"""Vectorized hover/recovery RL environment (no gym dependency).

* Control at `control_rate_hz` (500 Hz); each env step runs
  rate_loop_hz/control_rate_hz physics + rate-PID substeps (2 at 1 kHz).
* Action: CTBR, (N,4) in [-1,1] (INTERFACES.md §5).
* Observation: (N,19) float32 exactly per INTERFACES.md §4, built from the
  *measured* position/velocity (noisy), gyro (noise + bias) and true attitude
  (no estimator modelled yet).
* Auto-reset: envs that are done are reset inside `step`; the returned obs
  for those rows is the post-reset obs, the pre-reset one is in
  info["terminal_obs"]. info["timeout"] marks truncation (bootstrap), and
  info["crash"]/info["oob"] mark terminations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .ctbr import RateController, RatePIDGains, action_to_ctbr
from .params import load_nominal, load_randomization
from .quadrotor import Quadrotor
from .rotations import random_tilted_quat

OBS_DIM = 19
ACT_DIM = 4

DEFAULT_REWARD_WEIGHTS = {
    "alive": 2.0,         # per step, while not crashed
    "pos": 1.0,           # * ||p - p_target||
    "vel": 0.2,           # * ||v||
    "upright": 0.5,       # * (1 - R[2,2])   (0 upright, 2 upside-down)
    "rate": 0.02,         # * ||ω||^2
    "yaw_rate": 0.02,     # * ω_z^2 (extra)
    "action": 0.02,       # * ||a[1:]||^2 (rate magnitude)
    "smooth": 0.5,        # * ||a - a_prev||^2
    "crash": 50.0,        # one-off on crash / out-of-bounds
}


@dataclass
class EnvConfig:
    num_envs: int = 4096
    seed: int = 0
    use_nominal: bool = False              # disable domain randomization
    episode_len_s: float = 5.0
    randomize_initial_time: bool = True    # desync timeouts on first reset
    # targets
    target_xy: float = 1.0                 # uniform in [-v, v]
    target_z: tuple = (1.0, 2.0)
    # curriculum: initial-state distribution
    init_pos_offset: float = 1.0           # uniform per axis around target
    init_min_z: float = 0.5
    init_max_tilt_deg: float = 60.0        # uniform tilt in [0, this]
    init_upside_down_prob: float = 0.0     # tilt in [150°,180°]
    init_max_vel: float = 1.0              # uniform per axis
    init_max_rate: float = 2.0             # rad/s, uniform per axis
    # termination
    oob_dist: float = 3.0                  # |p - p_target| per axis
    max_rate_term: float = 40.0            # rad/s
    reward_weights: dict = field(default_factory=lambda: dict(DEFAULT_REWARD_WEIGHTS))
    rate_gains: RatePIDGains = field(default_factory=RatePIDGains)


class QuadEnv:
    def __init__(self, cfg: EnvConfig | None = None, nominal: dict | None = None,
                 randomization: dict | None = None, **overrides):
        cfg = cfg or EnvConfig()
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise AttributeError(f"unknown EnvConfig field {k}")
            setattr(cfg, k, v)
        self.cfg = cfg
        self.nom = nominal if nominal is not None else load_nominal()
        rand = randomization if randomization is not None else load_randomization()
        self.n = cfg.num_envs
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.quad = Quadrotor(self.n, self.nom, rand, cfg.use_nominal, generator=self.gen)
        self.rate_ctrl = RateController(self.n, self.nom, cfg.rate_gains)
        self.substeps = int(round(self.nom["rate_loop_hz"] / self.nom["control_rate_hz"]))
        self.control_dt = 1.0 / self.nom["control_rate_hz"]
        self.max_steps = int(round(cfg.episode_len_s / self.control_dt))
        self.w = cfg.reward_weights
        self.target = torch.zeros(self.n, 3)
        self.prev_action = torch.zeros(self.n, ACT_DIM)
        self.step_count = torch.zeros(self.n, dtype=torch.long)
        self.last_gyro = torch.zeros(self.n, 3)
        self.obs_buf = torch.zeros(self.n, OBS_DIM)
        self.max_rate = torch.tensor(self.nom["max_body_rate"])
        self._all = torch.arange(self.n)

    # ------------------------------------------------------------------ helpers
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen)

    def _u(self, shape, lo, hi):
        return lo + (hi - lo) * self._rand(*shape)

    def hover_action(self) -> torch.Tensor:
        """Nominal-hover CTBR action (useful for tests)."""
        F = self.nom["mass"] * self.nom["gravity"]
        a0 = F / (4 * self.nom["thrust_max"]) * 2 - 1
        a = torch.zeros(self.n, ACT_DIM)
        a[:, 0] = a0
        return a

    # ------------------------------------------------------------------ API
    def reset(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        first = idx is None
        idx = self._all if idx is None else idx
        if idx.numel() > 0:
            self._reset_idx(idx)
            if first and self.cfg.randomize_initial_time:
                self.step_count[idx] = (self._rand(idx.numel()) * self.max_steps).long()
        self._compute_obs(None if first else idx)
        return self.obs_buf.clone()

    def _reset_idx(self, idx: torch.Tensor) -> None:
        c, m = self.cfg, idx.numel()
        q = self.quad
        q.resample_params(idx)
        tgt = torch.stack((self._u((m,), -c.target_xy, c.target_xy),
                           self._u((m,), -c.target_xy, c.target_xy),
                           self._u((m,), *c.target_z)), -1)
        self.target[idx] = tgt
        pos = tgt + self._u((m, 3), -c.init_pos_offset, c.init_pos_offset)
        pos[:, 2] = pos[:, 2].clamp_min(c.init_min_z)
        quat = random_tilted_quat(m, math.radians(c.init_max_tilt_deg), self.gen,
                                  upside_down_prob=c.init_upside_down_prob)
        vel = self._u((m, 3), -c.init_max_vel, c.init_max_vel)
        omega = self._u((m, 3), -c.init_max_rate, c.init_max_rate)
        w_h = q.hover_motor_state(idx)
        q.set_state(idx, pos=pos, vel=vel, quat=quat, omega=omega, motor=w_h, cmd=w_h)
        self.rate_ctrl.reset(idx)
        self.prev_action[idx] = 0.0
        self.step_count[idx] = 0
        p = q.params
        self.last_gyro[idx] = omega + p.gyro_bias[idx] + torch.randn(m, 3, generator=self.gen) * p.gyro_noise_std[idx].unsqueeze(-1)

    def _compute_obs(self, idx: torch.Tensor | None = None) -> None:
        q, o = self.quad, self.obs_buf
        if idx is None:
            pos, vel = q.sample_pos(), q.sample_vel()
            o[:, 0:3] = (pos - self.target).clamp(-2.0, 2.0)
            o[:, 3:6] = vel.clamp(-5.0, 5.0)
            o[:, 6:9] = q.R[:, :, 0]
            o[:, 9:12] = q.R[:, :, 1]
            o[:, 12:15] = self.last_gyro
            o[:, 15:19] = self.prev_action
            return
        m, p = idx.numel(), q.params
        pos = q.pos[idx] + torch.randn(m, 3, generator=self.gen) * p.pos_noise_std[idx].unsqueeze(-1)
        vel = q.vel[idx] + torch.randn(m, 3, generator=self.gen) * p.vel_noise_std[idx].unsqueeze(-1)
        o[idx, 0:3] = (pos - self.target[idx]).clamp(-2.0, 2.0)
        o[idx, 3:6] = vel.clamp(-5.0, 5.0)
        o[idx, 6:9] = q.R[idx, :, 0]
        o[idx, 9:12] = q.R[idx, :, 1]
        o[idx, 12:15] = self.last_gyro[idx]
        o[idx, 15:19] = self.prev_action[idx]

    def step(self, action: torch.Tensor):
        a = action.to(torch.float32).clamp(-1.0, 1.0)
        F, rate_sp = action_to_ctbr(a, self.nom)
        q = self.quad
        # The policy saw gyro sample `last_gyro` taken at this tick; the rate
        # PID uses that same sample for the first substep (as firmware would).
        gyro = self.last_gyro
        for k in range(self.substeps):
            if k > 0:
                gyro = q.sample_gyro()
            q.step(self.rate_ctrl(F, rate_sp, gyro))
        self.last_gyro = q.sample_gyro()
        self.step_count += 1

        # reward (on true state)
        w = self.w
        err = q.pos - self.target
        dist = err.norm(dim=-1)
        R22 = q.R[:, 2, 2]
        om = q.omega
        crash = q.crashed.clone()
        oob = (err.abs() > self.cfg.oob_dist).any(-1) | (om.abs() > self.cfg.max_rate_term).any(-1)
        term = crash | oob
        timeout = (self.step_count >= self.max_steps) & ~term
        rew = (w["alive"]
               - w["pos"] * dist
               - w["vel"] * q.vel.norm(dim=-1)
               - w["upright"] * (1 - R22)
               - w["rate"] * (om * om).sum(-1)
               - w["yaw_rate"] * om[:, 2] ** 2
               - w["action"] * (a[:, 1:] ** 2).sum(-1)
               - w["smooth"] * ((a - self.prev_action) ** 2).sum(-1))
        rew = torch.where(term, rew - w["crash"], rew)
        self.prev_action = a
        self._compute_obs()
        done = term | timeout
        info = {"crash": crash, "oob": oob, "timeout": timeout, "dist": dist}
        if bool(done.any()):
            info["terminal_obs"] = self.obs_buf.clone()
            ids = done.nonzero(as_tuple=True)[0]
            self._reset_idx(ids)
            self._compute_obs(ids)
        return self.obs_buf.clone(), rew, done, info
