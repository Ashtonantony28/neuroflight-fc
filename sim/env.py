"""Vectorized hover/recovery RL environment (no gym dependency).

* Control at `control_rate_hz` (500 Hz); each env step runs
  rate_loop_hz/control_rate_hz physics + rate-PID substeps (2 at 1 kHz).
* Action: CTBR, (N,4) in [-1,1] (INTERFACES.md §5).
* Observation: (N,19) float32 exactly per INTERFACES.md §4.
  - body rates (obs 12-14) AND the rate-PID measurement are the firmware's
    `state.omega`: raw gyro (noise + bias) -> 150 Hz biquad -> RPM notches ->
    minus the gyro-bias estimate (true bias + N(0, gyro_bias_resid_std) per
    episode), via sim/fw_estimator.GyroFilterChain (`gyro_filter`, default on;
    FOLLOWUPS 2). `gyro_filter=False` restores the raw-gyro path (v1 training).
  - position/velocity: `estimator=False` -> truth + white noise per step
    (pos/vel_noise_std); `estimator=True` -> the firmware's flow/ToF Kalman
    filter (sim/fw_estimator.TranslationKf) fed with SITL-identical range and
    flow measurements and the 40 Hz-filtered accelerometer.
  - attitude: true R, optionally times a fixed per-episode misalignment
    (`att_misalign_deg`, also used by the KF), standing in for Mahony / IMU
    mounting error (SITL Mahony error is ~0.1 deg rms in hover).
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
from .fw_estimator import GyroFilterChain, TranslationKf
from .params import load_nominal, load_randomization
from .quadrotor import Quadrotor
from .rotations import quat_from_axis_angle, quat_to_rotmat, random_tilted_quat

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
    # supervisor-envelope shaping (default off; see _envelope_terms)
    "tilt_pred": 0.0,     # * relu(tilt + 0.1 s * tilt_rate - 50°) gated on tilt_rate (TILT_PREDICTED margin)
    "tilt_excess": 0.0,   # * relu(tilt - 55°)   (TILT monitor margin, 60°)
    "thrust_smooth": 0.0, # * (a0 - a0_prev)^2    (extra thrust-channel smoothness)
    "est_innov": 0.0,     # * relu(innov_lpf - 0.25 rad): firmware ESTIMATOR monitor margin (Mahony innovation > 0.35 rad)
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
    # firmware sensing path (sim/fw_estimator.py)
    gyro_filter: bool = True               # rate PID + obs use fw-filtered, bias-corrected gyro
    gyro_bias_resid_std: float = 0.001     # rad/s, bias-estimate error per episode (SITL: ~3e-4)
    estimator: bool = False                # pos/vel obs from the firmware flow/ToF KF
    att_misalign_deg: float = 0.0          # std of a fixed per-episode attitude error (obs + KF)
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
        fs = float(self.nom["rate_loop_hz"])
        self.gfilt = GyroFilterChain(self.n, fs) if cfg.gyro_filter else None
        self.bias_est = torch.zeros(self.n, 3)
        self.kf = TranslationKf(self.n, fs, float(self.nom["gravity"])) if cfg.estimator else None
        self.R_mis = torch.eye(3).expand(self.n, 3, 3).clone()
        self.innov_lpf = torch.zeros(self.n)
        self.innov_k = self.control_dt / (1.0 / (2.0 * math.pi * 1.0) + self.control_dt)

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
        self.prev_action[idx] = 0.0
        self.step_count[idx] = 0
        self.innov_lpf[idx] = 0.0
        self.reinit_sensing(idx)

    def reinit_sensing(self, idx: torch.Tensor) -> None:
        """(Re)initialise the sensing path + rate PID of envs `idx` for their
        current physics state (motors assumed near hover): gyro filters in
        steady state, bias estimate, attitude misalignment, KF at the true
        state, and the rate PID seeded with the first measurement (FOLLOWUPS
        1: no derivative kick). Call after setting the state by hand."""
        c, m, q = self.cfg, idx.numel(), self.quad
        p = q.params
        omega = q.omega[idx]
        q.specific_force_b[idx] = torch.tensor([0.0, 0.0, float(self.nom["gravity"])])
        if self.gfilt is None:
            self.last_gyro[idx] = omega + p.gyro_bias[idx] + torch.randn(m, 3, generator=self.gen) * p.gyro_noise_std[idx].unsqueeze(-1)
        else:
            # firmware filters have been running: steady state at the current rate
            self.bias_est[idx] = p.gyro_bias[idx] + torch.randn(m, 3, generator=self.gen) * c.gyro_bias_resid_std
            self.gfilt.reset(idx, omega + p.gyro_bias[idx], 400.0 * q.motor[idx])
            self.last_gyro[idx] = omega + p.gyro_bias[idx] - self.bias_est[idx]
        if c.att_misalign_deg > 0:
            ax = torch.randn(m, 3, generator=self.gen)
            ang = torch.randn(m, generator=self.gen).abs() * math.radians(c.att_misalign_deg)
            self.R_mis[idx] = quat_to_rotmat(quat_from_axis_angle(ax / ax.norm(dim=-1, keepdim=True), ang))
        if self.kf is not None:
            self.kf.reset(idx, q.pos[idx], q.vel[idx], q.specific_force_b[idx])
        self.rate_ctrl.reset(idx, self.last_gyro[idx].clone())

    def _compute_obs(self, idx: torch.Tensor | None = None) -> None:
        q, o = self.quad, self.obs_buf
        if idx is None:
            if self.kf is not None:
                pos, vel = self.kf.p, self.kf.v
            else:
                pos, vel = q.sample_pos(), q.sample_vel()
            R = self._R_obs()
            o[:, 0:3] = (pos - self.target).clamp(-2.0, 2.0)
            o[:, 3:6] = vel.clamp(-5.0, 5.0)
            o[:, 6:9] = R[:, :, 0]
            o[:, 9:12] = R[:, :, 1]
            o[:, 12:15] = self.last_gyro
            o[:, 15:19] = self.prev_action
            return
        m, p = idx.numel(), q.params
        if self.kf is not None:
            pos, vel = self.kf.p[idx], self.kf.v[idx]
        else:
            pos = q.pos[idx] + torch.randn(m, 3, generator=self.gen) * p.pos_noise_std[idx].unsqueeze(-1)
            vel = q.vel[idx] + torch.randn(m, 3, generator=self.gen) * p.vel_noise_std[idx].unsqueeze(-1)
        R = self._R_obs(idx)
        o[idx, 0:3] = (pos - self.target[idx]).clamp(-2.0, 2.0)
        o[idx, 3:6] = vel.clamp(-5.0, 5.0)
        o[idx, 6:9] = R[:, :, 0]
        o[idx, 9:12] = R[:, :, 1]
        o[idx, 12:15] = self.last_gyro[idx]
        o[idx, 15:19] = self.prev_action[idx]

    def _R_obs(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        R = self.quad.R if idx is None else self.quad.R[idx]
        if self.cfg.att_misalign_deg <= 0:
            return R
        Rm = self.R_mis if idx is None else self.R_mis[idx]
        return R @ Rm

    def _sense(self) -> torch.Tensor:
        """One 1 kHz firmware sensing tick on the current physics state:
        returns the rate measurement (rate PID input and obs 12-14) and
        advances the KF (if enabled)."""
        q = self.quad
        g = q.sample_gyro()
        if self.gfilt is not None:
            g = self.gfilt.step(g, 400.0 * q.motor) - self.bias_est
        if self.kf is not None:
            p = q.params
            R = self._R_obs()
            self.kf.predict(q.sample_accel(), R)
            self.kf.correct(R, g, q.pos, q.vel, q.R, q.omega, p.pos_noise_std, p.vel_noise_std, self.gen)
        return g

    def tilt_and_rate(self):
        """True tilt (rad) and tilt rate (rad/s, > 0 = away from level), as
        nf::tilt_rate (supervisor) computes them."""
        q = self.quad
        R, w = q.R, q.omega
        r20, r21 = R[:, 2, 0], R[:, 2, 1]
        tilt = torch.acos(R[:, 2, 2].clamp(-1.0, 1.0))
        s = torch.sqrt(r20 * r20 + r21 * r21)
        tr = torch.where(s < 1e-3, torch.sqrt(w[:, 0] ** 2 + w[:, 1] ** 2),
                         -(r20 * w[:, 1] - r21 * w[:, 0]) / s.clamp_min(1e-3))
        return tilt, tr

    def _envelope_terms(self):
        """Margins to the firmware supervisor's TILT_PREDICTED (tilt rate >=
        4 rad/s and tilt + 0.1 s * rate > 60°) and TILT (> 60°) monitors.
        Penalised 10° / 5° inside the thresholds; the predicted-tilt term is
        faded in between tilt rates of 2.5 and 4 rad/s."""
        tilt, tr = self.tilt_and_rate()
        pred = tilt + 0.1 * tr.clamp_min(0.0)
        gate = ((tr - 2.5) / 1.5).clamp(0.0, 1.0)
        pen_pred = (pred - math.radians(50.0)).clamp_min(0.0) * gate
        pen_tilt = (tilt - math.radians(55.0)).clamp_min(0.0)
        return pen_pred, pen_tilt

    def _update_innovation(self) -> None:
        """Proxy of the firmware Mahony health signal (nf::Mahony::innovation;
        supervisor ESTIMATOR: > 0.35 rad for 200 ms). On a quadrotor the
        accelerometer sees thrust, not gravity, so the innovation is ~ the
        tilt held while accelerating, and SITL shows the Mahony estimate
        really is pulled off by 8-10 deg when it trips (hard braking after a
        push). The proxy is the 1 Hz low-pass of the true tilt, WITHOUT the
        firmware's |omega| < 2 rad/s gate: a gated proxy was exploited in
        training (spinning freezes it; RATE trips in SITL)."""
        tilt = torch.acos(self.quad.R[:, 2, 2].clamp(-1.0, 1.0))
        self.innov_lpf = self.innov_lpf + self.innov_k * (tilt - self.innov_lpf)
        # the firmware innovation only moves while |omega| < 2 rad/s: penalise only then
        self.innov_calm = self.quad.omega.norm(dim=-1) < 2.0

    def step(self, action: torch.Tensor):
        a = action.to(torch.float32).clamp(-1.0, 1.0)
        F, rate_sp = action_to_ctbr(a, self.nom)
        q = self.quad
        # The policy saw the rate measurement `last_gyro` of this tick; the
        # rate PID uses that same sample for the first substep (as firmware).
        gyro = self.last_gyro
        for k in range(self.substeps):
            if k > 0:
                gyro = self._sense()
            q.step(self.rate_ctrl(F, rate_sp, gyro))
        self.last_gyro = self._sense()
        self.step_count += 1
        self._update_innovation()

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
        if w.get("tilt_pred", 0.0) or w.get("tilt_excess", 0.0):
            pen_pred, pen_tilt = self._envelope_terms()
            rew = rew - w.get("tilt_pred", 0.0) * pen_pred - w.get("tilt_excess", 0.0) * pen_tilt
        if w.get("est_innov", 0.0):
            rew = rew - w["est_innov"] * (self.innov_lpf - 0.25).clamp_min(0.0) * self.innov_calm
        if w.get("thrust_smooth", 0.0):
            rew = rew - w["thrust_smooth"] * (a[:, 0] - self.prev_action[:, 0]) ** 2
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
