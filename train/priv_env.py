"""Opt-in privileged-observation hook for the asymmetric critic.

`PrivilegedQuadEnv` subclasses sim.env.QuadEnv without changing its behaviour
(actor obs, rewards, resets and RNG consumption are identical). It adds:

* `privileged()` -> (N, PRIV_DIM) critic-only features from the *true* state
  and the randomized vehicle parameters (normalized by nominal);
* `info["terminal_priv"]` in `step()`: privileged features of done envs
  captured right before their auto-reset (needed to bootstrap timeouts).
* optional extra reward terms (training-side shaping, `extra_reward`):
  `pos_exp: {weight, sigma}` adds weight * exp(-(dist/sigma)^2), a bonus that
  is sharp near the target (the env's linear |p - p_target| term alone left a
  ~0.3 m steady-state offset). Uses info["dist"], i.e. the pre-reset state.

The critic input is `cat(actor_obs, privileged)`.
"""
from __future__ import annotations

import torch

from sim.env import QuadEnv

PRIV_DIM = 42


class PrivilegedQuadEnv(QuadEnv):
    def __init__(self, *args, extra_reward: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_reward = dict(extra_reward or {})
        nm = self.nom
        self._inertia_nom = torch.tensor(nm["inertia"])
        self._drag_nom = torch.tensor(nm["lin_drag"])
        self._term_priv = torch.zeros(self.n, PRIV_DIM)
        self._capture = False

    def privileged(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        q, p, nm = self.quad, self.quad.params, self.nom
        s = slice(None) if idx is None else idx
        R = q.R[s]
        feats = [
            (q.pos[s] - self.target[s]).clamp(-3.0, 3.0),           # 3 true pos err
            q.vel[s],                                                # 3 true vel
            R.reshape(R.shape[0], 9),                                # 9 full R
            q.omega[s].clamp(-40.0, 40.0),                           # 3 true rates
            q.motor[s],                                              # 4 motor state
            self.rate_ctrl.integ[s],                                 # 3 rate-PID integrator
            (p.mass[s] / nm["mass"]).unsqueeze(-1),                  # 1
            p.inertia[s] / self._inertia_nom,                        # 3
            p.thrust_max[s] / nm["thrust_max"],                      # 4
            (p.tau_up[s] / nm["motor_tau_up"]).unsqueeze(-1),        # 1
            (p.yaw_k[s] / nm["yaw_torque_coeff"]).unsqueeze(-1),     # 1
            p.lin_drag[s] / self._drag_nom,                          # 3
            p.latency_steps[s].float().unsqueeze(-1),                # 1
            p.gyro_bias[s],                                          # 3
        ]
        out = torch.cat(feats, -1)
        assert out.shape[-1] == PRIV_DIM
        return out

    def _reset_idx(self, idx: torch.Tensor) -> None:
        if self._capture and idx.numel() > 0:
            self._term_priv[idx] = self.privileged(idx)
        super()._reset_idx(idx)

    def step(self, action: torch.Tensor):
        self._capture = True
        try:
            obs, rew, done, info = super().step(action)
        finally:
            self._capture = False
        if "terminal_obs" in info:
            info["terminal_priv"] = self._term_priv.clone()
        pe = self.extra_reward.get("pos_exp")
        if pe and float(pe["weight"]) != 0.0:
            rew = rew + float(pe["weight"]) * torch.exp(-(info["dist"] / float(pe["sigma"])) ** 2)
        return obs, rew, done, info
