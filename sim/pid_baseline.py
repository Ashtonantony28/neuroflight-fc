"""Classical cascaded controller: position -> velocity -> attitude -> rate.

It consumes the *policy observation vector* (INTERFACES.md §4) plus nothing
else, and outputs a normalized CTBR action (§5), so it is a drop-in stand-in
for the learned policy (and mirrors what the firmware fallback can see).
The rate loop itself is the shared `RateController` in ctbr.py.

Attitude control uses a reduced-attitude (tilt-first) law that is well
defined for any tilt incl. upside down: the body rate setpoint rotates the
body z axis toward the desired thrust direction about their common normal,
with yaw control faded out while strongly tilted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .ctbr import ctbr_to_action


@dataclass
class PIDGains:
    kp_pos: tuple = (1.6, 1.6, 2.0)      # (m/s) / m
    max_vel: tuple = (2.0, 2.0, 1.5)     # m/s
    kp_vel: tuple = (4.0, 4.0, 5.0)      # (m/s^2) / (m/s)
    ki_vel: tuple = (0.8, 0.8, 3.0)      # (m/s^2) / m
    i_vel_limit: tuple = (2.0, 2.0, 4.0) # m/s^2
    max_tilt_deg: float = 40.0
    kp_att: float = 8.0                  # (rad/s) / rad
    kp_yaw: float = 3.0
    yaw_target: float = 0.0


class CascadedPID:
    def __init__(self, num_envs: int, nominal: dict, gains: PIDGains | None = None,
                 dtype=torch.float32):
        self.n, self.nom = num_envs, nominal
        self.g = gains or PIDGains()
        self.dt = 1.0 / float(nominal["control_rate_hz"])
        t = lambda v: torch.tensor(v, dtype=dtype)
        self.kp_pos, self.max_vel = t(self.g.kp_pos), t(self.g.max_vel)
        self.kp_vel, self.ki_vel, self.i_lim = t(self.g.kp_vel), t(self.g.ki_vel), t(self.g.i_vel_limit)
        self.max_rate = t(nominal["max_body_rate"])
        self.mass = float(nominal["mass"])
        self.grav = float(nominal["gravity"])
        self.f_max = 4 * float(nominal["thrust_max"])
        self.integ = torch.zeros(num_envs, 3, dtype=dtype)

    def reset(self, idx: torch.Tensor | None = None) -> None:
        if idx is None:
            self.integ.zero_()
        else:
            self.integ[idx] = 0

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        g = self.g
        pos_err = obs[:, 0:3]
        vel = obs[:, 3:6]
        c0, c1 = obs[:, 6:9], obs[:, 9:12]
        c2 = torch.linalg.cross(c0, c1, dim=-1)               # R[:,2] = body z in W
        # position -> velocity
        vel_sp = torch.maximum(torch.minimum(-self.kp_pos * pos_err, self.max_vel), -self.max_vel)
        # velocity -> acceleration (PI)
        ev = vel_sp - vel
        self.integ = torch.maximum(torch.minimum(self.integ + self.ki_vel * ev * self.dt, self.i_lim), -self.i_lim)
        acc = self.kp_vel * ev + self.integ
        acc_z = (acc[:, 2] + self.grav).clamp_min(0.2 * self.grav)
        # tilt limit
        lat = acc[:, 0:2]
        lat_max = acc_z * math.tan(math.radians(g.max_tilt_deg))
        ln = lat.norm(dim=-1).clamp_min(1e-9)
        lat = lat * (lat_max / ln).clamp(max=1.0).unsqueeze(-1)
        f_des = torch.cat((lat, acc_z.unsqueeze(-1)), -1) * self.mass  # desired force (W)
        z_des = f_des / f_des.norm(dim=-1, keepdim=True)
        # collective: project onto current body z; keep a floor so rates have authority
        F = (f_des * c2).sum(-1).clamp(0.15 * self.mass * self.grav, self.f_max)
        # reduced attitude: rotate e3_B onto z_des_B
        R = torch.stack((c0, c1, c2), -1)                      # columns
        z_b = torch.einsum("nji,nj->ni", R, z_des)             # R^T z_des
        cos_t = z_b[:, 2].clamp(-1, 1)
        angle = torch.acos(cos_t)
        axis = torch.stack((-z_b[:, 1], z_b[:, 0]), -1)        # e3 x z_b (x,y comps)
        an = axis.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(axis); fallback[:, 0] = 1.0  # upside-down: roll
        axis = torch.where(an > 1e-6, axis / an.clamp_min(1e-9), fallback)
        rate_xy = g.kp_att * angle.unsqueeze(-1) * axis
        yaw = torch.atan2(c0[:, 1], c0[:, 0])
        yaw_err = torch.remainder(g.yaw_target - yaw + math.pi, 2 * math.pi) - math.pi
        rate_z = g.kp_yaw * yaw_err * cos_t.clamp_min(0)
        rate_sp = torch.cat((rate_xy, rate_z.unsqueeze(-1)), -1)
        rate_sp = torch.maximum(torch.minimum(rate_sp, self.max_rate), -self.max_rate)
        return ctbr_to_action(F, rate_sp, self.nom)
