"""CTBR action interface (INTERFACES.md §5): normalized action -> collective
thrust + body-rate setpoint -> 1 kHz rate PID -> mixer -> motor commands.

Everything here models *firmware*, so it only uses nominal vehicle parameters
(the firmware does not know the randomized true ones).

Mixer saturation policy (thrust space, per motor T_i in [T_idle, T_max]):
  1. roll/pitch differential thrusts; if their spread exceeds the available
     range they are scaled down uniformly (direction preserved);
  2. collective is shifted (up or down) so roll/pitch fit — roll/pitch win
     over collective;
  3. yaw is added last, scaled by the largest factor in [0,1] that fits.
Thrust is then converted to command with the inverse nominal thrust curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .quadrotor import PITCH_SIGN, ROLL_SIGN, YAW_SIGN


@dataclass
class RatePIDGains:
    # angular-acceleration gains (rad/s^2 per rad/s), multiplied by nominal inertia
    # tuned against nominal + motor_tau x[0.7,1.3] + 0-8 ms latency:
    # 3 rad/s roll step -> t90 ~80 ms, overshoot 7-19 %
    kp: tuple = (35.0, 35.0, 20.0)
    ki: tuple = (60.0, 60.0, 20.0)
    kd: tuple = (0.7, 0.7, 0.0)          # on measurement, low-pass filtered
    d_lpf_hz: float = 20.0
    i_limit: tuple = (8.0, 8.0, 8.0)     # |integrator| limit (rad/s * s units of error)


def action_to_ctbr(action: torch.Tensor, nominal: dict):
    """(N,4) action in [-1,1] -> (collective N (N,), body-rate setpoint (N,3))."""
    a = action.clamp(-1.0, 1.0)
    F = (a[:, 0] + 1) * 0.5 * 4 * nominal["thrust_max"]
    rmax = torch.tensor(nominal["max_body_rate"], dtype=a.dtype)
    return F, a[:, 1:4] * rmax


def ctbr_to_action(F: torch.Tensor, rate_sp: torch.Tensor, nominal: dict) -> torch.Tensor:
    """Inverse of action_to_ctbr (with clipping to [-1,1])."""
    a0 = F / (4 * nominal["thrust_max"]) * 2 - 1
    rmax = torch.tensor(nominal["max_body_rate"], dtype=rate_sp.dtype)
    return torch.cat((a0.unsqueeze(-1), rate_sp / rmax), -1).clamp(-1, 1)


class Mixer:
    def __init__(self, nominal: dict, dtype=torch.float32):
        self.nom = nominal
        self.tmax = float(nominal["thrust_max"])
        self.A = float(nominal["thrust_curve_a"])
        self.a = float(nominal["arm_xy"])
        self.k = float(nominal["yaw_torque_coeff"])
        self.idle = float(nominal["idle_command"])
        self.t_lo = self.tmax * (self.A * self.idle ** 2 + (1 - self.A) * self.idle)
        self.t_hi = self.tmax
        s = lambda v: torch.tensor(v, dtype=dtype)
        self.rs, self.ps, self.ys = s(ROLL_SIGN), s(PITCH_SIGN), s(YAW_SIGN)

    def thrust_to_cmd(self, T: torch.Tensor) -> torch.Tensor:
        x = (T / self.tmax).clamp(0.0, 1.0)
        A = self.A
        return (-(1 - A) + torch.sqrt((1 - A) ** 2 + 4 * A * x)) / (2 * A)

    def mix(self, F: torch.Tensor, tau: torch.Tensor):
        """F (N,), tau (N,3) -> motor commands (N,4) in [idle,1], saturated flag (N,)."""
        lo, hi = self.t_lo, self.t_hi
        rng = hi - lo
        t_rp = (tau[:, 0:1] / (4 * self.a)) * self.rs + (tau[:, 1:2] / (4 * self.a)) * self.ps
        span = t_rp.amax(-1) - t_rp.amin(-1)
        scale_rp = (rng / span.clamp_min(1e-9)).clamp(max=1.0)
        t_rp = t_rp * scale_rp.unsqueeze(-1)
        base = F / 4
        base_c = torch.minimum(torch.maximum(base, lo - t_rp.amin(-1)), hi - t_rp.amax(-1))
        T = base_c.unsqueeze(-1) + t_rp
        t_y = (tau[:, 2:3] / (4 * self.k)) * self.ys
        # largest s in [0,1] with lo <= T + s t_y <= hi
        room = torch.where(t_y > 0, hi - T, T - lo).clamp_min(0.0)
        s_y = (room / t_y.abs().clamp_min(1e-12)).amin(-1).clamp(max=1.0)
        T = (T + s_y.unsqueeze(-1) * t_y).clamp(lo, hi)
        saturated = (scale_rp < 1) | (s_y < 1) | ((base_c - base).abs() > 1e-6)
        return self.thrust_to_cmd(T), saturated


class RateController:
    """Batched 1 kHz rate PID + mixer. State: integrator, filtered D."""

    def __init__(self, num_envs: int, nominal: dict, gains: RatePIDGains | None = None,
                 dtype=torch.float32):
        self.n = num_envs
        self.nom = nominal
        self.dt = 1.0 / float(nominal["rate_loop_hz"])
        g = gains or RatePIDGains()
        self.gains = g
        t = lambda v: torch.tensor(v, dtype=dtype)
        J = t(nominal["inertia"])
        self.kp, self.ki, self.kd = t(g.kp) * J, t(g.ki) * J, t(g.kd) * J
        self.i_lim = t(g.i_limit)
        self.d_alpha = 1.0 - torch.exp(torch.tensor(-2 * torch.pi * g.d_lpf_hz * self.dt)).item()
        self.mixer = Mixer(nominal, dtype)
        self.integ = torch.zeros(num_envs, 3, dtype=dtype)
        self.d_filt = torch.zeros(num_envs, 3, dtype=dtype)
        self.prev_meas = torch.zeros(num_envs, 3, dtype=dtype)
        self.use_d = bool(torch.any(self.kd != 0))

    def reset(self, idx: torch.Tensor, gyro: torch.Tensor | None = None) -> None:
        self.integ[idx] = 0
        self.d_filt[idx] = 0
        self.prev_meas[idx] = 0 if gyro is None else gyro

    def __call__(self, F: torch.Tensor, rate_sp: torch.Tensor, gyro: torch.Tensor) -> torch.Tensor:
        e = rate_sp - gyro
        u = self.kp * e
        if self.use_d:
            d = -(gyro - self.prev_meas) / self.dt
            self.d_filt = self.d_filt + self.d_alpha * (d - self.d_filt)
            u = u + self.kd * self.d_filt
            self.prev_meas = gyro
        cmd, sat = self.mixer.mix(F, u + self.ki * self.integ)
        # conditional integration: freeze integrator while mixer saturates
        upd = (~sat).unsqueeze(-1)
        self.integ = torch.where(upd, (self.integ + e * self.dt).clamp(-self.i_lim, self.i_lim), self.integ)
        return cmd
