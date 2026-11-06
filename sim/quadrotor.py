"""Batched rigid-body quadrotor (N envs, torch, CPU).

Conventions: INTERFACES.md §1-2. World ENU z-up, body FLU, q=[w,x,y,z] B->W.

Physics step (dt = 1/rate_loop_hz, 1 kHz by default), in order:
  1. push motor command into the per-env latency ring buffer, read the
     command delayed by `latency_steps`;
  2. first-order motor lag (exact discretization, tau_up / tau_down);
  3. thrust curve T_i = thrust_max_i (A w^2 + (1-A) w), body torques per §2,
     body-frame linear drag and rotational drag;
  4. semi-implicit Euler: ω, then q (exact exponential map with new ω), then
     v, then p (with new v).
Floor: z < floor_z sets `crashed`.
"""
from __future__ import annotations

import torch

from .params import (VehicleParams, load_nominal, load_randomization,
                     max_latency_steps_for, sample_params)
from .rotations import quat_integrate_body, quat_to_rotmat

# Torque sign vectors from INTERFACES.md §2
ROLL_SIGN = (1.0, -1.0, -1.0, 1.0)
PITCH_SIGN = (-1.0, -1.0, 1.0, 1.0)
YAW_SIGN = (1.0, -1.0, 1.0, -1.0)


class Quadrotor:
    def __init__(self, num_envs: int, nominal: dict | None = None, randomization: dict | None = None,
                 use_nominal: bool = False, seed: int = 0, floor_z: float = 0.0,
                 dtype=torch.float32, generator: torch.Generator | None = None,
                 max_latency_steps: int | None = None):
        self.n = num_envs
        self.dtype = dtype
        self.nominal = nominal if nominal is not None else load_nominal()
        self.rand = randomization if randomization is not None else load_randomization()
        self.use_nominal = use_nominal
        self.dt = 1.0 / float(self.nominal["rate_loop_hz"])
        self.g = float(self.nominal["gravity"])
        self.floor_z = floor_z
        # Hard clamp on body rates. Physically unreachable for this vehicle in
        # controlled flight; keeps explicit integration of the gyroscopic term
        # (unstable once |ω|·dt ~ 1) from producing NaNs under garbage commands.
        self.max_omega = 100.0
        if generator is None:
            generator = torch.Generator().manual_seed(seed)
        self.gen = generator

        # ring buffer sized for the randomization range even in nominal mode so
        # tests can override latency_steps; never silently wraps.
        self.max_delay = max(max_latency_steps_for(self.nominal, self.rand, self.dt), max_latency_steps or 0)
        self.buf_len = self.max_delay + 1

        n, dt_ = num_envs, dtype
        self.params: VehicleParams = sample_params(n, self.nominal, self.rand, self.dt, self.gen,
                                                   use_nominal, dt_, self.max_delay)
        self.pos = torch.zeros(n, 3, dtype=dt_)
        self.vel = torch.zeros(n, 3, dtype=dt_)
        self.quat = torch.zeros(n, 4, dtype=dt_)
        self.quat[:, 0] = 1
        self.omega = torch.zeros(n, 3, dtype=dt_)
        self.motor = torch.zeros(n, 4, dtype=dt_)          # motor state w in [0,1]
        self.cmd_buf = torch.zeros(n, self.buf_len, 4, dtype=dt_)
        self.buf_head = 0
        self.crashed = torch.zeros(n, dtype=torch.bool)
        self.specific_force_b = torch.zeros(n, 3, dtype=dt_)  # last true specific force (for accel)
        self.R = quat_to_rotmat(self.quat)
        self._ar = torch.arange(n)
        s = lambda v: torch.tensor(v, dtype=dt_)
        self._roll_s, self._pitch_s, self._yaw_s = s(ROLL_SIGN), s(PITCH_SIGN), s(YAW_SIGN)
        self._ez = s([0.0, 0.0, 1.0])
        self._update_derived(self._ar)

    # ------------------------------------------------------------------ params
    def _update_derived(self, idx: torch.Tensor) -> None:
        p = self.params
        if not hasattr(self, "alpha_up"):
            self.alpha_up = torch.zeros(self.n, 1, dtype=self.dtype)
            self.alpha_down = torch.zeros(self.n, 1, dtype=self.dtype)
        self.alpha_up[idx, 0] = 1 - torch.exp(-self.dt / p.tau_up[idx])
        self.alpha_down[idx, 0] = 1 - torch.exp(-self.dt / p.tau_down[idx])

    def resample_params(self, idx: torch.Tensor) -> None:
        if idx.numel() == 0:
            return
        new = sample_params(idx.numel(), self.nominal, self.rand, self.dt, self.gen,
                            self.use_nominal, self.dtype, self.max_delay)
        self.params.assign(idx, new)
        self._update_derived(idx)

    # ------------------------------------------------------------------ state
    def hover_motor_state(self, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Per-motor w that gives T_i = m g / 4 (true params)."""
        p = self.params
        sl = slice(None) if idx is None else idx
        T = (p.mass[sl] * self.g / 4).unsqueeze(-1) / p.thrust_max[sl]
        A = p.thrust_curve_a[sl].unsqueeze(-1)
        return (-(1 - A) + torch.sqrt((1 - A) ** 2 + 4 * A * T)) / (2 * A)

    def set_state(self, idx, pos=None, vel=None, quat=None, omega=None, motor=None, cmd=None):
        """Set state of envs idx; `cmd` (len(idx),4) fills the latency buffer."""
        if pos is not None: self.pos[idx] = pos
        if vel is not None: self.vel[idx] = vel
        if quat is not None:
            self.quat[idx] = quat / quat.norm(dim=-1, keepdim=True)
            self.R[idx] = quat_to_rotmat(self.quat[idx])
        if omega is not None: self.omega[idx] = omega
        if motor is not None: self.motor[idx] = motor
        if cmd is not None: self.cmd_buf[idx] = cmd.unsqueeze(1).expand(-1, self.buf_len, -1)
        self.crashed[idx] = False
        self.specific_force_b[idx] = 0.0

    # ------------------------------------------------------------------ sensors
    def sample_gyro(self) -> torch.Tensor:
        p = self.params
        noise = torch.randn(self.n, 3, generator=self.gen, dtype=self.dtype) * p.gyro_noise_std.unsqueeze(-1)
        return self.omega + p.gyro_bias + noise

    def sample_accel(self) -> torch.Tensor:
        p = self.params
        noise = torch.randn(self.n, 3, generator=self.gen, dtype=self.dtype) * p.accel_noise_std.unsqueeze(-1)
        return self.specific_force_b + noise

    def sample_pos(self) -> torch.Tensor:
        noise = torch.randn(self.n, 3, generator=self.gen, dtype=self.dtype)
        return self.pos + noise * self.params.pos_noise_std.unsqueeze(-1)

    def sample_vel(self) -> torch.Tensor:
        noise = torch.randn(self.n, 3, generator=self.gen, dtype=self.dtype)
        return self.vel + noise * self.params.vel_noise_std.unsqueeze(-1)

    # ------------------------------------------------------------------ dynamics
    def thrusts(self, w: torch.Tensor | None = None) -> torch.Tensor:
        p = self.params
        w = self.motor if w is None else w
        A = p.thrust_curve_a.unsqueeze(-1)
        return p.thrust_max * (A * w * w + (1 - A) * w)

    def body_wrench(self, T: torch.Tensor):
        """Collective thrust (N,) and body torque (N,3) from per-motor thrusts (§2)."""
        p = self.params
        a, k = p.arm_xy, p.yaw_k
        tx = a * (T @ self._roll_s)
        ty = a * (T @ self._pitch_s)
        tz = k * (T @ self._yaw_s)
        return T.sum(-1), torch.stack((tx, ty, tz), -1)

    def step(self, motor_cmd: torch.Tensor) -> None:
        """Advance one physics step with motor commands c in [0,1] (N,4)."""
        p, dt = self.params, self.dt
        c = motor_cmd.clamp(0.0, 1.0)
        # 1. latency ring buffer
        self.cmd_buf[:, self.buf_head] = c
        read = (self.buf_head - p.latency_steps.clamp(0, self.max_delay)) % self.buf_len
        c_eff = self.cmd_buf[self._ar, read]
        self.buf_head = (self.buf_head + 1) % self.buf_len
        # 2. motor lag
        alpha = torch.where(c_eff > self.motor, self.alpha_up, self.alpha_down)
        self.motor = self.motor + alpha * (c_eff - self.motor)
        # 3. forces / torques
        T = self.thrusts()
        F, tau = self.body_wrench(T)
        R = self.R
        omega = self.omega
        J = p.inertia
        tau = tau - p.rot_drag * omega
        Jw = J * omega
        omega_dot = (tau - torch.linalg.cross(omega, Jw, dim=-1)) / J
        v_b = torch.einsum("nji,nj->ni", R, self.vel)          # R^T v
        f_b = -p.lin_drag * v_b
        f_b = f_b + F.unsqueeze(-1) * self._ez
        sf_b = f_b / p.mass.unsqueeze(-1)                       # specific force (body)
        acc_w = torch.einsum("nij,nj->ni", R, sf_b)
        acc_w[:, 2] -= self.g
        # 4. semi-implicit Euler
        self.omega = (omega + dt * omega_dot).clamp(-self.max_omega, self.max_omega)
        self.quat = quat_integrate_body(self.quat, self.omega, dt)
        self.R = quat_to_rotmat(self.quat)
        self.vel = self.vel + dt * acc_w
        self.pos = self.pos + dt * self.vel
        self.specific_force_b = sf_b
        self.crashed |= self.pos[:, 2] < self.floor_z
