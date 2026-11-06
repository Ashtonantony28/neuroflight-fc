"""Batched quaternion helpers. q = [w, x, y, z], rotates B -> W (INTERFACES.md §1)."""
from __future__ import annotations

import math

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), -1)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) -> (N,3,3), R maps body vectors to world."""
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack((
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ), -1)
    return R.reshape(q.shape[:-1] + (3, 3))


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    h = 0.5 * angle
    return torch.cat((torch.cos(h).unsqueeze(-1), torch.sin(h).unsqueeze(-1) * axis), -1)


def quat_integrate_body(q: torch.Tensor, omega_b: torch.Tensor, dt: float) -> torch.Tensor:
    """q <- q ⊗ exp(ω dt / 2), renormalized. Exact for constant body rate."""
    ang = omega_b.norm(dim=-1, keepdim=True) * dt
    half = 0.5 * ang
    # sin(half)/|ω| without division by zero: sin(h)/h * dt/2
    small = half < 1e-6
    s = torch.where(small, torch.ones_like(half), torch.sin(half) / torch.where(small, torch.ones_like(half), half))
    dq = torch.cat((torch.cos(half), s * 0.5 * dt * omega_b), -1)
    q = quat_mul(q, dq)
    return q / q.norm(dim=-1, keepdim=True)


def random_tilted_quat(n: int, max_tilt: torch.Tensor | float, gen, dtype=torch.float32,
                       upside_down_prob: float = 0.0) -> torch.Tensor:
    """Random yaw in [-pi, pi), then a tilt about a random horizontal axis with
    angle uniform in [0, max_tilt]. With prob upside_down_prob the tilt is
    uniform in [150°, 180°]."""
    yaw = (torch.rand(n, generator=gen, dtype=dtype) * 2 - 1) * math.pi
    tilt = torch.rand(n, generator=gen, dtype=dtype) * torch.as_tensor(max_tilt, dtype=dtype)
    if upside_down_prob > 0:
        ud = torch.rand(n, generator=gen, dtype=dtype) < upside_down_prob
        ud_tilt = math.radians(150) + torch.rand(n, generator=gen, dtype=dtype) * math.radians(30)
        tilt = torch.where(ud, ud_tilt, tilt)
    phi = torch.rand(n, generator=gen, dtype=dtype) * 2 * math.pi
    axis = torch.stack((torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)), -1)
    q_tilt = quat_from_axis_angle(axis, tilt)
    q_yaw = quat_from_axis_angle(torch.tensor([0.0, 0.0, 1.0], dtype=dtype).expand(n, 3), yaw)
    return quat_mul(q_tilt, q_yaw)  # world-frame tilt applied after yaw
