"""Vehicle parameter loading and per-env domain randomization.

Nominal values come from config/vehicle_nominal.yaml, ranges from
config/randomization.yaml (INTERFACES.md §3). Never hand-copy numbers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields

import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOMINAL_PATH = os.path.join(REPO_ROOT, "config", "vehicle_nominal.yaml")
RANDOMIZATION_PATH = os.path.join(REPO_ROOT, "config", "randomization.yaml")


def load_nominal(path: str = NOMINAL_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_randomization(path: str = RANDOMIZATION_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class VehicleParams:
    """Per-env physical + sensor parameters, all tensors with leading dim N."""
    mass: torch.Tensor            # (N,)
    inertia: torch.Tensor         # (N,3) diagonal
    arm_xy: torch.Tensor          # (N,)
    thrust_max: torch.Tensor      # (N,4) per motor (includes imbalance)
    thrust_curve_a: torch.Tensor  # (N,)
    yaw_k: torch.Tensor           # (N,)
    tau_up: torch.Tensor          # (N,)
    tau_down: torch.Tensor        # (N,)
    lin_drag: torch.Tensor        # (N,3)
    rot_drag: torch.Tensor        # (N,3)
    latency_steps: torch.Tensor   # (N,) long, in physics steps
    gyro_noise_std: torch.Tensor  # (N,)
    gyro_bias: torch.Tensor       # (N,3)
    accel_noise_std: torch.Tensor # (N,)
    vel_noise_std: torch.Tensor   # (N,)
    pos_noise_std: torch.Tensor   # (N,)

    def assign(self, idx: torch.Tensor, other: "VehicleParams") -> None:
        """self[idx] = other (other has len(idx) rows)."""
        for f in fields(self):
            getattr(self, f.name)[idx] = getattr(other, f.name)


def _u(gen, shape, lo, hi, dtype):
    return lo + (hi - lo) * torch.rand(shape, generator=gen, dtype=dtype)


def sample_params(n: int, nominal: dict, rand: dict | None, physics_dt: float,
                  gen: torch.Generator | None = None, use_nominal: bool = False,
                  dtype=torch.float32, max_latency_steps: int | None = None) -> VehicleParams:
    """Sample n parameter sets. use_nominal=True (or rand=None) gives nominal
    params with zero sensor noise/bias and the nominal latency."""
    nm = nominal
    one = torch.ones(n, dtype=dtype)
    v3 = lambda key: torch.tensor(nm[key], dtype=dtype).expand(n, 3).clone()
    p = VehicleParams(
        mass=one * nm["mass"],
        inertia=v3("inertia"),
        arm_xy=one * nm["arm_xy"],
        thrust_max=torch.full((n, 4), nm["thrust_max"], dtype=dtype),
        thrust_curve_a=one * nm["thrust_curve_a"],
        yaw_k=one * nm["yaw_torque_coeff"],
        tau_up=one * nm["motor_tau_up"],
        tau_down=one * nm["motor_tau_down"],
        lin_drag=v3("lin_drag"),
        rot_drag=v3("rot_drag"),
        latency_steps=torch.full((n,), int(round(nm["latency_s"] / physics_dt)), dtype=torch.long),
        gyro_noise_std=torch.zeros(n, dtype=dtype),
        gyro_bias=torch.zeros(n, 3, dtype=dtype),
        accel_noise_std=torch.zeros(n, dtype=dtype),
        vel_noise_std=torch.zeros(n, dtype=dtype),
        pos_noise_std=torch.zeros(n, dtype=dtype),
    )
    if use_nominal or rand is None:
        if max_latency_steps is not None:
            p.latency_steps.clamp_(max=max_latency_steps)
        return p
    r = rand
    p.mass *= _u(gen, (n,), *r["mass"], dtype)
    p.inertia *= _u(gen, (n, 3), *r["inertia"], dtype)
    p.thrust_max *= _u(gen, (n, 1), *r["thrust_max"], dtype)
    p.thrust_max *= _u(gen, (n, 4), *r["per_motor_thrust_imbalance"], dtype)
    p.yaw_k *= _u(gen, (n,), *r["yaw_torque_coeff"], dtype)
    ts = _u(gen, (n,), *r["motor_tau"], dtype)
    p.tau_up *= ts
    p.tau_down *= ts
    p.lin_drag *= _u(gen, (n, 3), *r["lin_drag"], dtype)
    lat = _u(gen, (n,), *r["latency_s"], dtype)
    p.latency_steps = torch.round(lat / physics_dt).long()
    if max_latency_steps is not None:
        p.latency_steps.clamp_(max=max_latency_steps)
    p.gyro_noise_std = _u(gen, (n,), *r["gyro_noise_std"], dtype)
    p.gyro_bias = _u(gen, (n, 3), *r["gyro_bias"], dtype)
    p.accel_noise_std = _u(gen, (n,), *r["accel_noise_std"], dtype)
    p.vel_noise_std = _u(gen, (n,), *r["vel_noise_std"], dtype)
    p.pos_noise_std = _u(gen, (n,), *r["pos_noise_std"], dtype)
    return p


def max_latency_steps_for(nominal: dict, rand: dict | None, physics_dt: float) -> int:
    lat = nominal["latency_s"]
    if rand is not None:
        lat = max(lat, rand["latency_s"][1])
    return int(round(lat / physics_dt))
