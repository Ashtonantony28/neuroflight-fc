import math

import pytest
import torch

from sim.quadrotor import Quadrotor
from sim.rotations import quat_from_axis_angle, quat_to_rotmat


def make(n=1, nominal=None, **kw):
    return Quadrotor(n, nominal=nominal, use_nominal=True, **kw)


def test_free_fall_matches_g(nominal):
    q = make(4, nominal)
    idx = torch.arange(4)
    q.set_state(idx, pos=torch.tensor([[0, 0, 10.0]] * 4), motor=torch.zeros(4, 4), cmd=torch.zeros(4, 4))
    q.params.lin_drag.zero_()
    steps = 1000  # 1 s
    for _ in range(steps):
        q.step(torch.zeros(4, 4))
    t = steps * q.dt
    g = nominal["gravity"]
    assert torch.allclose(q.vel[:, 2], torch.full((4,), -g * t), rtol=1e-4)
    # semi-implicit Euler: z = z0 - g dt^2 n(n+1)/2
    expected_z = 10 - g * q.dt ** 2 * steps * (steps + 1) / 2
    assert torch.allclose(q.pos[:, 2], torch.full((4,), expected_z), atol=1e-3)
    assert abs(expected_z - (10 - 0.5 * g * t * t)) < 0.01
    # accelerometer reads ~0 specific force in free fall
    assert q.sample_accel().abs().max() < 1e-5


def test_hover_equilibrium_nominal(nominal):
    q = make(1, nominal)
    w = q.hover_motor_state()
    q.set_state(torch.arange(1), pos=torch.tensor([[0, 0, 1.0]]), motor=w, cmd=w)
    T = q.thrusts()
    assert abs(T.sum().item() - nominal["mass"] * nominal["gravity"]) < 1e-4
    for _ in range(2000):
        q.step(w)
    assert q.pos.sub(torch.tensor([[0, 0, 1.0]])).abs().max() < 1e-3
    assert q.vel.abs().max() < 1e-3
    assert q.omega.abs().max() < 1e-4
    # accelerometer reads +g along body z
    assert abs(q.sample_accel()[0, 2].item() - nominal["gravity"]) < 1e-3


@pytest.mark.parametrize("motor", [0, 1, 2, 3])
def test_single_motor_torque_sign(nominal, motor):
    """INTERFACES.md §2: increasing T_i produces the tabled torque signs."""
    roll = [1, -1, -1, 1]
    pitch = [-1, -1, 1, 1]
    yaw = [1, -1, 1, -1]
    q = make(1, nominal)
    w = q.hover_motor_state()
    _, tau0 = q.body_wrench(q.thrusts(w))
    assert tau0.abs().max() < 1e-7
    w2 = w.clone()
    w2[0, motor] += 0.05
    _, tau = q.body_wrench(q.thrusts(w2))
    assert math.copysign(1, tau[0, 0].item()) == roll[motor]
    assert math.copysign(1, tau[0, 1].item()) == pitch[motor]
    assert math.copysign(1, tau[0, 2].item()) == yaw[motor]
    # and the dynamics respond with that angular acceleration sign
    q.set_state(torch.arange(1), pos=torch.tensor([[0, 0, 5.0]]), motor=w2, cmd=w2)
    for _ in range(5):
        q.step(w2)
    assert torch.equal(torch.sign(q.omega[0]), torch.sign(tau[0]))


def test_torque_geometry_front_motors_pitch_nose_up(nominal):
    """Front motors up -> nose up (FLU: negative rotation about +y) -> body x gains +z."""
    q = make(1, nominal)
    w = q.hover_motor_state()
    w[0, 0] += 0.05
    w[0, 1] += 0.05
    q.set_state(torch.arange(1), pos=torch.tensor([[0, 0, 5.0]]), motor=w, cmd=w)
    for _ in range(50):
        q.step(w)
    assert q.R[0, 2, 0] > 0  # body x axis now points upward


def test_quaternion_stays_normalized(nominal, rand):
    q = Quadrotor(64, nominal, rand, use_nominal=False, seed=3)
    idx = torch.arange(64)
    g = torch.Generator().manual_seed(1)
    q.set_state(idx, pos=torch.full((64, 3), 100.0), omega=torch.randn(64, 3, generator=g) * 20)
    for _ in range(3000):
        q.step(torch.rand(64, 4, generator=g))
        assert (q.quat.norm(dim=-1) - 1).abs().max() < 1e-5
    R = q.R
    eye = torch.eye(3).expand(64, 3, 3)
    assert (R @ R.transpose(1, 2) - eye).abs().max() < 1e-4


def test_quat_to_rotmat_convention():
    # +90° yaw: body x -> world y
    q = quat_from_axis_angle(torch.tensor([[0, 0, 1.0]]), torch.tensor([math.pi / 2]))
    R = quat_to_rotmat(q)[0]
    assert torch.allclose(R @ torch.tensor([1.0, 0, 0]), torch.tensor([0, 1.0, 0]), atol=1e-6)


@pytest.mark.parametrize("direction", ["up", "down"])
def test_motor_lag_time_constant(nominal, direction):
    q = make(1, nominal)
    q.params.latency_steps.zero_()
    start, end = (0.2, 0.8) if direction == "up" else (0.8, 0.2)
    tau = nominal["motor_tau_up"] if direction == "up" else nominal["motor_tau_down"]
    q.set_state(torch.arange(1), pos=torch.tensor([[0, 0, 100.0]]),
                motor=torch.full((1, 4), start), cmd=torch.full((1, 4), start))
    c = torch.full((1, 4), end)
    n = int(round(tau / q.dt))
    for _ in range(n):
        q.step(c)
    frac = (q.motor[0, 0].item() - start) / (end - start)
    assert abs(frac - (1 - math.exp(-1))) < 0.01
    # fit tau from a long trace
    trace = []
    for _ in range(100):
        q.step(c)
        trace.append((end - q.motor[0, 0].item()) / (end - start))
    est = -q.dt / math.log(trace[-1] / trace[-2])
    assert abs(est - tau) / tau < 0.02


@pytest.mark.parametrize("delay", [0, 1, 3, 8])
def test_latency_delays_effect(nominal, rand, delay):
    q = Quadrotor(1, nominal, rand, use_nominal=False, seed=0)
    q.params.latency_steps.fill_(delay)
    q.set_state(torch.arange(1), pos=torch.tensor([[0, 0, 100.0]]),
                motor=torch.zeros(1, 4), cmd=torch.zeros(1, 4))
    first_move = None
    for k in range(20):
        q.step(torch.ones(1, 4))
        if first_move is None and q.motor[0, 0] > 0:
            first_move = k
    assert first_move == delay
