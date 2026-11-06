import torch

from sim.ctbr import Mixer, RateController, action_to_ctbr, ctbr_to_action
from sim.quadrotor import Quadrotor


def test_action_mapping_roundtrip(nominal):
    a = torch.tensor([[-1.0, 0, 0, 0], [1.0, 1, -1, 0.5], [0.0, 0.2, 0.3, -0.4]])
    F, w = action_to_ctbr(a, nominal)
    assert abs(F[0].item()) < 1e-6
    assert abs(F[1].item() - 4 * nominal["thrust_max"]) < 1e-5
    assert torch.allclose(w[1], torch.tensor([nominal["max_body_rate"][0], -nominal["max_body_rate"][1],
                                              0.5 * nominal["max_body_rate"][2]]))
    assert torch.allclose(ctbr_to_action(F, w, nominal), a, atol=1e-6)


def test_mixer_reproduces_wrench_when_unsaturated(nominal):
    m = Mixer(nominal)
    q = Quadrotor(3, nominal, use_nominal=True)
    F = torch.tensor([2.5, 3.0, 6.0])
    tau = torch.tensor([[0.0, 0, 0], [0.005, -0.004, 0.001], [-0.01, 0.01, -0.002]])
    cmd, sat = m.mix(F, tau)
    assert not sat.any()
    F2, tau2 = q.body_wrench(q.thrusts(cmd))
    assert torch.allclose(F2, F, atol=1e-4)
    assert torch.allclose(tau2, tau, atol=1e-5)


def test_mixer_saturation_prioritizes_roll_pitch(nominal):
    m = Mixer(nominal)
    q = Quadrotor(2, nominal, use_nominal=True)
    # low collective + big roll: collective is raised to keep roll; yaw dropped first
    F = torch.tensor([0.0, 17.5])
    tau = torch.tensor([[0.05, 0.0, 0.02], [0.0, -0.05, 0.02]])
    cmd, sat = m.mix(F, tau)
    assert sat.all()
    assert (cmd >= nominal["idle_command"] - 1e-6).all() and (cmd <= 1 + 1e-6).all()
    _, tau2 = q.body_wrench(q.thrusts(cmd))
    assert torch.allclose(tau2[0, 0], tau[0, 0], rtol=1e-3)   # roll fully delivered
    assert torch.allclose(tau2[1, 1], tau[1, 1], rtol=1e-3)   # pitch fully delivered
    assert (tau2[:, 2].abs() < tau[:, 2].abs()).all()        # yaw sacrificed
    # extreme roll+pitch: direction preserved under scaling
    cmd, _ = m.mix(torch.tensor([9.0]), torch.tensor([[1.0, 0.5, 0.0]]))
    _, t3 = q.body_wrench(q.thrusts(cmd))
    assert abs(t3[0, 1].item() / t3[0, 0].item() - 0.5) < 1e-3


def test_rate_loop_tracks_step(nominal):
    n = 1
    q = Quadrotor(n, nominal, use_nominal=True)
    rc = RateController(n, nominal)
    w = q.hover_motor_state()
    q.set_state(torch.arange(n), pos=torch.tensor([[0, 0, 50.0]]), motor=w, cmd=w)
    F = torch.tensor([nominal["mass"] * nominal["gravity"]])
    sp = torch.tensor([[3.0, -2.0, 1.5]])
    for _ in range(600):
        q.step(rc(F, sp, q.sample_gyro()))
    assert (q.omega - sp).abs().max() < 0.15  # combined-axis (gyroscopic coupling)
    # single-axis step: 90 % rise within 120 ms, within 3 % by 1 s
    q.set_state(torch.arange(n), omega=torch.zeros(1, 3), motor=w, cmd=w)
    rc.reset(torch.arange(n))
    sp = torch.tensor([[3.0, 0.0, 0.0]])
    trace = []
    for _ in range(1000):
        q.step(rc(F, sp, q.sample_gyro()))
        trace.append(q.omega[0, 0].item())
    assert trace[120] > 2.7
    assert abs(trace[-1] - 3.0) < 0.09
    assert max(trace) < 3.0 * 1.25
