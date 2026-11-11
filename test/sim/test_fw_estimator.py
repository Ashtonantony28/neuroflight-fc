"""sim/fw_estimator.py (firmware sensing path in the sim). Bit-level parity
with the C++ filters is tested in test/sitl (test_gyro_filter_chain_matches_firmware)."""
import math

import torch

from sim.env import EnvConfig, QuadEnv
from sim.fw_estimator import GyroFilterChain


def test_gyro_filter_reset_is_steady_state():
    n = 5
    f = GyroFilterChain(n)
    v = torch.randn(n, 3)
    hz = torch.tensor([[50.0, 120.0, 200.0, 90.0]]).expand(n, 4)
    f.reset(torch.arange(n), v, hz)
    for _ in range(50):
        out = f.step(v, hz)
        assert torch.allclose(out, v, atol=1e-5)


def test_obs_rates_are_filtered_bias_corrected_gyro():
    env = QuadEnv(EnvConfig(num_envs=64, seed=2, randomize_initial_time=False, init_max_rate=0.0))
    env.reset()
    q = env.quad
    # at reset: true rate + bias - bias estimate (residual ~1e-3)
    assert (env.last_gyro - q.omega).abs().max() < 6e-3
    a = env.hover_action()
    err = []
    for _ in range(200):
        obs, _, _, _ = env.step(a)
        err.append((obs[:, 12:15] - q.omega).pow(2).mean().sqrt())
    # noise std <= 0.05 rad/s filtered at 150 Hz, no bias
    assert torch.stack(err).mean() < 0.03


def test_kf_tracks_hover_and_feeds_obs():
    env = QuadEnv(EnvConfig(num_envs=32, seed=4, randomize_initial_time=False, estimator=True, init_max_tilt_deg=10,
                            init_max_vel=0.3, init_max_rate=0.3, init_pos_offset=0.2))
    obs = env.reset()
    q = env.quad
    assert torch.allclose(obs[:, 3:6], q.vel, atol=1e-6)            # KF starts at the true state
    a = env.hover_action()
    for _ in range(500):
        obs, _, done, _ = env.step(a)
    assert not done.any()
    assert torch.allclose(obs[:, 3:6], env.kf.v.clamp(-5, 5))
    assert (env.kf.v - q.vel).abs().median() < 0.05
    assert (env.kf.p[:, 2] - q.pos[:, 2]).abs().max() < 0.05     # ToF-corrected height
    assert (env.kf.height_age < 0.2).all() and (env.kf.flow_age < 0.2).all()


def test_kf_coasts_when_tilted():
    env = QuadEnv(EnvConfig(num_envs=8, seed=5, randomize_initial_time=False, estimator=True,
                            init_max_tilt_deg=0, init_upside_down_prob=1.0))
    env.reset()
    for _ in range(50):
        env.step(env.hover_action())
    # upside down: no ToF / flow updates (tilt gate), height/flow ages grow
    assert (env.kf.height_age > 0.09).all()


def test_envelope_terms():
    env = QuadEnv(EnvConfig(num_envs=3, seed=0, use_nominal=True, randomize_initial_time=False))
    env.reset()
    from sim.rotations import quat_from_axis_angle
    ax = torch.tensor([[1.0, 0.0, 0.0]]).expand(3, 3)
    ang = torch.tensor([0.0, math.radians(56), math.radians(40)])
    w = torch.tensor([[0.0, 0, 0], [0.0, 0, 0], [6.0, 0, 0]])
    env.quad.set_state(torch.arange(3), quat=quat_from_axis_angle(ax, ang), omega=w)
    tilt, tr = env.tilt_and_rate()
    assert torch.allclose(tilt, ang, atol=1e-4)
    assert abs(tr[2].item() - 6.0) < 1e-4                         # rolling away from level
    pen_pred, pen_tilt = env._envelope_terms()
    assert pen_pred[0] == 0 and pen_tilt[0] == 0
    assert pen_tilt[1] > 0 and pen_pred[1] == 0                   # static tilt: TILT margin only
    assert pen_pred[2] > 0 and pen_tilt[2] == 0                   # 40 deg + 0.6 rad > 50 deg at 6 rad/s


def _att_err_deg(Ra, Rb):
    c = (Ra[:, :, 2] * Rb[:, :, 2]).sum(-1).clamp(-1, 1)    # angle between body z axes
    return torch.rad2deg(torch.acos(c))


def test_mahony_obs_tracks_attitude_in_hover():
    """EnvConfig(mahony=True): attitude obs 6-11 are the Mahony estimate (FOLLOWUPS 8)."""
    # nominal vehicle, level start: the open-loop hover action really hovers (a tilted or randomized one
    # accelerates sideways for good, the sustained-acceleration case tested below)
    env = QuadEnv(EnvConfig(num_envs=64, seed=7, use_nominal=True, randomize_initial_time=False, estimator=True,
                            mahony=True, init_max_tilt_deg=0, init_max_vel=0.2, init_max_rate=0.0, init_pos_offset=0.1))
    obs = env.reset()
    q = env.quad
    assert torch.allclose(obs[:, 6:9], q.R[:, :, 0], atol=1e-5)      # starts at the true attitude
    a = env.hover_action()
    for _ in range(1000):
        obs, _, done, _ = env.step(a)
    assert not done.any()
    R_est = env.mahony.R()
    assert torch.allclose(obs[:, 6:9], R_est[:, :, 0]) and torch.allclose(obs[:, 9:12], R_est[:, :, 1])
    assert _att_err_deg(R_est, q.R).median() < 1.0
    assert env.mahony.innov.max() < 0.1
    assert (env.mahony.bias - q.params.gyro_bias).abs().max() < 0.01
    assert torch.allclose(obs[:, 12:15], env.last_gyro)


def _sustained_accel_error(kf_ref: bool) -> float:
    """Level-flight hold at 30 deg tilt (coordinated horizontal acceleration at
    g tan 30 deg, accel sees thrust only) for 0.6 s; returns the median final
    Mahony tilt error (deg)."""
    from sim.rotations import quat_from_axis_angle
    n = 16
    env = QuadEnv(EnvConfig(num_envs=n, seed=8, use_nominal=True, randomize_initial_time=False, estimator=True,
                            mahony=True, mahony_kf_ref=kf_ref, gyro_bias_resid_std=0.0))
    env.reset()
    th = math.radians(30)
    ax = torch.tensor([[0.0, 1.0, 0.0]]).expand(n, 3)
    idx = torch.arange(n)
    env.quad.set_state(idx, pos=torch.tensor([[0.0, 0.0, 1.5]]).expand(n, 3).clone(), vel=torch.zeros(n, 3),
                       quat=quat_from_axis_angle(ax, torch.full((n,), th)), omega=torch.zeros(n, 3))
    env.target[:] = torch.tensor([0.0, 0.0, 1.5])
    env.prev_action.zero_()
    env.reinit_sensing(idx)
    a = env.hover_action()
    nom = env.nom
    a[:, 0] = nom["mass"] * nom["gravity"] / math.cos(th) / (4 * nom["thrust_max"]) * 2 - 1
    for _ in range(300):
        _, _, done, _ = env.step(a)
    assert not done.any()
    return float(_att_err_deg(env.mahony.R(), env.quad.R).median())


def test_mahony_kf_reference_under_sustained_acceleration():
    """Mirrors fw test mahony_kf_acceleration_reference_while_braking: against
    gravity the estimate is pulled towards level; with the KF-acceleration
    reference it stays on the true attitude."""
    plain, ref = _sustained_accel_error(False), _sustained_accel_error(True)
    print(f"tilt error after 0.6 s at 30 deg: gravity ref {plain:.2f} deg, KF-accel ref {ref:.2f} deg")
    assert plain > 5.0
    assert ref < 0.4 * plain and ref < 3.0


def test_rate_margin_estimator_and_jerk_terms():
    """v3 reward terms: two identical envs (same seed), one with the new weights;
    the reward difference equals the documented penalties."""
    from sim.env import DEFAULT_REWARD_WEIGHTS
    from sim.rotations import quat_from_axis_angle
    w2 = dict(DEFAULT_REWARD_WEIGHTS, rate_margin=1.0, est_nis=1.0, est_innov=1.0, thrust_jerk=1.0)
    envs = [QuadEnv(EnvConfig(num_envs=4, seed=0, use_nominal=True, randomize_initial_time=False, estimator=True,
                              mahony=True, reward_weights=w)) for w in (dict(DEFAULT_REWARD_WEIGHTS), w2)]
    for e in envs:
        e.reset()
        idx = torch.arange(4)
        e.quad.set_state(idx, omega=torch.tensor([[0.0, 0, 0], [11.0, 0, 0], [0, -14.0, 0], [0, 0, 5.0]]),
                         quat=quat_from_axis_angle(torch.tensor([[1.0, 0, 0]]).expand(4, 3), torch.full((4,), 0.3)))
        e.reinit_sensing(idx)
        e._compute_obs()
    a = envs[0].hover_action()
    prev = [torch.zeros(4), a[:, 0].clone()]
    for k, da in enumerate((0.0, 0.05, -0.02)):
        ak = a.clone()
        ak[:, 0] += da
        r = [e.step(ak)[1] for e in envs]
        e0 = envs[0]
        assert torch.equal(e0.last_gyro, envs[1].last_gyro)
        pen = (e0.last_gyro.abs().amax(-1) - 10.0).clamp_min(0.0)
        pen = pen + (e0.kf.nis_lpf - 8.0).clamp_min(0.0) + (e0.mahony.innov - 0.25).clamp_min(0.0)
        if k >= 2:   # second difference only from the third step after reset
            pen = pen + (ak[:, 0] - 2 * prev[-1] + prev[-2]) ** 2
        prev.append(ak[:, 0].clone())
        assert torch.allclose(r[0] - r[1], pen, atol=1e-4), (k, r[0] - r[1], pen)
    assert (pen[1:3] > 0).all()
