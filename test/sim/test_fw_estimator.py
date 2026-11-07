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
