import math
import time

import pytest
import torch

from sim.env import ACT_DIM, OBS_DIM, EnvConfig, QuadEnv
from sim.rotations import quat_from_axis_angle


def test_obs_layout_matches_spec():
    env = QuadEnv(EnvConfig(num_envs=8, seed=1, use_nominal=True, randomize_initial_time=False))
    obs = env.reset()
    assert obs.shape == (8, OBS_DIM) and obs.dtype == torch.float32
    q = env.quad
    # nominal => no sensor noise/bias: obs is exact
    assert torch.allclose(obs[:, 0:3], (q.pos - env.target).clamp(-2, 2), atol=1e-6)
    assert torch.allclose(obs[:, 3:6], q.vel.clamp(-5, 5), atol=1e-6)
    assert torch.allclose(obs[:, 6:9], q.R[:, :, 0])
    assert torch.allclose(obs[:, 9:12], q.R[:, :, 1])
    assert torch.allclose(obs[:, 12:15], q.omega, atol=1e-6)
    assert (obs[:, 15:19] == 0).all()
    a = torch.rand(8, ACT_DIM) * 2 - 1
    obs, rew, done, info = env.step(a)
    live = ~done
    assert torch.allclose(obs[live, 15:19], a[live])
    assert torch.allclose(obs[live, 12:15], q.omega[live], atol=1e-6)
    assert torch.allclose(obs[live, 6:9], q.R[live, :, 0])
    # clipping
    env.quad.set_state(torch.arange(1), pos=env.target[:1] + torch.tensor([[2.5, -2.5, 0.0]]),
                       vel=torch.tensor([[7.0, 0, -7.0]]))
    env._compute_obs()
    assert torch.allclose(env.obs_buf[0, 0:2], torch.tensor([2.0, -2.0]))
    assert torch.allclose(env.obs_buf[0, 3:6], torch.tensor([5.0, 0, -5.0]))
    assert rew.shape == (8,) and done.dtype == torch.bool


def test_hover_action_holds_in_env():
    env = QuadEnv(EnvConfig(num_envs=4, seed=0, use_nominal=True, randomize_initial_time=False,
                            init_max_tilt_deg=0, init_max_vel=0, init_max_rate=0, init_pos_offset=0))
    env.reset()
    p0 = env.quad.pos.clone()
    a = env.hover_action()
    for _ in range(500):
        _, _, done, _ = env.step(a)
    assert not done.any()
    assert (env.quad.pos - p0).abs().max() < 0.02


def test_determinism_with_seed():
    def run(seed):
        env = QuadEnv(EnvConfig(num_envs=64, seed=seed, init_upside_down_prob=0.2))
        obs = [env.reset()]
        g = torch.Generator().manual_seed(123)
        rews = []
        for _ in range(300):
            o, r, d, _ = env.step(torch.rand(64, 4, generator=g) * 2 - 1)
            obs.append(o); rews.append(r)
        return torch.stack(obs), torch.stack(rews)
    o1, r1 = run(7)
    o2, r2 = run(7)
    o3, _ = run(8)
    assert torch.equal(o1, o2) and torch.equal(r1, r2)
    assert not torch.equal(o1, o3)


def test_termination_and_autoreset():
    env = QuadEnv(EnvConfig(num_envs=16, seed=0, use_nominal=True, randomize_initial_time=False,
                            episode_len_s=0.1))
    env.reset()
    # zero thrust -> falls -> crash or timeout; never both
    seen_crash = seen_timeout = False
    for _ in range(400):
        obs, r, d, info = env.step(torch.tensor([[-1.0, 0, 0, 0]]).expand(16, 4))
        assert not (info["crash"] & info["timeout"]).any()
        seen_crash |= bool(info["crash"].any())
        seen_timeout |= bool(info["timeout"].any())
        if d.any():
            assert "terminal_obs" in info
            assert (env.step_count[d] == 0).all()
            assert (obs[d, 15:19] == 0).all()
    assert seen_timeout
    env2 = QuadEnv(EnvConfig(num_envs=16, seed=0, use_nominal=True, randomize_initial_time=False,
                             episode_len_s=10.0, init_pos_offset=0.0, target_z=(0.6, 0.6)))
    env2.reset()
    crashed = torch.zeros(16, dtype=torch.bool)
    for _ in range(400):
        _, r, d, info = env2.step(torch.tensor([[-1.0, 0, 0, 0]]).expand(16, 4))
        crashed |= info["crash"]
        if info["crash"].any():
            assert (r[info["crash"]] < -env2.w["crash"] + 5).all()
    assert crashed.all()


def test_curriculum_upside_down_init():
    env = QuadEnv(EnvConfig(num_envs=2000, seed=0, init_upside_down_prob=1.0))
    env.reset()
    assert (env.quad.R[:, 2, 2] < math.cos(math.radians(149))).all()
    env = QuadEnv(EnvConfig(num_envs=2000, seed=0, init_max_tilt_deg=30))
    env.reset()
    assert (env.quad.R[:, 2, 2] > math.cos(math.radians(30.01))).all()


def test_reward_weights_tunable():
    w = {"alive": 0.0, "pos": 0.0, "vel": 0.0, "upright": 0.0, "rate": 0.0, "yaw_rate": 0.0,
         "action": 0.0, "smooth": 1.0, "crash": 0.0}
    env = QuadEnv(EnvConfig(num_envs=4, seed=0, use_nominal=True, reward_weights=w,
                            randomize_initial_time=False))
    env.reset()
    a = torch.full((4, 4), 0.5)
    _, r, _, _ = env.step(a)
    assert torch.allclose(r, torch.full((4,), -4 * 0.25))


@pytest.mark.slow
def test_throughput_4096():
    torch.set_num_threads(4)
    env = QuadEnv(EnvConfig(num_envs=4096, seed=0))
    env.reset()
    a = torch.zeros(4096, 4)
    for _ in range(10):
        env.step(a)
    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        env.step(a)
    sps = n * 4096 / (time.perf_counter() - t0)
    print(f"\nthroughput: {sps:,.0f} env-steps/s (4096 envs, 4 threads)")
    assert sps > 50_000
