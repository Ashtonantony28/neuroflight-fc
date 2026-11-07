import math

import torch

from sim.env import OBS_DIM
from train.ppo import PPO, CRITIC_DIM, RunningMeanStd, compute_gae, train
from train.priv_env import PRIV_DIM, PrivilegedQuadEnv


def test_gae_matches_hand_computation():
    g, lam = 0.9, 0.8
    # 3 steps, 1 env; episode ends after step 1 (index 1), then a new one starts
    r = torch.tensor([[1.0], [2.0], [3.0]])
    v = torch.tensor([[0.5], [1.0], [1.5]])
    d = torch.tensor([[False], [True], [False]])
    last = torch.tensor([2.0])
    adv, ret = compute_gae(r, v, d, last, g, lam)
    d2 = 3.0 + g * 2.0 - 1.5                 # t=2 bootstraps last value
    d1 = 2.0 - 1.0                           # t=1 terminal
    d0 = 1.0 + g * 1.0 - 0.5
    a2 = d2
    a1 = d1
    a0 = d0 + g * lam * a1
    exp = torch.tensor([[a0], [a1], [a2]])
    assert torch.allclose(adv, exp, atol=1e-6)
    assert torch.allclose(ret, exp + v, atol=1e-6)


def test_gae_lambda1_equals_discounted_return():
    g = 0.95
    T = 6
    r = torch.arange(1.0, T + 1).unsqueeze(-1)
    v = torch.zeros(T, 1)
    d = torch.zeros(T, 1, dtype=torch.bool)
    _, ret = compute_gae(r, v, d, torch.tensor([10.0]), g, 1.0)
    exp0 = sum(g ** k * (k + 1) for k in range(T)) + g ** T * 10.0
    assert math.isclose(ret[0, 0].item(), exp0, rel_tol=1e-6)


def test_running_mean_std():
    rms = RunningMeanStd(3)
    g = torch.Generator().manual_seed(0)
    xs = [torch.randn(100, 3, generator=g) * 2 + 1 for _ in range(5)]
    for x in xs:
        rms.update(x)
    allx = torch.cat(xs).double()
    assert torch.allclose(rms.mean, allx.mean(0), atol=1e-3)
    assert torch.allclose(rms.var, allx.var(0, unbiased=False), atol=1e-2)


def test_privileged_env_is_behaviour_neutral_and_captures_terminal():
    from sim.env import EnvConfig, QuadEnv
    kw = dict(num_envs=32, seed=3, episode_len_s=0.05, init_max_tilt_deg=120)
    a_env, b_env = QuadEnv(EnvConfig(**kw)), PrivilegedQuadEnv(EnvConfig(**kw))
    oa, ob = a_env.reset(), b_env.reset()
    assert b_env.privileged().shape == (32, PRIV_DIM)
    g = torch.Generator().manual_seed(1)
    saw_terminal = False
    for _ in range(40):
        act = torch.rand(32, 4, generator=g) * 2 - 1
        oa, ra, da, ia = a_env.step(act)
        ob, rb, db, ib = b_env.step(act)
        assert torch.equal(oa, ob) and torch.equal(ra, rb) and torch.equal(da, db)
        if db.any():
            saw_terminal = True
            tp = ib["terminal_priv"][db]
            # true pos err in priv is consistent with the (noisy, clipped) terminal obs
            assert ((tp[:, 0:3].clamp(-2, 2) - ib["terminal_obs"][db, 0:3]).abs() < 0.3).all()
            # terminal priv differs from the post-reset priv
            assert not torch.allclose(tp, b_env.privileged()[db])
    assert saw_terminal


def test_ppo_smoke(small_cfg, tmp_path):
    algo = train(small_cfg, str(tmp_path), iterations=3, log=lambda *a: None)
    assert algo.it == 3
    s = algo.iterate()
    for k in ("pg", "vf", "kl", "ent"):
        assert math.isfinite(s[k]), (k, s[k])
    assert (tmp_path / "progress.csv").exists() and (tmp_path / "last.pt").exists()
    for prm in list(algo.actor.parameters()) + list(algo.critic.parameters()):
        assert torch.isfinite(prm).all()


def test_ppo_deterministic(small_cfg, tmp_path):
    a = PPO(small_cfg); sa = a.iterate()
    b = PPO(small_cfg); sb = b.iterate()
    assert sa["pg"] == sb["pg"] and sa["vf"] == sb["vf"]
    for pa, pb in zip(a.actor.parameters(), b.actor.parameters()):
        assert torch.equal(pa, pb)


def test_actor_starts_near_hover(small_cfg, nominal):
    from train.ppo import hover_a0
    algo = PPO(small_cfg)
    m, s = algo.obs_rms.frozen()
    with torch.no_grad():
        a = algo.actor((algo.obs - m) / s)
    assert (a[:, 0] - hover_a0(nominal)).abs().max() < 0.1
    assert CRITIC_DIM == OBS_DIM + PRIV_DIM


def test_pos_exp_extra_reward():
    from sim.env import EnvConfig
    kw = dict(num_envs=16, seed=4)
    a = PrivilegedQuadEnv(EnvConfig(**kw))
    b = PrivilegedQuadEnv(EnvConfig(**kw), extra_reward={"pos_exp": {"weight": 2.0, "sigma": 0.5}})
    a.reset(); b.reset()
    act = a.hover_action()
    _, ra, _, ia = a.step(act)
    _, rb, _, _ = b.step(act)
    assert torch.allclose(rb - ra, 2.0 * torch.exp(-(ia["dist"] / 0.5) ** 2), atol=1e-5)
