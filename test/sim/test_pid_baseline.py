import math

import torch

from sim.env import EnvConfig, QuadEnv
from sim.pid_baseline import CascadedPID
from sim.rotations import quat_from_axis_angle


def _env(n, use_nominal, seed=0):
    return QuadEnv(EnvConfig(num_envs=n, seed=seed, use_nominal=use_nominal, episode_len_s=1e6,
                             randomize_initial_time=False))


def _run(env, pid, seconds, tail_s=1.0):
    """Run PID; return (failed mask, mean true position error over the last tail_s)."""
    obs = env.obs_buf.clone()
    failed = torch.zeros(env.n, dtype=torch.bool)
    steps = int(seconds / env.control_dt)
    tail = int(tail_s / env.control_dt)
    err_acc = torch.zeros(env.n)
    for k in range(steps):
        obs, _, done, info = env.step(pid(obs))
        failed |= info["crash"] | info["oob"]
        if done.any():
            pid.reset(done.nonzero(as_tuple=True)[0])
        if k >= steps - tail:
            err_acc += (env.quad.pos - env.target).norm(dim=-1)
    return failed, err_acc / tail


def _set(env, pos_off, tilt_deg, target=(0.0, 0.0, 1.5)):
    n = env.n
    idx = torch.arange(n)
    env.target[:] = torch.tensor(target)
    g = torch.Generator().manual_seed(5)
    phi = torch.rand(n, generator=g) * 2 * math.pi
    axis = torch.stack((torch.cos(phi), torch.sin(phi), torch.zeros(n)), -1)
    quat = quat_from_axis_angle(axis, torch.full((n,), math.radians(tilt_deg)))
    w = env.quad.hover_motor_state()
    env.quad.set_state(idx, pos=env.target + torch.tensor(pos_off), vel=torch.zeros(n, 3), quat=quat,
                       omega=torch.zeros(n, 3), motor=w, cmd=w)
    env.prev_action.zero_()
    env.last_gyro = env.quad.sample_gyro()
    env._compute_obs()


def test_pid_hover_from_1m_offset_nominal(nominal):
    env = _env(1, True)
    env.reset()
    pid = CascadedPID(1, nominal)
    _set(env, (0.6, -0.6, -0.529), 0.0)  # |offset| = 1 m
    failed, err = _run(env, pid, 6.0)
    assert not failed.any()
    assert err.item() < 0.05, err


def test_pid_recovers_from_60deg_tilt_nominal(nominal):
    env = _env(8, True)
    env.reset()
    pid = CascadedPID(8, nominal)
    _set(env, (0.0, 0.0, 0.0), 60.0)
    failed, err = _run(env, pid, 5.0)
    assert not failed.any()
    assert (err < 0.05).all(), err


def test_pid_randomized_success_rates(nominal):
    n = 256
    env = _env(n, False, seed=11)
    env.reset()
    results = {}
    for name, off, tilt in (("hover_1m", (0.6, -0.6, -0.529), 0.0), ("tilt_60", (0, 0, 0), 60.0)):
        pid = CascadedPID(n, nominal)
        env.reset()
        _set(env, off, tilt)
        failed, err = _run(env, pid, 6.0)
        ok = (~failed) & (err < 0.05)
        results[name] = ok.float().mean().item()
        print(f"\nPID randomized {name}: success {results[name]*100:.1f}% "
              f"(no-crash {(~failed).float().mean().item()*100:.1f}%, median err {err.median().item()*100:.1f} cm)")
    assert results["hover_1m"] >= 0.95
    assert results["tilt_60"] >= 0.95


def test_pid_recovers_from_upside_down_randomized(nominal):
    n = 256
    env = _env(n, False, seed=11)
    env.reset()
    pid = CascadedPID(n, nominal)
    _set(env, (0, 0, 0), 170.0, target=(0.0, 0.0, 2.0))
    failed, err = _run(env, pid, 6.0)
    no_crash = (~failed).float().mean().item()
    print(f"\nPID randomized upside-down(170deg, 2 m): no-crash {no_crash*100:.1f}%, "
          f"<5cm {((~failed) & (err < 0.05)).float().mean().item()*100:.1f}%")
    assert no_crash >= 0.99
    assert (err[~failed] < 0.10).all()
