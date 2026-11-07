"""Evaluation suite (PLAN §4.2 item 4) for CTBR controllers.

    python -m train.eval runs/<name>/policy.nfpol.json [--n 512] [--md out.md]

Scenarios (target fixed at (0,0,2) m, 5 s horizon, true-state metrics):
  hover_1m     1 m offset in a random direction, level, at rest
  tilt_60      exactly 60° tilt about a random horizontal axis, at rest
  throw        tilt U[0,30°], |v| U[0,3] m/s, |ω| U[0,10] rad/s (random directions)
  upside_down  tilt U[150°,180°], at rest
each in `nominal` (nominal params, no sensor noise) and `randomized`
(config/randomization.yaml) mode, plus every randomized parameter at its
min and max (others nominal) on tilt_60 and throw.

Success = no crash/out-of-bounds within 5 s, mean |p - p_target| over the
last 0.5 s < POS_TOL and final tilt < 10°. All controllers are run on the
same env seed, so they see identical vehicles and initial states.

Smoothness: RMS of the action content above 50 Hz (rfft over the last
4.096 s of each surviving env, mean removed; Parseval), per channel group.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.env import EnvConfig, QuadEnv  # noqa: E402
from sim.params import load_nominal, load_randomization, sample_params  # noqa: E402
from sim.pid_baseline import CascadedPID  # noqa: E402
from sim.rotations import quat_from_axis_angle, quat_mul  # noqa: E402

SCENARIOS = ("hover_1m", "tilt_60", "throw", "upside_down")
EXTREME_SCENARIOS = ("tilt_60", "throw")
TARGET = (0.0, 0.0, 2.0)
HORIZON_S = 5.0
POS_TOL = 0.10
TILT_TOL_DEG = 10.0
HF_CUTOFF_HZ = 50.0
HF_WINDOW = 2048


# ---------------------------------------------------------------------------- controllers
class TorchPolicy:
    """Deterministic CTBR policy from an nfpol dict (same math as export.numpy_forward)."""

    def __init__(self, pol: dict | str):
        if isinstance(pol, str):
            with open(pol) as f:
                pol = json.load(f)
        self.mean = torch.tensor(pol["obs_mean"], dtype=torch.float32)
        self.std = torch.tensor(pol["obs_std"], dtype=torch.float32)
        self.layers = [(torch.tensor(L["W"], dtype=torch.float32), torch.tensor(L["b"], dtype=torch.float32),
                        L["act"]) for L in pol["layers"]]

    def reset(self, idx=None):
        pass

    @torch.no_grad()
    def __call__(self, obs):
        x = (obs - self.mean) / self.std
        for W, b, act in self.layers:
            x = x @ W.T + b
            if act == "tanh":
                x = torch.tanh(x)
        return x


def make_pid(n, nominal):
    return CascadedPID(n, nominal)


# ---------------------------------------------------------------------------- setup
@dataclass
class Group:
    name: str          # e.g. "throw/randomized" or "extreme:mass=min/tilt_60"
    scenario: str
    mode: str          # nominal | randomized | extreme
    sl: slice


def _rand_unit(n, g):
    v = torch.randn(n, 3, generator=g)
    return v / v.norm(dim=-1, keepdim=True)


def _tilt_quat(tilt_rad, g):
    n = tilt_rad.shape[0]
    phi = torch.rand(n, generator=g) * 2 * math.pi
    yaw = (torch.rand(n, generator=g) * 2 - 1) * math.pi
    axis = torch.stack((torch.cos(phi), torch.sin(phi), torch.zeros(n)), -1)
    ez = torch.tensor([0.0, 0.0, 1.0]).expand(n, 3)
    return quat_mul(quat_from_axis_angle(axis, tilt_rad), quat_from_axis_angle(ez, yaw))


def initial_states(scenario, n, g):
    z = torch.zeros(n, 3)
    pos = torch.tensor(TARGET).expand(n, 3).clone()
    vel, omega = z.clone(), z.clone()
    if scenario == "hover_1m":
        pos = pos + _rand_unit(n, g)
        tilt = torch.zeros(n)
    elif scenario == "tilt_60":
        tilt = torch.full((n,), math.radians(60))
    elif scenario == "throw":
        tilt = torch.rand(n, generator=g) * math.radians(30)
        vel = _rand_unit(n, g) * (3.0 * torch.rand(n, 1, generator=g))
        omega = _rand_unit(n, g) * (10.0 * torch.rand(n, 1, generator=g))
    elif scenario == "upside_down":
        tilt = math.radians(150) + torch.rand(n, generator=g) * math.radians(30)
    else:
        raise ValueError(scenario)
    return pos, vel, _tilt_quat(tilt, g), omega


def extreme_keys(rand: dict):
    return list(rand.keys())


def apply_extreme(p, idx, key, which, rand, nominal, dt, max_delay):
    """Set param `key` to its randomization min/max on rows idx (others untouched)."""
    lo, hi = rand[key]
    v = lo if which == "min" else hi
    m = idx.numel()
    if key == "mass": p.mass[idx] *= v
    elif key == "inertia": p.inertia[idx] *= v
    elif key == "thrust_max": p.thrust_max[idx] *= v
    elif key == "per_motor_thrust_imbalance":
        # min: diagonal pattern (yaw imbalance); max: front/back pattern (pitch imbalance)
        pat = torch.tensor([lo, hi, lo, hi]) if which == "min" else torch.tensor([hi, hi, lo, lo])
        p.thrust_max[idx] *= pat
    elif key == "yaw_torque_coeff": p.yaw_k[idx] *= v
    elif key == "motor_tau":
        p.tau_up[idx] *= v; p.tau_down[idx] *= v
    elif key == "lin_drag": p.lin_drag[idx] *= v
    elif key == "latency_s":
        p.latency_steps[idx] = min(int(round(v / dt)), max_delay)
    elif key == "gyro_noise_std": p.gyro_noise_std[idx] = v
    elif key == "gyro_bias": p.gyro_bias[idx] = v
    elif key == "accel_noise_std": p.accel_noise_std[idx] = v
    elif key == "vel_noise_std": p.vel_noise_std[idx] = v
    elif key == "pos_noise_std": p.pos_noise_std[idx] = v
    else:
        raise KeyError(f"eval: no extreme handler for randomization key {key}")


SENSING = {
    # the v1 training env: raw gyro into PID/obs, truth + white noise pos/vel
    "v1": dict(gyro_filter=False, estimator=False),
    # firmware sensing path (sim/fw_estimator.py): filtered bias-corrected gyro,
    # flow/ToF KF position/velocity -- what the policy sees on the vehicle
    "fw": dict(gyro_filter=True, estimator=True),
}


def build(n: int, n_ext: int, seed: int, extremes: bool = True, scenarios=SCENARIOS, sensing: str = "v1"):
    nominal, rand = load_nominal(), load_randomization()
    groups, rows = [], 0
    for sc in scenarios:
        for mode in ("nominal", "randomized"):
            groups.append(Group(f"{sc}/{mode}", sc, mode, slice(rows, rows + n))); rows += n
    keys = extreme_keys(rand) if extremes else []
    for k in keys:
        for which in ("min", "max"):
            for sc in EXTREME_SCENARIOS:
                groups.append(Group(f"{k}={which}/{sc}", sc, "extreme", slice(rows, rows + n_ext)))
                rows += n_ext
    env = QuadEnv(EnvConfig(num_envs=rows, seed=seed, use_nominal=False, episode_len_s=1e6,
                            randomize_initial_time=False, **SENSING[sensing]), nominal=nominal, randomization=rand)
    env.reset()
    q = env.quad
    g = torch.Generator().manual_seed(seed + 12345)
    for gr in groups:
        idx = torch.arange(gr.sl.start, gr.sl.stop)
        if gr.mode != "randomized":
            nomp = sample_params(idx.numel(), nominal, None, q.dt, use_nominal=True, max_latency_steps=q.max_delay)
            q.params.assign(idx, nomp)
        if gr.mode == "extreme":
            k, which = gr.name.split("/")[0].split("=")
            apply_extreme(q.params, idx, k, which, rand, nominal, q.dt, q.max_delay)
    q._update_derived(torch.arange(rows))
    env.target[:] = torch.tensor(TARGET)
    for gr in groups:
        idx = torch.arange(gr.sl.start, gr.sl.stop)
        pos, vel, quat, omega = initial_states(gr.scenario, idx.numel(), g)
        w = q.hover_motor_state(idx)
        q.set_state(idx, pos=pos, vel=vel, quat=quat, omega=omega, motor=w, cmd=w)
    env.prev_action.zero_()
    env.reinit_sensing(torch.arange(rows))   # filters/KF/rate PID for the new states (no D kick)
    env._compute_obs()
    return env, groups, nominal


# ---------------------------------------------------------------------------- run
@torch.no_grad()
def rollout(env, ctrl, horizon_s=HORIZON_S):
    n = env.n
    steps = int(round(horizon_s / env.control_dt))
    tail = int(round(0.5 / env.control_dt))
    obs = env.obs_buf.clone()
    failed = torch.zeros(n, dtype=torch.bool)
    max_tilt = torch.zeros(n)
    err_tail = torch.zeros(n)
    last_bad = torch.zeros(n)        # last time (s) the env was outside the settle band
    win = min(HF_WINDOW, steps)
    acts = torch.zeros(win, n, 4)
    # firmware supervisor envelope monitors on the true state (fw/core supervisor.cpp):
    # TILT_PREDICTED: tilt rate >= 4 rad/s and tilt + 0.1 s * rate > 1.05 rad for 20 ms;
    # TILT: tilt > 1.05 rad for 50 ms. Counted from t = 0 (includes the initial state).
    hold_pred = int(round(0.020 / env.control_dt))
    hold_tilt = int(round(0.050 / env.control_dt))
    c_pred = torch.zeros(n, dtype=torch.long)
    c_tilt = torch.zeros(n, dtype=torch.long)
    trip = torch.zeros(n, dtype=torch.bool)
    for k in range(steps):
        a = ctrl(obs).clamp(-1, 1)
        obs, _, done, info = env.step(a)
        q = env.quad
        live = ~failed
        tilt = torch.rad2deg(torch.acos(q.R[:, 2, 2].clamp(-1, 1)))
        err = (q.pos - env.target).norm(dim=-1)
        max_tilt = torch.where(live, torch.maximum(max_tilt, tilt), max_tilt)
        bad = (err > 0.2) | (tilt > 15.0)
        last_bad = torch.where(live & bad, torch.full_like(last_bad, (k + 1) * env.control_dt), last_bad)
        if k >= steps - tail:
            err_tail = torch.where(live, err_tail + err / tail, err_tail)
        if k >= steps - win:
            acts[k - (steps - win)] = a
        tl, tr = env.tilt_and_rate()
        pred_on = (tr >= 4.0) & (tl + 0.1 * tr.clamp_min(0) > 1.05)
        c_pred = torch.where(pred_on, c_pred + 1, torch.zeros_like(c_pred))
        c_tilt = torch.where(tl > 1.05, c_tilt + 1, torch.zeros_like(c_tilt))
        trip |= live & ((c_pred >= hold_pred) | (c_tilt >= hold_tilt))
        newly = (info["crash"] | info["oob"]) & live
        failed |= newly
    final_tilt = torch.rad2deg(torch.acos(env.quad.R[:, 2, 2].clamp(-1, 1)))
    success = ~failed & (err_tail < POS_TOL) & (final_tilt < TILT_TOL_DEG)
    # high-frequency action content
    x = acts.permute(1, 2, 0).numpy().astype(np.float64)          # (n, 4, win)
    x = x - x.mean(-1, keepdims=True)
    X = np.fft.rfft(x, axis=-1)
    f = np.fft.rfftfreq(win, d=env.control_dt)
    hf_mask = f > HF_CUTOFF_HZ
    P = np.abs(X) ** 2 / win ** 2
    P[..., 1:] *= 2.0
    if win % 2 == 0:
        P[..., -1] /= 2.0
    hf_ms = (P[..., hf_mask]).sum(-1)                              # mean square above cutoff (n, 4)
    return {"success": success, "failed": failed, "err_tail": err_tail, "max_tilt": max_tilt,
            "settle": last_bad, "trip": trip, "hf_thrust": torch.from_numpy(np.sqrt(hf_ms[:, 0])).float(),
            "hf_rates": torch.from_numpy(np.sqrt(hf_ms[:, 1:].mean(-1))).float()}


def summarize(res, groups):
    out = {}
    for gr in groups:
        s = gr.sl
        ok = res["success"][s]
        surv = ~res["failed"][s]
        d = {"n": int(ok.numel()), "success": ok.float().mean().item(),
             "crash": res["failed"][s].float().mean().item(),
             "trip": res["trip"][s].float().mean().item(),
             "worst_tilt": res["max_tilt"][s][surv].max().item() if surv.any() else float("nan"),
             "p95_tilt": res["max_tilt"][s][surv].quantile(0.95).item() if surv.any() else float("nan"),
             "err_med": res["err_tail"][s][surv].median().item() if surv.any() else float("nan"),
             "settle_mean": res["settle"][s][ok].mean().item() if ok.any() else float("nan"),
             "hf_thrust": res["hf_thrust"][s][surv].mean().item() if surv.any() else float("nan"),
             "hf_rates": res["hf_rates"][s][surv].mean().item() if surv.any() else float("nan")}
        out[gr.name] = d
    return out


def evaluate(controllers: dict, n=512, n_ext=64, seed=123, extremes=True, scenarios=SCENARIOS, threads=4,
             sensing: str = "v1"):
    """controllers: {name: factory(num_envs, nominal) -> ctrl}. Returns {name: {group: metrics}}."""
    torch.set_num_threads(threads)
    results = {}
    for name, factory in controllers.items():
        env, groups, nominal = build(n, n_ext, seed, extremes, scenarios, sensing)
        ctrl = factory(env.n, nominal)
        results[name] = summarize(rollout(env, ctrl), groups)
    return results


# ---------------------------------------------------------------------------- report
def _f(x, fmt):
    return "  nan" if x != x else format(x, fmt)


def format_tables(results: dict, ref: str = "PID") -> str:
    names = list(results)
    first = results[names[0]]
    lines = []
    main_groups = [g for g in first if "=" not in g]
    ext_groups = [g for g in first if "=" in g]
    hdr = "| scenario | " + " | ".join(
        f"{c} {nm}" for nm in names for c in ("succ%",)) + " | " + " | ".join(
        f"{nm} err_med m" for nm in names) + " | " + " | ".join(
        f"{nm} worst/p95 tilt°" for nm in names) + " | " + " | ".join(
        f"{nm} settle s" for nm in names) + " | " + " | ".join(
        f"{nm} trip%" for nm in names) + " | " + " | ".join(
        f"{nm} HF rate/thr" for nm in names) + " |"
    lines.append(hdr)
    lines.append("|" + "---|" * (1 + 6 * len(names)))
    for g in main_groups:
        row = [g]
        row += [_f(100 * results[nm][g]["success"], ".1f") for nm in names]
        row += [_f(results[nm][g]["err_med"], ".3f") for nm in names]
        row += [f"{_f(results[nm][g]['worst_tilt'], '.0f')}/{_f(results[nm][g]['p95_tilt'], '.0f')}" for nm in names]
        row += [_f(results[nm][g]["settle_mean"], ".2f") for nm in names]
        row += [_f(100 * results[nm][g].get("trip", float("nan")), ".1f") for nm in names]
        row += [f"{_f(results[nm][g]['hf_rates'], '.4f')}/{_f(results[nm][g]['hf_thrust'], '.4f')}" for nm in names]
        lines.append("| " + " | ".join(row) + " |")
    if ext_groups:
        lines.append("")
        lines.append("Randomization extremes (one parameter at min/max, rest nominal, no other noise): success %")
        lines.append("")
        lines.append("| param=extreme | " + " | ".join(f"{nm} {sc}" for sc in EXTREME_SCENARIOS for nm in names) + " |")
        lines.append("|" + "---|" * (1 + len(names) * len(EXTREME_SCENARIOS)))
        keys = []
        for g in ext_groups:
            k = g.split("/")[0]
            if k not in keys:
                keys.append(k)
        for k in keys:
            row = [k] + [_f(100 * results[nm][f"{k}/{sc}"]["success"], ".0f") for sc in EXTREME_SCENARIOS for nm in names]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        for nm in names:
            worst = min(ext_groups, key=lambda g: results[nm][g]["success"])
            lines.append(f"- {nm}: worst extreme `{worst}` -> {100*results[nm][worst]['success']:.0f}%")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", nargs="?", help="*.nfpol.json (omit to evaluate PID only)")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n-ext", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no-extremes", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sensing", choices=sorted(SENSING), default="v1",
                    help="v1: training env of hover_ctbr_v1 (raw gyro, truth+noise pos/vel); "
                         "fw: firmware sensing path (filtered gyro, flow/ToF KF)")
    ap.add_argument("--compare", nargs="*", default=[], metavar="NAME=POLICY",
                    help="extra policies to evaluate side by side, e.g. v1=train/policies/hover_ctbr_v1.nfpol.json")
    ap.add_argument("--md", default=None, help="write the table to this markdown file")
    ap.add_argument("--json", default=None, help="write raw metrics as JSON")
    a = ap.parse_args(argv)
    ctrls = {}
    if a.policy:
        pol = TorchPolicy(a.policy)
        ctrls["RL"] = lambda n, nom: pol
    for spec in a.compare:
        nm, path = spec.split("=", 1)
        ctrls[nm] = (lambda pp: (lambda n, nom: pp))(TorchPolicy(path))
    ctrls["PID"] = make_pid
    res = evaluate(ctrls, a.n, a.n_ext, a.seed, not a.no_extremes, threads=a.threads, sensing=a.sensing)
    table = format_tables(res)
    print(table)
    if a.md:
        with open(a.md, "w") as f:
            f.write(table + "\n")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
