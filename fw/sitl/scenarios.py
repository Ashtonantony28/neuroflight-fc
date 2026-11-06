"""SITL scenario suite (used by test/sitl and fw/sitl/report.py).

Every scenario flies the real firmware core (libnf_sitl) in the Python sim.
Common flight profile: power on (vehicle still on the ground, firmware running,
`POWER_ON_S`), arm into FALLBACK_HOLD at t=0.5 s (takes off to the 1 m hold
point), switch to POLICY with the RC re-engage pulse at `T_POLICY`, inject the
fault at `T_FAULT`, keep flying to `T_END`.

Fault scenarios are batched: env i of one runner gets fault i, so the whole
fault table costs one closed-loop run.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from sitl import (calibrated_nominal, CONTROLLED_DESCENT, FALLBACK_HOLD, FAULT, II_KILL, II_LINK_OK, IN_GYRO_B, MODE_NAMES, MOTOR_STOP,
                  POLICY, ROOT, SitlRunner, fault_names, linear_hover_policy, load_nfpol, oscillating_policy,
                  random_policy, standard_script, tilt_of_quat)

TRAINED_POLICY = os.path.join(ROOT, "train", "policies", "hover_ctbr_v1.nfpol.json")
# Mahony gyro-bias convergence time constant is kp/ki = 20 s (estimator.hpp
# defaults); a shorter stationary power-on leaves gyro bias uncorrected, which
# the flow velocity inherits as h * delta_bias (see report).
POWER_ON_S = 120.0
T_ARM = 0.5
T_POLICY = 6.0
T_FAULT = 9.0
T_END = 15.0
HOVER_T0, HOVER_T1 = 6.0, 16.0   # 10 s hover window (arm at 0.5 s)
HOVER_TOL_M = 0.05               # RMS 3D position error over the window
TARGET = np.array([0.0, 0.0, 1.0])


# =========================================================================== hover
@dataclass
class HoverResult:
    name: str
    rms: np.ndarray        # per env, 3D RMS error over the window (m)
    max_err: np.ndarray
    z_rms: np.ndarray
    xy_drift: np.ndarray   # |xy| at end of window
    modes: list
    crashed: np.ndarray
    fault_history: np.ndarray
    arm_attempts: np.ndarray
    log: object = None

    @property
    def passed(self) -> np.ndarray:
        return (self.rms < HOVER_TOL_M) & ~self.crashed & (self.fault_history == 0)


def _hover_metrics(name, r: SitlRunner, log, t0=HOVER_T0, t1=HOVER_T1, mode=FALLBACK_HOLD) -> HoverResult:
    k = (log.t >= t0) & (log.t < t1)
    p = log["pos"][k]
    err = np.linalg.norm(p - TARGET, axis=-1)
    in_mode = (log["mode"][k] == mode).all(0)
    return HoverResult(name, np.sqrt((err ** 2).mean(0)), err.max(0), np.sqrt(((p[..., 2] - 1) ** 2).mean(0)),
                       np.linalg.norm(p[-1, :, :2], axis=-1),
                       [MODE_NAMES[m] if ok else "LEFT_" + MODE_NAMES[mode] for m, ok in zip(log["mode"][-1], in_mode)],
                       r.crashed.copy() | ~in_mode, log["fault_history"][-1].astype(np.int64), r.arm_attempts.copy(), log)


def fallback_hover(n: int = 1, randomize: bool = False, seed: int = 0, power_on_s: float = POWER_ON_S,
                   keep_log: bool = False) -> HoverResult:
    """(a) firmware fallback: take off from the ground, hover at 1 m for 10 s."""
    r = SitlRunner(n, randomize=randomize, seed=seed, log_decimate=5)
    r.power_on(power_on_s)
    log = r.run(HOVER_T1, standard_script(arm_t=T_ARM))
    name = f"fallback hover, {'randomized x%d' % n if randomize else 'nominal'}, power-on {power_on_s:g} s"
    res = _hover_metrics(name, r, log)
    if not keep_log:
        res.log = None
    return res


def _calibrate(r: SitlRunner):
    """Per-env synthetic linear policy using the vehicle's true mass/thrust
    (it has no integrator, so it would otherwise hover off-target)."""
    for e in range(r.n):
        ok, _ = r.fw.set_policy(linear_hover_policy(calibrated_nominal(r, e)), env=e)
        assert ok
    r.calibrated = True


def policy_hover(policy: dict, n: int = 1, randomize: bool = False, seed: int = 0, name: str = "policy",
                 power_on_s: float = POWER_ON_S, hold_s: float = 10.0) -> HoverResult:
    """(b) take off in FALLBACK_HOLD, switch to POLICY at T_POLICY, hover `hold_s`."""
    r = SitlRunner(n, randomize=randomize, seed=seed, policy=policy, log_decimate=5)
    assert r.policy_ok, "firmware rejected the policy"
    if randomize and (policy.get("meta") or {}).get("name") == "linear_hover":
        _calibrate(r)
    r.power_on(power_on_s)
    t_end = T_POLICY + 1.0 + hold_s
    log = r.run(t_end, standard_script(arm_t=T_ARM, policy_t=T_POLICY))
    res = _hover_metrics(f"{name} hover, {'randomized x%d' % n if randomize else 'nominal'}", r, log,
                         t0=T_POLICY + 1.0, t1=t_end, mode=POLICY)
    return res


# =========================================================================== faults
@dataclass
class Fault:
    name: str
    description: str
    expect_fault: str          # supervisor fault bit expected to trip first
    expect_mode: int           # mode entered from POLICY
    apply: callable            # apply(runner, env) once at T_FAULT
    each_step: callable = None  # each_step(runner, env) for t >= T_FAULT (sensor/RC corruption)
    lands: bool = False        # expected to end on the ground (MOTOR_STOP)


def _last_layer(pol):
    return len(pol["layers"]) - 1


def make_faults(base: dict) -> list[Fault]:
    L = _last_layer(base)

    def osc(r, e):
        nom = calibrated_nominal(r, e) if getattr(r, "calibrated", False) else None
        r.fw.set_policy(oscillating_policy(nom), env=e)

    def poke(idx, val):
        return lambda r, e: r.fw.poke_policy(L, "b", idx, val, env=e)

    def gyro_b(r, e):
        r.fw.in_f[e, IN_GYRO_B] += 1.0

    def rc_loss(r, e):
        r.fw.in_i[e, II_LINK_OK] = 0

    def kill(r, e):
        r.fw.in_i[e, II_KILL] = 1

    return [
        Fault("nan", "last-layer bias of a1 corrupted to NaN (CRC not re-checked in flight)",
              "POLICY_NONFINITE", FALLBACK_HOLD, poke(1, float("nan"))),
        Fault("saturate", "a3 (yaw rate) pinned at +1 (bias +6)", "POLICY_SATURATION", FALLBACK_HOLD, poke(3, 6.0)),
        Fault("oscillate", "hot-swap to a policy with a 250 Hz limit cycle on a1 (amplitude ~0.9)",
              "POLICY_OSCILLATION", FALLBACK_HOLD, osc),
        Fault("flip", "a1 (roll rate) pinned near +1: 10 rad/s roll", "TILT", FALLBACK_HOLD, poke(1, 3.0)),
        Fault("imu_b", "IMU B gyro x offset +1.0 rad/s", "IMU_DISAGREE", FALLBACK_HOLD, lambda r, e: None, gyro_b),
        Fault("rc_loss", "RC link lost (no valid frames)", "RC_LOSS", CONTROLLED_DESCENT, lambda r, e: None, rc_loss,
              lands=True),
        Fault("overrun", "simulated inference time 600 us (deadline 400 us)", "DEADLINE", FALLBACK_HOLD,
              lambda r, e: r.fw.set_inference_us(600, env=e)),
        Fault("kill", "RC kill switch", "KILL", MOTOR_STOP, lambda r, e: None, kill),
    ]


@dataclass
class FaultResult:
    fault: str
    env: int
    first_fault: str
    history: list
    mode_after: str
    final_mode: str
    latency_ms: float          # fault onset -> supervisor leaves POLICY
    alt_lost_m: float          # z(onset) - min z over the next 3 s (true state)
    crashed: bool
    crash_reason: str
    kill_cmd_zero_loops: int   # kill: loops from kill input until all motor cmds == 0 (-1 = n/a)
    max_tilt_deg: float        # max true tilt in the 3 s after onset
    end_tilt_deg: float
    end_z: float
    touchdown_vz: float
    passed: bool
    why: str = ""


def run_fault_suite(base: dict | None = None, randomize: bool = False, seed: int = 0,
                    power_on_s: float = POWER_ON_S, only: list[str] | None = None):
    """Run all faults (one env each) in one batched closed-loop flight."""
    synthetic = base is None
    base = base or linear_hover_policy()
    faults = [f for f in make_faults(base) if only is None or f.name in only]
    n = len(faults)
    r = SitlRunner(n, randomize=randomize, seed=seed, policy=base, log_decimate=1)
    assert r.policy_ok
    if synthetic and randomize:
        _calibrate(r)
    r.power_on(power_on_s)
    applied = [False] * n

    def hook(run: SitlRunner, t: float):
        if t + 1e-9 < T_FAULT:
            return
        for e, f in enumerate(faults):
            if not applied[e]:
                f.apply(run, e)
                applied[e] = True
            if f.each_step:
                f.each_step(run, e)

    r.hooks.append(hook)
    script = standard_script(arm_t=T_ARM, policy_t=T_POLICY)
    log = r.run(T_END, script)
    return [_analyze(f, e, r, log) for e, f in enumerate(faults)], log


def _analyze(f: Fault, e: int, r: SitlRunner, log) -> FaultResult:
    t = log.t
    mode = log["mode"][:, e]
    k0 = int(np.searchsorted(t, T_FAULT - 1e-9))
    z = log["pos"][:, e, 2]
    why = []
    in_policy_at_fault = mode[k0 - 1] == POLICY
    if not in_policy_at_fault:
        why.append(f"not in POLICY before the fault ({MODE_NAMES[mode[k0 - 1]]})")
    left = np.nonzero(mode[k0:] != POLICY)[0]
    ks = k0 + int(left[0]) if left.size else None
    latency = (t[ks] - T_FAULT) * 1e3 if ks is not None else float("nan")
    mode_after = MODE_NAMES[mode[ks]] if ks is not None else "POLICY"
    w = (t >= T_FAULT) & (t < T_FAULT + 3.0)
    alt_lost = max(0.0, float(z[k0] - z[w].min()))
    max_tilt = math.degrees(float(tilt_of_quat(log["quat"][w, e]).max()))
    first = int(log["first_fault"][-1, e])
    hist = int(log["fault_history"][-1, e])
    tilt_end = math.degrees(float(tilt_of_quat(log["quat"][-1, e])))
    td = [d for d in r.touchdowns[e] if d[0] > T_ARM + 1.0]
    td_vz = td[-1][1] if td else float("nan")
    kill_loops = -1
    if f.name == "kill":
        cmd = log["cmd"][k0:, e]
        zero = np.nonzero((cmd == 0).all(-1))[0]
        kill_loops = int(zero[0]) if zero.size else 10 ** 6
        if kill_loops > 1:
            why.append(f"motors not stopped within one loop ({kill_loops})")
        if mode_after != "MOTOR_STOP":
            why.append("kill did not force MOTOR_STOP")
    else:
        if fault_names(first) != [f.expect_fault]:
            why.append(f"first fault {fault_names(first)} != {f.expect_fault}")
        if mode_after != MODE_NAMES[f.expect_mode]:
            why.append(f"entered {mode_after}, expected {MODE_NAMES[f.expect_mode]}")
        if r.crashed[e]:
            why.append("crashed: " + r.crash_reason[e])
        if f.lands:
            if MODE_NAMES[mode[-1]] != "MOTOR_STOP" or z[-1] > 0.05:
                why.append(f"did not land and stop (final {MODE_NAMES[mode[-1]]}, z={z[-1]:.2f})")
        else:
            if mode[-1] not in (FALLBACK_HOLD,):
                why.append(f"final mode {MODE_NAMES[mode[-1]]}")
            if tilt_end > 10 or z[-1] < 0.3 or np.linalg.norm(log["vel"][-1, e]) > 0.3:
                why.append(f"not recovered to a stable hover (z={z[-1]:.2f}, tilt={tilt_end:.0f} deg)")
    return FaultResult(f.name, e, "+".join(fault_names(first)) or "-", fault_names(hist), mode_after,
                       MODE_NAMES[mode[-1]], latency, alt_lost, bool(r.crashed[e]), r.crash_reason[e], kill_loops, max_tilt,
                       tilt_end, float(z[-1]), td_vz, not why, "; ".join(why))


# =========================================================================== bad policy
def bad_policy_flight(seeds=(0, 1, 2), randomize=False):
    """Deliberately bad (random-weight) policies engaged in hover: the
    supervisor must hand over to the fallback and the vehicle must not crash."""
    pols = [random_policy(s) for s in seeds]
    n = len(pols)
    r = SitlRunner(n, randomize=randomize, seed=7, policy=pols[0], log_decimate=1)
    for e, p in enumerate(pols):
        ok, _ = r.fw.set_policy(p, env=e)
        assert ok
    r.power_on(POWER_ON_S)
    t_end = T_POLICY + 6.0
    log = r.run(t_end, standard_script(arm_t=T_ARM, policy_t=T_POLICY))
    out = []
    t = log.t
    for e in range(n):
        mode = log["mode"][:, e]
        k0 = int(np.searchsorted(t, T_POLICY))
        eng = np.nonzero(mode[k0:] == POLICY)[0]
        if eng.size == 0:
            out.append(dict(seed=seeds[e], engaged=False, passed=False, why="never engaged"))
            continue
        ke = k0 + int(eng[0])
        left = np.nonzero(mode[ke:] != POLICY)[0]
        ks = ke + int(left[0]) if left.size else None
        z = log["pos"][:, e, 2]
        w = (t >= t[ke]) & (t < t[ke] + 3)
        res = dict(seed=seeds[e], engaged=True, latency_ms=(t[ks] - t[ke]) * 1e3 if ks else float("nan"),
                   first_fault="+".join(fault_names(int(log["first_fault"][-1, e]))),
                   mode_after=MODE_NAMES[mode[ks]] if ks else "POLICY", final_mode=MODE_NAMES[mode[-1]],
                   alt_lost_m=max(0.0, float(z[ke] - z[w].min())), crashed=bool(r.crashed[e]),
                   crash_reason=r.crash_reason[e], end_z=float(z[-1]))
        res["passed"] = ks is not None and not r.crashed[e] and res["final_mode"] in ("FALLBACK_HOLD",
                                                                                         "CONTROLLED_DESCENT",
                                                                                         "MOTOR_STOP")
        out.append(res)
    return out


def trained_policy() -> dict | None:
    return load_nfpol(TRAINED_POLICY) if os.path.exists(TRAINED_POLICY) else None


# =========================================================================== CTBR parity (INTERFACES.md section 9)
CTBR_TOL = 1e-5   # max |cmd_sim - cmd_fw| (commands in [0,1]) for identical inputs


def _sim_ctbr(n, nominal):
    import torch
    from sim.ctbr import RateController
    return RateController(n, nominal), torch


def ctbr_parity_random(n: int = 64, steps: int = 4000, seed: int = 0, sim_reset_with_gyro: bool = True):
    """Identical (action, gyro) streams into sim/ctbr.py (RateController:
    action_to_ctbr -> rate PID -> mixer) and the firmware (decode_ctbr ->
    nf::RatePid -> nf::Mixer). Actions are held for 2 rate-loop steps (500 Hz
    policy); half the envs fly near hover, half get full-range random actions
    (heavy mixer saturation). Gyro: smooth random walk + noise."""
    from sim.ctbr import action_to_ctbr
    from sim.params import load_nominal
    from sitl import FirmwareCtbr
    nom = load_nominal()
    rc, torch = _sim_ctbr(n, nom)
    fw = FirmwareCtbr(n)
    rng = np.random.default_rng(seed)
    hover_a0 = nom["mass"] * nom["gravity"] / (2 * nom["thrust_max"]) - 1
    wide = np.arange(n) >= n // 2
    gyro = rng.normal(0, 1.0, (n, 3)).astype(np.float32)
    g_first = (gyro + rng.normal(0, 0.02, (n, 3))).astype(np.float32)
    if sim_reset_with_gyro:
        rc.reset(torch.arange(n), torch.from_numpy(g_first.copy()))  # == firmware first update
    else:
        rc.reset(torch.arange(n))     # what sim/env.py::_reset_idx does (prev meas = 0)
    # capture the sim mixer's saturation flag
    sim_mix = rc.mixer.mix
    cap = {}

    def mix_cap(F, tau):
        cmd, sat = sim_mix(F, tau)
        cap["sat"] = sat
        return cmd, sat
    rc.mixer.mix = mix_cap
    d = np.zeros((steps, n, 4), np.float32)
    sat_s = np.zeros((steps, n), bool)
    sat_f = np.zeros((steps, n), bool)
    a = np.zeros((n, 4), np.float32)
    for k in range(steps):
        if k % 2 == 0:
            near = np.concatenate([hover_a0 + rng.normal(0, 0.1, (n, 1)), rng.normal(0, 0.15, (n, 3))], 1)
            full = rng.uniform(-1, 1, (n, 4))
            a = np.where(wide[:, None], full, near).clip(-1, 1).astype(np.float32)
        if k > 0:
            gyro = (0.995 * gyro + rng.normal(0, 0.15, (n, 3))).astype(np.float32)
            g_noisy = (gyro + rng.normal(0, 0.02, (n, 3))).astype(np.float32)
        else:
            g_noisy = g_first
        F, rsp = action_to_ctbr(torch.from_numpy(a), nom)
        cmd_s = rc(F, rsp, torch.from_numpy(g_noisy)).numpy()
        cmd_f, sf = fw.step(a, g_noisy)
        sat_s[k] = cap["sat"].numpy()
        sat_f[k] = sf
        d[k] = np.abs(cmd_s - cmd_f)
        sat_f[k] = sf
    return {
        "max_abs": float(d.max()), "mean_abs": float(d.mean()), "p99_abs": float(np.quantile(d, 0.99)),
        "max_abs_near_hover": float(d[:, ~wide].max()), "max_abs_saturating": float(d[:, wide].max()),
        "first_step_max_abs": float(d[0].max()), "after_first_max_abs": float(d[1:].max()),
        "frac_gt_tol": float((d > CTBR_TOL).mean()), "fw_saturated_frac": float(sat_f.mean()),
        "sat_flag_mismatch_frac": float((sat_s != sat_f).mean()),
        "first_sat_mismatch_step": int(np.argmax((sat_s != sat_f).any(1))) if (sat_s != sat_f).any() else -1,
        "first_diff_step": int(np.argmax((d > CTBR_TOL).any((1, 2)))) if (d > CTBR_TOL).any() else -1,
        "n": n, "steps": steps,
    }


def ctbr_parity_closed_loop(seconds: float = 4.0, n: int = 16, seed: int = 0):
    """Realistic input distribution: sim QuadEnv-style closed loop (randomized
    vehicles, tilted/rotating initial states) flown by the sim RateController
    and the sim cascaded PID baseline; the firmware CTBR replays the exact same
    (action, gyro) stream. Open-loop replay == closed loop (identical inputs)."""
    import torch
    from sim.ctbr import RateController, action_to_ctbr
    from sim.env import EnvConfig, QuadEnv
    from sim.pid_baseline import CascadedPID
    from sitl import FirmwareCtbr
    env = QuadEnv(EnvConfig(num_envs=n, seed=seed, randomize_initial_time=False, episode_len_s=100.0))
    obs = env.reset()
    pid = CascadedPID(n, env.nom)
    # Re-seed the rate controller exactly like the firmware does on its first
    # update (prev measurement = first gyro sample).
    env.rate_ctrl.reset(torch.arange(n), env.last_gyro.clone())
    fw = FirmwareCtbr(n)
    q = env.quad
    dmax, dsum, cnt = 0.0, 0.0, 0
    steps = int(seconds * env.nom["control_rate_hz"])
    for _ in range(steps):
        a = pid(obs).clamp(-1, 1)
        F, rsp = action_to_ctbr(a, env.nom)
        gyro = env.last_gyro
        for k in range(env.substeps):
            if k > 0:
                gyro = q.sample_gyro()
            c_s = env.rate_ctrl(F, rsp, gyro)
            c_f, _ = fw.step(a.numpy(), gyro.numpy())
            dd = np.abs(c_s.numpy() - c_f)
            dmax = max(dmax, float(dd.max()))
            dsum += float(dd.sum())
            cnt += dd.size
            q.step(c_s)
        env.last_gyro = q.sample_gyro()
        env.prev_action = a
        env._compute_obs()
        obs = env.obs_buf.clone()
    return {"max_abs": dmax, "mean_abs": dsum / cnt, "n": n, "steps": steps * env.substeps}


def ctbr_full_loop_gap(policy: dict | None = None):
    """Informational: the firmware ControlLoop feeds the rate PID with
    filtered gyro (150 Hz biquad + RPM notches) minus the Mahony bias estimate,
    whereas sim/ctbr.py (and training) use the raw gyro. Replays the actions
    and raw IMU-A gyro of a SITL policy flight (randomized vehicle) through
    sim/ctbr.py and compares with the firmware motor commands."""
    import torch
    from sim.ctbr import action_to_ctbr
    from sim.params import load_nominal
    nom = load_nominal()
    pol = policy or linear_hover_policy()
    r = SitlRunner(1, randomize=True, seed=3, policy=pol)
    r.power_on(POWER_ON_S)
    log = r.run(T_POLICY + 4.0, standard_script(arm_t=T_ARM, policy_t=T_POLICY))
    mode = log["mode"][:, 0]
    ke = int(np.nonzero(mode == POLICY)[0][0])
    rc, _ = _sim_ctbr(1, nom)
    rc.reset(torch.arange(1), torch.from_numpy(log["gyro"][ke, 0][None].copy()))
    d = []
    for k in range(ke, len(log.t)):
        if mode[k] != POLICY:
            break
        a = torch.from_numpy(log["action"][k, 0][None].copy())
        F, rsp = action_to_ctbr(a, nom)
        c_s = rc(F, rsp, torch.from_numpy(log["gyro"][k, 0][None].copy())).numpy()[0]
        d.append(np.abs(c_s - log["cmd"][k, 0]))
    d = np.asarray(d)
    skip = 200   # ignore the integrator handover transient (fw PID kept its fallback state)
    return {"max_abs": float(d[skip:].max()), "rms": float(np.sqrt((d[skip:] ** 2).mean())),
            "gyro_noise_std": float(r.quad.params.gyro_noise_std[0]), "steps": int(len(d) - skip)}
