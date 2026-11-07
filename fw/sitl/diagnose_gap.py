"""Sim-to-real gap attribution for a trained policy in SITL.

    python3 fw/sitl/diagnose_gap.py [policy.nfpol.json] [--n 20] [--seed 11]

Flies the policy (real firmware, randomized vehicles, POLICY hover at 1 m)
with selected policy / rate-PID inputs replaced by what the training env
feeds (nf::InputOverride, diagnostics only):

  Q      policy attitude      := true attitude            (gap a: Mahony)
  W      policy body rates    := raw IMU-A gyro            (gap a/b: filter + bias)
  P      policy position      := true pos + white noise    (gap a: KF / flow dead reckoning)
  V      policy velocity      := true vel + white noise    (gap a: KF / flow)
  PID    rate-PID measurement := raw IMU-A gyro            (gap b, FOLLOWUPS 2)

"ALL" is the training-env input path inside the real firmware; removing one
override at a time from ALL (and adding one at a time to NONE) attributes the
SITL hover error and supervisor trips to each gap. Error = 3D RMS of the TRUE
position over the 10 s window (the metric of the SITL hover tests).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import scenarios as S  # noqa: E402
import sitl  # noqa: E402
from sitl import (IN_GYRO, OVR_PID_OMEGA, OVR_POLICY_OMEGA, OVR_POLICY_POS, OVR_POLICY_Q,  # noqa: E402
                  OVR_POLICY_VEL, POLICY, SitlRunner, fault_names, standard_script)

BITS = {"Q": OVR_POLICY_Q, "W": OVR_POLICY_OMEGA, "P": OVR_POLICY_POS, "V": OVR_POLICY_VEL, "PID": OVR_PID_OMEGA}
ALL = sum(BITS.values())


def override_hook(mask: int, seed: int = 0):
    rng = np.random.default_rng(seed + 999)

    def hook(r: SitlRunner, t: float):
        q = r.quad
        n = r.n
        p = q.params
        vals = np.zeros((n, 16), np.float32)
        vals[:, 0:4] = q.quat.numpy()
        g = r.fw.in_f[:, IN_GYRO:IN_GYRO + 3]
        vals[:, 4:7] = g
        vals[:, 7:10] = g
        vals[:, 10:13] = q.pos.numpy() + rng.standard_normal((n, 3)) * p.pos_noise_std.numpy()[:, None]
        vals[:, 13:16] = q.vel.numpy() + rng.standard_normal((n, 3)) * p.vel_noise_std.numpy()[:, None]
        r.fw.set_overrides(np.full(n, mask, np.uint32), vals)
    return hook


def run(policy: dict, mask: int, n: int, seed: int, hold_s: float = 10.0, int8: bool = False):
    r = SitlRunner(n, randomize=True, seed=seed, policy=policy, log_decimate=5, int8=int8)
    assert r.policy_ok
    if mask:
        r.hooks.append(override_hook(mask, seed))
    r.power_on(S.POWER_ON_S)
    t_end = S.T_POLICY + 1.0 + hold_s
    log = r.run(t_end, standard_script(arm_t=S.T_ARM, policy_t=S.T_POLICY))
    res = S._hover_metrics("diag", r, log, t0=S.T_POLICY + 1.0, t1=t_end, mode=POLICY)
    k = (log.t >= S.T_POLICY + 1.0)
    p = log["pos"][k]
    bias = np.linalg.norm(p.mean(0) - S.TARGET, axis=-1)            # steady offset per env
    est_err = log["est_pos"][k] - p                                  # estimator position error
    tilt = sitl.tilt_of_quat(log["quat"][k])
    return dict(rms=res.rms, trips=(res.fault_history != 0), hist=res.fault_history, offset=bias,
                est_pos_err=np.sqrt((est_err ** 2).sum(-1).mean(0)), max_tilt=np.degrees(tilt.max(0)))


def fmt(name, d):
    r = d["rms"] * 100
    return (f"| {name} | {np.min(r):.1f} / {np.median(r):.1f} / {np.max(r):.1f} | {np.median(d['offset'])*100:.1f} | "
            f"{np.median(d['est_pos_err'])*100:.1f} | {np.max(d['max_tilt']):.0f} | {int(d['trips'].sum())} |")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", nargs="?", default=os.path.join(ROOT, "train", "policies", "hover_ctbr_v1.nfpol.json"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--int8", action="store_true")
    a = ap.parse_args(argv)
    pol = sitl.load_nfpol(a.policy)
    torch.set_num_threads(1)
    configs = [("none (firmware as is)", 0), ("ALL (training inputs)", ALL)]
    for k, b in BITS.items():
        configs.append((f"ALL - {k}", ALL & ~b))
    for k, b in BITS.items():
        configs.append((f"none + {k}", b))
    print(f"policy {os.path.relpath(a.policy, ROOT)}, {a.n} randomized vehicles (seed {a.seed})\n")
    print("| inputs | RMS err min/med/max (cm) | med offset (cm) | med est-pos err (cm) | max tilt (deg) | trips |")
    print("|---|---|---|---|---|---|")
    for name, m in configs:
        d = run(pol, m, a.n, a.seed, int8=a.int8)
        print(fmt(name, d), flush=True)
        tr = np.nonzero(d["trips"])[0]
        for e in tr:
            print(f"    env {e}: {fault_names(int(d['hist'][e]))}")


if __name__ == "__main__":
    main()
