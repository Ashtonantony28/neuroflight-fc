"""Run the SITL scenario suite and write docs/sitl-report.md.

    python fw/sitl/report.py [--quick]

Blackbox logs (npz + CSV of env 0) of the fault runs go to runs/sitl/ (gitignored).
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import scenarios as S  # noqa: E402
import sitl  # noqa: E402

OUT = os.path.join(sitl.ROOT, "docs", "sitl-report.md")
RUNS = os.path.join(sitl.ROOT, "runs", "sitl")


def cm(x):
    return f"{x * 100:.1f}"


def hover_row(h: S.HoverResult):
    n = len(h.rms)
    p = int(h.passed.sum())
    if n == 1:
        rms, mx = cm(h.rms[0]), cm(h.max_err[0])
    else:
        rms = f"{cm(h.rms.min())} / {cm(np.median(h.rms))} / {cm(h.rms.max())}"
        mx = cm(h.max_err.max())
    faults = sorted({f for b in h.fault_history for f in sitl.fault_names(int(b))}) or ["-"]
    return f"| {h.name} | {rms} | {mx} | {cm(h.z_rms.max())} | {cm(h.xy_drift.max())} | {','.join(faults)} | {p}/{n} |"


def fault_rows(res, label):
    rows = []
    for r in res:
        lat = "same loop" if r.latency_ms == 0 else f"{r.latency_ms:.0f}"
        if r.fault == "kill":
            outcome = f"motors 0 after {r.kill_cmd_zero_loops} loops; falls (expected)"
        elif r.fault == "rc_loss":
            outcome = f"landed at {abs(r.touchdown_vz):.2f} m/s, {r.final_mode}"
        else:
            outcome = f"{r.final_mode}, z={r.end_z:.2f} m"
        rows.append(f"| {label} | {r.fault} | {r.first_fault} | {r.mode_after} | {lat} | {r.alt_lost_m:.3f} | "
                    f"{r.max_tilt_deg:.0f} | {outcome} | {'PASS' if r.passed else 'FAIL: ' + r.why} |")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer randomized vehicles")
    args = ap.parse_args()
    nr = 5 if args.quick else 20
    t0 = time.time()
    os.makedirs(RUNS, exist_ok=True)
    L = []
    w = L.append
    trained = S.trained_policy()

    print("hover ...")
    hov = [S.fallback_hover(1), S.fallback_hover(nr, randomize=True, seed=1),
           S.fallback_hover(nr, randomize=True, seed=1, power_on_s=1.5)]
    print("policy hover ...")
    pol_h = [S.policy_hover(sitl.linear_hover_policy(), name="synthetic linear policy"),
             S.policy_hover(sitl.linear_hover_policy(), n=4, randomize=True, seed=11, name="synthetic linear policy")]
    if trained is not None:
        pol_h += [S.policy_hover(trained, name="trained hover_ctbr_v1"),
                  S.policy_hover(trained, n=4, randomize=True, seed=11, name="trained hover_ctbr_v1")]
    print("bad policies ...")
    bad = S.bad_policy_flight()
    print("faults ...")
    f_nom, log_nom = S.run_fault_suite()
    log_nom.save_npz(os.path.join(RUNS, "faults_nominal.npz"))
    f_rnd, _ = S.run_fault_suite(randomize=True, seed=5)
    f_tr = None
    if trained is not None:
        f_tr, _ = S.run_fault_suite(base=trained, only=["nan", "saturate", "flip", "overrun", "imu_b"])
    print("ctbr parity ...")
    par = S.ctbr_parity_random()
    par_cl = S.ctbr_parity_closed_loop()
    par_env = S.ctbr_parity_random(sim_reset_with_gyro=False)
    gap = S.ctbr_full_loop_gap()

    w("# SITL report (WS6)\n")
    w(f"Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by `python fw/sitl/report.py"
      f"{' --quick' if args.quick else ''}` in {time.time() - t0:.0f} s. The real firmware core "
      "(`fw/core`, via `fw/sitl/libnf_sitl.so`) runs at 1 kHz (policy 500 Hz) inside `sim.Quadrotor` "
      "(sim latency + motor lag per vehicle). The sensors are IMU A and IMU B at 1 kHz, and ToF and optical flow at 100 Hz. "
      f"Each run starts with the vehicle powered on and still on the ground for {S.POWER_ON_S:.0f} s. It is then armed into FALLBACK_HOLD and takes off to 1 m. "
      f"Policy runs switch to POLICY with the RC re-engage at t={S.T_POLICY:.0f} s, and faults are injected at t={S.T_FAULT:.0f} s.\n")
    w(f"Trained policy `{os.path.relpath(S.TRAINED_POLICY, sitl.ROOT)}`: "
      f"{'present' if trained is not None else '**not present** (trained-policy rows skipped)'}.\n")

    w("## (a) Fallback controller: takeoff and 10 s hover at 1 m\n")
    w(f"Pass = 3D RMS error < {S.HOVER_TOL_M * 100:.0f} cm over the 10 s window, no faults, no crash. "
      "For batches the RMS column is min / median / max.\n")
    w("| scenario | RMS err (cm) | max err (cm) | worst z RMS (cm) | worst xy drift (cm) | faults | pass |")
    w("|---|---|---|---|---|---|---|")
    for h in hov:
        w(hover_row(h))
    w("\nThe last row is informational. It powers on for only 1.5 s, so the Mahony gyro bias has not converged "
      "(time constant kp/ki = 20 s). The flow velocity then inherits h * delta_bias and the vehicle drifts horizontally. See the findings.\n")

    w("## (b) Policy mode\n")
    w("| scenario | RMS err (cm) | max err (cm) | worst z RMS (cm) | worst xy drift (cm) | faults | pass |")
    w("|---|---|---|---|---|---|---|")
    for h in pol_h:
        w(hover_row(h))
    w("\nThe synthetic linear policy is a hand-built single tanh layer: PD on the §4 observation. It stands in for a "
      "trained policy and carries the fault injection. On randomized vehicles it is calibrated with the true mass "
      "and thrust, because it has no integrator.\n")
    w("Deliberately bad policies (random weights, 19-64-64-4, std 1), engaged at hover:\n")
    w("| seed | first fault | entered | latency (ms) | alt lost (m) | final | pass |")
    w("|---|---|---|---|---|---|---|")
    for b in bad:
        w(f"| {b['seed']} | {b.get('first_fault', '-')} | {b.get('mode_after', '-')} | {b.get('latency_ms', float('nan')):.0f} | "
          f"{b.get('alt_lost_m', float('nan')):.3f} | {b.get('final_mode', '-')} | {'PASS' if b['passed'] else 'FAIL'} |")

    w("\n## (c)+(d) Fault injection (supervisor transitions)\n")
    w("Latency is measured from fault onset to the supervisor leaving POLICY, in 1 ms loop steps. Alt lost = true z at onset "
      "minus the minimum true z over the next 3 s. Max tilt is over the same 3 s. "
      "Pass requires the expected first fault, the expected mode, no crash (touchdown > 1 m/s or > 34 deg), and a "
      "stable hover at the end (or a landing and MOTOR_STOP for RC loss).\n")
    w("| vehicle | fault | first fault | entered | latency (ms) | alt lost (m) | max tilt (deg) | outcome | result |")
    w("|---|---|---|---|---|---|---|---|---|")
    for row in fault_rows(f_nom, "nominal") + fault_rows(f_rnd, "randomized"):
        w(row)
    if f_tr:
        for row in fault_rows(f_tr, "nominal, trained base"):
            w(row)
    w("\nHow each fault is injected:")
    for f in S.make_faults(sitl.linear_hover_policy()):
        w(f"- **{f.name}**: {f.description}. Expected: {f.expect_fault} -> {sitl.MODE_NAMES[f.expect_mode]}.")
    w("- Kill in FALLBACK_HOLD (pytest): motor commands are 0 in the same loop iteration and stay at 0.\n")

    w("## (e) CTBR inner loop: sim/ctbr.py vs firmware (INTERFACES.md §9)\n")
    w(f"Tolerance: |cmd_sim - cmd_fw| <= {S.CTBR_TOL:g} (commands in [0,1]) for identical (action, gyro) inputs.\n")
    w("| comparison | max abs | mean abs | other |")
    w("|---|---|---|---|")
    w(f"| random stress, 64 envs x 4000 steps, {par['fw_saturated_frac'] * 100:.0f}% of steps mixer-saturated | "
      f"{par['max_abs']:.2e} | {par['mean_abs']:.1e} | sat-flag mismatch {par['sat_flag_mismatch_frac']:.0%} |")
    w(f"| closed loop (sim env + PID baseline, randomized, tilted/rotating starts), 16 x 4000 | {par_cl['max_abs']:.2e} | "
      f"{par_cl['mean_abs']:.1e} | |")
    w(f"| same random stress, but sim reset as `sim/env.py` does it (no gyro) | {par_env['max_abs']:.3f} | "
      f"{par_env['mean_abs']:.1e} | first step {par_env['first_step_max_abs']:.3f}; {par_env['frac_gt_tol']:.0%} of samples > tol |")
    w(f"| full firmware loop vs sim/ctbr.py on raw gyro (informational) | {gap['max_abs']:.4f} | rms {gap['rms']:.4f} | "
      f"gyro noise std {gap['gyro_noise_std']:.3f} rad/s |")
    w("\nThe last row shows the input-side gap. The firmware rate PID sees the gyro after a 150 Hz biquad, the RPM notches and "
      "Mahony bias correction. Training feeds the raw noisy gyro to the PID.\n")

    w("## Findings\n")
    w("1. **Fixed in fw/core (one line, `ControlLoop::do_arm`)**: arming kept the translation-KF velocity. On the ground "
      "the horizontal velocity cannot be observed (no flow below 8 cm), so it random-walks while disarmed. After 60 s it had reached ~5 m/s, "
      "and FAULT_ESTIMATOR tripped right after takeoff, which sent the vehicle into CONTROLLED_DESCENT. Arming now zeroes the "
      "horizontal velocity.")
    w("2. **Arming IMU check uses single raw samples** (`arming_checks`: |gyro_a - gyro_b| <= 0.1 rad/s). Under the "
      "randomized noise and bias it fails often, and some vehicles needed up to 9 arm flicks. It should compare filtered or "
      "bias-corrected values.")
    w("3. **Gyro bias converges slowly** (Mahony kp/ki = 20 s) and the firmware has no stationary calibration. Flow velocity "
      "error = h * delta_bias, so x/y drifts by tens of cm in 10 s if the vehicle arms soon after power-on (row 3 of (a)).")
    w("4. **sim/env.py resets RateController without the gyro**, so the first step gets a D-term kick (table (e)). "
      "The firmware seeds the previous measurement with the first sample.")
    w("5. **Filter path in §9**: the firmware PID input is filtered and bias-corrected; the sim uses the raw gyro (table (e), last row).")
    w("6. **Flip detection is slow**: the TILT trip takes 60 deg plus 50 ms, the commanded 10 rad/s stays under the 12 rad/s RATE limit, so "
      "the fallback takes over at about 110 deg and the vehicle loses about 0.6 m. That is fine at 1 m and marginal lower down.")
    w("7. Handover sag: the cascade's velocity integrator is reset when the fallback takes over, so a heavy vehicle can sag "
      "for about 1 s (alt lost up to ~0.35 m on randomized NaN/saturate rows).")
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
