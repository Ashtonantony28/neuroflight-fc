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
        sw = "-" if math.isnan(r.switch_tilt_deg) else f"{r.switch_tilt_deg:.0f}"
        rows.append(f"| {label} | {r.fault} | {r.first_fault} | {r.mode_after} | {lat} | {r.alt_lost_m:.3f} | "
                    f"{sw} | {r.max_tilt_deg:.0f} | {outcome} | {'PASS' if r.passed else 'FAIL: ' + r.why} |")
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
    v2 = S.trained_policy(S.TRAINED_POLICY_V2)
    v2q_path = S.TRAINED_POLICY_V2.replace(".nfpol.json", ".q.nfpol.json")
    v2q = sitl.load_nfpol(v2q_path) if os.path.exists(v2q_path) else None

    print("hover ...")
    hov = [S.fallback_hover(1), S.fallback_hover(nr, randomize=True, seed=1),
           S.fallback_hover(nr, randomize=True, seed=1, power_on_s=1.5)]
    print("policy hover ...")
    pol_h = [S.policy_hover(sitl.linear_hover_policy(), name="synthetic linear policy"),
             S.policy_hover(sitl.linear_hover_policy(), n=4, randomize=True, seed=11, name="synthetic linear policy")]
    if trained is not None:
        pol_h += [S.policy_hover(trained, name="trained hover_ctbr_v1"),
                  S.policy_hover(trained, n=nr, randomize=True, seed=11, name="trained hover_ctbr_v1")]
    if v2 is not None:
        pol_h += [S.policy_hover(v2, name="trained hover_ctbr_v2"),
                  S.policy_hover(v2, n=nr, randomize=True, seed=11, name="trained hover_ctbr_v2")]
    if v2q is not None:
        pol_h += [S.policy_hover(v2q, n=nr, randomize=True, seed=11, name="hover_ctbr_v2 int8 (nf::Int8Model)", int8=True)]
    print("policy disturbances ...")
    dist = []
    for label, pol, i8 in (("hover_ctbr_v1", trained, False), ("hover_ctbr_v2", v2, False), ("hover_ctbr_v2 int8", v2q, True)):
        if pol is None:
            continue
        for kind in ("tilt", "throw"):
            dist.append((label, kind, S.policy_disturbance(pol, kind, n=nr, seed=21, int8=i8)))
    if v2 is not None:
        dist.append(("hover_ctbr_v2", "throw, |dw| <= 10 rad/s (stress, informational)",
                     S.policy_disturbance(v2, "throw", n=nr, seed=21, throw_dw_max=10.0)))
    print("gap attribution ...")
    import diagnose_gap as DG
    gap_rows = []
    for label, pol in (("hover_ctbr_v1", trained), ("hover_ctbr_v2", v2)):
        if pol is None:
            continue
        for name, m in (("firmware as is", 0), ("training-env inputs (ALL)", DG.ALL), ("firmware + true velocity (V)", DG.BITS["V"]),
                        ("ALL - V (firmware velocity only)", DG.ALL & ~DG.BITS["V"]), ("firmware + true attitude (Q)", DG.BITS["Q"]),
                        ("firmware + raw gyro to PID (PID)", DG.BITS["PID"]), ("firmware + raw gyro obs (W)", DG.BITS["W"]),
                        ("firmware + true position (P)", DG.BITS["P"])):
            gap_rows.append((label, name, DG.run(pol, m, nr, 11)))
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
    gfp = S.gyro_filter_parity()
    gap_raw = S.ctbr_full_loop_gap(sim_filter=False)
    gap = S.ctbr_full_loop_gap(sim_filter=True)

    w("# SITL report (WS6)\n")
    w(f"Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by `python fw/sitl/report.py"
      f"{' --quick' if args.quick else ''}` in {time.time() - t0:.0f} s. The real firmware core "
      "(`fw/core`, via `fw/sitl/libnf_sitl.so`) runs at 1 kHz (policy 500 Hz) inside `sim.Quadrotor` "
      "(sim latency + motor lag per vehicle). The sensors are IMU A and IMU B at 1 kHz, and ToF and optical flow at 100 Hz. "
      f"Each run starts with the vehicle powered on and still on the ground for {S.POWER_ON_S:.0f} s. It is then armed into FALLBACK_HOLD and takes off to 1 m. "
      f"Policy runs switch to POLICY with the RC re-engage at t={S.T_POLICY:.0f} s, and faults are injected at t={S.T_FAULT:.0f} s.\n")
    for path, pol in ((S.TRAINED_POLICY, trained), (S.TRAINED_POLICY_V2, v2), (v2q_path, v2q)):
        w(f"Trained policy `{os.path.relpath(path, sitl.ROOT)}`: "
          f"{'present' if pol is not None else '**not present** (rows skipped)'}.\n")

    w("## (a) Fallback controller: takeoff and 10 s hover at 1 m\n")
    w(f"Pass = 3D RMS error < {S.HOVER_TOL_M * 100:.0f} cm over the 10 s window, no faults, no crash. "
      "For batches the RMS column is min / median / max.\n")
    w("| scenario | RMS err (cm) | max err (cm) | worst z RMS (cm) | worst xy drift (cm) | faults | pass |")
    w("|---|---|---|---|---|---|---|")
    for h in hov:
        w(hover_row(h))
    sp = hov[2]
    w(f"\nThe last row powers on for only 1.5 s and arms at t = 2.0 s. Before the at-rest gyro calibration it was 0/20 "
      f"(median 24 cm, up to 8 arm flicks): the Mahony bias had not converged (kp/ki = 20 s) and the flow velocity inherited "
      f"h * delta_bias. Now every vehicle arms on the first flick (max attempts {int(sp.arm_attempts.max())}) with the bias "
      f"from about 2 s of still averaging, and {int(sp.passed.sum())}/{len(sp.rms)} hover under 5 cm. The 5 cm bound is close to the "
      "noise floor of flow-only xy dead reckoning for max-spec sensor noise. Over seeds 1-4 (80 vehicles), 73/80 pass at 2 s and "
      "77/80 after a 120 s power-on. The pytest version (`test_arm_2s_after_power_on_and_hover_20_randomized`) allows 2 of 20 "
      "between 5 and 7.5 cm.\n")

    w("## (b) Policy mode\n")
    w("| scenario | RMS err (cm) | max err (cm) | worst z RMS (cm) | worst xy drift (cm) | faults | pass |")
    w("|---|---|---|---|---|---|---|")
    for h in pol_h:
        w(hover_row(h))
    w("\nThe synthetic linear policy is a hand-built single tanh layer: PD on the §4 observation. It stands in for a "
      "trained policy and carries the fault injection. On randomized vehicles it is calibrated with the true mass "
      "and thrust, because it has no integrator.\n")
    w("hover_ctbr_v1 was trained on true attitude/velocity (+ white noise) and the raw gyro; hover_ctbr_v2 on the firmware sensing "
      "path (sim/fw_estimator.py). The int8 row flies the quantized v2 through the firmware's int8 inference "
      "(`nf_sitl_set_policy_i8` -> `nf::Int8Model`).\n")
    if dist:
        w("### Policy recovery from disturbances (randomized vehicles)\n")
        w(f"At t = {S.T_DIST:.0f} s, 3 s after the POLICY engage. **tilt**: an external 'hand' holds the vehicle in place and rotates it "
          f"about a random horizontal axis at {S.TILT_DIST_RATE} rad/s to {S.TILT_DIST_DEG} deg, then releases it at rest (gyro and "
          f"accelerometer see the motion; both envelope monitors stay quiet during the hold: tilt < 60.16 deg, rate < 4 rad/s). "
          f"**throw**: a {S.THROW_PUSH_S * 1000:.0f} ms push, |dv| U[0, {S.THROW_DV_MAX:g}] m/s and |dw| U[0, {S.THROW_DW_MAX:g}] rad/s "
          "in random directions (accelerometer sees the push force). With |dw| <= 5 rad/s the push alone cannot meet TILT_PREDICTED "
          "(tilt + 0.1 s * rate < 60 deg), so any trip is the policy's. Pass = never leaves POLICY, no supervisor fault, no crash, "
          "back within 0.15 m / 10 deg for the last 2 s. Recovery = time after the release until |p - target| < 0.2 m and tilt < 15 deg for good.\n")
        w("| policy | disturbance | pass | trips | max tilt med / max (deg) | recovery med / max (s) | peak err med (m) | RMS last 3 s med (cm) |")
        w("|---|---|---|---|---|---|---|---|")
        for label, kind, r in dist:
            trips = sorted({f for b in r.fault_history for f in sitl.fault_names(int(b))})
            nt = int((r.fault_history != 0).sum())
            w(f"| {label} | {kind} | {int(r.passed.sum())}/{len(r.passed)} | {nt} {'(' + ','.join(trips) + ')' if trips else ''} | "
              f"{np.median(r.max_tilt_deg):.0f} / {r.max_tilt_deg.max():.0f} | {np.nanmedian(r.recover_s):.2f} / {np.nanmax(r.recover_s):.2f} | "
              f"{np.median(r.peak_err_m):.2f} | {cm(np.median(r.rms_after_m))} |")
        w("")
    if gap_rows:
        w("### Sim-to-real gap attribution (`fw/sitl/diagnose_gap.py`)\n")
        w(f"POLICY hover, {nr} randomized vehicles (seed 11). `nf::InputOverride` (diagnostics only) replaces selected policy / rate-PID "
          "inputs with what the v1 training env fed: Q = true attitude, W = raw gyro as obs 12-14, P / V = true position / velocity + white "
          "noise, PID = raw gyro into the rate PID. RMS error is the true 3D error over 10 s. 'offset' = |mean position - target|.\n")
        w("| policy | inputs | RMS err min / med / max (cm) | med offset (cm) | max tilt (deg) | trips |")
        w("|---|---|---|---|---|---|")
        for label, name, d in gap_rows:
            r_ = d["rms"] * 100
            w(f"| {label} | {name} | {r_.min():.1f} / {np.median(r_):.1f} / {r_.max():.1f} | {np.median(d['offset']) * 100:.1f} | "
              f"{d['max_tilt'].max():.0f} | {int(d['trips'].sum())} |")
        w("")
    w("Deliberately bad policies (random weights, 19-64-64-4, std 1), engaged at hover:\n")
    w("| seed | first fault | entered | latency (ms) | alt lost (m) | final | pass |")
    w("|---|---|---|---|---|---|---|")
    for b in bad:
        w(f"| {b['seed']} | {b.get('first_fault', '-')} | {b.get('mode_after', '-')} | {b.get('latency_ms', float('nan')):.0f} | "
          f"{b.get('alt_lost_m', float('nan')):.3f} | {b.get('final_mode', '-')} | {'PASS' if b['passed'] else 'FAIL'} |")

    w("\n## (c)+(d) Fault injection (supervisor transitions)\n")
    w("Latency is measured from fault onset to the supervisor leaving POLICY, in 1 ms loop steps. Alt lost = true z at onset "
      "minus the minimum true z over the next 3 s. Switch tilt is the true tilt when the supervisor leaves POLICY. Max tilt is over the same 3 s. "
      "Pass requires the expected first fault, the expected mode, no crash (touchdown > 1 m/s or > 34 deg), and a "
      "stable hover at the end (or a landing and MOTOR_STOP for RC loss).\n")
    w("| vehicle | fault | first fault | entered | latency (ms) | alt lost (m) | switch tilt (deg) | max tilt (deg) | outcome | result |")
    w("|---|---|---|---|---|---|---|---|---|---|")
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
    w(f"| same random stress, `RateController.reset` without a measurement (old sim/env.py reset; env fixed, FOLLOWUPS 1) | {par_env['max_abs']:.3f} | "
      f"{par_env['mean_abs']:.1e} | first step {par_env['first_step_max_abs']:.3f}; {par_env['frac_gt_tol']:.0%} of samples > tol |")
    w(f"| gyro filter chain: sim/fw_estimator.GyroFilterChain vs firmware LPF + RPM notches (rad/s), 32 x 5000 | {gfp['max_abs']:.1e} | | "
      f"{gfp['notch_toggles']} notch (de)activations |")
    w(f"| full firmware loop vs sim measurement path + sim/ctbr.py (whole armed flight replayed) | {gap['max_abs']:.1e} | rms {gap['rms']:.1e} | "
      f"Mahony bias drift {gap['bias_drift']:.1e} rad/s (replay uses a constant bias) |")
    w(f"| same, raw gyro into sim/ctbr.py (v1 training path) | {gap_raw['max_abs']:.4f} | rms {gap_raw['rms']:.4f} | "
      f"gyro noise std {gap_raw['gyro_noise_std']:.3f} rad/s |")
    w("\nSince hover_ctbr_v2 the sim feeds the rate PID (and obs 12-14) the firmware's measurement: 150 Hz biquad -> RPM notches -> "
      "minus the bias estimate (INTERFACES §9). The env reset seeds the PID with the first measurement "
      "(`test_sim_env_inner_loop_matches_firmware_from_reset`: every command within 1e-5 from the first step).\n")

    w("## Findings\n")
    w("1. **Fixed in fw/core (one line, `ControlLoop::do_arm`)**: arming kept the translation-KF velocity. On the ground "
      "the horizontal velocity cannot be observed (no flow below 8 cm), so it random-walks while disarmed. After 60 s it had reached ~5 m/s, "
      "and FAULT_ESTIMATOR tripped right after takeoff, which sent the vehicle into CONTROLLED_DESCENT. Arming now zeroes the "
      "horizontal velocity.")
    fn = next(r for r in f_nom if r.fault == "flip")
    fr = next(r for r in f_rnd if r.fault == "flip")
    nr_ = next(r for r in f_rnd if r.fault == "nan")
    w("2. **Fixed (FOLLOWUPS #4): the arming IMU A/B check now uses filtered values.** It compares the 1 Hz low-passed difference "
      "(gyro_a - calibrated bias_a) - gyro_b, and accel_a - accel_b, after at least 0.5 s of IMU B data, instead of single raw samples. "
      "Before, some randomized vehicles needed up to 9 arm flicks. Now all 160 vehicles of the seed 1-4 randomized hover runs (2 s and 120 s power-on) arm on the first flick. "
      "A real 0.3 rad/s IMU B offset is still refused (unit test).")
    w("3. **Fixed (FOLLOWUPS #3): at-rest gyro calibration (`nf::GyroCalibrator`).** While disarmed, still windows of 0.5 s "
      "(gyro/accel standard deviation, mean gyro and |g| gates, plus gross-motion gates on every sample) are averaged. The "
      "result seeds the Mahony bias. Arming is refused (ARM_FAIL_GYRO_CAL / ARM_FAIL_MOTION) until a still window completes, "
      "and again after any motion. See row 3 of (a).")
    w("4. **Fixed (FOLLOWUPS #1): sim/env.py seeds the rate PID on reset** with the first measurement, like the firmware "
      "(`QuadEnv.reinit_sensing`, also used by train/eval.py). Was a D-term kick of up to 0.22 in motor command on every episode start.")
    w("5. **Fixed (FOLLOWUPS #2): one filter path in §9.** The sim rate PID and obs 12-14 use the firmware's filtered, bias-corrected "
      "gyro (`sim/fw_estimator.GyroFilterChain`, bit-level parity in table (e)). Full-loop replay gap 0.035 -> <1e-4.")
    w(f"6. **Fixed (FOLLOWUPS #5): flips are caught early.** A new POLICY-only monitor, TILT_PREDICTED, trips when the tilt rate is at least 4 rad/s "
      f"and tilt + tilt_rate * 0.1 s > 60 deg for 20 ms. Before, the TILT monitor (60 deg + 50 ms) handed over at about 94 deg, the tilt peaked "
      f"at 110 deg and the vehicle lost 0.63 m. Now the switch-over tilt is {fn.switch_tilt_deg:.0f} deg nominal / {fr.switch_tilt_deg:.0f} deg randomized, "
      f"the tilt peaks at {fn.max_tilt_deg:.0f} / {fr.max_tilt_deg:.0f} deg and the vehicle loses {fn.alt_lost_m:.2f} / {fr.alt_lost_m:.2f} m. "
      "On the clean-manoeuvring unit-test data the predicted tilt peaks at 0.69 rad (trip at 1.05) and the tilt rate at 2.6 rad/s (gate 4).")
    w(f"7. **Fixed (FOLLOWUPS #6): bumpless fallback handover.** The loop keeps a hover-thrust estimate, a 0.5 Hz low-pass of "
      f"g * T_alloc / f_bz while flying near level. On handover from POLICY it preloads the cascade's vertical velocity integrator with it, "
      f"instead of starting from nominal weight. Altitude lost in the randomized NaN case was 0.348 m before and is {nr_.alt_lost_m:.3f} m now.")
    w("8. **hover_ctbr_v1 -> v2 (sim-to-real gap).** The gap-attribution table shows the firmware's flow/ToF KF velocity is the "
      "dominant gap for v1: its error is coloured (autocorrelation 0.66 at 10 ms, ~0 at 50 ms, rms ~0.6 x vel_noise_std) and "
      "the policy, trained on white per-step noise, answers it with 20-37 deg tilts and TILT_PREDICTED trips; true attitude, "
      "gyro path and position matter little. v2 is trained on a port of the firmware sensing path "
      "(sim/fw_estimator.py: gyro filter chain + TranslationKf with SITL-identical range/flow) plus supervisor-envelope shaping.")
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
