# SITL report (WS6)

Generated 2026-11-06 23:34 by `python fw/sitl/report.py` in 125 s. The real firmware core (`fw/core`, via `fw/sitl/libnf_sitl.so`) runs at 1 kHz (policy 500 Hz) inside `sim.Quadrotor` (sim latency + motor lag per vehicle). The sensors are IMU A and IMU B at 1 kHz, and ToF and optical flow at 100 Hz. Each run starts with the vehicle powered on and still on the ground for 120 s. It is then armed into FALLBACK_HOLD and takes off to 1 m. Policy runs switch to POLICY with the RC re-engage at t=6 s, and faults are injected at t=9 s.

Trained policy `train/policies/hover_ctbr_v1.nfpol.json`: **not present** (trained-policy rows skipped).

## (a) Fallback controller: takeoff and 10 s hover at 1 m

Pass = 3D RMS error < 5 cm over the 10 s window, no faults, no crash. For batches the RMS column is min / median / max.

| scenario | RMS err (cm) | max err (cm) | worst z RMS (cm) | worst xy drift (cm) | faults | pass |
|---|---|---|---|---|---|---|
| fallback hover, nominal, power-on 120 s | 0.2 | 0.4 | 0.2 | 0.0 | - | 1/1 |
| fallback hover, randomized x20, power-on 120 s | 0.6 / 2.2 / 4.3 | 6.8 | 2.2 | 6.4 | - | 20/20 |
| fallback hover, randomized x20, power-on 1.5 s | 0.7 / 2.7 / 5.9 | 10.2 | 2.2 | 10.2 | - | 18/20 |

The last row powers on for only 1.5 s and arms at t = 2.0 s. Before the at-rest gyro calibration it was 0/20 (median 24 cm, up to 8 arm flicks): the Mahony bias had not converged (kp/ki = 20 s) and the flow velocity inherited h * delta_bias. Now every vehicle arms on the first flick (max attempts 1) with the bias from about 2 s of still averaging, and 18/20 hover under 5 cm. The 5 cm bound is close to the noise floor of flow-only xy dead reckoning for max-spec sensor noise. Over seeds 1-4 (80 vehicles), 73/80 pass at 2 s and 77/80 after a 120 s power-on. The pytest version (`test_arm_2s_after_power_on_and_hover_20_randomized`) allows 2 of 20 between 5 and 7.5 cm.

## (b) Policy mode

| scenario | RMS err (cm) | max err (cm) | worst z RMS (cm) | worst xy drift (cm) | faults | pass |
|---|---|---|---|---|---|---|
| synthetic linear policy hover, nominal | 0.0 | 0.1 | 0.0 | 0.0 | - | 1/1 |
| synthetic linear policy hover, randomized x4 | 0.7 / 1.5 / 2.3 | 3.2 | 0.9 | 2.3 | - | 4/4 |

The synthetic linear policy is a hand-built single tanh layer: PD on the §4 observation. It stands in for a trained policy and carries the fault injection. On randomized vehicles it is calibrated with the true mass and thrust, because it has no integrator.

Deliberately bad policies (random weights, 19-64-64-4, std 1), engaged at hover:

| seed | first fault | entered | latency (ms) | alt lost (m) | final | pass |
|---|---|---|---|---|---|---|
| 0 | POLICY_OSCILLATION | FALLBACK_HOLD | 68 | 0.000 | FALLBACK_HOLD | PASS |
| 1 | POLICY_OSCILLATION | FALLBACK_HOLD | 80 | 0.004 | FALLBACK_HOLD | PASS |
| 2 | POLICY_OSCILLATION | FALLBACK_HOLD | 88 | 0.000 | FALLBACK_HOLD | PASS |

## (c)+(d) Fault injection (supervisor transitions)

Latency is measured from fault onset to the supervisor leaving POLICY, in 1 ms loop steps. Alt lost = true z at onset minus the minimum true z over the next 3 s. Switch tilt is the true tilt when the supervisor leaves POLICY. Max tilt is over the same 3 s. Pass requires the expected first fault, the expected mode, no crash (touchdown > 1 m/s or > 34 deg), and a stable hover at the end (or a landing and MOTOR_STOP for RC loss).

| vehicle | fault | first fault | entered | latency (ms) | alt lost (m) | switch tilt (deg) | max tilt (deg) | outcome | result |
|---|---|---|---|---|---|---|---|---|---|
| nominal | nan | POLICY_NONFINITE | FALLBACK_HOLD | same loop | 0.000 | 0 | 0 | FALLBACK_HOLD, z=1.00 m | PASS |
| nominal | saturate | POLICY_SATURATION | FALLBACK_HOLD | 200 | 0.007 | 0 | 0 | FALLBACK_HOLD, z=1.00 m | PASS |
| nominal | oscillate | POLICY_OSCILLATION | FALLBACK_HOLD | 48 | 0.002 | 0 | 0 | FALLBACK_HOLD, z=1.00 m | PASS |
| nominal | flip | TILT_PREDICTED | FALLBACK_HOLD | 90 | 0.107 | 27 | 49 | FALLBACK_HOLD, z=1.02 m | PASS |
| nominal | imu_b | IMU_DISAGREE | FALLBACK_HOLD | 50 | 0.000 | 0 | 0 | FALLBACK_HOLD, z=1.00 m | PASS |
| nominal | rc_loss | RC_LOSS | CONTROLLED_DESCENT | 300 | 0.946 | 0 | 0 | landed at 0.39 m/s, MOTOR_STOP | PASS |
| nominal | overrun | DEADLINE | FALLBACK_HOLD | 4 | 0.000 | 0 | 0 | FALLBACK_HOLD, z=1.00 m | PASS |
| nominal | kill | KILL | MOTOR_STOP | same loop | 1.000 | 0 | 0 | motors 0 after 0 loops; falls (expected) | PASS |
| randomized | nan | POLICY_NONFINITE | FALLBACK_HOLD | same loop | 0.008 | 0 | 1 | FALLBACK_HOLD, z=1.00 m | PASS |
| randomized | saturate | POLICY_SATURATION | FALLBACK_HOLD | 200 | 0.002 | 1 | 2 | FALLBACK_HOLD, z=1.02 m | PASS |
| randomized | oscillate | POLICY_OSCILLATION | FALLBACK_HOLD | 48 | 0.002 | 0 | 1 | FALLBACK_HOLD, z=1.00 m | PASS |
| randomized | flip | TILT_PREDICTED | FALLBACK_HOLD | 92 | 0.119 | 26 | 54 | FALLBACK_HOLD, z=1.02 m | PASS |
| randomized | imu_b | IMU_DISAGREE | FALLBACK_HOLD | 50 | 0.004 | 0 | 1 | FALLBACK_HOLD, z=1.00 m | PASS |
| randomized | rc_loss | RC_LOSS | CONTROLLED_DESCENT | 300 | 0.948 | 0 | 1 | landed at 0.40 m/s, MOTOR_STOP | PASS |
| randomized | overrun | DEADLINE | FALLBACK_HOLD | 4 | 0.000 | 0 | 1 | FALLBACK_HOLD, z=1.00 m | PASS |
| randomized | kill | KILL | MOTOR_STOP | same loop | 1.000 | 0 | 1 | motors 0 after 0 loops; falls (expected) | PASS |

How each fault is injected:
- **nan**: last-layer bias of a1 corrupted to NaN (CRC not re-checked in flight). Expected: POLICY_NONFINITE -> FALLBACK_HOLD.
- **saturate**: a3 (yaw rate) pinned at +1 (bias +6). Expected: POLICY_SATURATION -> FALLBACK_HOLD.
- **oscillate**: hot-swap to a policy with a 250 Hz limit cycle on a1 (amplitude ~0.9). Expected: POLICY_OSCILLATION -> FALLBACK_HOLD.
- **flip**: a1 (roll rate) pinned near +1: 10 rad/s roll. Expected: TILT_PREDICTED -> FALLBACK_HOLD.
- **imu_b**: IMU B gyro x offset +1.0 rad/s. Expected: IMU_DISAGREE -> FALLBACK_HOLD.
- **rc_loss**: RC link lost (no valid frames). Expected: RC_LOSS -> CONTROLLED_DESCENT.
- **overrun**: simulated inference time 600 us (deadline 400 us). Expected: DEADLINE -> FALLBACK_HOLD.
- **kill**: RC kill switch. Expected: KILL -> MOTOR_STOP.
- Kill in FALLBACK_HOLD (pytest): motor commands are 0 in the same loop iteration and stay at 0.

## (e) CTBR inner loop: sim/ctbr.py vs firmware (INTERFACES.md §9)

Tolerance: |cmd_sim - cmd_fw| <= 1e-05 (commands in [0,1]) for identical (action, gyro) inputs.

| comparison | max abs | mean abs | other |
|---|---|---|---|
| random stress, 64 envs x 4000 steps, 58% of steps mixer-saturated | 3.58e-07 | 9.0e-09 | sat-flag mismatch 0% |
| closed loop (sim env + PID baseline, randomized, tilted/rotating starts), 16 x 4000 | 1.49e-07 | 1.6e-09 | |
| same random stress, but sim reset as `sim/env.py` does it (no gyro) | 0.216 | 1.6e-04 | first step 0.216; 34% of samples > tol |
| full firmware loop vs sim/ctbr.py on raw gyro (informational) | 0.0350 | rms 0.0161 | gyro noise std 0.023 rad/s |

The last row shows the input-side gap. The firmware rate PID sees the gyro after a 150 Hz biquad, the RPM notches and Mahony bias correction. Training feeds the raw noisy gyro to the PID.

## Findings

1. **Fixed in fw/core (one line, `ControlLoop::do_arm`)**: arming kept the translation-KF velocity. On the ground the horizontal velocity cannot be observed (no flow below 8 cm), so it random-walks while disarmed. After 60 s it had reached ~5 m/s, and FAULT_ESTIMATOR tripped right after takeoff, which sent the vehicle into CONTROLLED_DESCENT. Arming now zeroes the horizontal velocity.
2. **Fixed (FOLLOWUPS #4): the arming IMU A/B check now uses filtered values.** It compares the 1 Hz low-passed difference (gyro_a - calibrated bias_a) - gyro_b, and accel_a - accel_b, after at least 0.5 s of IMU B data, instead of single raw samples. Before, some randomized vehicles needed up to 9 arm flicks. Now all 160 vehicles of the seed 1-4 randomized hover runs (2 s and 120 s power-on) arm on the first flick. A real 0.3 rad/s IMU B offset is still refused (unit test).
3. **Fixed (FOLLOWUPS #3): at-rest gyro calibration (`nf::GyroCalibrator`).** While disarmed, still windows of 0.5 s (gyro/accel standard deviation, mean gyro and |g| gates, plus gross-motion gates on every sample) are averaged. The result seeds the Mahony bias. Arming is refused (ARM_FAIL_GYRO_CAL / ARM_FAIL_MOTION) until a still window completes, and again after any motion. See row 3 of (a).
4. **sim/env.py resets RateController without the gyro**, so the first step gets a D-term kick (table (e)). The firmware seeds the previous measurement with the first sample. (Open: FOLLOWUPS #1, sim/ is owned elsewhere.)
5. **Filter path in §9**: the firmware PID input is filtered and bias-corrected; the sim uses the raw gyro (table (e), last row). (Open: FOLLOWUPS #2.)
6. **Fixed (FOLLOWUPS #5): flips are caught early.** A new POLICY-only monitor, TILT_PREDICTED, trips when the tilt rate is at least 4 rad/s and tilt + tilt_rate * 0.1 s > 60 deg for 20 ms. Before, the TILT monitor (60 deg + 50 ms) handed over at about 94 deg, the tilt peaked at 110 deg and the vehicle lost 0.63 m. Now the switch-over tilt is 27 deg nominal / 26 deg randomized, the tilt peaks at 49 / 54 deg and the vehicle loses 0.11 / 0.12 m. On the clean-manoeuvring unit-test data the predicted tilt peaks at 0.69 rad (trip at 1.05) and the tilt rate at 2.6 rad/s (gate 4).
7. **Fixed (FOLLOWUPS #6): bumpless fallback handover.** The loop keeps a hover-thrust estimate, a 0.5 Hz low-pass of g * T_alloc / f_bz while flying near level. On handover from POLICY it preloads the cascade's vertical velocity integrator with it, instead of starting from nominal weight. Altitude lost in the randomized NaN case was 0.348 m before and is 0.008 m now.
