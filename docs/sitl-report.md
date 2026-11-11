# SITL report (WS6)

Generated 2026-11-11 04:56 by `python fw/sitl/report.py` in 792 s. The real firmware core (`fw/core`, via `fw/sitl/libnf_sitl.so`) runs at 1 kHz (policy 500 Hz) inside `sim.Quadrotor` (sim latency + motor lag per vehicle). The sensors are IMU A and IMU B at 1 kHz, and ToF and optical flow at 100 Hz. Each run starts with the vehicle powered on and still on the ground for 120 s. It is then armed into FALLBACK_HOLD and takes off to 1 m. Policy runs switch to POLICY with the RC re-engage at t=6 s, and faults are injected at t=9 s.

Trained policy `train/policies/hover_ctbr_v1.nfpol.json`: present.

Trained policy `train/policies/hover_ctbr_v2.nfpol.json`: present.

Trained policy `train/policies/hover_ctbr_v2.q.nfpol.json`: present.

Trained policy `train/policies/hover_ctbr_v3.nfpol.json`: present.

Trained policy `train/policies/hover_ctbr_v3.q.nfpol.json`: present.

## (a) Fallback controller: takeoff and 10 s hover at 1 m

Pass = 3D RMS error < 5 cm over the 10 s window, no faults, no crash. For batches the RMS column is min / median / max.

| scenario | RMS err (cm) | max err (cm) | worst z RMS (cm) | worst xy drift (cm) | faults | pass |
|---|---|---|---|---|---|---|
| fallback hover, nominal, power-on 120 s | 0.2 | 0.4 | 0.2 | 0.0 | - | 1/1 |
| fallback hover, randomized x20, power-on 120 s | 0.6 / 2.2 / 4.2 | 6.7 | 2.2 | 6.4 | - | 20/20 |
| fallback hover, randomized x20, power-on 1.5 s | 0.7 / 2.7 / 5.9 | 10.1 | 2.2 | 10.1 | - | 18/20 |

The last row powers on for only 1.5 s and arms at t = 2.0 s. Before the at-rest gyro calibration it was 0/20 (median 24 cm, up to 8 arm flicks): the Mahony bias had not converged (kp/ki = 20 s) and the flow velocity inherited h * delta_bias. Now every vehicle arms on the first flick (max attempts 1) with the bias from about 2 s of still averaging, and 18/20 hover under 5 cm. The 5 cm bound is close to the noise floor of flow-only xy dead reckoning for max-spec sensor noise. Over seeds 1-4 (80 vehicles), 73/80 pass at 2 s and 77/80 after a 120 s power-on. The pytest version (`test_arm_2s_after_power_on_and_hover_20_randomized`) allows 2 of 20 between 5 and 7.5 cm.

## (b) Policy mode

| scenario | RMS err (cm) | max err (cm) | worst z RMS (cm) | worst xy drift (cm) | faults | pass |
|---|---|---|---|---|---|---|
| synthetic linear policy hover, nominal | 0.0 | 0.1 | 0.0 | 0.0 | - | 1/1 |
| synthetic linear policy hover, randomized x4 | 0.7 / 1.5 / 2.4 | 3.1 | 0.9 | 2.3 | - | 4/4 |
| trained hover_ctbr_v1 hover, nominal | 5.6 | 7.3 | 3.4 | 5.0 | - | 0/1 |
| trained hover_ctbr_v1 hover, randomized x20 | 4.8 / 10.0 / 20.0 | 46.7 | 8.4 | 29.8 | TILT_PREDICTED | 2/20 |
| trained hover_ctbr_v2 hover, nominal | 2.4 | 4.7 | 1.3 | 1.9 | - | 1/1 |
| trained hover_ctbr_v2 hover, randomized x20 | 2.1 / 3.5 / 4.8 | 8.0 | 3.4 | 4.9 | - | 20/20 |
| hover_ctbr_v2 int8 (nf::Int8Model) hover, randomized x20 | 2.5 / 3.7 / 5.2 | 9.2 | 3.4 | 4.6 | - | 19/20 |
| trained hover_ctbr_v3 hover, nominal | 1.7 | 3.6 | 0.3 | 1.5 | - | 1/1 |
| trained hover_ctbr_v3 hover, randomized x20 | 1.4 / 2.8 / 7.7 | 9.9 | 2.2 | 8.8 | - | 18/20 |
| hover_ctbr_v3 int8 (nf::Int8Model) hover, randomized x20 | 1.7 / 3.2 / 7.1 | 9.4 | 2.4 | 7.4 | - | 18/20 |

The synthetic linear policy is a hand-built single tanh layer: PD on the §4 observation. It stands in for a trained policy and carries the fault injection. On randomized vehicles it is calibrated with the true mass and thrust, because it has no integrator.

hover_ctbr_v1 was trained on true attitude/velocity (+ white noise) and the raw gyro; hover_ctbr_v2 on the firmware sensing path (sim/fw_estimator.py). The int8 row flies the quantized v2 through the firmware's int8 inference (`nf_sitl_set_policy_i8` -> `nf::Int8Model`). hover_ctbr_v3 adds the firmware Mahony estimate (with the KF-acceleration reference) to the training sim, RATE / ESTIMATOR margins and thrust-noise shaping (train/policies/hover_ctbr_v3.md). All rows fly the current firmware (Mahony KF-acceleration reference on, see (b2)).

Thrust-command noise in POLICY hover (FALLBACK_HOLD for the fallback), same 10 s window and vehicles as the hover rows: RMS of the collective-thrust command above 50 Hz (N), and the ratio to the fallback's cascaded PID.

| controller | HF thrust rms med (N) | ratio to fallback |
|---|---|---|
| fallback (cascaded PID) | 0.0054 | 1.00 |
| hover_ctbr_v2 | 0.0210 | 3.92 |
| hover_ctbr_v3 | 0.0088 | 1.64 |

### Policy recovery from disturbances (randomized vehicles)

At t = 9 s, 3 s after the POLICY engage. **tilt**: an external 'hand' holds the vehicle in place and rotates it about a random horizontal axis at 2.9 rad/s to 59.8 deg, then releases it at rest (gyro and accelerometer see the motion; both envelope monitors stay quiet during the hold: tilt < 60.16 deg, rate < 4 rad/s). **throw**: a 20 ms push, |dv| U[0, 3] m/s and |dw| U[0, 5] rad/s in random directions (accelerometer sees the push force). With |dw| <= 5 rad/s the push alone cannot meet TILT_PREDICTED (tilt + 0.1 s * rate < 60 deg), so any trip is the policy's. Pass = never leaves POLICY, no supervisor fault, no crash, back within 0.15 m / 10 deg for the last 2 s. Recovery = time after the release until |p - target| < 0.2 m and tilt < 15 deg for good.

| policy | disturbance | pass | trips | max tilt med / max (deg) | recovery med / max (s) | peak err med (m) | RMS last 3 s med (cm) |
|---|---|---|---|---|---|---|---|
| hover_ctbr_v1 | tilt | 14/20 | 4 (RATE,TILT,TILT_PREDICTED) | 60 / 67 | 0.47 / 5.64 | 0.18 | 7.2 |
| hover_ctbr_v1 | throw | 8/20 | 10 (ESTIMATOR,TILT_PREDICTED) | 37 / 45 | 1.84 / 5.98 | 0.29 | 12.8 |
| hover_ctbr_v2 | tilt | 16/20 | 4 (ESTIMATOR,RATE,TILT_PREDICTED) | 60 / 61 | 0.22 / 5.64 | 0.16 | 7.2 |
| hover_ctbr_v2 | throw | 19/20 | 1 (TILT_PREDICTED) | 25 / 45 | 0.67 / 5.98 | 0.29 | 3.6 |
| hover_ctbr_v2 int8 | tilt | 17/20 | 3 (ESTIMATOR,RATE,TILT_PREDICTED) | 60 / 61 | 0.20 / 5.64 | 0.16 | 6.5 |
| hover_ctbr_v2 int8 | throw | 19/20 | 1 (TILT_PREDICTED) | 25 / 45 | 0.61 / 5.98 | 0.28 | 3.8 |
| hover_ctbr_v3 | tilt | 19/20 | 0  | 60 / 63 | 0.38 / 0.55 | 0.16 | 6.7 |
| hover_ctbr_v3 | throw | 20/20 | 0  | 26 / 49 | 0.50 / 1.35 | 0.32 | 2.4 |
| hover_ctbr_v3 int8 | tilt | 20/20 | 0  | 60 / 63 | 0.38 / 0.56 | 0.15 | 5.5 |
| hover_ctbr_v3 int8 | throw | 20/20 | 0  | 26 / 49 | 0.49 / 1.36 | 0.31 | 3.3 |
| hover_ctbr_v2 | throw, |dw| <= 10 rad/s (stress, informational) | 18/20 | 2 (TILT_PREDICTED) | 26 / 44 | 0.68 / 5.98 | 0.29 | 4.2 |
| hover_ctbr_v3 | throw, |dw| <= 10 rad/s (stress, informational) | 20/20 | 0  | 27 / 45 | 0.62 / 1.34 | 0.33 | 3.0 |

### (b2) Mahony specific-force reference (FOLLOWUPS 11)

The firmware Mahony filter compares the accelerometer with the expected specific force. Up to v2 that was gravity alone, so while the vehicle accelerates (hard braking after a push) the estimate is pulled towards level. `ControlLoopConfig::mahony_kf_ref` (now default) uses KF acceleration + g instead: a 5 Hz low-pass of the derivative of the flow-corrected KF velocity while armed with valid flow/height (horizontal part clamped to 6 m/s^2, then a 0.3 m/s^2 soft deadband so hover noise stays out of the attitude: without it the fallback hover lost ~1 cm on some vehicles), else gravity. Same disturbances as above; attitude error = angle between true and estimated body z axes after the disturbance.

| controller | disturbance | reference | pass | trips | att err max med / max (deg) | innovation max med / max (rad) | hover att err rms (deg) |
|---|---|---|---|---|---|---|---|
| fallback | throw | gravity | 19/20 | 1 (ESTIMATOR) | 6.7 / 13.5 | 0.231 / 0.504 | 0.10 |
| fallback | throw | KF accel + g | 20/20 | 0  | 1.5 / 3.1 | 0.054 / 0.178 | 0.10 |
| fallback | tilt | gravity | 20/20 | 0  | 3.0 / 3.9 | 0.127 / 0.163 | 0.10 |
| fallback | tilt | KF accel + g | 20/20 | 0  | 1.5 / 2.4 | 0.061 / 0.073 | 0.10 |
| hover_ctbr_v2 | throw | gravity | 18/20 | 2 (ESTIMATOR,TILT_PREDICTED) | 3.1 / 10.4 | 0.160 / 0.458 | 0.32 |
| hover_ctbr_v2 | throw | KF accel + g | 19/20 | 1 (TILT_PREDICTED) | 1.3 / 2.4 | 0.064 / 0.153 | 0.25 |
| hover_ctbr_v2 | tilt | gravity | 17/20 | 3 (ESTIMATOR,RATE,TILT_PREDICTED) | 3.0 / 4.6 | 0.115 / 0.193 | 0.32 |
| hover_ctbr_v2 | tilt | KF accel + g | 16/20 | 4 (ESTIMATOR,RATE,TILT_PREDICTED) | 1.9 / 7.7 | 0.055 / 0.089 | 0.25 |
| hover_ctbr_v3 | throw | gravity | 15/20 | 5 (ESTIMATOR) | 4.4 / 9.9 | 0.211 / 0.540 | 0.38 |
| hover_ctbr_v3 | throw | KF accel + g | 20/20 | 0  | 1.2 / 2.8 | 0.075 / 0.186 | 0.26 |
| hover_ctbr_v3 | tilt | gravity | 19/20 | 0  | 3.3 / 5.6 | 0.191 / 0.267 | 0.38 |
| hover_ctbr_v3 | tilt | KF accel + g | 19/20 | 0  | 1.7 / 5.2 | 0.061 / 0.093 | 0.26 |

### Sim-to-real gap attribution (`fw/sitl/diagnose_gap.py`)

POLICY hover, 20 randomized vehicles (seed 11). `nf::InputOverride` (diagnostics only) replaces selected policy / rate-PID inputs with what the v1 training env fed: Q = true attitude, W = raw gyro as obs 12-14, P / V = true position / velocity + white noise, PID = raw gyro into the rate PID. RMS error is the true 3D error over 10 s. 'offset' = |mean position - target|.

| policy | inputs | RMS err min / med / max (cm) | med offset (cm) | max tilt (deg) | trips |
|---|---|---|---|---|---|
| hover_ctbr_v1 | firmware as is | 4.8 / 10.0 / 20.0 | 7.2 | 27 | 2 |
| hover_ctbr_v1 | training-env inputs (ALL) | 3.7 / 5.2 / 7.6 | 4.6 | 4 | 0 |
| hover_ctbr_v1 | firmware + true velocity (V) | 2.1 / 5.8 / 9.6 | 5.5 | 4 | 0 |
| hover_ctbr_v1 | ALL - V (firmware velocity only) | 2.6 / 5.0 / 8.1 | 3.8 | 25 | 0 |
| hover_ctbr_v1 | firmware + true attitude (Q) | 4.8 / 9.6 / 21.5 | 6.2 | 31 | 3 |
| hover_ctbr_v1 | firmware + raw gyro to PID (PID) | 4.7 / 7.7 / 17.1 | 6.4 | 27 | 1 |
| hover_ctbr_v1 | firmware + raw gyro obs (W) | 4.6 / 7.7 / 18.5 | 6.0 | 32 | 1 |
| hover_ctbr_v1 | firmware + true position (P) | 2.7 / 7.2 / 20.7 | 5.7 | 30 | 3 |
| hover_ctbr_v2 | firmware as is | 2.1 / 3.5 / 4.8 | 2.6 | 4 | 0 |
| hover_ctbr_v2 | training-env inputs (ALL) | 2.1 / 2.7 / 4.3 | 1.8 | 2 | 0 |
| hover_ctbr_v2 | firmware + true velocity (V) | 2.0 / 3.5 / 4.5 | 2.7 | 3 | 0 |
| hover_ctbr_v2 | ALL - V (firmware velocity only) | 2.1 / 2.8 / 4.4 | 1.8 | 3 | 0 |
| hover_ctbr_v2 | firmware + true attitude (Q) | 2.1 / 3.4 / 4.7 | 2.5 | 4 | 0 |
| hover_ctbr_v2 | firmware + raw gyro to PID (PID) | 2.0 / 3.6 / 4.7 | 2.4 | 3 | 0 |
| hover_ctbr_v2 | firmware + raw gyro obs (W) | 2.1 / 3.5 / 4.8 | 2.6 | 4 | 0 |
| hover_ctbr_v2 | firmware + true position (P) | 2.1 / 2.8 / 4.3 | 1.9 | 4 | 0 |

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
| nominal | flip | TILT_PREDICTED | FALLBACK_HOLD | 90 | 0.108 | 27 | 49 | FALLBACK_HOLD, z=1.01 m | PASS |
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
| nominal, trained base | nan | POLICY_NONFINITE | FALLBACK_HOLD | same loop | 0.024 | 6 | 6 | FALLBACK_HOLD, z=0.96 m | PASS |
| nominal, trained base | saturate | POLICY_SATURATION | FALLBACK_HOLD | 200 | 0.229 | 36 | 39 | FALLBACK_HOLD, z=0.93 m | PASS |
| nominal, trained base | flip | TILT_PREDICTED | FALLBACK_HOLD | 84 | 0.142 | 27 | 48 | FALLBACK_HOLD, z=0.96 m | PASS |
| nominal, trained base | imu_b | IMU_DISAGREE | FALLBACK_HOLD | 50 | 0.039 | 3 | 8 | FALLBACK_HOLD, z=0.96 m | PASS |
| nominal, trained base | overrun | DEADLINE | FALLBACK_HOLD | 4 | 0.026 | 6 | 6 | FALLBACK_HOLD, z=0.96 m | PASS |

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
| same random stress, `RateController.reset` without a measurement (old sim/env.py reset; env fixed, FOLLOWUPS 1) | 0.216 | 1.6e-04 | first step 0.216; 34% of samples > tol |
| gyro filter chain: sim/fw_estimator.GyroFilterChain vs firmware LPF + RPM notches (rad/s), 32 x 5000 | 4.8e-06 | | 10164 notch (de)activations |
| Mahony: sim/fw_estimator.Mahony vs nf::Mahony, 32 x 20000, random gyro/accel, KF-accel reference on every other step, 53% of samples inside the 2 rad/s accel gate | q 6.4e-06 | | bias 6.0e-07 rad/s, innovation 8.8e-06 rad (from t = 0 incl. the boosted-gain phase, and in flight) |
| full firmware loop vs sim measurement path + sim/ctbr.py (whole armed flight replayed) | 3.2e-05 | rms 2.0e-05 | Mahony bias drift 2.2e-04 rad/s (replay uses a constant bias) |
| same, raw gyro into sim/ctbr.py (v1 training path) | 0.0787 | rms 0.0376 | gyro noise std 0.023 rad/s |

Since hover_ctbr_v2 the sim feeds the rate PID (and obs 12-14) the firmware's measurement: 150 Hz biquad -> RPM notches -> minus the bias estimate (INTERFACES §9). The env reset seeds the PID with the first measurement (`test_sim_env_inner_loop_matches_firmware_from_reset`: every command within 1e-5 from the first step).

## Findings

1. **Fixed in fw/core (one line, `ControlLoop::do_arm`)**: arming kept the translation-KF velocity. On the ground the horizontal velocity cannot be observed (no flow below 8 cm), so it random-walks while disarmed. After 60 s it had reached ~5 m/s, and FAULT_ESTIMATOR tripped right after takeoff, which sent the vehicle into CONTROLLED_DESCENT. Arming now zeroes the horizontal velocity.
2. **Fixed (FOLLOWUPS #4): the arming IMU A/B check now uses filtered values.** It compares the 1 Hz low-passed difference (gyro_a - calibrated bias_a) - gyro_b, and accel_a - accel_b, after at least 0.5 s of IMU B data, instead of single raw samples. Before, some randomized vehicles needed up to 9 arm flicks. Now all 160 vehicles of the seed 1-4 randomized hover runs (2 s and 120 s power-on) arm on the first flick. A real 0.3 rad/s IMU B offset is still refused (unit test).
3. **Fixed (FOLLOWUPS #3): at-rest gyro calibration (`nf::GyroCalibrator`).** While disarmed, still windows of 0.5 s (gyro/accel standard deviation, mean gyro and |g| gates, plus gross-motion gates on every sample) are averaged. The result seeds the Mahony bias. Arming is refused (ARM_FAIL_GYRO_CAL / ARM_FAIL_MOTION) until a still window completes, and again after any motion. See row 3 of (a).
4. **Fixed (FOLLOWUPS #1): sim/env.py seeds the rate PID on reset** with the first measurement, like the firmware (`QuadEnv.reinit_sensing`, also used by train/eval.py). Was a D-term kick of up to 0.22 in motor command on every episode start.
5. **Fixed (FOLLOWUPS #2): one filter path in §9.** The sim rate PID and obs 12-14 use the firmware's filtered, bias-corrected gyro (`sim/fw_estimator.GyroFilterChain`, bit-level parity in table (e)). Full-loop replay gap 0.035 -> <1e-4.
6. **Fixed (FOLLOWUPS #5): flips are caught early.** A new POLICY-only monitor, TILT_PREDICTED, trips when the tilt rate is at least 4 rad/s and tilt + tilt_rate * 0.1 s > 60 deg for 20 ms. Before, the TILT monitor (60 deg + 50 ms) handed over at about 94 deg, the tilt peaked at 110 deg and the vehicle lost 0.63 m. Now the switch-over tilt is 27 deg nominal / 26 deg randomized, the tilt peaks at 49 / 54 deg and the vehicle loses 0.11 / 0.12 m. On the clean-manoeuvring unit-test data the predicted tilt peaks at 0.69 rad (trip at 1.05) and the tilt rate at 2.6 rad/s (gate 4).
7. **Fixed (FOLLOWUPS #6): bumpless fallback handover.** The loop keeps a hover-thrust estimate, a 0.5 Hz low-pass of g * T_alloc / f_bz while flying near level. On handover from POLICY it preloads the cascade's vertical velocity integrator with it, instead of starting from nominal weight. Altitude lost in the randomized NaN case was 0.348 m before and is 0.008 m now.
8. **hover_ctbr_v1 -> v2 (sim-to-real gap).** The gap-attribution table shows the firmware's flow/ToF KF velocity is the dominant gap for v1: its error is coloured (autocorrelation 0.66 at 10 ms, ~0 at 50 ms, rms ~0.6 x vel_noise_std) and the policy, trained on white per-step noise, answers it with 20-37 deg tilts and TILT_PREDICTED trips; true attitude, gyro path and position matter little. v2 is trained on a port of the firmware sensing path (sim/fw_estimator.py: gyro filter chain + TranslationKf with SITL-identical range/flow) plus supervisor-envelope shaping.
9. **Mahony in the training sim (FOLLOWUPS 8) and KF-acceleration reference (FOLLOWUPS 11).** `sim/fw_estimator.Mahony` ports `nf::Mahony` (gating, bias integral, innovation; parity in table (e)), so training sees the firmware's attitude estimate and its real innovation. The firmware now compares the accelerometer with KF acceleration + g (table (b2)): the 8-14 deg attitude error while braking hard drops to about 1-2 deg and the ESTIMATOR trips on pushes are gone, at the cost of ~0.1 deg more attitude noise in hover.
