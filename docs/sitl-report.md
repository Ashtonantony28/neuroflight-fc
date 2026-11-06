# SITL report (WS6)

Generated 2026-11-06 23:11 by `python fw/sitl/report.py` in 117 s. The real firmware core (`fw/core`, via `fw/sitl/libnf_sitl.so`) runs at 1 kHz (policy 500 Hz) inside `sim.Quadrotor` (sim latency + motor lag per vehicle). The sensors are IMU A and IMU B at 1 kHz, and ToF and optical flow at 100 Hz. Each run starts with the vehicle powered on and still on the ground for 120 s. It is then armed into FALLBACK_HOLD and takes off to 1 m. Policy runs switch to POLICY with the RC re-engage at t=6 s, and faults are injected at t=9 s.

Trained policy `train/policies/hover_ctbr_v1.nfpol.json`: **not present** (trained-policy rows skipped).

## (a) Fallback controller: takeoff and 10 s hover at 1 m

Pass = 3D RMS error < 5 cm over the 10 s window, no faults, no crash. For batches the RMS column is min / median / max.

| scenario | RMS err (cm) | max err (cm) | worst z RMS (cm) | worst xy drift (cm) | faults | pass |
|---|---|---|---|---|---|---|
| fallback hover, nominal, power-on 120 s | 0.2 | 0.4 | 0.2 | 0.0 | - | 1/1 |
| fallback hover, randomized x20, power-on 120 s | 0.8 / 2.3 / 4.1 | 7.4 | 2.2 | 5.8 | - | 20/20 |
| fallback hover, randomized x20, power-on 1.5 s | 8.9 / 24.1 / 50.1 | 60.5 | 2.3 | 60.5 | - | 0/20 |

The last row is informational. It powers on for only 1.5 s, so the Mahony gyro bias has not converged (time constant kp/ki = 20 s). The flow velocity then inherits h * delta_bias and the vehicle drifts horizontally. See the findings.

## (b) Policy mode

| scenario | RMS err (cm) | max err (cm) | worst z RMS (cm) | worst xy drift (cm) | faults | pass |
|---|---|---|---|---|---|---|
| synthetic linear policy hover, nominal | 0.0 | 0.1 | 0.0 | 0.0 | - | 1/1 |
| synthetic linear policy hover, randomized x4 | 0.7 / 1.5 / 2.1 | 3.0 | 0.9 | 2.7 | - | 4/4 |

The synthetic linear policy is a hand-built single tanh layer: PD on the §4 observation. It stands in for a trained policy and carries the fault injection. On randomized vehicles it is calibrated with the true mass and thrust, because it has no integrator.

Deliberately bad policies (random weights, 19-64-64-4, std 1), engaged at hover:

| seed | first fault | entered | latency (ms) | alt lost (m) | final | pass |
|---|---|---|---|---|---|---|
| 0 | POLICY_OSCILLATION | FALLBACK_HOLD | 68 | 0.016 | FALLBACK_HOLD | PASS |
| 1 | POLICY_OSCILLATION | FALLBACK_HOLD | 80 | 0.066 | FALLBACK_HOLD | PASS |
| 2 | POLICY_OSCILLATION | FALLBACK_HOLD | 88 | 0.020 | FALLBACK_HOLD | PASS |

## (c)+(d) Fault injection (supervisor transitions)

Latency is measured from fault onset to the supervisor leaving POLICY, in 1 ms loop steps. Alt lost = true z at onset minus the minimum true z over the next 3 s. Max tilt is over the same 3 s. Pass requires the expected first fault, the expected mode, no crash (touchdown > 1 m/s or > 34 deg), and a stable hover at the end (or a landing and MOTOR_STOP for RC loss).

| vehicle | fault | first fault | entered | latency (ms) | alt lost (m) | max tilt (deg) | outcome | result |
|---|---|---|---|---|---|---|---|---|
| nominal | nan | POLICY_NONFINITE | FALLBACK_HOLD | same loop | 0.000 | 0 | FALLBACK_HOLD, z=1.00 m | PASS |
| nominal | saturate | POLICY_SATURATION | FALLBACK_HOLD | 200 | 0.001 | 0 | FALLBACK_HOLD, z=1.00 m | PASS |
| nominal | oscillate | POLICY_OSCILLATION | FALLBACK_HOLD | 48 | 0.001 | 0 | FALLBACK_HOLD, z=1.00 m | PASS |
| nominal | flip | TILT | FALLBACK_HOLD | 197 | 0.633 | 110 | FALLBACK_HOLD, z=1.06 m | PASS |
| nominal | imu_b | IMU_DISAGREE | FALLBACK_HOLD | 50 | 0.000 | 0 | FALLBACK_HOLD, z=1.00 m | PASS |
| nominal | rc_loss | RC_LOSS | CONTROLLED_DESCENT | 300 | 0.946 | 0 | landed at 0.39 m/s, MOTOR_STOP | PASS |
| nominal | overrun | DEADLINE | FALLBACK_HOLD | 4 | 0.000 | 0 | FALLBACK_HOLD, z=1.00 m | PASS |
| nominal | kill | KILL | MOTOR_STOP | same loop | 1.000 | 0 | motors 0 after 0 loops; falls (expected) | PASS |
| randomized | nan | POLICY_NONFINITE | FALLBACK_HOLD | same loop | 0.348 | 1 | FALLBACK_HOLD, z=0.94 m | PASS |
| randomized | saturate | POLICY_SATURATION | FALLBACK_HOLD | 200 | 0.001 | 2 | FALLBACK_HOLD, z=1.06 m | PASS |
| randomized | oscillate | POLICY_OSCILLATION | FALLBACK_HOLD | 48 | 0.016 | 1 | FALLBACK_HOLD, z=1.00 m | PASS |
| randomized | flip | TILT | FALLBACK_HOLD | 197 | 0.635 | 113 | FALLBACK_HOLD, z=1.05 m | PASS |
| randomized | imu_b | IMU_DISAGREE | FALLBACK_HOLD | 50 | 0.000 | 1 | FALLBACK_HOLD, z=1.01 m | PASS |
| randomized | rc_loss | RC_LOSS | CONTROLLED_DESCENT | 300 | 1.005 | 1 | landed at 0.43 m/s, MOTOR_STOP | PASS |
| randomized | overrun | DEADLINE | FALLBACK_HOLD | 4 | 0.000 | 1 | FALLBACK_HOLD, z=1.02 m | PASS |
| randomized | kill | KILL | MOTOR_STOP | same loop | 1.000 | 1 | motors 0 after 0 loops; falls (expected) | PASS |

How each fault is injected:
- **nan**: last-layer bias of a1 corrupted to NaN (CRC not re-checked in flight). Expected: POLICY_NONFINITE -> FALLBACK_HOLD.
- **saturate**: a3 (yaw rate) pinned at +1 (bias +6). Expected: POLICY_SATURATION -> FALLBACK_HOLD.
- **oscillate**: hot-swap to a policy with a 250 Hz limit cycle on a1 (amplitude ~0.9). Expected: POLICY_OSCILLATION -> FALLBACK_HOLD.
- **flip**: a1 (roll rate) pinned near +1: 10 rad/s roll. Expected: TILT -> FALLBACK_HOLD.
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
| full firmware loop vs sim/ctbr.py on raw gyro (informational) | 0.0171 | rms 0.0047 | gyro noise std 0.023 rad/s |

The last row shows the input-side gap. The firmware rate PID sees the gyro after a 150 Hz biquad, the RPM notches and Mahony bias correction. Training feeds the raw noisy gyro to the PID.

## Findings

1. **Fixed in fw/core (one line, `ControlLoop::do_arm`)**: arming kept the translation-KF velocity. On the ground the horizontal velocity cannot be observed (no flow below 8 cm), so it random-walks while disarmed. After 60 s it had reached ~5 m/s, and FAULT_ESTIMATOR tripped right after takeoff, which sent the vehicle into CONTROLLED_DESCENT. Arming now zeroes the horizontal velocity.
2. **Arming IMU check uses single raw samples** (`arming_checks`: |gyro_a - gyro_b| <= 0.1 rad/s). Under the randomized noise and bias it fails often, and some vehicles needed up to 9 arm flicks. It should compare filtered or bias-corrected values.
3. **Gyro bias converges slowly** (Mahony kp/ki = 20 s) and the firmware has no stationary calibration. Flow velocity error = h * delta_bias, so x/y drifts by tens of cm in 10 s if the vehicle arms soon after power-on (row 3 of (a)).
4. **sim/env.py resets RateController without the gyro**, so the first step gets a D-term kick (table (e)). The firmware seeds the previous measurement with the first sample.
5. **Filter path in §9**: the firmware PID input is filtered and bias-corrected; the sim uses the raw gyro (table (e), last row).
6. **Flip detection is slow**: the TILT trip takes 60 deg plus 50 ms, the commanded 10 rad/s stays under the 12 rad/s RATE limit, so the fallback takes over at about 110 deg and the vehicle loses about 0.6 m. That is fine at 1 m and marginal lower down.
7. Handover sag: the cascade's velocity integrator is reset when the fallback takes over, so a heavy vehicle can sag for about 1 s (alt lost up to ~0.35 m on randomized NaN/saturate rows).
