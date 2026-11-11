# hover_ctbr_v3 — CTBR hover/recovery policy trained with the firmware Mahony estimate

`hover_ctbr_v3.nfpol.json` (float32, CRC f32 0xA33E320D) and `hover_ctbr_v3.q.nfpol.json` (int8 block, CRC i8 0x210D300F). Same network and interfaces as v1/v2 (19→64→64→4 tanh, INTERFACES §4–7); the firmware runs it unchanged. Deterministic (mean action). Flown on the current firmware (Mahony KF-acceleration reference on, `ControlLoopConfig::mahony_kf_ref`).

## What changed vs v2
* **Sim (FOLLOWUPS 8):** `sim/fw_estimator.Mahony` ports `nf::Mahony` (gyro-rate gating of the accel correction at 2 rad/s, bias integral, 1 Hz innovation; parity with the C++ ≤ 1e-5, `test_mahony_port_matches_firmware`). `EnvConfig(mahony=True)`: attitude obs and the KF rotation come from it, obs 12-14 / rate PID use filtered gyro minus the *Mahony* bias, and the innovation used for shaping is the real one. `TranslationKf` now also tracks the low-passed NIS (the ESTIMATOR monitor's other input).
* **Firmware (FOLLOWUPS 11):** the Mahony accel reference is KF acceleration + g instead of g (see docs/sitl-report.md (b2)); the sim mirrors it (`KfAccelRef`).
* **Reward** (`train/config/hover_ctbr_v3.yaml`): v2 terms plus `rate_margin` 5 per rad/s of max |ω_meas| over 10 rad/s (RATE monitor 12 rad/s), `est_innov` 30 per rad of real Mahony innovation over 0.25 (monitor 0.35), `est_nis` 0.3 per unit of KF NIS over 8 (monitor 12).
* **Smoothness** (`caps` in the config, `train/ppo.py`): CAPS-style losses on the *deterministic* action in the PPO update — temporal (second difference along the rollout, with the previous-action inputs replaced by the unrolled mean actions) weight 60, spatial (|μ(s) − μ(s+ε)|², ε ~ N(0, 0.05) in normalized obs, position-error and previous-action dims excluded) weight 60, channel weights [1, .25, .25, .25]. A reward penalty on the sampled thrust (`thrust_jerk`) was tried first and dropped: exploration noise (std 0.05–0.1) swamps it. A temporal term on the raw rollout obs (sampled previous action) collapsed training; spatial noise on the position obs cost hover precision (stage-1 policy: SITL hover 5.5 cm median).
* Warm start from v2 (`runs/v2g/last.pt`), driven by `python -m train.v3_driver` (resumable segments, checkpoint every 25 iterations ≈ 1 min): stage v3s1 400 iterations (std reset to 0.1, spatial term including position dims), stage v3s2 1200 iterations (position dims excluded) = 1600 PPO iterations ≈ 210 M env steps, 62 min on 4 cores.

## Evaluation in the training env (`python -m train.eval train/policies/hover_ctbr_v3.nfpol.json --compare v2=train/policies/hover_ctbr_v2.nfpol.json --sensing fw3`; fw3 mirrors the current firmware incl. the 0.3 m/s² reference deadband, which v3 was not trained with)
Definitions as in hover_ctbr_v2.md. New columns: **sup%** = runs meeting any envelope / estimator monitor condition (TILT, TILT_PREDICTED, RATE on the measured rate, ESTIMATOR on the Mahony innovation / KF NIS); **thr/PID** = HF (> 50 Hz) thrust-command RMS relative to the PID baseline.

| scenario | succ% v3 | succ% v2 | succ% PID | v3 err_med m | v2 err_med m | PID err_med m | v3 worst/p95 tilt° | v2 worst/p95 tilt° | PID worst/p95 tilt° | v3 settle s | v2 settle s | PID settle s | v3 trip% | v2 trip% | PID trip% | v3 sup% | v2 sup% | PID sup% | v3 HF rate/thr | v2 HF rate/thr | PID HF rate/thr | v3 thr/PID | v2 thr/PID |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hover_1m/nominal | 100.0 | 98.6 | 99.8 | 0.033 | 0.031 | 0.019 | 38/33 | 39/35 | 38/36 | 0.92 | 0.87 | 0.91 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0014/0.0012 | 0.0011/0.0004 | 0.0007/0.0003 | 4.45 | 1.46 |
| hover_1m/randomized | 98.8 | 93.6 | 98.4 | 0.033 | 0.044 | 0.027 | 39/33 | 44/37 | 41/37 | 0.95 | 0.89 | 0.99 | 0.0 | 2.1 | 0.0 | 0.0 | 2.5 | 1.8 | 0.0019/0.0017 | 0.0042/0.0042 | 0.0028/0.0014 | 1.26 | 3.08 |
| tilt_60/nominal | 92.8 | 95.3 | 99.6 | 0.059 | 0.050 | 0.020 | 60/60 | 60/60 | 60/60 | 0.78 | 0.63 | 0.77 | 0.0 | 0.0 | 0.0 | 2.9 | 10.2 | 0.0 | 0.0014/0.0010 | 0.0010/0.0003 | 0.0008/0.0001 | 8.78 | 2.42 |
| tilt_60/randomized | 86.5 | 86.3 | 97.5 | 0.061 | 0.062 | 0.031 | 60/60 | 60/60 | 60/60 | 0.81 | 0.73 | 0.86 | 0.0 | 0.6 | 0.0 | 2.3 | 14.6 | 0.6 | 0.0019/0.0017 | 0.0042/0.0040 | 0.0030/0.0013 | 1.33 | 3.22 |
| throw/nominal | 98.4 | 97.9 | 98.4 | 0.039 | 0.035 | 0.028 | 49/41 | 57/43 | 52/42 | 0.62 | 0.60 | 0.58 | 3.5 | 6.2 | 3.3 | 3.5 | 6.8 | 4.1 | 0.0015/0.0012 | 0.0011/0.0005 | 0.0010/0.0002 | 7.53 | 2.94 |
| throw/randomized | 97.9 | 90.8 | 94.7 | 0.040 | 0.046 | 0.033 | 47/41 | 61/45 | 53/44 | 0.69 | 0.69 | 0.71 | 3.5 | 7.4 | 3.5 | 3.5 | 8.8 | 5.9 | 0.0019/0.0018 | 0.0042/0.0041 | 0.0031/0.0013 | 1.36 | 3.14 |
| upside_down/nominal | 68.0 | 61.3 | 71.9 | 0.083 | 0.087 | 0.069 | 180/179 | 180/179 | 180/179 | 1.29 | 1.16 | 1.31 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0016/0.0014 | 0.0012/0.0011 | 0.0011/0.0006 | 2.46 | 1.82 |
| upside_down/randomized | 57.0 | 54.1 | 70.7 | 0.092 | 0.094 | 0.072 | 180/179 | 180/179 | 180/179 | 1.34 | 1.20 | 1.53 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0021/0.0021 | 0.0041/0.0043 | 0.0032/0.0014 | 1.44 | 3.03 |

Randomization extremes (one parameter at min/max, rest nominal, no other noise): success %

| param=extreme | v3 tilt_60 | v2 tilt_60 | PID tilt_60 | v3 throw | v2 throw | PID throw |
|---|---|---|---|---|---|---|
| mass=min | 97 | 94 | 98 | 98 | 98 | 100 |
| mass=max | 92 | 92 | 100 | 100 | 100 | 98 |
| inertia=min | 88 | 95 | 100 | 100 | 100 | 100 |
| inertia=max | 97 | 97 | 98 | 100 | 98 | 97 |
| thrust_max=min | 100 | 100 | 98 | 100 | 97 | 97 |
| thrust_max=max | 84 | 91 | 100 | 97 | 98 | 100 |
| per_motor_thrust_imbalance=min | 94 | 94 | 100 | 98 | 98 | 100 |
| per_motor_thrust_imbalance=max | 97 | 95 | 97 | 98 | 97 | 97 |
| yaw_torque_coeff=min | 97 | 94 | 98 | 100 | 95 | 100 |
| yaw_torque_coeff=max | 92 | 95 | 100 | 98 | 98 | 98 |
| motor_tau=min | 95 | 92 | 100 | 100 | 97 | 100 |
| motor_tau=max | 88 | 91 | 95 | 97 | 92 | 95 |
| lin_drag=min | 97 | 97 | 100 | 98 | 98 | 95 |
| lin_drag=max | 92 | 97 | 98 | 100 | 97 | 100 |
| latency_s=min | 95 | 97 | 98 | 100 | 100 | 100 |
| latency_s=max | 89 | 91 | 100 | 95 | 98 | 94 |
| gyro_noise_std=min | 92 | 94 | 100 | 100 | 97 | 98 |
| gyro_noise_std=max | 94 | 94 | 100 | 97 | 97 | 100 |
| gyro_bias=min | 92 | 97 | 98 | 98 | 97 | 98 |
| gyro_bias=max | 97 | 94 | 100 | 98 | 98 | 98 |
| accel_noise_std=min | 92 | 92 | 98 | 100 | 100 | 100 |
| accel_noise_std=max | 95 | 92 | 100 | 100 | 95 | 100 |
| vel_noise_std=min | 94 | 94 | 100 | 100 | 98 | 100 |
| vel_noise_std=max | 91 | 91 | 100 | 95 | 92 | 98 |
| pos_noise_std=min | 89 | 97 | 100 | 100 | 97 | 100 |
| pos_noise_std=max | 94 | 95 | 100 | 92 | 98 | 100 |

- v3: worst extreme `thrust_max=max/tilt_60` -> 84%
- v2: worst extreme `thrust_max=max/tilt_60` -> 91%
- PID: worst extreme `latency_s=max/throw` -> 94%

### Firmware sensing path without the Mahony estimate (`--sensing fw`, the v2 training env, no extremes)
| scenario | succ% v3 | succ% v2 | succ% PID | v3 err_med m | v2 err_med m | PID err_med m | v3 worst/p95 tilt° | v2 worst/p95 tilt° | PID worst/p95 tilt° | v3 settle s | v2 settle s | PID settle s | v3 trip% | v2 trip% | PID trip% | v3 sup% | v2 sup% | PID sup% | v3 HF rate/thr | v2 HF rate/thr | PID HF rate/thr | v3 thr/PID | v2 thr/PID |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hover_1m/nominal | 100.0 | 100.0 | 100.0 | 0.035 | 0.027 | 0.015 | 37/32 | 38/34 | 38/35 | 0.90 | 0.87 | 0.91 | 0.0 | 0.0 | 0.0 | 0.0 | 0.4 | 0.4 | 0.0014/0.0011 | 0.0010/0.0004 | 0.0007/0.0003 | 4.17 | 1.38 |
| hover_1m/randomized | 99.8 | 97.9 | 99.0 | 0.031 | 0.038 | 0.026 | 35/33 | 48/36 | 38/36 | 0.92 | 0.90 | 0.94 | 0.0 | 1.6 | 0.0 | 0.0 | 1.8 | 0.6 | 0.0019/0.0017 | 0.0039/0.0040 | 0.0026/0.0013 | 1.28 | 3.08 |
| tilt_60/nominal | 91.4 | 98.0 | 99.4 | 0.059 | 0.044 | 0.020 | 60/60 | 60/60 | 60/60 | 0.78 | 0.62 | 0.76 | 0.0 | 0.0 | 0.0 | 2.9 | 7.6 | 0.0 | 0.0013/0.0011 | 0.0009/0.0002 | 0.0008/0.0001 | 8.99 | 1.86 |
| tilt_60/randomized | 90.0 | 89.8 | 97.7 | 0.059 | 0.053 | 0.028 | 60/60 | 60/60 | 60/60 | 0.78 | 0.67 | 0.83 | 0.0 | 1.6 | 0.0 | 1.8 | 11.5 | 0.0 | 0.0019/0.0017 | 0.0041/0.0040 | 0.0028/0.0012 | 1.35 | 3.24 |
| throw/nominal | 98.6 | 99.0 | 97.7 | 0.038 | 0.035 | 0.034 | 49/40 | 55/42 | 47/41 | 0.61 | 0.60 | 0.56 | 3.5 | 4.9 | 3.3 | 3.7 | 5.9 | 3.7 | 0.0015/0.0012 | 0.0010/0.0003 | 0.0009/0.0001 | 8.00 | 2.34 |
| throw/randomized | 98.0 | 94.9 | 96.7 | 0.041 | 0.045 | 0.040 | 47/41 | 54/43 | 48/43 | 0.69 | 0.70 | 0.68 | 3.1 | 7.4 | 2.7 | 3.1 | 7.6 | 3.7 | 0.0019/0.0016 | 0.0041/0.0042 | 0.0030/0.0013 | 1.22 | 3.13 |
| upside_down/nominal | 86.5 | 69.9 | 73.2 | 0.057 | 0.067 | 0.058 | 180/179 | 180/179 | 180/179 | 1.34 | 1.29 | 1.42 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0016/0.0016 | 0.0011/0.0010 | 0.0012/0.0007 | 2.19 | 1.32 |
| upside_down/randomized | 81.2 | 64.8 | 73.2 | 0.062 | 0.074 | 0.064 | 180/179 | 180/179 | 180/179 | 1.38 | 1.31 | 1.61 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0021/0.0021 | 0.0041/0.0041 | 0.0033/0.0014 | 1.43 | 2.82 |


## SITL (real firmware, `python3 -m pytest test/sitl -k v3`, docs/sitl-report.md)
Hover: 20 randomized vehicles (seed 11), POLICY 10 s window, true 3D RMS error. Disturbances: 20 randomized vehicles (seed 21), definitions in hover_ctbr_v2.md. All on the current firmware (v2 numbers therefore differ slightly from hover_ctbr_v2.md: the KF-acceleration reference removed v2's ESTIMATOR trip on pushes).

| controller | hover RMS min / med / max (cm) | hover trips | < 5 cm | tilt pass (trips) | throw pass (trips) | HF thrust vs fallback PID |
|---|---|---|---|---|---|---|
| fallback (cascaded PID), FALLBACK_HOLD, seed 1 | 0.6 / 2.2 / 4.2 | 0 | 20/20 | 20/20 | 20/20 | 1.00 |
| hover_ctbr_v2 | 2.1 / 3.5 / 4.8 | 0 | 20/20 | 16/20 (ESTIMATOR, RATE, TILT_PREDICTED) | 19/20 (TILT_PREDICTED) | 3.92 |
| hover_ctbr_v2 int8 | 2.5 / 3.7 / 5.2 | 0 | 19/20 | 17/20 | 19/20 | - |
| **hover_ctbr_v3** | **1.4 / 2.8 / 7.7** | **0** | 18/20 | **19/20 (0 trips; 1 not back within 15 cm in the last 2 s)** | **20/20 (0)** | **1.64** |
| hover_ctbr_v3 int8 | 1.7 / 3.2 / 7.1 | 0 | 18/20 | 20/20 (0) | 20/20 (0) | 4.40 |

Stress throw (|dω| ≤ 10 rad/s): v3 20/20, v2 18/20. Recovery time med / max: tilt 0.38 / 0.55 s (v2 0.22 / 5.64), throw 0.50 / 1.35 s (v2 0.67 / 5.98). With the old gravity-only Mahony reference v3 trips ESTIMATOR on 5/20 pushes (sitl-report (b2)): it relies on the new firmware reference.

## int8 (`python3 tools/pipeline.py train/policies/hover_ctbr_v3.nfpol.json --calib <51 200 policy-rollout obs, fw3 sensing, all scenarios>`)
PASSED: C++ int8 inference bit-exact against the golden model (all vectors, int8 out max |d| 1.2e-7 at the float tail), float path within 6e-7. int8 vs float action error on the calibration set: mean 0.0030, p99 0.010, max 0.021.

## Acceptance (brief) — met / not met
* SITL hover median ≤ 3.5 cm, no trips: **met** (2.8 cm, 0 trips; int8 3.2 cm, 0 trips). The tail is wider than v2 (max 7.7 vs 4.8 cm, 18/20 under 5 cm vs 20/20).
* Tilt and push recoveries ≥ 19/20 staying in POLICY: **met** — 19/20 and 20/20, **no supervisor trips at all** (v2 on the same firmware: 16 and 19). The one tilt "fail" stays in POLICY but is 15+ cm off in the last 2 s.
* Thrust noise ≤ 1.5× PID under sensor noise: **met in the train eval** (randomized scenarios 1.26–1.44× vs v2 3.0–3.2×); **not met in SITL** (1.64× vs v2 3.92×); the int8 build is 4.4× (quantization steps on the thrust output). Without sensor noise (nominal rows) v3 has a small ripple that v2 does not (HF 0.0011 vs 0.0004).
* Training-env success no worse than v2: **not fully met** — better or equal on every randomized scenario and upside-down (hover 98.8 vs 93.6 %, throw 97.9 vs 90.8 %), but **worse on tilt_60/nominal** (92.8 vs 95.3 %; fw env 91.4 vs 98.0 %: slower settling, 0.78 vs 0.63 s, misses the 10 cm tail bar) and on the worst extreme (84 % vs 91 %, thrust_max=max / tilt_60).
* int8 pipeline bit-exact: **met**.
