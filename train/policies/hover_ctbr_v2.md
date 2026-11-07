# hover_ctbr_v2 — CTBR hover/recovery policy trained on the firmware sensing path

`hover_ctbr_v2.nfpol.json` (float32, CRC f32 0xAE325CF8) and `hover_ctbr_v2.q.nfpol.json` (int8 block, CRC i8 0x6DCC7235): same network and interfaces as v1 (19→64→64→4 tanh, INTERFACES §4–7), so the firmware runs it unchanged. Deterministic (mean action).

## Why v2: what was wrong with v1 in SITL
v1 scored 100 % in its training env but in SITL (real firmware) hovered at 4/18/14/5 cm on 4 randomized vehicles and tripped FAULT_TILT_PREDICTED on one. `fw/sitl/diagnose_gap.py` replaces the policy / rate-PID inputs inside the firmware with what the v1 training env fed (`nf::InputOverride`, diagnostics only) on 20 randomized vehicles (seed 11):

| inputs (v1) | RMS err min/med/max (cm) | med offset (cm) | max tilt (deg) | trips |
|---|---|---|---|---|
| firmware as is | 4.8 / 8.5 / 24.6 | 6.2 | 29 | 2 (TILT_PREDICTED) |
| all training-env inputs | 3.7 / 5.2 / 7.6 | 4.6 | 4 | 0 |
| firmware + true velocity only (V) | 2.1 / 5.8 / 9.6 | 5.4 | 4 | 0 |
| all training inputs except velocity | 3.1 / 5.4 / 7.7 | 4.0 | 22 | 0 |
| firmware + true attitude only (Q) | 4.9 / 10.1 / 33.1 | 6.6 | 37 | 1 |
| firmware + raw gyro into rate PID (PID, FOLLOWUPS 2) | 3.5 / 7.5 / 20.2 | 5.8 | 32 | 1 |
| firmware + raw gyro as obs (W) | 4.5 / 6.7 / 20.2 | 5.2 | 27 | 2 |
| firmware + true position (P) | 3.2 / 6.9 / 14.9 | 4.9 | 40 | 2 |

Attribution: (a) the **flow/ToF-KF velocity** is the dominant gap — swapping it for truth alone removes every trip and the large tilts (29° → 4°); its error is coloured (autocorrelation 0.66 at 10 ms, 0.41 at 20 ms, ~0 at 50 ms; rms ≈ 0.6 × vel_noise_std), i.e. ~5× the low-frequency power of the white per-step noise v1 saw, and the flow de-rotation couples it to the vehicle's own rates. Attitude (Mahony 0.1° rms in hover), the gyro path (b) and position matter little in hover. (e) The remaining ~5 cm with all training inputs is v1's own steady-state offset (6.5 cm on the nominal vehicle, +3/+3/−3 cm in x/y/z). (c) The D-kick only affected training episode starts. (d) The tilt trips come from (a) driving v1's aggressive attitude loop.

## What changed in training
* `sim/fw_estimator.py`: torch ports of the firmware gyro chain (150 Hz biquad + RPM notches, minus bias estimate; parity ≤ 1e-5 with the C++) feeding **both the rate PID and obs 12-14** (FOLLOWUPS 2), and of `nf::TranslationKf` fed with SITL-identical range/flow (`EnvConfig(estimator=True)`); 1° per-episode attitude misalignment; rate PID seeded on reset (FOLLOWUPS 1).
* Reward (`train/config/hover_ctbr_v2.yaml`): supervisor-envelope margins `tilt_pred` (tilt + 0.1 s·tilt-rate over 50°, faded in at 2.5–4 rad/s) and `tilt_excess` (> 55°), `thrust_smooth`, a sharp position bonus `pos_exp2` (σ 4 cm) against v1's offset, `est_innov` (1 Hz low-passed tilt over 0.25 rad while |ω| < 2 rad/s, a proxy for the Mahony innovation that the ESTIMATOR monitor watches); hover heights 0.8–2.2 m.
* Warm start from v1 (`runs/v3_ft3/last.pt`), 1700 PPO iterations ≈ 223 M env steps (2048 envs × 64 steps), ≈ 45 min of 4-core training at the uncontended 1.5 s/iteration (wall time was longer: container restarts, resumed from checkpoints):

| stage | run | iterations | change |
|---|---|---|---|
| 1 | v2 + v2b | 150 + 200 | fw sensing on, std reset to 0.1, `tilt_pred` 30 |
| 2 | v2c + v2f | 450 + 500 | `tilt_pred` 100 |
| 3 | v2g | 400 | `est_innov` 15 (calm-gated) → this policy |

Discarded: an ungated `est_innov` proxy (the policy learned to spin to freeze a gated one, and an ungated low-passed tilt penalty degraded hover to 9–10 cm). Reproduce: `python -m train.ppo --config train/config/hover_ctbr_v2.yaml --name v2 --init-from runs/v3_ft3/last.pt` with the stage overrides above, then `python -m train.export runs/<last>/last.pt`.

## Evaluation in the training env (`python -m train.eval train/policies/hover_ctbr_v2.nfpol.json --compare v1=train/policies/hover_ctbr_v1.nfpol.json --sensing v1|fw`)
RL = v2. 512 envs per scenario/mode, 64 per extreme, 5 s, target (0,0,2), same seeds for all controllers (definitions as in hover_ctbr_v1.md). New column **trip%**: fraction of runs that meet the firmware's TILT_PREDICTED (≥ 4 rad/s, tilt + 0.1 s·rate > 60° for 20 ms) or TILT (> 60° for 50 ms) condition on the true state, from t = 0 (upside-down starts always count).
Note the eval now resets the rate PID with the first measurement (FOLLOWUPS 1), so v1/PID numbers differ slightly from hover_ctbr_v1.md.

### v1 training env (`--sensing v1`: raw gyro, true attitude, truth + white-noise position/velocity)
| scenario | succ% RL | succ% v1 | succ% PID | RL err_med m | v1 err_med m | PID err_med m | RL worst/p95 tilt° | v1 worst/p95 tilt° | PID worst/p95 tilt° | RL settle s | v1 settle s | PID settle s | RL trip% | v1 trip% | PID trip% | RL HF rate/thr | v1 HF rate/thr | PID HF rate/thr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hover_1m/nominal | 100.0 | 100.0 | 100.0 | 0.022 | 0.066 | 0.005 | 36/33 | 60/51 | 36/35 | 0.84 | 0.76 | 0.87 | 0.0 | 45.1 | 0.0 | 0.0005/0.0002 | 0.0007/0.0002 | 0.0007/0.0003 |
| hover_1m/randomized | 100.0 | 100.0 | 100.0 | 0.026 | 0.059 | 0.010 | 39/35 | 61/51 | 40/36 | 0.84 | 0.77 | 0.91 | 0.0 | 41.8 | 0.0 | 0.0278/0.0299 | 0.0210/0.0261 | 0.0168/0.0096 |
| tilt_60/nominal | 100.0 | 100.0 | 100.0 | 0.022 | 0.066 | 0.017 | 60/60 | 60/60 | 60/60 | 0.45 | 0.41 | 0.66 | 0.0 | 5.3 | 0.0 | 0.0005/0.0000 | 0.0005/0.0001 | 0.0007/0.0001 |
| tilt_60/randomized | 100.0 | 100.0 | 100.0 | 0.027 | 0.056 | 0.022 | 60/60 | 60/60 | 60/60 | 0.48 | 0.44 | 0.71 | 0.0 | 7.4 | 0.0 | 0.0276/0.0289 | 0.0208/0.0253 | 0.0170/0.0094 |
| throw/nominal | 100.0 | 100.0 | 100.0 | 0.022 | 0.066 | 0.018 | 54/43 | 80/61 | 46/41 | 0.57 | 0.45 | 0.56 | 4.3 | 25.4 | 3.3 | 0.0006/0.0001 | 0.0006/0.0001 | 0.0009/0.0002 |
| throw/randomized | 100.0 | 100.0 | 100.0 | 0.026 | 0.057 | 0.024 | 52/44 | 80/61 | 48/43 | 0.65 | 0.52 | 0.66 | 5.1 | 29.9 | 2.3 | 0.0282/0.0296 | 0.0216/0.0259 | 0.0173/0.0096 |
| upside_down/nominal | 98.6 | 96.1 | 100.0 | 0.022 | 0.066 | 0.044 | 180/179 | 180/179 | 180/179 | 1.14 | 1.05 | 1.23 | 100.0 | 100.0 | 100.0 | 0.0007/0.0010 | 0.0014/0.0008 | 0.0012/0.0007 |
| upside_down/randomized | 98.0 | 95.9 | 100.0 | 0.027 | 0.060 | 0.046 | 180/179 | 180/180 | 180/179 | 1.19 | 1.13 | 1.50 | 100.0 | 100.0 | 100.0 | 0.0276/0.0294 | 0.0211/0.0259 | 0.0174/0.0096 |

Randomization extremes (one parameter at min/max, rest nominal, no other noise): success %

| param=extreme | RL tilt_60 | v1 tilt_60 | PID tilt_60 | RL throw | v1 throw | PID throw |
|---|---|---|---|---|---|---|
| mass=min | 100 | 100 | 100 | 100 | 100 | 100 |
| mass=max | 100 | 100 | 100 | 100 | 100 | 100 |
| inertia=min | 100 | 100 | 100 | 100 | 100 | 100 |
| inertia=max | 100 | 100 | 100 | 100 | 100 | 100 |
| thrust_max=min | 100 | 100 | 100 | 100 | 100 | 100 |
| thrust_max=max | 100 | 100 | 100 | 100 | 100 | 100 |
| per_motor_thrust_imbalance=min | 100 | 100 | 100 | 100 | 100 | 100 |
| per_motor_thrust_imbalance=max | 100 | 100 | 100 | 100 | 100 | 98 |
| yaw_torque_coeff=min | 100 | 100 | 100 | 100 | 100 | 100 |
| yaw_torque_coeff=max | 100 | 100 | 100 | 100 | 100 | 100 |
| motor_tau=min | 100 | 100 | 100 | 100 | 100 | 100 |
| motor_tau=max | 100 | 100 | 100 | 100 | 100 | 100 |
| lin_drag=min | 100 | 100 | 100 | 100 | 100 | 100 |
| lin_drag=max | 100 | 100 | 100 | 100 | 100 | 100 |
| latency_s=min | 100 | 100 | 100 | 100 | 100 | 100 |
| latency_s=max | 100 | 100 | 100 | 100 | 100 | 100 |
| gyro_noise_std=min | 100 | 100 | 100 | 100 | 100 | 100 |
| gyro_noise_std=max | 100 | 100 | 100 | 100 | 100 | 100 |
| gyro_bias=min | 100 | 100 | 100 | 100 | 100 | 100 |
| gyro_bias=max | 100 | 100 | 100 | 100 | 100 | 100 |
| accel_noise_std=min | 100 | 100 | 100 | 100 | 100 | 100 |
| accel_noise_std=max | 100 | 100 | 100 | 100 | 100 | 100 |
| vel_noise_std=min | 100 | 100 | 100 | 100 | 100 | 100 |
| vel_noise_std=max | 100 | 100 | 100 | 100 | 100 | 100 |
| pos_noise_std=min | 100 | 100 | 100 | 100 | 100 | 100 |
| pos_noise_std=max | 100 | 100 | 100 | 100 | 100 | 100 |

- RL: worst extreme `mass=min/tilt_60` -> 100%
- v1: worst extreme `mass=min/tilt_60` -> 100%
- PID: worst extreme `per_motor_thrust_imbalance=max/throw` -> 98%

### Firmware sensing path (`--sensing fw`: filtered bias-corrected gyro, flow/ToF KF)
| scenario | succ% RL | succ% v1 | succ% PID | RL err_med m | v1 err_med m | PID err_med m | RL worst/p95 tilt° | v1 worst/p95 tilt° | PID worst/p95 tilt° | RL settle s | v1 settle s | PID settle s | RL trip% | v1 trip% | PID trip% | RL HF rate/thr | v1 HF rate/thr | PID HF rate/thr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hover_1m/nominal | 99.6 | 19.3 | 99.8 | 0.027 | 0.166 | 0.015 | 38/34 | 65/56 | 37/35 | 0.87 | 2.07 | 0.91 | 0.0 | 68.4 | 0.0 | 0.0010/0.0004 | 0.0115/0.0059 | 0.0007/0.0003 |
| hover_1m/randomized | 97.1 | 14.8 | 98.8 | 0.040 | 0.214 | 0.026 | 42/37 | 67/56 | 39/36 | 0.92 | 1.69 | 0.95 | 1.6 | 66.4 | 0.0 | 0.0041/0.0042 | 0.0118/0.0069 | 0.0027/0.0014 |
| tilt_60/nominal | 98.0 | 19.7 | 99.6 | 0.044 | 0.190 | 0.020 | 60/60 | 60/60 | 60/60 | 0.62 | 1.82 | 0.76 | 0.0 | 10.9 | 0.0 | 0.0009/0.0002 | 0.0116/0.0059 | 0.0008/0.0001 |
| tilt_60/randomized | 91.4 | 17.6 | 97.5 | 0.054 | 0.197 | 0.029 | 60/60 | 60/60 | 60/60 | 0.72 | 1.50 | 0.85 | 0.8 | 20.5 | 0.0 | 0.0041/0.0040 | 0.0117/0.0066 | 0.0029/0.0013 |
| throw/nominal | 98.6 | 23.8 | 98.4 | 0.034 | 0.177 | 0.034 | 55/42 | 78/64 | 47/41 | 0.60 | 1.83 | 0.56 | 4.9 | 38.7 | 3.3 | 0.0010/0.0003 | 0.0115/0.0058 | 0.0009/0.0001 |
| throw/randomized | 94.5 | 12.7 | 94.7 | 0.045 | 0.225 | 0.038 | 51/43 | 80/65 | 49/43 | 0.68 | 1.19 | 0.68 | 6.4 | 49.6 | 2.7 | 0.0041/0.0041 | 0.0119/0.0068 | 0.0031/0.0013 |
| upside_down/nominal | 70.5 | 25.0 | 73.2 | 0.067 | 0.168 | 0.059 | 180/179 | 180/179 | 180/179 | 1.29 | 2.31 | 1.42 | 100.0 | 100.0 | 100.0 | 0.0011/0.0010 | 0.0111/0.0056 | 0.0012/0.0007 |
| upside_down/randomized | 61.7 | 12.7 | 73.4 | 0.078 | 0.204 | 0.067 | 180/179 | 180/179 | 180/179 | 1.31 | 1.80 | 1.56 | 100.0 | 100.0 | 100.0 | 0.0040/0.0043 | 0.0113/0.0067 | 0.0032/0.0015 |

Randomization extremes (one parameter at min/max, rest nominal, no other noise): success %

| param=extreme | RL tilt_60 | v1 tilt_60 | PID tilt_60 | RL throw | v1 throw | PID throw |
|---|---|---|---|---|---|---|
| mass=min | 95 | 25 | 100 | 98 | 22 | 98 |
| mass=max | 100 | 17 | 100 | 100 | 9 | 97 |
| inertia=min | 95 | 33 | 100 | 98 | 17 | 98 |
| inertia=max | 95 | 28 | 98 | 97 | 14 | 97 |
| thrust_max=min | 100 | 19 | 100 | 100 | 20 | 95 |
| thrust_max=max | 94 | 30 | 98 | 100 | 27 | 100 |
| per_motor_thrust_imbalance=min | 95 | 27 | 100 | 100 | 16 | 98 |
| per_motor_thrust_imbalance=max | 95 | 34 | 97 | 98 | 23 | 94 |
| yaw_torque_coeff=min | 98 | 12 | 98 | 98 | 16 | 97 |
| yaw_torque_coeff=max | 97 | 25 | 100 | 98 | 12 | 97 |
| motor_tau=min | 94 | 31 | 100 | 100 | 28 | 98 |
| motor_tau=max | 97 | 3 | 97 | 94 | 0 | 89 |
| lin_drag=min | 100 | 16 | 98 | 98 | 23 | 97 |
| lin_drag=max | 98 | 27 | 100 | 100 | 25 | 100 |
| latency_s=min | 97 | 39 | 100 | 98 | 27 | 100 |
| latency_s=max | 89 | 6 | 100 | 98 | 3 | 98 |
| gyro_noise_std=min | 97 | 30 | 100 | 95 | 19 | 98 |
| gyro_noise_std=max | 95 | 14 | 100 | 100 | 20 | 100 |
| gyro_bias=min | 98 | 19 | 100 | 100 | 20 | 97 |
| gyro_bias=max | 98 | 22 | 100 | 100 | 19 | 97 |
| accel_noise_std=min | 95 | 22 | 100 | 98 | 11 | 97 |
| accel_noise_std=max | 97 | 22 | 100 | 100 | 19 | 100 |
| vel_noise_std=min | 97 | 19 | 100 | 97 | 14 | 98 |
| vel_noise_std=max | 92 | 20 | 100 | 92 | 33 | 94 |
| pos_noise_std=min | 95 | 22 | 100 | 97 | 19 | 100 |
| pos_noise_std=max | 100 | 30 | 100 | 97 | 19 | 98 |

- RL: worst extreme `latency_s=max/tilt_60` -> 89%
- v1: worst extreme `motor_tau=max/throw` -> 0%
- PID: worst extreme `motor_tau=max/throw` -> 89%

Upside-down success is low for every controller with `--sensing fw` because ToF/flow are gated off beyond 0.7 rad tilt and the KF dead-reckons through the flip; the success test (< 10 cm on the *true* position) then fails on estimator drift (PID 73 %, v2 62–71 %).

## SITL (real firmware, `python3 -m pytest test/sitl -k v2`, fw/sitl/report.py)
Hover: fallback takeoff, POLICY at 6 s, 10 s window, true 3D RMS error; pass = < 5 cm, no fault, stays in POLICY. Randomized x20 = seed 11.

| scenario | RMS err (cm) min / med / max | max err (cm) | worst z RMS (cm) | worst xy drift (cm) | faults | pass |
|---|---|---|---|---|---|---|
| trained hover_ctbr_v1 hover, nominal | 5.3 | 7.3 | 3.4 | 5.3 | - | 0/1 |
| trained hover_ctbr_v1 hover, randomized x20 | 4.8 / 8.5 / 24.6 | 32.8 | 8.3 | 32.8 | TILT_PREDICTED | 3/20 |
| trained hover_ctbr_v2 hover, nominal | 2.4 | 4.7 | 1.3 | 1.9 | - | 1/1 |
| trained hover_ctbr_v2 hover, randomized x20 | 2.1 / 3.4 / 4.8 | 8.1 | 3.4 | 4.9 | - | 20/20 |
| hover_ctbr_v2 int8 (nf::Int8Model) hover, randomized x20 | 2.4 / 3.8 / 5.2 | 9.6 | 3.4 | 4.9 | - | 19/20 |

Disturbances (definitions in docs/sitl-report.md (b)): **tilt** = hand-tilted to 59.8° at 2.9 rad/s and released at rest; **throw** = 20 ms push |dv| ≤ 3 m/s, |dω| ≤ 5 rad/s. 20 randomized vehicles, seed 21. For reference the firmware fallback (FALLBACK_HOLD) on the same pushes: 19/20 (1 ESTIMATOR), on the same tilts 20/20.

| policy | disturbance | pass | trips | max tilt med / max (deg) | recovery med / max (s) | peak err med (m) | RMS last 3 s med (cm) |
|---|---|---|---|---|---|---|---|
| hover_ctbr_v1 | tilt | 10/20 | 2 (RATE,TILT) | 61 / 66 | 0.56 / 5.64 | 0.19 | 8.7 |
| hover_ctbr_v1 | throw | 8/20 | 10 (ESTIMATOR,TILT_PREDICTED) | 39 / 54 | 1.87 / 5.98 | 0.28 | 12.3 |
| hover_ctbr_v2 | tilt | 17/20 | 3 (ESTIMATOR,RATE,TILT_PREDICTED) | 60 / 61 | 0.22 / 5.64 | 0.16 | 5.8 |
| hover_ctbr_v2 | throw | 18/20 | 2 (ESTIMATOR,TILT_PREDICTED) | 25 / 45 | 0.64 / 5.98 | 0.29 | 3.7 |
| hover_ctbr_v2 int8 | tilt | 17/20 | 3 (ESTIMATOR,RATE,TILT_PREDICTED) | 60 / 61 | 0.20 / 5.64 | 0.17 | 5.5 |
| hover_ctbr_v2 int8 | throw | 18/20 | 2 (ESTIMATOR,TILT_PREDICTED) | 26 / 45 | 0.64 / 5.98 | 0.29 | 4.2 |
| hover_ctbr_v2 | throw, |dw| <= 10 rad/s (stress, informational) | 17/20 | 3 (ESTIMATOR,TILT_PREDICTED) | 26 / 44 | 0.66 / 5.98 | 0.29 | 4.1 |

The v2 gap-attribution rows (docs/sitl-report.md) show v2 is now insensitive to every input swap (median 2.7–3.6 cm, no trips): the remaining hover error is mostly flow dead-reckoning drift (fallback PID: 2.2 cm median) plus a 2.5 cm offset.

## int8 (`python3 tools/pipeline.py train/policies/hover_ctbr_v2.nfpol.json --calib <51 200 policy-rollout obs, fw sensing, all scenarios>`)
PASSED: C++ int8 inference bit-exact against the golden model (0 mismatches on 12 000 vectors), float path within 7.6e-7. int8 vs float action error on the calibration set: mean 0.0067, p99 0.026, max 0.052. CRC f32 0xAE325CF8, i8 0x6DCC7235. The SITL bridge now loads int8 models (`nf_sitl_set_policy_i8` → `nf::Int8Model`, bit-checked against `tools/golden.py`), and the int8 v2 is flown in SITL (table above).

## Summary vs v1 and PID
* Firmware sensing: v2 91–99 % success on hover / 60° tilt / throw where v1 gets 13–24 %; on par with PID (95–100 %); trip-condition rate 0–6 % vs v1 11–68 %.
* v1 training env: no worse than v1 on any success rate (100 % everywhere, upside-down 98 % vs 96 %); steady-state error 2.2–2.7 cm vs 5.6–6.6 cm; worst hover tilt 36–39° vs 60–61° (PID 36–40°); trip-condition rate 0–5 % vs 5–45 %. Settle time slightly longer than v1 (0.84 vs 0.77 s hover), still faster than PID on tilt_60.
* Smoothness: rate-channel HF with sensor noise 0.028 (v1 0.021, PID 0.017), thrust HF 0.030 (v1 0.026, PID 0.0096) in the v1 env; in the firmware env (filtered gyro) 0.004 / 0.004 vs PID 0.003 / 0.0014 — thrust still ~3× PID.
* Not reached: the strict SITL bar (all 20 vehicles, no supervisor trip) on tilt and throw recoveries — see the SITL table; remaining trips are RATE (> 12 rad/s while flipping back from 60°) and ESTIMATOR (Mahony accel innovation while braking hard after a 2–3 m/s push; the Mahony attitude error there really is 8–10°, and the firmware fallback trips it on 1/20 of the same pushes). Both monitors were left unchanged.
