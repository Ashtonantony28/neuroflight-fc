# hover_ctbr_v1 — CTBR hover/recovery policy (stage A)

`hover_ctbr_v1.nfpol.json`: actor 19→64→64→4, tanh everywhere (final tanh = action squashing), obs normalization frozen in `obs_mean`/`obs_std` (INTERFACES §4–6). Deterministic (mean action).

## Provenance
PPO, asymmetric actor-critic (critic gets true state + randomized params via `train/priv_env.py`), 2048 envs × 64 steps per iteration, domain randomization on (`config/randomization.yaml`), seed 0. ~235M env steps, ~60 min wall-clock on 4 CPU cores:

| stage | config | init | iterations | notes |
|---|---|---|---|---|
| 1 | `train/config/hover_ctbr.yaml` | scratch | 650 (of 1500, stopped) | curriculum levels 0→4 reached at it 86/126/166/376 |
| 2 | `hover_ctbr_ft.yaml`, no `extra_reward` | stage 1 | 150 | pos 3, vel 0.5, smooth 1.0, level 4 fixed |
| 3 | `hover_ctbr_ft.yaml`, `pos_exp σ=0.25` | stage 2 | 500 | |
| 4 | `hover_ctbr_ft.yaml`, `pos_exp σ=0.15` (as committed) | stage 3 | 500 | → this policy |

Reproduce: `python -m train.ppo --name s1 --iterations 650`, then `python -m train.ppo --config train/config/hover_ctbr_ft.yaml --name s2 --init-from runs/s1/last.pt` with the overrides above, then `python -m train.export runs/<last>/last.pt`.

## Evaluation (`python -m train.eval train/policies/hover_ctbr_v1.nfpol.json`)
512 envs per scenario/mode, 64 per extreme; 5 s; target (0,0,2). Success = no crash/OOB, mean position error over the last 0.5 s < 0.10 m, final tilt < 10°. RL and PID see identical vehicles and initial states (same seed). HF = RMS of action content above 50 Hz (rate channels / thrust channel).

| scenario | succ% RL | succ% PID | RL err_med m | PID err_med m | RL worst/p95 tilt° | PID worst/p95 tilt° | RL settle s | PID settle s | RL HF rate/thr | PID HF rate/thr |
|---|---|---|---|---|---|---|---|---|---|---|
| hover_1m/nominal | 100.0 | 100.0 | 0.066 | 0.005 | 60/51 | 36/35 | 0.76 | 0.87 | 0.0007/0.0002 | 0.0007/0.0003 |
| hover_1m/randomized | 100.0 | 100.0 | 0.059 | 0.010 | 61/51 | 40/36 | 0.77 | 0.91 | 0.0211/0.0261 | 0.0168/0.0096 |
| tilt_60/nominal | 100.0 | 100.0 | 0.066 | 0.017 | 60/60 | 60/60 | 0.41 | 0.66 | 0.0005/0.0001 | 0.0007/0.0001 |
| tilt_60/randomized | 100.0 | 100.0 | 0.056 | 0.022 | 60/60 | 60/60 | 0.44 | 0.71 | 0.0208/0.0253 | 0.0170/0.0094 |
| throw/nominal | 100.0 | 100.0 | 0.066 | 0.017 | 79/61 | 46/41 | 0.45 | 0.55 | 0.0006/0.0001 | 0.0009/0.0002 |
| throw/randomized | 100.0 | 100.0 | 0.056 | 0.023 | 80/61 | 48/43 | 0.51 | 0.65 | 0.0216/0.0259 | 0.0173/0.0096 |
| upside_down/nominal | 96.1 | 100.0 | 0.066 | 0.044 | 180/179 | 180/179 | 1.05 | 1.23 | 0.0014/0.0008 | 0.0012/0.0007 |
| upside_down/randomized | 95.7 | 100.0 | 0.059 | 0.046 | 180/180 | 180/179 | 1.13 | 1.50 | 0.0210/0.0259 | 0.0174/0.0096 |

Randomization extremes (one parameter at min/max, rest nominal, no other noise): success %

| param=extreme | RL tilt_60 | PID tilt_60 | RL throw | PID throw |
|---|---|---|---|---|
| mass=min | 100 | 100 | 100 | 100 |
| mass=max | 100 | 100 | 100 | 100 |
| inertia=min | 100 | 100 | 100 | 100 |
| inertia=max | 100 | 100 | 100 | 100 |
| thrust_max=min | 100 | 100 | 100 | 100 |
| thrust_max=max | 100 | 100 | 100 | 100 |
| per_motor_thrust_imbalance=min | 100 | 100 | 100 | 100 |
| per_motor_thrust_imbalance=max | 100 | 100 | 100 | 98 |
| yaw_torque_coeff=min | 100 | 100 | 100 | 100 |
| yaw_torque_coeff=max | 100 | 100 | 100 | 100 |
| motor_tau=min | 100 | 100 | 100 | 100 |
| motor_tau=max | 100 | 100 | 100 | 100 |
| lin_drag=min | 100 | 100 | 100 | 100 |
| lin_drag=max | 100 | 100 | 100 | 100 |
| latency_s=min | 100 | 100 | 100 | 100 |
| latency_s=max | 100 | 100 | 100 | 100 |
| gyro_noise_std=min | 100 | 100 | 100 | 100 |
| gyro_noise_std=max | 100 | 100 | 100 | 100 |
| gyro_bias=min | 100 | 100 | 100 | 100 |
| gyro_bias=max | 100 | 100 | 100 | 100 |
| accel_noise_std=min | 100 | 100 | 100 | 100 |
| accel_noise_std=max | 100 | 100 | 100 | 100 |
| vel_noise_std=min | 100 | 100 | 100 | 100 |
| vel_noise_std=max | 100 | 100 | 100 | 100 |
| pos_noise_std=min | 100 | 100 | 100 | 100 |
| pos_noise_std=max | 100 | 100 | 100 | 100 |

- RL: worst extreme `mass=min/tilt_60` -> 100%
- PID: worst extreme `per_motor_thrust_imbalance=max/throw` -> 98%

## Summary vs PID
- Success: matches PID (100 %) on hover, 60° tilt and throws, nominal and randomized, and on every single-parameter extreme; upside-down 96 % vs 100 % (all ~4 % failures are crash/out-of-bounds terminations).
- Faster: settle time shorter than PID in every scenario (e.g. tilt_60 0.44 s vs 0.71 s randomized).
- Worse: steady-state position error ~0.06 m vs 0.01–0.02 m (no integrator; the memoryless policy trims mass/thrust errors with a small offset), bigger tilt excursions (hover from 1 m: 61° vs 40°; throw: 80° vs 48°).
- Smoothness: same as PID with nominal (no noise) parameters; with sensor noise rate-channel HF is 1.25× PID and thrust HF 2.7× PID.

## int8 pipeline (`python3 tools/pipeline.py train/policies/hover_ctbr_v1.nfpol.json --calib <51 200 policy-rollout obs>`)
PASSED: C++ int8 inference bit-exact against the golden model (0 mismatches on 12 000 vectors), float path within 6.5e-7. int8 vs float action error on the calibration set: mean 0.0076, p99 0.031, max 0.049. CRC f32 0x0B7F8E3D, i8 0xC95185A4.
