# Shared interfaces (all workstreams MUST follow this)

Changes to this file need a note in the PR / report explaining why.

## 1. Frames and units
- SI units everywhere. Time in seconds.
- **World frame W: ENU-style, z up.** Gravity is `[0, 0, -g]`.
- **Body frame B: FLU** (x forward, y left, z up). Positive body rates follow the right-hand rule about B axes.
- Attitude: unit quaternion `q = [w, x, y, z]` rotating B → W (`v_W = R(q) v_B`). Rotation matrix `R` is B → W.
- Thrust acts along +z_B.

## 2. Motor layout (X configuration, viewed from above)
| index | position (x, y) in B | spin (viewed from above) | yaw torque on body |
|---|---|---|---|
| 0 | front-left  (+a, +a) | CW  | +k·T (CCW reaction, +z) |
| 1 | front-right (+a, −a) | CCW | −k·T |
| 2 | rear-right  (−a, −a) | CW  | +k·T |
| 3 | rear-left   (−a, +a) | CCW | −k·T |

`a = arm_xy`, `k = yaw_torque_coeff`. Body torques from per-motor thrusts T_i:
- τx (roll)  = a·(T0 − T1 − T2 + T3)
- τy (pitch) = a·(−T0 − T1 + T2 + T3)
- τz (yaw)   = k·(T0 − T1 + T2 − T3)

(Check: front motors up → nose up → negative pitch about +y in FLU. Yes: τy = a(−T0−T1+...).)

Motor **command** `c_i ∈ [0, 1]`; motor **state** `w_i ∈ [0,1]` follows c with first-order lag (`motor_tau_up` / `motor_tau_down`); thrust `T_i = thrust_max · (A·w² + (1−A)·w)`, `A = thrust_curve_a`.

## 3. Vehicle parameters
`config/vehicle_nominal.yaml` (nominal) and `config/randomization.yaml` (ranges). Python loads with PyYAML; C++ gets a generated header `fw/core/include/nf/vehicle_params.hpp` (tools/gen_params.py) — never hand-copy numbers.

## 4. Policy observation vector (float32, length 19, in this order)
| idx | content |
|---|---|
| 0-2 | position error in W: `p − p_target` (m), each clipped to ±2 |
| 3-5 | linear velocity in W (m/s), clipped to ±5 |
| 6-11 | first two **columns** of R (B→W), column-major: R[:,0], R[:,1] |
| 12-14 | body rates ω_B (rad/s) |
| 15-18 | previous action (the 4 values the policy output last step) |

Observations are normalized inside the exported model (mean/std stored with the weights), so firmware passes the raw vector above.

## 5. Policy action (float32, length 4, each in [−1, 1])
**CTBR mode (stage A):**
- a0 → collective thrust `F = (a0 + 1)/2 · 4·thrust_max` (N)
- a1..a3 → body-rate setpoint `ω_sp = a[1:4] · max_body_rate` (rad/s), tracked by the rate PID.

**Direct mode (stage B, later):** `c_i = (a_i + 1)/2`, clipped to `[idle_command, 1]`.

Policy runs at `control_rate_hz` (500 Hz).

## 6. Exported policy format (`*.nfpol.json` + generated C++)
JSON with: `version`, `mode` ("ctbr"|"direct"), `obs_mean[19]`, `obs_std[19]`, `layers: [{W: [out][in], b: [out], act: "tanh"|"none"}]`, plus an optional `int8` block (section 7). The final layer is followed by `tanh` (so actions are in [−1,1]); encode it as `act: "tanh"` on the last layer.

## 7. int8 inference spec (bit-exact contract between Python golden model and C++)
Symmetric quantization, zero point 0 everywhere.
1. Normalize in float32: `xn = (obs − mean) / std`, then `x_q = clamp(round_half_away_from_zero(xn / s_in), −127, 127)` as int8. (Float32 arithmetic, single-precision, same op order in C++ and numpy.)
2. Hidden layer j (per-output-channel weight scale `s_w[j]`): `acc = Σ_i W_q[j,i]·x_q[i] + b_q[j]` in int32, `b_q[j] = round(b[j] / (s_x · s_w[j]))` (int32).
3. Requantize to the pre-activation int8 scale `s_pre`: `y = clamp(rshift_round((int64)acc · M[j], S[j]), −128, 127)` with `rshift_round(v, s) = (v + (1 << (s−1))) >> s` (arithmetic shift, floor semantics, s ≥ 1). `M[j]` int32, `S[j]` int in [1, 62], chosen so `M·2^−S ≈ s_x·s_w[j]/s_pre`.
4. Activation: `tanh` via a 256-entry int8 LUT indexed by `y + 128`; output scale is `1/127` (LUT value = `round(127·tanh(y·s_pre))`). That output becomes the next layer's `x_q` with `s_x = 1/127`.
5. **Last layer**: int32 accumulator `acc` (bit-exact contract ends here), then float32 `out = tanh((float)acc · s_x · s_w[j])`, compared with tolerance 1e-6.

## 8. Directory ownership
- `sim/` – Python vectorized simulator, PID baseline (torch, CPU OK).
- `train/` – PPO, eval suite, checkpoints (large files go to `runs/`, gitignored).
- `tools/` – export, quantize, golden model, codegen, param header gen, log tools.
- `fw/core/` – portable C++17 firmware core (no HAL, no heap after init, no exceptions/RTTI); host-buildable with CMake; unit tests under `fw/core/tests`.
- `fw/sitl/` – host bridge that lets the sim call the firmware core (shared lib + ctypes).
- `hw/` – board design spec, pin map, BOM.

## 9. CTBR inner loop (must match between sim/ctbr.py and fw/core)
The policy is trained against `sim/ctbr.py`; firmware must reproduce it exactly or transfer suffers.
- Rate PID at `rate_loop_hz` (1 kHz). Gains are angular-acceleration gains multiplied by nominal inertia J (per axis):
  `kp = (35, 35, 20)·J`, `ki = (60, 60, 20)·J`, `kd = (0.7, 0.7, 0)·J`.
- P and I on error `e = ω_sp − ω`; D on **measurement** (−dω/dt), low-pass filtered with a first-order filter `alpha = 1 − exp(−2π·20 Hz·dt)`.
- Integrator clamped to `|i| ≤ 8` per axis and **frozen while the mixer reports saturation**.
- Mixer (thrusts → commands by inverting the thrust curve): roll/pitch differential thrusts are scaled down uniformly to fit `[T_idle, thrust_max]`; collective is then shifted to make room; yaw is added last with the largest scale s∈[0,1] that stays within limits.
- Collective thrust and rate scaling use **nominal** `thrust_max` / `max_body_rate`.
- Gains/filter live in `sim/ctbr.py::RatePIDGains`; the source of truth will move to `config/rate_pid.yaml` when the SITL workstream wires firmware ↔ sim.

## 10. Reset conventions
- "Previous action" (obs 15-18) is **zeros** at reset / on policy engage.
- Observation attitude is currently ground truth in sim (no estimator error modelled yet); gyro has noise + bias.

## 11. int8 clarifications (from WS2, binding)
- Rounding in §7 step 1 is **half away from zero**; numpy's `np.round` is half-to-even, so the golden model must implement it explicitly (`np.sign(x)*np.floor(np.abs(x)+0.5)` in float32).
- NaN observation entries map to 0 before quantization.
- Bias is added in int64 and the accumulator saturates to int32.
- Codegen emits `s_x_last` (and all scales) as float32 literals; nothing is recomputed on target.
- C++ structs `FloatModel` / `Int8Model` and the CRC32 rule are documented in `fw/core/include/nf/policy.hpp` — that header is the codegen target.
