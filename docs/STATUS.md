# Status: end of software phase (2026-11-07)

All of this runs without hardware. Test totals: pytest 151 passed + 2 expected failures; fw/core ctest 9/9 (normal and ASan/UBSan).

| Piece | Result |
|---|---|
| Simulator (sim/) | Batched quad sim, ~600k steps/s on 4 CPU cores; PID baseline 100% recovery from 60° tilt on randomized vehicles |
| Firmware core (fw/core) | C++17, no heap/exceptions; filters, Mahony, rate PID + mixer, cascaded fallback, float/int8 MLP, Simplex supervisor with fallback ladder, gyro calibration |
| int8 toolchain (tools/) | quantize → golden model → codegen; C++ int8 matches Python bit-for-bit on 12k vectors |
| Policy v2 (train/policies) | Training env: 100% hover/tilt/throw, 98% upside-down; 2.2–2.7 cm error |
| SITL (fw/sitl) | Real firmware flying the v2 policy in sim, 20 randomized vehicles: hover 3.4 cm median, no supervisor trips (int8: 3.8 cm). Tilt recovery 17/20 and push 18/20 stay in policy; the rest hand over to the fallback without crashing |
| Safety | 8 injected faults (NaN, saturation, oscillation, flip, IMU disagreement, RC loss, deadline overrun, kill) all handled; flip caught at ~27° tilt |
| Board (hw/) | Pin map with checker, BOM ~$65/board in a run of 5, order list; datasheet items flagged |

## Key finding
v1 looked perfect in the training env but flew poorly inside the real firmware. The cause was the firmware's flow-based velocity estimate having correlated error that training never saw. Putting the firmware's own estimator and gyro filtering into the training sim fixed hover (8.5 → 3.4 cm median, 2 trips → 0).

## Open (docs/FOLLOWUPS.md)
- Remaining tilt/push supervisor trips (rate limit while flipping back; estimator-health during hard braking).
- Thrust command noise ~3× PID; no integral action in policy.
- Attitude-estimator error under sustained acceleration not modelled in training.

## Needs the user
1. A GitHub repo to push to.
2. Hardware for milestone M3 (hw/ORDER-LIST.md, "needed for M3"). Note: most 3" ducted frames don't take a 30.5 mm board; buy one that does, or the PCB shrinks to 25.5 mm.
