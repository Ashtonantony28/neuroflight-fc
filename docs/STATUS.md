# Status: end of software phase (2026-11-07; v3 round 2026-11-11)

All of this runs without hardware. Test totals: pytest 163 passed + 2 expected failures (v3 round); fw/core ctest 9/9 (normal and ASan/UBSan).

| Piece | Result |
|---|---|
| Simulator (sim/) | Batched quad sim, ~600k steps/s on 4 CPU cores; PID baseline 100% recovery from 60° tilt on randomized vehicles |
| Firmware core (fw/core) | C++17, no heap/exceptions; filters, Mahony (accel reference = KF acceleration + g since v3), rate PID + mixer, cascaded fallback, float/int8 MLP, Simplex supervisor with fallback ladder, gyro calibration |
| int8 toolchain (tools/) | quantize → golden model → codegen; C++ int8 matches Python bit-for-bit on 12k vectors |
| Policy v3 (train/policies) | Trained with the firmware Mahony estimate in the loop, RATE/ESTIMATOR margins, smoothness regularization (62 min, warm start from v2). Training env (randomized): hover 98.8%, throw 97.9%, tilt 86.5% (v2 93.6/90.8/86.3%); thrust noise 1.3–1.4× PID (v2 3×); worse than v2 on nominal 60° tilt (92.8 vs 95.3%) and worst extreme (84 vs 91%) |
| SITL (fw/sitl) | Real firmware flying v3, 20 randomized vehicles: hover 2.8 cm median, no trips (int8 3.2 cm); tilt 19/20 and push 20/20 recoveries, no supervisor trip at all (v2: 16 and 19); thrust noise 1.6× the fallback PID (v2 3.9×, v3 int8 4.4×) |
| Safety | 8 injected faults (NaN, saturation, oscillation, flip, IMU disagreement, RC loss, deadline overrun, kill) all handled; flip caught at ~27° tilt |
| Board (hw/) | Pin map with checker, BOM ~$65/board in a run of 5, order list; datasheet items flagged |

## Key findings
v3: the firmware Mahony was pulled 8–14° off while braking hard (it compared the accelerometer with gravity); using KF acceleration + g as the reference cut that to ~2° and removed the ESTIMATOR trips, and training against the ported estimator plus monitor margins removed the remaining recovery trips.


v1 looked perfect in the training env but flew poorly inside the real firmware. The cause was the firmware's flow-based velocity estimate having correlated error that training never saw. Putting the firmware's own estimator and gyro filtering into the training sim fixed hover (8.5 → 3.4 cm median, 2 trips → 0).

## Open (docs/FOLLOWUPS.md)
- v3 thrust noise in SITL 1.6× PID (target 1.5×), int8 build 4.4× (quantization steps); small noise-free thrust ripple.
- v3 settles slower than v2 from a 60° tilt on the nominal vehicle (training-env success 91–93% vs 95–98%); hover error tail wider (max 7.7 vs 4.8 cm).
- No integral action in policy; flow sensor model idealized (FOLLOWUPS 9, 10).

## Needs the user
1. A GitHub repo to push to.
2. Hardware for milestone M3 (hw/ORDER-LIST.md, "needed for M3"). Note: most 3" ducted frames don't take a 30.5 mm board; buy one that does, or the PCB shrinks to 25.5 mm.
