# Open follow-ups (from SITL, 2026-11-06)
1. sim/env.py `_reset_idx` resets rate PID without the gyro → derivative kick on first step (sim/firmware mismatch up to 0.22).
2. Firmware rate PID uses filtered + bias-corrected gyro; sim/training uses raw gyro (gap 0.017 max). Decide one and align §9.
3. Gyro bias takes ~20 s to converge; add at-rest gyro calibration before arming.
4. Arming IMU A/B check compares single raw samples → spurious refusals; average over a window.
5. Flips caught late (only tilt monitor, ~110°); add a tilt-rate / predicted-tilt monitor.
6. Fallback velocity integrator reset on handover may cause 0.35 m sag; seed it from current state.
