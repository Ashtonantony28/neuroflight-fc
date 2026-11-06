# Learned-Control Flight Controller: project plan

**Status:** plan for review. Nothing gets designed, ordered or coded until you say proceed.
**Written:** 2026-11-06
**Working name:** `neuroflight-fc` (placeholder, rename freely)

---

## 1. The idea in one paragraph

Design and build your own quadrotor flight-controller PCB, write its firmware from scratch, and fly it with a small neural-network policy that was trained entirely in simulation with reinforcement learning. The network runs on the microcontroller at 500 Hz to 1 kHz and commands the motors directly. Underneath it sits a classical cascaded PID controller (and later a tiny MPC) that a runtime safety monitor switches to the instant the learned policy does something it shouldn't. The hard, interesting parts are: getting a policy to survive the sim-to-real gap, fitting and verifying neural inference inside a hard real-time loop, and building a safety architecture that lets you trust an unverified controller with spinning propellers.

**Why this fits you:** it joins the hardware/RTL side (tiny-gpu-extended) with the ML side (nanoGPT Extended). Several pieces reuse methods you already built: int8 quantization with a bit-exact golden model (the xf track), a verification-first workflow with cocotb-style test harnesses, and audit loops. There is also an optional stretch where the policy runs on an FPGA inference block you write in SystemVerilog.

---

## 2. Goals

### Must-have (definition of done for v1)
1. **Custom FC PCB**, 4-layer, 30.5 x 30.5 mm mount pattern, that boots, reads two IMUs, logs to flash and drives a 4-in-1 ESC over bidirectional DShot.
2. **From-scratch firmware** with a deterministic control loop (IMU data-ready interrupt → estimate → control → motor output) with measured jitter under 20 µs at 1 kHz.
3. **Classical fallback**: cascaded attitude/rate PID that flies the drone stably on its own (this is flown first, before any learned control).
4. **Learned policy** trained in simulation (PPO, small MLP) that **hovers and recovers from a hand-thrown launch** on the real drone, with zero real-world fine-tuning.
5. **Safety monitor** (Simplex architecture) that detects policy misbehaviour and hands over to the fallback, demonstrated by deliberately injecting faults in flight.
6. **Reproducible pipeline**: one command from `train` → `export` → `quantize` → `codegen` → `flash`, with the on-board inference checked bit-exact against a Python reference in CI.

### Stretch goals (pick later)
- **S1. Direct-motor-command waypoint tracking** (fly a figure-eight indoors using optical-flow position estimates).
- **S2. TinyMPC fallback** replacing or supplementing PID as the safe controller.
- **S3. FPGA inference co-processor**: an iCE40UP5K on the board (or a daughterboard) running the int8 MLP in your own RTL, with the MCU as host. Directly reuses tiny-gpu skills.
- **S4. Own ESC**: a single-motor FOC or 6-step ESC board. Big scope increase; listed only for completeness.
- **S5. Adaptive policy**: recurrent or history-conditioned policy that adapts to a changed payload or a damaged prop in flight.

### Non-goals
- Outdoor GPS navigation, FPV video, autonomy beyond a few metres indoors.
- Certifying anything. This is a research platform flown in a net.

---

## 3. System architecture

### 3.1 Hardware block diagram

```
                    ┌──────────────────── FC PCB (4-layer, 36x36 mm) ─────────────────────┐
  ELRS receiver ──► │ UART (CRSF)                                                          │
                    │                       STM32H743 (480 MHz Cortex-M7, FPU, 1 MB RAM)   │
  IMU A ICM-42688-P │◄─ SPI1 + DRDY IRQ ──┐                                               │
  IMU B BMI270      │◄─ SPI2 + DRDY IRQ ──┤  TIM/DMA ──► 4x DShot600 (bidir) ─────────────┼──► 4-in-1 ESC (AM32)
  Baro  DPS310      │◄─ I2C ──────────────┤                                               │       │ eRPM telemetry
  Flash W25N01G     │◄─ QSPI (blackbox) ──┤  ADC ◄── VBAT divider, current sense ──────────┼───────┘
  Flow PMW3901 +    │◄─ SPI3 / I2C ───────┘  USB-C (DFU, MAVLink-lite telemetry, HIL)     │
  ToF  VL53L1X      │                        SWD header (debug, trace)                     │
                    │  [optional S3] iCE40UP5K on SPI/QSPI as NN co-processor              │
                    │  LDOs: 3.3 V (digital), 3.3 V low-noise (IMUs, separate), 5 V BEC    │
                    └──────────────────────────────────────────────────────────────────────┘
```

Key board decisions:
- **STM32H743** because it is the best-supported high-end FC MCU (Betaflight, PX4 and ArduPilot all have H743 targets), so the hardware can be validated with existing firmware before your own exists. A 2x64 MLP takes roughly 10-30 µs per inference on it in float32, so 1 kHz is comfortable.
- **Two different IMUs** from two vendors on two SPI buses, for redundancy and for the safety monitor's consistency check. Soft-mounted IMU island with its own low-noise LDO.
- **Bidirectional DShot** gives per-motor RPM at the loop rate, which is gold for system identification and for an RPM-feedback input to the policy.
- **Off-the-shelf 4-in-1 ESC** (AM32 open-source firmware) so the project is not blocked on power electronics.
- **Optical flow + ToF** give indoor velocity and height without motion capture.

### 3.2 Airframe
3-inch ducted "cinewhoop" class, about 200-250 g all-up. Ducts protect props and people, the airframe is sturdy, and it stays under 250 g. Batteries 4S 650-850 mAh.

### 3.3 Firmware architecture

```
 IMU DRDY IRQ (1-8 kHz) ─► sensor driver ─► notch/LPF filters ─► estimator (Mahony/EKF attitude, flow+ToF velocity/height)
                                                                         │ state x̂
                         ┌───────────────────────────────────────────────┴────────────┐
                         ▼                                                            ▼
              Learned policy π(x̂, a_prev)                                  Fallback controller
              (int8/float MLP, codegen'd C)                                (cascaded PID; TinyMPC later)
                         │ u_π                                                        │ u_safe
                         └──────────────► Safety supervisor (Simplex switch) ◄────────┘
                                                   │ u
                                         mixer / saturation ─► DShot DMA
 RC input (CRSF) ─► arm / mode / KILL (hardware-priority path, bypasses everything) ───┘
 Blackbox logger (QSPI flash, ring buffer, 1 kHz state + actions + monitor flags)
```

- Bare-metal, interrupt-driven, no RTOS in the control path (a small cooperative scheduler for logging and telemetry). Language is an open decision (C++ or Rust/Embassy, see section 11).
- The **policy module is generated code**: weights as `const` arrays, layer kernels written by hand (CMSIS-DSP where it helps), with a fixed worst-case execution time measured on target.
- **Hardware-in-the-loop (HIL) mode**: over USB, the simulator streams synthetic IMU/flow data into the firmware and reads back motor commands. The exact flight binary is exercised against the simulator before any prop goes on.

### 3.4 Control-policy design

| Choice | Plan | Reason |
|---|---|---|
| Algorithm | PPO (asymmetric actor-critic: critic sees privileged true state and parameters) | Robust, well-trodden for quadrotors, scales to thousands of parallel envs |
| Actor | MLP 2x64 or 3x64, tanh | Small enough for µs inference; prior work flies with this size |
| Observations | 6D rotation, body rates, linear velocity, position error, last 2-3 actions, (optional) motor RPMs | Action history helps with latency and motor lag |
| Actions, stage A | Collective thrust + body rates (CTBR), tracked by the on-board rate PID | Easy first transfer |
| Actions, stage B | Direct normalized motor thrusts at 500 Hz to 1 kHz | The genuinely interesting "end-to-end" result |
| Reward | Position/attitude error, rate penalty, action smoothness and action-rate penalty, crash penalty | Smoothness terms matter a lot for real motors |

Prior work this follows: Hwangbo et al. 2017 (RL quadrotor control), Molchanov et al. 2019 ("Sim-to-(Multi)-Real", direct motor control with domain randomization), Kaufmann et al. 2023 (Swift, drone racing), Eschmann et al. 2024 ("Learning to Fly in Seconds", Crazyflie), and Nguyen et al. 2024 (TinyMPC on a Cortex-M4).

---

## 4. Training and sim-to-real pipeline

```
 system ID (thrust stand, bifilar pendulum, step tests)
        │ params + uncertainty ranges
        ▼
 Simulator(s) ─► PPO training (thousands of parallel envs, domain randomization)
        │ policy.pt
        ▼
 export ─► quantize (int8 PTQ, then QAT if needed) ─► golden model (numpy, integer-exact)
        │                                                  │ bit-exact check in CI
        ▼                                                  ▼
 codegen C (weights + kernels) ─────────────► firmware build ─► SITL ─► HIL (real board) ─► tethered ─► free flight
```

### 4.1 Simulators
- **Primary training: NVIDIA Isaac Lab** (GPU-parallel, thousands of quadrotors per GPU). Needs an RTX-class NVIDIA GPU, local or rented.
- **Fallback / cross-check: a custom vectorized rigid-body + motor model in PyTorch or JAX.** It is only a few hundred lines, makes the dynamics fully transparent, and gives a second simulator to test that the policy isn't exploiting one engine's quirks. Policies must pass in both.
- **Gazebo** is used only for SITL integration tests (firmware compiled for the host, talking to the sim), not for training.

### 4.2 Closing the reality gap
1. **System identification first.** Measure thrust and torque vs. throttle on a load-cell thrust stand; motor time constant from bidirectional-DShot RPM step responses; mass and inertia (bifilar pendulum); loop latency end-to-end (IMU sample → motor output).
2. **Domain randomization** around the measured values: mass ±15%, inertia ±20%, thrust coefficient ±15%, motor time constant ±30%, latency 0-8 ms, IMU noise and bias, per-motor thrust imbalance, battery-sag curve, initial states including upside-down.
3. **Model the firmware, not just the physics.** The sim runs the same filter, estimator and quantized policy as the board (via the golden model), at the same rate, with the same action quantization.
4. **Gate every policy** through an evaluation suite: success rate under random disturbances, worst-case attitude excursion, action smoothness spectrum (motors hate high-frequency content), behaviour at sim-parameter extremes.

### 4.3 Quantization and on-board verification
- Start with float32 on the H7 (it has an FPU, and it's fast enough). int8 is introduced as a deliberate experiment and is mandatory for the FPGA stretch.
- Golden-model methodology carried over from your xf track: a numpy integer reference produces exact expected outputs; the firmware's inference is run on the host (SITL build) and on target (over SWD/USB) and must match bit-for-bit.

---

## 5. Safety design

Safety is a first-class deliverable, not an afterthought. The rule: **the learned policy is never the last line of defence.**

### 5.1 Layers (outermost first)
1. **Physical**: ducted props, flights only inside a net enclosure, props off for all bench work, LiPo charging in a fire bag, a spotter for flights.
2. **RC kill switch**: handled in its own high-priority interrupt; it disarms the motors directly and cannot be overridden by any software mode. RC link loss for more than 300 ms triggers failsafe.
3. **Arming checks**: IMU agreement, estimator converged, battery voltage, policy checksum matches the build, throttle low, no fault flags.
4. **Runtime supervisor (Simplex)**: switches from policy to fallback when any monitor trips:
   - attitude outside an envelope (for example tilt > 60° when not commanded), body rate > limit
   - position outside a geofence (flow + ToF), height below a floor
   - policy output NaN/Inf, saturated for > N ms, or high-frequency oscillation detected
   - inference deadline overrun, control-loop jitter beyond limit
   - IMU A/B disagreement, estimator innovation out of bounds
   - the state leaving the region where the fallback is known to recover (estimated offline in sim)
5. **Fallback ladder**: learned policy → PID (or TinyMPC) stabilize-and-hold → controlled descent → motor stop. Switching back to the policy needs a manual RC action, never an automatic one.

### 5.2 Testing ladder
Unit tests → SITL → HIL with the real board (props off) → motors on, props off → 3-DOF gimbal rig (drone bolted at its centre of mass, can only rotate) → tether in the net → free hover in the net → fault-injection flights. A stage is passed only after a written checklist.

### 5.3 Legal
Flown indoors in a net, under 250 g. If you ever fly outdoors in the US, take the free FAA TRUST certificate first; check local rules if you are elsewhere.

---

## 6. Bill of materials and cost

Prices are rough late-2026 hobby/retail estimates in USD.

### 6.1 FC board (per board; build 5)
| Part | Example | Approx. |
|---|---|---|
| MCU | STM32H743VIT6 (LQFP100) | $12-16 |
| IMU A / IMU B | ICM-42688-P / BMI270 | $4 + $4 |
| Barometer | DPS310 or BMP390 | $3 |
| Blackbox flash | W25N01G (1 Gbit NAND) | $3 |
| Regulators, passives, connectors, USB-C, crystal | various | $8-12 |
| PCB fab + assembly (4-layer, JLCPCB-style, qty 5) | | $25-50 per board |
| **Per board** | | **~$60-90**, so **$300-450 for 5** |

### 6.2 Vehicle and sensors
| Part | Approx. |
|---|---|
| 3" ducted frame | $30-50 |
| 4x motors (1404-1507 class) | $50-70 |
| 4-in-1 ESC, AM32, bidirectional DShot | $40-60 |
| Props (lots of spares) | $15 |
| ELRS receiver | $15-25 |
| ELRS radio transmitter (if you don't own one) | $60-150 |
| Optical flow + ToF module (PMW3901 + VL53L1X) | $20-35 |
| 4S LiPo x4 + balance charger + fire bag | $100-150 |

### 6.3 Test and lab gear
| Part | Approx. |
|---|---|
| Debug probe (ST-Link V3 or J-Link EDU Mini) | $30-60 |
| Logic analyzer (8-ch) | $15-30 |
| Thrust stand (DIY load cell + HX711, or commercial) | $30-200 |
| 3-DOF gimbal test rig (DIY 3D printed) | $30-50 |
| Net enclosure (DIY poles + netting) | $60-120 |
| Bench power supply (if needed) | $50-80 |

### 6.4 Compute
| Item | Approx. |
|---|---|
| Training on your own RTX GPU | $0 |
| Or rented cloud GPU (~100-150 GPU-hours over the project) | $50-200 |

### Total
- **Lean** (own a radio and a GPU, DIY rigs): **about $650-850**
- **Comfortable** (commercial thrust stand, cloud GPU, spares): **about $1,000-1,400**
- Stretch S3 adds ~$50-100 (FPGA + daughterboard fab). Stretch S4 (own ESC) adds $200+ and a lot of time.

---

## 7. Milestones

Durations assume part-time evenings and weekends. Each milestone ends with a gate you review.

| # | Milestone | Deliverable / gate | Est. |
|---|---|---|---|
| M0 | Kickoff and decisions | Section 11 decisions made; repo scaffold, CI, sim chosen | 1 wk |
| M1 | Simulation baseline | Custom quadrotor sim + PID that hovers in sim; Isaac Lab env with same params | 2-3 wk |
| M2 | First learned policy in sim | PPO CTBR hover + throw recovery, >99% success under randomization; eval suite | 2-3 wk |
| M3 | Off-the-shelf hardware bring-up | Drone built on a commercial H743 FC running Betaflight; system ID done (thrust curve, motor τ, inertia, latency) | 2 wk |
| M4 | FC PCB design | Schematic + layout review (I'll do a design review pass), DRC clean, ordered | 3-4 wk + fab |
| M5 | Firmware v0 on the commercial FC | Drivers, filters, estimator, DShot, PID; flies on the gimbal rig, then hovers in the net | 4-5 wk |
| M6 | Policy on board (CTBR) | Codegen + golden-model CI; HIL passes; CTBR policy flies on the gimbal rig, then hovers | 3 wk |
| M7 | Safety supervisor | All monitors, fallback ladder, fault-injection flight video | 2-3 wk |
| M8 | Custom PCB bring-up | Own board runs the same firmware; flies; jitter and noise measured vs. commercial FC | 2-3 wk |
| M9 | Direct motor control | Stage B policy (motor commands at 500 Hz-1 kHz) hovers and recovers from throws, zero-shot | 3-5 wk |
| M10 | Write-up | Results report: sim vs. real metrics, ablations (randomization, history, quantization), videos | 1-2 wk |
| — | Stretch S1-S5 | Chosen after M10 | open |

**Total for v1: roughly 6-8 months part-time.** Software (M1, M2, M5, M6) and hardware (M3, M4) tracks run in parallel; PCB fab waits are hidden behind firmware work. Note M5 and M6 deliberately happen on a **commercial** H743 board first, so a mistake on your own PCB never blocks the ML work.

---

## 8. Repository layout (proposed)

```
neuroflight-fc/
  hw/            KiCad project, BOM, fab outputs, design-review notes
  fw/            firmware (drivers/, estimation/, control/, policy/ (generated), safety/, hil/)
  sim/           custom vectorized sim, Isaac Lab task, Gazebo SITL bridge
  train/         PPO configs, domain-randomization specs, eval suite
  tools/         export, quantize, golden model, codegen, log decoder, sysid scripts
  rigs/          thrust stand, gimbal rig CAD and firmware
  docs/          ADRs, safety checklists, flight logs index, PLAN.md
  test/          host unit tests, golden-model bit-exact tests, HIL tests
```

---

## 9. Verification and metrics

- **Control loop**: loop rate, jitter (p99, max), inference WCET, CPU load, all logged on target.
- **Flight quality**: hover RMS position and attitude error, throw-recovery time and success rate over 20 throws, motor command high-frequency energy, motor temperature after 60 s hover.
- **Sim-to-real gap**: same scripted manoeuvre in sim and real, compare trajectories.
- **Safety**: every monitor gets a fault-injection test in HIL and at least one in flight; record switch-over latency and altitude lost.
- **Ablations for the write-up**: no randomization vs. randomization; with/without action history; float32 vs. int8; CTBR vs. direct motor.

---

## 10. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Policy fails to transfer (oscillation, drift) | High | High | System ID first; latency + motor-lag randomization; smoothness rewards; CTBR stage before direct motor; two-simulator check |
| IMU vibration noise on own PCB | Medium | High | Soft-mount, separate LDO, RPM notch filters, compare against commercial FC at M8 |
| PCB respin needed | Medium | Medium | Validate firmware on a commercial FC first; design review before fab; test points and SWD on every rail |
| Crashes destroy hardware | High | Low-Med | Ducts, net, gimbal rig, spare props/motors/boards budgeted |
| Isaac Lab setup/GPU access friction | Medium | Medium | Custom PyTorch/JAX sim is a full fallback for training |
| Inference too slow at 1 kHz with bigger nets | Low | Medium | Keep 2x64; run policy at 500 Hz with 1 kHz rate loop; int8; FPGA stretch |
| Estimator quality limits position control indoors | Medium | Medium | Hover/recovery v1 needs mostly attitude + height; flow quality tested on gimbal + cart before flight |
| LiPo fire | Low | Severe | Fire bag, never unattended charging, voltage cutoffs in firmware |
| Scope creep (own ESC, FPGA) | High | Medium | Stretches locked behind M10 |

---

## 11. Decisions I need from you before starting

1. **Simulator**: do you have an NVIDIA RTX GPU for Isaac Lab, or should training default to the custom PyTorch/JAX sim (with cloud GPU only if needed)? *My recommendation: build the custom sim first (M1) regardless; add Isaac Lab when scaling.*
2. **Firmware language**: C++ (closest to Betaflight/PX4 references, easiest debugging) or Rust with Embassy (safer, more novel, fewer references). *Recommendation: C++17, bare-metal.*
3. **Airframe size**: 3" ducted under 250 g (safer indoors) vs. 5" open props (more authority, less safe). *Recommendation: 3" ducted.*
4. **Position sensing**: optical flow + ToF (self-contained) vs. a cheap external system (UWB anchors, or a webcam with ArUco markers). *Recommendation: flow + ToF for v1.*
5. **FPGA co-processor (S3)**: plan the footprint onto the v1 PCB now (cheap to add, unpopulated), or keep it as a later daughterboard? *Recommendation: leave a footprint and SPI lines on v1, populate later.*
6. **Repo**: new GitHub repo under your account (you'd need to create it; I can't), named however you like.

Once you've answered these (or said "go with your recommendations"), M0 starts.
