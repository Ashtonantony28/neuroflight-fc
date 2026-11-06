# Order list: airframe, flight parts and lab gear

Prices are rough late-2026 USD retail. The example products are **examples, not endorsements**: stock and model names change, so check the listed spec before buying. Items marked **SAFETY** are not optional.

**One compatibility rule decides most of this list.** Our own FC (hw/DESIGN.md) is 36 x 36 mm with a **30.5 x 30.5 mm** mount. Many 3" ducted cinewhoop frames only take 20x20 or 25.5x25.5 stacks. You have two options:
- **(a)** buy a 3-3.5" ducted frame that lists a **30.5 x 30.5 stack mount**, and a 30.5 mm 4-in-1 ESC; or
- **(b)** buy a 20x20 setup for M3 and accept that the custom board needs a 30.5-capable frame later (or a 25.5 mm rev of the PCB).

Option (a) is recommended so the M3 airframe also carries the M8 custom board. Check the frame's spec sheet for "30.5x30.5" before ordering.

---

## A. Needed for M3 bring-up

| # | Item | What to look for | Example products | Approx. |
|---|---|---|---|---|
| 1 | 3-3.5" ducted frame | **30.5 x 30.5 stack mount**, full ducts, fits 1404-1505 motors, < 100 g | iFlight ProTek35 / GEPRC CineLog35-class frames (**VERIFY** that the listing states 30.5 mm mount) | $35-60 |
| 2 | Motors x4 (+2 spare) | 1404-1505, ~3800-4600 KV for 4S, M2 mount pattern matching the frame | T-Motor / BrotherHobby / iFlight XING2 1404 4600KV | $55-80 for 4; ~$15 each spare |
| 3 | 4-in-1 ESC | **Ships with AM32 firmware**, supports **bidirectional DShot**, 30.5 mm mount, 35-45 A, 2-6S, comes with the 8-pin JST-SH harness. Check the harness pinout against hw/DESIGN.md J2 | Any 30.5 mm 4-in-1 sold as "AM32" (several vendors: e.g. Aikon AK32, SEQURE, HAKRC AM32 variants). **VERIFY "AM32" in the listing**, not BLHeli_S/BLHeli_32 | $40-70 |
| 4 | Commercial STM32H743 FC (for M3, M5, M6) | **H743** MCU, 30.5 mm, ICM-42688-P or BMI270 gyro, blackbox, Betaflight target exists | Matek H743-SLIM V3 or H743-WING; Holybro Kakute H7 (v1.5/v2); SpeedyBee F7 is **not** H7, so skip it | $55-90 |
| 5 | Props (lots) | 3" (or 3.5") tri-blade sized to the duct, matching shaft (T-mount vs 1.5 mm / M5) | Gemfan D76/D90 ducted props, HQProp DT series | $15-25 |
| 6 | ELRS receiver | ExpressLRS 2.4 GHz, UART (CRSF), small antenna | RadioMaster RP1/RP2, BetaFPV ELRS Nano | $15-25 |
| 7 | ELRS radio (if you don't own one) | ELRS internal module, EdgeTX | RadioMaster Pocket / Boxer / Zorro ELRS | $60-150 |
| 8 | Flow + ToF module | **Single UART module** (matches FC J4), MSP or MAVLink output, rangefinder ≥ 2 m indoors | MicoAir MTF-01 / MTF-02, Matek 3901-L0X | $20-40 |
| 9 | 4S LiPo x4 | 4S 650-850 mAh, XT30, 75C+, < 100 g | Tattu R-Line, CNHL, GNB 4S 650/850 | $70-110 for 4 |
| 10 | **SAFETY** Balance charger | Balance charging, storage mode, ≥ 50 W | ToolkitRC M6/M8, ISDT 608AC, HOTA D6 | $40-90 |
| 11 | **SAFETY** LiPo fire bag / ammo can, plus a sand or metal tray for charging | Charge **only** inside it, never unattended | any "LiPo safe bag" | $10-20 |
| 12 | **SAFETY** Smoke stopper (inline fuse/PTC for first power-up) | XT30 version | e.g. "smoke stopper XT30" (several vendors) | $8-15 |
| 13 | **SAFETY** Safety glasses + props-off rule | Wear them for every powered bench test | — | $10 |
| 14 | Debug probe | SWD + SWO, supported by probe-rs/OpenOCD | ST-LINK V3MINIE (cheap, good), or SEGGER J-Link EDU Mini (non-commercial use only) | $15-60 |
| 15 | Tag-Connect cable for our PCB | TC2030-IDC-NL (+ optional retaining clip board). For the ST-LINK V3 you also need a 14-pin STDC14 → 10-pin adapter, or use the TC2030-CTX-NL (Cortex 10-pin) | Tag-Connect TC2030-CTX-NL | $40-55 |
| 16 | Logic analyzer | 8 ch, ≥ 24 MS/s (DShot600 bit = 1.67 µs; 24 MS/s gives ~40 samples/bit, which is fine), sigrok/PulseView support | generic "Saleae-compatible" 24 MHz FX2 analyser; DSLogic Plus if you want 100 MS/s + triggers | $12-110 |
| 17 | Thrust stand (system ID: thrust curve, motor τ) | Load cell + HX711, or commercial with RPM + current | DIY: 1-5 kg bar load cell + HX711 board + 3D-printed mount + Arduino/ESP32 ($15-25). Commercial: Tyto Robotics Series 1585 ($200+) | $20-250 |
| 18 | **SAFETY** Bench power supply with current limit | 0-30 V, ≥ 3 A, **adjustable current limit** (first power-up of every board at 100 mA) | Korad KA3005D, Riden RD6006 + PSU | $50-100 |
| 19 | Soldering consumables | XT30 connectors, 18-20 AWG silicone wire, heat shrink, M3 grommets/nylon standoffs | — | $20 |

**M3 subtotal:** roughly **$420-750** (with your own radio and the DIY thrust stand), up to ~$1,000 with a radio and a commercial thrust stand.

## B. Later (M5-M8 and stretch)

| # | Item | When | Example | Approx. |
|---|---|---|---|---|
| 20 | 3-DOF gimbal test rig | M5 (before first free flight) | DIY 3D-printed gimbal with bearings; FC + frame mounted at CG | $30-50 |
| 21 | **SAFETY** Net enclosure | M5 hover tests | DIY: PVC/aluminium poles + golf/batting-cage netting, ≥ 2.5 x 2.5 x 2 m | $60-120 |
| 22 | Custom PCB fab + assembly (5 boards) | M4 → M8 | JLCPCB 4-layer + PCBA; see `hw/bom_cost.py` | ~$65/board, ~$325/5 |
| 23 | Spare motors, props, frame arms/ducts | ongoing | — | $40-80 |
| 24 | Extra LiPos (to 6-8 total) | M6+ | same as #9 | $70-110 |
| 25 | Hot-air station + microscope / loupe | M8 rework (LQFP/LGA) | any 858D-class hot air + 10-20x loupe | $50-150 |
| 26 | Oscilloscope (≥ 50 MHz, 2 ch) | M8 rail ripple / DShot edges | Rigol DHO800 / Siglent SDS1104X-E | $300-450 (optional) |
| 27 | iCE40UP5K parts + dev board | stretch S3 | iCEBreaker or UPduino 3.x for development; then populate U10/U11 on our board | $30-70 |
| 28 | Cloud GPU time | if no local RTX | — | $50-200 |

## Safety notes

- **Never** power a new or modified board straight from a LiPo. Use the current-limited bench supply first, then the smoke stopper, then the battery.
- **Props off** for all bench work, including ESC config, DShot tests and firmware flashing with a battery connected. Props go on only in the net/gimbal per PLAN §5.2.
- Charge LiPos in the fire bag, on a non-flammable surface, attended. Store at storage voltage (3.8 V/cell). Dispose of puffed packs.
- Check the ESC harness pinout (VBAT/GND position) with a multimeter before plugging it into any FC.
- ELRS: set a failsafe on the receiver (no pulses) and test that motors stop when the radio is switched off, before any props-on test.
