# neuroflight-fc v1 flight-controller PCB: design package

Status: **written design package for schematic capture** (KiCad was not available when this was written). Nothing here has been simulated, laid out, or checked against datasheets in hand. Every item marked **VERIFY** must be checked against the named datasheet before the schematic is signed off at the M4 review.

Files in `hw/`:

| File | What it is |
|---|---|
| `DESIGN.md` | This document |
| `pinmap.csv` | Complete STM32H743VIT6 (LQFP100) pin assignment, the single source of truth |
| `check_pinmap.py` | Pin map checker (conflicts, required functions, AF table, EXTI, DShot rules). `python3 hw/check_pinmap.py`; `--markdown` regenerates the table in §7 |
| `bom.csv` / `bom_cost.py` | BOM and per-board cost roll-up. `python3 hw/bom_cost.py [--boards N] [--include-dnp]` |
| `ORDER-LIST.md` | Off-the-shelf airframe and lab parts to buy |

---

## 1. Board outline and stackup

- **Outline:** 36.0 x 36.0 mm, corner radius 3 mm, 1.6 mm FR-4.
- **Mounting:** 4 holes on a **30.5 x 30.5 mm** square (hole centres at (±15.25, ±15.25) mm from board centre). 4.0 mm holes for M3 silicone grommets (the industry norm). If you plan to run hard-mounted M3 screws, use 3.2 mm instead. Keep copper 0.5 mm clear of the grommet OD (about 7 mm keep-out circle) on all layers, except a GND ring on L1/L4 if you want the frame grounded (default: **not** grounded).
- **Orientation:** an arrow on silkscreen marks body +x (forward). The MCU and IMU axes are documented in firmware as a board-to-body rotation, so the parts do not have to be aligned to the arrow. Aligning them anyway makes debugging easier.
- **Stackup (JLC JLC04161H-7628 or similar, 4-layer, 1 oz outer / 0.5 oz inner):**

| Layer | Use |
|---|---|
| L1 (top) | MCU, IMU island, connectors, signals |
| L2 | **Solid GND plane**, unbroken (the reference for all L1 signals) |
| L3 | Power pours: VBAT (short run from ESC connector to buck only), 5V, 3V3_D; 3V3_IMU as its own island; low-speed signals where needed |
| L4 (bottom) | Buck converter + inductor, USB-C, ESC connector, test points, some routing; GND pour stitched to L2 |

- Minimum feature: 0.1 mm (4 mil) track/space and 0.3/0.45 mm vias. That is fine for JLC 4-layer with no extra cost. No blind or buried vias. No via-in-pad except under the buck's thermal pad, where vias may be left open because JLC tents them by default.
- **Component placement:** the MCU sits in the centre of the top side. The IMU island sits in the top-side quadrant **diagonally opposite** the buck converter (bottom side) and the ESC connector. Because ESCs mount below the FC in a stack, put J2 on the bottom-side edge nearest the ESC.

```
           +x (forward)
   ┌───────────────────────────────┐
   │ o  [J3 RX] [J4 FLOW]  [J5 AUX]│ o = M3 grommet hole
   │  ┌───────┐    ┌───────────┐   │
   │  │IMU    │    │           │   │  top side
   │  │island │    │ STM32H743 │   │
   │  │A + B  │    │  LQFP100  │   │
   │  └───────┘    └───────────┘   │
   │ [BARO]  [FLASH]   [FPGA DNP]  │
   │ o [USB-C]  [SWD TC2030]  [J2] │ <- J2 ESC is on bottom side, this edge
   └───────────────────────────────┘
     (bottom: buck + L1 under J2/USB corner, far from IMU island)
```

## 2. Block diagram

```
 ESC 8-pin JST-SH (J2)
   VBAT ──► TVS D3 ──► LMR36015 buck ──► 5V ──┬──► J3 receiver, J4 flow/ToF, J5 aux, buzzer
   GND                  (60 V, 1.5 A)        │
   CUR  ──► RC ──► PC1 (ADC)                 ├──► LDL1117 ──► 3V3_D (MCU, flash, baro, LEDs, FPGA IO)
   TLM  ◄── PE7 UART7_RX                     │                    └──► [DNP] TLV755 1.2 V ──► FPGA core
   M1-4 ◄── 33R ◄── PE9/11/13/14 TIM1 CH1-4  ├──► FB1 ──► LP5907 ──► 3V3_IMU (IMU island only)
                                             │
 USB-C ── VBUS ──► B5819W ───────────────────┘  (USB alone powers 5V rail for bench work)
        └ D+/D- ─► USBLC6 ─► PA11/PA12 OTG_FS

                   ┌────────────── STM32H743VIT6 ──────────────┐
 ICM-42688-P ◄═SPI1═╡ PA5/6/7, CS PA4, DRDY PC4 (EXTI4)        │
 BMI270      ◄═SPI2═╡ PB13/14/15, CS PB12, DRDY PD10 (EXTI10)  │
 DPS310      ◄═I2C1═╡ PB8/PB9                                  │
 W25N01G     ◄═QSPI═╡ PB2, PB6, PD11, PD12, PE2, PD13          │
 ELRS (CRSF) ◄═U2═══╡ PD5/PD6                                  │
 Flow+ToF    ◄═U3═══╡ PD8/PD9                                  │
 Aux         ◄═U6/I2C2╡ PC6/PC7, PB10/PB11                     │
 iCE40UP5K   ◄═SPI3═╡ PC10/11/12, SS PA15, CRESET PD0,         │
   (DNP)            │ CDONE PD1, IRQ PD3 (EXTI3)               │
 SWD TC2030  ◄══════╡ PA13/PA14, SWO PB3, NRST                 │
 8 MHz XTAL  ◄══════╡ PH0/PH1                 BOOT0 ◄── SW1    │
 LEDs/Buzzer ◄══════╡ PE3, PE4, PD15 (AO3400), WS2812 PA0      │
 VBAT/5V sense ═════╡ PC0, PC5                                 │
                   └───────────────────────────────────────────┘
```

## 3. Power tree

| Rail | Source | Regulator | Feeds | Notes |
|---|---|---|---|---|
| VBAT | ESC J2 pin, 2S-6S (6.0-25.2 V nominal, spikes > 30 V) | — | buck, VBAT divider | SMBJ26A TVS + 3x 2.2 µF/50 V X7R at buck input. Bulk low-ESR electrolytic (≥ 470 µF, 35 V for 6S) goes **on the ESC battery pads**, not on the FC |
| 5V | VBAT | LMR36015 fixed 5 V, 1.5 A, 1 MHz (VERIFY variant suffix and inductor value) | receiver, flow, aux, buzzer, both 3V3 LDOs | Diode-OR'd with USB VBUS through B5819W so the board runs from USB alone |
| 3V3_D | 5V | LDL1117S33R (SOT-223, ceramic-stable) | MCU VDD (all pins), flash, baro, LEDs, FPGA VCCIO | VDDA through FB2 + 1 µF + 100 nF |
| 3V3_IMU | 5V via FB1 | LP5907-3.3 (6.5 µVrms) | ICM-42688-P VDD/VDDIO, BMI270 VDD/VDDIO | IMU island only. Nothing else on it |
| 1V2_FPGA | 3V3_D | TLV75512 (DNP) | iCE40UP5K VCC | Populate with U10. VPP_2V5 tied to 3V3_D (allowed range 1.71-3.46 V, VERIFY in UP5K datasheet) |

**Current / voltage budget** (estimates; MCU figure is for 480 MHz with most peripherals on, VERIFY against DS12110 §"Supply current characteristics"):

| Load | Rail | Typ (mA) | Max (mA) |
|---|---|---|---|
| STM32H743 @ 480 MHz (VOS0) | 3V3_D | 200 | 300 |
| W25N01G (program/erase) | 3V3_D | 15 | 35 |
| DPS310 + LEDs (3) + pull-ups | 3V3_D | 8 | 15 |
| iCE40UP5K IO (only when fitted) | 3V3_D | 5 | 20 |
| 1V2 FPGA core via TLV755 (only when fitted) | 3V3_D | 10 | 60 |
| **3V3_D total** | | **~240** | **~430** |
| ICM-42688-P + BMI270 (LN mode, both) | 3V3_IMU | 2 | 4 |
| ELRS receiver | 5V | 80 | 150 |
| Flow + ToF module | 5V | 70 | 120 |
| Aux connector budget | 5V | 0 | 200 |
| Buzzer | 5V | 0 | 30 |
| 3V3 LDOs (pass-through) | 5V | 245 | 435 |
| **5V total** | | **~400** | **~935** (LMR36015 1.5 A: 38 % margin) |
| Buck input at 2S min 6.4 V, 85 % eff. | VBAT | ~370 | ~860 |

**Thermal:** U7 dissipates (5.0 - 3.3) V × 0.43 A ≈ **0.73 W worst case** (about 0.4 W typical). In SOT-223 that needs ≥ 1 cm² of copper on the tab plus a via array to L2/L4. If bring-up shows U7 above 85 °C, the fallback is a small 3.3 V buck (e.g. TPS62162). Leave room for it on L4. The LP5907 dissipates < 10 mW.

**WS2812 LED strips** must be powered from their own BEC, not from this 5V budget.

## 4. IMU island and soft-mount

- Both IMUs sit inside a ~9 x 7 mm region with **its own copper island**: 3V3_IMU on L3 under the island, and an L2 GND that is part of the main plane but has **no other return currents routed through it**. Do not run a split ground; just keep motor/buck return paths physically away.
- Place both IMUs **≥ 10 mm from the buck inductor and switch node** and ≥ 8 mm from J2 and the DShot traces. No switching or high-di/dt traces on any layer under the island.
- Align both IMUs to the same axes if possible. Their orientation relative to the board arrow is recorded in `fw` board config.
- Decoupling per datasheets: ICM-42688-P has 100 nF on VDD and 100 nF on VDDIO, plus whatever additional bulk cap its reference schematic specifies (**VERIFY** values in the TDK datasheet). BMI270: 100 nF on VDD and VDDIO. Add 1 µF + 10 µF at the LP5907 output.
- **ICM-42688-P:** SPI mode 3, ≤ 24 MHz. INT1 → PC4. Optionally route CLKIN (pin 9, VERIFY) to a pad so a later rev can feed a 32 kHz reference.
- **BMI270:** SPI mode 3, ≤ 10 MHz. The first read after power-up is a dummy read that switches it into SPI mode. Firmware must upload the 8 KB config blob. INT1 → PD10.
- **Soft-mount:** the whole FC sits on M3 silicone grommets (board level). For v1 there is **no** separately soft-mounted IMU daughterboard. A board-level grommet mount plus the RPM-based notch filters is standard practice. As a stretch goal, the island is laid out so it could be cut out as a separate flex-connected module in v2. Leave a 1 mm keep-out ring around it.
- **Baro:** away from the IMU island and away from the USB-C opening. Firmware: put a dab of open-cell foam on top to block prop wash and light.

## 5. Decoupling and layout rules

1. **MCU:** 100 nF 0402 at **every** VDD pin, within 1 mm, with its via straight to L2. 1x 10 µF bulk near U1. **VCAP pins (48, 73): 2.2 µF each** (low ESR) to GND. VERIFY the pin numbers and whether the LQFP100 has a separate VDDLDO/VDD33USB pin (DS12110 pinout table). VDDA/VREF+: FB2 + 1 µF + 100 nF.
2. **Every IC** gets its datasheet decoupling with the smallest cap closest to the pin.
3. **Buck:** keep the hot loop (C_IN → VIN → SW → GND) as small as possible on L4. Put the inductor and switch node on L4, under the ESC-connector corner, and pour no copper under the SW node on L2/L3 beyond what is needed. Feedback/Vout sense goes at the output cap.
4. **Ground strategy:** one solid L2 GND plane, with no splits. Stitch GND vias every ~3 mm along board edges and around the buck. Battery/ESC return current enters at J2 GND and must reach the buck GND without crossing under the MCU or IMU island. Place J2 and the buck in the same corner.
5. **DShot traces (PE9/11/13/14 → J2):** 33 Ω series resistor at the MCU pin (tune 22-100 Ω on a scope for clean edges). Route them as a group over unbroken L2, ≤ 25 mm, ≥ 3x trace width from IMU/SPI traces. Bidirectional DShot means the ESC drives the line back: no RC filter and no fixed buffer on these lines. Use the MCU internal pull-up and no external pull-down. Optional 5 V-tolerant check: STM32H7 FT pins tolerate 5 V, but AM32 drives 3.3 V anyway.
6. **SPI (IMU) traces:** short, direct, over L2, not parallel to DShot or the buck. Add 0402 22 Ω series resistor pads on the SCK lines (populate 0 Ω first).
7. **QSPI:** keep the 6 lines length-matched to ±2 mm, which is easy at this size.
8. **USB:** 90 Ω differential pair, kept short. USBLC6 at the connector. CC1/CC2 each 5.1 kΩ to GND. Shield to GND via 1 MΩ ∥ 4.7 nF (or direct GND; JLC choice).
9. **ESD and robustness:** the receiver/flow/aux UART lines use 100 Ω series resistors (optional pads). The FC must survive a hot-plugged receiver.
10. **Crystal:** 8 MHz with load caps right next to it, a GND guard ring, and no traces under it on L1/L3.

## 6. Connectors

| Ref | Function | Part | Pinout (pin 1 first) |
|---|---|---|---|
| J2 | ESC (4-in-1) | JST-SH 8-pin SM08B-SRSS-TB | **VBAT, GND, CUR, TLM, M1, M2, M3, M4**. This is the common "Betaflight 4-in-1" order, but vendors differ (some put GND first). **VERIFY against the ESC you buy and its supplied cable before first power-up; a VBAT/GND swap destroys the board.** |
| J3 | Receiver (ELRS CRSF) | JST-SH 4-pin | 5V, GND, RX_TX (FC TX, PD5), RX_RX (FC RX, PD6) |
| J4 | Flow + ToF | JST-SH 4-pin | 5V, GND, FLOW_TX (PD8), FLOW_RX (PD9) |
| J5 | Aux | JST-SH 6-pin | 5V, GND, AUX_TX (PC6), AUX_RX (PC7), SCL2 (PB10), SDA2 (PB11) |
| J1 | USB-C | 16-pin USB2 receptacle | DFU, MSC/CDC, HIL link |
| J6 | SWD | Tag-Connect TC2030-NL footprint (no part) | 1 VCC (3V3_D), 2 SWDIO, 3 NRST, 4 SWCLK, 5 GND, 6 SWO (standard TC2030 ARM pinout, VERIFY) |
| J7 | SWD alt (DNP) | 1.27 mm 2x5 Cortex-Debug header | standard ARM 10-pin |
| SW1 | BOOT | SMD tactile to 3V3_D, BOOT0 has 10 kΩ pull-down | hold while plugging USB → ROM DFU |
| LED1-3 | PWR (green, on 3V3_D), STATUS (blue, PE3), WARN (red, PE4) | 0603 | 1 kΩ |
| BZ | Buzzer pads | external 5 V active buzzer | 5V and drain of AO3400 (gate PD15, 10 kΩ gate pull-down), 1N4148 across buzzer |
| LED_STRIP | WS2812 pad | pad | PA0 (TIM2_CH1); 5V comes from separate BEC |

## 7. Pin assignment (STM32H743VIT6, LQFP100)

Source of truth: `hw/pinmap.csv`. Checked by `hw/check_pinmap.py`. The table below was generated with `--markdown`.

**Design reasoning:**
- **Motors on TIM1 CH1-CH4 (PE9/PE11/PE13/PE14, AF1).** These are non-complementary channels, so input capture works for bidirectional DShot eRPM on the same pin. TIM1 has DMA requests for every channel and for UP. On the H7, DMAMUX1 routes any request to any DMA1/DMA2 stream, so the F4/F7 "fixed stream" conflicts **do not apply**. All four channels on one timer allow Betaflight-style DShot burst DMA (TIM1_UP + DMABURST) or per-channel DMA. PE8/PE10/PE12 (CH1N-CH3N) are deliberately avoided. Fallback if TIM1 misbehaves: TIM8 CH1-4 on PC6-PC9 (AF3, would displace USART6). The AF table already has these entries.
- **QSPI** uses the standard bank-1 set (CLK PB2, NCS PB6 at AF10, IO0-3 PD11/PD12/PE2/PD13). This is why the motors are not on TIM4 (PD12-PD15). The W25N01G is NAND, so firmware uses QUADSPI indirect mode only.
- **EXTI lines are unique** (4, 10, 3), and the checker enforces it because EXTIn is shared across ports.
- **SPI3 to the FPGA** on PC10-12 (AF6). The MCU configures the iCE40 over this same bus as an SPI slave (hold SS_B low while releasing CRESET_B, stream the bitstream, wait for CDONE). The same pins then serve as the runtime SPI link to the user design. No config flash is needed on the FC. With the FPGA unpopulated, the PA15 pull-up keeps the bus idle.
- **Spare pins** for later revisions: PA1-PA3, PA8, PA10, PB0, PB1, PB4, PB5, PB7, PC2, PC3, PC8, PC9, PC13-15, PD2, PD4, PD7, PD14, PE0, PE1, PE5, PE6, PE8, PE10, PE12, PE15.
- **Motor naming:** nets MOTOR0-3 follow the motor index in `docs/INTERFACES.md` §2 and are wired to ESC M1-M4 in order. The physical motor-to-index mapping depends on the ESC and frame, so it is a firmware remap table, not a PCB decision.

| LQFP | Pin | Signal | Peripheral | AF | Net | Conf. |
|---|---|---|---|---|---|---|
| 1 | PE2 | QUADSPI_BK1_IO2 | QUADSPI | 9 | QSPI_IO2 | medium |
| 2 | PE3 | LED_STATUS | GPIO | - | LED0 | high |
| 3 | PE4 | LED_WARN | GPIO | - | LED1 | high |
| 12 | PH0 | HSE_IN | RCC | - | HSE_IN | high |
| 13 | PH1 | HSE_OUT | RCC | - | HSE_OUT | high |
| 14 | NRST | NRST | SYS | - | NRST | medium |
| 15 | PC0 | ADC_VBAT | ADC123_INP10 | - | VBAT_SENSE | medium |
| 16 | PC1 | ADC_CURR | ADC123_INP11 | - | CURR_SENSE | medium |
| 22 | PA0 | LED_STRIP | TIM2_CH1 | 1 | LED_STRIP | high |
| 28 | PA4 | IMU_A_CS | GPIO | - | IMUA_CS | high |
| 29 | PA5 | SPI1_SCK | SPI1 | 5 | IMUA_SCK | high |
| 30 | PA6 | SPI1_MISO | SPI1 | 5 | IMUA_MISO | high |
| 31 | PA7 | SPI1_MOSI | SPI1 | 5 | IMUA_MOSI | high |
| 32 | PC4 | IMU_A_DRDY | EXTI4 | - | IMUA_INT1 | high |
| 33 | PC5 | ADC_5V | ADC12_INP8 | - | 5V_SENSE | medium |
| 36 | PB2 | QUADSPI_CLK | QUADSPI | 9 | QSPI_CLK | medium |
| 37 | PE7 | UART7_RX | UART7 | 7 | ESC_TLM | high |
| 39 | PE9 | MOTOR0 | TIM1_CH1 | 1 | MOTOR0 | high |
| 41 | PE11 | MOTOR1 | TIM1_CH2 | 1 | MOTOR1 | high |
| 43 | PE13 | MOTOR2 | TIM1_CH3 | 1 | MOTOR2 | high |
| 44 | PE14 | MOTOR3 | TIM1_CH4 | 1 | MOTOR3 | high |
| 46 | PB10 | I2C2_SCL | I2C2 | 4 | AUX_SCL | high |
| 47 | PB11 | I2C2_SDA | I2C2 | 4 | AUX_SDA | high |
| 51 | PB12 | IMU_B_CS | GPIO | - | IMUB_CS | high |
| 52 | PB13 | SPI2_SCK | SPI2 | 5 | IMUB_SCK | high |
| 53 | PB14 | SPI2_MISO | SPI2 | 5 | IMUB_MISO | high |
| 54 | PB15 | SPI2_MOSI | SPI2 | 5 | IMUB_MOSI | high |
| 55 | PD8 | USART3_TX | USART3 | 7 | FLOW_TX | high |
| 56 | PD9 | USART3_RX | USART3 | 7 | FLOW_RX | high |
| 57 | PD10 | IMU_B_DRDY | EXTI10 | - | IMUB_INT1 | high |
| 58 | PD11 | QUADSPI_BK1_IO0 | QUADSPI | 9 | QSPI_IO0 | medium |
| 59 | PD12 | QUADSPI_BK1_IO1 | QUADSPI | 9 | QSPI_IO1 | medium |
| 60 | PD13 | QUADSPI_BK1_IO3 | QUADSPI | 9 | QSPI_IO3 | medium |
| 62 | PD15 | BEEPER | GPIO | - | BEEPER | high |
| 63 | PC6 | USART6_TX | USART6 | 7 | AUX_TX | high |
| 64 | PC7 | USART6_RX | USART6 | 7 | AUX_RX | high |
| 68 | PA9 | USB_VBUS_SENSE | GPIO | - | VBUS_SENSE | high |
| 70 | PA11 | OTG_FS_DM | USB_OTG_FS | 10 | USB_DM | high |
| 71 | PA12 | OTG_FS_DP | USB_OTG_FS | 10 | USB_DP | high |
| 72 | PA13 | SWDIO | SYS | 0 | SWDIO | high |
| 76 | PA14 | SWCLK | SYS | 0 | SWCLK | high |
| 77 | PA15 | FPGA_CS | GPIO | - | FPGA_SS_B | high |
| 78 | PC10 | SPI3_SCK | SPI3 | 6 | FPGA_SCK | high |
| 79 | PC11 | SPI3_MISO | SPI3 | 6 | FPGA_MISO | high |
| 80 | PC12 | SPI3_MOSI | SPI3 | 6 | FPGA_MOSI | high |
| 81 | PD0 | FPGA_CRESET_B | GPIO | - | FPGA_CRESET_B | high |
| 82 | PD1 | FPGA_CDONE | GPIO | - | FPGA_CDONE | high |
| 84 | PD3 | FPGA_IRQ | EXTI3 | - | FPGA_IRQ | high |
| 86 | PD5 | USART2_TX | USART2 | 7 | RX_TX | high |
| 87 | PD6 | USART2_RX | USART2 | 7 | RX_RX | high |
| 89 | PB3 | SWO | SYS | 0 | SWO | high |
| 92 | PB6 | QUADSPI_BK1_NCS | QUADSPI | 10 | QSPI_NCS | medium |
| 94 | BOOT0 | BOOT0 | SYS | - | BOOT0 | medium |
| 95 | PB8 | I2C1_SCL | I2C1 | 4 | BARO_SCL | high |
| 96 | PB9 | I2C1_SDA | I2C1 | 4 | BARO_SDA | high |

**Confidence and open items (pins):**
- `high`: an assignment used on shipping H743 flight controllers or a standard ST AF that I am confident of.
- `medium`: QUADSPI AF numbers (especially PB6 = AF10), ADC input channel numbers (PC0 = INP10, PC1 = INP11, PC5 = INP8), and the BOOT0/NRST/VCAP **package pin numbers**. Check all of these in DS12110 Tables "Pin/ball definitions" and "Alternate function" before capture. An AF error on QSPI only costs the blackbox. An ADC channel error only changes a firmware constant.

## 8. Test points

1.0 mm round pads on the bottom side, each labelled on silkscreen:

| TP | Net | Expected |
|---|---|---|
| TP1 | VBAT | battery V |
| TP2 | 5V | 5.00 V ± 2 % |
| TP3 | 3V3_D | 3.30 V ± 2 % |
| TP4 | 3V3_IMU | 3.30 V ± 2 %, < 1 mV ripple (scope, 20 MHz BW) |
| TP5 | 1V2_FPGA | 1.20 V (only if U10/U11 fitted) |
| TP6 | VDDA | 3.30 V |
| TP7 | GND (x2, near TP2 and TP4) | probe ground |
| TP8 | SW node (buck) | 1 MHz square wave |
| TP9 | VBAT_SENSE | VBAT / 11 |
| TP10 | MOTOR0 (after series R) | DShot frames |
| TP11 | IMUA_INT1 | DRDY pulses at ODR |
| TP12 | IMUB_INT1 | DRDY pulses at ODR |
| TP13 | NRST | 3.3 V |
| TP14 | BOOT0 | 0 V (3.3 V with SW1 held) |

## 9. Bring-up checklist

Do these in order on each board. Record results per board serial in `hw/bringup/<serial>.md`.

1. **Visual / microscope:** check MCU and LGA orientation, solder bridges on LQFP pins, and that J2 is the correct way round and matches the ESC harness pinout.
2. **Unpowered resistance** to GND: VBAT > 10 kΩ, 5V > 1 kΩ, 3V3_D > 500 Ω, 3V3_IMU > 1 kΩ. Shorts → stop.
3. **Bench supply, current-limited** (7.4 V, 100 mA limit) into J2 VBAT/GND through the **smoke stopper**: expect < 60 mA with blank MCU. Measure TP1-TP6, TP8 (switching frequency), and TP4 ripple.
4. Raise to 25 V (6S max) for 10 s: rails stable, nothing hot. Feel/IR check U6, U7.
5. **USB only** (no battery): 5V rail ≈ 4.7 V through D1, board enumerates. Hold SW1 while plugging in: `dfu-util -l` shows "STM32 BOOTLOADER".
6. **SWD:** connect ST-Link via TC2030, `probe-rs info` / `st-flash` reads IDCODE (H743 DBGMCU_IDCODE DEV_ID 0x450). Flash an LED blink and check HSE start (MCO out on PA8 if needed: 8 MHz).
7. **Clock:** firmware reports SYSCLK 480 MHz, HSE-based PLL lock; set VOS0.
8. **IMU A:** WHO_AM_I (0x75) = **0x47**. DRDY at the configured ODR seen on TP11.
9. **IMU B:** CHIP_ID (0x00) = **0x24** after dummy read. Config blob loads, INTERNAL_STATUS ok. DRDY on TP12.
10. **Baro:** DPS310 PRODUCT_ID (0x0D) = **0x10**. Pressure plausible.
11. **Flash:** W25N01G JEDEC ID = **EF AA 21**. Erase/program/read back one block over QSPI.
12. **UARTs:** loopback each on J3/J4/J5. Then a real ELRS receiver binds and CRSF channels move.
13. **ADC:** VBAT reading vs. multimeter within 1 %. Calibrate the scale. CUR offset with ESC idle.
14. **DShot, no ESC:** logic analyzer on TP10/J2: DShot600 frames (1.67 µs bit) on all 4 channels, correct CRC.
15. **DShot with ESC, PROPS OFF**, smoke stopper inline: motors arm, spin in index order, eRPM telemetry (bidir) decodes on all four.
16. **Noise:** gyro noise floor, props off and motors spinning, compared with the commercial H743 FC from M3. Pass if within 1.5x.
17. **Thermal:** 10 min at max load on bench: U7 < 85 °C, MCU < 85 °C.
18. **(Only if FPGA fitted)** 1V2 rail ok, MCU configures iCE40 over SPI3, CDONE goes high, loopback register reads back.
19. Only then: tethered / net flight per PLAN §5.2.

## 10. Known uncertainties (honest list)

- No datasheets were open while writing this. The pin assignments marked `medium`, the power-pin numbering of the H743 LQFP100 (VCAP/VDD/VSS/VDD33USB), and all LCSC part numbers marked `verify` in `bom.csv` need checking.
- LMR36015 exact orderable suffix (fixed 5 V, 1 MHz) and its inductor/cap values. An acceptable alternative is any ≥ 40 V, ≥ 1.5 A synchronous buck that JLC stocks (e.g. LMR33630 with 36 V max, but that is tight for 6S spikes, so prefer ≥ 40 V).
- ESC 8-pin pinout varies by vendor (see J2).
- 36 x 36 mm / 30.5 mm mount: many 3" ducted frames only take 20x20 or 25.5x25.5 stacks. **Pick a frame that accepts 30.5 x 30.5** (see ORDER-LIST.md) or plan a 25.5 mm rev.
- ICM-42688-P decoupling details and CLKIN pin; iCE40UP5K SG48 SPI config pin numbers (SPI_SO/SI/SCK/SS_B, CRESET_B, CDONE) and VPP_2V5 handling.
- Current estimates for the H743 at 480 MHz are from memory. Measure them at bring-up step 3/17.
