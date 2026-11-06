#!/usr/bin/env python3
"""Sanity checker for hw/pinmap.csv (STM32H743VIT6, LQFP100).

Checks:
  1. No GPIO / package pin is used twice.
  2. Every required function is assigned.
  3. Every alternate-function assignment matches the small embedded AF table
     below (the AF options this design relies on). The table is transcribed
     from memory of ST DS12110 (STM32H742/743/753/750 datasheet), "Alternate
     function" tables, and cross-checked against pin choices used by shipping
     Betaflight/INAV H743 targets. Entries marked 'medium' in pinmap.csv must
     still be verified against the datasheet before schematic sign-off.
  4. LQFP100 package pin numbers match the embedded pin-number table.
  5. EXTI lines are unique (on STM32 EXTIn is shared across ports) and match the
     pin number.
  6. DShot outputs: non-complementary channels (input capture for bidirectional
     DShot is impossible on CHxN), on a timer with DMA requests in DMAMUX1, and
     all on one timer (enables DShot burst DMA).
  7. ADC rows are on pins that have the listed ADC channel.

Usage: python3 hw/check_pinmap.py [--markdown]
Exit code 0 = pass.
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "pinmap.csv")

# LQFP100 package pin numbers for STM32H743Vx GPIO/dedicated pins we may use.
# (GPIO ordering is the standard ST LQFP100 ordering shared with F4/F7.)
LQFP100 = {
    "PE2": 1, "PE3": 2, "PE4": 3, "PE5": 4, "PE6": 5, "PC13": 7, "PC14": 8,
    "PC15": 9, "PH0": 12, "PH1": 13, "NRST": 14, "PC0": 15, "PC1": 16,
    "PC2": 17, "PC3": 18, "PA0": 22, "PA1": 23, "PA2": 24, "PA3": 25,
    "PA4": 28, "PA5": 29, "PA6": 30, "PA7": 31, "PC4": 32, "PC5": 33,
    "PB0": 34, "PB1": 35, "PB2": 36, "PE7": 37, "PE8": 38, "PE9": 39,
    "PE10": 40, "PE11": 41, "PE12": 42, "PE13": 43, "PE14": 44, "PE15": 45,
    "PB10": 46, "PB11": 47, "PB12": 51, "PB13": 52, "PB14": 53, "PB15": 54,
    "PD8": 55, "PD9": 56, "PD10": 57, "PD11": 58, "PD12": 59, "PD13": 60,
    "PD14": 61, "PD15": 62, "PC6": 63, "PC7": 64, "PC8": 65, "PC9": 66,
    "PA8": 67, "PA9": 68, "PA10": 69, "PA11": 70, "PA12": 71, "PA13": 72,
    "PA14": 76, "PA15": 77, "PC10": 78, "PC11": 79, "PC12": 80, "PD0": 81,
    "PD1": 82, "PD2": 83, "PD3": 84, "PD4": 85, "PD5": 86, "PD6": 87,
    "PD7": 88, "PB3": 89, "PB4": 90, "PB5": 91, "PB6": 92, "PB7": 93,
    "BOOT0": 94, "PB8": 95, "PB9": 96, "PE0": 97, "PE1": 98,
}

# (pin, signal) -> AF number. Only the options this design relies on, plus a
# few documented alternatives so a later re-assignment can be checked too.
AF_TABLE = {
    # SPI1 (AF5)
    ("PA5", "SPI1_SCK"): 5, ("PA6", "SPI1_MISO"): 5, ("PA7", "SPI1_MOSI"): 5,
    ("PD7", "SPI1_MOSI"): 5,
    # SPI2 (AF5)
    ("PB13", "SPI2_SCK"): 5, ("PB14", "SPI2_MISO"): 5, ("PB15", "SPI2_MOSI"): 5,
    # SPI3 (AF6 on PC10-12; AF6 SCK/MISO and AF7 MOSI on PB3-5)
    ("PC10", "SPI3_SCK"): 6, ("PC11", "SPI3_MISO"): 6, ("PC12", "SPI3_MOSI"): 6,
    ("PB3", "SPI3_SCK"): 6, ("PB4", "SPI3_MISO"): 6, ("PB5", "SPI3_MOSI"): 7,
    # I2C (AF4)
    ("PB8", "I2C1_SCL"): 4, ("PB9", "I2C1_SDA"): 4,
    ("PB6", "I2C1_SCL"): 4, ("PB7", "I2C1_SDA"): 4,
    ("PB10", "I2C2_SCL"): 4, ("PB11", "I2C2_SDA"): 4,
    # QUADSPI bank 1
    ("PB2", "QUADSPI_CLK"): 9, ("PB6", "QUADSPI_BK1_NCS"): 10,
    ("PD11", "QUADSPI_BK1_IO0"): 9, ("PD12", "QUADSPI_BK1_IO1"): 9,
    ("PE2", "QUADSPI_BK1_IO2"): 9, ("PD13", "QUADSPI_BK1_IO3"): 9,
    # TIM1 (AF1) on port E, TIM2/3/4/5 alternatives
    ("PE9", "TIM1_CH1"): 1, ("PE11", "TIM1_CH2"): 1, ("PE13", "TIM1_CH3"): 1,
    ("PE14", "TIM1_CH4"): 1, ("PE8", "TIM1_CH1N"): 1, ("PE10", "TIM1_CH2N"): 1,
    ("PE12", "TIM1_CH3N"): 1,
    ("PA0", "TIM2_CH1"): 1, ("PA0", "TIM5_CH1"): 2,
    ("PB0", "TIM3_CH3"): 2, ("PB1", "TIM3_CH4"): 2,
    ("PD12", "TIM4_CH1"): 2, ("PD13", "TIM4_CH2"): 2,
    ("PD14", "TIM4_CH3"): 2, ("PD15", "TIM4_CH4"): 2,
    ("PC6", "TIM8_CH1"): 3, ("PC7", "TIM8_CH2"): 3,
    ("PC8", "TIM8_CH3"): 3, ("PC9", "TIM8_CH4"): 3,
    # U(S)ARTs (AF7)
    ("PD5", "USART2_TX"): 7, ("PD6", "USART2_RX"): 7,
    ("PD8", "USART3_TX"): 7, ("PD9", "USART3_RX"): 7,
    ("PE8", "UART7_TX"): 7, ("PE7", "UART7_RX"): 7,
    ("PC6", "USART6_TX"): 7, ("PC7", "USART6_RX"): 7,
    ("PA9", "USART1_TX"): 7, ("PA10", "USART1_RX"): 7,
    # USB OTG FS (AF10)
    ("PA11", "OTG_FS_DM"): 10, ("PA12", "OTG_FS_DP"): 10,
    # Debug (AF0, system)
    ("PA13", "SWDIO"): 0, ("PA14", "SWCLK"): 0, ("PB3", "SWO"): 0,
}

# ADC channels relied on (pin -> set of ADC channel names)
ADC_TABLE = {
    "PC0": {"ADC123_INP10"}, "PC1": {"ADC123_INP11"}, "PC4": {"ADC12_INP4"},
    "PC5": {"ADC12_INP8"}, "PA0": {"ADC1_INP16"},
}

# Timers with DMA request lines in DMAMUX1 (TIM12/13/14 have none).
DMA_TIMERS = {"TIM1", "TIM2", "TIM3", "TIM4", "TIM5", "TIM8", "TIM15", "TIM16", "TIM17"}

# Pins that may only carry their dedicated function.
RESERVED = {"PA13": "SWDIO", "PA14": "SWCLK", "PH0": "HSE_IN", "PH1": "HSE_OUT",
            "BOOT0": "BOOT0", "NRST": "NRST"}

REQUIRED = [
    "SPI1_SCK", "SPI1_MISO", "SPI1_MOSI", "IMU_A_CS", "IMU_A_DRDY",
    "SPI2_SCK", "SPI2_MISO", "SPI2_MOSI", "IMU_B_CS", "IMU_B_DRDY",
    "I2C1_SCL", "I2C1_SDA",
    "QUADSPI_CLK", "QUADSPI_BK1_NCS", "QUADSPI_BK1_IO0", "QUADSPI_BK1_IO1",
    "QUADSPI_BK1_IO2", "QUADSPI_BK1_IO3",
    "MOTOR0", "MOTOR1", "MOTOR2", "MOTOR3",
    "USART2_TX", "USART2_RX",          # CRSF
    "USART3_TX", "USART3_RX",          # flow sensor
    "SPI3_SCK", "SPI3_MISO", "SPI3_MOSI", "FPGA_CS", "FPGA_CRESET_B",
    "FPGA_CDONE", "FPGA_IRQ",
    "ADC_VBAT", "ADC_CURR",
    "OTG_FS_DM", "OTG_FS_DP",
    "SWDIO", "SWCLK",
    "HSE_IN", "HSE_OUT", "BOOT0", "NRST",
    "LED_STATUS", "BEEPER",
]


def load(path=CSV_PATH):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check(rows):
    errors, warnings = [], []
    seen_pin, seen_num, seen_sig, exti_lines = {}, {}, {}, {}

    for r in rows:
        pin, sig, per = r["pin"], r["signal"], r["peripheral"]
        af = r["af"].strip()
        # 1. uniqueness
        if pin in seen_pin:
            errors.append(f"pin {pin} used twice: {seen_pin[pin]} and {sig}")
        seen_pin[pin] = sig
        num = int(r["lqfp_pin"])
        if num in seen_num:
            errors.append(f"package pin {num} used twice: {seen_num[num]} and {sig}")
        seen_num[num] = sig
        if sig in seen_sig:
            errors.append(f"signal {sig} assigned twice ({seen_sig[sig]}, {pin})")
        seen_sig[sig] = pin
        # 4. package pin number
        if pin not in LQFP100:
            errors.append(f"{pin}: not in LQFP100 pin table")
        elif LQFP100[pin] != num:
            errors.append(f"{pin}: CSV says pin {num}, table says {LQFP100[pin]}")
        # reserved pins
        if pin in RESERVED and RESERVED[pin] != sig:
            errors.append(f"{pin} is reserved for {RESERVED[pin]}, got {sig}")
        # 3. AF check
        if af:
            # motor rows carry the timer channel in 'peripheral'
            af_sig = per if sig.startswith("MOTOR") or sig == "LED_STRIP" else sig
            key = (pin, af_sig)
            if key not in AF_TABLE:
                errors.append(f"{pin}/{af_sig}: AF{af} not in embedded AF table")
            elif AF_TABLE[key] != int(af):
                errors.append(f"{pin}/{af_sig}: CSV AF{af}, table AF{AF_TABLE[key]}")
        # 5. EXTI
        m = re.fullmatch(r"EXTI(\d+)", per)
        if m:
            line = int(m.group(1))
            pin_no = int(re.sub(r"^P[A-H]", "", pin))
            if line != pin_no:
                errors.append(f"{pin}: EXTI{line} does not match pin number {pin_no}")
            if line in exti_lines:
                errors.append(f"EXTI{line} shared by {exti_lines[line]} and {pin}")
            exti_lines[line] = pin
        # 7. ADC
        if r["mode"] == "ANALOG":
            if per not in ADC_TABLE.get(pin, set()):
                errors.append(f"{pin}: {per} not in embedded ADC table")

    # 2. required
    for sig in REQUIRED:
        if sig not in seen_sig:
            errors.append(f"required signal {sig} not assigned")

    # 6. DShot
    motors = [r for r in rows if r["signal"].startswith("MOTOR")]
    timers = set()
    for r in motors:
        m = re.fullmatch(r"(TIM\d+)_CH(\d)(N?)", r["peripheral"])
        if not m:
            errors.append(f"{r['signal']}: peripheral {r['peripheral']} is not a timer channel")
            continue
        tim, _, n = m.groups()
        timers.add(tim)
        if n:
            errors.append(f"{r['signal']}: complementary channel cannot input-capture (bidir DShot)")
        if tim not in DMA_TIMERS:
            errors.append(f"{r['signal']}: {tim} has no DMA request")
    if len(motors) != 4:
        errors.append(f"expected 4 motor outputs, got {len(motors)}")
    if len(timers) > 1:
        warnings.append(f"motors span timers {sorted(timers)}: no single-timer burst DMA")

    medium = [r["pin"] + ":" + r["signal"] for r in rows if r["confidence"] != "high"]
    return errors, warnings, medium


def markdown(rows):
    out = ["| LQFP | Pin | Signal | Peripheral | AF | Net | Conf. |",
           "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: int(r["lqfp_pin"])):
        out.append(f"| {r['lqfp_pin']} | {r['pin']} | {r['signal']} | {r['peripheral']} | "
                   f"{r['af'] or '-'} | {r['net']} | {r['confidence']} |")
    return "\n".join(out)


def main():
    rows = load()
    if "--markdown" in sys.argv:
        print(markdown(rows))
        return 0
    errors, warnings, medium = check(rows)
    print(f"pinmap: {len(rows)} assignments, {len({r['pin'] for r in rows})} unique pins")
    for w in warnings:
        print("WARN:", w)
    print(f"needs datasheet verification ({len(medium)}): {', '.join(medium)}")
    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print("PASS: no pin conflicts, all required functions assigned, all AFs in table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
