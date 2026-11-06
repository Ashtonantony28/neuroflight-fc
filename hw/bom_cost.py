#!/usr/bin/env python3
"""Per-board cost roll-up for hw/bom.csv.

Sums populated parts (populate == yes) x qty x unit price, adds JLC-style
extended-part loading fees amortised over the build quantity, and an assumed
4-layer fab + assembly cost per board. All prices are rough estimates.

Usage: python3 hw/bom_cost.py [--boards N] [--fab-usd X] [--include-dnp]
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXT_FEE_USD = 3.00  # JLCPCB one-off loading fee per unique extended part per order (approx.)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", type=int, default=5)
    ap.add_argument("--fab-usd", type=float, default=110.0,
                    help="total 4-layer PCB fab + PCBA setup/stencil for the order (approx.)")
    ap.add_argument("--include-dnp", action="store_true", help="also cost DNP parts (FPGA etc.)")
    a = ap.parse_args(argv)

    with open(os.path.join(HERE, "bom.csv"), newline="") as f:
        rows = list(csv.DictReader(f))

    parts = dnp = 0.0
    ext = set()
    lines = []
    for r in rows:
        cost = int(r["qty"]) * float(r["unit_usd"])
        pop = r["populate"].strip().lower() == "yes"
        if pop or a.include_dnp:
            parts += cost
            if r["jlc_lib"] == "extended":
                ext.add(r["mpn"])
            lines.append((cost, r["ref"], r["mpn"]))
        else:
            dnp += cost
    lines.sort(reverse=True)
    ext_fee = len(ext) * EXT_FEE_USD
    per_board = parts + (ext_fee + a.fab_usd) / a.boards
    unverified = sum(1 for r in rows if r["lcsc_status"] == "verify")

    print(f"BOM lines: {len(rows)}, unverified LCSC numbers: {unverified}")
    print("Top cost drivers:")
    for c, ref, mpn in lines[:5]:
        print(f"  {ref:<10} {mpn:<28} ${c:6.2f}")
    print(f"Components per board          : ${parts:7.2f}")
    print(f"DNP parts (FPGA + LDO, if fit) : ${dnp:7.2f}")
    print(f"Extended-part fees ({len(ext)} x ${EXT_FEE_USD:.0f}) / {a.boards} boards: ${ext_fee / a.boards:7.2f}")
    print(f"Fab + assembly ${a.fab_usd:.0f} / {a.boards} boards      : ${a.fab_usd / a.boards:7.2f}")
    print(f"PER-BOARD TOTAL                : ${per_board:7.2f}  (order of {a.boards}: ${per_board * a.boards:.2f})")
    assert parts > 0 and per_board < 150, "cost outside plausible range - check bom.csv"
    return 0


if __name__ == "__main__":
    sys.exit(main())
