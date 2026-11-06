#!/usr/bin/env python3
"""Integer golden model of the int8 policy (INTERFACES.md sections 7 and 11).

This is the Python side of the bit-exact contract with
`fw/core/src/policy.cpp::mlp_forward_int8`. Everything that the contract makes
bit-exact (quantized input `x_q`, every hidden activation, the last-layer
int32 accumulators) is computed here with explicit integer arithmetic:

  1. xn = (obs - mean) / std             float32, same op order as C++
     t  = xn / s_in                       float32
     x_q = clamp(round_half_away(t), -127, 127)   NaN -> 0
  2. acc = sum_i W_q[j,i] * x_q[i] + b_q[j]   int64, saturated to int32
  3. y = clamp(rshift_round(acc * M[j], S[j]), -128, 127)   int64 product,
     rshift_round(v, s) = (v + 2^(s-1)) >> s  with floor (arithmetic) shift
  4. x_next = lut[y + 128]
  5. last layer: out = tanh(((float32)acc * s_x_last) * s_w[j])   float32

Rounding note: `round_half_away` must equal C `std::round(float)`. The literal
formula in INTERFACES §11 (`sign(x)*floor(|x|+0.5)` evaluated in float32) is
wrong for exactly one class of inputs: |x| = 0.49999997 (= 0.5 - 2^-25), where
`|x| + 0.5` rounds up to 1.0 in float32. We evaluate `|x| + 0.5` in float64,
which is exact for every float32 |x| < 2^52 and therefore matches std::round.

Also provides the float32 reference forward (`forward_float`) of the
unquantized model.

Library use:
    from golden import load_policy, Int8Arrays, forward_int8, forward_float
CLI:
    python3 tools/golden.py policy.q.nfpol.json obs.npy [--out res.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np

INT32_MIN = -(2 ** 31)
INT32_MAX = 2 ** 31 - 1
F32 = np.float32


# --------------------------------------------------------------------------- primitives
def round_half_away(x) -> np.ndarray:
    """Round half away from zero, bit-identical to C `std::round` on float32.

    Input float32 (any shape). Returns float32. NaN propagates.
    """
    x = np.asarray(x, dtype=F32)
    a = np.abs(x).astype(np.float64)  # exact widening
    r = np.floor(a + 0.5)              # exact for a < 2^52; larger a are integers
    r = np.where(a >= 2.0 ** 52, a, r)
    return (np.sign(x).astype(np.float64) * r).astype(F32)


def round_half_away_f64(x) -> np.ndarray:
    """Round half away from zero for float64 values (offline quantizer use)."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize_input(t) -> np.ndarray:
    """t = xn / s_in (float32) -> int8 x_q. NaN -> 0, +-inf saturate."""
    t = np.asarray(t, dtype=F32)
    t = np.where(np.isnan(t), F32(0), t)
    r = round_half_away(t)
    return np.clip(r, -127, 127).astype(np.int8)


def rshift_round(v, s) -> np.ndarray:
    """(v + 2^(s-1)) >> s on int64 with floor semantics (numpy >> is arithmetic).

    s in [1, 62]. Caller guarantees |v| <= 2^62 (int32*int32) so no overflow.
    """
    v = np.asarray(v, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    if np.any(s < 1) or np.any(s > 62):
        raise ValueError("shift out of [1, 62]")
    return (v + (np.int64(1) << (s - 1))) >> s


def rshift_round_ref(v: int, s: int) -> int:
    """Pure-Python-int reference (unbounded ints, >> is floor division)."""
    return (int(v) + (1 << (int(s) - 1))) >> int(s)


def sat_int32(v) -> np.ndarray:
    return np.clip(np.asarray(v, dtype=np.int64), INT32_MIN, INT32_MAX)


# --------------------------------------------------------------------------- model containers
@dataclass
class QLayer:
    W_q: np.ndarray            # int8  [out, in]
    b_q: np.ndarray            # int32 [out]
    s_w: np.ndarray            # float32 [out]
    M: np.ndarray | None       # int32 [out]   (hidden)
    S: np.ndarray | None       # uint8 [out]   (hidden)
    lut: np.ndarray | None     # int8 [256]    (hidden)
    s_pre: float | None = None


@dataclass
class Int8Arrays:
    obs_mean: np.ndarray   # float32 [obs]
    obs_std: np.ndarray    # float32 [obs]
    s_in: np.float32
    s_x_last: np.float32
    layers: list

    @staticmethod
    def from_policy(pol: dict) -> "Int8Arrays":
        if "int8" not in pol:
            raise ValueError("policy has no int8 block; run tools/quantize.py first")
        q = pol["int8"]
        n = len(q["layers"])
        layers = []
        for l, L in enumerate(q["layers"]):
            hidden = l + 1 < n
            layers.append(QLayer(
                W_q=np.asarray(L["W_q"], dtype=np.int8),
                b_q=np.asarray(L["b_q"], dtype=np.int64).astype(np.int32),
                s_w=np.asarray(L["s_w"], dtype=F32),
                M=np.asarray(L["M"], dtype=np.int64).astype(np.int32) if hidden else None,
                S=np.asarray(L["S"], dtype=np.uint8) if hidden else None,
                lut=np.asarray(L["lut"], dtype=np.int8) if hidden else None,
                s_pre=L.get("s_pre"),
            ))
        return Int8Arrays(
            obs_mean=np.asarray(pol["obs_mean"], dtype=F32),
            obs_std=np.asarray(pol["obs_std"], dtype=F32),
            s_in=F32(q["s_in"]), s_x_last=F32(q["s_x_last"]), layers=layers)


def load_policy(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- forward passes
@dataclass
class GoldenResult:
    x_q: np.ndarray        # int8  [N, obs]
    last_acc: np.ndarray   # int32 [N, act]
    out: np.ndarray        # float32 [N, act]
    hidden: list           # int8 activations per hidden layer [N, width]


def forward_int8(q: Int8Arrays, obs) -> GoldenResult:
    obs = np.atleast_2d(np.asarray(obs, dtype=F32))
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        xn = (obs - q.obs_mean) / q.obs_std   # float32 elementwise
        t = xn / q.s_in                       # float32 / float32 scalar
    x_q = quantize_input(t)
    x = x_q.astype(np.int64)
    hidden = []
    n = len(q.layers)
    for l, L in enumerate(q.layers):
        acc = sat_int32(x @ L.W_q.astype(np.int64).T + L.b_q.astype(np.int64))
        if l + 1 < n:
            v = rshift_round(acc * L.M.astype(np.int64), L.S.astype(np.int64))
            y = np.clip(v, -128, 127)
            a = L.lut[(y + 128).astype(np.intp)]
            hidden.append(a)
            x = a.astype(np.int64)
        else:
            last_acc = acc.astype(np.int32)
            z = (last_acc.astype(F32) * q.s_x_last) * L.s_w   # float32, C++ op order
            out = np.tanh(z).astype(F32)
    return GoldenResult(x_q=x_q, last_acc=last_acc, out=out, hidden=hidden)


def forward_float(pol: dict, obs) -> np.ndarray:
    """float32 reference forward of the unquantized model (same as train/export.py)."""
    obs = np.atleast_2d(np.asarray(obs, dtype=F32))
    with np.errstate(invalid="ignore", over="ignore"):
        x = (obs - np.asarray(pol["obs_mean"], F32)) / np.asarray(pol["obs_std"], F32)
        for L in pol["layers"]:
            x = x @ np.asarray(L["W"], F32).T + np.asarray(L["b"], F32)
            if L["act"] == "tanh":
                x = np.tanh(x)
            elif L["act"] != "none":
                raise ValueError(f"unknown act {L['act']}")
    return x.astype(F32)


def forward_float_seq(pol: dict, obs) -> np.ndarray:
    """float32 forward with the exact C++ summation order (acc = b; acc += w*x).

    Slow-ish (loops over the input dim) but useful to confirm the C++ float
    path to ~1 ulp.
    """
    obs = np.atleast_2d(np.asarray(obs, dtype=F32))
    with np.errstate(invalid="ignore", over="ignore"):
        x = (obs - np.asarray(pol["obs_mean"], F32)) / np.asarray(pol["obs_std"], F32)
        for L in pol["layers"]:
            W = np.asarray(L["W"], F32)
            acc = np.broadcast_to(np.asarray(L["b"], F32), (x.shape[0], W.shape[0])).copy()
            for i in range(W.shape[1]):
                acc = acc + W[:, i][None, :] * x[:, i:i + 1]
            x = np.tanh(acc) if L["act"] == "tanh" else acc
    return x.astype(F32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", help="quantized *.nfpol.json (with int8 block)")
    ap.add_argument("obs", help=".npy of observations [N, 19]")
    ap.add_argument("--out", help="write x_q / last_acc / out_q / out_f to this .npz")
    a = ap.parse_args(argv)
    pol = load_policy(a.policy)
    obs = np.load(a.obs).astype(F32)
    r = forward_int8(Int8Arrays.from_policy(pol), obs)
    f = forward_float(pol, obs)
    err = np.abs(r.out - f)
    ok = np.isfinite(err)
    print(f"N={len(obs)}  int8-vs-float |err|: max {err[ok].max():.4g}  mean {err[ok].mean():.4g}  "
          f"p99 {np.percentile(err[ok], 99):.4g}")
    if a.out:
        np.savez(a.out, x_q=r.x_q, last_acc=r.last_acc, out_q=r.out, out_f=f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
