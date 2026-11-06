#!/usr/bin/env python3
"""Post-training int8 quantization of a float `*.nfpol.json` (INTERFACES §6/§7/§11).

    python3 tools/quantize.py policy.nfpol.json [--calib obs.npy] [--out policy.q.nfpol.json]

Adds an `int8` block:
    {
      "version": 1,
      "s_in": f32, "s_x_last": f32,
      "calib": {...provenance...},
      "layers": [
        # hidden layer l:
        {"s_x": f32, "s_w": [out] f32, "W_q": [out][in] int8, "b_q": [out] int32,
         "s_pre": f32, "M": [out] int32, "S": [out] int in [1,62], "lut": [256] int8},
        ...
        # last layer:
        {"s_x": f32, "s_w": [out] f32, "W_q": ..., "b_q": ...}
      ]
    }

Scales:
  * weights: per-output-channel symmetric, s_w[j] = max|W[j,:]| / 127 (float32).
  * s_in: one scalar for all inputs, = percentile(|xn|, p_in=100) / 127 over the
    normalized calibration observations (`--calib` .npy of raw obs [N,19]),
    else over a synthetic set: obs = mean + std*N(0,1), clipped to the §4
    physical limits (pos +-2, vel +-5, R +-1, prev action +-1).
  * s_pre (per hidden layer): percentile(|z|, p) / 127 of the float
    pre-activations z on the same calibration set, capped at TANH_SAT/127
    (beyond |z|=4, 127*tanh(z) already rounds to 127, so a wider range only
    costs resolution).
  * M, S: M * 2^-S ~= s_x * s_w[j] / s_pre with the largest S in [1,62] such
    that M fits in int32 (=> |rel err| <= 2^-31 whenever M >= 2^30).
  * b_q[j] = round_half_away(b[j] / (s_x * s_w[j])), saturated to int32.
  * lut[k] = round_half_away(127 * tanh((k-128) * s_pre)).
All offline arithmetic is float64 from the float32-rounded scales; only the
integer tables and the float32 scales are part of the contract.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import (INT32_MAX, INT32_MIN, Int8Arrays, forward_float, forward_int8,  # noqa: E402
                    round_half_away_f64)

F32 = np.float32
INT8_BLOCK_VERSION = 1
TANH_SAT = 4.0
DEFAULT_PERCENTILE = 99.99      # s_pre (tanh saturates, so clipping is mild)
DEFAULT_PERCENTILE_IN = 100.0   # s_in: input clipping dominates worst-case action error
# §4 physical clip limits per observation index (None = unbounded)
OBS_LIMITS = [2.0] * 3 + [5.0] * 3 + [1.0] * 6 + [None] * 3 + [1.0] * 4


def f32(x) -> float:
    """Round to float32 and return as a Python float (exactly representable)."""
    return float(F32(x))


def choose_ms(ratio: float) -> tuple[int, int]:
    """Fixed-point (M, S): M*2^-S ~= ratio, M int32 > 0, S in [1, 62], max precision."""
    if not (ratio > 0.0 and np.isfinite(ratio)):
        raise ValueError(f"requant ratio must be positive finite, got {ratio}")
    for S in range(62, 0, -1):
        M = int(round_half_away_f64(ratio * 2.0 ** S))
        if M <= INT32_MAX:
            if M == 0:
                raise ValueError(f"requant ratio {ratio:.3g} too small for S<=62")
            return M, S
    raise ValueError(f"requant ratio {ratio:.3g} too large (needs S < 1)")


def tanh_lut(s_pre: float) -> list[int]:
    y = np.arange(-128, 128, dtype=np.float64)
    v = round_half_away_f64(127.0 * np.tanh(y * s_pre))
    return np.clip(v, -127, 127).astype(int).tolist()


def synthetic_obs(pol: dict, n: int = 200_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = np.asarray(pol["obs_mean"], np.float64)
    std = np.asarray(pol["obs_std"], np.float64)
    obs = mean + std * rng.standard_normal((n, len(mean)))
    if len(mean) == len(OBS_LIMITS):
        for i, lim in enumerate(OBS_LIMITS):
            if lim is not None:
                obs[:, i] = np.clip(obs[:, i], -lim, lim)
    return obs.astype(F32)


def quantize_weights(W: np.ndarray):
    """Per-output-channel symmetric int8. Returns (W_q int8, s_w float32)."""
    W = np.asarray(W, np.float64)
    amax = np.abs(W).max(axis=1)
    s_w = np.where(amax > 0, amax / 127.0, 1.0).astype(F32)   # all-zero row: any scale
    W_q = np.clip(round_half_away_f64(W / s_w.astype(np.float64)[:, None]), -127, 127).astype(np.int8)
    return W_q, s_w


def quantize_policy(pol: dict, calib_obs: np.ndarray | None = None, percentile: float = DEFAULT_PERCENTILE,
                    calib_source: str = "synthetic", tanh_sat: float = TANH_SAT,
                    percentile_in: float = DEFAULT_PERCENTILE_IN) -> dict:
    layers = pol["layers"]
    n = len(layers)
    for l, L in enumerate(layers):
        want = "tanh"
        if L["act"] != want:
            raise ValueError(f"layer {l}: act={L['act']!r}; the int8 path (§7) requires tanh on every layer")
    if calib_obs is None:
        calib_obs = synthetic_obs(pol)
        calib_source = "synthetic"
    calib_obs = np.asarray(calib_obs, F32)
    calib_obs = calib_obs[np.all(np.isfinite(calib_obs), axis=1)]
    if len(calib_obs) == 0:
        raise ValueError("no finite calibration observations")

    mean = np.asarray(pol["obs_mean"], F32)
    std = np.asarray(pol["obs_std"], F32)
    xn = ((calib_obs - mean) / std).astype(np.float64)
    r_in = float(np.percentile(np.abs(xn), percentile_in))
    r_in = max(r_in, 1e-3)
    s_in = f32(r_in / 127.0)

    out_layers = []
    s_x = s_in
    h = xn
    for l, L in enumerate(layers):
        W = np.asarray(L["W"], np.float64)
        b = np.asarray(L["b"], np.float64)
        W_q, s_w = quantize_weights(W)
        denom = s_x * s_w.astype(np.float64)
        bq = round_half_away_f64(b / denom)
        if np.any(bq > INT32_MAX) or np.any(bq < INT32_MIN):
            print(f"warning: layer {l} bias saturates int32", file=sys.stderr)
        bq = np.clip(bq, INT32_MIN, INT32_MAX).astype(np.int64)
        QL = {"s_x": s_x, "s_w": [float(v) for v in s_w], "W_q": W_q.tolist(), "b_q": bq.tolist()}
        z = h @ W.T + b
        if l + 1 < n:
            r_pre = float(np.percentile(np.abs(z), percentile))
            r_pre = min(max(r_pre, 1e-3), tanh_sat)
            s_pre = f32(r_pre / 127.0)
            MS = [choose_ms(s_x * float(sw) / s_pre) for sw in s_w]
            QL.update({"s_pre": s_pre, "M": [m for m, _ in MS], "S": [s for _, s in MS],
                       "lut": tanh_lut(s_pre)})
            s_x = f32(1.0 / 127.0)
        out_layers.append(QL)
        h = np.tanh(z)

    q = {
        "version": INT8_BLOCK_VERSION,
        "s_in": s_in,
        "s_x_last": out_layers[-1]["s_x"],
        "calib": {"source": calib_source, "n": int(len(calib_obs)), "percentile_in": percentile_in,
                  "percentile_pre": percentile,
                  "range_in": r_in, "tanh_sat": tanh_sat},
        "layers": out_layers,
    }
    res = dict(pol)
    res["int8"] = q
    return res


def action_error(pol_q: dict, obs: np.ndarray) -> dict:
    """int8-vs-float action error statistics on finite observations."""
    obs = np.asarray(obs, F32)
    r = forward_int8(Int8Arrays.from_policy(pol_q), obs)
    f = forward_float(pol_q, obs)
    e = np.abs(r.out.astype(np.float64) - f.astype(np.float64))
    return {"n": int(len(obs)), "max": float(e.max()), "mean": float(e.mean()),
            "p99": float(np.percentile(e, 99)), "per_action_max": e.max(axis=0).tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy")
    ap.add_argument("--calib", help=".npy of raw observations [N,19] for calibration")
    ap.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE, help="for s_pre")
    ap.add_argument("--percentile-in", type=float, default=DEFAULT_PERCENTILE_IN, help="for s_in")
    ap.add_argument("--out", help="output path (default: <stem>.q.nfpol.json next to input)")
    a = ap.parse_args(argv)
    with open(a.policy) as f:
        pol = json.load(f)
    calib = np.load(a.calib) if a.calib else None
    src = f"npy:{os.path.basename(a.calib)}" if a.calib else "synthetic"
    q = quantize_policy(pol, calib, a.percentile, src, percentile_in=a.percentile_in)
    out = a.out or (a.policy[:-len(".nfpol.json")] if a.policy.endswith(".nfpol.json") else a.policy) + ".q.nfpol.json"
    with open(out, "w") as f:
        json.dump(q, f)
    st = action_error(q, calib if calib is not None else synthetic_obs(pol, 20_000, seed=1))
    print(f"wrote {out}: s_in={q['int8']['s_in']:.6g}, "
          f"s_pre={[L.get('s_pre') for L in q['int8']['layers'][:-1]]}; "
          f"int8-vs-float action err max {st['max']:.4g} mean {st['mean']:.4g} p99 {st['p99']:.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
