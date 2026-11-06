#!/usr/bin/env python3
"""Generate bit-exact test vectors for the C++ policy runtime.

    python3 tools/gen_vectors.py policy.q.nfpol.json --out vectors.bin [--n 12000] [--seed 0]

Observation mix (rows): in-distribution, wide (4 sigma), uniform beyond the §4
ranges, saturating magnitudes (1e3..1e30, +-inf), NaN entries, rows whose
`xn / s_in` hits exact rounding ties (k + 0.5, and the +-0.49999997 trap),
rows on exact integer grid points, zeros and the mean.

Binary layout (little-endian), consumed by test/tools/cpp/test_policy_bitexact.cpp:
    char[8]  magic "NFVEC01\\0"
    u32      N, obs_dim, act_dim, crc_f32, crc_i8, K (rshift cases), reserved
    f32      obs   [N*obs_dim]
    i8       x_q   [N*obs_dim]
    i32      acc   [N*act_dim]      last-layer int32 accumulators (golden)
    f32      out_q [N*act_dim]      golden int8-path float output
    f32      out_f [N*act_dim]      numpy float32 reference of the float model
    i64      rs_v  [K]; i32 rs_s [K]; i64 rs_r [K]   rshift_round cases (Python-int reference)
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codegen import crc_float, crc_int8  # noqa: E402
from golden import Int8Arrays, forward_float, forward_int8, rshift_round_ref  # noqa: E402
from quantize import OBS_LIMITS  # noqa: E402

F32 = np.float32
MAGIC = b"NFVEC01\0"
HDR = struct.Struct("<8s7I")
TIE_TRAP = float(F32(0.5) - F32(2.0 ** -25))   # 0.49999997f


def _t_of(obs, mean, std, s_in):
    with np.errstate(all="ignore"):
        return ((obs - mean) / std) / s_in


def solve_obs_for_t(targets: np.ndarray, mean: np.float32, std: np.float32, s_in: np.float32,
                    span: int = 64) -> np.ndarray:
    """For each float32 target t find float32 obs with f32(f32(f32(obs-mean)/std)/s_in) == t.

    Returns obs (NaN where no solution was found within +-span ulps of the
    float64 estimate).
    """
    targets = np.asarray(targets, F32)
    o0 = (np.float64(mean) + np.float64(std) * np.float64(s_in) * targets.astype(np.float64)).astype(F32)
    bits = o0.view(np.int32).astype(np.int64)
    res = np.full(len(targets), np.nan, F32)
    offs = np.concatenate([[0], np.ravel(np.column_stack([np.arange(1, span + 1), -np.arange(1, span + 1)]))])
    for k in offs:
        todo = np.isnan(res)
        if not todo.any():
            break
        # step in the ordered-float sense (handles sign via int view for same-sign values)
        cand_bits = bits[todo] + np.where(o0[todo] >= 0, k, -k)
        cand = cand_bits.astype(np.int32).view(F32)
        ok = _t_of(cand, mean, std, s_in) == targets[todo]
        idx = np.flatnonzero(todo)[ok]
        res[idx] = cand[ok]
    return res


def tie_rows(q: Int8Arrays, rng, n_rows: int) -> tuple[np.ndarray, int]:
    """Rows where most entries land exactly on rounding ties after normalization."""
    od = len(q.obs_mean)
    ks = np.arange(0, 128)
    base = np.concatenate([ks + 0.5, [TIE_TRAP, 126.5, 127.5, 0.5, 1.5, 2.5]]).astype(F32)
    vals = np.concatenate([base, -base])
    rows = np.tile(q.obs_mean, (n_rows, 1)).astype(F32)
    hits = 0
    for i in range(od):
        t = rng.choice(vals, size=n_rows).astype(F32)
        o = solve_obs_for_t(t, q.obs_mean[i], q.obs_std[i], q.s_in)
        ok = ~np.isnan(o)
        rows[ok, i] = o[ok]
        hits += int(ok.sum())
    # force the trap value into the first rows explicitly (both signs)
    for s, r in ((1, 0), (-1, 1)):
        for i in range(od):
            o = solve_obs_for_t(np.array([s * TIE_TRAP], F32), q.obs_mean[i], q.obs_std[i], q.s_in)
            if not np.isnan(o[0]):
                rows[r, i] = o[0]
    return rows, hits


def grid_rows(q: Int8Arrays, rng, n_rows: int) -> np.ndarray:
    """Rows whose x_q is an exact integer chosen uniformly in [-127, 127]."""
    od = len(q.obs_mean)
    rows = np.tile(q.obs_mean, (n_rows, 1)).astype(F32)
    for i in range(od):
        t = rng.integers(-127, 128, n_rows).astype(F32)
        o = solve_obs_for_t(t, q.obs_mean[i], q.obs_std[i], q.s_in)
        ok = ~np.isnan(o)
        rows[ok, i] = o[ok]
    return rows


def make_obs(q: Int8Arrays, n: int, seed: int = 0) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    od = len(q.obs_mean)
    mean, std = q.obs_mean.astype(np.float64), q.obs_std.astype(np.float64)
    lim = np.array([l if l is not None else 20.0 for l in OBS_LIMITS]) if od == len(OBS_LIMITS) else np.full(od, 5.0)
    parts, info = [], {}

    def add(tag, a):
        parts.append(np.asarray(a, F32))
        info[tag] = len(a)

    add("zero_mean", np.stack([np.zeros(od), mean]))
    add("tie", (tr := tie_rows(q, rng, max(400, n // 20)))[0])
    info["tie_entries"] = tr[1]
    add("grid", grid_rows(q, rng, max(400, n // 20)))
    m = max(1, n // 25)
    sat = rng.choice([-1, 1], (m, od)) * 10.0 ** rng.uniform(3, 30, (m, od))
    sat[rng.random((m, od)) < 0.1] = np.inf
    sat[rng.random((m, od)) < 0.1] = -np.inf
    keep = rng.random((m, od)) < 0.3
    sat = np.where(keep, mean + std * rng.standard_normal((m, od)), sat)
    add("saturating", sat)
    nan = mean + std * rng.standard_normal((m, od))
    nan[rng.random((m, od)) < 0.3] = np.nan
    nan[0, :] = np.nan
    add("nan", nan)
    rest = n - sum(len(p) for p in parts)
    n_in = rest // 2
    n_wide = rest // 4
    n_beyond = rest - n_in - n_wide
    add("in_dist", np.clip(mean + std * rng.standard_normal((n_in, od)), -lim, lim))
    add("wide", mean + 4.0 * std * rng.standard_normal((n_wide, od)))
    add("beyond", rng.uniform(-3.0, 3.0, (n_beyond, od)) * lim)
    return np.concatenate(parts).astype(F32), info


def rshift_cases(rng, k: int = 4000):
    v, s = [], []
    edge_v = [0, 1, -1, 2, -2, 3, -3, 5, -5, 6, -6, 7, -7, 9, -9, 12, -12, 13, -13,
              2 ** 62, -(2 ** 62), -(2 ** 61), -(2 ** 61) - 1, (-(2 ** 31)) * (2 ** 31 - 1),
              (2 ** 31 - 1) ** 2, -(2 ** 31) * -(2 ** 31) // 2]
    for ev in edge_v:
        for ss in (1, 2, 3, 31, 40, 61, 62):
            v.append(ev); s.append(ss)
    while len(v) < k:
        a = int(rng.integers(-(2 ** 31), 2 ** 31))
        b = int(rng.integers(-(2 ** 31), 2 ** 31))
        ss = int(rng.integers(1, 63))
        prod = a * b
        if rng.random() < 0.25:   # exact half: v = (2m+1) * 2^(s-1) (ties)
            prod = (2 * int(rng.integers(-(2 ** 20), 2 ** 20)) + 1) << (ss - 1) if ss <= 40 else prod
        if rng.random() < 0.25:
            prod = int(rng.integers(-5000, 5000))
        v.append(prod); s.append(ss)
    r = [rshift_round_ref(a, b) for a, b in zip(v, s)]
    return np.array(v, np.int64), np.array(s, np.int32), np.array(r, np.int64)


def write_vectors(path: str, pol: dict, n: int = 12000, seed: int = 0) -> dict:
    q = Int8Arrays.from_policy(pol)
    obs, info = make_obs(q, n, seed)
    g = forward_int8(q, obs)
    f = forward_float(pol, obs)
    rv, rs, rr = rshift_cases(np.random.default_rng(seed + 1))
    N, od = obs.shape
    ad = g.last_acc.shape[1]
    cf, ci = crc_float(pol), crc_int8(q)
    with open(path, "wb") as fh:
        fh.write(HDR.pack(MAGIC, N, od, ad, cf, ci, len(rv), 0))
        for a, dt in ((obs, "<f4"), (g.x_q, "i1"), (g.last_acc, "<i4"), (g.out, "<f4"), (f, "<f4"),
                      (rv, "<i8"), (rs, "<i4"), (rr, "<i8")):
            fh.write(np.ascontiguousarray(a, dtype=dt).tobytes())
    xq = g.x_q.astype(np.int64)
    info.update({"N": int(N), "crc_f32": cf, "crc_i8": ci, "rshift_cases": int(len(rv)),
                 "x_q_saturated": int((np.abs(xq) == 127).sum()),
                 "nan_entries": int(np.isnan(obs).sum()),
                 "acc_absmax": int(np.abs(g.last_acc.astype(np.int64)).max())})
    return info


def read_vectors(path: str) -> dict:
    with open(path, "rb") as fh:
        buf = fh.read()
    magic, N, od, ad, cf, ci, K, _ = HDR.unpack_from(buf, 0)
    assert magic == MAGIC
    off = HDR.size
    out = {"N": N, "obs_dim": od, "act_dim": ad, "crc_f32": cf, "crc_i8": ci}
    for name, dt, cnt in (("obs", "<f4", N * od), ("x_q", "i1", N * od), ("acc", "<i4", N * ad),
                          ("out_q", "<f4", N * ad), ("out_f", "<f4", N * ad),
                          ("rs_v", "<i8", K), ("rs_s", "<i4", K), ("rs_r", "<i8", K)):
        a = np.frombuffer(buf, dtype=dt, count=cnt, offset=off)
        off += a.nbytes
        out[name] = a
    assert off == len(buf)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    with open(a.policy) as f:
        pol = json.load(f)
    info = write_vectors(a.out, pol, a.n, a.seed)
    print(f"wrote {a.out}: {json.dumps(info)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
