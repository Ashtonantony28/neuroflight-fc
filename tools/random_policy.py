#!/usr/bin/env python3
"""Write a random-init float policy in the §6 `*.nfpol.json` format (test fixture).

    python3 tools/random_policy.py out.nfpol.json [--hidden 64 64] [--seed 0]

Weights follow torch.nn.Linear's default init (U(-1/sqrt(in), 1/sqrt(in))),
optionally scaled by --gain; obs_mean/obs_std are plausible hover statistics.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

OBS_DIM, ACT_DIM = 19, 4


def plausible_norm_stats():
    mean = np.zeros(OBS_DIM)
    std = np.ones(OBS_DIM)
    mean[6], mean[10] = 1.0, 1.0           # R[:,0] ~ e_x, R[:,1] ~ e_y at hover
    std[0:3] = 0.4
    std[3:6] = 0.6
    std[6:12] = [0.05, 0.12, 0.12, 0.12, 0.05, 0.12]
    std[12:15] = [1.2, 1.2, 0.6]
    std[15:19] = 0.3
    mean[15] = -0.1
    return mean, std


def random_policy(hidden=(64, 64), seed: int = 0, gain: float = 1.0, obs_dim=OBS_DIM, act_dim=ACT_DIM) -> dict:
    rng = np.random.default_rng(seed)
    dims = [obs_dim, *hidden, act_dim]
    layers = []
    for i in range(len(dims) - 1):
        k = 1.0 / np.sqrt(dims[i])
        W = rng.uniform(-k, k, (dims[i + 1], dims[i])) * gain
        b = rng.uniform(-k, k, dims[i + 1])
        layers.append({"W": W.astype(np.float32).tolist(), "b": b.astype(np.float32).tolist(), "act": "tanh"})
    mean, std = plausible_norm_stats() if obs_dim == OBS_DIM else (np.zeros(obs_dim), np.ones(obs_dim))
    return {"version": 1, "mode": "ctbr",
            "obs_mean": mean.astype(np.float32).tolist(), "obs_std": std.astype(np.float32).tolist(),
            "layers": layers, "meta": {"source": "tools/random_policy.py", "seed": seed, "gain": gain}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--hidden", type=int, nargs="*", default=[64, 64])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gain", type=float, default=1.0)
    a = ap.parse_args(argv)
    with open(a.out, "w") as f:
        json.dump(random_policy(tuple(a.hidden), a.seed, a.gain), f)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
