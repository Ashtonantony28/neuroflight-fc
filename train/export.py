"""Export a trained actor to `*.nfpol.json` (INTERFACES §6).

    python -m train.export runs/<name>/last.pt [--out runs/<name>/policy.nfpol.json]

The exported model is the deterministic actor (mean action):
    xn = (obs - obs_mean) / obs_std ;  h = tanh(W1 xn + b1) ; ... ; a = tanh(Wk h + bk)
Normalization stats are frozen at export. After writing, the JSON is loaded
back and a plain-numpy forward pass is checked against the torch actor (1e-5).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.env import ACT_DIM, OBS_DIM  # noqa: E402

FORMAT_VERSION = 1


def actor_from_state(st: dict):
    from train.ppo import Actor
    actor = Actor(tuple(st["actor_hidden"]))
    actor.load_state_dict(st["actor"])
    actor.eval()
    return actor


def to_nfpol(actor, obs_mean, obs_std, mode: str = "ctbr", meta: dict | None = None) -> dict:
    lins = [m for m in actor.net if isinstance(m, torch.nn.Linear)]
    layers = []
    for i, lin in enumerate(lins):
        layers.append({
            "W": lin.weight.detach().float().tolist(),
            "b": lin.bias.detach().float().tolist(),
            "act": "tanh",  # hidden tanh, and the final tanh squashing (§6)
        })
    d = {
        "version": FORMAT_VERSION,
        "mode": mode,
        "obs_mean": torch.as_tensor(obs_mean).float().tolist(),
        "obs_std": torch.as_tensor(obs_std).float().tolist(),
        "layers": layers,
    }
    if meta:
        d["meta"] = meta
    return d


def load_nfpol(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def numpy_forward(pol: dict, obs: np.ndarray) -> np.ndarray:
    """Reference float32 forward pass of an nfpol dict. obs: (..., 19)."""
    x = (np.asarray(obs, np.float32) - np.asarray(pol["obs_mean"], np.float32)) / np.asarray(pol["obs_std"], np.float32)
    for L in pol["layers"]:
        W = np.asarray(L["W"], np.float32)
        b = np.asarray(L["b"], np.float32)
        x = x @ W.T + b
        if L["act"] == "tanh":
            x = np.tanh(x)
        elif L["act"] != "none":
            raise ValueError(L["act"])
    return x.astype(np.float32)


def validate_nfpol(pol: dict) -> None:
    assert pol["version"] == FORMAT_VERSION and pol["mode"] in ("ctbr", "direct")
    assert len(pol["obs_mean"]) == OBS_DIM and len(pol["obs_std"]) == OBS_DIM
    assert all(s > 0 for s in pol["obs_std"])
    prev = OBS_DIM
    for L in pol["layers"]:
        W = np.asarray(L["W"])
        assert W.ndim == 2 and W.shape[1] == prev and len(L["b"]) == W.shape[0]
        assert L["act"] in ("tanh", "none")
        prev = W.shape[0]
    assert prev == ACT_DIM and pol["layers"][-1]["act"] == "tanh"


def check_roundtrip(path: str, actor, obs_mean, obs_std, n: int = 4096, seed: int = 0, tol: float = 1e-5) -> float:
    """Max |numpy(JSON) - torch actor| over random + realistic observations."""
    pol = load_nfpol(path)
    validate_nfpol(pol)
    g = torch.Generator().manual_seed(seed)
    mean, std = torch.as_tensor(obs_mean).float(), torch.as_tensor(obs_std).float()
    obs = mean + std * torch.randn(n, OBS_DIM, generator=g) * 2.0
    with torch.no_grad():
        ref = actor((obs - mean) / std).numpy()
    out = numpy_forward(pol, obs.numpy())
    err = float(np.abs(out - ref).max())
    if err > tol:
        raise AssertionError(f"export round-trip error {err:.3g} > {tol}")
    return err


def export_checkpoint(ckpt: str, out: str | None = None) -> str:
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    actor = actor_from_state(st)
    out = out or os.path.join(os.path.dirname(ckpt), "policy.nfpol.json")
    meta = {"source": os.path.relpath(ckpt, ROOT) if ckpt.startswith(ROOT) else ckpt,
            "iteration": st.get("iteration"), "curriculum_level": st.get("curriculum_level")}
    pol = to_nfpol(actor, st["obs_mean"], st["obs_std"], meta=meta)
    validate_nfpol(pol)
    with open(out, "w") as f:
        json.dump(pol, f)
    err = check_roundtrip(out, actor, st["obs_mean"], st["obs_std"])
    print(f"wrote {out} ({os.path.getsize(out)/1024:.1f} KiB), numpy-vs-torch max err {err:.2e}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    export_checkpoint(a.ckpt, a.out)


if __name__ == "__main__":
    main()
