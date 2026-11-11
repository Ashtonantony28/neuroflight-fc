"""Resumable multi-segment PPO driver (hover_ctbr_v3).

The training container is restarted roughly hourly, killing processes. This
driver splits each stage into segments: every segment is a `train.ppo` run
that checkpoints `last.pt` every `checkpoint_every` iterations; on (re)start
the driver counts the iterations already saved in each stage's segments
(`iteration` stored in last.pt) and resumes from the newest checkpoint with
`--init-from` for the remainder. Just rerun it after a restart:

    python -m train.v3_driver [--stages A,B] [--dry-run]

Stages: (name, config, iterations, overrides). Overrides are applied to the
config of the stage's FIRST segment only when given under `first`, to every
segment under `all` (e.g. a reward ramp only makes sense from iteration 0).
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.ppo import load_config, train  # noqa: E402

INIT = "runs/v2g/last.pt"   # hover_ctbr_v2
CFG = "train/config/hover_ctbr_v3.yaml"
STAGES = [
    # exploration std back to 0.1 for the new terms
    # (stage 1 ran with spatial noise on the position-error obs as well: spatial_exclude [15..18])
    ("v3s1", CFG, 400, {"first": {"reset_log_std": True}, "all": {"caps_spatial_exclude": [15, 16, 17, 18]}}),
    ("v3s2", CFG, 1200, {}),
]


def _saved_iters(path: str) -> int:
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False).get("iteration", 0))
    except Exception:
        return 0


def segments(name: str):
    segs = sorted(glob.glob(os.path.join(ROOT, "runs", f"{name}_seg*")), key=lambda p: int(p.rsplit("seg", 1)[1]))
    return [(s, os.path.join(s, "last.pt")) for s in segs]


def stage_state(name: str):
    done, ckpt = 0, None
    for d, lp in segments(name):
        if os.path.exists(lp):
            done += _saved_iters(lp)
            ckpt = lp
    return done, ckpt


def final_ckpt(name: str) -> str | None:
    return stage_state(name)[1]


def run(stages=None, dry=False):
    prev = os.path.join(ROOT, INIT)
    for name, cfg_path, iters, ov in STAGES:
        done, ckpt = stage_state(name)
        if stages and name not in stages:
            prev = ckpt or prev
            continue
        while done < iters:
            segs = segments(name)
            j = len(segs)
            init = ckpt or prev
            cfg = load_config(os.path.join(ROOT, cfg_path))
            for k, v in (ov.get("all") or {}).items():
                if k.startswith("caps_"):
                    cfg.setdefault("caps", {})[k[5:]] = copy.deepcopy(v)
                else:
                    cfg[k] = copy.deepcopy(v)
            if done == 0:
                for k, v in (ov.get("first") or {}).items():
                    cfg[k] = copy.deepcopy(v)
            cfg["init_from"] = init
            cfg["seed"] = int(cfg.get("seed", 0)) + 101 * j
            out = os.path.join(ROOT, "runs", f"{name}_seg{j}")
            print(f"[v3_driver] stage {name}: {done}/{iters} done, segment {j} from {os.path.relpath(init, ROOT)}"
                  f" for {iters - done} iterations -> {os.path.relpath(out, ROOT)}", flush=True)
            if dry:
                return
            train(cfg, out, iters - done)
            done, ckpt = stage_state(name)
        print(f"[v3_driver] stage {name} complete ({done} iterations): {ckpt}", flush=True)
        prev = ckpt or prev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=None, help="comma-separated stage names to run (others are skipped)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    run(a.stages.split(",") if a.stages else None, a.dry_run)


if __name__ == "__main__":
    main()
