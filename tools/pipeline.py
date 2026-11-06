#!/usr/bin/env python3
"""One-shot policy pipeline: export -> quantize -> codegen -> vectors -> build -> bit-exact test.

    python3 tools/pipeline.py train/policies/hover_ctbr_v1.nfpol.json [--calib obs.npy]
    python3 tools/pipeline.py runs/<name>/last.pt            # exports first (needs torch)
    python3 tools/pipeline.py --random                       # random-init 2x64 fixture

Outputs go to `--out-dir` (default runs/pipeline/<name>/):
    <name>.nfpol.json        float policy (export step; copied if input is JSON)
    <name>.q.nfpol.json      with int8 block
    gen/nf_policy_gen.{hpp,cpp}
    vectors.bin              golden vectors (tools/gen_vectors.py)
    build/                   fw/core configured with NF_GEN_POLICY_DIR (test_policy_bitexact)
    report.json              CRCs, vector mix, int8-vs-float action error
`--install DIR` additionally copies the generated pair into DIR (e.g. a
firmware source folder). Exit code 0 iff every step passes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
import codegen  # noqa: E402
import gen_vectors  # noqa: E402
import quantize  # noqa: E402

OBS_DIM, ACT_DIM = 19, 4


def step(msg):
    print(f"[pipeline] {msg}", flush=True)


def validate_float_policy(pol: dict) -> None:
    assert pol.get("mode") in ("ctbr", "direct"), "mode"
    assert len(pol["obs_mean"]) == OBS_DIM and len(pol["obs_std"]) == OBS_DIM, "obs dims"
    assert all(s > 0 for s in pol["obs_std"]), "obs_std must be > 0"
    prev = OBS_DIM
    for l, L in enumerate(pol["layers"]):
        W = np.asarray(L["W"])
        assert W.ndim == 2 and W.shape[1] == prev and len(L["b"]) == W.shape[0], f"layer {l} shape"
        assert W.shape[0] <= 128, f"layer {l} wider than nf::kMaxLayerWidth"
        assert np.all(np.isfinite(W)) and np.all(np.isfinite(L["b"])), f"layer {l} non-finite"
        assert L["act"] in ("tanh", "none"), f"layer {l} act"
        prev = W.shape[0]
    assert prev == ACT_DIM and pol["layers"][-1]["act"] == "tanh", "last layer must be 4-wide tanh"


def export(src: str, dst: str) -> dict:
    if src.endswith(".pt"):
        sys.path.insert(0, ROOT)
        from train.export import export_checkpoint  # torch import only when needed
        export_checkpoint(src, dst)
    elif src != dst:
        shutil.copyfile(src, dst)
    with open(dst) as f:
        pol = json.load(f)
    validate_float_policy(pol)
    return pol


def run(cmd, **kw):
    step("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def pipeline(src: str | None, out_dir: str | None = None, calib: str | None = None, n: int = 12000,
             seed: int = 0, jobs: int = 4, install: str | None = None, name: str | None = None) -> dict:
    t0 = time.time()
    if src is None:
        from random_policy import random_policy
        name = name or "random_2x64"
    else:
        name = name or os.path.basename(src).split(".")[0]
        if name in ("last", "best", "policy"):
            name = os.path.basename(os.path.dirname(os.path.abspath(src))) + "_" + name
    out_dir = os.path.abspath(out_dir or os.path.join(ROOT, "runs", "pipeline", name))
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, f"{name}.nfpol.json")
    qpath = os.path.join(out_dir, f"{name}.q.nfpol.json")
    gdir = os.path.join(out_dir, "gen")
    vpath = os.path.join(out_dir, "vectors.bin")
    bdir = os.path.join(out_dir, "build")

    step(f"1/6 export -> {fpath}")
    if src is None:
        with open(fpath, "w") as f:
            json.dump(random_policy(seed=seed), f)
    pol = export(src or fpath, fpath)

    step("2/6 quantize")
    cal = np.load(calib).astype(np.float32) if calib else None
    pol_q = quantize.quantize_policy(pol, cal, calib_source=f"npy:{os.path.basename(calib)}" if calib else "synthetic")
    with open(qpath, "w") as f:
        json.dump(pol_q, f)
    eval_obs = cal if cal is not None else quantize.synthetic_obs(pol, 20_000, seed=seed + 1)
    err = quantize.action_error(pol_q, eval_obs)
    step(f"   s_in={pol_q['int8']['s_in']:.6g}  int8-vs-float action |err| max {err['max']:.4g} "
         f"mean {err['mean']:.4g} p99 {err['p99']:.4g} (n={err['n']}, {'calib' if cal is not None else 'synthetic'})")

    step(f"3/6 codegen -> {gdir}")
    hdr, cpp, crcs = codegen.generate(pol_q, "nf_policy_gen", "nf_gen", name, os.path.basename(qpath))
    os.makedirs(gdir, exist_ok=True)
    for ext, txt in (("hpp", hdr), ("cpp", cpp)):
        with open(os.path.join(gdir, f"nf_policy_gen.{ext}"), "w") as f:
            f.write(txt)

    step(f"4/6 golden vectors (N={n}) -> {vpath}")
    vinfo = gen_vectors.write_vectors(vpath, pol_q, n, seed)

    step("5/6 build fw/core + test_policy_bitexact")
    run(["cmake", "-S", os.path.join(ROOT, "fw", "core"), "-B", bdir, f"-DNF_GEN_POLICY_DIR={gdir}",
         f"-DNF_GEN_POLICY_VECTORS={vpath}"], stdout=subprocess.DEVNULL)
    run(["cmake", "--build", bdir, "-j", str(jobs), "--target", "test_policy_bitexact"], stdout=subprocess.DEVNULL)

    step("6/6 bit-exact test (ctest -R policy_bitexact)")
    run(["ctest", "--test-dir", bdir, "-R", "^policy_bitexact$", "--output-on-failure", "-V"])

    if install:
        os.makedirs(install, exist_ok=True)
        for ext in ("hpp", "cpp"):
            shutil.copyfile(os.path.join(gdir, f"nf_policy_gen.{ext}"), os.path.join(install, f"nf_policy_gen.{ext}"))
        step(f"installed generated pair into {install}")

    report = {"name": name, "source": src or "random", "crc_f32": f"0x{crcs['crc_f32']:08X}",
              "crc_i8": f"0x{crcs['crc_i8']:08X}", "s_in": pol_q["int8"]["s_in"],
              "s_pre": [L.get("s_pre") for L in pol_q["int8"]["layers"][:-1]],
              "action_error": err, "vectors": vinfo, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    step(f"PASSED in {report['seconds']} s; report {os.path.join(out_dir, 'report.json')}")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", nargs="?", help="*.nfpol.json or a training checkpoint *.pt")
    ap.add_argument("--random", action="store_true", help="use a random-init 2x64 policy")
    ap.add_argument("--calib", help=".npy of raw observations [N,19] for calibration + error stats")
    ap.add_argument("--out-dir")
    ap.add_argument("--name")
    ap.add_argument("--n", type=int, default=12000, help="number of test vectors (>= 10000 recommended)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--install", help="copy nf_policy_gen.{hpp,cpp} into this directory on success")
    a = ap.parse_args(argv)
    if not a.policy and not a.random:
        ap.error("give a policy path or --random")
    try:
        pipeline(None if a.random else a.policy, a.out_dir, a.calib, a.n, a.seed, a.jobs, a.install, a.name)
    except subprocess.CalledProcessError as e:
        step(f"FAILED: {' '.join(e.cmd)} exited {e.returncode}")
        return 1
    except AssertionError as e:
        step(f"FAILED: invalid policy: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
