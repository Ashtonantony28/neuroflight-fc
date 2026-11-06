"""Quantization accuracy, codegen/CRC, test-vector coverage, end-to-end C++ bit-exact pipeline."""
import os
import shutil
import struct
import subprocess
import zlib

import numpy as np
import pytest

import codegen
import gen_vectors
import golden
import pipeline
import quantize
from golden import F32
from random_policy import random_policy

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINED = os.path.join(ROOT, "train", "policies", "hover_ctbr_v1.nfpol.json")
HAVE_CMAKE = shutil.which("cmake") is not None and shutil.which("ctest") is not None


# ---------------------------------------------------------------- accuracy
@pytest.mark.parametrize("hidden,seed", [((64, 64), 0), ((64, 64), 1), ((64, 64), 7), ((32,), 2),
                                         ((128, 96, 32), 3), ((), 4)])
def test_int8_action_error_under_005(hidden, seed):
    pol = random_policy(hidden, seed=seed)
    pq = quantize.quantize_policy(pol)
    st = quantize.action_error(pq, quantize.synthetic_obs(pol, 20000, seed=99))
    assert st["max"] < 0.05, st
    assert st["mean"] < 0.01, st


def test_calibration_from_npy_samples(tmp_path, rand_pol):
    rng = np.random.default_rng(0)
    obs = quantize.synthetic_obs(rand_pol, 5000, seed=3)
    obs[:10] = np.nan  # non-finite rows are dropped from calibration
    p = tmp_path / "obs.npy"
    np.save(p, obs)
    pq = quantize.quantize_policy(rand_pol, np.load(p), calib_source="npy")
    assert pq["int8"]["calib"]["n"] == 4990
    assert quantize.action_error(pq, obs[10:])["max"] < 0.05
    # wider calibration data -> larger input scale
    pq2 = quantize.quantize_policy(rand_pol, obs[10:] * 3 - 2 * np.asarray(rand_pol["obs_mean"], F32))
    assert pq2["int8"]["s_in"] > pq["int8"]["s_in"]
    del rng


@pytest.mark.skipif(not os.path.exists(TRAINED), reason="trained policy not exported yet")
def test_trained_policy_error_under_005():
    pol = golden.load_policy(TRAINED)
    pq = quantize.quantize_policy(pol)
    st = quantize.action_error(pq, quantize.synthetic_obs(pol, 20000, seed=99))
    assert st["max"] < 0.05, st


def test_int8_block_structure(rand_pol_q):
    q = rand_pol_q["int8"]
    assert q["s_x_last"] == float(F32(1 / 127)) and q["s_in"] == float(F32(q["s_in"]))
    for L in q["layers"][:-1]:
        assert all(1 <= s <= 62 for s in L["S"]) and len(L["lut"]) == 256
        assert all(0 < m < 2 ** 31 for m in L["M"])
    assert "M" not in q["layers"][-1]


def test_rejects_non_tanh_hidden(rand_pol):
    pol = dict(rand_pol, layers=[dict(rand_pol["layers"][0], act="none")] + rand_pol["layers"][1:])
    with pytest.raises(ValueError):
        quantize.quantize_policy(pol)


# ---------------------------------------------------------------- codegen
def test_float_literals_roundtrip():
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.integers(0, 0x7F800000, 20000).astype(np.int32).view(F32),
                           np.array([0.0, -0.0, 1.0, -1.0, 1e-45, -1e-45, 3.4028235e38, 1 / 127], F32)])
    vals = np.concatenate([vals, -vals])
    for v in vals:
        s = codegen.flit(v)
        assert s.endswith("f") and any(c in s for c in ".eE")
        back = F32(float(s[:-1]))
        assert back.tobytes() == v.tobytes() or (v == 0 and back == 0)
    assert codegen.flit(-0.0) == "-0.0f"
    with pytest.raises(ValueError):
        codegen.flit(np.inf)


def test_crc_rule_independent(rand_pol_q):
    """CRC via struct.pack (independent of numpy dtypes) == codegen's numpy/zlib CRC."""
    def pk(fmt, xs):
        return struct.pack("<%d%s" % (len(xs), fmt), *xs)
    p = rand_pol_q
    c = zlib.crc32(pk("f", p["obs_mean"]))
    c = zlib.crc32(pk("f", p["obs_std"]), c)
    cf = c
    for L in p["layers"]:
        cf = zlib.crc32(pk("f", [w for r in L["W"] for w in r]), cf)
        cf = zlib.crc32(pk("f", L["b"]), cf)
    ci = c
    n = len(p["int8"]["layers"])
    for l, L in enumerate(p["int8"]["layers"]):
        ci = zlib.crc32(pk("b", [w for r in L["W_q"] for w in r]), ci)
        ci = zlib.crc32(pk("i", L["b_q"]), ci)
        if l + 1 < n:
            ci = zlib.crc32(pk("i", L["M"]), ci)
            ci = zlib.crc32(pk("B", L["S"]), ci)
            ci = zlib.crc32(pk("b", L["lut"]), ci)
        else:
            ci = zlib.crc32(pk("f", L["s_w"]), ci)
    assert cf == codegen.crc_float(p)
    assert ci == codegen.crc_int8(golden.Int8Arrays.from_policy(p))


def test_codegen_output_shape(rand_pol_q):
    hdr, cpp, crcs = codegen.generate(rand_pol_q, name="t")
    assert f"0x{crcs['crc_i8']:08X}u" in hdr and "extern const nf::Int8Model kPolicyI8;" in hdr
    assert "static const int8_t kQW0[1216]" in cpp and "static const float kSw2[4]" in cpp
    assert "{64, 4, kQW2, kQB2, nullptr, nullptr, nullptr, kSw2}" in cpp


# ---------------------------------------------------------------- vectors have teeth
def test_vectors_cover_edge_cases(rand_pol_q, tmp_path):
    p = tmp_path / "v.bin"
    info = gen_vectors.write_vectors(str(p), rand_pol_q, 10000)
    v = gen_vectors.read_vectors(str(p))
    assert v["N"] >= 10000 and info["nan_entries"] > 0 and info["x_q_saturated"] > 0
    q = golden.Int8Arrays.from_policy(rand_pol_q)
    obs = v["obs"].reshape(v["N"], -1)
    with np.errstate(all="ignore"):
        t = ((obs - q.obs_mean) / q.obs_std) / q.s_in
    with np.errstate(invalid="ignore"):
        frac = np.abs(t) - np.floor(np.abs(t))
    assert np.sum(frac == 0.5) > 2000                    # exact .5 ties
    trap = F32(0.5) - F32(2.0 ** -25)
    assert np.sum(np.abs(t) == trap) >= 10               # the 0.49999997 trap
    # A half-to-even golden (np.round) disagrees on many rows -> the C++ test would catch it.
    bad = np.clip(np.round(np.nan_to_num(t, nan=0.0)), -127, 127).astype(np.int8)
    assert np.sum(bad != v["x_q"].reshape(obs.shape)) > 1000
    # So does the literal float32 §11 formula, exactly on the trap entries.
    naive = np.clip(np.sign(t) * np.floor(np.abs(t) + F32(0.5)), -127, 127)
    naive = np.nan_to_num(naive, nan=0.0).astype(np.int8)
    diff = naive != v["x_q"].reshape(obs.shape)
    assert diff.sum() >= 10 and np.all(np.abs(t[diff]) == trap)
    # rshift cases include negatives and exact ties
    assert np.sum(v["rs_v"] < 0) > 500


# ---------------------------------------------------------------- end-to-end
@pytest.mark.skipif(not HAVE_CMAKE, reason="cmake not available")
@pytest.mark.parametrize("hidden", [(64, 64), (24, 40, 16), ()])
def test_pipeline_end_to_end(tmp_path, hidden):
    import json
    src = tmp_path / "p.nfpol.json"
    src.write_text(json.dumps(random_policy(hidden, seed=11)))
    rep = pipeline.pipeline(str(src), str(tmp_path / "out"), n=10000, jobs=os.cpu_count() or 4)
    assert rep["vectors"]["N"] == 10000
    assert rep["action_error"]["max"] < 0.05


@pytest.mark.skipif(not HAVE_CMAKE, reason="cmake not available")
def test_cpp_test_detects_corruption(tmp_path):
    """Flip one golden accumulator -> the C++ bit-exact test must fail."""
    out = tmp_path / "out"
    assert pipeline.main(["--random", "--out-dir", str(out), "--n", "10000"]) == 0
    exe = out / "build" / "test_policy_bitexact"
    v = (out / "vectors.bin").read_bytes()
    hdr = gen_vectors.HDR.unpack_from(v, 0)
    N, od = hdr[1], hdr[2]
    off = gen_vectors.HDR.size + N * od * 4 + N * od + 123 * 4
    acc = struct.unpack_from("<i", v, off)[0]
    bad = bytearray(v)
    struct.pack_into("<i", bad, off, acc + 1)
    (tmp_path / "bad.bin").write_bytes(bytes(bad))
    r = subprocess.run([str(exe), str(tmp_path / "bad.bin")], capture_output=True, text=True)
    assert r.returncode == 1 and "acc mismatches 1/" in r.stdout
    ok = subprocess.run([str(exe), str(out / "vectors.bin")], capture_output=True, text=True)
    assert ok.returncode == 0 and "PASSED" in ok.stdout
