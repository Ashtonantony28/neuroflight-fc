"""Rounding, rshift_round, M/S fixed point, LUT and golden-model self-consistency."""
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pytest

import golden
import quantize
from golden import F32, INT32_MAX, INT32_MIN

TRAP = F32(0.5) - F32(2.0 ** -25)  # 0.49999997f


def py_round_half_away(x: float) -> float:
    # Decimal ROUND_HALF_UP rounds half away from zero; exact for any float.
    return float(Decimal(float(x)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------- input rounding
@pytest.mark.parametrize("x,want", [(0.5, 1), (-0.5, -1), (1.5, 2), (-1.5, -2), (2.5, 3), (-2.5, -3),
                                    (126.5, 127), (-126.5, -127), (0.0, 0), (-0.0, 0), (0.49999997, 0),
                                    (-0.49999997, 0), (1.4999999, 1), (127.5, 128), (-127.5, -128)])
def test_round_half_away_ties(x, want):
    assert golden.round_half_away(F32(x)) == want


def test_round_half_away_matches_decimal_on_random_and_ties():
    rng = np.random.default_rng(0)
    xs = np.concatenate([
        rng.uniform(-200, 200, 20000).astype(F32),
        (np.arange(-300, 300) + F32(0.5)).astype(F32),
        np.array([TRAP, -TRAP, np.nextafter(F32(0.5), F32(1)), np.nextafter(F32(1.5), F32(0))], F32),
        # random bit patterns in a sane exponent range
        rng.integers(0x30000000, 0x4B800000, 5000).astype(np.int32).view(F32),
    ]).astype(F32)
    got = golden.round_half_away(xs)
    want = np.array([py_round_half_away(x) for x in xs], np.float64)
    assert np.array_equal(got.astype(np.float64), want)


def test_spec_literal_formula_differs_only_at_trap():
    """Documents the §11 ambiguity: sign*floor(|x|+0.5) in float32 rounds 0.49999997 to 1."""
    x = np.array([TRAP, -TRAP], F32)
    naive = np.sign(x) * np.floor(np.abs(x) + F32(0.5))
    assert naive.dtype == F32 and list(naive) == [1.0, -1.0]
    assert list(golden.round_half_away(x)) == [0.0, 0.0]   # == std::round


def test_quantize_input_nan_inf_clamp():
    t = np.array([np.nan, np.inf, -np.inf, 200.0, -200.0, 127.4, -127.4, 126.5, -126.5, 0.5, -0.5], F32)
    assert golden.quantize_input(t).tolist() == [0, 127, -127, 127, -127, 127, -127, 127, -127, 1, -1]


# ---------------------------------------------------------------- rshift_round
def test_rshift_round_negatives_vs_python_reference():
    rng = np.random.default_rng(1)
    for s in range(1, 63):
        a = rng.integers(INT32_MIN, INT32_MAX, 500, endpoint=True)
        b = rng.integers(INT32_MIN, INT32_MAX, 500, endpoint=True)
        v = np.concatenate([a * b, -np.abs(a), np.arange(-50, 50)]).astype(np.int64)
        got = golden.rshift_round(v, np.full(v.shape, s))
        want = [golden.rshift_round_ref(int(x), s) for x in v]
        assert got.tolist() == want, f"s={s}"


@pytest.mark.parametrize("v,s,want", [(-1, 1, 0), (-2, 1, -1), (-3, 1, -1), (-5, 1, -2), (5, 1, 3),
                                      (-6, 2, -1), (-7, 2, -2), (-12, 3, -1), (-13, 3, -2),
                                      (-(2 ** 61), 62, 0), (-(2 ** 61) - 1, 62, -1), (2 ** 62, 62, 1)])
def test_rshift_round_known(v, s, want):
    assert golden.rshift_round(v, s) == want == golden.rshift_round_ref(v, s)


def test_rshift_round_rejects_bad_shift():
    with pytest.raises(ValueError):
        golden.rshift_round(1, 0)
    with pytest.raises(ValueError):
        golden.rshift_round(1, 63)


# ---------------------------------------------------------------- M / S
def test_choose_ms_error_bounds():
    for r in np.logspace(-9, 8.9, 3000):
        M, S = quantize.choose_ms(float(r))
        assert 1 <= S <= 62 and 0 < M <= INT32_MAX
        rel = abs(M * 2.0 ** -S - r) / r
        if M >= 2 ** 30:
            assert rel <= 2.0 ** -31, (r, M, S, rel)
        else:  # only when S hit 62 (tiny ratios)
            assert S == 62 and rel <= 0.5 / M
    # realistic ratios always get full precision
    for r in np.logspace(-6, 3, 500):
        M, _ = quantize.choose_ms(float(r))
        assert M >= 2 ** 30


@pytest.mark.parametrize("r", [0.0, -1.0, float("nan"), float("inf"), 2.0 ** 31, 1e-25])
def test_choose_ms_rejects(r):
    with pytest.raises(ValueError):
        quantize.choose_ms(r)


def test_requant_matches_real_scaling(rand_pol_q):
    """Integer requant of an accumulator ~= acc * s_x*s_w/s_pre within 0.5 LSB (+tiny)."""
    q = golden.Int8Arrays.from_policy(rand_pol_q)
    rng = np.random.default_rng(2)
    for l, L in enumerate(q.layers[:-1]):
        QL = rand_pol_q["int8"]["layers"][l]
        ratio = QL["s_x"] * np.asarray(QL["s_w"], np.float64) / QL["s_pre"]
        acc = rng.integers(-200000, 200000, (1000, len(ratio)))
        y = golden.rshift_round(acc * q.layers[l].M.astype(np.int64), q.layers[l].S.astype(np.int64))
        assert np.max(np.abs(y - acc * ratio)) <= 0.5 + 1e-4


# ---------------------------------------------------------------- weights / LUT
def test_weight_quant_per_channel(rand_pol_q, rand_pol):
    for L, QL in zip(rand_pol["layers"], rand_pol_q["int8"]["layers"]):
        W = np.asarray(L["W"], np.float64)
        Wq = np.asarray(QL["W_q"])
        sw = np.asarray(QL["s_w"], np.float64)
        assert np.abs(Wq).max(axis=1).tolist() == [127] * len(sw)   # full range per channel
        assert np.max(np.abs(Wq * sw[:, None] - W)) <= 0.5 * sw.max() * (1 + 1e-6)
        assert np.all(sw == sw.astype(F32))


def test_zero_weight_row_ok():
    Wq, sw = quantize.quantize_weights(np.array([[0.0, 0.0], [0.5, -1.0]]))
    assert Wq.tolist() == [[0, 0], [64, -127]] and sw[0] > 0


def test_lut_shape_symmetry():
    for s_pre in (0.005, 0.0185, 4.0 / 127):
        lut = np.array(quantize.tanh_lut(s_pre))
        assert len(lut) == 256 and lut[128] == 0
        assert np.all(np.diff(lut) >= 0)
        assert np.array_equal(lut[1:], -lut[1:][::-1])   # odd symmetry for y in [-127,127]
        assert lut.min() >= -127 and lut.max() <= 127
    assert quantize.tanh_lut(4.0 / 127)[-1] == 127


# ---------------------------------------------------------------- golden vs scalar Python-int reference
def scalar_reference(pol_q, obs_row):
    """Independent per-element reference using Python ints for all integer math."""
    q = pol_q["int8"]
    mean = np.asarray(pol_q["obs_mean"], F32)
    std = np.asarray(pol_q["obs_std"], F32)
    s_in = F32(q["s_in"])
    xq = []
    for i, o in enumerate(np.asarray(obs_row, F32)):
        with np.errstate(all="ignore"):
            t = F32(F32(F32(o - mean[i]) / std[i]) / s_in)
        if np.isnan(t):
            xq.append(0)
            continue
        # clamp-then-round == round-then-clamp for integer bounds; keeps Decimal in range
        xq.append(int(py_round_half_away(max(-200.0, min(200.0, float(t))))) if abs(t) < 200 else (127 if t > 0 else -127))
        xq[-1] = max(-127, min(127, xq[-1]))
    x = xq
    n = len(q["layers"])
    for l, L in enumerate(q["layers"]):
        accs = []
        for j, row in enumerate(L["W_q"]):
            a = sum(int(w) * int(xi) for w, xi in zip(row, x)) + int(L["b_q"][j])
            accs.append(max(INT32_MIN, min(INT32_MAX, a)))
        if l + 1 < n:
            x = [L["lut"][max(-128, min(127, golden.rshift_round_ref(a * L["M"][j], L["S"][j]))) + 128]
                 for j, a in enumerate(accs)]
    return xq, accs


def test_golden_matches_scalar_reference(rand_pol_q):
    import gen_vectors
    q = golden.Int8Arrays.from_policy(rand_pol_q)
    obs, _ = gen_vectors.make_obs(q, 2000, seed=5)
    idx = np.random.default_rng(3).choice(len(obs), 300, replace=False)
    idx = np.concatenate([np.arange(5), idx])
    r = golden.forward_int8(q, obs[idx])
    for k, i in enumerate(idx):
        xq, acc = scalar_reference(rand_pol_q, obs[i])
        assert r.x_q[k].tolist() == xq
        assert r.last_acc[k].tolist() == acc


def test_float_reference_matches_sequential_order(rand_pol):
    rng = np.random.default_rng(4)
    obs = (np.asarray(rand_pol["obs_mean"]) + np.asarray(rand_pol["obs_std"]) * rng.standard_normal((500, 19))).astype(F32)
    a = golden.forward_float(rand_pol, obs)
    b = golden.forward_float_seq(rand_pol, obs)
    assert np.max(np.abs(a - b)) < 2e-6
