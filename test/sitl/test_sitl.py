"""WS6 SITL: the real firmware core (fw/core via libnf_sitl) flying in the Python sim.

Scenarios live in fw/sitl/scenarios.py; fw/sitl/report.py runs the same suite
and writes docs/sitl-report.md.
"""
import math

import numpy as np
import pytest

import scenarios as S
import sitl
from sitl import FALLBACK_HOLD, MODE_NAMES, MOTOR_STOP, POLICY


# ---------------------------------------------------------------- bridge basics
def test_abi_and_policy_load_crc_forward():
    fw = sitl.Firmware(2)
    for pol in (sitl.linear_hover_policy(), sitl.random_policy(3)):
        ok, crc = fw.set_policy(pol)
        assert ok
        assert crc == sitl.nfpol_crc32(pol), "C++ policy_crc32 != Python zlib CRC rule"
        rng = np.random.default_rng(0)
        for _ in range(50):
            obs = rng.normal(0, 1, 19).astype(np.float32)
            np.testing.assert_allclose(fw.policy_forward(obs, env=1), sitl.numpy_forward(pol, obs), atol=1e-5)  # float32 sum order (export.py uses 1e-5)
    # corruption (poke) changes the output but not the CRC verdict made at load time
    obs = np.zeros(19, np.float32)
    before = fw.policy_forward(obs)
    j = int(np.argmin(np.abs(before)))          # an unsaturated output channel
    assert fw.poke_policy(len(pol["layers"]) - 1, "b", j, -5.0)
    assert not np.allclose(fw.policy_forward(obs), before)


def test_refuses_to_arm_into_policy_without_policy():
    r = sitl.SitlRunner(1)
    r.power_on(2.0)
    log = r.run(1.0, sitl.RcScript(arm_retry_s=0).at(0.5, arm=1, policy_enable=1))
    assert log["armed"][-1, 0] == 0
    assert log["arm_fail"][-1, 0] & (1 << 4)   # ARM_FAIL_POLICY


# ---------------------------------------------------------------- (a) fallback hover
def test_fallback_hover_nominal():
    res = S.fallback_hover(1)
    print(f"nominal: rms={res.rms[0]*100:.2f} cm max={res.max_err[0]*100:.2f} cm")
    assert res.passed.all(), (res.rms, res.modes, res.fault_history, res.crashed)


def test_fallback_hover_20_randomized():
    res = S.fallback_hover(20, randomize=True, seed=1)
    print("rms cm:", np.round(res.rms * 100, 2), "max cm:", np.round(res.max_err * 100, 2))
    assert res.passed.all(), [(i, res.rms[i], res.modes[i], res.fault_history[i]) for i in np.nonzero(~res.passed)[0]]


# ---------------------------------------------------------------- (b) policy mode
def test_policy_hover_synthetic_linear_policy():
    res = S.policy_hover(sitl.linear_hover_policy(), name="linear")
    assert res.passed.all(), (res.rms, res.modes, res.fault_history)


def test_policy_hover_trained():
    pol = S.trained_policy()
    if pol is None:
        pytest.skip(f"{S.TRAINED_POLICY} not present yet")
    fw = sitl.Firmware(1)
    ok, crc = fw.set_policy(pol)
    assert ok and crc == sitl.nfpol_crc32(pol)
    res = S.policy_hover(pol, n=4, randomize=True, seed=11, name="trained")
    print("trained rms cm:", np.round(res.rms * 100, 2), res.modes, res.fault_history)
    assert not res.crashed.any(), res.modes
    assert (res.fault_history == 0).all(), [sitl.fault_names(int(h)) for h in res.fault_history]
    assert (res.rms < 0.10).all(), res.rms


def test_bad_random_policy_is_caught():
    for r in S.bad_policy_flight():
        print(r)
        assert r["engaged"]
        assert r["passed"], r


# ---------------------------------------------------------------- (c)+(d) faults
@pytest.fixture(scope="module", params=[False, True], ids=["nominal", "randomized"])
def fault_results(request):
    res, _ = S.run_fault_suite(randomize=request.param, seed=5)
    for r in res:
        print(f"{r.fault:10s} first={r.first_fault:20s} -> {r.mode_after:18s} "
              f"latency={r.latency_ms:7.1f} ms alt_lost={r.alt_lost_m:.3f} m crashed={r.crashed} {r.why}")
    return {r.fault: r for r in res}


FAULT_NAMES = [f.name for f in S.make_faults(sitl.linear_hover_policy())]


@pytest.mark.parametrize("fault", FAULT_NAMES)
def test_fault_injection(fault_results, fault):
    r = fault_results[fault]
    assert r.passed, r.why
    # latency budgets: hold times of the corresponding monitors + 1 control period
    budget_ms = {"nan": 2, "saturate": 202, "oscillate": 100, "flip": 300, "imu_b": 52, "rc_loss": 302,
                 "overrun": 8, "kill": 0}[fault]
    assert r.latency_ms <= budget_ms, r.latency_ms


def test_kill_in_fallback_mode_stops_motors_same_loop():
    r = sitl.SitlRunner(1)
    r.power_on(2.0)
    log = r.run(5.0, sitl.standard_script(arm_t=0.5, kill_t=4.0))
    k = int(np.searchsorted(log.t, 4.0 - 1e-9))
    assert log["mode"][k - 1, 0] == FALLBACK_HOLD and (log["cmd"][k - 1, 0] > 0).all()
    assert (log["cmd"][k, 0] == 0).all() and log["mode"][k, 0] == MOTOR_STOP
    assert (log["cmd"][k:, 0] == 0).all()   # stays stopped


# ---------------------------------------------------------------- (e) CTBR parity
def test_ctbr_parity_random_inputs():
    m = S.ctbr_parity_random()
    print(m)
    assert m["max_abs"] <= S.CTBR_TOL, m
    assert m["sat_flag_mismatch_frac"] == 0.0
    assert m["fw_saturated_frac"] > 0.3   # the stress set really exercises the saturation paths


def test_ctbr_parity_closed_loop():
    m = S.ctbr_parity_closed_loop()
    print(m)
    assert m["max_abs"] <= S.CTBR_TOL, m


def test_sim_env_reset_derivative_kick_is_known_mismatch():
    """sim/env.py resets RateController without the gyro (prev meas = 0) ->
    D-term kick on the first step; firmware seeds prev meas with the first
    sample. Documented mismatch (reported); this pins it so a fix is noticed."""
    m = S.ctbr_parity_random(steps=200, sim_reset_with_gyro=False)
    assert m["first_step_max_abs"] > 1e-2
