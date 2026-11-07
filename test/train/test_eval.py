import math

import torch

from train.eval import build, evaluate, format_tables, make_pid, rollout, summarize


def test_eval_groups_share_initial_conditions():
    e1, g1, _ = build(8, 2, seed=5, extremes=True)
    e2, g2, _ = build(8, 2, seed=5, extremes=True)
    assert torch.equal(e1.quad.pos, e2.quad.pos) and torch.equal(e1.quad.quat, e2.quad.quat)
    assert torch.equal(e1.quad.params.mass, e2.quad.params.mass)
    # mass=min group has 0.85x nominal mass, nominal groups exactly nominal
    gm = next(g for g in g1 if g.name == "mass=min/tilt_60")
    gn = next(g for g in g1 if g.name == "tilt_60/nominal")
    nom_mass = e1.nom["mass"]
    assert torch.allclose(e1.quad.params.mass[gm.sl], torch.full((2,), 0.85 * nom_mass))
    assert torch.allclose(e1.quad.params.mass[gn.sl], torch.full((8,), nom_mass))
    assert (e1.quad.params.gyro_noise_std[gn.sl] == 0).all()
    tilt = torch.rad2deg(torch.acos(e1.quad.R[gn.sl, 2, 2]))
    assert torch.allclose(tilt, torch.full_like(tilt, 60.0), atol=1e-3)


def test_eval_pid_is_sane():
    res = evaluate({"PID": make_pid}, n=16, n_ext=4, seed=1, extremes=False,
                   scenarios=("hover_1m", "tilt_60"))
    r = res["PID"]
    for g in ("hover_1m/nominal", "tilt_60/nominal"):
        assert r[g]["success"] == 1.0, (g, r[g])
        assert r[g]["err_med"] < 0.05
        assert r[g]["settle_mean"] < 4.0
    assert r["tilt_60/nominal"]["worst_tilt"] >= 59.9
    assert r["hover_1m/nominal"]["worst_tilt"] < 45
    assert r["tilt_60/randomized"]["success"] > 0.8
    assert math.isfinite(r["hover_1m/randomized"]["hf_rates"])
    assert "PID" in format_tables(res)


def test_eval_detects_bad_controller():
    env, groups, nom = build(8, 2, seed=2, extremes=False, scenarios=("tilt_60",))
    zero_thrust = lambda obs: torch.cat((-torch.ones(obs.shape[0], 1), torch.zeros(obs.shape[0], 3)), -1)
    r = summarize(rollout(env, zero_thrust, horizon_s=1.5), groups)
    assert r["tilt_60/nominal"]["success"] == 0.0 and r["tilt_60/nominal"]["crash"] == 1.0
