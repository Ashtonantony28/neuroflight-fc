import json

import numpy as np
import torch

from train.export import check_roundtrip, numpy_forward, to_nfpol, validate_nfpol
from train.eval import TorchPolicy
from train.ppo import Actor


def _actor():
    torch.manual_seed(0)
    a = Actor((64, 64))
    with torch.no_grad():   # make the last layer non-trivial
        a.net[-2].weight.normal_(0, 0.5)
    return a


def test_export_roundtrip(tmp_path):
    actor = _actor()
    mean = torch.randn(19) * 0.3
    std = torch.rand(19) + 0.2
    pol = to_nfpol(actor, mean, std)
    validate_nfpol(pol)
    assert pol["mode"] == "ctbr" and pol["layers"][-1]["act"] == "tanh"
    assert [np.asarray(L["W"]).shape for L in pol["layers"]] == [(64, 19), (64, 64), (4, 64)]
    p = tmp_path / "p.nfpol.json"
    p.write_text(json.dumps(pol))
    err = check_roundtrip(str(p), actor, mean, std)
    assert err < 1e-5
    # eval's torch runner agrees with the numpy reference
    obs = torch.randn(100, 19)
    a_np = numpy_forward(json.loads(p.read_text()), obs.numpy())
    a_t = TorchPolicy(str(p))(obs).numpy()
    assert np.abs(a_np - a_t).max() < 1e-5
    assert np.abs(a_np).max() <= 1.0


def test_export_checkpoint(small_cfg, tmp_path):
    from train.export import export_checkpoint
    from train.ppo import PPO
    algo = PPO(small_cfg)
    algo.iterate()
    ck = tmp_path / "c.pt"
    algo.save(str(ck))
    out = export_checkpoint(str(ck), str(tmp_path / "policy.nfpol.json"))
    pol = json.load(open(out))
    validate_nfpol(pol)
    assert np.allclose(pol["obs_mean"], algo.obs_rms.frozen()[0].numpy())
