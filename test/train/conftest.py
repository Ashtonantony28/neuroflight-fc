import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

torch.set_num_threads(min(4, os.cpu_count() or 1))


@pytest.fixture(scope="session")
def nominal():
    from sim.params import load_nominal
    return load_nominal()


@pytest.fixture
def small_cfg():
    from train.ppo import load_config
    cfg = load_config(os.path.join(ROOT, "train", "config", "hover_ctbr.yaml"))
    cfg["env"]["num_envs"] = 64
    cfg["ppo"]["rollout_steps"] = 16
    cfg["ppo"]["minibatches"] = 2
    cfg["ppo"]["epochs"] = 2
    cfg["log_every"] = 1000
    return cfg
