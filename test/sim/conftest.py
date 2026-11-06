import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.params import load_nominal, load_randomization  # noqa: E402

torch.set_num_threads(min(4, os.cpu_count() or 1))


@pytest.fixture(scope="session")
def nominal():
    return load_nominal()


@pytest.fixture(scope="session")
def rand():
    return load_randomization()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test")
