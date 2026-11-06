import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
for p in (ROOT, TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test")


@pytest.fixture(scope="session")
def rand_pol():
    from random_policy import random_policy
    return random_policy(seed=0)


@pytest.fixture(scope="session")
def rand_pol_q(rand_pol):
    import quantize
    return quantize.quantize_policy(rand_pol)
