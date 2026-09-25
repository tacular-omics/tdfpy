"""Shared pytest setup: Hypothesis profiles and the opt-in ``slow`` marker.

The default run is kept fast. ``--run-slow`` (or ``RUN_SLOW=1``) also runs the
tests marked ``slow``; ``HYPOTHESIS_PROFILE=thorough`` (200 examples) or ``deep`` (2000) raises the
Hypothesis example counts.
CI sets both.
"""

import os

import pytest
from hypothesis import HealthCheck, settings

settings.register_profile("default", max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
# CI: at least double every per-test count the suite used before profiles.
settings.register_profile("thorough", max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
# Local soak runs.
settings.register_profile("deep", max_examples=2000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-slow", action="store_true", default=False, help="also run tests marked slow")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow") or os.environ.get("RUN_SLOW", "") not in ("", "0"):
        return
    skip = pytest.mark.skip(reason="slow: run with --run-slow or RUN_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
