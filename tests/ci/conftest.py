"""CI-specific test configuration.

Tests in tests/ci/ are the critical path: run on every CI push.
Use @pytest.mark.ci to mark tests for the CI pipeline.

Usage:
    pytest tests/ci/ --ci          # Only CI-marked tests
    pytest tests/unit/              # Unit tests (fast)
    pytest tests/integration/       # Integration tests (slower)
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--ci", action="store_true", default=False,
                     help="Run only CI-marked tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--ci"):
        # In CI mode, only run tests with @pytest.mark.ci marker
        ci_items = [item for item in items
                    if item.get_closest_marker("ci") is not None]
        items[:] = ci_items
