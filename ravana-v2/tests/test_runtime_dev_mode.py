"""
Pytest: dynamic runtime / dev-mode behavior for RAVANA GRACE.
"""
from __future__ import annotations

import os
import sys

import pytest

try:
    from .conftest import _PACKAGE_TEST
except ImportError:
    from conftest import _PACKAGE_TEST


def _import_ravana_grace(force_reimport: bool = False):
    """Import ``ravana_grace`` from the active source (src or installed)."""
    if force_reimport:
        for mod in list(sys.modules):
            if mod == "ravana_grace" or mod.startswith("ravana_grace."):
                del sys.modules[mod]
    return __import__("ravana_grace", fromlist=["__version__"])


def test_publish_mode_wheel_import():
    os.environ.pop("RAVANA_RUNTIME_MODE", None)
    pkg = _import_ravana_grace(force_reimport=False)
    assert getattr(pkg, "__version__", None) is not None


@pytest.mark.parametrize("mode", ["DEV", "DEV_STRICT"])
def test_dev_mode_flag_import(mode: str):
    os.environ["RAVANA_RUNTIME_MODE"] = mode
    pkg = _import_ravana_grace(force_reimport=True)
    assert getattr(pkg, "__version__", None) is not None
