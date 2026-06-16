"""
Conftest for research/core_k0 — exclude experiment scripts from pytest collection.

All test_*.py files in this directory are standalone research scripts meant to be
run via `python -m` or direct execution, NOT as pytest tests. Their test_ functions
have non-fixture parameters that cause pytest collection errors.
"""

import os

# Collect all test_*.py files in this directory as scripts to ignore
_dir = os.path.dirname(__file__)
_test_scripts = [
    f for f in os.listdir(_dir)
    if f.startswith("test_") and f.endswith(".py") and f != "conftest.py"
]

collect_ignore = _test_scripts
