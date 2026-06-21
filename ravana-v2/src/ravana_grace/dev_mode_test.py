"""Dev-mode smoke test for core.governor.

Three runs:
1) module only (publish wheel verified)
2) dev mode via PYTHONPATH
3) dev mode via RAVANA_RUNTIME_MODE flag
"""
from __future__ import annotations

import os
import sys
import importlib


def _ensure_paths():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    src = os.path.join(root, 'src')
    if src not in sys.path:
        sys.path.insert(0, src)
    return root, src


def _import_governor(force_reimport: bool = False):
    _ensure_paths()
    if 'ravana_grace' in sys.modules and force_reimport:
        for mod in list(sys.modules):
            if mod == 'ravana_grace' or mod.startswith('ravana_grace.'):
                del sys.modules[mod]
    return __import__('ravana_grace.core.governor', fromlist=['Governor', 'GovernorConfig'])


def test_module_installed():
    print("\n=== TEST 1: MODULE INSTALLED (publish wheel) ===")
    gov_mod = _import_governor(force_reimport=False)
    cfg = gov_mod.GovernorConfig()
    gov = gov_mod.Governor(cfg)
    print("publish_mode_test=OK")
    return gov


def test_dev_path():
    print("\n=== TEST 2: DEV / PYTHONPATH MODE ===")
    os.environ.pop('RAVANA_RUNTIME_MODE', None)
    gov_mod = _import_governor(force_reimport=True)
    cfg = gov_mod.GovernorConfig()
    gov = gov_mod.Governor(cfg)
    print("dev_mode_test=OK via PYTHONPATH")
    return gov


def test_dev_flag():
    print("\n=== TEST 3: DEV FLAG MODE ===")
    os.environ['RAVANA_RUNTIME_MODE'] = 'DEV'
    gov_mod = _import_governor(force_reimport=True)
    cfg = gov_mod.GovernorConfig()
    gov = gov_mod.Governor(cfg)
    print("dev_mode_test=OK via RAVANA_RUNTIME_MODE flag")
    return gov


def main():
    test_module_installed()
    test_dev_path()
    test_dev_flag()


if __name__ == '__main__':
    main()
