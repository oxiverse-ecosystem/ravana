"""Pytest for dynamic runtime / dev-mode behavior."""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, SRC)


def _import_governor(force_reimport: bool = False):
    if force_reimport:
        for mod in list(sys.modules):
            if mod == 'ravana_grace' or mod.startswith('ravana_grace.'):
                del sys.modules[mod]
    gov_mod = __import__('ravana_grace.core.governor', fromlist=['Governor', 'GovernorConfig'])
    return gov_mod


def test_publish_mode_wheel_import():
    os.environ.pop('RAVANA_RUNTIME_MODE', None)
    gov_mod = _import_governor(force_reimport=False)
    gov = gov_mod.Governor(gov_mod.GovernorConfig())
    assert gov is not None


@pytest.mark.parametrize('mode', ['DEV', 'DEV_STRICT'])
def test_dev_mode_flag_import(mode: str):
    os.environ['RAVANA_RUNTIME_MODE'] = mode
    gov_mod = _import_governor(force_reimport=True)
    gov = gov_mod.Governor(gov_mod.GovernorConfig())
    assert gov is not None
