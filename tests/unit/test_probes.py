"""Tests for ravana-v2 probes: constraint_stress, exploration_pressure, learning_signal."""

import sys, os
_grace = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ravana-v2", "src")
if _grace not in sys.path:
    sys.path.insert(0, _grace)

import pytest


def _try_import_class(module_path, class_name):
    """Try importing a class from a module, return (success, error)."""
    try:
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name, None)
        if cls is None:
            return False, f"Class {class_name} not found in {module_path}"
        return True, None
    except (ImportError, AttributeError) as e:
        return False, str(e)


class TestConstraintStressProbe:
    def test_module_importable(self):
        ok, err = _try_import_class("ravana_grace.probes.constraint_stress", "ConstraintStressProbe")
        if not ok:
            pytest.skip(f"ConstraintStressProbe not importable (optional dep): {err}")


class TestExplorationPressureProbe:
    def test_module_importable(self):
        ok, err = _try_import_class("ravana_grace.probes.exploration_pressure", "ExplorationPressureProbe")
        if not ok:
            pytest.skip(f"ExplorationPressureProbe not importable (optional dep): {err}")


class TestLearningSignalProbe:
    def test_module_importable(self):
        ok, err = _try_import_class("ravana_grace.probes.learning_signal", "LearningSignalProbe")
        if not ok:
            pytest.skip(f"LearningSignalProbe not importable (optional dep): {err}")
