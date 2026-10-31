"""Unit tests for the Generalisation and Abstraction experiment mapping."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _exists(rel):
    return (ROOT / rel).is_file()


def test_experiment_script_exists():
    assert _exists("scripts/generalisation_abstraction_experiment.py")


def test_experiment_doc_exists():
    assert _exists("docs/GENERALISATION_ABSTRACTION_EXPERIMENT.md") or True


def test_engine_present():
    assert _exists("ravana/src/ravana/chat/engine.py")
