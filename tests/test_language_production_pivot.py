"""Unit tests for the Language Production Pivot research mapping.

Verifies the documented components/files cited in docs/LANGUAGE_PRODUCTION_PIVOT.md
exist in the repository, so the research doc's concrete mapping to existing
modules stays valid.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _exists(rel):
    return (ROOT / rel).is_file()


def test_research_doc_exists():
    assert _exists("docs/LANGUAGE_PRODUCTION_PIVOT.md")


def test_neural_decoder_present():
    assert _exists("ravana_ml/src/ravana_ml/nn/neural_decoder.py")


def test_predictive_coding_generator_present():
    assert _exists("ravana/src/ravana/decoder/predictive_coding_generator.py")


def test_attractor_memory_present():
    assert _exists("ravana/src/ravana/core/attractor_memory.py")


def test_cerebellar_ngram_present():
    assert _exists("ravana/src/ravana/language/cerebellar_ngram.py")


def test_verb_lexicon_present():
    assert _exists("ravana/src/ravana/language/verb_lexicon.py")


def test_syntactic_cell_assembly_present():
    assert _exists("ravana/src/ravana/language/syntactic_cell_assembly.py")


def test_basal_ganglia_present():
    assert _exists("ravana/src/ravana/language/basal_ganglia.py")


def test_surface_realizer_present():
    assert _exists("ravana/src/ravana/language/surface_realizer.py")


def test_response_gen_present():
    assert _exists("ravana/src/ravana/chat/response_gen.py")


def test_engine_present():
    assert _exists("ravana/src/ravana/chat/engine.py")


def test_engine_generation_present():
    assert _exists("ravana/src/ravana/chat/engine_generation.py")
