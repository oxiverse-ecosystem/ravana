import pytest
from ravana.chat.engine import CognitiveChatEngine

@pytest.fixture
def engine():
    return CognitiveChatEngine(baby_mode=False)

def test_performative_verb_subject_extraction(engine):
    cases = [
        ("explain consciousness", "consciousness"),
        ("can you explain consciousness", "consciousness"),
        ("tell me about photosynthesis", "photosynthesis"),
        ("describe quantum entanglement", "quantum entanglement"),
        ("give an overview of machine learning", "machine learning"),
        ("search for dark matter", "dark matter"),
        ("define epistemology", "epistemology"),
    ]
    for query, expected_subj in cases:
        subj, _ = engine._extract_topic(query, [])
        cleaned = engine._clean_scenario_subject(subj, query) if subj else ""
        assert cleaned.lower() == expected_subj.lower(), f"Expected '{expected_subj}' for '{query}', got '{cleaned}'"

def test_performative_query_variant_generation(engine):
    variants = engine._refine_query_variants("explain consciousness", "consciousness")
    assert "consciousness definition meaning" in variants or "what is consciousness" in variants
    assert variants[0] in ("consciousness definition meaning", "what is consciousness")

def test_conditional_compound_subject_extraction(engine):
    query = "what happens if time travel possible"
    subj, _ = engine._extract_topic(query, [])
    cleaned = engine._clean_scenario_subject(subj, query) if subj else ""
    assert "time travel" in cleaned.lower()
