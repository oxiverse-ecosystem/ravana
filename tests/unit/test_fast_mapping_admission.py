"""A text's own subject is admitted on first sight (schema-consistent fast mapping).

Tse et al. (2011, Science 333:891-895) and Kumaran, Hassabis & McClelland (2016,
TiCS 20(7):512-534): a fact that fills slots which already exist is learned in
ONE trial; only schema-inconsistent material must be interleaved.

RAVANA's node-admission gate used a flat corroboration counter
(`_promote_min_sources = 2` distinct sources) for EVERY token, including the
subject the text is actually about. Measured consequence: ingesting the 512-pair
harvested corpus once left 384 subjects parked in the provisional buffer forever,
and the grounding gate withheld them.

`_learn_from_text` now routes the text's own subject through fast mapping, while
co-occurring tokens still require corroboration. These tests pin that split.
"""
import pytest

from ravana.chat.engine import CognitiveChatEngine


def _engine(tmp_path):
    eng = CognitiveChatEngine(data_dir=str(tmp_path), user_suffix="")
    eng._concept_keywords = {}
    eng._concept_labels = set()
    eng._concept_sources = {}
    eng._provisional_nodes = {}
    eng._definitions = {}
    return eng


def _known(eng, word):
    return (word in eng._concept_keywords
            or word.lower() in eng._concept_labels
            or word in eng._concept_sources)


def test_topic_subject_is_admitted_on_first_sighting(tmp_path):
    """The subject the text is ABOUT must not wait for a second source."""
    eng = _engine(tmp_path)
    eng._learn_from_text(
        "Gravity is a force that attracts masses toward each other.",
        "gravity", source_url="wikipedia:gravity")
    assert _known(eng, "gravity"), \
        "the text's own subject must be admitted on first sight"


def test_co_occurring_token_still_requires_corroboration(tmp_path):
    """A token that merely appears in the text is NOT fast-mapped."""
    eng = _engine(tmp_path)
    eng._learn_from_text(
        "Gravity is a force that attracts masses toward each other.",
        "gravity", source_url="wikipedia:gravity")
    # 'attracts' is a co-occurring content word, not the subject
    assert not _known(eng, "attracts"), \
        "co-occurring tokens must still go through the corroboration buffer"


def test_junk_is_still_rejected_on_the_fast_map_route(tmp_path):
    """Fast mapping must not become a junk bypass: the structural pre-filter
    runs BEFORE the fast-map branch, so unambiguous junk is never admitted."""
    eng = _engine(tmp_path)
    before = len(eng._concept_keywords)
    # 'the' is a stopword and never in important_words; a junk-scoring token as
    # the topic must still be rejected.
    eng._learn_from_text(
        "Noun the of and is was were be been being have has had do does did.",
        "the", source_url="wikipedia:the")
    assert len(eng._concept_keywords) == before or "the" not in eng._concept_keywords


def test_second_source_still_promotes_a_buffered_token(tmp_path):
    """The corroboration route must still work for non-subject tokens."""
    eng = _engine(tmp_path)
    text = "Photosynthesis converts light into chemical energy in plants."
    eng._learn_from_text(text, "photosynthesis", source_url="srcA")
    assert not _known(eng, "chemical"), "first sighting must be buffered"
    eng._learn_from_text(text, "photosynthesis", source_url="srcB")
    assert _known(eng, "chemical"), \
        "reactivation across two distinct sources must promote"


def test_fast_map_does_not_disturb_existing_concepts(tmp_path):
    """A subject already in the graph is reinforced, not duplicated."""
    eng = _engine(tmp_path)
    eng._learn_from_text("Gravity is a force.", "gravity", source_url="s1")
    n_ids = len(eng._concept_keywords.get("gravity", []))
    eng._learn_from_text("Gravity pulls objects together.", "gravity",
                         source_url="s2")
    assert len(eng._concept_keywords.get("gravity", [])) == n_ids
