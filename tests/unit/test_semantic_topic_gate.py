"""Test: semantic topic gate prevents false recall matches.

Round 2026-10-15 (RL-1 fix): the structured recall system matched
"what was my opinion about patience" to a stance on "taught" because
the token-overlap matching was too permissive. The fix adds a
semantic similarity gate (GloVe cosine) that verifies token-overlap
matches are genuine.

This test verifies that:
1. A query about "patience" does NOT match a stance on "taught"
   (the RL-1 false match)
2. A query about a semantically related topic still matches correctly
3. The _semantic_topic_match method works as expected
"""
import os, sys, io, contextlib, tempfile, shutil
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.personal_fact_store import Stance


def _build():
    d = tempfile.mkdtemp(prefix="ravana_rl1_")
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=d), d


def test_semantic_topic_match_related():
    """Semantically related topics should match."""
    eng, d = _build()
    try:
        assert eng._semantic_topic_match("happy", "joyful"), \
            "'happy' and 'joyful' should be semantically related"
        assert eng._semantic_topic_match("car", "vehicle"), \
            "'car' and 'vehicle' should be semantically related"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_semantic_topic_match_unrelated():
    """Semantically unrelated topics should NOT match."""
    eng, d = _build()
    try:
        assert not eng._semantic_topic_match("patience", "taught"), \
            "'patience' and 'taught' should NOT be semantically related"
        assert not eng._semantic_topic_match("computer", "banana"), \
            "'computer' and 'banana' should NOT be semantically related"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_match_stance_false_positive_prevention():
    """A query about 'patience' should NOT match a stance on 'taught'."""
    eng, d = _build()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            eng.user_model.opinions.stances["taught"] = \
                Stance(topic="taught", polarity=0.8, confidence=0.9)
            result = eng._match_stance("what was my opinion about patience")
            assert result is None, \
                f"'patience' query matched stance on '{result[0]}' — false positive!"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_match_stance_true_positive_preserved():
    """A query about 'taught' SHOULD still match the 'taught' stance."""
    eng, d = _build()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            eng.user_model.opinions.stances["taught"] = \
                Stance(topic="taught", polarity=0.8, confidence=0.9)
            result = eng._match_stance("what was my opinion about taught")
            assert result is not None, \
                "'taught' query did not match 'taught' stance — false negative!"
            assert result[0] == "taught"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_end_to_end_no_false_match():
    """End-to-end: query about patience should not return 'taught'."""
    eng, d = _build()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            eng.user_model.opinions.stances["taught"] = \
                Stance(topic="taught", polarity=0.8, confidence=0.9)
            r = eng.process_turn("what was my opinion about patience")
            reply = str(r) if r else ""
            assert "taught" not in reply.lower(), \
                f"End-to-end: reply mentions 'taught': {reply}"
    finally:
        shutil.rmtree(d, ignore_errors=True)
