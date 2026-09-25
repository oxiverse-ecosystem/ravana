"""BM25 lexical evidence for episodic recall.

Adopted from total-agent-memory's independent lexical retrieval tier. The
engine's semantic GloVe path remains authoritative when lexical evidence is
weak, so these tests focus on rare, exact content terms that semantic cosine
cannot reliably rank.
"""
import contextlib
import io
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _engine():
    sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
    os.environ["RAVANA_OFFLINE"] = "1"
    from ravana.chat.engine import CognitiveChatEngine
    return CognitiveChatEngine(
        dim=64, seed=42, baby_mode=True, user_suffix="test_episodic_bm25"
    )


def test_bm25_ranks_rare_cue_above_common_cue():
    eng = _engine()
    store = [
        {"text": "the telescope sits beside the greenhouse", "facts": {}},
        {"text": "the cedar notebook rests beside the telescope", "facts": {}},
        {"text": "the greenhouse needs water before noon", "facts": {}},
    ]

    ranked = eng._bm25_rank("cedar notebook telescope", store)

    assert ranked[0][0] is store[1], ranked
    assert ranked[0][1] > ranked[1][1], ranked


def test_cued_recall_uses_bm25_evidence_for_rare_terms():
    eng = _engine()
    for turn in (
        "the telescope sits beside the greenhouse.",
        "the cedar notebook rests beside the telescope.",
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            eng.process_turn(turn)

    recalled = eng._retrieve_episodic(
        "what did i tell you about the cedar notebook", eng._episodic_transcript
    )

    assert recalled and "cedar notebook" in recalled.lower(), recalled


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
