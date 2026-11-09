"""Tests for the Phase 1 dialogue memory.

These exercise the real store (SQLite via CognitiveDB) and the real retrieval
path, not a mock. The neuroscience grounding is in the module docstring:
episodic encoding of conversational information (Tavares et al. 2015),
prediction-error-driven updating of conversation memories (Boeltzig, Liedtke &
Schubotz 2025; Liedtke et al. 2025), interactive alignment (Pickering & Garrod
2004), grounding/common ground (Clark & Brennan 1991), and Ullman's
declarative/procedural model (2001, 2004).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "ravana_ml", "src"))

from ravana.storage.db import CognitiveDB  # noqa: E402
from ravana.learn.dialogue_memory import (  # noqa: E402
    DialogueMemory, DialogueMemoryConfig, DialogueTurn, derive_outcome,
    OUTCOME_CORRECTED, OUTCOME_NEUTRAL, OUTCOME_CONTINUED,
)


DIM = 8

# A fixed topic vocabulary. Each topic owns one axis of the embedding space,
# so two utterances are similar exactly when they share a topic axis. This is
# deterministic (unlike Python's salted hash()) and makes the expected
# similarity values exact.
_TOPIC_AXES = {
    "music": 0, "song": 0, "melody": 0, "tune": 0,
    "ocean": 1, "sea": 1, "wave": 1, "tide": 1,
    "mountain": 2, "peak": 2, "cliff": 2, "summit": 2,
    "desert": 3, "dune": 3, "sand": 3, "arid": 3,
    "forest": 4, "tree": 4, "leaf": 4, "grove": 4,
    "river": 5, "stream": 5, "brook": 5, "water": 5,
    "cloud": 6, "sky": 6, "rain": 6, "storm": 6,
    "stone": 7, "rock": 7, "boulder": 7, "pebble": 7,
}


def _stable_axis(word: str) -> int:
    """A deterministic axis for an out-of-vocabulary word.

    Uses a fixed-seed hash (NOT the salted built-in hash()) so the same word
    always maps to the same axis across processes and runs.
    """
    import zlib
    return zlib.crc32(word.encode("utf-8")) % DIM


def _vec_fn(word):
    """Deterministic pseudo-embedding: a distinct unit vector per word.

    Words in the same topic share an axis, so cosine similarity between two
    utterances is 1.0 when their content words all fall on one shared axis
    and 0.0 when their axes are disjoint.
    """
    w = (word or "").lower()
    if not w:
        return None
    axis = _TOPIC_AXES.get(w)
    if axis is None:
        axis = _stable_axis(w)
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = 1.0
    return v


def _content_words(text):
    """The exact content-word list the memory's vector builder will use."""
    from ravana.learn.dialogue_memory import _WORD_RE, _STOPWORDS
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS]


def _db(tmpdir, name="test.db"):
    return CognitiveDB(os.path.join(tmpdir, name))


class _FakeUserModel:
    """Minimal stand-in for the real UserModel's correction flags.

    The point is to exercise derive_outcome() against the SAME attribute
    names the real detector sets (detected_correction, correction_severity,
    detected_correction_type), without importing the heavy UserModel.
    """

    def __init__(self, detected=False, severity=0.0, ctype=None):
        self.detected_correction = detected
        self.correction_severity = severity
        self.detected_correction_type = ctype


class _FakeCorrectionType:
    def __init__(self, value):
        self.value = value


# ── round-trip / persistence ────────────────────────────────────────────────

def test_store_persists_across_instances(tmpdir):
    """A turn stored by one DialogueMemory is visible to a fresh instance
    backed by the same DB file (the whole point of Phase 1)."""
    path = os.path.join(tmpdir, "rt.db")
    db = CognitiveDB(path)
    mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    assert mem.available

    ok = mem.record_turn(
        turn_index=1, user_utterance="i love music",
        ravana_reply="music is a pattern of sound",
        subject="music", strategy="definition",
        outcome=OUTCOME_CONTINUED,
    )
    assert ok
    assert mem.count() == 1
    db.close()

    # Fresh instance, same file.
    db2 = CognitiveDB(path)
    mem2 = DialogueMemory(db2, vector_fn=_vec_fn, dim=DIM)
    assert mem2.available
    assert mem2.count() == 1
    turns = mem2.recent_turns(10)
    assert len(turns) == 1
    t = turns[0]
    assert t.turn_index == 1
    assert t.user_utterance == "i love music"
    assert t.ravana_reply == "music is a pattern of sound"
    assert t.subject == "music"
    assert t.strategy == "definition"
    assert t.outcome == OUTCOME_CONTINUED
    assert t.user_vector is not None
    assert t.user_vector.shape == (DIM,)
    db2.close()


def test_vector_round_trips_through_blob(tmpdir):
    """The stored user vector survives the float32 blob round-trip."""
    db = _db(tmpdir)
    mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    mem.record_turn(turn_index=7, user_utterance="the ocean is vast",
                    ravana_reply="the sea covers most of the planet",
                    subject="ocean")
    t = mem.get_turn(7)
    assert t is not None
    # Mean-pooled unit vector over content words -> unit norm.
    assert abs(float(np.linalg.norm(t.user_vector)) - 1.0) < 1e-5
    db.close()


# ── retrieval ───────────────────────────────────────────────────────────────

def test_retrieval_returns_most_similar(tmpdir):
    """A related query returns the matching past exchange as the top hit."""
    db = _db(tmpdir)
    mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    # Each stored utterance's content words all sit on ONE topic axis, so
    # the stored vector is exactly that axis and cosine is 1.0 for a query
    # on the same axis and 0.0 otherwise.
    mem.record_turn(1, "music melody", "music is a pattern of sound",
                    subject="music")
    mem.record_turn(2, "ocean wave", "the sea covers most of the planet",
                    subject="ocean")
    mem.record_turn(3, "mountain peak", "a peak rises above the valley",
                    subject="mountain")

    hits = mem.recall_similar_exchanges("music", k=1)
    assert len(hits) == 1
    turn, sim = hits[0]
    assert turn.turn_index == 1
    assert turn.subject == "music"
    assert abs(sim - 1.0) < 1e-5
    db.close()


def test_retrieval_empty_for_unrelated_query(tmpdir):
    """A query with no related past exchange returns nothing.

    The query's content words are all out-of-vocabulary, so their vectors
    land on crc32-derived axes. We assert the OBSERVABLE contract: no
    returned turn is a stored turn (i.e. either nothing is returned, or
    nothing scores above the similarity of a genuinely related query).
    """
    db = _db(tmpdir)
    mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    mem.record_turn(1, "music melody", "music is a pattern of sound",
                    subject="music")
    mem.record_turn(2, "ocean wave", "the sea covers most of the planet",
                    subject="ocean")

    # A genuinely related query must score 1.0 (exact axis match).
    related = mem.recall_similar_exchanges("music", k=1)
    assert len(related) == 1
    assert related[0][1] == pytest.approx(1.0, abs=1e-5)

    # An unrelated query must NOT return a turn at that same similarity.
    unrelated = mem.recall_similar_exchanges("quantum chromodynamics", k=3)
    for turn, sim in unrelated:
        assert sim < 1.0, f"unrelated query matched {turn.subject} at {sim}"
    db.close()


def test_retrieval_empty_when_glove_absent(tmpdir):
    """With no vector function, no similarity can be computed -> empty."""
    db = _db(tmpdir)
    mem = DialogueMemory(db, vector_fn=None, dim=DIM)
    mem.record_turn(1, "i love music", "music is a pattern of sound",
                    subject="music")
    assert mem.recall_similar_exchanges("music", k=3) == []
    db.close()


def test_retrieval_k_respected(tmpdir):
    db = _db(tmpdir)
    mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    for i in range(5):
        mem.record_turn(i + 1, f"music song melody {i}", f"reply {i}",
                        subject="music")
    hits = mem.recall_similar_exchanges("music", k=2)
    assert len(hits) == 2
    # Sorted descending by similarity.
    sims = [s for _, s in hits]
    assert sims == sorted(sims, reverse=True)
    db.close()


def test_exclude_turn_index(tmpdir):
    """The current turn can be excluded so it never matches itself."""
    db = _db(tmpdir)
    mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    mem.record_turn(1, "i love music", "reply", subject="music")
    hits = mem.recall_similar_exchanges("music", k=3, exclude_turn_index=1)
    assert hits == []
    db.close()


# ── outcome ─────────────────────────────────────────────────────────────────

def test_outcome_recorded_from_simulated_correction(tmpdir):
    """A simulated correction (the real detector's flags) is stored as
    'corrected' with its severity."""
    db = _db(tmpdir)
    mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    um = _FakeUserModel(detected=True, severity=0.8,
                        ctype=_FakeCorrectionType("fact"))
    outcome, severity, ctype = derive_outcome(um)
    assert outcome == OUTCOME_CORRECTED
    assert severity == 0.8
    assert ctype == "fact"

    mem.record_turn(1, "no, my sister's name is priya",
                    "thanks for correcting me", subject="sister",
                    outcome=outcome, outcome_severity=severity,
                    correction_type=ctype)
    t = mem.get_turn(1)
    assert t.outcome == OUTCOME_CORRECTED
    assert abs(t.outcome_severity - 0.8) < 1e-6
    assert t.correction_type == "fact"
    db.close()


def test_outcome_continued_when_no_correction(tmpdir):
    db = _db(tmpdir)
    mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    um = _FakeUserModel(detected=False, severity=0.0, ctype=None)
    outcome, severity, ctype = derive_outcome(um)
    assert outcome == OUTCOME_CONTINUED
    mem.record_turn(1, "hello", "hi there", outcome=outcome)
    assert mem.get_turn(1).outcome == OUTCOME_CONTINUED
    db.close()


def test_derive_outcome_handles_missing_user_model():
    outcome, severity, ctype = derive_outcome(None)
    assert outcome == OUTCOME_NEUTRAL
    assert severity == 0.0
    assert ctype == ""


def test_outcome_counts(tmpdir):
    db = _db(tmpdir)
    mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    mem.record_turn(1, "a", "b", outcome=OUTCOME_CONTINUED)
    mem.record_turn(2, "a", "b", outcome=OUTCOME_CORRECTED)
    mem.record_turn(3, "a", "b", outcome=OUTCOME_CORRECTED)
    counts = mem.outcome_counts()
    assert counts[OUTCOME_CONTINUED] == 1
    assert counts[OUTCOME_CORRECTED] == 2
    db.close()


# ── cap / eviction ──────────────────────────────────────────────────────────

def test_cap_evicts_oldest(tmpdir):
    """The store keeps at most max_turns rows, evicting the oldest by
    turn_index (FIFO)."""
    db = _db(tmpdir)
    cfg = DialogueMemoryConfig(max_turns=5)
    mem = DialogueMemory(db, config=cfg, vector_fn=_vec_fn, dim=DIM)
    for i in range(1, 11):  # 10 turns, cap 5
        mem.record_turn(i, f"music {i}", f"reply {i}", subject="music")
    assert mem.count() == 5
    remaining = mem.recent_turns(10)
    indices = [t.turn_index for t in remaining]
    assert indices == [6, 7, 8, 9, 10]
    db.close()


def test_cap_zero_disables_eviction(tmpdir):
    """max_turns <= 0 means 'no eviction' (documented behaviour)."""
    db = _db(tmpdir)
    cfg = DialogueMemoryConfig(max_turns=0)
    mem = DialogueMemory(db, config=cfg, vector_fn=_vec_fn, dim=DIM)
    for i in range(1, 4):
        mem.record_turn(i, "a", "b")
    assert mem.count() == 3
    db.close()


# ── degradation ─────────────────────────────────────────────────────────────

def test_unusable_db_path_noops(tmpdir):
    """A DB path that cannot be opened makes the memory inert, not fatal."""
    bad_path = os.path.join(tmpdir, "not_a_dir", "sub", "test.db")
    try:
        db = CognitiveDB(bad_path)
    except Exception:
        # The DB constructor itself may raise; the memory must still be
        # constructible and inert.
        db = None
    if db is None:
        # Simulate the unavailable-DB shape directly.
        class _Broken:
            conn = None
        mem = DialogueMemory(_Broken(), vector_fn=_vec_fn, dim=DIM)
    else:
        mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    assert not mem.available
    assert mem.record_turn(1, "a", "b") is False
    assert mem.recall_similar_exchanges("a", k=3) == []
    assert mem.count() == 0
    assert mem.recent_turns(5) == []
    assert mem.get_turn(1) is None
    assert mem.clear() is False


def test_missing_table_noops(tmpdir):
    """A DB without the dialogue_turns table degrades to a no-op."""
    db = _db(tmpdir)
    # Drop the table if it exists, then build a memory against the same conn.
    with db.conn:
        db.conn.execute("DROP TABLE IF EXISTS dialogue_turns")
    mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    # CREATE TABLE IF NOT EXISTS recreated it, so this should be available.
    assert mem.available
    # Now drop it behind the memory's back and confirm reads/writes no-op.
    with db.conn:
        db.conn.execute("DROP TABLE dialogue_turns")
    assert mem.record_turn(1, "a", "b") is False
    assert mem.recall_similar_exchanges("a", k=3) == []
    assert mem.count() == 0
    db.close()


def test_closed_connection_noops(tmpdir):
    db = _db(tmpdir)
    mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    assert mem.available
    mem.close()
    assert not mem.available
    assert mem.record_turn(1, "a", "b") is False
    assert mem.recall_similar_exchanges("a", k=3) == []
    # close is idempotent
    mem.close()


# ── recall@k sanity ─────────────────────────────────────────────────────────

def test_recall_at_k_after_n_turns_on_topic(tmpdir):
    """After N turns on a topic, the right exchange is in the top-k.

    12 turns: 6 about music, 6 about the ocean. A music query must rank a
    music turn first, and the correct one must be within the top-3.
    """
    db = _db(tmpdir)
    mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    # Interleave topics so recency alone cannot explain the result.
    for i in range(1, 7):
        mem.record_turn(i * 2 - 1, f"music song melody {i}",
                        f"music reply {i}", subject="music")
        mem.record_turn(i * 2, f"ocean sea wave {i}",
                        f"ocean reply {i}", subject="ocean")
    assert mem.count() == 12

    hits = mem.recall_similar_exchanges("music song melody", k=3)
    assert len(hits) == 3
    # Every top-3 hit must be a music turn (ocean turns are orthogonal).
    for turn, sim in hits:
        assert turn.subject == "music", f"got {turn.subject} (sim={sim})"
    # All three content words share the music axis, so the mean-pooled query
    # vector is exactly that axis -> cosine 1.0 with every music turn.
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    db.close()


def test_recall_at_k_distinguishes_confusable_topics(tmpdir):
    """Two similar-but-distinct topics (mountain vs desert) are separated."""
    db = _db(tmpdir)
    mem = DialogueMemory(db, vector_fn=_vec_fn, dim=DIM)
    mem.record_turn(1, "the mountain peak is snowy", "reply", subject="mountain")
    mem.record_turn(2, "the desert dune is arid", "reply", subject="desert")
    hits = mem.recall_similar_exchanges("mountain summit", k=1)
    assert len(hits) == 1
    assert hits[0][0].turn_index == 1
    db.close()


# ── DialogueTurn dataclass ──────────────────────────────────────────────────

def test_dialogue_turn_to_from_row_round_trip():
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    t = DialogueTurn(
        turn_index=3, timestamp=123.5, user_utterance="hi",
        ravana_reply="hello", subject="greeting", strategy="chitchat",
        user_vector=v, outcome=OUTCOME_CONTINUED, outcome_severity=0.0,
        correction_type="",
    )
    row = t.to_row()
    t2 = DialogueTurn.from_row(row)
    assert t2.turn_index == 3
    assert t2.subject == "greeting"
    assert t2.outcome == OUTCOME_CONTINUED
    assert np.allclose(t2.user_vector, v)


def test_dialogue_turn_invalid_outcome_normalised():
    v = np.zeros(4, dtype=np.float32)
    t = DialogueTurn(turn_index=1, timestamp=0.0, user_utterance="",
                     ravana_reply="", subject="", strategy="",
                     user_vector=v, outcome="bogus")
    assert t.outcome == OUTCOME_NEUTRAL


# ── engine wiring ───────────────────────────────────────────────────────────

def test_engine_records_turn_end_to_end(tmpdir):
    """The engine boots with the memory wired and records a real turn.

    This is the Phase 1 contract: process_turn stores the exchange after the
    reply is produced, and the store survives a fresh engine instance on the
    same DB file.
    """
    from ravana.chat.engine import CognitiveChatEngine

    suffix = "_dm_engine_test"
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                            data_dir=str(tmpdir), user_suffix=suffix)
    assert e.dialogue_memory.available
    assert e.dialogue_memory.count() == 0

    reply = e.process_turn("hello")
    assert isinstance(reply, str) and reply
    assert e.dialogue_memory.count() == 1

    turn = e.dialogue_memory.recent_turns(1)[0]
    assert turn.user_utterance == "hello"
    assert turn.ravana_reply == reply
    assert turn.outcome in (OUTCOME_CONTINUED, OUTCOME_NEUTRAL,
                            OUTCOME_CORRECTED)
    # The strategy is the one the generator actually used.
    assert turn.strategy == e._last_strategy

    # Retrieval is exposed (Phase 2 entry point) and returns a list.
    assert isinstance(e.recall_similar_exchanges("hello", k=3), list)
    e.dialogue_memory.close()


def test_engine_records_correction_outcome_end_to_end(tmpdir):
    """A simulated correction is stored with outcome='corrected'.

    Uses the REAL correction detector (no mock): the fact-miner sets
    detected_correction on a name correction, and the engine must record the
    verdict even though the correction-persist path resets the flags before
    the turn ends.
    """
    from ravana.chat.engine import CognitiveChatEngine

    suffix = "_dm_engine_corr"
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                            data_dir=str(tmpdir), user_suffix=suffix)
    e.process_turn("my dog is called rocky")
    e.process_turn("no, that is wrong, my dog is called max")

    turns = e.dialogue_memory.recent_turns(5)
    assert len(turns) == 2
    corrected = [t for t in turns if t.outcome == OUTCOME_CORRECTED]
    assert len(corrected) == 1, f"outcomes: {[t.outcome for t in turns]}"
    assert corrected[0].turn_index == 2
    assert corrected[0].outcome_severity > 0.0
    e.dialogue_memory.close()


def test_engine_memory_survives_restart(tmpdir):
    """The store is durable: a fresh engine on the same DB sees the turns."""
    from ravana.chat.engine import CognitiveChatEngine

    suffix = "_dm_engine_restart"
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                            data_dir=str(tmpdir), user_suffix=suffix)
    e.process_turn("my dog is called rocky")
    e.process_turn("no, that is wrong, my dog is called max")
    assert e.dialogue_memory.count() == 2
    e.save()
    e.dialogue_memory.close()

    e2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                             data_dir=str(tmpdir), user_suffix=suffix)
    assert e2.dialogue_memory.count() == 2
    outcomes = {t.turn_index: t.outcome
                for t in e2.dialogue_memory.recent_turns(5)}
    assert outcomes[2] == OUTCOME_CORRECTED
    e2.dialogue_memory.close()


def test_engine_degrades_when_table_missing(tmpdir):
    """With the table dropped, the engine still boots and chats."""
    from ravana.chat.engine import CognitiveChatEngine

    suffix = "_dm_engine_degrade"
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                            data_dir=str(tmpdir), user_suffix=suffix)
    with e.db.conn:
        e.db.conn.execute("DROP TABLE dialogue_turns")
    assert e._record_dialogue_turn("hi", "hello", "") is False
    assert e.recall_similar_exchanges("hi", k=2) == []
    reply = e.process_turn("hello")
    assert isinstance(reply, str) and reply
    e.dialogue_memory.close()
