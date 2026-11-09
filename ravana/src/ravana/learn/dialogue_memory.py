"""Dialogue memory: a persistent, retrievable store of conversation exchanges.

Phase 1 of the dialogue-memory feature. This module stores each exchange
(user utterance + RAVANA reply + grounded subject + reply strategy + outcome)
and retrieves the k most similar PAST exchanges for a new utterance. It is
memory only: retrieval is exposed but NOT wired into the response generator
(that is Phase 2).

Neuroscience grounding
----------------------
  * **Episodic encoding of social/conversational information.** The
    hippocampus does not only map physical space; it computes a general,
    multidimensional cognitive map that also frames social relationships and
    interactions (Tavares et al. 2015, *Neuron* 87(1):231-243). A dialogue
    memory is the conversational instance of that claim: each exchange is an
    EPISODE (who said what, about what, with what outcome), not a bag of
    words. -> the store keeps the verbatim utterance, the reply, the subject
    and the outcome together, as one bound trace.

  * **Prediction-error-driven updating of conversation memories.** Memories
    for naturalistic dialogues are updated when a prediction about the
    conversation is violated, and the SIZE and TYPE of the prediction error
    shape whether the original or the modified statement is learned
    (Boeltzig, Liedtke & Schubotz 2025, *Memory* 33(1):73-83;
    Liedtke, Boeltzig, Mecklenbrauck, Siestrup & Schubotz 2025,
    *NeuroImage* 311:121194; Boeltzig et al. 2025, *NeuroImage*
    121375). -> the OUTCOME label on each turn is the local prediction-error
    signal: a user correction is a large PE (the model's expectation about
    the exchange was wrong), a neutral turn is a small one. We reuse the
    engine's existing correction detector (user_model.detected_correction /
    correction_severity) rather than building a second one.

  * **Interactive alignment.** Interlocutors align their linguistic
    representations at many levels via automatic priming, and alignment of
    situation models is what makes dialogue successful (Pickering & Garrod
    2004, *Behavioral and Brain Sciences* 27(2):212-225). -> retrieval is
    similarity-based over the user's utterance vector, so a new utterance
    reactivates the exchanges it aligns with.

  * **Grounding / common ground.** Communication is a collective process of
    reaching the grounding criterion, and common ground is updated
    moment-by-moment (Clark & Brennan 1991, in *Perspectives on Socially
    Shared Cognition*, pp. 127-149). -> the outcome label is the grounding
    evidence for the turn: 'corrected' means the exchange FAILED to ground
    (the user repaired it), 'continued' means it grounded and the dialogue
    moved on.

  * **Declarative / procedural model.** Ullman (2001, *Nature Reviews
    Neuroscience* 2(10):717-726; 2004, *Cognition* 92(1-2):231-270) separates
    a declarative memory system (lexicon/semantic+episodic, fast, medial
    temporal lobe) from a procedural memory system (grammar/sequential skill,
    basal ganglia / frontal). Turn-taking is the sequential, timed skill of
    dialogue; the stored (subject, strategy, outcome) triple is the
    declarative record the procedural skill can be tuned against. Phase 2
    will use it that way; Phase 1 only records.

Design notes
------------
  * **New table, not the dead `episodes` table.** `CognitiveDB.episodes`
    (id, timestamp, content, concepts) exists but `save_episode` has no
    production caller -- it is dead schema. Repurposing it would risk
    breaking `tests/unit/test_db_persistence.py` (which calls save_episode)
    and any future consumer. A separate `dialogue_turns` table keeps the
    existing contract intact and lets the two stores evolve independently.
  * **Bounded store.** The table is capped at ``max_turns`` (default 2000)
    most-recent turns. Eviction is FIFO by ``turn_index``: when the count
    exceeds the cap, the oldest rows are deleted. This mirrors the finite
    capacity of conversational episodic memory and prevents unbounded growth.
  * **Fail-open.** Every public method degrades to a no-op (or an empty
    result) if the DB is unavailable, the table is missing, or the vector
    function is absent. The engine must boot and chat even when memory is
    entirely unavailable.

Retrieval
---------
Similarity is cosine over the user utterance's embedding. The embedding is
mean-pooled ``engine._glove_vector`` over the utterance's content words
(stopwords excluded), matching ``engine._build_sentence_vector`` exactly.
If the engine exposes a ready-made sentence vector
(``engine._sentence_vector`` / ``engine._content_vector``) that is NOT used
here because those are the CURRENT turn's discourse-blended vectors, not a
pure function of the utterance; mean-pooling is deterministic and
utterance-local. If GloVe is unavailable, the utterance vector is all zeros
and retrieval returns an empty list (no similarity can be computed).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# Outcome labels. 'corrected' = the user repaired the exchange (large PE);
# 'neutral' = no correction signal; 'continued' = the exchange grounded and
# the dialogue moved on (no repair needed). The label is derived from the
# existing user_model correction detector, never from a second detector.
OUTCOME_CORRECTED = "corrected"
OUTCOME_NEUTRAL = "neutral"
OUTCOME_CONTINUED = "continued"

_VALID_OUTCOMES = {OUTCOME_CORRECTED, OUTCOME_NEUTRAL, OUTCOME_CONTINUED}

# Default cap on the number of stored turns (most recent N kept).
DEFAULT_MAX_TURNS = 2000

# Content-word token pattern (matches engine._build_sentence_vector).
_WORD_RE = re.compile(r"\b[a-zA-Z']{2,}\b")

# A small, static stopword set. We deliberately do NOT import
# ravana.chat.constants.STOP_WORDS here: that import pulls in the whole chat
# package (and its heavy transitive deps) for a set of ~120 words. This set
# is the same closed-class vocabulary the engine uses for content-word
# filtering; it is a seed, not authored prose.
_STOPWORDS = {
    'a', 'about', 'across', 'after', 'all', 'along', 'also', 'am', 'among',
    'an', 'and', 'any', 'are', 'around', 'as', 'ask', 'at', 'be', 'because',
    'become', 'been', 'before', 'begin', 'beneath', 'between', 'beyond',
    'both', 'but', 'by', 'call', 'can', 'change', 'come', 'could', 'despite',
    'did', 'do', 'does', 'during', 'each', 'end', 'every', 'except', 'feel',
    'few', 'find', 'for', 'from', 'get', 'give', 'go', 'had', 'has', 'have',
    'he', 'hear', 'help', 'her', 'him', 'his', 'how', 'if', 'in', 'inside',
    'into', 'is', 'it', 'its', 'just', 'keep', 'know', 'let', 'like',
    'listen', 'look', 'make', 'may', 'mean', 'might', 'more', 'most', 'move',
    'near', 'need', 'new', 'no', 'nor', 'not', 'of', 'on', 'once', 'one',
    'or', 'other', 'our', 'over', 'play', 'put', 'run', 'same', 'say', 'see',
    'seem', 'set', 'shall', 'she', 'should', 'show', 'since', 'so', 'some',
    'start', 'stop', 'such', 'take', 'talk', 'tell', 'than', 'that', 'the',
    'their', 'them', 'then', 'these', 'they', 'think', 'this', 'those',
    'through', 'til', 'till', 'to', 'too', 'toward', 'towards', 'try',
    'turn', 'twice', 'underneath', 'until', 'upon', 'use', 'versus', 'very',
    'via', 'walk', 'want', 'was', 'we', 'were', 'what', 'when', 'where',
    'whether', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
    'within', 'without', 'work', 'would', 'you', 'your',
    # conversational filler (engine.__init__ adds these to STOP_WORDS too)
    'please', 'sorry', 'thanks', 'thank', 'hello', 'hi', 'hey', 'bye',
    'goodbye',
}


@dataclass
class DialogueTurn:
    """One stored exchange. All fields are plain data; no behaviour."""

    turn_index: int
    timestamp: float
    user_utterance: str
    ravana_reply: str
    subject: str
    strategy: str
    user_vector: Optional[np.ndarray] = None
    outcome: str = OUTCOME_NEUTRAL
    outcome_severity: float = 0.0
    correction_type: str = ""

    def __post_init__(self) -> None:
        """Normalise the outcome label to a known value.

        A label outside the vocabulary (e.g. a bad caller, a corrupt row)
        must never be stored: it would make outcome_counts silently
        under-report. Normalise to 'neutral' -- the absence of a signal.
        """
        if self.outcome not in _VALID_OUTCOMES:
            self.outcome = OUTCOME_NEUTRAL
        self.outcome_severity = float(self.outcome_severity or 0.0)

    def to_row(self) -> Tuple:
        """Serialize to a DB row tuple (vector as float32 blob)."""
        vec_blob = None
        if self.user_vector is not None:
            vec_blob = np.asarray(self.user_vector, dtype=np.float32).tobytes()
        return (
            int(self.turn_index),
            float(self.timestamp),
            self.user_utterance,
            self.ravana_reply,
            self.subject,
            self.strategy,
            vec_blob,
            self.outcome,
            float(self.outcome_severity),
            self.correction_type,
        )

    @classmethod
    def from_row(cls, row: Tuple) -> "DialogueTurn":
        """Rebuild a turn from a DB row tuple."""
        (turn_index, timestamp, user_utterance, ravana_reply, subject,
         strategy, vec_blob, outcome, outcome_severity, correction_type) = row
        vec = None
        if vec_blob is not None:
            vec = np.frombuffer(vec_blob, dtype=np.float32).copy()
        return cls(
            turn_index=int(turn_index),
            timestamp=float(timestamp),
            user_utterance=user_utterance or "",
            ravana_reply=ravana_reply or "",
            subject=subject or "",
            strategy=strategy or "",
            user_vector=vec,
            outcome=outcome if outcome in _VALID_OUTCOMES else OUTCOME_NEUTRAL,
            outcome_severity=float(outcome_severity or 0.0),
            correction_type=correction_type or "",
        )


@dataclass
class DialogueMemoryConfig:
    """Knobs for the dialogue memory. Defaults are conservative."""

    # Maximum number of turns retained (most recent N). FIFO eviction.
    max_turns: int = DEFAULT_MAX_TURNS
    # Default k for retrieval.
    default_k: int = 3
    # If True, retrieval returns only turns whose cosine similarity is >= 0
    # (i.e. not anti-correlated). A negative similarity is not evidence of
    # relatedness. Set False to return the raw top-k regardless.
    drop_negative_similarity: bool = True


class DialogueMemory:
    """Persistent store of dialogue exchanges with similarity retrieval.

    The store is backed by the existing ``CognitiveDB`` (SQLite, WAL). All
    public methods are fail-open: if the DB or the table is unavailable they
    no-op or return an empty result rather than raising.
    """

    def __init__(self, db: Any, config: Optional[DialogueMemoryConfig] = None,
                 vector_fn: Optional[Callable[[str], Optional[np.ndarray]]] = None,
                 dim: int = 64):
        self.db = db
        self.config = config or DialogueMemoryConfig()
        self._vector_fn = vector_fn
        self.dim = int(dim)
        self._available = False
        self._init_error: Optional[str] = None
        self._ensure_table()

    # ── schema ────────────────────────────────────────────────────────────

    def _ensure_table(self) -> None:
        """Create the dialogue_turns table if it does not exist.

        Fail-open: any exception marks the memory unavailable and every
        subsequent call becomes a no-op.
        """
        try:
            conn = self.db.conn
            if conn is None:
                self._init_error = "db connection is None"
                return
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dialogue_turns (
                        turn_index INTEGER PRIMARY KEY,
                        timestamp REAL,
                        user_utterance TEXT,
                        ravana_reply TEXT,
                        subject TEXT,
                        strategy TEXT,
                        user_vector BLOB,
                        outcome TEXT,
                        outcome_severity REAL,
                        correction_type TEXT
                    )
                """)
            self._available = True
        except Exception as e:  # noqa: BLE001 - fail-open by design
            self._init_error = repr(e)
            self._available = False

    @property
    def available(self) -> bool:
        """True when the table exists and the DB connection is live."""
        return bool(self._available)

    @property
    def init_error(self) -> Optional[str]:
        """The exception that made the memory unavailable, if any."""
        return self._init_error

    # ── vector ────────────────────────────────────────────────────────────

    def _utterance_vector(self, text: str) -> Optional[np.ndarray]:
        """Mean-pooled GloVe vector over the utterance's content words.

        Mirrors ``engine._build_sentence_vector``: tokenize, drop stopwords,
        look up each word via the injected ``_vector_fn``, mean-pool, unit
        normalize. Returns None when no word has a vector (GloVe absent or
        all-OOV), which makes retrieval return an empty list.
        """
        if not text or self._vector_fn is None:
            return None
        words = _WORD_RE.findall(text.lower())
        vecs: List[np.ndarray] = []
        for w in words:
            if w in _STOPWORDS:
                continue
            try:
                gv = self._vector_fn(w)
            except Exception:
                gv = None
            if gv is None:
                continue
            gv = np.asarray(gv, dtype=np.float32).ravel()
            if gv.size == 0:
                continue
            vecs.append(gv)
        if not vecs:
            return None
        # All vectors must share a dimension to be pooled. If the function
        # returns mixed widths (should not happen), keep the majority width.
        widths = [v.size for v in vecs]
        width = max(set(widths), key=widths.count)
        vecs = [v for v in vecs if v.size == width]
        if not vecs:
            return None
        mean_v = np.mean(vecs, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean_v))
        if norm > 1e-6:
            mean_v /= norm
        return mean_v

    # ── store ─────────────────────────────────────────────────────────────

    def record_turn(self, turn_index: int, user_utterance: str,
                    ravana_reply: str, subject: str = "",
                    strategy: str = "", outcome: str = OUTCOME_NEUTRAL,
                    outcome_severity: float = 0.0,
                    correction_type: str = "",
                    timestamp: Optional[float] = None,
                    user_vector: Optional[np.ndarray] = None) -> bool:
        """Store one exchange. Returns True on success, False on no-op.

        The outcome is whatever the caller derived from the existing
        user_model correction detector; this method does not re-derive it.
        """
        if not self._available:
            return False
        if turn_index is None:
            return False
        ts = float(timestamp) if timestamp is not None else time.time()
        vec = user_vector if user_vector is not None else \
            self._utterance_vector(user_utterance)
        turn = DialogueTurn(
            turn_index=int(turn_index),
            timestamp=ts,
            user_utterance=user_utterance or "",
            ravana_reply=ravana_reply or "",
            subject=subject or "",
            strategy=strategy or "",
            user_vector=vec,
            outcome=outcome if outcome in _VALID_OUTCOMES else OUTCOME_NEUTRAL,
            outcome_severity=float(outcome_severity or 0.0),
            correction_type=correction_type or "",
        )
        try:
            with self.db.conn:
                self.db.conn.execute("""
                    INSERT OR REPLACE INTO dialogue_turns
                        (turn_index, timestamp, user_utterance, ravana_reply,
                         subject, strategy, user_vector, outcome,
                         outcome_severity, correction_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, turn.to_row())
            self._evict_if_needed()
            return True
        except Exception:
            return False

    def _evict_if_needed(self) -> None:
        """FIFO eviction: keep the most recent ``max_turns`` rows.

        Eviction is by ``turn_index`` (monotonic), not by rowid, so a
        restored store with non-contiguous indices still evicts the true
        oldest turns.
        """
        try:
            cap = int(self.config.max_turns)
            if cap <= 0:
                return
            cur = self.db.conn.execute(
                "SELECT COUNT(*) FROM dialogue_turns")
            row = cur.fetchone()
            if not row:
                return
            count = int(row[0])
            if count <= cap:
                return
            excess = count - cap
            self.db.conn.execute(
                "DELETE FROM dialogue_turns WHERE turn_index IN ("
                "  SELECT turn_index FROM dialogue_turns"
                "  ORDER BY turn_index ASC LIMIT ?)",
                (excess,))
        except Exception:
            pass

    def count(self) -> int:
        """Number of stored turns (0 when unavailable)."""
        if not self._available:
            return 0
        try:
            cur = self.db.conn.execute("SELECT COUNT(*) FROM dialogue_turns")
            row = cur.fetchone()
            return int(row[0]) if row else 0
        except Exception:
            return 0

    def clear(self) -> bool:
        """Delete all stored turns. Returns True on success."""
        if not self._available:
            return False
        try:
            with self.db.conn:
                self.db.conn.execute("DELETE FROM dialogue_turns")
            return True
        except Exception:
            return False

    # ── retrieval ─────────────────────────────────────────────────────────

    def recall_similar_exchanges(self, query: str, k: int = 3,
                                 exclude_turn_index: Optional[int] = None
                                 ) -> List[Tuple[DialogueTurn, float]]:
        """Return the k most similar past exchanges for ``query``.

        Similarity is cosine between the query's utterance vector and each
        stored turn's user vector. Turns with no stored vector are skipped.
        When the query vector cannot be built (GloVe absent / all-OOV) the
        result is empty -- no similarity can be computed, and returning
        arbitrary turns would be confabulation.

        Returns a list of ``(DialogueTurn, similarity)`` sorted by
        similarity descending. Empty list when unavailable or no match.
        """
        if not self._available:
            return []
        qv = self._utterance_vector(query)
        if qv is None:
            return []
        k = max(0, int(k))
        if k == 0:
            return []
        try:
            cur = self.db.conn.execute(
                "SELECT turn_index, timestamp, user_utterance, ravana_reply,"
                " subject, strategy, user_vector, outcome, outcome_severity,"
                " correction_type"
                " FROM dialogue_turns ORDER BY turn_index ASC")
            rows = cur.fetchall()
        except Exception:
            return []

        scored: List[Tuple[DialogueTurn, float]] = []
        for row in rows:
            try:
                turn = DialogueTurn.from_row(row)
            except Exception:
                continue
            if exclude_turn_index is not None and \
                    turn.turn_index == exclude_turn_index:
                continue
            if turn.user_vector is None:
                continue
            tv = np.asarray(turn.user_vector, dtype=np.float32).ravel()
            if tv.size != qv.size or tv.size == 0:
                continue
            denom = float(np.linalg.norm(tv)) * float(np.linalg.norm(qv))
            if denom <= 1e-9:
                continue
            sim = float(np.dot(tv, qv) / denom)
            if self.config.drop_negative_similarity and sim < 0.0:
                continue
            scored.append((turn, sim))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def get_turn(self, turn_index: int) -> Optional[DialogueTurn]:
        """Fetch a single stored turn by index (None when unavailable)."""
        if not self._available:
            return None
        try:
            cur = self.db.conn.execute(
                "SELECT turn_index, timestamp, user_utterance, ravana_reply,"
                " subject, strategy, user_vector, outcome, outcome_severity,"
                " correction_type FROM dialogue_turns WHERE turn_index = ?",
                (int(turn_index),))
            row = cur.fetchone()
            return DialogueTurn.from_row(row) if row else None
        except Exception:
            return None

    def recent_turns(self, n: int = 10) -> List[DialogueTurn]:
        """Return the n most recent stored turns (oldest-first)."""
        if not self._available:
            return []
        try:
            cur = self.db.conn.execute(
                "SELECT turn_index, timestamp, user_utterance, ravana_reply,"
                " subject, strategy, user_vector, outcome, outcome_severity,"
                " correction_type FROM dialogue_turns"
                " ORDER BY turn_index DESC LIMIT ?", (max(0, int(n)),))
            rows = cur.fetchall()
            turns = [DialogueTurn.from_row(r) for r in rows]
            turns.reverse()
            return turns
        except Exception:
            return []

    def outcome_counts(self) -> Dict[str, int]:
        """Count stored turns by outcome label."""
        out = {OUTCOME_CORRECTED: 0, OUTCOME_NEUTRAL: 0,
               OUTCOME_CONTINUED: 0}
        if not self._available:
            return out
        try:
            cur = self.db.conn.execute(
                "SELECT outcome, COUNT(*) FROM dialogue_turns"
                " GROUP BY outcome")
            for outcome, cnt in cur.fetchall():
                if outcome in out:
                    out[outcome] = int(cnt)
        except Exception:
            pass
        return out

    def close(self) -> None:
        """Close the underlying DB connection (idempotent, fail-open)."""
        try:
            if self.db is not None and getattr(self.db, "conn", None) is not None:
                self.db.conn.close()
                self.db.conn = None
        except Exception:
            pass
        self._available = False


def derive_outcome(user_model: Any) -> Tuple[str, float, str]:
    """Derive (outcome_label, severity, correction_type) from the user model.

    This is the ONLY place the outcome is computed, and it reads the existing
    correction detector's flags -- it does not re-detect anything. A detected
    correction is a large prediction error -> 'corrected'. Otherwise the turn
    is 'continued' (the exchange grounded and the dialogue moved on).

    Returns (OUTCOME_NEUTRAL, 0.0, "") when the user_model is missing or the
    flags are absent, so the caller can always store something.
    """
    if user_model is None:
        return (OUTCOME_NEUTRAL, 0.0, "")
    try:
        detected = bool(getattr(user_model, "detected_correction", False))
    except Exception:
        detected = False
    if not detected:
        return (OUTCOME_CONTINUED, 0.0, "")
    try:
        severity = float(getattr(user_model, "correction_severity", 0.0) or 0.0)
    except Exception:
        severity = 0.0
    ctype = ""
    try:
        ct = getattr(user_model, "detected_correction_type", None)
        ctype = getattr(ct, "value", None) or (ct if isinstance(ct, str) else "")
    except Exception:
        ctype = ""
    return (OUTCOME_CORRECTED, severity, str(ctype or ""))
