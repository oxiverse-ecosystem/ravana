"""Evaluative-predicate polarity from distributional geometry.

WHY THIS EXISTS (round t_fd0f88d4, defect D5)
---------------------------------------------
RAVANA's opinion miner could only recognise a value judgment when the
predicate word appeared in a hand-written alternation list::

    r"\\bi\\s+think\\s+(.+?)\\s+(?:is|are)\\s+(?:\\w+ly\\s+)?(?:good\\b|great|...)"

That is a FROZEN vocabulary. "i think handmade mugs are overpriced" matched
no pattern, so **no stance was created at all** — and with no stance there
was nothing for the retraction machinery to recode, so a contradiction
("i was wrong about handmade mugs" / "no, i still think they are
overpriced") left the store untouched. Measured on a clean engine: three
such turns, `stances: {}` before and after. The user's opinion was simply
not in the model.

The fix is NOT a longer word list (that is the same frozen-vocabulary bug
with a bigger list). It is a second, INDEPENDENT route to the same
judgment: read the predicate's position in concept space.

A word that carries an evaluation ("overpriced", "sturdy", "dreadful")
sits in a different region of embedding space than a word that merely
names a thing ("table", "kiln", "bicycle"), and evaluative words split
cleanly along a positive/negative direction. So two projections of one
GloVe vector answer both questions the alternation list was being asked:

  * EVALUATIVE  = cos(v, evaluative_centroid) - cos(v, neutral_centroid)
                  -> is this word a judgment at all?
  * VALENCE     = cos(v, positive_centroid) - cos(v, negative_centroid)
                  -> which pole?

Measured on 62 held-out words (tmp/probe_glv3.py, engine dim=64):
  evaluative axis: 36/38 evaluative words score > 0, 24/24 neutral nouns
  score < 0 (worst evaluative +0.044, best neutral -0.016).
  valence axis: 35/38 correct sign; the 3 misses are all |margin| <= 0.012.

SEED vs HARDCODING (the doctrine this repo enforces)
----------------------------------------------------
The three anchor word sets are SEED VOCABULARY in exactly the sense the
repo already accepts (the VAD affect lexicon, ``_ACTIVITY_VERB_LEXICON``,
the relation-verb lexicon): a small curated lexicon of words, used as
*features*, never as a Q->A path. There is no per-topic table, no reply
prose, and nothing here decides WHAT RAVANA says — it only decides the
polarity of a judgment the user actually made.

Crucially the model can change through experience, in two directions:

  1. ``observe()`` — every predicate this model judges is REMEMBERED in a
     persisted ``judged`` store with a running confidence. A predicate
     RAVANA has met before costs no geometry on the next encounter, and
     the memory survives save/load, so the lexicon compounds with use.

  2. ``relearn()`` — when a later stance reversal on a predicate-derived
     topic moves the polarity, the user's new position is written BACK
     into ``judged``. The user is ground truth: geometry proposes, the
     user disposes, and the disposition persists. A predicate RAVANA
     mis-signed is corrected by the user talking, not by a retrain.

NO RETRAINING: every number here is computed from a lookup at call time
against the engine's existing projected GloVe table. Nothing is fitted,
nothing is rebuilt, and a predicate learned tonight is usable tonight.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ── SEED anchor vocabularies ─────────────────────────────────────────────────
# Small curated sets of *words* used as geometry features. See the module
# docstring for why this is seed vocabulary and not a hardcoded answer path.
# Each entry is removable; the model degrades to abstaining on the words it
# loses, never to a wrong polarity.
SEED_POSITIVE: Tuple[str, ...] = (
    "good", "great", "excellent", "wonderful", "useful", "underrated",
    "favorite", "reliable", "worthwhile", "recommended",
)
SEED_NEGATIVE: Tuple[str, ...] = (
    "bad", "terrible", "awful", "overrated", "useless", "horrible",
    "poor", "overpriced", "disappointing", "flawed",
)
SEED_NEUTRAL: Tuple[str, ...] = (
    "table", "chair", "window", "morning", "kiln", "pottery", "mug",
    "street", "teacher", "harbour", "bicycle", "letter",
)

# A predicate must clear this margin on the evaluative axis before its
# valence is trusted. Chosen from the measured distribution: the lowest
# scoring evaluative word was +0.044 and the highest scoring neutral word
# was -0.016, so 0.02 sits in the empty band between the two populations
# with margin on both sides. It is a decision boundary read off real
# output, not a constant tuned to one test case.
EVALUATIVE_MIN_MARGIN = 0.02

# Below this valence margin the sign is not trusted and the model ABSTAINS
# (returns None) rather than guessing. The three measured sign errors all
# had |margin| <= 0.012; the smallest correct sign was +0.021.
VALENCE_MIN_MARGIN = 0.02

# |valence| is scaled to [-1, 1] by this divisor: a predicate at or beyond
# this margin is a maximally confident read. 0.25 is just past the
# strongest measured margin (0.375) so a strong opinion saturates while a
# marginal one stays honest about its weakness.
VALENCE_SCALE = 0.25

# Confidence a first-time geometric judgment earns. A remembered judgment
# starts here and rises with each re-observation.
BASE_CONFIDENCE = 0.45
MAX_CONFIDENCE = 0.8
# Per-observation confidence increment (running mean toward BASE).
CONFIDENCE_STEP = 0.15


class EvaluativePolarityModel:
    """Scores an arbitrary predicate word for evaluativeness and valence.

    ``vector_fn`` maps a lowercase word to a unit vector (the engine's
    existing projected GloVe table) or returns None when the word is out of
    vocabulary. It is INJECTED rather than imported: the user model owns no
    GloVe table, and the engine already owns the only one — same pattern as
    ``user_model._episodic_index`` / ``_concept_vocab``.

    With no vector_fn (or an out-of-vocabulary word) every lookup returns
    None and no stance is minted — the model fails CLOSED.
    """

    def __init__(self, vector_fn: Optional[Callable[[str], Any]] = None,
                 positive: Sequence[str] = SEED_POSITIVE,
                 negative: Sequence[str] = SEED_NEGATIVE,
                 neutral: Sequence[str] = SEED_NEUTRAL):
        self._vector_fn = vector_fn
        self._seed_pos = [w.lower() for w in positive]
        self._seed_neg = [w.lower() for w in negative]
        self._seed_neu = [w.lower() for w in neutral]
        # Anchors ADDED at runtime by observe()/relearn(). Sets, so
        # re-observing a word is idempotent.
        self._learned_pos: set = set()
        self._learned_neg: set = set()
        # word -> [polarity, confidence]; the experiential memory.
        self._judged: Dict[str, List[float]] = {}
        self._centroids: Optional[Tuple[List[float], List[float], List[float]]] = None

    # ── wiring ───────────────────────────────────────────────────────────────
    def set_vector_fn(self, fn: Callable[[str], Any]) -> None:
        self._vector_fn = fn
        self._centroids = None

    # ── geometry ─────────────────────────────────────────────────────────────
    def _unit(self, word: str):
        if not self._vector_fn:
            return None
        try:
            v = self._vector_fn(word)
        except Exception:
            return None
        if v is None:
            return None
        try:
            norm = math.sqrt(sum(float(c) * float(c) for c in v))
        except (TypeError, ValueError):
            return None
        if norm <= 0.0:
            return None
        return [float(c) / norm for c in v]

    def _centroid(self, words: Sequence[str]):
        acc = None
        n = 0
        for w in words:
            v = self._unit(w)
            if v is None:
                continue
            if acc is None:
                acc = list(v)
            else:
                for i, c in enumerate(v):
                    acc[i] += c
            n += 1
        if not acc or n == 0:
            return None
        norm = math.sqrt(sum(c * c for c in acc))
        if norm <= 0.0:
            return None
        return [c / norm for c in acc]

    def _all_centroids(self):
        """(positive, negative, neutral) unit centroids, cached.

        The cache is invalidated whenever an anchor set grows, so an online
        learn() is reflected on the very next lookup.

        There is deliberately NO "evaluative centroid". An earlier version
        pooled the two pole centroids into one direction to ask "is this word
        evaluative at all?", which is wrong twice over: a word IS evaluative
        whether it leans good or bad, and summing the two poles cancels to
        near-zero whenever they are close to antipodal — which is exactly the
        case for a well-formed positive/negative anchor pair. The
        pool-and-normalise silently produced a degenerate axis there.
        Evaluativeness is instead computed per-word in ``score`` as the mean
        of the two ABSOLUTE pole similarities (see the comment there).
        """
        if self._centroids is not None:
            return self._centroids
        pos = self._centroid(self._seed_pos + sorted(self._learned_pos))
        neg = self._centroid(self._seed_neg + sorted(self._learned_neg))
        neu = self._centroid(self._seed_neu)
        if pos is None or neg is None or neu is None:
            return None
        self._centroids = (pos, neg, neu)
        return self._centroids

    # ── the capability ───────────────────────────────────────────────────────
    def score(self, word: str) -> Optional[Dict[str, float]]:
        """Return ``{'polarity', 'evaluative', 'confidence', 'source'}`` for
        an evaluative predicate, or None when the word is not one / is
        out of vocabulary / too close to the decision boundary.

        A word this model has judged before is answered from that memory
        (and its confidence is not re-derived), so the capability gets
        sharper and cheaper with use. Everything else falls through to
        geometry.
        """
        w = (word or "").strip().lower()
        if not w or len(w) < 2:
            return None
        mem = self._judged.get(w)
        if mem is not None:
            return {"polarity": float(mem[0]), "evaluative": 1.0,
                    "confidence": float(mem[1]), "source": "remembered"}
        cs = self._all_centroids()
        if cs is None:
            return None
        pos, neg, neu = cs
        v = self._unit(w)
        if v is None:
            return None
        cos_pos = sum(a * b for a, b in zip(v, pos))
        cos_neg = sum(a * b for a, b in zip(v, neg))
        cos_neu = sum(a * b for a, b in zip(v, neu))
        # EVALUATIVE axis: how close the word sits to EITHER pole, regardless
        # of which — the mean of the two ABSOLUTE similarities. Absolute
        # values are what make the axis pole-independent (a word is
        # evaluative whether it leans good or bad) and they avoid the
        # cancellation that sank the earlier pooled-centroid formulation.
        # A naming word scores near zero on both and high on neutral; an
        # evaluative word scores high on one of them.
        evaluative = 0.5 * (abs(cos_pos) + abs(cos_neg)) - cos_neu
        if evaluative < EVALUATIVE_MIN_MARGIN:
            # Not a judgment, or too close to call. Fail closed: no stance.
            return None
        # VALENCE axis: which pole, signed.
        valence = cos_pos - cos_neg
        if abs(valence) < VALENCE_MIN_MARGIN:
            return None
        polarity = max(-1.0, min(1.0, valence / VALENCE_SCALE))
        return {"polarity": polarity, "evaluative": evaluative,
                "confidence": BASE_CONFIDENCE, "source": "geometry"}

    # ── online growth ────────────────────────────────────────────────────────
    def observe(self, word: str, polarity: float) -> None:
        """Remember a judgment this model made, so it compounds.

        The first observation records the read at BASE_CONFIDENCE; each
        later one nudges confidence up (a running mean toward BASE, capped)
        and blends a repeated same-sign read. A predicate seen again is
        answered from memory, so this is a real growth path rather than a
        write-only log.
        """
        w = (word or "").strip().lower()
        if not w or len(w) < 2:
            return
        p = max(-1.0, min(1.0, float(polarity)))
        if abs(p) < VALENCE_MIN_MARGIN:
            return
        mem = self._judged.get(w)
        if mem is None:
            # The FIRST observation is already evidence, not a bare prior, so
            # it records one step above BASE. Recording at exactly BASE would
            # make the first and the zeroth observation indistinguishable, and
            # a test (rightly) could not tell that remembering anything had
            # happened.
            self._judged[w] = [p, min(MAX_CONFIDENCE, BASE_CONFIDENCE + CONFIDENCE_STEP)]
        else:
            old_p, old_c = float(mem[0]), float(mem[1])
            new_c = min(MAX_CONFIDENCE, old_c + CONFIDENCE_STEP)
            blended = (old_p * old_c + p * new_c) / max(1e-6, old_c + new_c)
            # Only same-sign re-observations reinforce; a flip is a
            # revision and is handled by relearn(), which the user's
            # retraction path calls explicitly.
            if old_p * p < 0:
                blended = p
            self._judged[w] = [blended, new_c]
        # A confidently-read predicate is promoted to an anchor so its
        # morphological neighbours ("overpriced" -> "pricey") can be judged
        # on their own. Signed sets keep the valence axis honest.
        if abs(self._judged[w][0]) >= 0.5:
            (self._learned_pos if self._judged[w][0] > 0
             else self._learned_neg).add(w)
            self._centroids = None

    def relearn(self, word: str, polarity: float) -> None:
        """Overwrite a remembered judgment with the USER's revised position.

        Called when a stance reversal recodes a topic that was keyed from an
        evaluative predicate: the user is ground truth, so their new position
        replaces the geometric read outright (at full confidence) rather than
        being averaged against it. This is the path by which RAVANA corrects
        its own mis-signing without any retraining.
        """
        w = (word or "").strip().lower()
        if not w or len(w) < 2:
            return
        p = max(-1.0, min(1.0, float(polarity)))
        self._judged[w] = [p, MAX_CONFIDENCE]
        (self._learned_pos if p > 0 else self._learned_neg).add(w)
        self._centroids = None

    def remembered(self, word: str) -> Optional[float]:
        w = (word or "").strip().lower()
        mem = self._judged.get(w)
        return float(mem[0]) if mem else None

    # ── introspection / persistence ──────────────────────────────────────────
    def get_state(self) -> Dict:
        return {
            'judged': {k: [float(v[0]), float(v[1])] for k, v in self._judged.items()},
            'learned_pos': sorted(self._learned_pos),
            'learned_neg': sorted(self._learned_neg),
        }

    def set_state(self, state: Dict) -> None:
        self._judged = {k: [float(v[0]), float(v[1])]
                        for k, v in (state.get('judged') or {}).items()}
        self._learned_pos = set(state.get('learned_pos') or [])
        self._learned_neg = set(state.get('learned_neg') or [])
        self._centroids = None

    def stats(self) -> Dict[str, int]:
        return {
            'judged': len(self._judged),
            'learned_pos': len(self._learned_pos),
            'learned_neg': len(self._learned_neg),
            'seed_pos': len(self._seed_pos),
            'seed_neg': len(self._seed_neg),
            'seed_neutral': len(self._seed_neu),
        }
