"""Retirement: a user retraction is a STORE-LEVEL event, not a fact-store flag.

The defect this module addresses was measured, not guessed (see
tmp/reports/ravana-2026-09-29T1239Z.md, defect 2, and the probe
scratch/_retire_probe.py):

    USER: my brother nikhil plays the shehnai, he is in his thirties
    USER: no, nikhil plays the surbahari, not the shehnai
    USER: which instrument does my brother play now?
    RAVANA: your brother nikhil plays shehnai.       <- the retracted value

The correction was ACKNOWLEDGED and then ignored. Two independent causes:

  1. `UserModel._extract_correction_fact` recognises a closed set of
     correction *shapes* ("X is not Y, it's Z", "X's name is not Y"). The
     natural contrastive tail shape — "<subject> <rel> <NEW>, not <OLD>" —
     is not one of them, so `detected_correction_fact` stayed None and
     NOTHING was written: no new value, no retirement.
  2. Even when a correction IS recognised, `PersonalFactStore.contradict`
     retires the old value ONLY in the fact store. The episodic transcript
     keeps the original utterance verbatim, and `_self_cued_episodic` /
     `_retrieve_episodic` answer from it — so "what did i tell you about my
     brother?" re-surfaces the retracted claim.

Cause 2 is the structural one: RAVANA had no notion that "the user told me
this and then took it back" is a property of the RECORD, independent of which
store happens to hold the string. This module supplies that notion.

Design constraints (skill doctrine):
  * NO authored replies. This module renders nothing user-facing; it only
    mutates stores and records retirements.
  * Seed, not answers. The ledger starts EMPTY. The contrast-marker lexicon is
    a grammar vocabulary (like a stopword list) and it has a real growth path:
    `learn_contrast_marker` registers any token observed in the contrast slot
    of a GROUNDED retraction (one whose retired value was actually found in a
    live store), so an unseen marker becomes usable from experience.
  * Online / incremental. A retraction is learned from one utterance during
    normal operation. Nothing is rebuilt, retrained or re-fitted.
  * Fails closed. A retraction only ever fires when the retired value is
    FOUND in a live store. Nothing to retire -> no-op. RAVANA can never
    "retire" something it does not actually hold, which is what stops this
    from becoming a hallucination source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── contrast grammar ────────────────────────────────────────────────────
# A retraction is a CONTRAST between what the user now asserts and what they
# asserted before. The markers below fall into two GRAMMATICAL classes, and
# the class — not the individual word — decides which side of the marker is
# the rejected value:
#
#   NEGATION markers take a COMPLEMENT. What follows them is what they deny:
#     "nikhil plays the surbahari, not the shehnai"   -> rejects shehnai
#
#   REVISION markers are INTERJECTIVE. They mark the material BEFORE them as
#   needing repair, and the replacement follows:
#     "he moved to berlin, actually he moved to lisbon" -> rejects berlin
#
# Collapsing the two classes into one list is what produced inverted answers
# on interjections ("berlin, nah he moved to lisbon" retiring lisbon).
#
# This is a grammar vocabulary, not a topic list: it says nothing about WHAT
# is being retracted. `learn_contrast_marker` GROWS it at runtime from
# grounded retractions, so a marker RAVANA has never seen still becomes usable
# once a real conversation uses it to retract a value RAVANA actually holds.
#
# A newly LEARNED marker is treated as interjective: that is the conservative
# direction, because an interjection only ever retires a value RAVANA already
# confirmed it held, whereas mis-reading a negation could retire the value the
# user is actually asserting.
NEGATION_MARKERS: set[str] = {
    "not", "n't", "never", "neither", "nor",
}

REVISION_MARKERS: set[str] = {
    # contrast
    "instead", "rather", "else",
    # discourse repair / retraction interjections
    "actually", "correction", "sorry", "meant", "mistake", "wrong",
    "retracted", "withdrawn", "forget", "ignore", "undo", "backtrack",
    "no", "yeah-no",
}

CONTRAST_MARKERS: set[str] = NEGATION_MARKERS | REVISION_MARKERS

# Discourse-repair OPENERS: they signal a retraction but sit BEFORE the new
# assertion, so the replaced value has to be inferred from the store rather
# than read off the utterance.
RETRACTION_OPENERS: Tuple[str, ...] = (
    "no", "actually", "correction", "sorry", "i mean", "meant",
    "scratch that", "forget that", "retract", "take that back",
    "i was wrong", "my mistake", "to correct", "correction:",
)

# Phrases that retire a value without asserting a replacement.
_PURE_RETRACTION = re.compile(
    r"\b(?:forget|ignore|scratch|drop|unlearn)\b[^.]*?\bwhat i (?:said|told)\b"
    r"|\bscratch that\b|\btake (?:that|it) back\b|\bretract\b"
    r"|\bforget (?:that|what i said)\b|\bunlearn\b",
    re.IGNORECASE,
)

_WORD = re.compile(r"[a-z']+")

# Determiners and possessive pronouns. Never the head of a VALUE, so they are
# stripped before a value is compared (see RetirementLedger._same_value).
_DETERMINERS = {
    "the", "a", "an", "my", "his", "her", "their", "its", "our", "your",
    "that", "this", "those", "these",
}

# Closed-class relation heads: copulas, auxiliaries, prepositions and the
# participles that head a predicate. A GRAMMAR vocabulary — it says where a
# predicate ends, never what the value is. Like CONTRAST_MARKERS it is
# extendable at runtime via RetirementLedger.learn_contrast_marker for the
# marker side; this side is deliberately closed because mis-detecting a
# relation head would mis-slice a real value.
_RELATION_HEADS: set[str] = {
    # copula / auxiliary
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "does", "do", "did",
    # prepositions
    "to", "in", "at", "on", "for", "with", "from", "by", "of", "into",
    "near", "about", "around", "under", "over", "as",
    # discourse / coordination
    "but", "and", "then", "that", "which",
    # common predicate heads that are not inflected forms
    "plays", "play", "uses", "use", "owns", "own", "lives", "live",
    "prefers", "prefer", "likes", "like", "wants", "want", "needs", "need",
    "reads", "read", "writes", "write", "speaks", "speak", "drives", "drive",
    "named", "called", "born", "based", "works", "work", "studies", "study",
}


def _toks(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


def _stem(word: str) -> str:
    """Extremely small suffix stripper.

    Morphology variation ("shehnai"/"shehnais", "surbahari"/"surbaharis")
    must not read as two different values, or the retirement would silently
    miss its target. Kept deliberately crude: it exists to make a value
    match itself under inflection, not to be linguistics.
    """
    w = word.lower().strip("'")
    for suf in ("ies", "es", "s", "ed", "ing"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def _stemset(text: str) -> set[str]:
    return {_stem(w) for w in _toks(text)}


@dataclass
class Retraction:
    """A grounded retraction the user made in one utterance.

    ``replaced_by`` is None when the user retracted without asserting a new
    value ("forget what i said about the shehnai").
    """

    retired_value: str
    replaced_by: Optional[str] = None
    marker: Optional[str] = None
    slot: Optional[str] = None          # resolved (subject, attribute) if known
    turn_index: int = 0
    # The utterance that carried the retraction, kept so a later "what did i
    # used to say / what did you get wrong" can be answered from the real
    # trace instead of world-knowledge. Empty until grounded.
    evidence: str = ""


@dataclass
class RetirementLedger:
    """Persistent record of what the user has retracted, and why.

    A STORE, not a table of answers: it is empty until real conversations
    populate it, it grows online, and it rides the engine's existing
    pickle/SQLite state (see ``to_state`` / ``from_state``).
    """

    #: normalized slot key -> retired value (lower). The live fact for a slot
    #: is whatever is in the stores and NOT in here.
    retired: Dict[str, List[str]] = field(default_factory=dict)
    #: turn_index at which each retirement happened, for "what did you get
    #: wrong recently" (time-cells, same shape as the episodic index).
    retired_at: Dict[str, int] = field(default_factory=dict)
    #: Every grounded retraction, newest last, with its evidence text.
    log: List[Retraction] = field(default_factory=list)
    #: Contrast markers RAVANA has confirmed ONLINE (learned, not authored).
    learned_markers: set[str] = field(default_factory=set)

    # ── mutation ──────────────────────────────────────────────────────
    def retire(self, retraction: Retraction, slot_key: str) -> None:
        val = (retraction.retired_value or "").strip().lower()
        if not val:
            return
        bucket = self.retired.setdefault(slot_key, [])
        if val not in bucket:
            bucket.append(val)
        self.retired_at[val] = retraction.turn_index
        if retraction.marker:
            self.learn_contrast_marker(retraction.marker)
        self.log.append(retraction)
        if len(self.log) > 200:
            del self.log[:-200]

    def learn_contrast_marker(self, marker: str) -> bool:
        """Register a contrast marker seen in a GROUNDED retraction.

        Growth path for the marker lexicon: only markers that actually
        co-occurred with a retirement of a value RAVANA held are learned, so
        the vocabulary cannot grow from noise.

        A learned marker is registered as INTERJECTIVE (see
        ``learned_interjections``) — the conservative direction. An
        interjection can only retire a value RAVANA already confirmed it
        held; a mis-read negation could retire the value the user is
        currently asserting.
        """
        m = (marker or "").strip().lower()
        if not m or m in CONTRAST_MARKERS:
            return False
        self.learned_markers.add(m)
        return True

    def all_markers(self) -> set[str]:
        return set(CONTRAST_MARKERS) | self.learned_markers

    def is_interjective(self, marker: str) -> bool:
        """Does this marker reject the PRECEDING material (vs a following
        complement)? Learned markers count as interjective by construction."""
        m = (marker or "").strip().lower()
        return m in REVISION_MARKERS or m in self.learned_markers

    def interjective_markers(self) -> set[str]:
        """The full interjective class: the AUTHORED revision markers plus
        every marker learned online.

        Callers that pass an interjective set to :func:`parse_contrast` must
        pass THIS, not :attr:`learned_markers`. ``parse_contrast`` replaces
        its ``REVISION_MARKERS`` default whenever an explicit set is given, so
        handing it the learned set alone (empty until something is learned)
        silently reclassifies every authored interjection as a negation — which
        inverts the direction of the parse and retires the value the user just
        ASSERTED. Measured: "pune, actually he lives in patna" parsed as
        rejected="pune" / asserted="he lives in patna" with the learned-only
        set, and correctly as rejected="pune" / asserted="patna" with this one.
        """
        return set(REVISION_MARKERS) | self.learned_markers

    # ── query ─────────────────────────────────────────────────────────
    def _same_value(self, a: str, b: str) -> bool:
        """Do two value strings denote the same thing to RAVANA?

        Compared stem-wise on the whole phrase AND on its head noun, so an
        inflected or determiner-prefixed surface form ("the shehnai" /
        "shehnais") still matches the retirement recorded for "shehnai".
        Without this the retirement silently misses its own target and the
        stale value keeps being served — the exact failure it exists to fix.
        """
        na = " ".join(_stem(w) for w in _toks(a))
        nb = " ".join(_stem(w) for w in _toks(b))
        if not na or not nb:
            return False
        if na == nb:
            return True
        # Head-noun fallback: "the shehnai" vs "shehnai", "old surbahari" vs
        # "surbahari". A single-word stem match is enough evidence, because a
        # retirement only ever suppresses a value; it never asserts one.
        aw = [w for w in _toks(a) if w not in _DETERMINERS]
        bw = [w for w in _toks(b) if w not in _DETERMINERS]
        if aw and bw and _stem(aw[-1]) == _stem(bw[-1]):
            return True
        return False

    def is_retired(self, value: str, slot_key: Optional[str] = None) -> bool:
        """Is this value retired, for this slot (or in any slot)?"""
        if not (value or "").strip():
            return False
        if slot_key is not None and any(
                self._same_value(value, v)
                for v in self.retired.get(slot_key, [])):
            return True
        return any(self._same_value(value, v)
                   for vals in self.retired.values() for v in vals)

    def retired_slots(self) -> List[str]:
        return sorted(self.retired)

    def active_in(self, slot_key: str) -> bool:
        return bool(self.retired.get(slot_key))

    # ── serialization (rides existing pickle/SQLite persistence) ─────
    def to_state(self) -> Dict[str, Any]:
        return {
            "retired": {k: list(v) for k, v in self.retired.items()},
            "retired_at": dict(self.retired_at),
            "log": [
                {
                    "retired_value": r.retired_value,
                    "replaced_by": r.replaced_by,
                    "marker": r.marker,
                    "slot": list(r.slot) if r.slot else None,
                    "turn_index": r.turn_index,
                    "evidence": r.evidence,
                }
                for r in self.log
            ],
            "learned_markers": sorted(self.learned_markers),
        }

    @classmethod
    def from_state(cls, state: Optional[Dict[str, Any]]) -> "RetirementLedger":
        led = cls()
        if not state:
            return led
        led.retired = {k: list(v) for k, v in (state.get("retired") or {}).items()}
        led.retired_at = dict(state.get("retired_at") or {})
        led.learned_markers = set(state.get("learned_markers") or [])
        for r in state.get("log") or []:
            led.log.append(
                Retraction(
                    retired_value=r.get("retired_value", ""),
                    replaced_by=r.get("replaced_by"),
                    marker=r.get("marker"),
                    slot=tuple(r["slot"]) if r.get("slot") else None,
                    turn_index=int(r.get("turn_index") or 0),
                    evidence=r.get("evidence", ""),
                )
            )
        return led


# ── detection ──────────────────────────────────────────────────────────


def _object_of(phrase: str) -> str:
    """Reduce "<subject> <relation> <value>" to the VALUE (the object).

    The contrast grammar tells us WHICH side is rejected; it does not tell us
    where the rejected value ENDS inside its own clause. In "nikhil plays the
    surbahari" the value is the object of the relation verb, and in
    "he moved to lisbon" it is the object of a preposition. Taking the tail
    after the last relation/participle head does both without a per-topic
    verb list; when the phrase is already a bare noun phrase it is returned
    unchanged.
    """
    words = phrase.split()
    if len(words) <= 2:
        return phrase.strip(" .,!?")
    # Drop leading determiners/possessives, which never head a value.
    while words and words[0].lower() in ("the", "a", "an", "my", "his", "her",
                                         "their", "its", "our", "your"):
        words = words[1:]
    if len(words) <= 1:
        return phrase.strip(" .,!?")
    # A past participle / copula / preposition immediately before the final
    # noun marks the relation head. Walk from the right and keep the first
    # noun-phrase-sized tail.
    for i in range(len(words) - 1, 0, -1):
        w = words[i - 1].lower().strip(",")
        if w in _RELATION_HEADS or w.endswith("ed") and len(w) > 4:
            return " ".join(words[i:]).strip(" .,!?")
    return " ".join(words[-2:]).strip(" .,!?") if len(words) > 2 else phrase.strip(" .,!?")


def parse_contrast(text: str, markers: Optional[set[str]] = None,
                   interjective: Optional[set[str]] = None
                   ) -> Optional[Tuple[str, str, str]]:
    """Split an utterance into (asserted, rejected, marker) around a contrast.

    Direction is decided by the marker's GRAMMATICAL CLASS, which is the
    general rule rather than a per-shape table:

    * NEGATION marker ("not") takes a complement, so the REJECTED value is
      what FOLLOWS it:
          "nikhil plays the surbahari, not the shehnai"
              rejected = "the shehnai", asserted = "the surbahari"
    * INTERJECTIVE marker ("actually", a learned marker like "nah") marks the
      PRECEDING material as needing repair, so the REJECTED value is what
      PRECEDES it:
          "he moved to berlin, nah he moved to lisbon"
              rejected = "berlin", asserted = "lisbon"

    The caller still has to GROUND the rejected phrase in a live store before
    anything is retired — parsing a contrast is cheap and easy to get wrong,
    grounding it is not.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    mk = set(markers if markers is not None else CONTRAST_MARKERS)
    inter = set(interjective) if interjective is not None else set(REVISION_MARKERS)

    # Split into clauses, keeping each clause's tokens.
    clauses: List[List[str]] = []
    for c in re.split(r"[,;]|\bbut\b", raw):
        parts = c.strip().split()
        if parts:
            clauses.append(parts)
    if len(clauses) < 2:
        return None

    mi = mj = None
    marker = None
    for i, cl in enumerate(clauses):
        for j, w in enumerate(cl):
            bare = w.lower().strip(".").strip("'")
            canon = bare.replace("n't", "not")
            if canon in mk or bare in mk:
                mi, mj, marker = i, j, canon
    if mi is None or marker is None:
        return None

    if marker in inter:
        # INTERJECTIVE: rejects what precedes, replaced by what follows.
        # Bounded by the clause break so the marker's own clause tail is not
        # swallowed.
        if mi == 0:
            return None
        rejected = _object_of(" ".join(w for c in clauses[:mi] for w in c)).lower()
        asserted = " ".join(w for c in clauses[mi:] for w in c[mj + 1:])
    else:
        # NEGATION: rejects the complement that follows it, bounded by the
        # clause break; the assertion is everything before the marker.
        if mj == 0 and mi == 0:
            rejected = " ".join(clauses[0][1:]).strip(" .,!?").lower()
            asserted = " ".join(w for c in clauses[1:] for w in c)
        elif mj == 0:
            rejected = " ".join(clauses[mi][1:]).strip(" .,!?").lower()
            asserted = " ".join(w for c in clauses[:mi] for w in c)
        else:
            rejected = _object_of(" ".join(clauses[mi][:mj])).lower()
            asserted = " ".join(w for c in clauses[mi:] for w in c[mj + 1:])
    if not asserted or not rejected:
        return None

    asserted = re.sub(
        r"\b(?:it'?s|it\s+is|that'?s|he'?s|she'?s|they'?re)\s+",
        "", asserted, flags=re.IGNORECASE,
    ).strip()
    asserted = re.sub(r"^(?:no|nah|actually|correction|sorry)\b[, ]*", "",
                      asserted, flags=re.IGNORECASE).strip()
    asserted = _object_of(asserted)
    if not asserted or not rejected:
        return None
    if asserted.lower() == rejected:
        return None
    return asserted, rejected, marker


def looks_like_retraction(text: str, markers: Optional[set[str]] = None) -> bool:
    """True when the utterance opens as a correction or names a retraction.

    Cheap pre-filter so the common case (an ordinary disclosure) costs one
    regex rather than a store scan.
    """
    low = (text or "").lower().strip()
    if not low:
        return False
    if _PURE_RETRACTION.search(low):
        return True
    head = " ".join(_toks(low)[:4])
    return any(op in head for op in RETRACTION_OPENERS)
