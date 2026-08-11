"""Learned personal-fact store for the user profile.

Brain basis
-----------
- vmPFC confidence coding (Clairis & Pessiglione 2022; Lebreton et al. 2015):
  personal knowledge is graded, not binary. Every fact carries a confidence.
- Complementary Learning Systems (McClelland et al. 1995): personal facts start
  as fast hippocampal episodes and become stable through repetition / rehearsal.
- Prediction-error learning (Rescorla-Wagner; ACC ERN analog already in
  UserModel._detect_correction): confidence rises on confirmation, drops on
  correction.

Design (mirrors BeliefStore exactly)
------------------------------------
High-precision regex seeds assert facts at confidence ~0.6 (source="seed_regex").
From there the store LEARNS: repetition reinforces, "yes/that's right" confirms
(strong boost), "no, it's X" contradictions, low-confidence unconfirmed facts
prune after a quiet window. Serialization is via get_state()/set_state() so the
store rides the engine's existing pickle+SQLite persistence -- no new database.

Multiple values for the same (subject, attribute) coexist as separate fact
records (keyed by subject|attribute|value) so a knowledge update preserves the
prior value for reconcile() to resolve by confidence x recency, rather than
silently overwriting it.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import re


@dataclass
class PersonalFact:
    subject: str                 # entity the fact is about, e.g. "cat"
    attribute: str               # property, e.g. "name"
    value: str                   # value, e.g. "pixel"
    confidence: float = 0.6      # 0-1, starts at seed confidence
    turn_number: int = 0
    rehearsal_count: int = 1
    source: str = "seed_regex"   # seed_regex | user_confirmation | correction | repetition
    superseded: bool = False      # marked when a newer value wins the battle


class PersonalFactStore:
    def __init__(self, decay_turns: int = 50):
        # (subject.lower(), attribute.lower(), value.lower()) -> PersonalFact
        self.facts: Dict[Tuple[str, str, str], PersonalFact] = {}
        self.contradictions: List[Tuple] = []
        self.turn_num: int = 0
        self._decay_turns = decay_turns

    # ── turn clock ────────────────────────────────────────────────
    def advance_turn(self):
        self.turn_num += 1

    # ── core assertions ───────────────────────────────────────────
    def _key(self, subject: str, attribute: str, value: str) -> Tuple[str, str, str]:
        return (subject.lower().strip(), attribute.lower().strip(), value.lower().strip())

    def assert_fact(self, subject: str, attribute: str, value: str,
                    confidence: float = 0.6, source: str = "seed_regex") -> None:
        """Store or reinforce a personal fact.

        If the exact (subject, attribute, value) already exists, reinforce it.
        If a DIFFERENT value for the same (subject, attribute) exists, open a
        contradiction (both values kept as separate records for reconcile()).
        """
        key = self._key(subject, attribute, value)
        val = (value or "").strip()
        existing = self.facts.get(key)
        if existing is not None:
            existing.confidence = min(1.0, existing.confidence + 0.1)
            existing.rehearsal_count += 1
            existing.turn_number = self.turn_num
            if source in ("user_confirmation", "correction"):
                existing.source = source
            existing.superseded = False
            return
        # New value. Check for a conflicting prior value on this (subj, attr).
        for (s, a, v), f in self.facts.items():
            if s == subject.lower().strip() and a == attribute.lower().strip() \
                    and v != val.lower().strip() and not f.superseded:
                self.contradictions.append(
                    ((s, a, v), (s, a, val.lower().strip()), self.turn_num))
        self.facts[key] = PersonalFact(
            subject=subject, attribute=attribute, value=val,
            confidence=confidence, turn_number=self.turn_num,
            rehearsal_count=1, source=source)

    def reinforce(self, subject: str, attribute: str, value: Optional[str] = None) -> None:
        best = self.get(subject, attribute, value)
        if best is None:
            return
        best.confidence = min(1.0, best.confidence + 0.1)
        best.rehearsal_count += 1
        best.turn_number = self.turn_num

    def confirm(self, subject: str, attribute: str, value: str) -> None:
        """User feedback 'yes / that's right' on a previously asserted fact.

        Dramatically boosts confidence (prediction-error confirmation). If no
        matching slot exists we take the user at their word at high confidence
        (they are the ground truth for their own profile).
        """
        val = (value or "").strip()
        existing = self.get(subject, attribute, value=val if val else None)
        if existing is None:
            self.assert_fact(subject, attribute, val,
                             confidence=0.85, source="user_confirmation")
            return
        if val and existing.value.lower() != val.lower():
            self.contradict(subject, attribute, val)
            return
        existing.confidence = min(1.0, existing.confidence + 0.25)
        existing.rehearsal_count += 1
        existing.turn_number = self.turn_num
        existing.source = "user_confirmation"
        existing.superseded = False

    def contradict(self, subject: str, attribute: str, new_value: str) -> None:
        """User feedback 'no, it's X' -> assert the corrected value and mark the
        prior value(s) superseded. A user correction is authoritative: the new
        value becomes the active fact and the previously-held value is retired
        (kept in the contradiction log for reconcile(), but not returned by
        queries)."""
        subj, attr = subject.lower().strip(), attribute.lower().strip()
        new_val = (new_value or "").strip().lower()
        # Retire any active prior value for this (subject, attribute).
        for (s, a, v), f in self.facts.items():
            if s == subj and a == attr and not f.superseded and v != new_val:
                f.superseded = True
        self.assert_fact(subject, attribute, new_value,
                         confidence=0.7, source="correction")

    # ── queries ───────────────────────────────────────────────────
    def query_fact(self, subject: str, attribute: Optional[str] = None
                   ) -> List[PersonalFact]:
        """Return matching facts sorted by confidence x recency (best first)."""
        subj = subject.lower().strip()
        out = [f for (s, a, v), f in self.facts.items()
               if s == subj and (attribute is None or a == attribute.lower().strip())
               and not f.superseded]
        out.sort(key=self._decay_score, reverse=True)
        return out

    def get(self, subject: str, attribute: str,
            value: Optional[str] = None) -> Optional[PersonalFact]:
        subj, attr = subject.lower().strip(), attribute.lower().strip()
        cands = [f for (s, a, v), f in self.facts.items()
                 if s == subj and a == attr and not f.superseded
                 and (value is None or v == value.lower().strip())]
        if not cands:
            return None
        return max(cands, key=self._decay_score)

    # ── decay / consolidation helpers ──────────────────────────────
    def _decay_score(self, f: PersonalFact) -> float:
        recency = 1.0 / (1.0 + (self.turn_num - f.turn_number) * 0.1)
        return f.confidence * recency

    def reconcile(self) -> Dict[Tuple[str, str], PersonalFact]:
        """Resolve contradictions by confidence x recency (recent wins)."""
        resolved: Dict[Tuple[str, str], PersonalFact] = {}
        groups: Dict[Tuple[str, str], List[PersonalFact]] = {}
        for old_triple, new_triple, _c in self.contradictions:
            key = (old_triple[0], old_triple[1])
            old_val, new_val = old_triple[2], new_triple[2]
            for f in self.facts.values():
                if (f.subject.lower(), f.attribute.lower()) == key and \
                        f.value.lower() in (old_val.lower(), new_val.lower()):
                    groups.setdefault(key, []).append(f)
        for key, cands in groups.items():
            seen = set()
            uniq = [c for c in cands if not (c.value.lower() in seen or seen.add(c.value.lower()))]
            if len(uniq) < 2:
                continue
            winner = max(uniq, key=self._decay_score)
            for c in uniq:
                c.superseded = (c is not winner)
            resolved[key] = winner
        self.contradictions = []
        return resolved

    def prune_stale(self, min_confidence: float = 0.4,
                    stale_after: int = 10) -> int:
        """Forget low-confidence facts never reinforced (mirrors BeliefStore)."""
        to_remove = [k for k, f in self.facts.items()
                     if f.confidence < min_confidence
                     and (self.turn_num - f.turn_number) >= stale_after]
        for k in to_remove:
            del self.facts[k]
        return len(to_remove)

    def get_consolidation_candidates(self, min_confidence: float = 0.6
                                     ) -> List[PersonalFact]:
        """Facts confident + rehearsed enough to graduate to the graph.

        Same contract shape as HippocampalBuffer.get_consolidation_candidates:
        returns the strongest candidates for _sleep_consolidate to drain.
        """
        cands = [f for f in self.facts.values()
                 if not f.superseded and f.confidence >= min_confidence
                 and f.rehearsal_count >= 2]
        cands.sort(key=lambda f: f.confidence * f.rehearsal_count, reverse=True)
        return cands

    # ── serialization (rides existing pickle/SQLite persistence) ───
    def get_state(self) -> Dict:
        return {
            'facts': {f"{k[0]}|{k[1]}|{k[2]}": (f.subject, f.attribute, f.value,
                                               f.confidence, f.turn_number,
                                               f.rehearsal_count, f.source,
                                               f.superseded)
                      for k, f in self.facts.items()},
            'contradictions': self.contradictions,
            'turn_num': self.turn_num,
        }

    def set_state(self, state: Dict) -> None:
        self.facts = {}
        for k, v in state.get('facts', {}).items():
            s, a, val, conf, tn, rc, src, sup = v
            self.facts[(s.lower(), a.lower(), val.lower())] = PersonalFact(
                subject=s, attribute=a, value=val, confidence=conf,
                turn_number=tn, rehearsal_count=rc, source=src, superseded=sup)
        self.contradictions = state.get('contradictions', [])
        self.turn_num = state.get('turn_num', 0)


@dataclass
class Stance:
    topic: str               # subject of the opinion, e.g. "cats"
    polarity: float = 0.0     # -1.0 (against) .. +1.0 (for); 0 = neutral/uncertain
    confidence: float = 0.5   # 0-1, how strongly held
    valence: float = 0.0      # emotional valence at expression time
    arousal: float = 0.0      # emotional arousal at expression time
    turn_number: int = 0
    rehearsal_count: int = 1


class UserStanceStore:
    """Learned store of the user's OPINIONS (value judgments), kept separate
    from biographical facts (PersonalFactStore).

    Brain basis (C): OFC/vmPFC computes subjective value (for/against) on a
    circuit distinct from the hippocampal semantic system (Levy & Glimcher
    2012; Clairis & Pessiglione 2022). An opinion is a polarity + confidence,
    NOT an attribute=value assertion, so it must never be flattened into the
    fact store or the knowledge graph as if it were a property of the world.
    Stances decay FASTER than facts (halflife ~20 turns vs ~50) because social
    attitudes are more malleable than biographical memory.
    """
    def __init__(self, decay_turns: int = 20):
        # topic.lower() -> Stance
        self.stances: Dict[str, Stance] = {}
        self.turn_num: int = 0
        self._decay_turns = decay_turns
        # Transient record of the most recent stance REVERSAL this turn, so the
        # engine can acknowledge a retraction LINKED to the prior stance (it is
        # consumed by the ack composer and cleared, never serialized as truth).
        # topic -> (old_polarity, new_polarity)
        self.last_reversal: Optional[Tuple[str, float, float]] = None
        # Idempotency guard keyed by the NORMALIZED UTTERANCE, not turn_num:
        # a concession/retraction is mined TWICE within one process_turn — once
        # by the early gate (mine_personal_facts @ engine.py:2977) and once by
        # the self_disclosure -> observe_user_query -> mine_personal_facts path
        # (@ engine.py:2100/1284) — and the fact-store turn clock is advanced
        # BETWEEN them (0 -> 1 in observe_user_query). A turn_num-keyed guard
        # therefore misses the second fire, double-flips the stance and
        # overwrites last_reversal's old-polarity with the already-reversed
        # value (corrupting the ack framing). Keying on the utterance text makes
        # repeated mining of the SAME utterance idempotent regardless of when in
        # the turn it is seen (matches the docstring on reverse_stance).
        self._reversed_utterance: Dict[str, str] = {}

    def clear_last_reversal(self):
        self.last_reversal = None

    def clear_reversal_guard(self):
        """Reset the per-utterance idempotency guard at the START of a turn.

        The guard suppresses repeated mining of the same utterance within one
        process_turn (it is mined twice: early gate + self_disclosure observe
        path). It MUST NOT be reset inside advance_turn(), because observe_user_query
        calls advance_turn() between the two mining calls — resetting there would
        clear the guard and allow the second mine to double-flip. Resetting here,
        once per user turn, correctly scopes the guard to a single turn.
        """
        self._reversed_utterance = {}

    def advance_turn(self):
        self.turn_num += 1

    def express_stance(self, topic: str, polarity: float,
                       confidence: float = 0.5, valence: float = 0.0,
                       arousal: float = 0.0, source: str = "seed_regex") -> None:
        """Store or weighted-merge a stance on `topic`.

        Repeats shift polarity toward the new signal and raise confidence
        (running mean), mirroring how repeated expression entrenches attitude.
        """
        key = topic.lower().strip()
        existing = self.stances.get(key)
        if existing is None:
            self.stances[key] = Stance(
                topic=topic, polarity=float(polarity), confidence=float(confidence),
                valence=valence, arousal=arousal,
                turn_number=self.turn_num, rehearsal_count=1)
            return
        _n = existing.rehearsal_count + 1
        _w_old = existing.confidence * existing.rehearsal_count
        _w_new = confidence
        existing.polarity = (_w_old * existing.polarity + _w_new * polarity) / max(1e-6, _w_old + _w_new)
        existing.confidence = min(1.0, (existing.confidence + confidence) / 2.0 + 0.05)
        existing.valence = (existing.valence * existing.rehearsal_count + valence) / _n
        existing.arousal = (existing.arousal * existing.rehearsal_count + arousal) / _n
        existing.rehearsal_count = _n
        existing.turn_number = self.turn_num

    def query_stance(self, topic: str) -> Optional[Stance]:
        return self.stances.get(topic.lower().strip())

    def resolve_topic(self, phrase: str) -> Optional[str]:
        """Link a mention/topic phrase to the single most likely stored stance.

        The spoken phrase rarely reproduces the stored stance key verbatim
        ("plastic bans" vs the phrase "plastic bans"). Match stored topics by
        exact key, substring, then content-word overlap, returning the strongest
        link. Returns None when no stored stance plausibly matches — so recall
        and reversal never fabricate a read on a topic the user has no stance on.
        """
        stances = self.stances
        if not stances:
            return None
        head = (phrase or "").lower().strip()
        if head in stances:
            return head
        for k in stances:
            if head and (head in k or k in head):
                return k
        hw = set(re.findall(r"[a-z']+", head))
        best, best_j = None, 0.0
        for k in stances:
            kw = set(re.findall(r"[a-z']+", k))
            if not hw or not kw:
                continue
            j = len(hw & kw) / len(hw | kw)
            if j > best_j:
                best, best_j = k, j
        return best if best_j >= 0.4 else None

    def reinforce(self, topic: str) -> None:
        s = self.stances.get(topic.lower().strip())
        if s is None:
            return
        s.confidence = min(1.0, s.confidence + 0.1)
        s.rehearsal_count += 1
        s.turn_number = self.turn_num

    def reverse_stance(self, topic: str, reversal_strength: float = 0.85,
                       utterance: Optional[str] = None) -> Optional[Stance]:
        """Flip the user's stance on `topic` (e.g. "i take back X").

        A retraction is an attitude CHANGE recoding, not a fresh merge: the
        subjective value the user expressed on the topic is recalibrated toward
        the OPPOSITE pole, and confidence drops (the brain's valuation of the
        topic becomes uncertain after reversal — vmPFC re-evaluation). The link
        to the PRIOR stance is preserved so the ack can reference what was
        reversed.

        The recalibration strength is NOT fixed at the full 180°. A hard recant
        ("i was wrong about X", "i take it all back") should land opposite;
        a *softening* ("X isn't so bad, i was too hasty", "i came around a bit")
        should relax the prior stance toward neutral, never invert it into the
        opposite conviction. So the blend magnitude is driven by the utterance:
        a soft cue halves the reversal strength, a hard cue keeps it. Callers
        pass `soft=True` for softening idioms. Returns the PRIOR stance (so
        callers can render a linked acknowledgment), or None if the user had no
        stance on this topic (a benign "take back" with nothing to reverse —
        never fabricates a stance).

        Idempotent within a turn: repeated mining of the same utterance cannot
        flip the stance more than once.
        """
        key = topic.lower().strip()
        existing = self.stances.get(key)
        if existing is None:
            return None
        # Already reversed this SAME utterance / this turn's mining epoch.
        # Key = normalized utterance when one is supplied (so repeated mining
        # of the same utterance idempotently suppresses even when the fact-store
        # clock advanced between the two mining calls within a process_turn),
        # else fall back to the turn clock (original unit-level within-turn
        # idempotency for direct callers that pass no utterance).
        _norm = re.sub(r"\s+", " ", (utterance or "").lower().strip())
        _guard_key = _norm if _norm else self.turn_num
        if self._reversed_utterance.get(key) == _guard_key:
            return existing
        old_polarity = existing.polarity
        old_confidence = existing.confidence
        # Softening relaxes toward neutral; hard recant flips decisively. A
        # partial reversal never crosses the pivot, so "olives aren't that bad"
        # lands near neutral instead of converting the user into an olive-lover.
        blend = min(reversal_strength, 0.5) if getattr(self, "_soft_reversal", False) else reversal_strength
        pivot = -old_polarity
        existing.polarity = old_polarity * (1.0 - blend) + pivot * blend
        # Attitude change injects uncertainty: drop confidence toward the pivot.
        existing.confidence = max(0.1, existing.confidence * (1.0 - blend * 0.6))
        existing.rehearsal_count += 1
        existing.turn_number = self.turn_num
        self._reversed_utterance[key] = _guard_key
        self.last_reversal = (existing.topic, old_polarity, existing.polarity)
        return existing

    def _decay_score(self, s: Stance) -> float:
        recency = 1.0 / (1.0 + (self.turn_num - s.turn_number) * 0.25)
        return s.confidence * recency

    def prune_stale(self, min_confidence: float = 0.3,
                    stale_after: int = 8) -> int:
        to_remove = [k for k, s in self.stances.items()
                     if s.confidence < min_confidence
                     and (self.turn_num - s.turn_number) >= stale_after]
        for k in to_remove:
            del self.stances[k]
        return len(to_remove)

    def get_consolidation_candidates(self) -> List[Stance]:
        cands = [s for s in self.stances.values()
                 if s.confidence >= 0.5 and s.rehearsal_count >= 2]
        cands.sort(key=self._decay_score, reverse=True)
        return cands

    def get_state(self) -> Dict:
        return {
            'stances': {k: (v.topic, v.polarity, v.confidence, v.valence,
                            v.arousal, v.turn_number, v.rehearsal_count)
                       for k, v in self.stances.items()},
            'turn_num': self.turn_num,
        }

    def set_state(self, state: Dict) -> None:
        self.stances = {}
        for k, v in state.get('stances', {}).items():
            topic, pol, conf, val, aro, tn, rc = v
            self.stances[k.lower().strip()] = Stance(
                topic=topic, polarity=pol, confidence=conf, valence=val,
                arousal=aro, turn_number=tn, rehearsal_count=rc)
        self.turn_num = state.get('turn_num', 0)


# ── Quantitative possession / activity memory ────────────────────────────────
# A structured store for COUNT-bearing disclosures ("i keep twelve racing
# pigeons", "i have three cats", "i bake two sourdough loaves", "i lost five
# hens"). The activity/event miner already PRESERVES the number in the stored
# 'does'/'event' value (e.g. "keep twelve racing pigeons"), but that is a raw
# text blob: there is no way to SYNTHESIZE a clean count answer ("how many
# racing pigeons do i keep" -> "twelve") and no way to AGGREGATE across the
# store ("how many pets do i have in total"). This store captures the
# (subject, kind, count, noun) as structured fields so both are possible.
#
# Brain basis: the hippocampus stores the episode, but the vmPFC also binds a
# QUANTITY to a possession schema (how many of my X) independently of the
# episodic gist — that is what lets a human answer "how many" without replaying
# the whole memory. Decoupling the number from the gist sentence is the
# structural fix for the report's line-86 limitation.
#
# SEED + ONLINE (no retraining, no authored answers): the number-word lexicon
# is a tiny seed (one..twelve, dozen, half dozen, a/an=1) extended at runtime by
# learn_number() so RAVANA grows its own number vocabulary; the verb->category
# map is a small seed of everyday possession/activity verbs, not a per-topic
# table. Both are overridable by the user at runtime via correct()/supersede.
_NUMBER_WORDS: Dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "dozen": 12, "half dozen": 6, "half a dozen": 6,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_NUMBER_WORDS_LEARNED: Dict[str, int] = {}
# inverse map int -> word, for rendering recalled counts as words (1-12).
_NUMBER_WORDS_INV: Dict[int, str] = {
    v: k for k, v in _NUMBER_WORDS.items()
    if " " not in k and "dozen" not in k and "half" not in k}


def learn_number(word: str, value: int) -> None:
    """Grow the number-word lexicon from a live disclosure."""
    w = (word or "").strip().lower()
    if w and isinstance(value, int) and value > 0:
        _NUMBER_WORDS_LEARNED[w] = value


def number_to_int(token: str) -> Optional[int]:
    """Parse a number word or digit to int, or None if not a number."""
    t = (token or "").strip().lower()
    if not t:
        return None
    if t.isdigit():
        return int(t)
    return _NUMBER_WORDS.get(t) or _NUMBER_WORDS_LEARNED.get(t)


def render_count(count: int) -> str:
    """Render a small integer count as a word when natural (1-12), else digit.

    Mirrors the mined surface form: the activity/event miner stores counts as
    words ("keep six hives"), so recall should echo "six", not "6", to stay
    consistent with what the user said. Larger counts render as digits.
    """
    return _NUMBER_WORDS_INV.get(count, str(count))


@dataclass
class QuantityRecord:
    subject: str               # entity the quantity is about, e.g. "i"
    kind: str                  # possession verb family, e.g. "keep"/"have"/"bake"/"lose"
    count: int                 # parsed integer count
    noun_phrase: str           # full surface noun phrase, e.g. "racing pigeons"
    noun_canonical: str        # canonical singular noun for matching/aggregation
    category: str = "possession"  # possession | made | loss (for totals)
    confidence: float = 0.6
    turn_number: int = 0
    source: str = "seed_regex"
    superseded: bool = False


class QuantityMemory:
    """Structured count store for the user's possessions / activities.

    Keyed by (subject, kind, noun_canonical, count) so a correction or update
    opens a contradiction the same way PersonalFactStore does (active value
    wins; the user is ground truth). Distinct from the 'does'/'event' text
    facts: those preserve the gist sentence; this store preserves the NUMBER so
    it can be retrieved and summed. Rides the engine's existing pickle/SQLite
    persistence via get_state()/set_state() — no new database.
    """

    # verb family -> (kind, category). Seed; covers everyday disclosures.
    # Past-tense forms are included so "i kept two rabbits" / "i had three
    # dogs" / "i made five loaves" are captured the same way (the verb is
    # normalized to its present family before lookup). Extendable, not a
    # per-topic table.
    VERB_KIND = {
        "keep": ("keep", "possession"), "keep on": ("keep", "possession"),
        "kept": ("keep", "possession"),
        "have": ("have", "possession"), "have on": ("have", "possession"),
        "had": ("have", "possession"),
        "own": ("own", "possession"), "raise": ("raise", "possession"),
        "raised": ("raise", "possession"),
        "breed": ("breed", "possession"), "bred": ("breed", "possession"),
        "run": ("run", "possession"), "ran": ("run", "possession"),
        "grow": ("grow", "possession"), "grew": ("grow", "possession"),
        "bake": ("bake", "made"), "make": ("make", "made"),
        "made": ("make", "made"), "cook": ("cook", "made"),
        "brew": ("brew", "made"),
        "lose": ("lose", "loss"), "lost": ("lose", "loss"),
        "drop": ("drop", "loss"), "dropped": ("drop", "loss"),
        "break": ("break", "loss"), "broke": ("break", "loss"),
    }

    def __init__(self, decay_turns: int = 50):
        self.records: Dict[Tuple[str, str, str, int], QuantityRecord] = {}
        self.turn_num: int = 0
        self._decay_turns = decay_turns

    def advance_turn(self):
        self.turn_num += 1

    def _key(self, subject, kind, noun_canonical, count):
        return (subject.lower().strip(), kind.lower().strip(),
                noun_canonical.lower().strip(), int(count))

    def assert_quantity(self, subject, kind, count, noun_phrase,
                        noun_canonical, category="possession",
                        confidence=0.6, source="seed_regex") -> None:
        key = self._key(subject, kind, noun_canonical, count)
        # Conflicting prior count for the same (subject, kind, noun_canonical)?
        # Mark the old one superseded (the new disclosure is the active truth).
        # Run this BEFORE checking the existing key, so re-asserting a prior
        # count supersedes all other active counts for the same triplet.
        for (s, k, nc, c), r in list(self.records.items()):
            if (s == subject.lower().strip()
                    and k == kind.lower().strip()
                    and nc == noun_canonical.lower().strip()
                    and c != int(count) and not r.superseded):
                r.superseded = True
        existing = self.records.get(key)
        if existing is not None:
            existing.confidence = min(1.0, existing.confidence + 0.1)
            existing.turn_number = self.turn_num
            existing.superseded = False
            return
        self.records[key] = QuantityRecord(
            subject=subject, kind=kind, count=int(count),
            noun_phrase=noun_phrase, noun_canonical=noun_canonical,
            category=category, confidence=confidence,
            turn_number=self.turn_num, source=source)

    def correct(self, subject: str, noun: str, count: int,
                category: str = "possession") -> None:
        """Supersede any active record matching `noun` and store the new count.

        Online correction path: when the user corrects a count ("it's seven
        hives now"), the prior record is retired (not echoed) and the new count
        becomes active. Matches by canonical noun OR by the noun appearing in
        the stored phrase, so "hives" retires a record whose phrase is
        "hives of bees". A user correction is authoritative — no retrain.
        """
        subj = subject.lower().strip()
        _n = (noun or "").lower().strip()
        if not _n or count is None:
            return
        _matched = False
        for (s, k, nc, c), r in list(self.records.items()):
            if s != subj or r.superseded:
                continue
            if (r.noun_canonical == _n
                    or _n in self._noun_tokens(r.noun_phrase)):
                r.superseded = True
                _matched = True
        if _matched:
            # Re-assert the corrected count. Use the canonical noun from the
            # most-recent matching record to keep the key stable.
            _canon = _n
            for (s, k, nc, c), r in self.records.items():
                if (s == subj and r.superseded
                        and (r.noun_canonical == _n
                             or _n in self._noun_tokens(r.noun_phrase))):
                    _prev_kind = k
                    _prev_cat = r.category
                    _canon = nc
                    break
            else:
                _prev_kind = "keep"
                _prev_cat = category
            self.assert_quantity(subj, _prev_kind, count, _n, _canon,
                                 category=_prev_cat, confidence=0.85,
                                 source="correction")

    def _noun_tokens(self, phrase: str) -> Set[str]:
        return {w for w in re.findall(r"[a-z']+", (phrase or "").lower())
                if len(w) > 1}

    def query_count(self, noun_phrase: str, kind: Optional[str] = None,
                    subject: str = "i") -> Optional[QuantityRecord]:
        """Return the best active record whose noun overlaps the query phrase.

        Matches by token overlap (so 'racing pigeons' resolves a stored
        'pigeons' record, and 'cats' resolves a stored 'cats' record). When a
        kind is given it is preferred; otherwise the strongest overlap wins.
        Returns None when nothing plausibly matches — never fabricates.
        """
        subj = subject.lower().strip()
        q_tokens = self._noun_tokens(noun_phrase)
        if not q_tokens:
            return None
        best, best_score = None, 0.0
        for (s, k, nc, c), r in self.records.items():
            if s != subj or r.superseded:
                continue
            if kind is not None and k != kind.lower().strip():
                continue
            r_tokens = self._noun_tokens(r.noun_canonical) | self._noun_tokens(r.noun_phrase)
            if not r_tokens:
                continue
            overlap = len(q_tokens & r_tokens)
            if overlap == 0:
                continue
            # precision: fraction of the record's tokens that the query hits
            precision = overlap / len(r_tokens)
            score = overlap * (1.0 + precision) * (0.5 + r.confidence)
            if score > best_score:
                best, best_score = r, score
        return best

    def aggregate(self, category: str = "possession",
                  subject: str = "i") -> int:
        """Sum the counts of all active records of a category (for 'in total').

        A 'pets/animals in total' question sums the possession category across
        every species the user disclosed. Returns 0 when there are none (the
        caller decides how to phrase an honest empty answer).
        """
        subj = subject.lower().strip()
        total = 0
        for (s, k, nc, c), r in self.records.items():
            if s == subj and not r.superseded and r.category == category:
                total += r.count
        return total

    def aggregate_tokens(self, subject: str = "i") -> Dict[str, int]:
        """Per-category totals, for summary rendering."""
        subj = subject.lower().strip()
        out: Dict[str, int] = {}
        for (s, k, nc, c), r in self.records.items():
            if s == subj and not r.superseded:
                out[r.category] = out.get(r.category, 0) + r.count
        return out

    def get_state(self) -> Dict:
        return {
            'records': {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": (
                r.subject, r.kind, r.count, r.noun_phrase, r.noun_canonical,
                r.category, r.confidence, r.turn_number, r.source, r.superseded)
                for k, r in self.records.items()},
            'turn_num': self.turn_num,
        }

    def set_state(self, state: Dict) -> None:
        self.records = {}
        for k, v in state.get('records', {}).items():
            s, kind, cnt, np_, nc, cat, conf, tn, src, sup = v
            self.records[(s.lower(), kind.lower(), nc.lower(), int(cnt))] = \
                QuantityRecord(subject=s, kind=kind, count=int(cnt),
                               noun_phrase=np_, noun_canonical=nc, category=cat,
                               confidence=conf, turn_number=tn, source=src,
                               superseded=sup)
        self.turn_num = state.get('turn_num', 0)
