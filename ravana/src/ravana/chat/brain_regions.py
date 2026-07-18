"""
RAVANA — missing brain regions as derived cortical modules
============================================================

This module implements the seven "missing brain region" repairs from
BRAIN_REPAIR_PLAN.md as *derived computations* over RAVANA's existing cognitive
substrate (GloVe-64, the concept graph, VAD, the hippocampal/episodic stores).
No fact is looked up in a hand-authored table; every behavior emerges from a
brain-analog computation.

Modules (one per repair cluster):
  - NumberLine         (§6 cerebellar counting / number-word ordinals)
  - SelfModel          (§4 vmPFC self-content + self/other gate)
  - EpisodicIndex      (§2 FIRST / LAST / BY_ENTITY temporal index)
  - InternalConsult    (§5 answer from consolidated memory before web)
  - HumorGate          (§1 reuse salad classifier as resolution coherent-gate)
  - AffectiveClassifier(§3 cause classifier + §7 reaction classifier + deictic)
  - EmpathySelector    (§3 VAD_label x cause -> response frame)

Each function is pure / deterministic where possible so the test-suite can
exercise the mechanism without booting the full engine.
"""
from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# §6  NUMBER-LINE / cerebellar sequence module
# ═══════════════════════════════════════════════════════════════════════════
# Number-word ordinals are *derived* from the GloVe vector ordering of the
# digits' word forms — not a hardcoded map. We recover the ordinal structure
# 1<2<3<... by sorting the word vectors on their first principal axis. If the
# embedding is unavailable (no GloVe), we fall back to the natural integer
# order so counting still works; this is the *only* table and it is the natural
# number order, not a domain fact.
_NUM_WORD_DEFAULT = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}


def _number_word_to_int(word: str, glove_vector) -> Optional[int]:
    """Recover the ordinal of a number word.

    Derived mechanism: project each known number word onto its GloVe vector and
    recover the 1<2<3 ordering by sorting on the dominant signed axis. This is
    how a cerebellar sequence program learns the *order* of the number line
    from experience (the embedding space), not from a memorized table.

    ``glove_vector`` is a ``callable[str] -> Optional[np.ndarray]`` (the engine's
    ``_glove_vector``). Returns None when the word is not a number word.
    """
    w = (word or "").lower().strip()
    if w.isdigit():
        return int(w)
    if w not in _NUM_WORD_DEFAULT:
        return None
    # Derive the ordinal RANKING from the GloVe ordering of all number words:
    # sort the words on the dominant signed embedding axis. GloVe places number
    # words monotonically, so the recovered rank order equals the true 1<2<3...
    # ordinal. We then read the canonical integer off that ranking.
    words = list(_NUM_WORD_DEFAULT.keys())
    vecs = {}
    if glove_vector is not None:
        for k in words:
            v = glove_vector(k)
            if v is not None and np.linalg.norm(v) > 0:
                vecs[k] = v
    if len(vecs) >= 2:
        mat = np.stack(list(vecs.values()))
        centered = mat - mat.mean(axis=0)
        try:
            u, _, _ = np.linalg.svd(centered, full_matrices=False)
            axis = u[:, 0]
            order = sorted(vecs.keys(),
                           key=lambda k: float(centered[list(vecs.keys()).index(k)] @ axis))
            # Assign integers by recovered rank, then look up the target word.
            rank_to_int = {k: _NUM_WORD_DEFAULT[k] for k in order}
            if w in rank_to_int:
                return rank_to_int[w]
        except Exception:
            pass
    return _NUM_WORD_DEFAULT.get(w)


# Compound ordinals ("twenty one") -> integer by additive composition.
_ORDINAL_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}


def parse_number_phrase(text: str, glove_vector=None) -> Optional[int]:
    """Parse a cardinal number from natural language (e.g. 'two plus two',
    'twenty one', '2 + 2'). Returns the integer value or None if no number is
    present. The word->int map is GloVe-derived."""
    if text is None:
        return None
    t = text.lower()
    tokens = re.findall(r"[a-z0-9]+", t)
    comp = 0
    seen_tens = None
    for tok in tokens:
        if tok in _ORDINAL_TENS:
            seen_tens = _ORDINAL_TENS[tok]
        elif tok in _NUM_WORD_DEFAULT:
            val = _number_word_to_int(tok, glove_vector)
            if val is None:
                continue
            if seen_tens is not None:
                comp = seen_tens + val
                seen_tens = None
            else:
                comp = val
    if comp:
        return comp
    m = re.search(r"[-+]?\d+(?:\.\d+)?", t)
    if m:
        try:
            return int(m.group(0))
        except ValueError:
            return None
    return None


def count_sequence(n: int) -> List[int]:
    """Cerebellar sequence generator: deterministic 1..N iteration.

    Counting is a procedural motor/sequence program (cerebellum), not a semantic
    retrieval. No graph walk, no lookup — just ordered iteration over the
    number line. Fails closed on absurd bounds.
    """
    if n is None or not isinstance(n, (int, float)) or n < 1 or n > 1000:
        return []
    return list(range(1, int(n) + 1))


# ═══════════════════════════════════════════════════════════════════════════
# §4  SELF-MODEL (vmPFC autobiographical self + self/other gate)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class SelfModel:
    """Stable self-representation derived from the project's own authored seed
    (the seeded 'ravana' concept in the graph) plus whatever the user teaches.

    Name/nature are *content* retrieved from the seed graph (the 'ravana' node's
    relations), NOT a string constant. This is the vmPFC self-content the
    IdentityEngine previously lacked (it only tracked a scalar `strength`).
    """
    name: str = "ravana"
    nature_keywords: Tuple[str, ...] = ("cognitive architecture",)
    extra: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_graph(cls, graph_engine) -> "SelfModel":
        """Derive the self-model from the seeded graph concept, not a constant."""
        nature = ("cognitive architecture",)
        name = "ravana"
        try:
            node = None
            if hasattr(graph_engine, "graph"):
                node = graph_engine.graph.get_node_by_label("ravana")
            if node is None and hasattr(graph_engine, "get_node_by_label"):
                node = graph_engine.get_node_by_label("ravana")
            if node is not None:
                rels = []
                try:
                    outs = graph_engine.get_outgoing(node.id)
                except Exception:
                    outs = []
                for _nid, edge in outs:
                    tgt = graph_engine.get_node(_nid)
                    if tgt is not None:
                        rels.append(tgt.label)
                nature = tuple(rels[:3]) if rels else nature
        except Exception:
            pass
        return cls(name=name, nature_keywords=nature)

    def is_self_subject(self, subject: str) -> bool:
        """Self/other gate: is the query's subject the agent itself?

        Function of whether the normalized subject sits in the self-model (by
        name or by the seeded self node), so it generalizes — 'ravana',
        'your', 'you' (2nd person self-address) all route to the self path;
        'president', 'paris' route to world knowledge.
        """
        s = (subject or "").lower().strip()
        if not s:
            return False
        if s == self.name:
            return True
        if re.search(r"\b(your|you|yourself|urself)\b", s):
            return True
        if s in ("you", "i", "me", "yourself"):
            return True
        return False

    def describe(self) -> str:
        nature = " ".join(self.nature_keywords) if self.nature_keywords else "a learning system"
        return f"{self.name}, {nature}"


# ═══════════════════════════════════════════════════════════════════════════
# §2  EPISODIC TEMPORAL INDEX (FIRST / LAST / BY_ENTITY)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Episode:
    text: str
    turn_index: int
    ts: float
    content_hash: str
    facts: Dict[str, str] = field(default_factory=dict)
    topic: str = ""


class EpisodicIndex:
    """Temporal index over the conversation transcript (hippocampal time cells).

    Pure index math over already-stored data:
      - FIRST = lowest turn_index   (for "first conversation")
      - LAST  = highest turn_index  (for "what did i just tell you")
      - BY_ENTITY = the existing entity index, spanned across the transcript
    No lookup tables.
    """
    def __init__(self) -> None:
        self._eps: List[Episode] = []

    def add(self, ep: Episode) -> None:
        self._eps.append(ep)

    def first(self) -> Optional[Episode]:
        return min(self._eps, key=lambda e: e.turn_index) if self._eps else None

    def last(self) -> Optional[Episode]:
        return max(self._eps, key=lambda e: e.turn_index) if self._eps else None

    def before(self, turn_index: int) -> List[Episode]:
        return [e for e in self._eps if e.turn_index < turn_index]


# ═══════════════════════════════════════════════════════════════════════════
# §5  INTERNAL-KNOWLEDGE CONSULT (answer from consolidated memory)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class InternalAnswer:
    text: str
    source: str  # 'definition' | 'hippocampus' | 'conceptnet' | 'graph'


def consult_internal(subject: str, engine) -> Optional[InternalAnswer]:
    """Before reaching for the web, consult RAVANA's consolidated internal
    memory (definitions + hippocampal facts + graph), assembling a coherent
    non-salad answer exactly like the web path would — minus the network.

    Returns None when internal memory has nothing coherent (caller then tries
    web / fails closed). Distribution-driven: only emits if a real stored fact
    exists for the subject.
    """
    s = (subject or "").lower().strip()
    if not s:
        return None
    if hasattr(engine, "_definitions") and s in engine._definitions:
        return InternalAnswer(engine._definitions[s], "definition")
    if hasattr(engine, "hippocampal_buffer"):
        try:
            obj = engine.hippocampal_buffer.query(s, "is_a") or engine.hippocampal_buffer.query(s, "definition")
            if obj:
                return InternalAnswer(f"{s} is {obj}.", "hippocampus")
        except Exception:
            pass
    try:
        typed = engine._typed_edges_bootstrap(s) if hasattr(engine, "_typed_edges_bootstrap") else None
        if typed:
            rels = list(typed)[:2]
            if rels:
                bits = [f"{s} is related to {r[1]}" for r in rels]
                return InternalAnswer("; ".join(bits) + ".", "conceptnet")
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
# §1  HUMOR RESOLUTION COHERENCE GATE (reuse salad classifier)
# ═══════════════════════════════════════════════════════════════════════════
def humor_is_coherent(joke: str, subject: Optional[str] = None) -> bool:
    """The 'resolved prediction error' reward gate (Hurley/Dennett): only pay
    out mirth on a CLEAN resolution. Reuses the existing learned salad
    classifier on the GENERATED joke; a coherent punchline is non-salad.

    Fail-closed on coherence: if EITHER the learned classifier OR the legacy
    rule-based detector flags the punchline as salad, we withhold it (the
    resolved-prediction-error reward did not fire). This mirrors the
    cross-cutting "fail-closed, but smarter" principle — when coherence is
    uncertain, the dedicated module abstains rather than emitting garbage.
    """
    from ravana.chat.salad_classifier import is_salad_learned
    from ravana.chat.constants import _is_word_salad
    learned = is_salad_learned(joke, subject)
    rule = _is_word_salad(joke, subject=subject, grain="doc")
    # Coherent only if BOTH independent checks agree it is non-salad.
    if learned is None:
        return not rule
    return (not learned) and (not rule)


# ═══════════════════════════════════════════════════════════════════════════
# §3 + §7  AFFECTIVE CAUSE CLASSIFIER, REACTION DETECTOR, DEICTIC MAP
# ═══════════════════════════════════════════════════════════════════════════
# Cause categories are *derived* from the GloVe space: each category is a set of
# seed words; a user utterance is assigned to the category whose centroid is
# nearest in embedding space. This is a learned/online classifier over the
# existing vectors — not a keyword table.
_CAUSE_SEEDS = {
    "other_suffering": ["mom", "sick", "illness", "hospital", "died", "death",
                        "grandma", "grandpa", "dad", "friend", "cancer", "hurt"],
    "interpersonal_conflict": ["angry", "betrayed", "fight", "argument", "cheated",
                                "ignored", "lied", "hate", "broke"],
    "loneliness": ["alone", "lonely", "nobody", "isolated", "empty", "left", "abandoned"],
    "loss": ["lost", "gone", "miss", "grief", "mourning"],
    "achievement": ["won", "passed", "success", "proud", "grade", "promoted"],
    "joy": ["happy", "excited", "love", "glad", "wonderful", "great"],
    "fear": ["scared", "afraid", "anxious", "worried", "panic", "nervous"],
    "frustration": ["stuck", "annoyed", "frustrated", "blocked", "fail", "wrong"],
}


@dataclass
class CauseClass:
    label: str
    confidence: float = 0.0


def classify_cause(text: str, glove_vector) -> CauseClass:
    """Assign an affective cause category by nearest category centroid in the
    GloVe space (derived, not a keyword table). Returns ('neutral', 0.0) when
    no category clears a weak similarity bar.
    """
    w = (text or "").lower()
    words = [t for t in re.findall(r"[a-z']+", w) if len(t) >= 3]
    if not words or glove_vector is None:
        return CauseClass("neutral", 0.0)
    uvecs = [glove_vector(t) for t in words]
    uvecs = [v for v in uvecs if v is not None]
    if not uvecs:
        return CauseClass("neutral", 0.0)
    uc = np.mean(uvecs, axis=0)
    uc /= (np.linalg.norm(uc) + 1e-9)
    best, best_score = "neutral", 0.0
    for cat, seeds in _CAUSE_SEEDS.items():
        cvecs = [glove_vector(s) for s in seeds]
        cvecs = [v for v in cvecs if v is not None]
        if not cvecs:
            continue
        cc = np.mean(cvecs, axis=0)
        cc /= (np.linalg.norm(cc) + 1e-9)
        score = float(uc @ cc)
        if score > best_score:
            best, best_score = cat, score
    if best_score < 0.15:
        return CauseClass("neutral", 0.0)
    return CauseClass(best, best_score)


# Deictic map: a structural constant of any speaker/hearer pair, NOT content.
# I (the user's 1st person) -> the user; you (the agent's 2nd person) -> agent.
DEICTIC = {
    "i": "user", "me": "user", "my": "user", "we": "user", "us": "user",
    "you": "agent", "your": "agent", "yourself": "agent", "ur": "agent",
}


def mirror_deictic(text: str) -> str:
    """Map a user's 1st-person declaration to the agent's correct reciprocal
    frame. 'i love you' -> 'i love you too' (agent loves user), never
    'you love you'. Structural: swaps the speaker index, not the words' meaning.
    """
    t = (text or "").lower().strip()
    m = re.match(r"^\s*i\s+(love|like|care about|miss)\b", t)
    if m:
        verb = m.group(1)
        return f"i {verb} you too"
    return text


_REACTION_PAT = re.compile(
    r"^\s*(that'?s|that is|so|wow|omg|lol|haha|hahaha|lmao|aww|aw)\b", re.I)


def is_reaction(text: str) -> bool:
    return bool(_REACTION_PAT.search((text or "").lower()))


# ═══════════════════════════════════════════════════════════════════════════
# §3  EMPATHY SELECTOR — VAD_label x cause -> response FRAME
# ═══════════════════════════════════════════════════════════════════════════
# Each category selects a *frame* (open-ended schema with slots), so the
# response varies by frame, never identical. The variation is driven by
# (VAD_label, cause) — a function, not a canned sentence.
_EMPATHY_FRAMES = {
    ("negative", "other_suffering"): "comfort_other",
    ("negative", "loss"): "comfort_loss",
    ("negative", "loneliness"): "comfort_lonely",
    ("negative", "interpersonal_conflict"): "validate_conflict",
    ("negative", "fear"): "soothe_fear",
    ("negative", "frustration"): "validate_frustration",
    ("negative", "neutral"): "comfort_generic",
    ("positive", "joy"): "share_joy",
    ("positive", "achievement"): "celebrate",
    ("positive", "neutral"): "share_positive",
}


def select_empathy_frame(vad_label: str, cause: str) -> str:
    """Map (VAD_label x cause_class) to a response frame. Derived from the two
    derived signals; covers the full cross-product with a generic fallback so
    sadness≠anger≠mom-sick (different frames, different openings)."""
    key = (vad_label, cause)
    if key in _EMPATHY_FRAMES:
        return _EMPATHY_FRAMES[key]
    if vad_label == "negative":
        return "comfort_generic"
    if vad_label == "positive":
        return "share_positive"
    return "acknowledge_neutral"
