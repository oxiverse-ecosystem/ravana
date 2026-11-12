#!/usr/bin/env python3
"""Measure how ungrounded RAVANA's neural decoder output is.

THIS IS A MEASUREMENT INSTRUMENT, NOT A FIX. It does not modify any
production code and does not propose one. It exists so that a future change
to the decoder / grounding gate can be judged against a NUMBER rather than a
vibe.

What it does
------------
For each subject, it builds a CognitiveResponseContext the same way
``CognitiveChatEngine.process_turn`` does (activate -> spread -> situation
model -> ctx) and calls the decoder entry point
``engine._generate_with_situation_model(ctx)`` twice:

  * GATED   -- the production configuration. The Levelt/Wernicke monitor
               (``_sm_response_grounded``) and the word-salad checks are
               live, so text that fails them is withheld and the method
               falls through to the narrative / syntax paths.
  * UNGATED -- ``engine._disable_grounding_gate = True``. This is the
               decoder's own output, i.e. what the network would say if the
               monitor were not there. This is the quantity the task cares
               about: the decoder's intrinsic grounding, not the gate's.

Each output is scored on three axes:

  a. FLUENCY (proxy) -- word count, unique-ratio, and whether the
     rule-based ``_is_word_salad`` fires. This is a STRUCTURAL proxy only;
     it is not a learned fluency model and it is blind to semantic
     falsehood (see LIMITATIONS).
  b. ON-TOPIC -- overlap between the output's content words and the
     subject's graph neighbours / stored definition, plus the mean cosine
     of the output's content words to the subject's graph vector.
  c. UNSUPPORTED ASSERTIONS -- relation words (is / are / opposite / tied /
     connected / differs / leads / paves / like ...) attached to content
     words that are NOT graph neighbours of the subject. Each such
     (relation, content-word) pair is counted as one unsupported assertion.

The script is RESUMABLE: with ``--out`` it reads back any existing TSV and
skips subjects already recorded, so a gateway restart does not lose a long
run. It always exits 0; per-subject failures are recorded in the table.

Usage
-----
    uv run python scripts/eval_decoder_groundedness.py
    uv run python scripts/eval_decoder_groundedness.py --out tmp/groundedness.tsv
    uv run python scripts/eval_decoder_groundedness.py --engine persisted --out tmp/g.tsv
    uv run python scripts/eval_decoder_groundedness.py --engine both --out tmp/g.tsv

READ-ONLY: the script never writes to the repo's weights/ or data/. The
persisted-state engine is constructed with ``data_dir=<repo>/weights`` and
``user_suffix=''``; ``CognitiveChatEngine.__init__`` only *reads* the
snapshot there (it writes on ``save()``, which this script never calls).

LIMITATIONS -- read this before quoting a number from this script
---------------------------------------------------------------
1. Axis (c) does NOT detect falsehood. It detects UNGROUNDED claims: a
   relation word bound to a content word that is not a graph neighbour of
   the subject. A claim that is TRUE but unanchored in the graph is counted
   as unsupported (false positive). A claim that is anchored but FALSE is
   not counted at all (false negative). The graph is itself partly built
   from noisy GloVe wiring, so "anchored" is not a truth certificate.
2. Axis (a) is a STRUCTURAL fluency proxy. ``_is_word_salad`` is a
   rule-based function (repetition, TTR, structural-word frequency); it is
   NOT a learned fluency model and it is blind to semantic falsehood. It
   fired on 0/N outputs in the runs below -- fluent-but-false text is
   exactly the class it cannot see. A high unique-ratio is not evidence of
   quality.
3. Axis (b) measures lexical overlap with the subject's graph neighbours
   and stored definition. A correct definition of an abstract concept
   (rumour -> "pieces of purportedly true information") shares almost no
   words with the subject, so on-topic fraction UNDERSTATES correctness for
   definitional output. The mean-cosine column is the more robust signal.
4. The UNGATED number is the decoder's intrinsic grounding only when the
   decoder path is actually reached. ``_generate_with_situation_model``
   tries the decoder first, then falls back to narrative / syntax paths.
   The ``strategy`` column says which produced the text; the narrative and
   syntax paths are NOT the neural decoder, so their rows measure the
   fallbacks, not the decoder. Compare strategies before comparing means.
5. The gate is part of the system under test. GATED numbers measure the
   whole production path (decoder + monitor); UNGATED numbers measure the
   decoder alone. Neither is "the truth"; the gap between them is the
   monitor's contribution.
6. Subject coverage is engine-dependent. The baby engine and the persisted
   engine have different graphs (259 vs 295 nodes) and different
   definitions (229 vs 80), so their means are NOT directly comparable --
   the comparison block prints both so the difference is visible rather
   than hidden.
7. N=1 per (subject, phrasing). The decoder samples with a temperature, so
   a single sample is not a stable estimate of the mean. Run several seeds
   or repeat the script before treating a small difference as real.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── repo root (scripts/ -> repo root) ─────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
from ravana.chat.models import CognitiveResponseContext  # noqa: E402
from ravana.chat.constants import (  # noqa: E402
    STOP_WORDS,
    WEB_GARBAGE,
    _is_word_salad,
    _is_word_salad_any_sentence,
)

# ── subjects ─────────────────────────────────────────────────────────────────
# Graph-known seeds are DISCOVERED from the graph (see discover_subjects);
# this list only nominates candidates and supplies the unknown controls.
NOMINATED_SUBJECTS: Tuple[str, ...] = (
    "gravity", "ocean", "water", "light", "time", "trust", "memory",
    "dream", "love", "fear", "music", "colour", "language",
)
UNKNOWN_CONTROLS: Tuple[str, ...] = (
    "zzzunknown", "qxzblorp", "notarealword",
)

# ── relation / assertion vocabulary ───────────────────────────────────────────
# The relation words the task named, plus the close variants the decoder
# actually emits (measured: "is tied to", "paves the way for", "is opposite
# to", "runs counter to", "is connected to", "is similar to", "stems from",
# "brings about", "turns into", "leads to", "springs from", "draws on",
# "carries through", "opens up", "guides", "recalls", "grounds in").
#
# NOTE: pure prepositions (of / to / with / into / through / toward) are
# deliberately EXCLUDED. A preposition next to a content word is not a
# predicate -- "grows of awareness" is not an assertion about awareness in
# the way "is tied to awareness" is. Including them inflated the unsupported
# count with syntactic glue (measured: "of:awareness", "to:pursue",
# "through:learning" were ~40% of all flags on the baby engine).
RELATION_WORDS = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "am",
    "opposite", "tied", "tie", "ties", "connected", "connect", "connects",
    "differs", "differ", "leads", "lead", "paves", "like", "similar",
    "resembles", "akin", "related", "relates", "linked", "link", "links",
    "means", "mean", "implies", "suggests", "causes", "cause", "caused",
    "creates", "created", "makes", "make", "made", "forms", "formed",
    "contains", "contain", "includes", "include", "produces", "produce",
    "generates", "generate", "drives", "drive", "driven", "shapes", "shape",
    "influences", "influence", "affects", "affect", "determines", "determine",
    "defines", "define", "describes", "describe", "represents", "represent",
    "symbolizes", "symbolize", "mirrors", "mirror", "reflects", "reflect",
    "contrasts", "contrast", "contradicts", "contradict", "opposes", "oppose",
    "opposed", "reverses", "reverse", "aligns", "align", "parallels", "parallel",
    "echoes", "echo", "stems", "stem", "springs", "spring", "arises", "arise",
    "derives", "derive", "originates", "originate", "comes", "come",
    "brings", "bring", "brought", "gives", "give", "given", "takes", "take",
    "taken", "turns", "turn", "turned", "becomes", "become", "became",
    "grows", "grow", "grew", "flows", "flow", "flowed", "runs", "run", "ran",
    "reaches", "reach", "reached", "extends", "extend", "extended",
    "opens", "open", "opened", "closes", "close", "closed",
    "guides", "guide", "guided", "directs", "direct", "directed",
    "recalls", "recall", "recalled", "reminds", "remind", "reminded",
    "grounds", "ground", "grounded",
    "draws", "draw", "drawn", "carries", "carry", "carried",
    "holds", "hold", "held", "keeps", "keep", "kept",
    "needs", "need", "needed", "requires", "require", "required",
    "uses", "use", "used", "using", "serves", "serve", "served",
    "works", "work", "worked", "plays", "play", "played",
    "lives", "live", "lived", "dies", "die", "died",
    "appears", "appear", "appeared", "seems", "seem", "seemed",
    "belongs", "belong", "belonged", "exists", "exist", "existed",
    "happens", "happen", "happened", "occurs", "occur", "occurred",
    "starts", "start", "started", "begins", "begin", "began",
    "ends", "end", "ended", "stops", "stop", "stopped",
    "changes", "change", "changed", "moves", "move", "moved",
    "supports", "support", "supported", "helps", "help", "helped",
    "protects", "protect", "protected", "threatens", "threaten", "threatened",
    "increases", "increase", "increased", "decreases", "decrease", "decreased",
    "reduces", "reduce", "reduced", "adds", "add", "added",
    "removes", "remove", "removed", "replaces", "replace", "replaced",
    "binds", "bind", "bound",
    "associates", "associate", "associated",
    "compared", "compare", "compares", "versus", "vs",
    "counter", "against", "unlike",
})

# Prepositions / particles that never count as a predicate on their own.
# Kept explicit so the exclusion is auditable rather than implicit.
NON_PREDICATE_WORDS = frozenset({
    "of", "to", "with", "into", "through", "toward", "towards", "from", "for",
    "on", "in", "at", "by", "as", "about", "over", "under", "between",
    "during", "after", "before", "against", "across", "upon", "within",
    "without", "around", "behind", "beyond", "despite", "except", "inside",
    "near", "off", "onto", "since", "until", "via", "with", "per", "than",
})

# Words that carry no referential content and are never counted as the
# "content word" of an assertion.
_NON_CONTENT = frozenset({
    "it", "its", "they", "them", "their", "he", "she", "him", "her", "his",
    "we", "our", "you", "your", "i", "me", "my", "myself", "this", "that",
    "these", "those", "there", "here", "what", "which", "who", "whom", "whose",
    "one", "ones", "something", "anything", "everything", "nothing",
    "other", "others", "another", "such", "same", "the", "a", "an",
})


# ═════════════════════════════════════════════════════════════════════════════
# subject discovery
# ═════════════════════════════════════════════════════════════════════════════

def discover_subjects(eng: CognitiveChatEngine) -> List[str]:
    """Discover subjects from the graph rather than hardcoding them.

    Preference order:
      1. nominated subjects that the graph actually knows (graph node, or a
         stored definition / web source);
      2. graph labels with real structure (degree >= 2) -- these are the
         concepts RAVANA has actually wired up, so they are the ones a
         grounding claim can be checked against;
      3. the unknown controls, always appended (they are the negative
         control: the decoder must not be able to say anything grounded
         about a subject with no graph presence).
    """
    known: List[str] = []
    for s in NOMINATED_SUBJECTS:
        if _subject_known(eng, s):
            known.append(s)

    # Graph-discovered: labels with at least one edge, single-token, and
    # not junk-shaped. Sorted for determinism.
    discovered = []
    for nid, node in sorted(eng.graph.nodes.items()):
        label = (getattr(node, "label", "") or "").strip().lower()
        if not label or " " in label:
            continue
        if len(label) < 3 or label in STOP_WORDS or label in WEB_GARBAGE:
            continue
        if label in known or label in UNKNOWN_CONTROLS:
            continue
        deg = len(eng.graph.get_outgoing(nid)) + len(eng.graph.get_incoming(nid))
        if deg >= 2:
            discovered.append(label)
    discovered.sort()

    subjects = known + discovered + list(UNKNOWN_CONTROLS)
    # de-duplicate, preserve order
    seen = set()
    out = []
    for s in subjects:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _subject_known(eng: CognitiveChatEngine, s: str) -> bool:
    return (
        s in getattr(eng, "_concept_keywords", {})
        or s in getattr(eng, "_concept_labels", set())
        or s in getattr(eng, "_definitions", {})
        or s in getattr(eng, "_concept_sources", {})
    )


# ═════════════════════════════════════════════════════════════════════════════
# context construction (mirrors process_turn's ctx assembly)
# ═════════════════════════════════════════════════════════════════════════════

def build_ctx(eng: CognitiveChatEngine, subject: str, raw_input: str) -> CognitiveResponseContext:
    """Build a CognitiveResponseContext the way process_turn does.

    Mirrors engine.py:10165-10700 + 11167-11192: activate from input,
    spread from the subject's node ids, build the sentence/context vectors,
    push both into the situation model, then assemble the ctx. The situation
    model is RESET first so each subject is measured independently rather
    than inheriting the previous subject's DMN state.
    """
    eng.situation_model.reset()

    activated = eng._activate_from_input(raw_input)
    subject_ids = set(eng._concept_keywords.get(subject.lower(), []))
    associations = eng._spread_and_collect(activated, primary_ids=subject_ids)

    sentence_vector = eng._build_sentence_vector(raw_input)
    context_vector = eng._build_context_vector_from_input(raw_input, subject)

    concept_embs: Dict[str, np.ndarray] = {}
    activations: Dict[str, float] = {}
    for label, score in associations[:12]:
        ll = label.lower()
        nids = eng._concept_keywords.get(ll, [])
        if nids:
            node = eng.graph.get_node(nids[0])
            if node is not None and node.vector is not None:
                concept_embs[ll] = node.vector
                activations[ll] = max(activations.get(ll, 0.0), float(score))
    for nid in activated[:10]:
        node = eng.graph.get_node(nid)
        if node and node.label and node.vector is not None:
            ll = node.label.lower()
            if ll not in concept_embs:
                concept_embs[ll] = node.vector
                activations[ll] = 0.5

    eng.situation_model.update(
        concept_embeddings=concept_embs,
        activations=activations,
        graph_get_vector_fn=eng._glove_vector,
        sentence_vector=sentence_vector,
        context_vector_input=context_vector,
    )

    return CognitiveResponseContext(
        subject=subject,
        raw_input=raw_input,
        associated_concepts=associations,
        bridge_concept=eng._find_bridge(associations, subject),
        situation_vector=eng.situation_model.get_blended_vector(),
        situation_narrative=eng.situation_model.get_narrative_suggestions(),
        sentence_vector=sentence_vector,
        context_vector=context_vector,
    )


# ═════════════════════════════════════════════════════════════════════════════
# scoring
# ═════════════════════════════════════════════════════════════════════════════

def _content_words(text: str) -> List[str]:
    """Content words: len >= 3, not a stop word, not web garbage, not a
    pronoun/determiner. Same filter the grounding gate uses."""
    out = []
    for w in re.findall(r"[a-z']+", text.lower()):
        w = w.strip("'")
        if len(w) < 3 or w in STOP_WORDS or w in WEB_GARBAGE or w in _NON_CONTENT:
            continue
        out.append(w)
    return out


def _unique_ratio(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _subject_neighbours(eng: CognitiveChatEngine, subject: str) -> set:
    """The subject's graph neighbours (both directions), as lowercased labels.

    This is the system's own record that a concept is ABOUT the subject --
    the same evidence the grounding gate trusts in step 1.
    """
    nids = eng._concept_keywords.get(subject.lower(), [])
    if not nids:
        return set()
    neigh = set()
    for nid in nids:
        for tgt, _edge in (eng.graph.get_outgoing(nid) or []):
            node = eng.graph.get_node(tgt)
            if node is not None and node.label:
                neigh.add(node.label.lower().strip())
        for tgt, _edge in (eng.graph.get_incoming(nid) or []):
            node = eng.graph.get_node(tgt)
            if node is not None and node.label:
                neigh.add(node.label.lower().strip())
    neigh.discard(subject.lower())
    return neigh


def _definition_words(eng: CognitiveChatEngine, subject: str) -> set:
    defn = getattr(eng, "_definitions", {}).get(subject.lower())
    if not defn:
        return set()
    return {w for w in re.findall(r"[a-z']+", str(defn).lower())
            if len(w) >= 3 and w not in STOP_WORDS and w not in WEB_GARBAGE}


def _subject_vector(eng: CognitiveChatEngine, subject: str) -> Optional[np.ndarray]:
    nids = eng._concept_keywords.get(subject.lower(), [])
    if not nids:
        return None
    node = eng.graph.get_node(nids[0])
    if node is None or node.vector is None:
        return None
    v = np.asarray(node.vector, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_on_topic(eng: CognitiveChatEngine, subject: str, text: str) -> Dict[str, Any]:
    """Axis (b): overlap with graph neighbours / definition + mean cosine."""
    cw = _content_words(text)
    neigh = _subject_neighbours(eng, subject)
    defw = _definition_words(eng, subject)

    # A content word counts as "on-topic" if it IS a neighbour, or is a
    # constituent of a multi-word neighbour, or appears in the definition.
    overlap = []
    for w in cw:
        hit = False
        if w in neigh or w in defw:
            hit = True
        else:
            for nb in neigh:
                if " " in nb and w in nb.split():
                    hit = True
                    break
            if not hit:
                for dw in defw:
                    if " " in dw and w in dw.split():
                        hit = True
                        break
        if hit:
            overlap.append(w)

    subj_vec = _subject_vector(eng, subject)
    sims: List[float] = []
    if subj_vec is not None:
        for w in cw:
            v = eng._glove_vector(w)
            if v is None:
                # Fall back to the decoder's own embedding table.
                emb = getattr(eng, "_decoder_word_to_embed", {}).get(w)
                if emb is not None:
                    v = np.asarray(emb, dtype=np.float64).reshape(-1)[:len(subj_vec)]
            if v is None or len(v) != len(subj_vec):
                continue
            sims.append(_cosine(np.asarray(v, dtype=np.float64), subj_vec))

    return {
        "content_words": len(cw),
        "overlap_count": len(overlap),
        "overlap_words": overlap,
        "on_topic_frac": (len(overlap) / len(cw)) if cw else 0.0,
        "mean_cosine": float(np.mean(sims)) if sims else 0.0,
        "n_cosine": len(sims),
        "neighbour_count": len(neigh),
        "has_definition": bool(defw),
    }


def score_unsupported_assertions(
    eng: CognitiveChatEngine, subject: str, text: str
) -> Dict[str, Any]:
    """Axis (c): relation words attached to content words that are NOT
    neighbours of the subject in the graph.

    An "assertion" is a (relation word, content word) pair. The relation word
    is the predicate; the content word is its argument. The assertion is
    UNSUPPORTED when the argument is not something the graph relates to the
    subject (and is not in the subject's stored definition).

    This is a STRUCTURAL proxy for "asserts something false". It cannot know
    truth -- only whether the claim is anchored in the system's own graph.
    A claim that is true but unanchored is counted as unsupported (a false
    positive); a claim that is anchored but false is NOT counted (a false
    negative). See LIMITATIONS in the module docstring / final report.
    """
    neigh = _subject_neighbours(eng, subject)
    defw = _definition_words(eng, subject)
    subj_l = subject.lower()
    subj_tokens = set(subj_l.split())

    tokens = re.findall(r"[a-z']+", text.lower())
    unsupported: List[Tuple[str, str]] = []
    supported: List[Tuple[str, str]] = []

    for i, tok in enumerate(tokens):
        if tok not in RELATION_WORDS:
            continue
        # Find the nearest content word after the relation word (the
        # argument), then before it (the subject side) if none follows.
        # A preposition between the predicate and its argument is skipped
        # ("is tied to star" -> relation "tied", argument "star").
        arg = None
        for j in range(i + 1, min(i + 5, len(tokens))):
            w = tokens[j]
            if w in NON_PREDICATE_WORDS:
                continue
            if w in STOP_WORDS or w in _NON_CONTENT or w in RELATION_WORDS:
                continue
            if len(w) < 3 or w in WEB_GARBAGE:
                continue
            arg = w
            break
        if arg is None:
            for j in range(i - 1, max(i - 5, -1), -1):
                w = tokens[j]
                if w in NON_PREDICATE_WORDS:
                    continue
                if w in STOP_WORDS or w in _NON_CONTENT or w in RELATION_WORDS:
                    continue
                if len(w) < 3 or w in WEB_GARBAGE:
                    continue
                arg = w
                break
        if arg is None:
            continue
        # The subject itself / its own tokens are not "assertions about
        # something else" -- skip self-referential pairs.
        if arg in subj_tokens:
            continue
        anchored = (
            arg in neigh
            or arg in defw
            or any(" " in nb and arg in nb.split() for nb in neigh)
        )
        if anchored:
            supported.append((tok, arg))
        else:
            unsupported.append((tok, arg))

    return {
        "unsupported_count": len(unsupported),
        "unsupported": unsupported,
        "supported_count": len(supported),
        "supported": supported,
    }


def score_fluency(text: str, subject: str) -> Dict[str, Any]:
    """Axis (a): structural fluency proxy."""
    words = re.findall(r"\b\w+\b", text)
    return {
        "word_count": len(words),
        "unique_ratio": _unique_ratio(text),
        "is_word_salad": bool(_is_word_salad(text, subject=subject)),
        "is_word_salad_any_sentence": bool(
            _is_word_salad_any_sentence(text, subject=subject)
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# per-subject measurement
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SubjectResult:
    engine: str
    subject: str
    raw_input: str
    # graph context
    in_graph: bool = False
    has_definition: bool = False
    has_source: bool = False
    neighbour_count: int = 0
    degree: int = 0
    # gated run
    gated_text: str = ""
    gated_strategy: str = ""
    gated_ok: bool = False
    # ungated run
    ungated_text: str = ""
    ungated_strategy: str = ""
    ungated_ok: bool = False
    # scores (ungated = the decoder's own output; gated = production output)
    fluency: Dict[str, Any] = field(default_factory=dict)
    on_topic: Dict[str, Any] = field(default_factory=dict)
    unsupported: Dict[str, Any] = field(default_factory=dict)
    gated_fluency: Dict[str, Any] = field(default_factory=dict)
    gated_on_topic: Dict[str, Any] = field(default_factory=dict)
    gated_unsupported: Dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    error: str = ""

    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        if not self.ungated_text and not self.gated_text:
            return "NO_OUTPUT"
        return "OK"


def measure_subject(eng: CognitiveChatEngine, engine_label: str,
                    subject: str, raw_input: str,
                    use_cpg: bool = False,
                    cpg_config=None) -> SubjectResult:
    t0 = time.time()
    res = SubjectResult(engine=engine_label, subject=subject, raw_input=raw_input)
    try:
        res.in_graph = _subject_known(eng, subject)
        res.has_definition = subject.lower() in getattr(eng, "_definitions", {})
        res.has_source = subject.lower() in getattr(eng, "_concept_sources", {})
        nids = eng._concept_keywords.get(subject.lower(), [])
        if nids:
            nid = nids[0]
            res.degree = len(eng.graph.get_outgoing(nid)) + len(eng.graph.get_incoming(nid))
        res.neighbour_count = len(_subject_neighbours(eng, subject))

        ctx = build_ctx(eng, subject, raw_input)

        # ── UNGATED: the decoder's own output ────────────────────────────
        prev_gate = getattr(eng, "_disable_grounding_gate", False)
        prev_cpg = getattr(eng, "_cpg_enabled", False)
        prev_cfg = getattr(eng, "_cpg_config", None)
        try:
            eng._disable_grounding_gate = True
            if use_cpg:
                eng._cpg_enabled = True
                if cpg_config is not None:
                    eng._cpg_config = cpg_config
            out = eng._generate_with_situation_model(ctx)
        finally:
            eng._disable_grounding_gate = prev_gate
            eng._cpg_enabled = prev_cpg
            eng._cpg_config = prev_cfg
        if out:
            res.ungated_text, res.ungated_strategy = out[0], out[1]
            res.ungated_ok = True
            res.fluency = score_fluency(res.ungated_text, subject)
            res.on_topic = score_on_topic(eng, subject, res.ungated_text)
            res.unsupported = score_unsupported_assertions(eng, subject, res.ungated_text)

        # ── GATED: production configuration ──────────────────────────────
        out2 = eng._generate_with_situation_model(ctx)
        if out2:
            res.gated_text, res.gated_strategy = out2[0], out2[1]
            res.gated_ok = True
            res.gated_fluency = score_fluency(res.gated_text, subject)
            res.gated_on_topic = score_on_topic(eng, subject, res.gated_text)
            res.gated_unsupported = score_unsupported_assertions(eng, subject, res.gated_text)

    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
        if os.environ.get("EG_DEBUG"):
            traceback.print_exc()
    finally:
        res.elapsed_s = time.time() - t0
    return res


# ═════════════════════════════════════════════════════════════════════════════
# TSV I/O (resumable)
# ═════════════════════════════════════════════════════════════════════════════

TSV_COLUMNS: Tuple[str, ...] = (
    "engine", "subject", "raw_input", "status",
    "in_graph", "has_definition", "has_source", "neighbour_count", "degree",
    "ungated_strategy", "ungated_text",
    "ungated_word_count", "ungated_unique_ratio", "ungated_word_salad",
    "ungated_content_words", "ungated_overlap_count", "ungated_on_topic_frac",
    "ungated_mean_cosine", "ungated_unsupported_count", "ungated_supported_count",
    "ungated_unsupported_pairs",
    "gated_strategy", "gated_text",
    "gated_word_count", "gated_unique_ratio", "gated_word_salad",
    "gated_content_words", "gated_overlap_count", "gated_on_topic_frac",
    "gated_mean_cosine", "gated_unsupported_count",
    "elapsed_s", "error",
)


def result_to_row(r: SubjectResult) -> Dict[str, str]:
    def b(x):
        return "" if x is None else ("1" if x else "0")

    return {
        "engine": r.engine,
        "subject": r.subject,
        "raw_input": r.raw_input,
        "status": r.status,
        "in_graph": b(r.in_graph),
        "has_definition": b(r.has_definition),
        "has_source": b(r.has_source),
        "neighbour_count": str(r.neighbour_count),
        "degree": str(r.degree),
        "ungated_strategy": r.ungated_strategy,
        "ungated_text": r.ungated_text,
        "ungated_word_count": str(r.fluency.get("word_count", "")),
        "ungated_unique_ratio": _fmt(r.fluency.get("unique_ratio")),
        "ungated_word_salad": b(r.fluency.get("is_word_salad")),
        "ungated_content_words": str(r.on_topic.get("content_words", "")),
        "ungated_overlap_count": str(r.on_topic.get("overlap_count", "")),
        "ungated_on_topic_frac": _fmt(r.on_topic.get("on_topic_frac")),
        "ungated_mean_cosine": _fmt(r.on_topic.get("mean_cosine")),
        "ungated_unsupported_count": str(r.unsupported.get("unsupported_count", "")),
        "ungated_supported_count": str(r.unsupported.get("supported_count", "")),
        "ungated_unsupported_pairs": ";".join(
            f"{a}:{b}" for a, b in r.unsupported.get("unsupported", [])
        ),
        "gated_strategy": r.gated_strategy,
        "gated_text": r.gated_text,
        "gated_word_count": str(r.gated_fluency.get("word_count", "")),
        "gated_unique_ratio": _fmt(r.gated_fluency.get("unique_ratio")),
        "gated_word_salad": b(r.gated_fluency.get("is_word_salad")),
        "gated_content_words": str(r.gated_on_topic.get("content_words", "")),
        "gated_overlap_count": str(r.gated_on_topic.get("overlap_count", "")),
        "gated_on_topic_frac": _fmt(r.gated_on_topic.get("on_topic_frac")),
        "gated_mean_cosine": _fmt(r.gated_on_topic.get("mean_cosine")),
        "gated_unsupported_count": str(r.gated_unsupported.get("unsupported_count", "")),
        "elapsed_s": _fmt(r.elapsed_s),
        "error": r.error,
    }


def _fmt(v) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def read_existing(path: str) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    """Read back a previous TSV. Key = (engine, subject, raw_input)."""
    done: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    if not path or not os.path.exists(path):
        return done
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                key = (row.get("engine", ""), row.get("subject", ""),
                       row.get("raw_input", ""))
                # A row that errored is NOT considered complete -- re-run it.
                if row.get("error"):
                    continue
                done[(row.get("engine", ""), row.get("subject", "") or "",
                      row.get("raw_input", "") or "")] = row
    except Exception as e:
        print(f"  [resume] could not read {path}: {e}")
    return done


def append_rows(path: str, rows: Sequence[Dict[str, str]]) -> None:
    if not path or not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TSV_COLUMNS, delimiter="\t",
                                extrasaction="ignore", lineterminator="\n")
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ═════════════════════════════════════════════════════════════════════════════
# reporting
# ═════════════════════════════════════════════════════════════════════════════

def _mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else 0.0


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return None
    xa = np.array([p[0] for p in pairs], dtype=np.float64)
    ya = np.array([p[1] for p in pairs], dtype=np.float64)
    if xa.std() <= 1e-12 or ya.std() <= 1e-12:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def print_table(results: Sequence[SubjectResult]) -> None:
    # ── ungated table (the decoder's own output) ─────────────────────────
    header = (
        f"{'subject':<16} {'graph':<5} {'deg':>3} {'nbr':>3} "
        f"{'words':>5} {'uniq':>5} {'salad':>5} "
        f"{'over':>4} {'topic%':>6} {'cos':>6} "
        f"{'unsup':>5} {'sup':>4}  {'strategy':<26} text"
    )
    print()
    print("=" * len(header))
    print("UNGATED DECODER OUTPUT  (engine._disable_grounding_gate = True)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        if r.error:
            print(f"{r.subject:<16} {'ERR':<5} ---  {r.error[:80]}")
            continue
        if not r.ungated_text:
            print(f"{r.subject:<16} {'y' if r.in_graph else 'n':<5} "
                  f"{r.degree:>3} {r.neighbour_count:>3} "
                  f"{'-- no decoder output --':<40} ({r.ungated_strategy or '-'})")
            continue
        print(
            f"{r.subject:<16} {'y' if r.in_graph else 'n':<5} "
            f"{r.degree:>3} {r.neighbour_count:>3} "
            f"{r.fluency.get('word_count', 0):>5} "
            f"{r.fluency.get('unique_ratio', 0):>5.2f} "
            f"{'YES' if r.fluency.get('is_word_salad') else 'no':>5} "
            f"{r.on_topic.get('overlap_count', 0):>4} "
            f"{r.on_topic.get('on_topic_frac', 0):>6.2f} "
            f"{r.on_topic.get('mean_cosine', 0):>6.3f} "
            f"{r.unsupported.get('unsupported_count', 0):>5} "
            f"{r.unsupported.get('supported_count', 0):>4}  "
            f"{r.ungated_strategy:<26} {r.ungated_text[:70]}"
        )

    # ── gated table (production configuration) ───────────────────────────
    print()
    print("=" * len(header))
    print("GATED OUTPUT  (production configuration: _sm_response_grounded live)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        if r.error:
            continue
        if not r.gated_text:
            print(f"{r.subject:<16} {'y' if r.in_graph else 'n':<5} "
                  f"{r.degree:>3} {r.neighbour_count:>3} "
                  f"{'-- withheld by gate / no output --':<40}")
            continue
        print(
            f"{r.subject:<16} {'y' if r.in_graph else 'n':<5} "
            f"{r.degree:>3} {r.neighbour_count:>3} "
            f"{r.gated_fluency.get('word_count', 0):>5} "
            f"{r.gated_fluency.get('unique_ratio', 0):>5.2f} "
            f"{'YES' if r.gated_fluency.get('is_word_salad') else 'no':>5} "
            f"{r.gated_on_topic.get('overlap_count', 0):>4} "
            f"{r.gated_on_topic.get('on_topic_frac', 0):>6.2f} "
            f"{r.gated_on_topic.get('mean_cosine', 0):>6.3f} "
            f"{r.gated_unsupported.get('unsupported_count', 0):>5} "
            f"{'':>4}  "
            f"{r.gated_strategy:<26} {r.gated_text[:70]}"
        )


def print_summary(results: Sequence[SubjectResult], engine_label: str) -> None:
    ok = [r for r in results if not r.error]
    ungated = [r for r in ok if r.ungated_text]
    gated = [r for r in ok if r.gated_text]
    errors = [r for r in results if r.error]
    no_out = [r for r in ok if not r.ungated_text]

    print()
    print("=" * 78)
    print(f"SUMMARY  engine={engine_label!r}  subjects={len(results)}  "
          f"ok={len(ok)}  error={len(errors)}  no_ungated_output={len(no_out)}")
    print("=" * 78)

    if not ungated:
        print("  no ungated decoder output to score.")
        return

    # Restrict the headline numbers to subjects the graph actually knows --
    # the unknown controls have no neighbours by construction, so their
    # on-topic score is 0 by definition and would drag the mean down for a
    # reason that is not the decoder's fault.
    known = [r for r in ungated if r.in_graph]
    unknown = [r for r in ungated if not r.in_graph]

    def _block(label, rows):
        if not rows:
            print(f"  {label}: (none)")
            return
        topic = _mean([r.on_topic.get("on_topic_frac", 0.0) for r in rows])
        cos = _mean([r.on_topic.get("mean_cosine", 0.0) for r in rows])
        uns = _mean([r.unsupported.get("unsupported_count", 0) for r in rows])
        sup = _mean([r.unsupported.get("supported_count", 0) for r in rows])
        wc = _mean([r.fluency.get("word_count", 0) for r in rows])
        ur = _mean([r.fluency.get("unique_ratio", 0.0) for r in rows])
        salad = sum(1 for r in rows if r.fluency.get("is_word_salad"))
        print(f"  {label} (n={len(rows)}):")
        print(f"      mean on-topic fraction      : {topic:.4f}")
        print(f"      mean content-word cosine    : {cos:.4f}")
        print(f"      mean unsupported assertions : {uns:.4f}")
        print(f"      mean supported assertions   : {sup:.4f}")
        print(f"      mean word count             : {wc:.2f}")
        print(f"      mean unique ratio           : {ur:.4f}")
        print(f"      word-salad flags            : {salad}/{len(rows)}")

    _block("UNGATED, graph-known subjects", known)
    _block("UNGATED, unknown controls", unknown)
    _block("UNGATED, all subjects", ungated)

    # correlation between on-topic and unsupported assertions
    xs = [r.on_topic.get("on_topic_frac", 0.0) for r in ungated]
    ys = [float(r.unsupported.get("unsupported_count", 0)) for r in ungated]
    corr = _pearson(xs, ys)
    print(f"      corr(on-topic frac, unsupported count) = "
          f"{'n/a' if corr is None else f'{corr:.4f}'}")

    if gated:
        gknown = [r for r in gated if r.in_graph]
        if gknown:
            print(f"  GATED, graph-known subjects (n={len(gknown)}):")
            print(f"      mean on-topic fraction      : "
                  f"{_mean([r.gated_on_topic.get('on_topic_frac', 0.0) for r in gknown]):.4f}")
            print(f"      mean unsupported assertions : "
                  f"{_mean([r.gated_unsupported.get('unsupported_count', 0) for r in gknown]):.4f}")
            print(f"      word-salad flags            : "
                  f"{sum(1 for r in gknown if r.gated_fluency.get('is_word_salad'))}/{len(gknown)}")
        print(f"  gate withholding rate: {len(gated)}/{len(ok)} subjects produced "
              f"gated output; {len(ok) - len(gated)} withheld/none")

    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for r in errors:
            print(f"      {r.subject}: {r.error[:100]}")

    # strategy distribution
    from collections import Counter
    c = Counter(r.ungated_strategy for r in ungated)
    print(f"  ungated strategy distribution: {dict(c)}")

    # ── strategy split ────────────────────────────────────────────────────
    # The decoder is only ONE of three paths. The narrative / syntax paths
    # are NOT the neural decoder, so a mean over all rows mixes the
    # decoder with its fallbacks. Print the decoder-only mean separately
    # (limitation #4 in the module docstring).
    print()
    for strat in ("situation_model_decoder", "situation_model_narrative",
                  "situation_model_syntax"):
        rows_s = [r for r in ungated if r.ungated_strategy == strat and r.in_graph]
        if not rows_s:
            continue
        print(f"  UNGATED, strategy={strat} (graph-known, n={len(rows_s)}):")
        print(f"      mean on-topic fraction      : "
              f"{_mean([r.on_topic.get('on_topic_frac', 0.0) for r in rows_s]):.4f}")
        print(f"      mean content-word cosine    : "
              f"{_mean([r.on_topic.get('mean_cosine', 0.0) for r in rows_s]):.4f}")
        print(f"      mean unsupported assertions : "
              f"{_mean([r.unsupported.get('unsupported_count', 0) for r in rows_s]):.4f}")
        print(f"      mean supported assertions   : "
              f"{_mean([r.unsupported.get('supported_count', 0) for r in rows_s]):.4f}")
        print(f"      mean word count             : "
              f"{_mean([r.fluency.get('word_count', 0) for r in rows_s]):.2f}")
        print(f"      mean unique ratio           : "
              f"{_mean([r.fluency.get('unique_ratio', 0.0) for r in rows_s]):.4f}")
        print(f"      word-salad flags            : "
              f"{sum(1 for r in rows_s if r.fluency.get('is_word_salad'))}/{len(rows_s)}")


def print_engine_info(eng: CognitiveChatEngine, label: str) -> None:
    print(f"  [{label}] nodes={len(eng.graph.nodes)} edges={len(eng.graph.edges)} "
          f"definitions={len(getattr(eng, '_definitions', {}))} "
          f"sources={len(getattr(eng, '_concept_sources', {}))} "
          f"concept_keywords={len(getattr(eng, '_concept_keywords', {}))} "
          f"decoder_vocab={len(getattr(eng, '_decoder_word_to_idx', {}))} "
          f"trained_decoder_restored={getattr(eng, '_trained_decoder_restored', None)} "
          f"decoder_dim={eng._decoder_dim()}")


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════

def build_engine(kind: str) -> Tuple[Optional[CognitiveChatEngine], str]:
    """kind in {'baby', 'persisted'}."""
    if kind == "baby":
        data_dir = tempfile.mkdtemp(prefix="ravana_eg_")
        label = "baby(tempdir,_eg)"
        print(f"  [baby] data_dir={data_dir}")
        t0 = time.time()
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                                  data_dir=data_dir, user_suffix="_eg")
        print(f"  [baby] booted in {time.time()-t0:.1f}s")
        return eng, label

    weights = os.path.join(_REPO_ROOT, "weights")
    label = "persisted(weights,'')"
    print(f"  [persisted] data_dir={weights}")
    if not os.path.exists(os.path.join(weights, "ravana_weights.pkl")):
        print("  [persisted] no ravana_weights.pkl -- skipping")
        return None, label
    t0 = time.time()
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              data_dir=weights, user_suffix="")
    print(f"  [persisted] booted in {time.time()-t0:.1f}s")
    return eng, label


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None,
                    help="TSV path; the script appends and resumes from it")
    ap.add_argument("--engine", choices=["baby", "persisted", "both"],
                    default="both",
                    help="which engine state to measure (default: both)")
    ap.add_argument("--max-subjects", type=int, default=None,
                    help="cap the number of graph-discovered subjects")
    ap.add_argument("--raw-input", choices=["what_is", "tell_me", "both"],
                    default="both",
                    help="which synthetic raw_input to use per subject")
    ap.add_argument("--cpg", action="store_true",
                    help="enable Committed-Proposition Gating while measuring "
                         "(default: off, the production default)")
    ap.add_argument("--cpg-relation", action="store_true",
                    help="with --cpg, also gate on the committed RELATION "
                         "(relation_words), not just the target")
    ap.add_argument("--cpg-theta", type=float, default=None,
                    help="with --cpg, override theta_base (the ACC conflict "
                         "criterion). Higher demands more evidence before "
                         "committing; default 0.30")
    args = ap.parse_args(argv)

    print("=" * 78)
    print("RAVANA decoder groundedness instrument (MEASUREMENT ONLY -- no fix)")
    print("=" * 78)
    print(f"  repo root     : {_REPO_ROOT}")
    print(f"  engine(s)     : {args.engine}")
    print(f"  out TSV       : {args.out or '(none -- not resumable)'}")
    print(f"  raw_input(s)  : {args.raw_input}")

    # CPG configuration for this run. `--cpg` alone is target-only (the
    # configuration that measurably worked); `--cpg-relation` adds the
    # relation_words map so the committed RELATION is gated too.
    cpg_config = None
    if args.cpg:
        from ravana.learn.committed_proposition import CPGConfig
        rel = None
        if args.cpg_relation:
            rel = {
                "causal": ("leads", "causes", "makes", "creates", "gives"),
                "causes": ("leads", "causes", "makes", "creates", "gives"),
                "temporal": ("follows", "precedes", "comes"),
                "semantic": ("like", "is", "are", "becomes"),
                "is_a": ("is", "are", "becomes"),
                "part_of": ("part", "contains", "includes"),
                "related_to": ("like", "connected", "linked"),
            }
        cpg_config = CPGConfig(relation_words=rel)
        if args.cpg_theta is not None:
            cpg_config.theta_base = args.cpg_theta
        print(f"  cpg           : on (relation_gating={args.cpg_relation}, "
              f"theta_base={cpg_config.theta_base})")
    else:
        print(f"  cpg           : off")

    existing = read_existing(args.out) if args.out else {}
    if existing:
        print(f"  [resume] {len(existing)} completed rows found in {args.out}")

    engines: List[Tuple[str, CognitiveChatEngine, str]] = []
    if args.engine in ("baby", "both"):
        eng, label = build_engine("baby")
        if eng is not None:
            print_engine_info(eng, label)
            engines.append(("baby", eng, label))
    if args.engine in ("persisted", "both"):
        eng, label = build_engine("persisted")
        if eng is not None:
            print_engine_info(eng, label)
            engines.append(("persisted", eng, label))

    if not engines:
        print("  no engine available -- nothing to measure.")
        return 0

    all_results: List[SubjectResult] = []
    for kind, eng, label in engines:
        subjects = discover_subjects(eng)
        if args.max_subjects:
            # keep the nominated + unknown controls, cap the discovered tail
            nominated = [s for s in subjects if s in NOMINATED_SUBJECTS]
            rest = [s for s in subjects if s not in NOMINATED_SUBJECTS]
            subjects = nominated + rest[: max(0, args.max_subjects - len(nominated))]
        print()
        print(f"  [{label}] {len(subjects)} subjects: {subjects}")

        raw_inputs = []
        if args.raw_input in ("what_is", "both"):
            raw_inputs.append("what is {s}")
        if args.raw_input in ("tell_me", "both"):
            raw_inputs.append("tell me about {s}")

        new_rows: List[Dict[str, str]] = []
        for idx, subject in enumerate(subjects, 1):
            for tmpl in raw_inputs:
                raw = tmpl.format(s=subject)
                key = (label, subject, raw)
                if key in existing:
                    print(f"  [{idx}/{len(subjects)}] {subject!r} ({raw!r}) "
                          f"-- already in TSV, skipping")
                    # Reconstruct a minimal result so the table is complete.
                    # Values come back from csv as STRINGS -- coerce them to
                    # the numeric/bool types the scorers would have produced.
                    row = existing[key]
                    r = SubjectResult(engine=label, subject=subject, raw_input=raw)
                    r.ungated_text = row.get("ungated_text", "")
                    r.ungated_strategy = row.get("ungated_strategy", "")
                    r.gated_text = row.get("gated_text", "")
                    r.gated_strategy = row.get("gated_strategy", "")
                    r.in_graph = row.get("in_graph") == "1"
                    r.has_definition = row.get("has_definition") == "1"
                    r.has_source = row.get("has_source") == "1"
                    try:
                        r.degree = int(row.get("degree") or 0)
                        r.neighbour_count = int(row.get("neighbour_count") or 0)
                    except ValueError:
                        pass

                    def _num(col):
                        v = row.get(col)
                        if v in (None, ""):
                            return None
                        try:
                            f = float(v)
                        except ValueError:
                            return None
                        return int(f) if col.endswith(("count", "words")) else f

                    for fld, col in (("word_count", "ungated_word_count"),
                                     ("unique_ratio", "ungated_unique_ratio"),
                                     ("is_word_salad", "ungated_word_salad"),
                                     ("is_word_salad_any_sentence",
                                      "ungated_word_salad")):
                        v = _num(col)
                        if v is not None:
                            r.fluency[fld] = bool(v) if fld.startswith("is_") else v
                    for fld, col in (("content_words", "ungated_content_words"),
                                     ("overlap_count", "ungated_overlap_count"),
                                     ("on_topic_frac", "ungated_on_topic_frac"),
                                     ("mean_cosine", "ungated_mean_cosine")):
                        v = _num(col)
                        if v is not None:
                            r.on_topic[fld] = v
                    for fld, col in (("unsupported_count",
                                      "ungated_unsupported_count"),
                                     ("supported_count",
                                      "ungated_supported_count")):
                        v = _num(col)
                        if v is not None:
                            r.unsupported[fld] = int(v)
                    for fld, col in (("word_count", "gated_word_count"),
                                     ("unique_ratio", "gated_unique_ratio"),
                                     ("is_word_salad", "gated_word_salad")):
                        v = _num(col)
                        if v is not None:
                            r.gated_fluency[fld] = bool(v) if fld.startswith("is_") else v
                    for fld, col in (("content_words", "gated_content_words"),
                                     ("overlap_count", "gated_overlap_count"),
                                     ("on_topic_frac", "gated_on_topic_frac"),
                                     ("mean_cosine", "gated_mean_cosine")):
                        v = _num(col)
                        if v is not None:
                            r.gated_on_topic[fld] = v
                    v = _num("gated_unsupported_count")
                    if v is not None:
                        r.gated_unsupported["unsupported_count"] = int(v)
                    all_results.append(r)
                    continue

                r = measure_subject(eng, label, subject, raw, use_cpg=args.cpg,
                                    cpg_config=cpg_config)
                all_results.append(r)
                row = result_to_row(r)
                new_rows.append(row)
                status = r.status
                if r.ungated_text:
                    print(f"  [{idx}/{len(subjects)}] {subject!r} ({raw!r}) "
                          f"{status} strategy={r.ungated_strategy} "
                          f"unsup={r.unsupported.get('unsupported_count')} "
                          f"topic={r.on_topic.get('on_topic_frac', 0):.2f} "
                          f"t={r.elapsed_s:.1f}s")
                else:
                    print(f"  [{idx}/{len(subjects)}] {subject!r} ({raw!r}) "
                          f"{status} t={r.elapsed_s:.1f}s")

            # flush after each subject so a kill mid-run keeps progress
            if new_rows:
                append_rows(args.out, new_rows)
                new_rows = []

        print_table([r for r in all_results if r.engine == label])
        print_summary([r for r in all_results if r.engine == label], label)

    if len(engines) > 1:
        print()
        print("=" * 78)
        print("ENGINE COMPARISON")
        print("=" * 78)
        for kind, _eng, label in engines:
            rows = [r for r in all_results if r.engine == label and r.ungated_text]
            known = [r for r in rows if r.in_graph]
            if not known:
                continue
            print(f"  {label}: n={len(known)} "
                  f"mean_on_topic={_mean([r.on_topic.get('on_topic_frac', 0) for r in known]):.4f} "
                  f"mean_unsupported={_mean([r.unsupported.get('unsupported_count', 0) for r in known]):.4f} "
                  f"mean_cosine={_mean([r.on_topic.get('mean_cosine', 0) for r in known]):.4f}")

    print()
    print("DONE. This instrument measures; it does not fix. See the module")
    print("docstring for what it can and cannot measure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
