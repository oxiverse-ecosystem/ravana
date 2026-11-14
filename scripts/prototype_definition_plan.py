#!/usr/bin/env python3
"""PROTOTYPE: build the surface plan from the DEFINITION, not from a typed edge.

The plan route currently requires a typed graph edge. Coverage is ~40% because
typed edges are sparse, and the online-typing mechanism is measured as not
working (docs/TYPING_ACCURACY_IS_CIRCULAR.md).

But RAVANA already has 324 stored definitions -- real retrieved text. The
brain's route for a novel concept is exactly this: it does not need a stored
relation label, it READS the relation off the concept's description.

So: parse the definition's own predicate to recover (relation, target), then
build the SurfacePlan from that. No typed edge required.

READ-ONLY. Usage:
    uv run python scripts/prototype_definition_plan.py [subject ...]
"""
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from eval_decoder_groundedness import (  # noqa: E402
    build_engine, build_ctx, score_on_topic, score_unsupported_assertions,
)

from prototype_linearization import _frame_score  # noqa: E402

# Copula + predicate patterns. These read the DEFINITION's surface form, which
# is how a reader (human or system) recovers what kind of claim is being made.
_PATTERNS = [
    # "X is a Y", "X was a Y", "X is the Y that ..."
    ("is_a", re.compile(
        r"\b(?:is|was|are|were)\s+(?:a|an|the)\s+([a-z][a-z\s]{2,40})", re.I)),
    # "X is the Y by which ..." (gravity)
    ("has_property", re.compile(
        r"\b(?:is|was)\s+the\s+([a-z][a-z\s]{2,30}?)\s+(?:by|that|which|of)\b",
        re.I)),
    # "X, a Y, ..." appositive
    ("is_a", re.compile(r"\b[a-z]+\s*,\s*(?:a|an)\s+([a-z][a-z\s]{2,30}),", re.I)),
]

# Stop words that begin a predicate phrase but are not the target.
_NOT_TARGET = re.compile(
    r"^(?:the|a|an|that|which|who|whose|what|process|act|quality|state|"
    r"result|product|form|part|type|kind|sort|branch|member|aspect|"
    r"phenomenon|concept|term|word|name|way|means|method|system)$", re.I)


def _clean_target(phrase: str) -> str:
    """Trim a predicate phrase to its head noun-ish content."""
    words = re.findall(r"[a-z][a-z-]*", phrase.lower())
    if not words:
        return ""
    # Drop leading function words.
    while words and words[0] in ("the", "a", "an", "of", "to", "in", "of"):
        words.pop(0)
    return " ".join(words[:3])


def _graph_target(eng, subject: str, phrase: str) -> str:
    """Resolve a predicate phrase to a GRAPH CONCEPT.

    Copying the phrase verbatim produces targets the graph has never heard of
    ("force by which", "body of salt"), and a target outside the graph is a
    claim the graph cannot support. The brain does not copy a description's
    words either: it maps the description onto concepts it already has.

    So: score each content word of the phrase against the subject's graph
    neighbours and return the best-resolved one. Falls back to the phrase.
    """
    words = [w for w in re.findall(r"[a-z][a-z-]*", phrase.lower())
             if w not in ("the", "a", "an", "of", "to", "in", "by", "which",
                          "that", "and", "or")]
    if not words:
        return ""
    nids = eng._concept_keywords.get(subject.lower(), [])
    neigh = set()
    for nid in nids:
        for tgt, _ed in (eng.graph.get_outgoing(nid) or []):
            tn = eng.graph.get_node(tgt)
            if tn is not None and tn.label:
                neigh.add(tn.label.lower())
    # Prefer a phrase word that IS a graph neighbour of the subject.
    for w in words:
        if w in neigh:
            return w
    for w in words:
        if w in eng._concept_keywords:
            return w
    return words[0]


def _head_noun(eng, words):
    """Pick the NOUN from a predicate phrase, preferring the LAST one.

    English noun phrases are right-headed: "basic unit of classification" heads
    on 'unit', "continuous progression of existence" on 'progression'. Taking
    the first content word yields 'basic'/'continuous' -- adjectives, which is
    a claim about nothing. The engine's learned POS map is the authority; the
    right-most content word is the fallback (English right-headedness).
    """
    pos = getattr(eng, "_concept_pos", {}) or {}
    nouns = [w for w in words if pos.get(w) in ("noun", "proper_noun")]
    if nouns:
        return nouns[-1]
    if words:
        return words[-1]
    return ""


def definition_proposition(eng, subject: str, definition: str):
    """Recover (relation, target) from a definition's surface form.

    The relation comes from the copula pattern; the target is the predicate
    phrase resolved to a graph concept. Returns None when the definition does
    not assert a categorizing or attributive claim.
    """
    if not definition:
        return None
    for rel, rx in _PATTERNS:
        m = rx.search(str(definition))
        if not m:
            continue
        raw = _clean_target(m.group(1))
        if not raw or _NOT_TARGET.match(raw):
            continue
        tgt = _graph_target(eng, subject, raw)
        if not tgt:
            continue
        # Re-resolve with head-noun preference: _graph_target returns the first
        # phrase word that is a graph concept, which is usually the adjective.
        words = [w for w in re.findall(r"[a-z][a-z-]*", raw.lower())
                 if w not in ("the", "a", "an", "of", "to", "in", "by",
                              "which", "that", "and", "or")]
        head = _head_noun(eng, words)
        if head:
            tgt = head
        return (rel, tgt)
    return None


def main(subjects):
    eng, label = build_engine("persisted")
    if eng is None:
        return 1
    defs = getattr(eng, "_definitions", {}) or {}
    print(f"[defplan] engine={label} definitions={len(defs)}")

    for subj in subjects:
        ctx = build_ctx(eng, subj, f"what is {subj}")
        d = defs.get(subj.lower())
        prop = definition_proposition(eng, subj, d) if d else None
        print(f"\n=== {subj} ===")
        print(f"  defn: {str(d)[:95]}")
        print(f"  proposition: {prop}")
        if not prop:
            print("  -> no proposition")
            continue
        rel, tgt = prop
        from ravana.learn.linearization import message_to_plan, plan_is_committed
        from ravana.learn.macroplanning import message_from_best_edge
        plan = message_to_plan(
            message_from_best_edge(subj, [(rel, tgt, 0.9, 0.8)]), subject=subj)
        if not plan_is_committed(plan):
            print(f"  -> plan not committed: {plan}")
            continue
        print(f"  plan: {' '.join(plan.tokens)}")
        from ravana.language.surface_realizer import DiscourseState
        relmap = {"is_a": "is_a", "has_property": "has_property"}
        try:
            f = eng.syntactic_assembly.bind_to_sentence(
                subject=subj, relation=relmap.get(rel, "related_to"), target=tgt,
                pos_map=getattr(eng, "_concept_pos", {}))
            s = eng.surface_realizer.realize(
                frame=f, discourse_context=DiscourseState(
                    sentence_index=0, discourse_type="explain",
                    total_sentences=1, free_energy=0.3),
                dopamine_tone=0.5,
                cerebellar_ngram=getattr(eng, "cerebellar_ngram", None))
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}")
            continue
        if s:
            ot = score_on_topic(eng, subj, s)
            un = score_unsupported_assertions(eng, subj, s)
            fr = _frame_score(s)
            print(f"  on={ot['on_topic_frac']:.2f} "
                  f"unsup={un.get('unsupported_count',0)} "
                  f"func={fr['func_share']:.2f} verb={fr['has_verb']}")
            print(f"  -> {s[:120]}")
    return 0


if __name__ == "__main__":
    subs = sys.argv[1:] or ["gravity", "ocean", "species", "time", "music", "dog"]
    raise SystemExit(main(subs))
