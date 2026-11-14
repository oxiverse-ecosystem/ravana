#!/usr/bin/env python3
"""PROTOTYPE: does conditioning on a LINEARIZED PLAN beat a word-list boost?

This is the falsifiable test of the linearization hypothesis. It compares
three ways of steering the SAME decoder on the SAME subjects:

  A. BASELINE   -- the decoder's own distribution (no steering)
  B. WORD BOOST -- a logit boost on the definition's content words
                  (the measured failure: on-topic 1.00, no syntax)
  C. PLAN       -- condition the decoder on an ordered surface plan derived
                  from the graph message (macroplanning -> linearization),
                  then elaborate

The instrument reports, per arm: on-topic fraction, unsupported assertions,
function-word share, and a grammatical-frame check. The last is the axis the
original 9.0-knee sweep could not see (its metric was invocab * distinct-1 *
on-topic, where distinct-1 is the type-token ratio).

READ-ONLY on production code. Usage:
    uv run python scripts/prototype_linearization.py [subject ...]
"""
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from eval_decoder_groundedness import (  # noqa: E402
    build_engine, build_ctx, score_fluency, score_on_topic,
    score_unsupported_assertions, _content_words, _subject_neighbours,
)

# Function words that carry grammatical structure. Their presence is the
# cheapest available signal that the output is a CLAUSE rather than a list.
_FUNC = frozenset("""
a an the and but or is are was were be been being have has had do does did
will would could should may might shall can not no nor so if then than too
very just about also into over after before between through during because
while which who whom what when where why how all each every both few more
most some any this that these those it its they them their we our you your
he she him her his i me my of to for with from at by as on in out up off
""".split())

_REL_RE = re.compile(
    r"\b(is|are|was|were|leads|causes|makes|creates|gives|has|have|contains|"
    r"includes|produces|forms|drives|shapes|influences|affects|determines|"
    r"defines|describes|represents|mirrors|reflects|contrasts|opposes|"
    r"becomes|grows|flows|runs|reaches|extends|opens|closes|like|related|"
    r"connected|linked|akin|similar|resembles|means|implies|suggests|"
    r"located|part|member|kind|type|category)\b", re.I)


def _frame_score(text: str) -> dict:
    """Structural frame check: is this a clause or a word list?"""
    toks = re.findall(r"[a-z']+", text.lower())
    if not toks:
        return {"func_share": 0.0, "has_verb": False, "frame": 0.0}
    n_func = sum(1 for t in toks if t in _FUNC)
    has_verb = bool(_REL_RE.search(text.lower()))
    # A clause needs both grammatical glue and a predicate.
    return {
        "func_share": n_func / len(toks),
        "has_verb": has_verb,
        "frame": (1.0 if has_verb else 0.0) * min(1.0, n_func / max(3, len(toks) // 3)),
    }


def _arm_baseline(eng, ctx):
    return eng._generate_with_situation_model(ctx)


def _arm_word_boost(eng, ctx, strength=9.0):
    from ravana.learn import anchor_boost as ab
    saved = ab.anchor_token_boost

    def _forced(desc, w2i, *, strength=strength, _f=saved):
        return _f(desc, w2i, strength=strength)
    ab.anchor_token_boost = _forced
    try:
        return eng._generate_with_situation_model(ctx)
    finally:
        ab.anchor_token_boost = saved


def _arm_plan(eng, ctx):
    """Condition on a linearized plan.

    The plan's tokens are fed to the decoder as a PREFIX the model must
    continue from, rather than as a logit bias over the whole vocabulary.
    """
    from ravana.learn.macroplanning import build_message
    from ravana.learn.linearization import message_to_plan

    subj = (getattr(ctx, "subject", None) or "").strip()
    if not subj:
        return None
    nids = eng._concept_keywords.get(subj.lower(), [])
    if not nids:
        return None
    nid = nids[0]
    cands = []
    for tgt, ed in (eng.graph.get_outgoing(nid) or []):
        tn = eng.graph.get_node(tgt)
        lab = (getattr(tn, "label", "") or "").strip()
        if lab:
            cands.append((getattr(ed, "relation_type", "related_to")
                          or "related_to", lab,
                          float(getattr(ed, "weight", 0.0) or 0.0),
                          float(getattr(ed, "confidence", 0.0) or 0.0)))
    msg = build_message(subj, cands)
    plan = message_to_plan(msg, subject=subj)
    if plan is None:
        return None
    print(f"    [plan] {plan.construction}: {' '.join(plan.tokens)}")
    # Feed the plan as conditioning: its tokens' embeddings become the
    # conditioning set, so the decoder elaborates FROM the ordered plan.
    return _generate_with_plan_prefix(eng, ctx, plan)


def _generate_with_plan_prefix(eng, ctx, plan):
    """Run the decoder with the plan's tokens as a prefix to continue from."""
    nd = getattr(eng, "neural_decoder", None)
    if nd is None:
        return None
    iw = getattr(eng, "_decoder_idx_to_word", {}) or {}
    w2i = getattr(eng, "_decoder_word_to_idx", {}) or {}
    prefix = [w2i[t] for t in plan.tokens if t in w2i]
    if not prefix:
        return None
    # Use the production entry point; the plan prefix is passed by temporarily
    # seeding the decoder's recent-prediction buffer, which the generate loop
    # reads as the emitted-so-far context.
    nd._recent_predictions.extend(prefix)
    try:
        return eng._generate_with_situation_model(ctx)
    finally:
        del nd._recent_predictions[-len(prefix):]


def main(subjects):
    eng, label = build_engine("persisted")
    if eng is None:
        return 1
    print(f"[proto] engine={label}")

    arms = [("BASELINE", _arm_baseline), ("WORD_BOOST", _arm_word_boost),
            ("PLAN", _arm_plan)]

    for subj in subjects:
        ctx = build_ctx(eng, subj, f"what is {subj}")
        print(f"\n=== {subj} ===")
        for name, fn in arms:
            try:
                out = fn(eng, ctx)
            except Exception as e:
                print(f"  {name:<12} ERROR {type(e).__name__}: {e}")
                continue
            if not out:
                print(f"  {name:<12} <no output>")
                continue
            text, strat = out
            ot = score_on_topic(eng, subj, text)
            un = score_unsupported_assertions(eng, subj, text)
            fr = _frame_score(text)
            print(f"  {name:<12} on={ot['on_topic_frac']:.2f} "
                  f"unsup={un.get('unsupported_count', 0)} "
                  f"func={fr['func_share']:.2f} verb={fr['has_verb']} "
                  f"frame={fr['frame']:.2f} strat={strat}")
            print(f"               {text[:130]}")
    return 0


if __name__ == "__main__":
    subs = sys.argv[1:] or ["gravity", "ocean", "species", "time"]
    raise SystemExit(main(subs))
