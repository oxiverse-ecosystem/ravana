#!/usr/bin/env python3
"""PROTOTYPE: drive the NARRATIVE formulator from a linearized plan.

`docs/PLAN_PREFIX_MEASURED_NEGATIVE.md` showed the GRU cannot continue an
imposed frame. But the narrative/syntax path emits grammatical text with no
neural network at all, and it scores 3.5x the decoder on on-topic
(0.242 vs 0.069, docs/DECODER_ONLY_NUMBERS.md).

So the one remaining brain-faithful option is to condition THAT formulator on
the SurfacePlan rather than the GRU. `bind_to_sentence(subject, relation,
target, pos_map)` already takes exactly what a plan provides.

Three arms:
  BASELINE    -- the production narrative path as it stands
  PLAN_NARR   -- the narrative path driven by a committed SurfacePlan
  PLANselect  -- same, but showing which plan was chosen

READ-ONLY on production code. Usage:
    uv run python scripts/prototype_plan_narrative.py [subject ...]
"""
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from eval_decoder_groundedness import (  # noqa: E402
    build_engine, build_ctx, score_on_topic, score_unsupported_assertions,
)

from prototype_linearization import _frame_score  # noqa: E402


def _plan_for(eng, subj):
    """Build a committed SurfacePlan for a subject, or None."""
    from ravana.learn.macroplanning import build_message
    from ravana.learn.linearization import (
        message_to_plan, plan_is_committed, VACUOUS_RELATIONS)

    nids = eng._concept_keywords.get(subj.lower(), [])
    if not nids:
        return None
    cands = []
    for tgt, ed in (eng.graph.get_outgoing(nids[0]) or []):
        tn = eng.graph.get_node(tgt)
        lab = (getattr(tn, "label", "") or "").strip()
        rel = getattr(ed, "relation_type", "related_to") or "related_to"
        if lab:
            cands.append((rel, lab, float(getattr(ed, "weight", 0.0) or 0.0),
                          float(getattr(ed, "confidence", 0.0) or 0.0)))
    if not cands:
        return None
    typed = [c for c in cands if c[0] not in VACUOUS_RELATIONS]
    for pool in (typed, cands):
        if not pool:
            continue
        plan = message_to_plan(build_message(subj, pool), subject=subj)
        if plan_is_committed(plan):
            return plan
    return None


def _realize_from_plan(eng, ctx, plan):
    """Run the narrative formulator with the plan's triple as its input."""
    if not hasattr(eng, "syntactic_assembly") or not hasattr(eng, "surface_realizer"):
        return None
    # The plan's fill gives the (subject, relation, target) triple that
    # bind_to_sentence already accepts.
    subj = plan.subject
    target = None
    for role in ("ground", "relation", "theme"):
        if role in plan.fill and plan.fill[role].lower() != subj.lower():
            target = plan.fill[role]
            break
    if target is None:
        return None
    # Map the construction back to a relation_type the realizer understands.
    rel_map = {
        "categorize": "is_a", "cause": "causes", "attribute": "has_property",
        "locate": "located_in", "relate": "related_to",
    }
    relation = rel_map.get(plan.construction, "related_to")

    frame = eng.syntactic_assembly.bind_to_sentence(
        subject=subj, relation=relation, target=target,
        pos_map=getattr(eng, "_concept_pos", {}),
    )
    from ravana.language.surface_realizer import DiscourseState
    s = eng.surface_realizer.realize(
        frame=frame,
        discourse_context=DiscourseState(
            sentence_index=0, discourse_type="explain",
            total_sentences=1, free_energy=0.3),
        dopamine_tone=getattr(eng, "_dopamine_tone", 0.5),
        cerebellar_ngram=getattr(eng, "cerebellar_ngram", None),
    )
    return s


def main(subjects):
    eng, label = build_engine("persisted")
    if eng is None:
        return 1
    print(f"[proto] engine={label}")

    for subj in subjects:
        ctx = build_ctx(eng, subj, f"what is {subj}")
        print(f"\n=== {subj} ===")

        plan = _plan_for(eng, subj)
        print(f"  plan: {(' '.join(plan.tokens) if plan else 'None')}")

        # BASELINE: production narrative path.
        base = eng._generate_narrative_paragraph(ctx)
        if base:
            ot = score_on_topic(eng, subj, base)
            un = score_unsupported_assertions(eng, subj, base)
            fr = _frame_score(base)
            print(f"  BASELINE  on={ot['on_topic_frac']:.2f} "
                  f"unsup={un.get('unsupported_count',0)} "
                  f"func={fr['func_share']:.2f} verb={fr['has_verb']}")
            print(f"            {base[:140]}")
        else:
            print("  BASELINE  <no output>")

        # PLAN_NARR: the formulator driven by the plan.
        if plan:
            try:
                s = _realize_from_plan(eng, ctx, plan)
            except Exception as e:
                s = None
                print(f"  PLAN_NARR ERROR {type(e).__name__}: {e}")
            if s:
                ot = score_on_topic(eng, subj, s)
                un = score_unsupported_assertions(eng, subj, s)
                fr = _frame_score(s)
                print(f"  PLAN_NARR on={ot['on_topic_frac']:.2f} "
                      f"unsup={un.get('unsupported_count',0)} "
                      f"func={fr['func_share']:.2f} verb={fr['has_verb']}")
                print(f"            {s[:140]}")
            else:
                print("  PLAN_NARR <no output>")
    return 0


if __name__ == "__main__":
    subs = sys.argv[1:] or ["gravity", "ocean", "species", "time", "music"]
    raise SystemExit(main(subs))
