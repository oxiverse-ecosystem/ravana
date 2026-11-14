#!/usr/bin/env python3
"""Scale the plan-driven NARRATIVE formulator vs the production baseline.

The narrative path emits grammatical text with no neural network and scores
3.5x the decoder on on-topic (0.242 vs 0.069). This measures whether driving
it from a committed SurfacePlan improves grounding without losing grammar.

Arms:
  BASELINE   -- _generate_narrative_paragraph as it stands
  PLAN_NARR  -- the same formulator driven by a committed SurfacePlan

READ-ONLY. Usage:
    uv run python scripts/measure_plan_narrative.py --max-subjects 40
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from eval_decoder_groundedness import (  # noqa: E402
    build_engine, build_ctx, discover_subjects, score_on_topic,
    score_unsupported_assertions,
)
from prototype_linearization import _frame_score  # noqa: E402
from prototype_plan_narrative import _plan_for, _realize_from_plan  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-subjects", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    eng, label = build_engine("persisted")
    if eng is None:
        return 1
    subs = discover_subjects(eng)
    if args.max_subjects:
        subs = subs[:args.max_subjects]
    print(f"[measure] engine={label} subjects={len(subs)}")

    rows = []
    for i, subj in enumerate(subs):
        ctx = build_ctx(eng, subj, f"what is {subj}")
        row = {"subject": subj}
        base = eng._generate_narrative_paragraph(ctx)
        row["baseline"] = (score_on_topic(eng, subj, base)["on_topic_frac"],
                           score_unsupported_assertions(eng, subj, base)
                           .get("unsupported_count", 0),
                           _frame_score(base)["frame"], base[:120]) if base else None
        plan = _plan_for(eng, subj)
        row["plan"] = " ".join(plan.tokens) if plan else ""
        s = None
        if plan:
            try:
                s = _realize_from_plan(eng, ctx, plan)
            except Exception as e:
                row["plan_error"] = f"{type(e).__name__}: {e}"
        if s:
            row["plan_narr"] = (
                score_on_topic(eng, subj, s)["on_topic_frac"],
                score_unsupported_assertions(eng, subj, s)
                .get("unsupported_count", 0),
                _frame_score(s)["frame"], s[:120])
        else:
            row["plan_narr"] = None
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(subs)}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        print(f"[measure] wrote {args.out}")

    def _m(key, idx):
        v = [r[key][idx] for r in rows if r.get(key)]
        return (sum(v) / len(v)) if v else 0.0

    print("\n" + "=" * 74)
    print(f"{'arm':<12} {'n':>4} {'on_topic':>9} {'unsup':>7} {'frame':>7}")
    print("-" * 74)
    for arm in ("baseline", "plan_narr"):
        n = sum(1 for r in rows if r.get(arm))
        print(f"{arm:<12} {n:>4} {_m(arm,0):>9.4f} {_m(arm,1):>7.3f} {_m(arm,2):>7.3f}")
    nb = sum(1 for r in rows if r.get("baseline"))
    np_ = sum(1 for r in rows if r.get("plan_narr"))
    print("-" * 74)
    print(f"baseline produced {nb}/{len(subs)}; plan_narr produced {np_}/{len(subs)}")
    print(f"subjects with a committed plan: "
          f"{sum(1 for r in rows if r.get('plan'))}/{len(subs)}")

    # Paired comparison on subjects where BOTH produced output.
    both = [r for r in rows if r.get("baseline") and r.get("plan_narr")]
    if both:
        d_on = sum(r["plan_narr"][0] - r["baseline"][0] for r in both) / len(both)
        d_un = sum(r["plan_narr"][1] - r["baseline"][1] for r in both) / len(both)
        print(f"\npaired on n={len(both)}: d_on_topic={d_on:+.4f} "
              f"d_unsupported={d_un:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
