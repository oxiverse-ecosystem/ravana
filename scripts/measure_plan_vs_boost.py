#!/usr/bin/env python3
"""Scale the three-arm comparison and report paired deltas.

Arms:
  BASELINE   -- the decoder's own distribution
  WORD_BOOST -- logit boost on the definition's content words (9.0)
  PLAN       -- continue from a linearized surface plan

Reports per-arm means and the paired delta vs BASELINE, plus a FRAME rate
(the fraction of outputs that are clauses rather than word lists). The frame
rate is the axis the original 9.0-knee sweep could not see.

READ-ONLY. Usage:
    uv run python scripts/measure_plan_vs_boost.py --max-subjects 40 --out tmp/plan.tsv
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from eval_decoder_groundedness import (  # noqa: E402
    build_engine, build_ctx, discover_subjects, score_fluency, score_on_topic,
    score_unsupported_assertions,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from prototype_linearization import _frame_score  # noqa: E402


def _run_arm(eng, ctx, arm):
    if arm == "baseline":
        return eng._generate_with_situation_model(ctx)
    if arm == "word_boost":
        from ravana.learn import anchor_boost as ab
        saved = ab.anchor_token_boost
        ab.anchor_token_boost = (
            lambda d, w, *, strength=9.0, _f=saved: _f(d, w, strength=strength))
        try:
            return eng._generate_with_situation_model(ctx)
        finally:
            ab.anchor_token_boost = saved
    if arm == "plan":
        eng._plan_prefix_enabled = True
        try:
            return eng._generate_with_situation_model(ctx)
        finally:
            eng._plan_prefix_enabled = False
    raise ValueError(arm)


def _score(eng, subj, out):
    if not out:
        return None
    text, strat = out
    ot = score_on_topic(eng, subj, text)
    un = score_unsupported_assertions(eng, subj, text)
    fr = _frame_score(text)
    return {
        "text": text,
        "strategy": strat,
        "on_topic": ot["on_topic_frac"],
        "unsupported": un.get("unsupported_count", 0),
        "func_share": fr["func_share"],
        "has_verb": fr["has_verb"],
        "frame": fr["frame"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-subjects", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", default="baseline,word_boost,plan")
    args = ap.parse_args(argv)

    eng, label = build_engine("persisted")
    if eng is None:
        return 1
    subs = discover_subjects(eng)
    if args.max_subjects:
        subs = subs[:args.max_subjects]
    arms = [a.strip() for a in args.arms.split(",")]
    print(f"[measure] engine={label} subjects={len(subs)} arms={arms}")

    rows = []
    for i, subj in enumerate(subs):
        ctx = build_ctx(eng, subj, f"what is {subj}")
        row = {"subject": subj}
        for arm in arms:
            try:
                s = _score(eng, subj, _run_arm(eng, ctx, arm))
            except Exception as e:
                s = {"error": f"{type(e).__name__}: {e}"}
            row[arm] = s
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(subs)}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        print(f"[measure] wrote {args.out}")

    # Aggregate.
    def _mean(arm, key):
        vals = [r[arm][key] for r in rows
                if r.get(arm) and key in r[arm] and isinstance(r[arm][key], (int, float))]
        return (sum(vals) / len(vals)) if vals else 0.0

    def _rate(arm, key):
        vals = [r[arm] for r in rows if r.get(arm) and key in r[arm]]
        return (sum(1 for v in vals if v[key]) / len(vals)) if vals else 0.0

    print("\n" + "=" * 78)
    print(f"{'arm':<12} {'n':>4} {'on_topic':>9} {'unsup':>7} {'func':>6} "
          f"{'verb':>6} {'frame':>7}")
    print("-" * 78)
    base = None
    for arm in arms:
        n = sum(1 for r in rows if r.get(arm) and "on_topic" in r[arm])
        ot, un = _mean(arm, "on_topic"), _mean(arm, "unsupported")
        fs, vb, fr = _mean(arm, "func_share"), _rate(arm, "has_verb"), _mean(arm, "frame")
        print(f"{arm:<12} {n:>4} {ot:>9.4f} {un:>7.3f} {fs:>6.3f} "
              f"{vb:>6.2f} {fr:>7.3f}")
        if arm == "baseline":
            base = (ot, un, fs, vb, fr)
    if base:
        print("-" * 78)
        print(f"{'delta vs baseline':<12}")
        for arm in arms:
            if arm == "baseline":
                continue
            ot, un = _mean(arm, "on_topic"), _mean(arm, "unsupported")
            fs, vb, fr = _mean(arm, "func_share"), _rate(arm, "has_verb"), _mean(arm, "frame")
            print(f"{arm:<12} {'':>4} {ot-base[0]:>+9.4f} {un-base[1]:>+7.3f} "
                  f"{fs-base[2]:>+6.3f} {vb-base[3]:>+6.2f} {fr-base[4]:>+7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
