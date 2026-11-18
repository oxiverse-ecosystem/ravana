#!/usr/bin/env python3
"""Replicate PR #364's plan-route claim at n>=40 held-out subjects.

Single cold engine boot, RAVANA_OFFLINE=1, fixed seed, NO eng.save().

A/B, per subject, on the SAME situation-model entry point the original
instruments used (``_generate_with_situation_model``):

  FALLBACK  eng._plan_narrative_enabled = False
  PLAN      eng._plan_narrative_enabled = True

Both arms get a freshly built context (build_ctx resets the situation model).
Scored with the ORIGINAL functions from eval_decoder_groundedness.py and the
ORIGINAL frame score from prototype_linearization.py. Prints raw inputs and
outputs for a hand-label sample and writes a full JSON evidence file.

READ-ONLY: never calls eng.save().
Usage:
    RAVANA_OFFLINE=1 uv run python scripts/replicate_plan_route.py \
        --out docs/plan_route_replication_data.json
"""
import argparse
import json
import os
import random
import sys
import traceback

os.environ["RAVANA_OFFLINE"] = "1"

import numpy as np  # noqa: E402

random.seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "ravana", "src"))

from eval_decoder_groundedness import (  # noqa: E402
    build_engine, build_ctx, discover_subjects, score_on_topic,
    score_unsupported_assertions, score_fluency,
)
from prototype_linearization import _frame_score  # noqa: E402

SEED = 42


def _scores(eng, subj, text):
    """All three axes, using the original instruments, plus raw detail."""
    ot = score_on_topic(eng, subj, text)
    un = score_unsupported_assertions(eng, subj, text)
    fr = _frame_score(text)
    fl = score_fluency(text, subj)
    return {
        "unsupported_count": int(un.get("unsupported_count", 0)),
        "unsupported_pairs": [list(p) for p in un.get("unsupported", [])],
        "supported_pairs": [list(p) for p in un.get("supported", [])],
        "on_topic_frac": float(ot.get("on_topic_frac", 0.0)),
        "content_words": int(ot.get("content_words", 0)),
        "overlap_words": list(ot.get("overlap_words", [])),
        "mean_cosine": float(ot.get("mean_cosine", 0.0)),
        "func_share": float(fr["func_share"]),
        "has_verb": bool(fr["has_verb"]),
        "frame": float(fr["frame"]),
        "word_count": int(fl.get("word_count", 0)),
        "unique_ratio": float(fl.get("unique_ratio", 0.0)),
        "is_word_salad": bool(fl.get("is_word_salad", False)),
    }


def _run_arm(eng, subj, plan_flag):
    """Fresh context, set flag, run, restore flag. Returns (text, strategy)."""
    ctx = build_ctx(eng, subj, f"what is {subj}")
    prev = eng._plan_narrative_enabled
    eng._plan_narrative_enabled = plan_flag
    try:
        out = eng._generate_with_situation_model(ctx)
    finally:
        eng._plan_narrative_enabled = prev
    if not out:
        return "", ""
    return (out[0] or ""), (out[1] if len(out) > 1 else "")


def _plan_commit_text(eng, subj):
    """Does a committed SurfacePlan exist for this subject? Diagnostic only."""
    ctx = build_ctx(eng, subj, f"what is {subj}")
    try:
        return eng._plan_narrative_sentence(ctx) or ""
    except Exception as e:
        return f"<ERROR {type(e).__name__}: {e}>"


def _mean(vals):
    return (sum(vals) / len(vals)) if vals else 0.0


def _report_group(name, rows, key):
    """rows: list of dicts; key: 'fallback' or 'plan' sub-dict."""
    sel = [r for r in rows if r.get(key, {}).get("text")]
    if not sel:
        print(f"  {name:<34} n=0")
        return None
    un = [r[key]["scores"]["unsupported_count"] for r in sel]
    ot = [r[key]["scores"]["on_topic_frac"] for r in sel]
    fr = [r[key]["scores"]["frame"] for r in sel]
    verb = sum(1 for r in sel if r[key]["scores"]["has_verb"])
    salad = sum(1 for r in sel if r[key]["scores"]["is_word_salad"])
    print(f"  {name:<34} n={len(sel):>3}  "
          f"unsup={_mean(un):.4f}  on_topic={_mean(ot):.4f}  "
          f"frame={_mean(fr):.4f}  verb={verb}/{len(sel)}  "
          f"salad={salad}/{len(sel)}")
    return {"n": len(sel), "unsup": _mean(un), "on_topic": _mean(ot),
            "frame": _mean(fr), "verb": verb, "salad": salad}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/plan_route_replication_data.json")
    ap.add_argument("--max-subjects", type=int, default=0,
                    help="cap subjects (0 = all discovered); for debugging")
    ap.add_argument("--hand-label", type=int, default=10)
    args = ap.parse_args(argv)

    print("=" * 78)
    print("PLAN ROUTE REPLICATION  seed=42  RAVANA_OFFLINE=1  no save()")
    print("=" * 78)
    eng, label = build_engine("persisted")
    if eng is None:
        return 1
    print(f"[repl] engine={label}")
    print(f"[repl] default _plan_narrative_enabled={eng._plan_narrative_enabled}")

    subjects = discover_subjects(eng)
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
    print(f"[repl] n_subjects={len(subjects)}")
    print(f"[repl] subjects={subjects}")
    print()

    defs = getattr(eng, "_definitions", {}) or {}
    rows = []
    for i, subj in enumerate(subjects, 1):
        rec = {"subject": subj,
               "has_definition": subj.lower() in defs}
        try:
            # FALLBACK first, then PLAN; each with a fresh ctx.
            ot, ost = _run_arm(eng, subj, False)
            rec["fallback"] = {"text": ot, "strategy": ost,
                               "scores": _scores(eng, subj, ot) if ot else None}
            pt, pst = _run_arm(eng, subj, True)
            rec["plan"] = {"text": pt, "strategy": pst,
                           "scores": _scores(eng, subj, pt) if pt else None}
            rec["fired"] = (pst == "situation_model_plan")
            rec["plan_commit"] = _plan_commit_text(eng, subj)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["trace"] = traceback.format_exc()
        rows.append(rec)
        if (i % 10 == 0) or (i == len(subjects)):
            print(f"  ... {i}/{len(subjects)}")

    # ── aggregate ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("AGGREGATES")
    print("=" * 78)
    fired = [r for r in rows if r.get("fired")]
    not_fired = [r for r in rows if not r.get("fired")]
    commits = [r for r in rows
               if r.get("plan_commit") and
               not r["plan_commit"].startswith("<ERROR")]
    print(f"  total subjects                : {len(rows)}")
    print(f"  with stored definition        : "
          f"{sum(1 for r in rows if r.get('has_definition'))}")
    print(f"  plan commits (committed plan) : {len(commits)}")
    print(f"  plan route FIRED (on arm)     : {len(fired)}/{len(rows)}")

    print("\n  -- all subjects, paired A/B --")
    all_fb = _report_group("FALLBACK (flag OFF), all", rows, "fallback")
    all_pl = _report_group("PLAN (flag ON), all", rows, "plan")

    print("\n  -- where the plan route FIRED --")
    fired_fb = _report_group("FALLBACK on fired subset", fired, "fallback")
    fired_pl = _report_group("PLAN on fired subset", fired, "plan")

    print("\n  -- where the plan route did NOT fire --")
    _report_group("FALLBACK on non-fired subset", not_fired, "fallback")
    _report_group("PLAN on non-fired subset", not_fired, "plan")

    # The headline-style split: plan-fired subjects (on arm) vs the subjects
    # that fell back (off arm). This is how '0.087 vs 1.833 (21x)' is built.
    print("\n  -- headline-style precision split --")
    print("     plan arm on FIRED subjects vs fallback arm on NON-FIRED subjects")
    pf = _report_group("PLAN (fired subset)", fired, "plan")
    nf = _report_group("FALLBACK (non-fired subset)", not_fired, "fallback")
    if pf and nf and nf["unsup"] > 0:
        print(f"\n  >>> reduction factor (fallback_nonfired / plan_fired) = "
              f"{nf['unsup'] / pf['unsup']:.2f}x" if pf["unsup"] > 0
              else "\n  >>> plan_fired unsup = 0 (division undefined; infinite reduction)")
    if all_fb and all_pl and all_fb["unsup"] > 0:
        print(f"  >>> reduction factor (all fallback / all plan)      = "
              f"{all_fb['unsup'] / all_pl['unsup']:.2f}x" if all_pl["unsup"] > 0
              else "  >>> all-plan unsup = 0 (infinite reduction)")

    # First-40 arm table to compare with the original n=40 measurement.
    print("\n  -- first-40 arm table (original n=40 comparison) --")
    first40 = rows[:40]
    _report_group("FALLBACK (flag OFF), first 40", first40, "fallback")
    _report_group("PLAN (flag ON), first 40", first40, "plan")

    # ── hand-label sample: raw inputs/outputs, printed for validation ────
    print("\n" + "=" * 78)
    print("HAND-LABEL SAMPLE (raw inputs/outputs; validate metric vs property)")
    print("=" * 78)
    # Prefer a mix: fired subjects + a few non-fired, so both directions shown.
    sample = [r for r in rows if r.get("fired")][: args.hand_label]
    if len(sample) < args.hand_label:
        sample += [r for r in not_fired][: args.hand_label - len(sample)]
    for r in sample:
        subj = r["subject"]
        print(f"\n### {subj}  def={r.get('has_definition')}  fired={r.get('fired')}")
        for arm in ("fallback", "plan"):
            a = r.get(arm) or {}
            s = a.get("scores") or {}
            print(f"  [{arm:<8}] strategy={a.get('strategy','')!r} "
                  f"unsup={s.get('unsupported_count')} "
                  f"on_topic={s.get('on_topic_frac',0):.3f} "
                  f"frame={s.get('frame',0):.3f} verb={s.get('has_verb')}")
            print(f"       RAW: {a.get('text','')[:260]!r}")
            print(f"       unsup_pairs={s.get('unsupported_pairs')}")
            print(f"       overlap_words={s.get('overlap_words')}")
        print(f"  [plan_commit] {(r.get('plan_commit') or '')[:160]!r}")

    # ── persist full evidence ────────────────────────────────────────────
    out_path = args.out
    if out_path:
        out_path = os.path.abspath(out_path)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"seed": SEED, "engine": label, "subjects": subjects,
                       "rows": rows}, fh, indent=1)
        print(f"\n[repl] wrote evidence -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
