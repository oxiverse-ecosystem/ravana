#!/usr/bin/env python3
"""TRA-K stage-2 BOOT-FREE read-surface probe. SCRATCH ONLY.

Question: with an IDENTICAL trusted typed-slot edge in the reader's view,
does the production `_typed_slot_candidate` hook fire on the graded P3 prompt
but not on the P4 immediate / P6 delayed paraphrase prompt?

No engine boot, no store write, no network, no production edit. The only
monkeypatch is the graph->view adapter inside THIS process: the hook itself,
the forced-choice parser and the fail-closed scorer are the UNMODIFIED
production functions. Stored views are supplied as an ORACLE UPPER BOUND
(non-deployable): the point is to give the reader a perfect trace and ask
whether the cue surface still fails. Deployable evidence comes from the live
rows already emitted by the parent A/B (read-only).

Run (venv python, from this worktree):
  PYTHONPATH=<wt>/scripts;<wt>/ravana/src;<wt>/ravana_ml/src \
  RAVANA_TRA_SLOTS=1 RAVANA_OFFLINE=1 TRAK_CHANNEL_JSON=<main>/scratch/trak_channel_from_substrate.json \
  C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/trak_stage2_cue_probe.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (os.path.join(REPO, "scripts"),
           os.path.join(REPO, "ravana", "src"),
           os.path.join(REPO, "ravana_ml", "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ["RAVANA_TRA_SLOTS"] = "1"
SOURCE_ROOT = os.environ.get(
    "TRAK_REPLAY_SOURCE_ROOT", os.path.join(os.path.dirname(REPO), "ravana"))
LIVE_DATA = os.environ.get(
    "TRAK_LIVE_DATA",
    os.path.join(os.path.dirname(REPO), "ravana_trak_live_ab", "scratch",
                 "trak_live_ab_data"))

import measure_tra_k as mtk                      # noqa: E402
from ravana.chat import response_gen as rg      # noqa: E402
from ravana.chat.schema_induction import (      # noqa: E402
    TypedSlotEdgeView, slot_keyed_forced_choice)


class _Host:
    """Duck-typed host: no engine, no graph, only the hook under test."""

    graph = None


def trusted_view(subject: str, label: str, confidence: float = 0.9):
    """The exact provenance shape the R1/R2 overlay earns at write time."""
    return TypedSlotEdgeView(
        source=subject.strip().lower(), relation_type="is_a",
        target=label.strip().lower(), type_label=label.strip().lower(),
        slot_label="type", edge_kind="curated", edge_source="conversation",
        source_type="user", confidence=confidence)


def canonical_cue(prompt: str) -> str:
    """Stage-2 candidate: reinstate the encoding-surface cue at read time.

    Rewrites ONLY the interrogative frame; the displayed option set is left
    byte-identical. Brain ground: encoding specificity / transfer-appropriate
    processing -- alter the cue, not the stored trace or the answer key.
    """
    head, sep, tail = prompt.partition("Options:")
    if not sep or "best fits" not in head.lower():
        return prompt
    new_head = ("Considering the examples above, what is true of "
                + head.split("best fits ", 1)[1].strip().rstrip("? "))
    return new_head + "? Options:" + tail


def run():
    rows = []
    for arm in ("OFF", "R1"):
        path = os.path.join(LIVE_DATA, "rows_%s.jsonl" % arm)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                rows += [json.loads(l) for l in fh if l.strip()]
    paired = {}
    for r in rows:
        if r.get("phase") != "delayed" or r.get("condition") not in ("near", "far"):
            continue
        key = (r["id"], r["condition"])
        paired.setdefault(key, r)          # OFF and R1 prompts are identical
    if not paired:
        raise SystemExit("no live delayed rows found under %s" % LIVE_DATA)

    original_adapter = rg.typed_slot_views_from_graph
    results = []
    try:
        for (item_id, condition), row in sorted(paired.items()):
            subject = (row.get("input") or {}).get("query_subject")
            options = list((row.get("input") or {}).get("candidates") or [])
            gold = str((row.get("gold") or {}).get("target") or "")
            if not subject or not options or not gold:
                continue
            # ORACLE UPPER BOUND: give the reader one trusted stored slot
            # matching the item's own gold label. This proves reachability of
            # the read path; it is NOT deployable evidence of learning.
            rg.typed_slot_views_from_graph = lambda _g, s=subject, l=gold: [
                trusted_view(s, l)]
            host = _Host()
            p3 = mtk.query_surface(
                {"item_id": item_id, "typed_queries": {},
                 "unseen_test_words": {condition: {
                     "subject": subject, "candidates": options, "gold": gold}}},
                condition, paraphrase=False)
            p6 = mtk.query_surface(
                {"item_id": item_id, "typed_queries": {},
                 "unseen_test_words": {condition: {
                     "subject": subject, "candidates": options, "gold": gold}}},
                condition, paraphrase=True)
            direct = rg.ResponseGenMixin._typed_slot_candidate(
                host, p3, subject)
            para_subject = rg.ResponseGenMixin._typed_slot_candidate(
                host, p6, subject)
            para_none = rg.ResponseGenMixin._typed_slot_candidate(
                host, p6, None)
            para_fixed = rg.ResponseGenMixin._typed_slot_candidate(
                host, canonical_cue(p6), None)
            scorer = slot_keyed_forced_choice(
                [trusted_view(subject, gold)], subject, options)
            results.append({
                "id": item_id, "condition": condition, "subject": subject,
                "gold": gold, "options": options,
                "p3_pick": direct, "p6_pick_with_subject": para_subject,
                "p6_pick_no_subject": para_none,
                "p6_canonical_cue_pick": para_fixed,
                "scorer_pick": scorer.get("pick"),
                "scorer_reason": scorer.get("reason"),
                "live_response": (row.get("engine_responses") or [None])[0],
                "live_verdict": (row.get("engine_answer") or {}).get("verdict"),
            })
    finally:
        rg.typed_slot_views_from_graph = original_adapter

    n = len(results)
    summary = {
        "n_paired_delayed_items": n,
        "p3_pick_eq_gold": sum(r["p3_pick"] == r["gold"] for r in results),
        "p6_pick_eq_gold_with_subject": sum(
            r["p6_pick_with_subject"] == r["gold"] for r in results),
        "p6_pick_eq_gold_no_subject": sum(
            r["p6_pick_no_subject"] == r["gold"] for r in results),
        "p6_canonical_cue_pick_eq_gold": sum(
            r["p6_canonical_cue_pick"] == r["gold"] for r in results),
        "live_correct": sum(bool(r["live_verdict"] == "match") for r in results),
    }
    out = os.path.join(HERE, "trak_stage2_cue_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": results}, fh, indent=2,
                  ensure_ascii=False)
    print(json.dumps({"summary": summary,
                      "rows": results[:4]}, indent=2, ensure_ascii=False))
    print("RESULT FILE:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
