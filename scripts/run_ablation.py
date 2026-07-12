#!/usr/bin/env python3
"""
Ablation: off-topic-snippet rate for P5 paradox grounding, before vs after.

The task's L2 fix replaces a hardcoded single-query + "emit the first sentence
regardless" path with a learned, gated retrieval pipeline (data-derived query
reformulation -> Wikipedia-REST / web search -> _best_answer_snippet -> M4/M5
-> coherence gate -> pseudo-relevance feedback). This script measures the
OFF-TOPIC-SNIPPET RATE: the fraction of paradox queries whose grounding clause
fails the coherence bar (GloVe cosine < 0.15 vs the paradox topic).

Two pipelines are compared on the SAME paradox query set:
  - AFTER (current): the gated pipeline in _reflect_on_paradox.
  - BEFORE (baseline): a naive "hardcoded-ish" first-result-first-sentence
    emulator (single query, first snippet, first sentence, no gate) — the
    behaviour the fix replaced.

Lower off-topic rate = better. Reports per-query coherence and the aggregate
rate for each pipeline so the improvement is visible.

Usage:
    python scripts/run_ablation.py
"""
import os
import sys
import re

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))

from ravana.chat.engine import CognitiveChatEngine

# Paradox query set (the plan's P5 examples + common cases).
PARADOXES = [
    "how many angels can dance on the head of a pin",
    "can god create a rock he cannot lift",
    "is the statement i am lying true or false",
    "what happens when an unstoppable force meets an immovable object",
    "what is the sound of one hand clapping",
]

COHERENCE_THETA = 0.15


def _topic(e, q):
    return e._paradox_topic(q.lower().strip(" ?!."))


def _grounding_clause(reply):
    m = re.search(r"\(from what i've read:\s*(.+?)\)\s*$", reply)
    return m.group(1).strip() if m else ""


def _before_pipeline(e, q):
    """Emulate the OLD behaviour: one query, first result, first sentence,
    no coherence gate. Returns the snippet text (or '')."""
    topic = _topic(e, q)
    qry = f"{topic} paradox" if topic else q
    try:
        res = e.search_engine.search(qry, max_results=1)
        if not res:
            return ""
        snip = (res[0].get("content") or res[0].get("snippet")
                or res[0].get("extract") or "")
        snip = snip.strip()
        if len(snip) > 25:
            return re.split(r"(?<=[.!?])\s+", snip)[0].rstrip(".!?")
    except Exception:
        return ""
    return ""


def main():
    e = CognitiveChatEngine(baby_mode=True)
    print("=== P5 paradox-grounding ablation: off-topic-snippet rate ===")
    print(f"(coherence bar: GloVe cosine >= {COHERENCE_THETA} vs paradox topic)\n")
    print(f"{'query':52} | {'BEFORE coh':>10} | {'AFTER coh':>10} | grounded?")
    print("-" * 100)
    before_off = 0
    after_off = 0
    after_grounded = 0
    for q in PARADOXES:
        topic = _topic(e, q)
        # AFTER: real gated pipeline.
        reply = e._reflect_on_paradox(q)
        after_clause = _grounding_clause(reply)
        after_coh = e._definition_coherence_score(topic, after_clause) if after_clause else 0.0
        if after_clause:
            after_grounded += 1
            if after_coh < COHERENCE_THETA:
                after_off += 1
        # BEFORE: naive baseline.
        before_clause = _before_pipeline(e, q)
        before_coh = e._definition_coherence_score(topic, before_clause) if before_clause else 0.0
        before_off_flag = (before_clause != "" and before_coh < COHERENCE_THETA)
        if before_off_flag:
            before_off += 1
        grounded = "yes" if after_clause else "no (fail-closed)"
        print(f"{q[:52]:52} | {before_coh:10.3f} | {after_coh:10.3f} | {grounded}")

    n = len(PARADOXES)
    before_rate = (before_off / n) * 100
    after_rate = (after_off / max(1, after_grounded)) * 100 if after_grounded else 0.0
    print("-" * 100)
    print(f"BEFORE (naive first-sentence) off-topic rate : {before_rate:5.1f}%  ({before_off}/{n} queries grounded w/ incoherent snippet)")
    print(f"AFTER  (gated pipeline)        off-topic rate : {after_rate:5.1f}%  ({after_off}/{after_grounded} groundings incoherent)")
    print(f"AFTER grounded queries                         : {after_grounded}/{n}")
    print()
    print("VERDICT:", "IMPROVED" if after_rate <= before_rate else "REGRESSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
