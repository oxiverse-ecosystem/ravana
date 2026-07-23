#!/usr/bin/env python3
"""
LoCoMo + LongMemEval adapter harness for RAVANA.

These are external long-term-conversational-memory benchmarks. Both feed a
multi-session dialogue history, then ask a question whose answer must be
recalled from an earlier session. We:

  1. Feed each session's utterances into RAVANA process_turn IN ORDER
     (so the hippocampal/episodic buffer accumulates the history).
  2. Ask the benchmark question.
  3. Grade with a lenient RECALL metric: RAVANA is generative (not
     extractive), so we score a hit if the gold answer string (or its
     salient tokens) appears in RAVANA's response. We report both an
     exact-substring score and a token-overlap (F1-ish) score.

Gold data (downloaded via IntentForge search -> authors' releases):
  - data/benchmarks_external/locomo10.json          (snap-research/locomo)
  - data/benchmarks_external/longmemeval_oracle_50.jsonl (xiaowu0162/longmemeval-cleaned)

Usage:
  python scripts/eval_longmem.py --bench locomo   --limit 3
  python scripts/eval_longmem.py --bench longmem  --limit 20
  python scripts/eval_longmem.py --bench both     --limit 5
"""
import sys
import os
import re
import json
import time
import argparse

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))
sys.path.insert(0, os.path.join(_proj_root, "ravana_ml", "src"))

os.environ["RAVANA_SILENT"] = "1"

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or",
    "is", "was", "were", "are", "be", "been", "with", "as", "by", "it",
    "that", "this", "she", "he", "they", "her", "his", "their", "i",
    "did", "do", "does", "had", "has", "have", "what", "when", "where",
    "who", "why", "how", "which",
}


def _norm(s):
    return re.sub(r"[^a-z0-9' ]", " ", str(s).lower()).strip()


def _tokens(s):
    return [w for w in _norm(s).split() if w and w not in _STOP and len(w) > 1]


def grade(response, gold):
    """Return (exact_hit, token_f1). Lenient recall grading."""
    r = _norm(response)
    g = _norm(gold)
    if not g:
        return 0.0, 0.0
    exact = 1.0 if g in r else 0.0
    gt = set(_tokens(gold))
    rt = set(_tokens(response))
    if not gt:
        return exact, exact
    overlap = len(gt & rt)
    recall = overlap / len(gt)
    prec = overlap / len(rt) if rt else 0.0
    f1 = (2 * prec * recall / (prec + recall)) if (prec + recall) else 0.0
    return exact, f1


def make_engine(suffix):
    from scripts.ravana_chat import CognitiveChatEngine
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                            user_suffix=suffix)
    try:
        e.stop_background_learning()
    except Exception:
        pass
    return e


def feed_and_ask(engine, utterances, question):
    """Feed all history utterances, then ask the question."""
    for u in utterances:
        if not u:
            continue
        try:
            engine.process_turn(u)
        except Exception:
            pass
    try:
        return engine.process_turn(question) or ""
    except Exception as e:
        return f"<error: {e}>"


def run_locomo(path, limit, max_q=None):
    data = json.load(open(path, encoding="utf-8"))
    results = []
    for si, sample in enumerate(data[:limit]):
        conv = sample["conversation"]
        # Collect utterances across sessions in order.
        utter = []
        n = 1
        while f"session_{n}" in conv:
            for turn in conv[f"session_{n}"]:
                spk = turn.get("speaker", "")
                txt = turn.get("text", "")
                if txt:
                    utter.append(f"{spk}: {txt}")
            n += 1
        # Build memory ONCE per dialogue, then ask all questions against it
        # (this is how LoCoMo is scored: one accumulated history, many Qs).
        engine = make_engine(f"_locomo_{si}")
        for u in utter:
            try:
                engine.process_turn(u)
            except Exception:
                pass
        qa_list = sample["qa"]
        if max_q:
            qa_list = qa_list[:max_q]
        for qa in qa_list:
            q = qa.get("question")
            gold = qa.get("answer")
            if gold is None or q is None:
                continue  # some adversarial cat-5 have no answer
            try:
                resp = engine.process_turn(q) or ""
            except Exception as e:
                resp = f"<error: {e}>"
            ex, f1 = grade(resp, gold)
            results.append({"cat": qa.get("category"), "exact": ex,
                            "f1": f1, "q": q, "gold": gold,
                            "resp": resp[:120]})
    return results


def run_longmem(path, limit):
    results = []
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    for i, ex in enumerate(rows[:limit]):
        q = ex.get("question")
        gold = ex.get("answer")
        utter = []
        for sess in ex.get("haystack_sessions", []):
            for turn in sess:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if content:
                    utter.append(f"{role}: {content}")
        engine = make_engine(f"_lme_{i}")
        resp = feed_and_ask(engine, utter, q)
        e_, f1 = grade(resp, gold)
        results.append({"type": ex.get("question_type"), "exact": e_,
                        "f1": f1, "q": q, "gold": gold, "resp": resp[:120]})
    return results


def summarize(name, results):
    if not results:
        print(f"{name}: no results")
        return
    n = len(results)
    ex = sum(r["exact"] for r in results) / n
    f1 = sum(r["f1"] for r in results) / n
    print(f"\n{'='*66}")
    print(f"  {name}: {n} questions")
    print(f"  Exact-substring recall : {ex:.3f}")
    print(f"  Token-overlap F1       : {f1:.3f}")
    # per-category
    from collections import defaultdict
    byc = defaultdict(list)
    key = "cat" if "cat" in results[0] else "type"
    for r in results:
        byc[r.get(key)].append(r)
    print(f"  By {key}:")
    for c, rs in sorted(byc.items(), key=lambda x: str(x[0])):
        cf1 = sum(x["f1"] for x in rs) / len(rs)
        cex = sum(x["exact"] for x in rs) / len(rs)
        print(f"    {str(c):22s} n={len(rs):3d}  exact={cex:.3f}  f1={cf1:.3f}")
    print(f"{'='*66}")
    # a few examples
    print("  Examples:")
    for r in results[:4]:
        print(f"    Q: {r['q'][:60]}")
        print(f"      gold: {str(r['gold'])[:50]}  | resp: {r['resp'][:60]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["locomo", "longmem", "both"],
                    default="both")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--max_q", type=int, default=0,
                    help="cap questions per LoCoMo dialogue (0=all)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base = os.path.join(_proj_root, "data", "benchmarks_external")
    allres = {}
    t0 = time.time()
    if args.bench in ("locomo", "both"):
        p = os.path.join(base, "locomo10.json")
        print(f"Running LoCoMo (limit={args.limit} dialogues)...")
        r = run_locomo(p, args.limit, max_q=(args.max_q or None))
        summarize("LoCoMo", r)
        allres["locomo"] = r
    if args.bench in ("longmem", "both"):
        p = os.path.join(base, "longmemeval_oracle_50.jsonl")
        print(f"Running LongMemEval (limit={args.limit} questions)...")
        r = run_longmem(p, args.limit)
        summarize("LongMemEval", r)
        allres["longmem"] = r
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    if args.out:
        json.dump(allres, open(args.out, "w"), indent=2)
        print(f"Saved raw results -> {args.out}")


if __name__ == "__main__":
    main()
