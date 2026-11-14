#!/usr/bin/env python3
"""Flag-on measurement restricted to subjects the plan route can REACH.

The first run (`measure_plan_flag_on.py`) showed 0/40 changed, because every
corpus subject has a stored definition and `consult_internal` answers from it
before the situation-model path is ever consulted. That is the correct
production behaviour -- the plan route is a fallback, not a replacement.

So the decision to enable the flag rests on the subjects where the fallback IS
reached: those with NO stored definition. This measures the paired OFF/ON
difference on exactly that set.

READ-ONLY. Usage:
    uv run python scripts/measure_plan_flag_on_reachable.py --limit 60
"""
import argparse
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CORPUS = os.path.join(PROJ, "data", "corpora", "grounded_descriptions.txt")


def load_subjects(path, limit=0):
    subs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "\t" not in line:
                continue
            c = line.split("\t", 1)[0].strip()
            if c:
                subs.append(c)
            if limit and len(subs) >= limit:
                break
    return subs


def run(eng, subs):
    out = {}
    for s in subs:
        try:
            out[s] = (eng.process_turn(f"what is {s}") or "").strip()
        except Exception:
            out[s] = ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    os.environ.pop("RAVANA_OFFLINE", None)
    from ravana.chat.engine import CognitiveChatEngine

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    defs = getattr(eng, "_definitions", {}) or {}
    subs = [s for s in load_subjects(args.corpus, args.limit)
            if s.lower() not in defs]
    print(f"[reach] corpus subjects without a stored definition: {len(subs)}")
    if not subs:
        return 0

    eng._web_direct_answer = lambda ctx: None
    os.environ["RAVANA_OFFLINE"] = "1"
    off = run(eng, subs)
    eng._plan_narrative_enabled = True
    on = run(eng, subs)

    same = sum(1 for s in subs if off[s] == on[s])
    changed = [s for s in subs if off[s] != on[s]]
    print(f"[reach] identical: {same}   changed: {len(changed)}")
    print()
    for s in changed[:12]:
        print(f"  {s}")
        print(f"    OFF: {off[s][:105]}")
        print(f"    ON : {on[s][:105]}")
    if not changed:
        print("  (no subject's output changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
