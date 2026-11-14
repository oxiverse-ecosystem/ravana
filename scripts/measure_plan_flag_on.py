#!/usr/bin/env python3
"""Does enabling the plan route change what PRODUCTION emits?

`measure_production_output.py` measures `process_turn` with the flag OFF (the
production default). This runs the SAME corpus through the SAME entry point
with the flag ON, and reports the paired difference -- so the decision to
enable is made against the real dispatch path, not against
`_generate_with_situation_model` in isolation.

READ-ONLY. Usage:
    uv run python scripts/measure_plan_flag_on.py --limit 100
"""
import argparse
import collections
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
    """The corpus is TSV: '<subject>\\t<description>'."""
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
    """One pass over the corpus. Returns {subject: text}."""
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
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    os.environ.pop("RAVANA_OFFLINE", None)
    from ravana.chat.engine import CognitiveChatEngine

    subs = load_subjects(args.corpus, args.limit)

    # ── Arm 1: production default (flag OFF) ────────────────────────────
    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    eng._web_direct_answer = lambda ctx: None
    os.environ["RAVANA_OFFLINE"] = "1"
    off = run(eng, subs)
    del eng

    # ── Arm 2: plan route enabled ───────────────────────────────────────
    eng2 = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                               user_suffix="")
    eng2._web_direct_answer = lambda ctx: None
    eng2._plan_narrative_enabled = True
    on = run(eng2, subs)

    same = sum(1 for s in subs if off[s] == on[s])
    changed = [s for s in subs if off[s] != on[s]]
    print(f"[flag] subjects={len(subs)}")
    print(f"[flag] identical output : {same}")
    print(f"[flag] changed output   : {len(changed)}")
    print()
    if changed:
        print("[flag] changed samples (OFF -> ON):")
        for s in changed[:15]:
            print(f"  {s}")
            print(f"    OFF: {off[s][:100]}")
            print(f"    ON : {on[s][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
