#!/usr/bin/env python3
"""Flag-on measurement for the reachable set, CHUNKED so it fits a time budget.

`measure_plan_flag_on_reachable.py` boots the engine twice and runs every
reachable subject through both arms; over 179 subjects that exceeds the
process window. This does the same paired comparison but processes a
``--chunk`` of ``--chunk-size`` subjects per process invocation, appends each
chunk's result to a JSONL file, and resumes by skipping subjects already
recorded.

READ-ONLY. Usage (repeat until CHUNKS_EXHAUSTED):
    uv run python scripts/measure_plan_flag_on_chunked.py \
        --limit 512 --chunk-size 30 --out tmp/reach.jsonl
"""
import argparse
import json
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
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--chunk-size", type=int, default=30)
    ap.add_argument("--out", default=None, required=True)
    args = ap.parse_args()

    done = set()
    if args.out and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["subject"])
                    except Exception:
                        pass

    os.environ.pop("RAVANA_OFFLINE", None)
    from ravana.chat.engine import CognitiveChatEngine

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    defs = getattr(eng, "_definitions", {}) or {}
    reachable = [s for s in load_subjects(args.corpus, args.limit)
                 if s.lower() not in defs and s not in done]
    chunk = reachable[:args.chunk_size]
    print(f"[chunk] done={len(done)} remaining={len(reachable)} "
          f"this_chunk={len(chunk)}")
    if not chunk:
        print("[chunk] EXHAUSTED")
        return 0

    eng._web_direct_answer = lambda ctx: None
    os.environ["RAVANA_OFFLINE"] = "1"
    off = run(eng, chunk)
    eng._plan_narrative_enabled = True
    on = run(eng, chunk)

    with open(args.out, "a", encoding="utf-8") as fh:
        for s in chunk:
            fh.write(json.dumps({
                "subject": s, "off": off[s], "on": on[s],
                "changed": off[s] != on[s],
            }) + "\n")
    n_changed = sum(1 for s in chunk if off[s] != on[s])
    print(f"[chunk] wrote {len(chunk)} rows; changed={n_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
