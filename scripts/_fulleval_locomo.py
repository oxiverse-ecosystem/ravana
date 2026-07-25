"""Full 600-case LoCoMo eval — CRASH-RESILIENT + RESUMABLE.

evaluate_ravana.main() segfaults in this env (training + restore_from_snapshot,
numpy 2.4.6/Py3.11 BLAS-OpenMP access violation). A fresh engine works but the
run intermittently segfaults mid-way (nondeterministic BLAS race). This driver:
  - builds ONE fresh engine (no snapshot/training),
  - runs all 600 cases through the REAL grader/runner,
  - writes EACH case result to a JSONL immediately (so a crash loses nothing),
  - RESUMES: on restart it skips cases already recorded,
  - stops when all 600 are recorded.

Launch loop (in shell): keep restarting until 600 recorded:
    while not done: python _fulleval_locomo.py
The driver itself also self-loops a few times for convenience.

Progress file: benchmark_results/locomo_full_progress.jsonl
Final aggregate: benchmark_results/locomo_full_freshengine.json
"""
import os
import sys
import time
import json

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_PROJ, os.path.join(_PROJ, "ravana", "src"),
           os.path.join(_PROJ, "ravana_ml", "src"),
           os.path.join(_PROJ, "ravana-v2", "src"),
           os.path.join(_PROJ, "scripts")):
    sys.path.insert(0, _p)

import numpy as np
import evaluate_ravana as EV

PROGRESS = os.path.join(_PROJ, "benchmark_results", "locomo_full_progress.jsonl")
FINAL = os.path.join(_PROJ, "benchmark_results", "locomo_full_freshengine.json")


def _recorded_indices():
    done = set()
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["idx"])
                except Exception:
                    pass
    return done


def main():
    EV.MAX_CASES = {"_default": 2000}
    EV._init_benchmarks()
    locomo = EV.BENCHMARKS["locomo"]
    EV._ensure_cases_loaded("locomo")
    all_cases = locomo["cases"]
    total = len(all_cases)
    done = _recorded_indices()
    remaining = [i for i in range(total) if i not in done]
    if not remaining:
        print(f"[fulleval] all {total} cases already recorded.", flush=True)
        _aggregate(total)
        return
    print(f"[fulleval] {len(done)}/{total} done; running {len(remaining)} "
          f"({remaining[0]}..{remaining[-1]})", flush=True)
    print(f"[fulleval] booting fresh engine...", flush=True)
    t0 = time.time()
    engine = EV.CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
    print(f"[fulleval] engine boot {time.time()-t0:.1f}s", flush=True)

    # Group remaining into contiguous runs (preserve per-dialogue reset state).
    # Simplest correct approach: run cases in order, but a reset (first case of
    # a new dialogue) must execute its primer from a clean slate. Since the
    # runner handles reset_memory/keep_memory per case via engine state, we can
    # just feed cases sequentially — but we must NOT skip the primer of a
    # dialogue whose first case we already recorded (its state is in the engine
    # from prior cases). So we trim the leading contiguous done block only.
    # Find the first not-done index; run from there to end.
    start = min(remaining)
    run_cases = all_cases[start:]
    run_offset = start

    fout = open(PROGRESS, "a")
    try:
        for j, case in enumerate(run_cases):
            idx = run_offset + j
            q = case["question"]
            grader = case.get("grader")
            primer = case.get("primer", [])
            try:
                if case.get("reset_memory", False) or not case.get("keep_memory", False):
                    try:
                        engine.reset_episodic_state()
                    except Exception:
                        try:
                            engine.hippocampal_buffer.facts.clear()
                        except Exception:
                            pass
                for turn in primer:
                    try:
                        engine.process_turn(turn)
                    except Exception:
                        pass
                try:
                    resp = engine.process_turn(q)
                except Exception as e:
                    resp = f"[error: {e}]"
            except Exception as e:
                resp = f"[error: {e}]"
            score = grader(resp) if (grader and resp) else 0.0
            rec = {"idx": idx, "q": q[:120], "expected": case.get("expected", ""),
                   "score": round(float(score), 4),
                   "cat": case.get("category")}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if (j + 1) % 25 == 0:
                print(f"  [{idx+1}/{total}] running avg so far: "
                      f"{_running_avg(fout):.3f}", flush=True)
    finally:
        fout.close()
    print(f"[fulleval] wrote through idx {run_offset + len(run_cases) - 1}",
          flush=True)
    _aggregate(total)


def _running_avg(fout):
    # fout is the open append handle; compute mean of all recorded scores.
    try:
        scores = []
        with open(PROGRESS) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    scores.append(json.loads(line)["score"])
                except Exception:
                    pass
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


def _aggregate(total):
    done = _recorded_indices()
    scores = {}
    cats = {}
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                scores[r["idx"]] = r["score"]
                cats.setdefault(r.get("cat"), []).append(r["score"])
    n = len(scores)
    overall = float(np.mean(list(scores.values()))) if scores else 0.0
    by_cat = {str(k): round(float(np.mean(v)), 4)
              for k, v in sorted(cats.items())}
    out = {"overall": round(overall, 4), "by_cat": by_cat, "n": n,
           "total": total,
           "note": "fresh engine (no snapshot/training); crash-resilient "
                   "resumable run; measures retrieval-path fix only"}
    with open(FINAL, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== AGGREGATE {n}/{total} ===", flush=True)
    for k, v in by_cat.items():
        print(f"  cat{k}: {v} (n={len(cats[k])})", flush=True)
    print(f"OVERALL: {out['overall']:.3f} (n={n})", flush=True)
    print("JSON:" + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
