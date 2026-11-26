#!/usr/bin/env python3
"""SCRATCH-ONLY A/B: memoize ConceptEdge._init_relation_vector.

No production file is edited. The candidate is installed as a monkeypatch of
the UNMODIFIED production class at runtime. Frozen patches block save/network.

Part 1: boot-free functional equivalence over every relation type and several
        dims (exact array equality, dtype, shape, writable, distinct objects).
Part 2: end-to-end A/B on the SAME three WFA turns used by PR #549's profile,
        two engines booted from the same store in one process, same seeds.
        Compare responses, grounding method, node/edge census, relation-vector
        bytes, belief assertions and activation scores; report wall times.
"""
import hashlib
import json
import os
import random
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAIN = r"C:\Users\Likhith\Documents\projects\ravana"
for _p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "ravana", "src"), ROOT):
    sys.path.insert(0, _p)
os.environ["RAVANA_OFFLINE"] = "1"

import measure_fronts_web_frozen as mfwf  # noqa: E402
mfwf.install_frozen_patches()
import measure_learn5_wfa as W  # noqa: E402
from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
from ravana_ml.graph import ConceptEdge  # noqa: E402

ORIGINAL = ConceptEdge.__dict__["_init_relation_vector"].__func__
RELATIONS = ["semantic", "causal", "temporal", "analogical", "contextual",
             "inferred", "negation", "antonym", "transitive", "physical_cause",
             "comparison", "pragmatic", "not_a_relation"]
DIMS = [0, 1, 2, 4, 8, 16, 32, 64]


def make_cached():
    """Memoize the deterministic template; return a fresh writable copy."""
    templates = {(rel, dim): ORIGINAL(rel, dim)
                 for rel in RELATIONS for dim in DIMS}

    def cached(relation_type, dim):
        key = (relation_type, dim)
        tpl = templates.get(key)
        if tpl is None:
            # Any unseen (relation_type, dim) must use the ORIGINAL semantics.
            tpl = templates[key] = ORIGINAL(relation_type, dim)
        CACHED.calls += 1
        return tpl.copy()

    return cached, templates


CACHED, TEMPLATES = make_cached()
CACHED.calls = 0


def functional_equivalence():
    failures = []
    checked = 0
    for rel in RELATIONS:
        for dim in DIMS:
            a = ORIGINAL(rel, dim)
            b = CACHED(rel, dim)
            checked += 1
            if not (np.array_equal(a, b) and a.dtype == b.dtype
                    and a.shape == b.shape and b.flags.writeable
                    and a is not b):
                failures.append((rel, dim))
    # Mutating one returned copy must not affect a sibling edge's vector.
    x = CACHED("causal", 16)
    y = CACHED("causal", 16)
    x[0] = 99.0
    isolation_ok = not np.array_equal(x, y) and y[0] != 99.0
    return {"pairs_checked": checked, "failures": failures,
            "copy_isolation_ok": isolation_ok}


def microbench(reps=3, n=4352):
    out = {}
    for name, fn in (("original", ORIGINAL), ("cached", CACHED)):
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for _i in range(n):
                v = fn("causal", 16)
                v[0] += 0.0  # touch the result like the caller does
            times.append(time.perf_counter() - t0)
        out[name] = {"median_s": round(statistics.median(times), 5),
                     "reps": reps, "calls": n}
    out["speedup"] = round(out["original"]["median_s"] /
                           max(out["cached"]["median_s"], 1e-9), 2)
    return out


def pick(kind, pool=None):
    for it in W.build_items():
        if it["kind"] == kind and (pool is None or it["pool"] == pool):
            return it
    raise RuntimeError("no WFA item for kind=%r pool=%r" % (kind, pool))


TURNS = [
    ("inclusion_short", W.endorsement_prompt(pick("gap", "primary"), "inclusion")),
    ("exclusion_long_context", W.endorsement_prompt(pick("gap", "sealed"), "exclusion")),
    ("forced_choice_provenance",
     W.forced_choice_prompt(pick("decoy", "primary"), "exclusion")),
]


def snapshot(eng):
    nodes = tuple(sorted((int(nid), str(n.label))
                         for nid, n in eng.graph.nodes.items()))
    edges = []
    for (s, t), e in sorted(eng.graph.edges.items()):
        md = {k: v for k, v in e.source_metadata.items()
              if "time" not in str(k).lower()}
        edges.append((
            int(s), int(t), round(float(e.weight), 8),
            round(float(e.confidence), 8), str(e.relation_type),
            str(e.edge_type), str(e.predicate_token_id),
            hashlib.sha256(np.asarray(e.relation_vector).tobytes()).hexdigest(),
            json.dumps(md, sort_keys=True, default=str),
        ))
    beliefs = tuple(tuple(str(x) for x in b)
                    for b in getattr(eng, "_belief_assertions", []))
    acts = eng._activate_from_input(TURNS[-1][1])  # returns List[int] of node ids
    activations = tuple(int(k) for k in acts)
    return {"nodes": nodes, "edges": tuple(edges), "beliefs": beliefs,
            "activations": activations}


def run_arm(label):
    random.seed(0)
    np.random.seed(0)
    if label == "candidate":
        ConceptEdge._init_relation_vector = staticmethod(CACHED)
        CACHED.calls = 0
    else:
        ConceptEdge._init_relation_vector = staticmethod(ORIGINAL)
    t_boot = time.time()
    eng = CognitiveChatEngine(data_dir=os.path.join(MAIN, "weights"),
                              user_suffix="")
    boot = time.time() - t_boot
    rows = []
    for name, prompt in TURNS:
        t0 = time.time()
        resp = eng.process_turn(prompt)
        rows.append({"name": name, "wall": round(time.time() - t0, 4),
                     "prompt_len": len(prompt),
                     "response": resp or "",
                     "grounding": getattr(eng, "_last_grounding_method", None)})
    return {"boot": round(boot, 3), "turns": rows, "state": snapshot(eng),
            "cached_vec_calls": getattr(CACHED, "calls", 0)}


def compare(a, b):
    diffs = []
    for key in ("nodes", "edges", "beliefs", "activations"):
        if a["state"][key] != b["state"][key]:
            av, bv = a["state"][key], b["state"][key]
            diffs.append({"field": key, "len_a": len(av), "len_b": len(bv),
                          "first_diff": next(((i, av[i], bv[i]) for i in
                                              range(min(len(av), len(bv)))
                                              if av[i] != bv[i]), None)})
    for ra, rb in zip(a["turns"], b["turns"]):
        for f in ("name", "prompt_len", "response", "grounding"):
            if ra[f] != rb[f]:
                diffs.append({"turn": ra["name"], "field": f,
                              "a": str(ra[f])[:400], "b": str(rb[f])[:400]})
    return diffs


def main():
    eq = functional_equivalence()
    bench = microbench()
    print("FUNCTIONAL: %s" % eq, flush=True)
    print("MICROBENCH: %s" % bench, flush=True)

    print("Running baseline arm...", flush=True)
    base = run_arm("baseline")
    print("  boot=%.3fs turns=%s" % (
        base["boot"], [(r["name"], r["wall"]) for r in base["turns"]]),
        flush=True)
    print("Running candidate arm...", flush=True)
    cand = run_arm("candidate")
    print("  boot=%.3fs cached_vec_calls=%d turns=%s" % (
        cand["boot"], cand["cached_vec_calls"],
        [(r["name"], r["wall"]) for r in cand["turns"]]), flush=True)
    diffs = compare(base, cand)
    print("IDENTICAL OUTPUTS: %s (diff count=%d)" % (not diffs, len(diffs)),
          flush=True)
    for d in diffs[:10]:
        print("  DIFF %s" % d, flush=True)

    report = {
        "functional": eq, "microbench": bench,
        "baseline": {"boot": base["boot"],
                     "turns": [{k: v for k, v in r.items() if k != "response"}
                               for r in base["turns"]],
                     "response_sha256": [hashlib.sha256(
                         r["response"].encode("utf-8")).hexdigest()
                         for r in base["turns"]]},
        "candidate": {"boot": cand["boot"],
                      "cached_vec_calls": cand["cached_vec_calls"],
                      "turns": [{k: v for k, v in r.items() if k != "response"}
                                for r in cand["turns"]],
                      "response_sha256": [hashlib.sha256(
                          r["response"].encode("utf-8")).hexdigest()
                          for r in cand["turns"]]},
        "state_counts": {k: len(v) for k, v in base["state"].items()},
        "diffs": diffs,
        "identical_outputs": not diffs,
    }
    with open(os.path.join(HERE, "ab_relation_vector_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("Wrote ab_relation_vector_results.json", flush=True)


if __name__ == "__main__":
    main()
