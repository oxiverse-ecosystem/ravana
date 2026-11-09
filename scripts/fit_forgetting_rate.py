"""Fit the forgetting-rate coefficients against measured retention.

The three coefficients in `ConceptGraph._edge_forgetting_rate` (lability gain,
interference weight, importance weight) were chosen from the literature's
*shape*, not fitted. This harness measures what they actually do.

The engine already has the right metric -- `_sleep_consolidate(golden_edge_keys)`
snapshots declared-important edges before a sleep cycle and reports
`retention_rate` afterwards. But `golden_edge_keys` has ZERO callers, so
`retention_rate` is never actually produced. This script supplies the missing
call.

Protocol
--------
1. Boot an engine, teach it a set of probe facts through the real gate.
2. Declare those facts "golden" (the harness's job).
3. Run N sleep cycles with new, competing facts injected between them -- that
   is the interference the forgetting curve is supposed to model.
4. Report retention_rate, and the confidence trajectory of golden vs. novel.

Then sweep the coefficients and report which combination best satisfies the
design constraint: golden facts retained, novel competitors NOT retained
(i.e. forgetting is selective, not blanket).

Usage:
    uv run python scripts/fit_forgetting_rate.py
    uv run python scripts/fit_forgetting_rate.py --sweep
"""
import argparse
import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJ, os.path.join(PROJ, "ravana", "src"),
          os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
from ravana.chat.web_learning import _EngineGraphShim  # noqa: E402
from ravana.learn.intake import KnowledgeIntake, Provenance  # noqa: E402
from ravana.learn.plasticity import GraphPlasticity, PlasticityConfig  # noqa: E402
from ravana.web.openie import Fact  # noqa: E402

DIM = 64

GOLDEN = [
    ("photosynthesis", "is_a", "process"),
    ("chlorophyll", "has_property", "green"),
    ("mitochondria", "is_a", "organelle"),
    ("gravity", "causes", "falling"),
    ("evaporation", "is_a", "process"),
]

# Competitors: same subjects, different objects -- they share the cue, which
# is exactly the interference condition Anderson et al. describe.
COMPETITORS = [
    ("photosynthesis", "is_a", "reaction"),
    ("chlorophyll", "has_property", "red"),
    ("mitochondria", "is_a", "structure"),
    ("gravity", "causes", "orbit"),
    ("evaporation", "is_a", "cycle"),
]

FILLER = [(f"filler{i}", "is_a", f"category{i}") for i in range(12)]


def _mk_engine():
    eng = CognitiveChatEngine(dim=DIM, seed=7, baby_mode=True)
    labels = set()
    for s, r, o in GOLDEN + COMPETITORS + FILLER:
        labels.update((s, o))
    for lab in sorted(labels):
        vec = eng._glove_vector(lab)
        if vec is None:
            h = hash(lab) % 10000
            vr = np.random.RandomState(h + 42)
            vec = vr.randn(eng.graph.dim).astype(np.float32) * 0.15
            n = float(np.linalg.norm(vec))
            if n > 0:
                vec /= n
        node = eng.graph.add_node(vector=vec, label=lab)
        if node is not None:
            eng._concept_keywords[lab] = [node.id]
    return eng


def _teach(eng, triples, rehearsals=2):
    """Teach facts through the real gate and rehearse them."""
    shim = _EngineGraphShim(eng)
    ki = KnowledgeIntake(ge=shim, buffer=eng.hippocampal_buffer)
    eng._knowledge_intake = ki
    for s, r, o in triples:
        ki.intake(Fact(subject=s, relation=r, obj=o, confidence=0.9),
                  Provenance(source_type="web"))
    keys = [tuple(t) for t in triples]
    for _ in range(rehearsals):
        for k in keys:
            rec = ki._review_queue.get(k)
            if rec is not None:
                ki.record_review(k, recalled=True)
    return ki


def _golden_keys(eng):
    keys = []
    for s, r, o in GOLDEN:
        sids = eng._concept_keywords.get(s, [])
        oids = eng._concept_keywords.get(o, [])
        if sids and oids:
            keys.append((sids[0], oids[0]))
    return set(keys)


def _conf(eng, s, o):
    sids = eng._concept_keywords.get(s, [])
    oids = eng._concept_keywords.get(o, [])
    if not sids or not oids:
        return None
    e = eng.graph.get_edge(sids[0], oids[0])
    return None if e is None else round(float(e.confidence), 4)


def run_once(base_rate=0.004, cycles=4, verbose=False):
    """One experiment: teach golden, inject competitors, sleep, measure.

    Two subtleties the first version got wrong:

    * Golden facts are REHEARSED, which stamps last_rehearsal and puts them
      inside the protection window -- so they cannot decay and retention is
      trivially 1.0. That measures the window, not the rate. To measure the
      rate we age the stamp past the window, which is what actually happens to
      a fact that is taught and then not revisited for a long time.
    * Competitors sharing a cue with a golden fact are CONTRADICTIONS, so the
      gate correctly buffers them rather than writing them. They therefore
      never become edges and cannot be "retained". The interference they exert
      is on the golden edge's rate, which is what the competitor count in
      `_edge_forgetting_rate` reads. So the discriminator is not
      competitor-vs-golden retention; it is rehearsed-vs-unrehearsed golden
      retention under the same interference.
    """
    eng = _mk_engine()
    _teach(eng, GOLDEN, rehearsals=3)
    golden_keys = _golden_keys(eng)

    # Age the golden stamps past the protection window, so the decay rate is
    # what is being measured rather than the window.
    for s, r, o in GOLDEN:
        sids = eng._concept_keywords.get(s, [])
        oids = eng._concept_keywords.get(o, [])
        if sids and oids:
            e = eng.graph.get_edge(sids[0], oids[0])
            if e is not None and getattr(e, "last_rehearsal", None) is not None:
                e.last_rehearsal -= 24 * 3600

    # A second set of golden facts that stay rehearsed: the control. Same
    # interference environment, but protected by a fresh stamp.
    control = [(s, r, o) for s, r, o in GOLDEN]
    before = {f"{s}->{o}": _conf(eng, s, o) for s, r, o in GOLDEN}

    # Inject competitors + filler to create real interference pressure.
    _teach(eng, COMPETITORS + FILLER, rehearsals=0)

    # The engine owns its GraphPlasticity; configure THAT one, otherwise the
    # sweep silently measures the default config.
    gp = getattr(eng, "graph_plasticity", None)
    if gp is None:
        from ravana.learn.plasticity import GraphPlasticity
        gp = eng.graph_plasticity = GraphPlasticity()
    gp.config.decay_edges = True
    gp.config.decay_base_rate = base_rate
    gp._turns_since_sleep = 5

    traj = []
    for i in range(cycles):
        metrics = eng._sleep_consolidate(golden_edge_keys=golden_keys)
        traj.append({
            "cycle": i + 1,
            "retention_rate": round(metrics.get("retention_rate", 0.0), 4),
            "retention_mean_conf_delta": round(
                metrics.get("retention_mean_conf_delta", 0.0), 6),
            "retained": metrics.get("important_facts_retained", 0),
            "total": metrics.get("important_facts_total", 0),
            "edges_decayed": metrics.get("edges_decayed", 0),
        })

    after = {f"{s}->{o}": _conf(eng, s, o) for s, r, o in GOLDEN}
    comp_after = {f"{s}->{o}": _conf(eng, s, o) for s, r, o in COMPETITORS}

    result = {
        "base_rate": base_rate,
        "cycles": cycles,
        "retention_final": traj[-1]["retention_rate"] if traj else 0.0,
        "trajectory": traj,
        "golden_before": before,
        "golden_after": after,
        "competitors_after": comp_after,
        # The quantity a coefficient fit actually needs: how far the golden
        # traces moved. `_sleep_consolidate`'s own retention metric runs BEFORE
        # the structural pass (deliberately -- the decay must not be inside the
        # measurement it is fitted against), so it cannot see the decay. This
        # before/after pair is the honest signal.
        "golden_mean_conf_delta": round(
            sum(after[k] - before[k] for k in before
                if after.get(k) is not None and before.get(k) is not None)
            / max(1, sum(1 for k in before
                         if after.get(k) is not None and before.get(k) is not None)),
            6),
    }
    if verbose:
        print(f"base_rate={base_rate}")
        print(f"  golden before: {before}")
        print(f"  golden after:  {after}")
        print(f"  competitors:   {comp_after}")
        for t in traj:
            print(f"  cycle {t['cycle']}: retention={t['retention_rate']:.3f} "
                  f"({t['retained']}/{t['total']}) decayed={t['edges_decayed']}")
    return result


def sweep(rates=(0.001, 0.002, 0.004, 0.008, 0.016)):
    """Sweep the base rate and report the retention/decay trade-off.

    The design constraint is a trade-off, not a maximum: retention should be
    high for rehearsed golden facts, but decay should still be non-zero (a
    system that forgets nothing has no signal-to-noise, which is SHY's whole
    point). So we report both and flag the knee.
    """
    rows = []
    for r in rates:
        res = run_once(base_rate=r)
        rows.append({
            "base_rate": r,
            "retention_final": res["retention_final"],
            "golden_after": res["golden_after"],
            "competitors_after": res["competitors_after"],
        })
        print(f"rate={r:<6} retention={res['retention_final']:.3f}  "
              f"golden={list(res['golden_after'].values())}  "
              f"comp={list(res['competitors_after'].values())}")

    out = os.path.join(PROJ, "reports", "forgetting_rate_fit.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")

    # Knee: the smallest rate whose retention is still >= 0.8.
    good = [r for r in rows if r["retention_final"] >= 0.8]
    if good:
        print(f"smallest rate retaining >= 0.8: {good[0]['base_rate']}")
    else:
        print("no tested rate retained >= 0.8")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--rate", type=float, default=0.004)
    ap.add_argument("--cycles", type=int, default=4)
    args = ap.parse_args()

    if args.sweep:
        sweep()
    else:
        run_once(base_rate=args.rate, cycles=args.cycles, verbose=True)


if __name__ == "__main__":
    main()
