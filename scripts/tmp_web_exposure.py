"""Phase 1.1-1.2 (section 6.4 plan): bounded web-learning exposure session.

Runs a FIXED list of factual queries through the engine's real
learn_from_web path (which already feeds OpenIE facts into the triplet
operator at web_learning.py:990) — no synthetic conditioning. Then probes
the operator's profiles to show whether the web path moves the learned
counters in the wild (not just in unit tests).

The engine is loaded from the warmed eval snapshot (which already has the
Phase 1.3 multi-predicate growth). After exposure we re-save the snapshot
so the A/B runs can use the grown profiles, and emit a profile-growth JSON.

Web is best-effort: if the local search engine (localhost:4000) is down,
learn_from_web falls back silently and adds nothing — Phase 1.3 already
provided the deterministic multi-predicate growth, so this script never
blocks on network.
"""
from __future__ import annotations
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ravana", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import evaluate_ravana as ev

# Fixed, factual, taxonomy/property-heavy queries (deterministic replay).
QUERIES = [
    "mammal characteristics",
    "bird migration facts",
    "metal properties conductivity",
    "liquid water boiling point",
    "plant photosynthesis process",
    "insect anatomy",
    "acid base reaction",
    "electric current defined",
    "volcano eruption causes",
    "virus structure",
    "planet gas giant composition",
    "enzyme function biology",
    "crystal lattice structure",
    "gravity force newton",
    "cell membrane function",
    "fiber optical cable",
    "solar panel photovoltaic",
    "ocean current causes",
    "earthquake fault line",
    "protein amino acid",
    "carbohydrate structure",
    "laser light amplification",
    "magnet field poles",
    "river delta formation",
    "forest ecosystem",
    "battery electric chemical",
    "glass transparent material",
    "sound wave propagation",
    "gene dna heredity",
    "neuron signal transmission",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=len(QUERIES),
                    help="how many of the fixed queries to run")
    ap.add_argument("--save", action="store_true",
                    help="re-save the (grown) snapshot and profile JSON")
    args = ap.parse_args()

    ev.SNAPSHOT_PATH = "data/ravana_eval_snapshot.pkl"
    engine = ev.restore_from_snapshot()
    op = engine.triplet_op
    m = op.memory

    def snapshot_profiles():
        out = {}
        for pn, p in m.profiles.items():
            ev_pos = p.transitivity_pos + p.transitivity_neg \
                + p.symmetry_n
            if ev_pos == 0 and not p.composition_counts:
                continue
            out[pn] = {
                "transitivity_score": round(p.transitivity_score, 3),
                "transitivity_pos": p.transitivity_pos,
                "transitivity_neg": p.transitivity_neg,
                "transitivity_lower": round(p.transitivity_lower(), 3),
                "symmetry_n": p.symmetry_n,
                "depth_n": p.depth_n,
                "composition_counts": {r: dict(b)
                                      for r, b in p.composition_counts.items()},
            }
        return out

    before = snapshot_profiles()
    print(f"[before] predicates with evidence: {len(before)}")

    n = min(args.queries, len(QUERIES))
    grew = {"web_concepts": 0, "errors": 0}
    for i, q in enumerate(QUERIES[:n], 1):
        try:
            summary, _ = engine.learn_from_web(q, max_results=3,
                                               train_decoder=False)
            grew["web_concepts"] += 1
        except Exception as e:
            grew["errors"] += 1
            if getattr(engine, "_trace_enabled", False):
                print(f"  [{i}/{n}] {q!r} error: {e}")
        if i % 10 == 0:
            print(f"  ran {i}/{n} queries")

    after = snapshot_profiles()
    print(f"[after]  predicates with evidence: {len(after)} "
          f"(web calls ok={grew['web_concepts']}, err={grew['errors']})")

    # Show deltas for shared predicates.
    for pn in sorted(set(before) | set(after)):
        b = before.get(pn)
        a = after.get(pn)
        if b and a:
            dpos = a["transitivity_pos"] - b["transitivity_pos"]
            dneg = a["transitivity_neg"] - b["transitivity_neg"]
            if dpos or dneg:
                print(f"  Δ{pn}: trans +{dpos}/-{dneg} "
                      f"(wilson {b['transitivity_lower']}->{a['transitivity_lower']})")
        elif a and not b:
            print(f"  NEW {pn}: trans +{a['transitivity_pos']}/"
                  f"-{a['transitivity_neg']} wilson={a['transitivity_lower']}")

    if args.save:
        engine.stop_background_learning()
        engine.save()
        import shutil
        shutil.copy2(engine._save_path, ev.SNAPSHOT_PATH)
        print(f"[saved] snapshot -> {ev.SNAPSHOT_PATH}")
        with open("reports/web_exposure_profiles.json", "w",
                  encoding="utf-8") as f:
            json.dump({"before": before, "after": after,
                       "web_calls": grew}, f, indent=2)
        print("[saved] reports/web_exposure_profiles.json")


if __name__ == "__main__":
    main()
