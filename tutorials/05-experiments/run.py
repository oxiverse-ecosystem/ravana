"""Tutorial 05: Run a tiny experiment-style harness.

Layer 2 of 3 added on top of the mini-system. Demonstrates the probe loop
pattern used by all experiments in the experiments/ directory.

Usage:
    python tutorials/05-experiments/run.py
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ravana", "src"))
sys.path.insert(0, os.path.join(ROOT, "ravana_ml", "src"))
sys.path.insert(0, os.path.join(ROOT, "ravana-v2", "src"))

from ravana.chat.engine import CognitiveChatEngine


def main() -> None:
    # 1. Create a clean engine (fresh state for reproducible measurements)
    engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
    print("=== Experiment-Style Harness Demo ===\n")

    # 2. Define structured probes (the independent variables)
    probes = ["what is trust", "what is betrayal", "what is loyalty", "what is memory"]
    print(f"  Probes: {len(probes)}")
    for p in probes:
        print(f"    - {p}")
    print()

    # 3. Run each probe and record graph statistics (the dependent variables)
    results = []
    for q in probes:
        engine.process_turn(q)
        results.append({
            "query": q,
            "nodes": len(engine.graph.nodes),
            "edges": len(engine.graph.edges),
            "sleep": engine.sleep_cycles_completed,
        })
        print(f"  [OK] {q:40s}  nodes={results[-1]['nodes']:4d}  "
              f"edges={results[-1]['edges']:4d}  sleep={results[-1]['sleep']}")

    # 4. Write results to JSON (same format as experiments/)
    path = os.path.join(ROOT, "experiment_results", "tutorial_persistence.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  [OK] Results written to {path}")

    # 5. Summary statistics
    print(f"\n  Summary:")
    print(f"    Total nodes: {results[-1]['nodes']}")
    print(f"    Total edges: {results[-1]['edges']}")
    print(f"    Sleep cycles: {results[-1]['sleep']}")


if __name__ == "__main__":
    main()
