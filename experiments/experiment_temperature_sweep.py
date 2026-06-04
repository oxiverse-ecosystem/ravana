"""
Temperature Sweep for Cross-Domain Transfer

Trains Domain A -> Domain B -> sleep, then sweeps softmax temperature
to show how eval-time sharpening recovers Top-1 accuracy.

Usage:
    python experiments/experiment_temperature_sweep.py
    python experiments/experiment_temperature_sweep.py --quick  # fewer repeats
"""

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
import json
import numpy as np
from ravana_ml.nn.rlm import RLM
from ravana_ml.tokenizer import SimpleTokenizer
from experiments.experiment_cross_domain import (
    build_domain_a_science,
    build_domain_b_social,
    train_rlm_on_domain,
    evaluate_rlm,
)


def run_temperature_sweep(n_repeats=3, seed=42):
    """Train cross-domain model, then sweep temperature."""
    print("=" * 70)
    print("  CROSS-DOMAIN TEMPERATURE SWEEP")
    print("=" * 70)

    tokenizer = SimpleTokenizer()
    vocab_size = tokenizer.vocab_size
    domain_a = build_domain_a_science()
    domain_b = build_domain_b_social()

    print(f"  Domain A: {len(domain_a['train'])} train facts")
    print(f"  Domain B: {len(domain_b['train'])} train facts")
    print(f"  Repeats: {n_repeats}")
    print()

    # Train cross-domain model
    np.random.seed(seed)
    model = RLM(
        vocab_size=vocab_size,
        embed_dim=32,
        concept_dim=32,
        n_concepts=vocab_size,
        n_hidden=32,
        n_layers=3,
        sleep_interval=100,
        tokenizer=tokenizer,
    )

    # Phase 1: Train Domain A
    print("[Phase 1] Training Domain A (Science)...")
    t0 = time.time()
    train_rlm_on_domain(model, domain_a["train"], tokenizer,
                        n_repeats=n_repeats, domain_tag="science",
                        buffer_for_replay=True)
    print(f"  Time: {time.time()-t0:.1f}s")

    # Phase 2: Train Domain B with interleaved replay
    model.activate_domain_memories("science")
    print("[Phase 2] Training Domain B (Social) with replay...")
    t0 = time.time()
    train_rlm_on_domain(model, domain_b["train"], tokenizer,
                        n_repeats=n_repeats, domain_tag="social",
                        buffer_for_replay=True)
    print(f"  Time: {time.time()-t0:.1f}s")

    # Phase 3: Sleep consolidation
    print("[Phase 3] Sleep cycle...")
    model.sleep_cycle()

    # Baseline evaluation (T=1.0)
    base_a = evaluate_rlm(model, domain_a["test"], tokenizer, temperature=1.0)
    base_b = evaluate_rlm(model, domain_b["test"], tokenizer, temperature=1.0)
    print(f"\n  Post-sleep baseline (T=1.0):")
    print(f"    Domain A: top1={base_a['top1_accuracy']:.1%}, top10={base_a['top10_accuracy']:.1%}")
    print(f"    Domain B: top1={base_b['top1_accuracy']:.1%}, top10={base_b['top10_accuracy']:.1%}")

    # ── Temperature sweep ──
    temperatures = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]

    print(f"\n  {'T':>5}  {'A Top-1':>8}  {'A Top-10':>9}  {'B Top-1':>8}  {'B Top-10':>9}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*9}")

    sweep_results = {}
    for T in temperatures:
        res_a = evaluate_rlm(model, domain_a["test"], tokenizer, temperature=T)
        res_b = evaluate_rlm(model, domain_b["test"], tokenizer, temperature=T)
        sweep_results[str(T)] = {
            "domain_a": res_a,
            "domain_b": res_b,
        }
        print(f"  {T:>5.2f}  {res_a['top1_accuracy']:>8.1%}  {res_a['top10_accuracy']:>9.1%}  "
              f"{res_b['top1_accuracy']:>8.1%}  {res_b['top10_accuracy']:>9.1%}")

    # Find best T for Domain B Top-1
    best_T = max(temperatures, key=lambda T: sweep_results[str(T)]['domain_b']['top1_accuracy'])
    best_val = sweep_results[str(best_T)]['domain_b']['top1_accuracy']
    print(f"\n  Best T for Domain B Top-1: {best_T} ({best_val:.1%})")

    # Improvement from T=1.0
    baseline_t1 = sweep_results['1.0']['domain_b']['top1_accuracy']
    improvement = best_val - baseline_t1
    if improvement > 0:
        print(f"  Improvement over T=1.0: {improvement*100:+.1f}pp")

    output = {
        "config": {"n_repeats": n_repeats, "seed": seed},
        "sweep": sweep_results,
        "best_T": best_T,
        "baseline_T1_domain_b_top1": float(baseline_t1),
        "improvement_pp": float(improvement * 100),
    }
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fewer repeats")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_repeats = 2 if args.quick else 3
    results = run_temperature_sweep(n_repeats=n_repeats, seed=args.seed)

    out_path = os.path.join(_PROJECT_ROOT, "revisions", "temperature_sweep_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
