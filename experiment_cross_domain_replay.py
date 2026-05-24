"""
Cross-Domain Transfer with Sleep-Time Interleaved Replay

Compares two conditions:
  (A) Baseline: train Domain A -> train Domain B -> evaluate (no replay)
  (B) Replay:   train Domain A -> snapshot -> train Domain B with
                 sleep-time interleaved replay of Domain A -> evaluate

Measures:
  - Domain A retention after Domain B training
  - Domain B accuracy
  - Cross-domain transfer probes
  - Improvement in percentage points (replay vs baseline)

Usage:
    python experiment_cross_domain_replay.py             # full experiment
    python experiment_cross_domain_replay.py --n 3       # quick test
    python experiment_cross_domain_replay.py --skip-baselines
"""

import os
import sys
import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ravana_ml.nn.rlm import RLM
from ravana_ml.tokenizer import SimpleTokenizer

# Reuse domain definitions and helpers from existing experiment
from experiment_cross_domain import (
    build_domain_a_science,
    build_domain_b_social,
    encode_fact,
    evaluate_rlm,
    test_structural_transfer,
    measure_graph_overlap,
)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReplayConfig:
    n_train_repeats: int = 3
    seed: int = 42
    skip_baselines: bool = False

    # RLM architecture (same as experiment_cross_domain.py)
    embed_dim: int = 32
    concept_dim: int = 32
    n_hidden: int = 32
    n_layers: int = 3
    sleep_interval: int = 100

    # Replay parameters
    replay_n_samples: int = 20  # samples per sleep cycle replay


# ═══════════════════════════════════════════════════════════════════════════
# Training with Replay Buffering
# ═══════════════════════════════════════════════════════════════════════════

def train_rlm_with_buffer(model: RLM, facts: List[Tuple[str, str, str]],
                          tokenizer, n_repeats: int = 3,
                          buffer_domain: Optional[str] = None):
    """Train RLM on facts, optionally buffering all experiences into replay buffer.

    Args:
        model: RLM instance
        facts: list of (input_text, target_text, relation_type)
        tokenizer: SimpleTokenizer
        n_repeats: how many times to repeat the fact set
        buffer_domain: if provided, buffer experiences under this domain label

    Returns:
        (acc_history, errors)
    """
    acc_history = []
    errors = []

    for repeat in range(n_repeats):
        for input_text, target_text, rel_type in facts:
            input_ids, target_ids = encode_fact(tokenizer, input_text, target_text)
            err = model.learn(input_ids, target_ids)
            errors.append(err)
            acc_history.append(model.conceptual_accuracy)

            if buffer_domain is not None:
                model.buffer_experience(input_ids, target_ids, domain=buffer_domain)

    return acc_history, errors


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(config: ReplayConfig) -> Dict[str, Any]:
    """Run the cross-domain transfer experiment with and without interleaved replay."""

    print("=" * 70)
    print("  CROSS-DOMAIN TRANSFER: INTERLEAVED REPLAY EXPERIMENT")
    print("=" * 70)
    print()

    tokenizer = SimpleTokenizer()
    vocab_size = tokenizer.vocab_size

    domain_a = build_domain_a_science()
    domain_b = build_domain_b_social()

    print(f"Domain A (Science): {len(domain_a['train'])} train, {len(domain_a['test'])} test")
    print(f"Domain B (Social):  {len(domain_b['train'])} train, {len(domain_b['test'])} test")
    print(f"Sleep interval: {config.sleep_interval}")
    print(f"Replay samples per cycle: {config.replay_n_samples}")
    print()

    results = {
        "config": asdict(config),
        "baseline": {},
        "replay": {},
        "comparison": {},
    }

    # ─── Condition A: Baseline (no replay) ──────────────────────────────

    print("-" * 70)
    print("  CONDITION A: Baseline (no interleaved replay)")
    print("-" * 70)
    print()

    np.random.seed(config.seed)
    model_baseline = RLM(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        concept_dim=config.concept_dim,
        n_concepts=vocab_size,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        sleep_interval=config.sleep_interval,
        tokenizer=tokenizer,
    )

    # Phase 1: Train on Domain A
    print("[Baseline Phase 1] Training on Domain A (Science)...")
    t0 = time.time()
    acc_a_base, _ = train_rlm_with_buffer(
        model_baseline, domain_a["train"], tokenizer,
        n_repeats=config.n_train_repeats,
    )
    base_p1_time = time.time() - t0
    print(f"  Time: {base_p1_time:.1f}s ({len(acc_a_base)} steps)")

    base_post_a_on_a = evaluate_rlm(model_baseline, domain_a["test"], tokenizer)
    print(f"  Domain A test: top1={base_post_a_on_a['top1_accuracy']:.1%}, top10={base_post_a_on_a['top10_accuracy']:.1%}")

    # Phase 2: Train on Domain B (no replay)
    print("\n[Baseline Phase 2] Training on Domain B (Social)...")
    t0 = time.time()
    acc_b_base, _ = train_rlm_with_buffer(
        model_baseline, domain_b["train"], tokenizer,
        n_repeats=config.n_train_repeats,
    )
    base_p2_time = time.time() - t0
    print(f"  Time: {base_p2_time:.1f}s ({len(acc_b_base)} steps)")

    base_post_b_on_a = evaluate_rlm(model_baseline, domain_a["test"], tokenizer)
    base_post_b_on_b = evaluate_rlm(model_baseline, domain_b["test"], tokenizer)
    print(f"  Domain B test: top1={base_post_b_on_b['top1_accuracy']:.1%}, top10={base_post_b_on_b['top10_accuracy']:.1%}")
    print(f"  Domain A retention: top1={base_post_b_on_a['top1_accuracy']:.1%}, top10={base_post_b_on_a['top10_accuracy']:.1%}")

    # Sleep cycle
    print("\n[Baseline Phase 3] Sleep cycle...")
    model_baseline.sleep_cycle()
    base_sleep_a = evaluate_rlm(model_baseline, domain_a["test"], tokenizer)
    base_sleep_b = evaluate_rlm(model_baseline, domain_b["test"], tokenizer)
    print(f"  Domain A after sleep: top1={base_sleep_a['top1_accuracy']:.1%}, top10={base_sleep_a['top10_accuracy']:.1%}")
    print(f"  Domain B after sleep: top1={base_sleep_b['top1_accuracy']:.1%}, top10={base_sleep_b['top10_accuracy']:.1%}")

    # Cross-domain probes
    base_transfer = test_structural_transfer(model_baseline, tokenizer)
    print(f"\n  Cross-domain probes: top1={base_transfer['top1_accuracy']:.1%}, top10={base_transfer['top10_accuracy']:.1%}")

    results["baseline"] = {
        "post_a_on_a": base_post_a_on_a,
        "post_b_on_a": base_post_b_on_a,
        "post_b_on_b": base_post_b_on_b,
        "post_sleep_a": base_sleep_a,
        "post_sleep_b": base_sleep_b,
        "transfer_probes": base_transfer,
        "graph": measure_graph_overlap(model_baseline),
    }

    # ─── Condition B: With Interleaved Replay ───────────────────────────

    print("\n" + "-" * 70)
    print("  CONDITION B: Sleep-Time Interleaved Replay")
    print("-" * 70)
    print()

    np.random.seed(config.seed)
    model_replay = RLM(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        concept_dim=config.concept_dim,
        n_concepts=vocab_size,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        sleep_interval=config.sleep_interval,
        tokenizer=tokenizer,
    )

    # Phase 1: Train on Domain A, buffer all experiences
    print("[Replay Phase 1] Training on Domain A (Science)...")
    t0 = time.time()
    acc_a_replay, _ = train_rlm_with_buffer(
        model_replay, domain_a["train"], tokenizer,
        n_repeats=config.n_train_repeats,
        buffer_domain="science",
    )
    replay_p1_time = time.time() - t0
    print(f"  Time: {replay_p1_time:.1f}s ({len(acc_a_replay)} steps)")

    replay_post_a_on_a = evaluate_rlm(model_replay, domain_a["test"], tokenizer)
    print(f"  Domain A test: top1={replay_post_a_on_a['top1_accuracy']:.1%}, top10={replay_post_a_on_a['top10_accuracy']:.1%}")
    print(f"  Replay buffer size: {len(model_replay._replay_buffer)}")

    # Snapshot: freeze Domain A experiences into "science" domain memory
    print("\n[Replay Phase 1b] Snapshotting Domain A replay buffer...")
    model_replay.snapshot_replay_buffer("science")

    # Activate Domain A memories for interleaved replay during Domain B training
    print("[Replay Phase 1c] Activating Domain A memories for interleaved replay...")
    model_replay.activate_domain_memories("science")
    print(f"  Replay buffer now has {len(model_replay._replay_buffer)} old experiences")

    # Phase 2: Train on Domain B with interleaved replay active
    # During this training, the model's sleep_cycle() will call
    # _replay_old_memories() during SWS, replaying Domain A experiences
    print("\n[Replay Phase 2] Training on Domain B (Social) with interleaved replay...")
    t0 = time.time()
    acc_b_replay, _ = train_rlm_with_buffer(
        model_replay, domain_b["train"], tokenizer,
        n_repeats=config.n_train_repeats,
        buffer_domain="social",
    )
    replay_p2_time = time.time() - t0
    print(f"  Time: {replay_p2_time:.1f}s ({len(acc_b_replay)} steps)")

    replay_post_b_on_a = evaluate_rlm(model_replay, domain_a["test"], tokenizer)
    replay_post_b_on_b = evaluate_rlm(model_replay, domain_b["test"], tokenizer)
    print(f"  Domain B test: top1={replay_post_b_on_b['top1_accuracy']:.1%}, top10={replay_post_b_on_b['top10_accuracy']:.1%}")
    print(f"  Domain A retention: top1={replay_post_b_on_a['top1_accuracy']:.1%}, top10={replay_post_b_on_a['top10_accuracy']:.1%}")

    # Sleep cycle (with replay of Domain A)
    print("\n[Replay Phase 3] Sleep cycle (with Domain A replay)...")
    model_replay.sleep_cycle()
    replay_sleep_a = evaluate_rlm(model_replay, domain_a["test"], tokenizer)
    replay_sleep_b = evaluate_rlm(model_replay, domain_b["test"], tokenizer)
    print(f"  Domain A after sleep: top1={replay_sleep_a['top1_accuracy']:.1%}, top10={replay_sleep_a['top10_accuracy']:.1%}")
    print(f"  Domain B after sleep: top1={replay_sleep_b['top1_accuracy']:.1%}, top10={replay_sleep_b['top10_accuracy']:.1%}")

    # Cross-domain probes
    replay_transfer = test_structural_transfer(model_replay, tokenizer)
    print(f"\n  Cross-domain probes: top1={replay_transfer['top1_accuracy']:.1%}, top10={replay_transfer['top10_accuracy']:.1%}")

    results["replay"] = {
        "post_a_on_a": replay_post_a_on_a,
        "post_b_on_a": replay_post_b_on_a,
        "post_b_on_b": replay_post_b_on_b,
        "post_sleep_a": replay_sleep_a,
        "post_sleep_b": replay_sleep_b,
        "transfer_probes": replay_transfer,
        "graph": measure_graph_overlap(model_replay),
    }

    # ─── Comparison ─────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  COMPARISON: INTERLEAVED REPLAY vs BASELINE")
    print("=" * 70)
    print()

    # Domain A retention: how well Domain A is remembered after Domain B training
    # Use top-10 accuracy for retention (matches existing experiment conventions)
    base_retention = base_post_b_on_a["top10_accuracy"]
    replay_retention = replay_post_b_on_a["top10_accuracy"]
    retention_improvement = replay_retention - base_retention

    # Domain A retention after sleep
    base_retention_sleep = base_sleep_a["top10_accuracy"]
    replay_retention_sleep = replay_sleep_a["top10_accuracy"]
    retention_sleep_improvement = replay_retention_sleep - base_retention_sleep

    # Domain B accuracy
    base_b_acc = base_post_b_on_b["top10_accuracy"]
    replay_b_acc = replay_post_b_on_b["top10_accuracy"]
    b_acc_improvement = replay_b_acc - base_b_acc

    # Domain B after sleep
    base_b_sleep = base_sleep_b["top10_accuracy"]
    replay_b_sleep = replay_sleep_b["top10_accuracy"]
    b_sleep_improvement = replay_b_sleep - base_b_sleep

    # Cross-domain probe accuracy
    base_probe = base_transfer["top10_accuracy"]
    replay_probe = replay_transfer["top10_accuracy"]
    probe_improvement = replay_probe - base_probe

    # Top-1 versions
    base_retention_t1 = base_post_b_on_a["top1_accuracy"]
    replay_retention_t1 = replay_post_b_on_a["top1_accuracy"]
    retention_t1_improvement = replay_retention_t1 - base_retention_t1

    comparison = {
        "domain_a_retention_top10": {
            "baseline": base_retention,
            "replay": replay_retention,
            "improvement_pp": retention_improvement * 100,
        },
        "domain_a_retention_after_sleep_top10": {
            "baseline": base_retention_sleep,
            "replay": replay_retention_sleep,
            "improvement_pp": retention_sleep_improvement * 100,
        },
        "domain_a_retention_top1": {
            "baseline": base_retention_t1,
            "replay": replay_retention_t1,
            "improvement_pp": retention_t1_improvement * 100,
        },
        "domain_b_accuracy_top10": {
            "baseline": base_b_acc,
            "replay": replay_b_acc,
            "improvement_pp": b_acc_improvement * 100,
        },
        "domain_b_after_sleep_top10": {
            "baseline": base_b_sleep,
            "replay": replay_b_sleep,
            "improvement_pp": b_sleep_improvement * 100,
        },
        "cross_domain_probes_top10": {
            "baseline": base_probe,
            "replay": replay_probe,
            "improvement_pp": probe_improvement * 100,
        },
    }
    results["comparison"] = comparison

    # Print comparison table
    print(f"  {'Metric':<40} {'Baseline':>10} {'Replay':>10} {'Improvement':>12}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*12}")
    print(f"  {'Domain A Retention (top10)':<40} {base_retention:>10.1%} {replay_retention:>10.1%} {retention_improvement:>+10.1%}  ({retention_improvement*100:+.1f}pp)")
    print(f"  {'Domain A Retention after Sleep (top10)':<40} {base_retention_sleep:>10.1%} {replay_retention_sleep:>10.1%} {retention_sleep_improvement:>+10.1%}  ({retention_sleep_improvement*100:+.1f}pp)")
    print(f"  {'Domain A Retention (top1)':<40} {base_retention_t1:>10.1%} {replay_retention_t1:>10.1%} {retention_t1_improvement:>+10.1%}  ({retention_t1_improvement*100:+.1f}pp)")
    print(f"  {'Domain B Accuracy (top10)':<40} {base_b_acc:>10.1%} {replay_b_acc:>10.1%} {b_acc_improvement:>+10.1%}  ({b_acc_improvement*100:+.1f}pp)")
    print(f"  {'Domain B after Sleep (top10)':<40} {base_b_sleep:>10.1%} {replay_b_sleep:>10.1%} {b_sleep_improvement:>+10.1%}  ({b_sleep_improvement*100:+.1f}pp)")
    print(f"  {'Cross-Domain Probes (top10)':<40} {base_probe:>10.1%} {replay_probe:>10.1%} {probe_improvement:>+10.1%}  ({probe_improvement*100:+.1f}pp)")

    # Print detailed probe results for replay model
    print("\n  Cross-domain probe details (with replay):")
    for probe in replay_transfer["probes"]:
        status = "OK" if probe["correct"] else ("~" if probe["in_top10"] else "X")
        print(f"    [{status}] '{probe['input'].strip()}' -> expected '{probe['expected']}'"
              f"  got '{probe['predicted']}'  ({probe['description']})")

    # Verdict
    print("\n" + "-" * 70)
    if retention_improvement > 0.05:
        print(f"  VERDICT: INTERLEAVED REPLAY IMPROVES RETENTION BY {retention_improvement*100:+.1f}pp")
        print(f"  Sleep-time replay significantly reduces catastrophic forgetting.")
    elif retention_improvement > 0.0:
        print(f"  VERDICT: INTERLEAVED REPLAY HELPS SLIGHTLY ({retention_improvement*100:+.1f}pp)")
        print(f"  Modest improvement in Domain A retention.")
    else:
        print(f"  VERDICT: NO CLEAR BENEFIT FROM INTERLEAVED REPLAY ({retention_improvement*100:+.1f}pp)")
        print(f"  Interleaved replay did not improve Domain A retention.")

    if retention_sleep_improvement > 0.10:
        print(f"\n  Notable: After sleep, retention gap widens to {retention_sleep_improvement*100:+.1f}pp")
        print(f"  Sleep consolidation amplifies the replay benefit.")
    print("-" * 70)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Domain Transfer with Interleaved Replay")
    parser.add_argument("--n", type=int, default=3, help="Repeats of each fact during training")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-samples", type=int, default=20, help="Samples per sleep cycle replay")
    parser.add_argument("--sleep-interval", type=int, default=100, help="Steps between sleep cycles")
    parser.add_argument("--output", type=str, default="experiment_results/cross_domain_replay.json")
    args = parser.parse_args()

    config = ReplayConfig(
        n_train_repeats=args.n,
        seed=args.seed,
        skip_baselines=args.skip_baselines,
        replay_n_samples=args.replay_samples,
        sleep_interval=args.sleep_interval,
    )

    results = run_experiment(config)

    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to {args.output}")
