"""
Dimension Scaling Experiment for RAVANA RLM

Tests whether the architecture generalizes to different embedding dimensions.
Runs the within-domain benchmark at embed_dim = {32, 64, 128}.

Usage:
    python experiments/experiment_dimension_scaling.py
    python experiments/experiment_dimension_scaling.py --quick  # fewer epochs
"""

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import time
import json
from ravana_ml.nn.rlm import RLM
from ravana_ml.tokenizer import SimpleTokenizer
from experiments.experiment_cross_domain import (
    build_domain_a_science,
    build_domain_b_social,
    encode_fact,
)


def evaluate_topk(model, facts, tokenizer, k=1):
    """Evaluate top-k accuracy on a set of facts."""
    correct = 0
    total = 0
    for input_text, target_text, rel_type in facts:
        input_ids, target_ids = encode_fact(tokenizer, input_text, target_text)
        if len(input_ids) == 0:
            continue
        logits = model.forward(input_ids[np.newaxis, :])
        data = logits.data if hasattr(logits, 'data') else np.array(logits)
        if data.ndim > 1:
            data = data[0]

        target_id = ord(target_text[0]) if target_text else 0
        ranked = np.argsort(data)[::-1]
        if target_id in set(ranked[:k]):
            correct += 1
        total += 1
    return correct / max(1, total)


def train_and_evaluate(tokenizer, facts, embed_dim, epochs=80, seed=42, deep_sleep_every=5):
    """Train and evaluate at a given embed_dim. Returns metrics."""
    np.random.seed(seed)
    concept_dim = embed_dim  # match concept dim to embed dim

    t0 = time.time()
    model = RLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        concept_dim=concept_dim,
        n_concepts=tokenizer.vocab_size,
        n_hidden=embed_dim,
        n_layers=3,
        sleep_interval=100,
        tokenizer=tokenizer,
        deep_sleep_every=deep_sleep_every,
    )
    init_time = time.time() - t0

    # Train
    errors = []
    t0 = time.time()
    for epoch in range(epochs):
        np.random.shuffle(facts)
        for input_text, target_text, rel_type in facts:
            input_ids, target_ids = encode_fact(tokenizer, input_text, target_text)
            err = model.learn(input_ids, target_ids)
            errors.append(err)
        if (epoch + 1) % 10 == 0:
            model.sleep_cycle()
    train_time = time.time() - t0

    # Evaluate
    top1 = evaluate_topk(model, facts, tokenizer, k=1)
    top10 = evaluate_topk(model, facts, tokenizer, k=10)

    # Graph stats
    n_nodes = len(model.graph.nodes)
    n_edges = len(model.graph.edges)

    return {
        "embed_dim": embed_dim,
        "top1": float(top1),
        "top10": float(top10),
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "train_time_s": round(train_time, 1),
        "init_time_s": round(init_time, 3),
        "mean_error": float(np.mean(errors[-100:])) if errors else 0.0,
        "param_count": sum(p.data.size for p in model.parameters()),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fewer epochs")
    parser.add_argument("--seeds", type=int, default=3, help="Random seeds per dim")
    parser.add_argument("--dims", type=str, default="32,64,128",
                        help="Comma-separated embed_dims")
    parser.add_argument("--fast-sleep", action="store_true",
                        help="Use light/deep sleep alternation for faster runs")
    args = parser.parse_args()

    epochs = 40 if args.quick else 80
    dims = [int(d) for d in args.dims.split(",")]
    n_seeds = args.seeds
    deep_sleep_every = 10 if args.fast_sleep else 5

    tokenizer = SimpleTokenizer()
    domain_a = build_domain_a_science()
    domain_b = build_domain_b_social()
    all_facts = domain_a["train"] + domain_b["train"]

    print("=" * 70)
    print("  RAVANA Dimension Scaling Experiment")
    print("=" * 70)
    print(f"  Vocab: {tokenizer.vocab_size} tokens")
    print(f"  Facts: {len(all_facts)}")
    print(f"  Dims: {dims}")
    print(f"  Epochs: {epochs}, Seeds: {n_seeds}")
    print()

    results = []
    print(f"  {'Dim':>4}  {'Top-1':>7}  {'Top-10':>8}  {'Nodes':>6}  {'Edges':>6}  "
          f"{'Params':>8}  {'Time':>6}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}")

    for dim in dims:
        dim_results = []
        for seed in range(n_seeds):
            res = train_and_evaluate(tokenizer, all_facts, dim,
                                     epochs=epochs, seed=seed,
                                     deep_sleep_every=deep_sleep_every)
            dim_results.append(res)

        # Aggregate across seeds
        agg = {
            "embed_dim": dim,
            "top1_mean": float(np.mean([r["top1"] for r in dim_results])),
            "top1_std": float(np.std([r["top1"] for r in dim_results])),
            "top10_mean": float(np.mean([r["top10"] for r in dim_results])),
            "top10_std": float(np.std([r["top10"] for r in dim_results])),
            "n_nodes_mean": float(np.mean([r["n_nodes"] for r in dim_results])),
            "n_edges_mean": float(np.mean([r["n_edges"] for r in dim_results])),
            "param_count": dim_results[0]["param_count"],
            "train_time_mean": float(np.mean([r["train_time_s"] for r in dim_results])),
            "per_seed": dim_results,
        }
        results.append(agg)
        print(f"  {dim:>4}  {agg['top1_mean']:>7.1%}  {agg['top10_mean']:>8.1%}  "
              f"{agg['n_nodes_mean']:>6.0f}  {agg['n_edges_mean']:>6.0f}  "
              f"{agg['param_count']:>8}  {agg['train_time_mean']:>5.1f}s")

    # Scaling analysis
    if len(results) >= 2:
        print(f"\n  Scaling analysis:")
        base = results[0]
        for r in results[1:]:
            dim_ratio = r["embed_dim"] / base["embed_dim"]
            time_ratio = r["train_time_mean"] / max(base["train_time_mean"], 0.1)
            param_ratio = r["param_count"] / max(base["param_count"], 1)
            top1_delta = r["top1_mean"] - base["top1_mean"]
            print(f"    {base['embed_dim']} -> {r['embed_dim']}: "
                  f"dim x{dim_ratio:.0f}, params x{param_ratio:.1f}, "
                  f"time x{time_ratio:.1f}, top1 {top1_delta*100:+.1f}pp")

    output = {
        "results": results,
        "config": {
            "dims": dims,
            "epochs": epochs,
            "seeds": n_seeds,
            "vocab_size": tokenizer.vocab_size,
            "deep_sleep_every": deep_sleep_every,
            "fast_sleep": args.fast_sleep,
            "n_facts": len(all_facts),
        },
    }
    out_path = os.path.join(_PROJECT_ROOT, "revisions", "dimension_scaling_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
