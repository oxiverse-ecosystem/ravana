#!/usr/bin/env python3
"""
benchmark_vs_transformers.py — P3 Benchmark Harness (Sprint 6)

Compares RLMv2 against small transformer baselines on triple prediction tasks.

Usage:
    python scripts/benchmark_vs_transformers.py                # full suite
    python scripts/benchmark_vs_transformers.py --quick        # small subset
    python scripts/benchmark_vs_transformers.py --model rlm    # only RLM

Reports:
    - Accuracy, loss, parameter count, inference speed
    - Cross-domain generalization gap
    - Per-relation-type breakdown
"""

import argparse
import json
import time
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ravana_ml" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "ravana" / "src"))


# ── Synthetic benchmark data generators ─────────────────────────────

def _make_synthetic_data(
    n_triples: int = 500,
    vocab_size: int = 100,
    n_relations: int = 6,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic (subject, relation, object) triples.

    Returns (subjects, relations, objects) each shape (n_triples,) as int arrays.
    """
    rng = np.random.RandomState(seed)
    subjects = rng.randint(0, vocab_size, size=n_triples)
    objects = rng.randint(0, vocab_size, size=n_triples)
    relations = rng.randint(0, n_relations, size=n_triples)
    return subjects, relations, objects


def _make_cross_domain_data(
    n_train: int = 400,
    n_test: int = 100,
    vocab_size: int = 100,
    n_relations: int = 6,
    seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate train/test splits with held-out subject-verb combinations.

    Subjects 0-39 seen during training; subjects 40-49 held out for testing.
    Returns ((train_sub, train_rel, train_obj), (test_sub, test_rel, test_obj)).
    """
    rng = np.random.RandomState(seed)
    
    # Training: subjects 0-39, all relations
    train_sub = rng.randint(0, 40, size=n_train)
    train_rel = rng.randint(0, n_relations, size=n_train)
    train_obj = rng.randint(0, vocab_size, size=n_train)
    
    # Test: subjects 40-49 (held-out), all relations
    test_sub = rng.randint(40, 50, size=n_test)
    test_rel = rng.randint(0, n_relations, size=n_test)
    test_obj = rng.randint(0, vocab_size, size=n_test)
    
    return (train_sub, train_rel, train_obj), (test_sub, test_rel, test_obj)


# ── Evaluation utilities ────────────────────────────────────────────

def evaluate_rlm(model, subjects, relations, objects,
                 vocab_size: int, batch_size: int = 32) -> Dict:
    """Evaluate RLMv2 on triple prediction.

    Returns accuracy, mean loss, and per-relation breakdown.
    """
    from ravana_ml.nn.rlm_v2 import RELATION_TYPES

    n = len(subjects)
    correct = 0
    total_loss = 0.0
    per_rel_correct = {rt: [0, 0] for rt in RELATION_TYPES}
    latencies = []

    for i in range(0, n, batch_size):
        batch_s = subjects[i:i + batch_size]
        batch_r = relations[i:i + batch_size]
        batch_o = objects[i:i + batch_size]

        for sid, rid, oid in zip(batch_s, batch_r, batch_o):
            t0 = time.perf_counter()
            logits = model._rp_forward(int(sid), int(rid))
            lat = time.perf_counter() - t0
            latencies.append(lat)

            if logits is not None:
                pred = int(np.argmax(logits))
                is_correct = pred == int(oid)
                correct += is_correct

                # Cross-entropy loss
                exp_l = np.exp(logits - np.max(logits))
                probs = exp_l / (np.sum(exp_l) + 1e-10)
                loss = -np.log(max(probs[int(oid)], 1e-10))
                total_loss += loss

                rel_name = RELATION_TYPES[int(rid)] if int(rid) < len(RELATION_TYPES) else "unknown"
                per_rel_correct[rel_name][0] += is_correct
                per_rel_correct[rel_name][1] += 1

    return {
        "accuracy": correct / max(n, 1),
        "mean_loss": total_loss / max(n, 1),
        "n_correct": correct,
        "n_total": n,
        "mean_latency_ms": float(np.mean(latencies) * 1000) if latencies else 0.0,
        "per_relation": {
            rt: (cnt[0] / max(cnt[1], 1), cnt[1])
            for rt, cnt in per_rel_correct.items()
        },
    }


def evaluate_transformer_baseline(model, subjects, relations, objects,
                                  vocab_size: int, batch_size: int = 32) -> Dict:
    """Evaluate a PyTorch transformer model on triple prediction.

    The model must implement forward(subjects, relations) -> logits over vocab.
    """
    import torch

    model.eval()
    n = len(subjects)
    correct = 0
    total_loss = 0.0
    latencies = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_s = torch.tensor(subjects[i:i + batch_size], dtype=torch.long)
            batch_r = torch.tensor(relations[i:i + batch_size], dtype=torch.long)
            batch_o = torch.tensor(objects[i:i + batch_size], dtype=torch.long)

            t0 = time.perf_counter()
            logits = model(batch_s, batch_r)
            lat = time.perf_counter() - t0
            latencies.append(lat)

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch_o).sum().item()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fct(logits, batch_o)
            total_loss += loss.item()

    return {
        "accuracy": correct / max(n, 1),
        "mean_loss": total_loss / max(n, 1),
        "n_correct": correct,
        "n_total": n,
        "mean_latency_ms": float(np.mean(latencies) * 1000) if latencies else 0.0,
    }


# ── Transformer baselines (minimal PyTorch if available) ────────────

def _build_linear_baseline(vocab_size: int, embed_dim: int, n_relations: int) -> object:
    """Build a simple linear baseline: (subject_embed + rel_embed) @ W -> logits.

    Returns the model object.
    """
    import torch
    import torch.nn as nn

    class LinearBaseline(nn.Module):
        def __init__(self, vocab_size, embed_dim, n_relations):
            super().__init__()
            self.token_embed = nn.Embedding(vocab_size, embed_dim)
            self.rel_embed = nn.Embedding(n_relations, embed_dim)
            self.W = nn.Linear(embed_dim, vocab_size, bias=False)

            # Init
            nn.init.normal_(self.token_embed.weight, std=0.1)
            nn.init.normal_(self.rel_embed.weight, std=0.1)

        def forward(self, subjects, relations):
            s = self.token_embed(subjects)
            r = self.rel_embed(relations)
            h = s + r
            return self.W(h)

    return LinearBaseline(vocab_size, embed_dim, n_relations)


def _build_mlp_baseline(vocab_size: int, embed_dim: int, n_relations: int,
                        hidden_dim: int = 64) -> object:
    """Build a 2-layer MLP baseline."""
    import torch
    import torch.nn as nn

    class MLPBaseline(nn.Module):
        def __init__(self, vocab_size, embed_dim, n_relations, hidden_dim):
            super().__init__()
            self.token_embed = nn.Embedding(vocab_size, embed_dim)
            self.rel_embed = nn.Embedding(n_relations, embed_dim)
            self.net = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, vocab_size),
            )

        def forward(self, subjects, relations):
            s = self.token_embed(subjects)
            r = self.rel_embed(relations)
            h = torch.cat([s, r], dim=-1)
            return self.net(h)

    return MLPBaseline(vocab_size, embed_dim, n_relations, hidden_dim)


# ── Main benchmark runner ───────────────────────────────────────────

def run_benchmark(args):
    """Run the full benchmark suite."""
    results = {}

    # Generate data
    print("Generating synthetic benchmark data...")
    train_data, test_data = _make_cross_domain_data(
        n_train=400, n_test=100, vocab_size=100, n_relations=6
    )
    train_sub, train_rel, train_obj = train_data
    test_sub, test_rel, test_obj = test_data

    vocab_size = 100
    embed_dim = 16
    concept_dim = 16

    # ── RLMv2 ────────────────────────────────────────────────────────
    if args.model in ("rlm", "all"):
        print("\n=== RLMv2 ===")
        from ravana_ml.nn.rlm_v2 import RLMv2

        rlm = RLMv2(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            concept_dim=concept_dim,
            n_concepts=vocab_size,
        )

        # Quick training loop (simple bilinear)
        print("  Training RLMv2 on synthetic triples...")
        lr = 0.01
        n_epochs = args.epochs
        for epoch in range(n_epochs):
            losses = []
            for s, r, o in zip(train_sub, train_rel, train_obj):
                logits = rlm._rp_forward(int(s), int(r))
                if logits is None:
                    continue
                exp_l = np.exp(logits - np.max(logits))
                probs = exp_l / (np.sum(exp_l) + 1e-10)
                loss = -np.log(max(probs[int(o)], 1e-10))
                losses.append(loss)

                rlm._rp_backward(int(o), lr_scale=1.0)

            if epoch % 5 == 0:
                print(f"  Epoch {epoch:3d}: mean loss={np.mean(losses):.4f}")
                
        # Compute verb offsets
        for s, r, o in zip(train_sub, train_rel, train_obj):
            verb_word = ["causes", "is", "then", "has", "like", "in"][int(r) % 6]
            rlm._accumulate_verb_offset(int(s), int(o), verb_word)
        rlm._compute_verb_offsets()

        print("  Evaluating...")
        train_results = evaluate_rlm(rlm, train_sub, train_rel, train_obj, vocab_size)
        test_results = evaluate_rlm(rlm, test_sub, test_rel, test_obj, vocab_size)

        # Count numeric array parameters only (exclude int/bool config values)
        rlm_params = sum(
            v.size if hasattr(v, 'size') else 0
            for v in rlm.state_dict().values()
        )
        results["rlmv2"] = {
            "train_accuracy": train_results["accuracy"],
            "test_accuracy": test_results["accuracy"],
            "generalization_gap": train_results["accuracy"] - test_results["accuracy"],
            "train_loss": train_results["mean_loss"],
            "test_loss": test_results["mean_loss"],
            "parameters": rlm_params,
            "mean_latency_ms": test_results["mean_latency_ms"],
            "per_relation_test": {
                k: v[0] for k, v in test_results["per_relation"].items()
            },
        }
        print(f"  Train acc: {train_results['accuracy']:.3f}, "
              f"Test acc: {test_results['accuracy']:.3f}, "
              f"Params: {rlm_params}")

    # ── Transformer baselines ───────────────────────────────────────
    if args.model in ("linear", "all"):
        print("\n=== Linear Baseline ===")
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            lin = _build_linear_baseline(vocab_size, embed_dim, 6)
            opt = optim.Adam(lin.parameters(), lr=0.01)
            loss_fn = nn.CrossEntropyLoss()

            s_train = torch.tensor(train_sub, dtype=torch.long)
            r_train = torch.tensor(train_rel, dtype=torch.long)
            o_train = torch.tensor(train_obj, dtype=torch.long)

            for epoch in range(args.epochs):
                opt.zero_grad()
                logits = lin(s_train, r_train)
                loss = loss_fn(logits, o_train)
                loss.backward()
                opt.step()
                if epoch % 5 == 0:
                    print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}")

            s_test = torch.tensor(test_sub, dtype=torch.long)
            r_test = torch.tensor(test_rel, dtype=torch.long)
            o_test = torch.tensor(test_obj, dtype=torch.long)
            with torch.no_grad():
                logits = lin(s_test, r_test)
                preds = torch.argmax(logits, dim=-1)
                test_acc = (preds == o_test).float().mean().item()

            lin_params = sum(p.numel() for p in lin.parameters())
            results["linear_baseline"] = {
                "test_accuracy": test_acc,
                "parameters": lin_params,
            }
            print(f"  Test acc: {test_acc:.3f}, Params: {lin_params}")

        except ImportError:
            print("  [SKIP] PyTorch not available — skipping linear baseline")

    if args.model in ("mlp", "all"):
        print("\n=== MLP Baseline ===")
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            mlp = _build_mlp_baseline(vocab_size, embed_dim, 6, hidden_dim=64)
            opt = optim.Adam(mlp.parameters(), lr=0.01)
            loss_fn = nn.CrossEntropyLoss()

            s_train = torch.tensor(train_sub, dtype=torch.long)
            r_train = torch.tensor(train_rel, dtype=torch.long)
            o_train = torch.tensor(train_obj, dtype=torch.long)

            for epoch in range(args.epochs):
                opt.zero_grad()
                logits = mlp(s_train, r_train)
                loss = loss_fn(logits, o_train)
                loss.backward()
                opt.step()
                if epoch % 5 == 0:
                    print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}")

            s_test = torch.tensor(test_sub, dtype=torch.long)
            r_test = torch.tensor(test_rel, dtype=torch.long)
            o_test = torch.tensor(test_obj, dtype=torch.long)
            with torch.no_grad():
                logits = mlp(s_test, r_test)
                preds = torch.argmax(logits, dim=-1)
                test_acc = (preds == o_test).float().mean().item()

            mlp_params = sum(p.numel() for p in mlp.parameters())
            results["mlp_baseline"] = {
                "test_accuracy": test_acc,
                "parameters": mlp_params,
            }
            print(f"  Test acc: {test_acc:.3f}, Params: {mlp_params}")

        except ImportError:
            print("  [SKIP] PyTorch not available — skipping MLP baseline")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for model_name, metrics in results.items():
        acc = metrics.get("test_accuracy", metrics.get("train_accuracy", 0))
        params = metrics.get("parameters", 0)
        print(f"  {model_name:20s}  test_acc={acc:.3f}  params={params}")

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RLMv2 vs transformer baselines"
    )
    parser.add_argument("--model", choices=["rlm", "linear", "mlp", "all"],
                        default="all", help="Which model(s) to benchmark")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs per model")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results")
    parser.add_argument("--quick", action="store_true",
                        help="Run a quick subset with fewer epochs")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 10

    run_benchmark(args)
