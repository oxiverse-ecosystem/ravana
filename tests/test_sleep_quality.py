"""
Sleep Quality Test for RAVANA RLM

Tests whether sleep_cycle() consolidation helps or hurts retrieval accuracy.
Trains simple patterns (0->1, 2->3, 4->5), measures retrieval before/after sleep,
and reports graph metrics.

Usage:
    python test_sleep_quality.py
"""

import sys
import os
import time
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'ravana-v2'))

from ravana_ml.nn.rlm import RLM
from ravana_ml.tensor import StateTensor


def get_graph_metrics(graph):
    """Extract key graph metrics."""
    n_edges = len(graph.edges)
    weights = [e.weight for e in graph.edges.values()]
    inhibitory = sum(1 for e in graph.edges.values() if e.edge_type == "inhibitory")

    # Graph entropy from activations
    activations = np.array([n.activation for n in graph.nodes.values()])
    active_mask = activations > 0.001
    if active_mask.any():
        probs = activations[active_mask]
        probs = probs / (probs.sum() + 1e-15)
        raw_entropy = -np.sum(probs * np.log(probs + 1e-15))
        max_entropy = np.log(max(active_mask.sum(), 2))
        graph_entropy = float(raw_entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        graph_entropy = 0.0

    return {
        "edge_count": n_edges,
        "mean_edge_weight": float(np.mean(weights)) if weights else 0.0,
        "inhibitory_edges": inhibitory,
        "graph_entropy": graph_entropy,
        "node_count": len(graph.nodes),
    }


def measure_retrieval_accuracy(model, pairs):
    """For each (input, expected_output) pair, run forward() and check top-1 logit.

    Returns (accuracy, details_list).
    """
    correct = 0
    details = []
    for inp, expected in pairs:
        token_ids = np.array([[inp]], dtype=np.int64)
        logits = model.forward(token_ids)
        logits_data = logits.data.flatten() if hasattr(logits.data, 'flatten') else logits.data[0]
        predicted = int(np.argmax(logits_data))
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
        top5_idx = np.argsort(logits_data)[-5:][::-1]
        top5_vals = [float(logits_data[i]) for i in top5_idx]
        details.append({
            "input": inp,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "top5_tokens": top5_idx.tolist(),
            "top5_logits": top5_vals,
        })
    accuracy = correct / len(pairs) if pairs else 0.0
    return accuracy, details


def main():
    print("=" * 70)
    print("RAVANA RLM — Sleep Quality Test")
    print("=" * 70)
    print()

    # ── 1. Create model ──
    print("[1] Creating RLM(vocab_size=20, embed_dim=32, concept_dim=32, n_hidden=64, n_concepts=20)")
    model = RLM(
        vocab_size=20,
        embed_dim=32,
        concept_dim=32,
        n_hidden=64,
        n_concepts=20,
        sleep_interval=99999,  # disable auto-sleep; we'll trigger manually
    )
    # Disable auto-sleep triggered by sleep_pressure
    model.sleep_pressure_threshold = 999.0
    print(f"    Model: {model}")
    print()

    # ── 2. Train on patterns ──
    pairs = [(0, 1), (2, 3), (4, 5)]
    n_steps = 200
    print(f"[2] Training {n_steps} steps on patterns: {pairs}")

    losses = []
    t0 = time.time()
    for step in range(n_steps):
        # Cycle through pairs
        inp, out = pairs[step % len(pairs)]
        token_ids = np.array([[inp]], dtype=np.int64)
        next_ids = np.array([out], dtype=np.int64)
        err = model.learn(token_ids, next_ids)
        losses.append(err)

        if (step + 1) % 50 == 0:
            recent = losses[-50:]
            print(f"    Step {step+1:4d}: mean_loss={np.mean(recent):.4f}  "
                  f"acc={model.conceptual_accuracy:.3f}  "
                  f"edges={len(model.graph.edges)}")

    train_time = time.time() - t0
    print(f"    Training done in {train_time:.1f}s, final mean_loss={np.mean(losses[-50:]):.4f}")
    print()

    # ── 3. Measure BEFORE sleep ──
    print("[3] Retrieval accuracy BEFORE sleep:")
    pre_acc, pre_details = measure_retrieval_accuracy(model, pairs)
    for d in pre_details:
        mark = "OK" if d["correct"] else "MISS"
        print(f"    {d['input']} -> {d['expected']}: predicted={d['predicted']}  [{mark}]"
              f"  top5={d['top5_tokens']} logits=[{', '.join(f'{v:.3f}' for v in d['top5_logits'])}]")
    print(f"    Accuracy: {pre_acc:.1%} ({int(pre_acc*len(pairs))}/{len(pairs)})")
    print()

    # Graph metrics before sleep
    print("[3b] Graph metrics BEFORE sleep:")
    pre_metrics = get_graph_metrics(model.graph)
    for k, v in pre_metrics.items():
        print(f"    {k}: {v}")
    print()

    # ── 4. Run sleep cycle ──
    print("[4] Running sleep_cycle()...")
    # First trigger a forward pass on one of the pairs to activate concepts before sleep
    model.forward(np.array([[0]], dtype=np.int64))

    t0 = time.time()
    model.sleep_cycle()
    sleep_time = time.time() - t0
    print(f"    Sleep completed in {sleep_time:.2f}s")
    print(f"    Sleep cycles completed: {model.sleep_cycles_completed}")
    print()

    # ── 5. Measure AFTER sleep ──
    print("[5] Retrieval accuracy AFTER sleep:")
    post_acc, post_details = measure_retrieval_accuracy(model, pairs)
    for d in post_details:
        mark = "OK" if d["correct"] else "MISS"
        print(f"    {d['input']} -> {d['expected']}: predicted={d['predicted']}  [{mark}]"
              f"  top5={d['top5_tokens']} logits=[{', '.join(f'{v:.3f}' for v in d['top5_logits'])}]")
    print(f"    Accuracy: {post_acc:.1%} ({int(post_acc*len(pairs))}/{len(pairs)})")
    print()

    # Graph metrics after sleep
    print("[5b] Graph metrics AFTER sleep:")
    post_metrics = get_graph_metrics(model.graph)
    for k, v in post_metrics.items():
        print(f"    {k}: {v}")
    print()

    # ── 6. Comparison ──
    print("[6] COMPARISON — Before vs After Sleep:")
    print("-" * 70)

    acc_delta = post_acc - pre_acc
    if acc_delta > 0:
        verdict = "IMPROVED"
    elif acc_delta < 0:
        verdict = "DEGRADED"
    else:
        verdict = "UNCHANGED"

    print(f"    Accuracy:      {pre_acc:.1%} -> {post_acc:.1%}  (delta={acc_delta:+.1%})  [{verdict}]")
    print()

    for key in ["edge_count", "mean_edge_weight", "inhibitory_edges", "graph_entropy", "node_count"]:
        pre_v = pre_metrics.get(key, 0)
        post_v = post_metrics.get(key, 0)
        delta = post_v - pre_v
        if isinstance(pre_v, float):
            print(f"    {key:20s}: {pre_v:.4f} -> {post_v:.4f}  (delta={delta:+.4f})")
        else:
            print(f"    {key:20s}: {pre_v:6d} -> {post_v:6d}  (delta={delta:+d})")

    print()
    print("-" * 70)

    # Per-pair comparison
    print("    Per-pair before/after:")
    for i, (pre_d, post_d) in enumerate(zip(pre_details, post_details)):
        b = "OK" if pre_d["correct"] else "MISS"
        a = "OK" if post_d["correct"] else "MISS"
        changed = "" if pre_d["correct"] == post_d["correct"] else (" <- FIXED" if post_d["correct"] else " <- BROKEN")
        print(f"      {pre_d['input']}->{pre_d['expected']}: {b} -> {a}{changed}")

    print()
    print("=" * 70)
    print("CONCLUSION: Sleep consolidation", end=" ")
    if acc_delta > 0:
        print("HELPED retrieval accuracy.")
    elif acc_delta < 0:
        print(f"HURT retrieval accuracy ({acc_delta:+.1%} degradation).")
    else:
        print("had NO EFFECT on retrieval accuracy.")

    # Additional analysis
    edge_delta = post_metrics["edge_count"] - pre_metrics["edge_count"]
    inh_delta = post_metrics["inhibitory_edges"] - pre_metrics["inhibitory_edges"]
    if edge_delta > 0:
        print(f"  Sleep created {edge_delta} new edges (structural plasticity active).")
    if inh_delta > 0:
        print(f"  Sleep created {inh_delta} new inhibitory edges (contradiction resolution active).")
    elif inh_delta < 0:
        print(f"  Sleep removed {-inh_delta} inhibitory edges.")

    weight_delta = post_metrics["mean_edge_weight"] - pre_metrics["mean_edge_weight"]
    if abs(weight_delta) > 0.01:
        direction = "strengthened" if weight_delta > 0 else "weakened"
        print(f"  Mean edge weight {direction} by {abs(weight_delta):.4f} (homeostatic scaling).")

    print("=" * 70)

    return {
        "pre_accuracy": pre_acc,
        "post_accuracy": post_acc,
        "delta": acc_delta,
        "verdict": verdict,
        "pre_metrics": pre_metrics,
        "post_metrics": post_metrics,
    }


if __name__ == "__main__":
    result = main()
