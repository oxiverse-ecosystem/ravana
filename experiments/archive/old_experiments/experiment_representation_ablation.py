"""
Representation Ablation Diagnostic

Measures whether the hidden state `h` differentiates between inputs,
and isolates GRU output vs. concept pathway contribution.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ravana_ml.nn.rlm import RLM
from ravana_ml.tokenizer import SimpleTokenizer
from experiments.experiment_cross_domain import (
    build_domain_a_science, build_domain_b_social,
    train_rlm_on_domain, encode_fact, compute_unigram_baseline
)

def main():
    np.random.seed(42)
    tokenizer = SimpleTokenizer()
    vocab_size = tokenizer.vocab_size
    model = RLM(
        vocab_size=vocab_size,
        embed_dim=32, concept_dim=32, n_concepts=vocab_size,
        n_hidden=32, n_layers=3, tokenizer=tokenizer,
    )

    # Train on both domains
    domain_a = build_domain_a_science()
    domain_b = build_domain_b_social()
    train_rlm_on_domain(model, domain_a["train"], tokenizer, n_repeats=3,
                         domain_tag="science", buffer_for_replay=True)
    train_rlm_on_domain(model, domain_b["train"], tokenizer, n_repeats=3,
                         domain_tag="social", buffer_for_replay=True)
    model.sleep_cycle()

    # Test probes
    probes = [
        "kindness causes ",
        "anger produces ",
        "heat causes ",
        "trust is ",
        "gravity pulls ",
        "collaboration produces ",
        "gossip spreads ",
        "fire produces ",
    ]

    print("\n" + "="*70)
    print("  REPRESENTATION ABLATION DIAGNOSTIC")
    print("="*70)

    # ── 1. Collect hidden states ──
    h_states = []
    for probe in probes:
        input_ids = np.array(tokenizer.encode(probe), dtype=np.int64)
        # Run forward pass to populate _last_hidden_state
        model.forward(input_ids[np.newaxis, :])
        h_states.append(model._last_hidden_state.copy())

    h_matrix = np.array(h_states)  # (n_probes, n_hidden)

    # ── 2. Hidden state variance analysis ──
    print("\n[1] Hidden State Variance")
    h_std_per_dim = np.std(h_matrix, axis=0)
    h_mean_per_dim = np.mean(h_matrix, axis=0)
    print(f"  Overall h std (per-dim mean): {np.mean(h_std_per_dim):.4f}")
    print(f"  Overall h std (per-dim max):  {np.max(h_std_per_dim):.4f}")
    print(f"  Overall h std (per-dim min):  {np.min(h_std_per_dim):.4f}")
    print(f"  h mean magnitude: {np.mean(np.abs(h_mean_per_dim)):.4f}")

    # Cross-input cosine similarities
    print("\n  Cross-input cosine similarities:")
    norms = np.linalg.norm(h_matrix, axis=1, keepdims=True)
    h_normed = h_matrix / (norms + 1e-15)
    cos_sims = h_normed @ h_normed.T
    for i in range(len(probes)):
        for j in range(i+1, len(probes)):
            print(f"    cos({probes[i].strip()}, {probes[j].strip()}) = {cos_sims[i,j]:.4f}")

    # ── 3. Context logits directly from h ──
    print("\n[2] context_logits(h) — Top-5 per probe")
    for i, probe in enumerate(probes):
        h = h_states[i]
        ctx = model.context_logits.forward_raw(h[np.newaxis, :]).flatten()
        top5 = np.argsort(ctx)[-5:][::-1]
        top5_text = [chr(int(t)) if 32 <= int(t) < 127 else f"#{int(t)}" for t in top5]
        top5_scores = [f"{ctx[t]:.2f}" for t in top5]
        print(f"  '{probe.strip()}' -> {list(zip(top5_text, top5_scores))}")

    # ── 4. Full pipeline logits ──
    print("\n[3] Full pipeline logits — Top-5 per probe")
    for i, probe in enumerate(probes):
        input_ids = np.array(tokenizer.encode(probe), dtype=np.int64)
        logits = model.forward(input_ids[np.newaxis, :]).data
        if logits.ndim > 1:
            logits = logits[0]
        top5 = np.argsort(logits)[-5:][::-1]
        top5_text = [chr(int(t)) if 32 <= int(t) < 127 else f"#{int(t)}" for t in top5]
        top5_scores = [f"{logits[t]:.2f}" for t in top5]
        print(f"  '{probe.strip()}' -> {list(zip(top5_text, top5_scores))}")

    # ── 5. Concept pathway contribution ──
    print("\n[4] Concept pathway contribution")
    for i, probe in enumerate(probes[:3]):
        input_ids = np.array(tokenizer.encode(probe), dtype=np.int64)
        logits = model.forward(input_ids[np.newaxis, :]).data
        if logits.ndim > 1:
            logits = logits[0]

        # Get individual pathway contributions
        h = model._last_hidden_state
        z = model.concept_predictor(h[np.newaxis, :]).data[0]
        z_norm = z / (np.linalg.norm(z) + 1e-15)

        # Concept logits (from graph spreading)
        model.graph.reset_activation()
        if model.graph._vectors_dirty or model.graph._vector_matrix_normed is None:
            model.graph._rebuild_vector_matrix()
        sims = model.graph._vector_matrix_normed @ z_norm.astype(np.float32)
        top_k = min(3, len(sims))
        top_indices = np.argsort(sims)[-top_k:][::-1]
        activated_nodes = [(model.graph._node_id_order[idx], float(sims[idx])) for idx in top_indices]

        # Context logits
        ctx_logits = model.context_logits.forward_raw(h[np.newaxis, :]).flatten()

        print(f"\n  '{probe.strip()}':")
        print(f"    h magnitude: {np.linalg.norm(h):.4f}")
        print(f"    z magnitude: {np.linalg.norm(z):.4f}")
        print(f"    Top concepts: {activated_nodes}")
        print(f"    ctx_logits top-3: {np.argsort(ctx_logits)[-3:][::-1]} (scores: {np.sort(ctx_logits)[-3:][::-1]})")

    # ── 6. Skip connection test: h + last input embedding ──
    print("\n[6] Skip Connection Test: context_logits(h ⊕ input_embed)")
    # Get input embeddings for the first differentiating character of each probe
    # (last char is always space, so use first char of the probe word)
    input_embeds = []
    for probe in probes:
        input_ids = np.array(tokenizer.encode(probe), dtype=np.int64)
        # Use the first token that differs between probes
        first_char_id = int(input_ids[0])
        input_embeds.append(model.token_embed.embed_raw(first_char_id))
    input_embeds = np.array(input_embeds)  # (n_probes, embed_dim)

    # Project input_embed to n_hidden dim (simple random projection)
    # Use concept_predictor's weight matrix as a proxy (already exists)
    # Or just test: does input_embed differentiate probes?
    print("\n  Input embedding cosine similarities (last token):")
    ie_norms = np.linalg.norm(input_embeds, axis=1, keepdims=True)
    ie_normed = input_embeds / (ie_norms + 1e-15)
    ie_cos = ie_normed @ ie_normed.T
    for i in range(len(probes)):
        for j in range(i+1, len(probes)):
            print(f"    cos({probes[i].strip()}, {probes[j].strip()}) = {ie_cos[i,j]:.4f}")

    # Test: expand context_logits to accept h⊕embed
    # Create new linear: n_hidden + embed_dim → vocab_size
    old_w = model.context_logits.weight.data  # (vocab_size, n_hidden)
    old_b = model.context_logits.bias.data
    n_new = model.n_hidden + model.embed_dim
    new_w = np.zeros((model.vocab_size, n_new), dtype=np.float32)
    new_w[:, :model.n_hidden] = old_w  # copy old weights for h part
    # Use concept_predictor weight for embed part (random projection)
    # Actually, use a small random init for the embed→logits part
    np.random.seed(42)
    new_w[:, model.n_hidden:] = np.random.randn(model.vocab_size, model.embed_dim).astype(np.float32) * 0.1

    print("\n  context_logits(h ⊕ embed) — Top-5 per probe:")
    for i, probe in enumerate(probes):
        h = h_states[i]
        ie = input_embeds[i]
        combined = np.concatenate([h, ie])
        logits = combined @ new_w.T + old_b
        top5 = np.argsort(logits)[-5:][::-1]
        top5_text = [chr(int(t)) if 32 <= int(t) < 127 else f"#{int(t)}" for t in top5]
        top5_scores = [f"{logits[t]:.2f}" for t in top5]
        print(f"  '{probe.strip()}' -> {list(zip(top5_text, top5_scores))}")

    # Also test: input_embed alone (bypass GRU entirely)
    print("\n  input_embed alone → random projection — Top-5 per probe:")
    for i, probe in enumerate(probes):
        ie = input_embeds[i]
        logits = ie @ new_w[:, model.n_hidden:].T + old_b
        top5 = np.argsort(logits)[-5:][::-1]
        top5_text = [chr(int(t)) if 32 <= int(t) < 127 else f"#{int(t)}" for t in top5]
        top5_scores = [f"{logits[t]:.2f}" for t in top5]
        print(f"  '{probe.strip()}' -> {list(zip(top5_text, top5_scores))}")

    # ── 7. Is h input-invariant? ──
    print("\n[5] Is h input-invariant?")
    h_range = np.max(h_matrix, axis=0) - np.min(h_matrix, axis=0)
    print(f"  h dimension range (max-min): mean={np.mean(h_range):.4f}, max={np.max(h_range):.4f}")
    print(f"  Dimensions with range > 0.1: {np.sum(h_range > 0.1)}/{len(h_range)}")
    print(f"  Dimensions with range > 0.01: {np.sum(h_range > 0.01)}/{len(h_range)}")

    # Fraction of variance explained by input
    total_var = np.var(h_matrix)
    mean_var = np.var(np.mean(h_matrix, axis=0))
    input_var = total_var - mean_var
    print(f"  Total variance: {total_var:.6f}")
    print(f"  Input-dependent variance: {input_var:.6f} ({input_var/total_var*100:.1f}% of total)")


if __name__ == "__main__":
    main()
