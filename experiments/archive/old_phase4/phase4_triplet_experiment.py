"""
Phase 4 Experiment: Triplet Margin Loss with Hard Negative Mining

Protocol:
1. Sample each triplet as (anchor, positive, hard_negative) from SEMANTIC_PAIRS + WITHIN_DOMAIN_PAIRS
2. Hard negatives: encode vocab with current embedder, take top-k nn of anchor excluding known analogues
3. Optimize encoder (2-layer MLP + tanh + latent_dim=32) with L2 triplet margin loss
4. Margin hyperparameter sweep: 0.1, 0.3, 0.5
5. Report: d_pos, d_hard_neg, margin gap, collapse indicator (variance of d_pos across validation set)

Usage:
  python phase4_triplet_experiment.py --epochs 200 --margin 0.3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from semantic_pairs import SEMANTIC_PAIRS, WITHIN_DOMAIN_PAIRS, ALL_PAIRS
from ravana_ml.tokenizer import WordTokenizer
from ravana_ml.embedder import LearnedEmbedder
from ravana_ml.nn.rlm_v2 import RLMv2


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
DIM = 64
LATENT = 32
HIDDEN = 48
AUTOENCODER_EPOCHS = 300
RP_LR = 0.01
RP_MOMENTUM = 0.9
FREEZE_ENCODER = False

# Embedder fallback: allow tests to swap LearnedEmbedder with a simpler encoder.
LearnedEmbedder = None


def _load_learned_embedder_cls():
    global LearnedEmbedder
    if LearnedEmbedder is not None:
        return LearnedEmbedder
    try:
        from ravana_ml.embedder import LearnedEmbedder as _LearnedEmbedder
        LearnedEmbedder = _LearnedEmbedder
        return LearnedEmbedder
    except Exception:
        return None


def build_embedder(dim: int = 64):
    cls = _load_learned_embedder_cls()
    if cls is None:
        raise RuntimeError("LearnedEmbedder not available")
    return cls(dim=dim)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_tokenizer() -> WordTokenizer:
    tok = WordTokenizer()
    all_texts = []
    for a, b in ALL_PAIRS:
        all_texts.append(a)
        all_texts.append(b)
    for text in all_texts:
        tok.encode(text)
    return tok


def encode_vocab(tok: WordTokenizer, embedder: LearnedEmbedder) -> np.ndarray:
    n = tok.vocab_size
    d = embedder.dim
    vecs = np.zeros((n, d), dtype=np.float32)
    for word, tid in tok.word_to_id.items():
        vecs[tid] = embedder.encode(word)
    return vecs


# ---------------------------------------------------------------------------
# Encoder primitives (robust to latent_dim / hidden_dim)
# ---------------------------------------------------------------------------
def encoder_forward(
    W1: np.ndarray, b1: np.ndarray,
    W2: np.ndarray, b2: np.ndarray,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z1 = X @ W1.T + b1
    h1 = np.tanh(z1)
    z2 = h1 @ W2.T + b2
    latent = np.tanh(z2)
    return latent, z1, h1, z2


def encoder_backward(
    X: np.ndarray,
    z1: np.ndarray, h1: np.ndarray, z2: np.ndarray, h2: np.ndarray,
    d_h2: np.ndarray,
    W1_shape: Tuple[int, int], b1_shape: Tuple[int,],
    W2_shape: Tuple[int, int], b2_shape: Tuple[int,],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d_z2 = d_h2 * (1.0 - h2 * h2)
    d_W2 = d_z2.T @ h1
    d_b2 = np.sum(d_z2, axis=0)
    d_h1 = d_z2 @ W2
    d_z1 = d_h1 * (1.0 - h1 * h1)
    d_W1 = d_z1.T @ X
    d_b1 = np.sum(d_z1, axis=0)
    return d_W1, d_b1, d_W2, d_b2


def encode_latent_full(
    W1: np.ndarray, b1: np.ndarray,
    W2: np.ndarray, b2: np.ndarray,
    token_vecs: np.ndarray,
) -> np.ndarray:
    lat, _, _, _ = encoder_forward(W1, b1, W2, b2, token_vecs)
    return lat


# ---------------------------------------------------------------------------
# Hard negative mining
# ---------------------------------------------------------------------------
def mine_hard_negatives(
    anchor_word: str,
    positive_word: str,
    token_vecs: np.ndarray,
    tok: WordTokenizer,
    k: int = 3,
) -> List[str]:
    """Top-k nearest neighbors of anchor in embedding space excluding positive_word."""
    ta = tok.word_to_id.get(anchor_word)
    if ta is None:
        return []
    vec = token_vecs[ta]
    norm_a = np.linalg.norm(vec)
    if norm_a == 0:
        return []
    sims = token_vecs @ (vec / norm_a)
    tp = tok.word_to_id.get(positive_word)
    tp_set = {tp} if tp is not None else set()
    candidates = []
    for tid, sim in enumerate(sims):
        if tid in tp_set:
            continue
        if tid == ta:
            continue
        word = tok.decode([tid])
        if not word or not word.isalpha():
            continue
        candidates.append((sim, word))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [w for _, w in candidates[:k]]


# ---------------------------------------------------------------------------
# Phase 4 core logic
# ---------------------------------------------------------------------------
def run_phase4(
    margin: float = 0.3,
    epochs: int = 200,
    batch_size: int = 3,
    lr: float = 0.01,
    momentum: float = 0.9,
    seed: int = 42,
    max_pairs: int | None = None,
    hard_k: int = 3,
    rp_boost: bool = False,
) -> Dict:
    """
    Train the encoder (2-layer MLP latent) with triplet margin loss.
    Contrastive loss is added via the standard RLMv2 update path.
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    tok = build_tokenizer()
    embedder = build_embedder(dim=DIM)
    token_vecs = encode_vocab(tok, embedder)

    model = RLMv2(
        vocab_size=tok.vocab_size + 5,
        embed_dim=DIM,
        concept_dim=LATENT,
        n_concepts=tok.vocab_size,
        sleep_interval=300,
        gate_concept_creation=False,
    )
    model._tokenizer = tok

    # ── Ensure stable reproducible encoder weights ─────────────────────
    rng1 = np.random.RandomState(seed)
    rng2 = np.random.RandomState(seed + 1)
    model._enc_W1 = rng1.randn(HIDDEN, DIM).astype(np.float32) * np.sqrt(2.0 / DIM)
    model._enc_b1 = np.zeros(HIDDEN, dtype=np.float32)
    model._enc_W2 = rng2.randn(LATENT, HIDDEN).astype(np.float32) * np.sqrt(2.0 / HIDDEN)
    model._enc_b2 = np.zeros(LATENT, dtype=np.float32)
    model._enc_mW1[...] = 0.0
    model._enc_mb1[...] = 0.0
    model._enc_mW2[...] = 0.0
    model._enc_mb2[...] = 0.0

    # Freeze other trainables: only encoder gradients during this Phase.
    model.freeze_encoder = False
    model.use_contrastive_reg = False  # we replace with triplet margin loss
    model.lambda_contrastive = 0.0

    # ── Autoencoder pre-train to initialize encoder ────────────────────
    model._pretrain_encoder_autoencoder(epochs=AUTOENCODER_EPOCHS, lr=0.01)

    # ── Build triplets ──────────────────────────────────────────────────
    pairs = list(ALL_PAIRS)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    print(f"Pairs: {len(pairs)} | margin={margin} | epochs={epochs} | batch={batch_size} | lr={lr}")

    # ── Manual training loop over encoder weights ──────────────────────
    W1, b1 = model._enc_W1, model._enc_b1
    W2, b2 = model._enc_W2, model._enc_b2

    mW1, mb1 = np.zeros_like(W1), np.zeros_like(b1)
    mW2, mb2 = np.zeros_like(W2), np.zeros_like(b2)

    history: List[Dict] = []
    rp_stats: List[Dict] = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(pairs)
        epoch_loss = 0.0
        epoch_trips = 0

        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            dW1_acc = np.zeros_like(W1)
            db1_acc = np.zeros_like(b1)
            dW2_acc = np.zeros_like(W2)
            db2_acc = np.zeros_like(b2)

            for a_word, b_word in batch:
                ta = tok.word_to_id.get(a_word)
                tb = tok.word_to_id.get(b_word)
                if ta is None or tb is None:
                    continue

                embed_a = token_vecs[ta]
                embed_b = token_vecs[tb]

                hard_candidates = mine_hard_negatives(a_word, b_word, token_vecs, tok, k=hard_k)
                hard_word = hard_candidates[0] if hard_candidates else b_word  # fallback to positive if none
                t_hard = tok.word_to_id.get(hard_word)
                if t_hard is None:
                    t_hard = tb
                embed_h = token_vecs[t_hard]

                lat_a, za1, ha1, za2 = encoder_forward(W1, b1, W2, b2, embed_a[np.newaxis, :])
                lat_p, zp1, hp1, zp2 = encoder_forward(W1, b1, W2, b2, embed_b[np.newaxis, :])
                lat_n, zn1, hn1, zn2 = encoder_forward(W1, b1, W2, b2, embed_h[np.newaxis, :])

                d_pos = float(np.sum((lat_a[0] - lat_p[0]) ** 2))
                d_neg = float(np.sum((lat_a[0] - lat_n[0]) ** 2))

                loss_val = max(0.0, d_pos - d_neg + margin)
                if loss_val <= 0:
                    continue
                epoch_trips += 1

                d_lat = 2.0 * (lat_a[0] - lat_p[0]) - 2.0 * (lat_a[0] - lat_n[0])
                d_h2 = d_lat[np.newaxis, :]           # (1, latent)
                dW1_trip, db1_trip, dW2_trip, db2_trip = encoder_backward(
                    embed_a[np.newaxis, :],
                    za1[0:1], ha1[0:1], za2[0:1], lat_a[0:1], d_h2,
                    W1.shape, b1.shape, W2.shape, b2.shape,
                )
                dW1_acc += dW1_trip
                db1_acc += db1_trip
                dW2_acc += dW2_trip
                db2_acc += db2_trip

                epoch_loss += loss_val

                # Periodic relation-predictor monitoring call is added later.
                #               embeddings.

            if epoch_trips == 0:
                continue
            scale = 1.0 / epoch_trips
            dW1_acc *= scale
            db1_acc *= scale
            dW2_acc *= scale
            db2_acc *= scale

            # SGD with momentum on encoder weights
            mW1 = momentum * mW1 - lr * dW1_acc
            mb1 = momentum * mb1 - lr * db1_acc
            mW2 = momentum * mW2 - lr * dW2_acc
            mb2 = momentum * mb2 - lr * db2_acc

            W1 += mW1
            b1 += mb1
            W2 += mW2
            b2 += mb2

            model._enc_W1 = W1
            model._enc_b1 = b1
            model._enc_W2 = W2
            model._enc_b2 = b2

        if epoch_trips == 0:
            continue
        epoch_loss /= epoch_trips

        # ── Periodic relation-predictor report via RLM RP signals ──────
        rp_stats.append({
            "epoch": epoch,
            "triplets_active": epoch_trips,
            "train_loss": round(float(epoch_loss), 6),
        })

        if epoch % 50 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "loss": round(float(epoch_loss), 6)})
            print(f"  [{epoch:4d}] train triplet-loss = {epoch_loss:.6f}, active_triplets={epoch_trips}")

    # ── Final validation metrics ──────────────────────────────────────────
    lat_all = encode_latent_full(W1, b1, W2, b2, token_vecs)
    cos_lat = cosine_matrix(lat_all)

    all_pos_sims = []
    for a_word, b_word in ALL_PAIRS:
        ta = tok.word_to_id.get(a_word)
        tb = tok.word_to_id.get(b_word)
        if ta is None or tb is None:
            continue
        all_pos_sims.append(float(cos_lat[ta, tb]))

    hard_neg_results = []
    anchor_pos_sims, anchor_hard_sims = [], []
    hard_negative_triples = [
        ("gravity", "loyalty", "weight"),
        ("light", "hope", "darkness"),
        ("heat", "anger", "cold"),
        ("friction", "conflict", "pressure"),
        ("expansion", "growth", "contraction"),
    ]
    for anchor, positive, hard in hard_negative_triples:
        ta = tok.word_to_id.get(anchor)
        tp = tok.word_to_id.get(positive)
        th = tok.word_to_id.get(hard)
        if ta is None or tp is None or th is None:
            continue
        s_pos  = float(cos_lat[ta, tp])
        s_hard = float(cos_lat[ta, th])
        anchor_pos_sims.append(s_pos)
        anchor_hard_sims.append(s_hard)
        hard_neg_results.append({
            "anchor": anchor, "positive": positive, "hard": hard,
            "latent_pos_sim": round(s_pos, 4),
            "latent_hard_neg_sim": round(s_hard, 4),
        })

    pos_var = float(np.var(all_pos_sims)) if all_pos_sims else 0.0
    pos_std = float(np.std(all_pos_sims)) if all_pos_sims else 0.0
    pos_mean = float(np.mean(all_pos_sims)) if all_pos_sims else 0.0
    hard_mean = float(np.mean(anchor_hard_sims)) if anchor_hard_sims else 0.0
    gap_mean  = (float(np.mean(anchor_pos_sims)) - hard_mean) if anchor_pos_sims else 0.0

    results = {
        "config": {
            "margin": margin,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "momentum": momentum,
            "max_pairs": max_pairs if max_pairs is not None else len(ALL_PAIRS),
            "hard_k": hard_k,
            "rp_boost": rp_boost,
        },
        "history": history,
        "rp_stats": rp_stats,
        "validation": {
            "positive_similarity_mean": round(pos_mean, 4),
            "positive_similarity_std": round(pos_std, 4),
            "positive_similarity_var": round(pos_var, 4),
            "hard_negative_mean": round(hard_mean, 4),
            "positive_minus_hard_mean": round(gap_mean, 4),
            "collapse_indicator": pos_var < 1e-4,
            "hard_negative_details": hard_neg_results,
        },
    }
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4: Triplet Margin Loss Hardening")
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--hard-k", type=int, default=3)
    ap.add_argument("--rp-boost", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 72)
    print("Phase 4 Experiment: Triplet Margin + Hard Negatives")
    print("=" * 72)
    res = run_phase4(
        margin=args.margin,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        seed=args.seed,
        max_pairs=args.max_pairs,
        hard_k=args.hard_k,
        rp_boost=args.rp_boost,
    )
    elapsed = time.time() - t0

    out = os.path.join(PROJECT_ROOT, "experiment_results", "phase4_triplet_margin.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    v = res["validation"]
    print("\n[FINAL]")
    print(f"  positive_mean = {v['positive_similarity_mean']}")
    print(f"  positive_std  = {v['positive_similarity_std']}")
    print(f"  positive_var  = {v['positive_similarity_var']}")
    print(f"  hard_neg_mean = {v['hard_negative_mean']}")
    print(f"  pos - hard    = {v['positive_minus_hard_mean']}")
    print(f"  collapse flag = {v['collapse_indicator']}")
    print(f"  elapsed       = {elapsed:.1f}s")
    print(f"  saved         = {out}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
