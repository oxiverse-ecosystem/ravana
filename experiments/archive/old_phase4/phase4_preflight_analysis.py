"""
Phase 4 Preflight: validation collapse analysis for RAVANA embeddings.

This script inspects the current token / latent geometry using the
current LearnedEmbedder and encoder pre-training path in RLMv2.
It reports:
  * intra-pair similarity for SAME-DOMAIN pairs
  * intra-pair similarity for CROSS-DOMAIN analogies
  * anchor-to-hard-negative similarity
  * collapse indicator: variance across all positive-pair similarities
"""

from __future__ import annotations

import sys
import os
import json
from typing import List, Tuple, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from semantic_pairs import SEMANTIC_PAIRS, WITHIN_DOMAIN_PAIRS, ALL_PAIRS  # noqa: E402
from ravana_ml.tokenizer import WordTokenizer  # noqa: E402
from ravana_ml.embedder import LearnedEmbedder  # noqa: E402


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


def build_embedder(dim: int = 64) -> LearnedEmbedder:
    emb = LearnedEmbedder(dim=dim)
    return emb


def encode_vocab(
    tok: WordTokenizer,
    embedder: LearnedEmbedder,
) -> np.ndarray:
    """Return (vocab_size, dim) token embeddings from embedder."""
    n = tok.vocab_size
    d = embedder.dim
    vecs = np.zeros((n, d), dtype=np.float32)
    for word, tid in tok.word_to_id.items():
        vecs[tid] = embedder.encode(word)
    return vecs


def cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    n = vecs.shape[0]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = vecs / norms
    return normed @ normed.T


def encode_latent(
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    token_vecs: np.ndarray,
) -> np.ndarray:
    """Pass token_vecs (V, embed_dim) through encoder to latent space."""
    h1 = np.tanh(token_vecs @ W1.T + b1)           # (V, hidden)
    z2 = h1 @ W2.T + b2                              # (V, latent)
    latent = np.tanh(z2)
    return latent


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis() -> Dict:
    print("=" * 72)
    print("Phase 4 Preflight — Embedding Geometry Collapse Analysis")
    print("=" * 72)

    tok = build_tokenizer()
    embedder = build_embedder(dim=64)

    token_vecs = encode_vocab(tok, embedder)
    cos_token = cosine_matrix(token_vecs)

    # ── Positive-pair distributions ─────────────────────────────────────
    def pair_sims(pairs: List[Tuple[str, str]]) -> List[float]:
        sims = []
        missing = []
        for a, b in pairs:
            ta = tok.word_to_id.get(a)
            tb = tok.word_to_id.get(b)
            if ta is None or tb is None:
                missing.append((a, b))
                continue
            sims.append(float(cos_token[ta, tb]))
        if missing:
            print(f"  [WARN] missing tokenizer entries: {missing}")
        return sims

    x_sims = pair_sims(SEMANTIC_PAIRS)
    w_sims = pair_sims(WITHIN_DOMAIN_PAIRS)
    all_sims = x_sims + w_sims

    print("\n┌─ Positive-pair token similarity (LearnedEmbedder init) ──────")
    print(f"│  Cross-domain analogies : mean={np.mean(x_sims):.4f}, "
          f"std={np.std(x_sims):.4f}, min={np.min(x_sims):.4f}, "
          f"max={np.max(x_sims):.4f}")
    print(f"│  Within-domain pairs     : mean={np.mean(w_sims):.4f}, "
          f"std={np.std(w_sims):.4f}, min={np.min(w_sims):.4f}, "
          f"max={np.max(w_sims):.4f}")
    print(f"│  All positive pairs      : mean={np.mean(all_sims):.4f}, "
          f"std={np.std(all_sims):.4f}, var={np.var(all_sims):.4f}")

    # ── Hard negatives ──────────────────────────────────────────────────
    hard_negative_triples = [
        ("gravity", "loyalty", "weight"),
        ("light",    "hope",    "darkness"),
        ("heat",     "anger",   "cold"),
        ("friction", "conflict","pressure"),
        ("expansion","growth",  "contraction"),
    ]

    print("\n┌─ Hard-negative token similarity (anchor ↔ hard-neg) ─────────")
    anchor_hard_sims = []
    anchor_pos_sims  = []
    for anchor, positive, hard in hard_negative_triples:
        ta = tok.word_to_id.get(anchor)
        tp = tok.word_to_id.get(positive)
        th = tok.word_to_id.get(hard)
        if ta is None or tp is None or th is None:
            print(f"  [WARN] missing vocab entry in triple: {(anchor, positive, hard)}")
            continue
        s_pos  = float(cos_token[ta, tp])
        s_hard = float(cos_token[ta, th])
        anchor_hard_sims.append(s_hard)
        anchor_pos_sims.append(s_pos)
        print(f"│  ({anchor:12s}, +{positive:12s}, -{hard:12s}) "
              f"pos={s_pos:.4f} anchor↔hard={s_hard:.4f}")

    # ── If RLMv2 is trainable: run encoder forward and compute latent space metrics
    # We can't construct a trained RLMv2 without fully initializing it, so instead
    # we approximate: use a fresh untrained encoder on current token embeddings.
    # In Phase 4 run, we will accumulate ground-truth latent metrics.
    # Here we run a sanity-check with *untrained* latent space to confirm
    # that hard negatives can be separated once the encoder is trained.

    try:
        from ravana_ml.nn.rlm_v2 import RLMv2  # noqa: E402
    except Exception as exc:  # pragma: no cover
        print(f"\n[INFO] RLMv2 not available for latent-space inspection: {exc}")
        RLMv2 = None  # type: ignore

    latent_report = {}
    if RLMv2 is not None:
        # Build a minimally initialized model (no training)
        model = RLMv2(
            vocab_size=tok.vocab_size + 5,
            embed_dim=64,
            concept_dim=64,
            n_concepts=tok.vocab_size,
            sleep_interval=300,
            gate_concept_creation=False,
        )
        model._tokenizer = tok

        # Run the autoencoder pre-train that's triggered by tokenizer
        # (already done during tokenizer set, but can be invoked again for clean state)
        model._pretrain_encoder_autoencoder(epochs=300, lr=0.01)

        lat = encode_latent(
            model._enc_W1, model._enc_b1,
            model._enc_W2, model._enc_b2,
            token_vecs,
        )
        cos_lat = cosine_matrix(lat)
        lat_x_sims = pair_sims(SEMANTIC_PAIRS)
        lat_w_sims = pair_sims(WITHIN_DOMAIN_PAIRS)
        lat_all    = lat_x_sims + lat_w_sims

        print("\n┌─ Positive-pair latent similarity (post-autoencoder pre-train)")
        print(f"│  Cross-domain analogies : mean={np.mean(lat_x_sims):.4f}, std={np.std(lat_x_sims):.4f}")
        print(f"│  Within-domain pairs     : mean={np.mean(lat_w_sims):.4f}, std={np.std(lat_w_sims):.4f}")
        print(f"│  All positive pairs      : mean={np.mean(lat_all):.4f}, "
              f"std={np.std(lat_all):.4f}, var={np.var(lat_all):.4f}")

        anchor_pos_lat, anchor_hard_lat = [], []
        print("\n┌─ Hard-negative latent similarity (anchor ↔ hard-neg) ─────────")
        for anchor, positive, hard in hard_negative_triples:
            ta = tok.word_to_id.get(anchor)
            tp = tok.word_to_id.get(positive)
            th = tok.word_to_id.get(hard)
            if ta is None or tp is None or th is None:
                continue
            sp = float(cos_lat[ta, tp])
            sh = float(cos_lat[ta, th])
            anchor_pos_lat.append(sp)
            anchor_hard_lat.append(sh)
            print(f"│  ({anchor:12s}, +{positive:12s}, -{hard:12s}) "
                  f"pos={sp:.4f} anchor↔hard={sh:.4f}")

        if anchor_pos_lat and anchor_hard_lat:
            gap = np.mean(anchor_pos_lat) - np.mean(anchor_hard_lat)
            print(f"\n  MEAN pos gap (latent)   : {gap:.4f}")

        latent_report = {
            "cross_domain_mean": round(float(np.mean(lat_x_sims)), 4),
            "within_domain_mean": round(float(np.mean(lat_w_sims)), 4),
            "all_positive_mean":  round(float(np.mean(lat_all)), 4),
            "all_positive_std":   round(float(np.std(lat_all)), 4),
            "all_positive_var":   round(float(np.var(lat_all)), 4),
            "hard_negative_mean": round(float(np.mean(anchor_hard_lat)), 4),
            "positive_minus_hard": round(float(np.mean(anchor_pos_lat) - np.mean(anchor_hard_lat)), 4),
        }

    # ── Summary ──────────────────────────────────────────────────────────
    if all_sims:
        collapse_flag = bool(np.var(all_sims) < 1e-4)
    else:
        collapse_flag = False

    summary = {
        "token": {
            "cross_domain_mean": round(float(np.mean(x_sims)), 4),
            "within_domain_mean": round(float(np.mean(w_sims)), 4),
            "all_positive_mean":  round(float(np.mean(all_sims)), 4),
            "all_positive_std":   round(float(np.std(all_sims)), 4),
            "all_positive_var":   round(float(np.var(all_sims)), 4),
            "hard_negative_mean": round(float(np.mean(anchor_hard_sims)), 4),
            "positive_minus_hard": round(float(np.mean(anchor_pos_sims) - np.mean(anchor_hard_sims)), 4),
        },
        "latent": latent_report,
        "collapse_likely": collapse_flag,
    }

    print("\n┌─ Collapse indicator")
    print(f"│  collapse_likely = {collapse_flag} "
          f"(all positive sims variance = {summary['token']['all_positive_var']})")

    out_path = os.path.join(PROJECT_ROOT, "phase4_preflight_report.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] {out_path}")
    print("=" * 72)
    return summary


if __name__ == "__main__":
    run_analysis()
