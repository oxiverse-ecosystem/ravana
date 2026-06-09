"""
Smoke test: verify encoder weights actually move with a direct
latent-space triplet-margin gradient on a single triple.
"""
from __future__ import annotations

import os, sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from ravana_ml.tokenizer import WordTokenizer
from ravana_ml.nn.rlm_v2 import RLMv2


_REQUIRED = ["heat", "anger", "cold", "friction", "conflict", "water", "stress", "muscles"]


def main() -> int:
    tok = WordTokenizer()
    for w in _REQUIRED:
        for _ in range(3):
            tok.encode(w)
    model = RLMv2(
        vocab_size=tok.vocab_size + 5,
        embed_dim=64,
        concept_dim=64,
        n_concepts=tok.vocab_size,
        sleep_interval=999,
        gate_concept_creation=False,
    )
    model._tokenizer = tok

    def proj(word: str) -> np.ndarray:
        tid = tok.word_to_id[word]
        return model._project_to_concept(model.token_embed.embed_raw(tid))

    # Snapshot before update
    w1_before = model._enc_W1.copy()
    w2_before = model._enc_W2.copy()
    b1_before = model._enc_b1.copy()
    b2_before = model._enc_b2.copy()

    # Triple: anchor=heat, positive=anger, hard_neg=cold
    a_emb = model.token_embed.weight.data[tok.word_to_id["heat"]].copy()
    p_emb = model.token_embed.weight.data[tok.word_to_id["anger"]].copy()
    h_emb = model.token_embed.weight.data[tok.word_to_id["cold"]].copy()

    lat_a, z1_a, h1_a, _ = model._encoder_forward_full(a_emb)
    lat_p, *_ , _ = model._encoder_forward_full(p_emb)
    lat_h, *_ , _ = model._encoder_forward_full(h_emb)

    # Latent squared Euclidean distances
    d_pos = float(np.dot(lat_a - lat_p, lat_a - lat_p))
    d_neg = float(np.dot(lat_a - lat_h, lat_a - lat_h))
    margin = 0.3
    loss = max(0.0, d_pos - d_neg + margin)
    print(f"d_pos={d_pos:.5f} d_neg={d_neg:.5f} loss={loss:.5f}")

    if loss > 0:
        # Gradient points toward positive and away from hardest negative.
        # We approximate hardest by using cold only for this smoke test.
        hardest = lat_h
        d_lat = ((lat_a - lat_p) - (lat_a - hardest)) * loss          # (,latent)
        d_lat = d_lat.reshape(1, -1)
        z1_a = z1_a.reshape(1, -1)
        h1_a = h1_a.reshape(1, -1)
        # Backprop must be called on the inputs that generated these activations,
        # with the correct intermediate cache. Use fresh reencoded a_emb to be safe.
        a_for_bp = model.token_embed.weight.data[tok.word_to_id["heat"]].reshape(1, -1)
        z1_a_bp, h1_a_bp, *_ = model._encoder_forward_full(a_for_bp[0])
        z1_a_bp = z1_a_bp.reshape(1, -1)
        h1_a_bp = h1_a_bp.reshape(1, -1)
        dW1, db1, dW2, db2 = model._encoder_backward(
            a_for_bp, z1_a_bp, h1_a_bp, z1_a_bp, z1_a_bp, d_lat
        )
        lr = 0.01
        model._enc_W1 -= lr * dW1
        model._enc_b1 -= lr * db1
        model._enc_W2 -= lr * dW2
        model._enc_b2 -= lr * db2

    # Snapshot after update
    w1_after = model._enc_W1.copy()
    w2_after = model._enc_W2.copy()
    b1_after = model._enc_b1.copy()
    b2_after = model._enc_b2.copy()

    print("W1 delta norm:", float(np.linalg.norm(w1_after - w1_before)))
    print("W2 delta norm:", float(np.linalg.norm(w2_after - w2_before)))
    print("b1 delta norm:", float(np.linalg.norm(b1_after - b1_before)))
    print("b2 delta norm:", float(np.linalg.norm(b2_after - b2_before)))

    # Effect on projections
    def gap(a, b, h):
        pa, pp, ph = proj(a), proj(b), proj(h)
        s_pp = float(np.dot(pa, pp) / (np.linalg.norm(pa) * np.linalg.norm(pp) + 1e-9))
        s_ph = float(np.dot(pa, ph) / (np.linalg.norm(pa) * np.linalg.norm(ph) + 1e-9))
        return s_pp, s_ph

    s_pp0, s_ph0 = gap("heat", "anger", "cold")
    s_pp1, s_ph1 = gap("heat", "anger", "cold")
    print("before gap:", round(s_pp0 - s_ph0, 4), "after gap:", round(s_pp1 - s_ph1, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
