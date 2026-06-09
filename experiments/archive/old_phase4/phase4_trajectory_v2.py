"""
Minimal Phase 4 trajectory: triplet-margin training loop with
per-epoch hard-negative gap and collapse reporting.
No dependency on the broken `result.get("proposal")` path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ravana_ml.tokenizer import WordTokenizer
from ravana_ml.nn.rlm_v2 import RLMv2


# Fixed validation set (anchor, positive, hard_negatives)
TRIPLES = [
    ("gravity", "loyalty", ["weight", "darkness", "pressure", "cold", "contraction"]),
    ("light", "hope", ["darkness", "weight", "anger", "pressure", "cold"]),
    ("heat", "anger", ["cold", "pressure", "darkness", "weight", "contraction"]),
    ("friction", "conflict", ["pressure", "contraction", "darkness", "weight", "cold"]),
    ("expansion", "growth", ["contraction", "pressure", "darkness", "weight", "cold"]),
]

TRAIN_TEXTS = [
    ("heat causes expansion", "expansion"),
    ("kindness causes trust", "trust"),
    ("rain causes flooding", "flooding"),
    ("exercise strengthens muscles", "muscles"),
    ("code creates software", "software"),
    ("ice is cold", "cold"),
    ("trust is valuable", "valuable"),
    ("wind is powerful", "powerful"),
    ("blood is essential", "essential"),
    ("python is popular", "popular"),
    ("love causes trust", "trust"),
    ("anger produces conflict", "conflict"),
]


def init_tokenizer() -> WordTokenizer:
    tok = WordTokenizer()
    for text, target in TRAIN_TEXTS:
        for word in text.split():
            tok.encode(word)
        tok.encode(target)
    for anchor, positive, hards in TRIPLES:
        for w in [anchor, positive] + hards:
            tok.encode(w)
    return tok


def init_model(vocab: int, embed_dim: int = 64, concept_dim: int = 64, seed: int = 42) -> RLMv2:
    np.random.seed(seed)
    return RLMv2(
        vocab_size=vocab + 5,
        embed_dim=embed_dim,
        concept_dim=concept_dim,
        n_concepts=vocab,
        sleep_interval=300,
        gate_concept_creation=False,
    )


def proto(model: RLMv2, tok: WordTokenizer, word: str) -> np.ndarray:
    tid = tok.word_to_id.get(word)
    if tid is None:
        return np.zeros(model.concept_dim, dtype=np.float32)
    raw = model.token_embed.embed_raw(tid)
    return model._project_to_concept(raw)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def similarity_surface(model: RLMv2, tok: WordTokenizer, anchor: str, positive: str) -> float:
    return cosine(proto(model, tok, anchor), proto(model, tok, positive))


def hard_gap_surface(model: RLMv2, tok: WordTokenizer, anchor: str, positives: list[str], hards: list[str]) -> dict:
    pa = proto(model, tok, anchor)
    pos_sims = [cosine(pa, proto(model, tok, p)) for p in positives]
    hard_sims = [cosine(pa, proto(model, tok, h)) for h in hards]
    mean_pos = float(np.mean(pos_sims)) if pos_sims else float("nan")
    mean_gap = float(np.mean(pos_sims) - np.mean(hard_sims)) if pos_sims and hard_sims else float("nan")
    return {
        "mean_pos": mean_pos,
        "mean_gap": mean_gap,
        "pos_sims": [round(float(x), 4) for x in pos_sims],
        "hard_sims": [round(float(x), 4) for x in hard_sims],
    }


def encoder_gradients(
    model: RLMv2,
    tok: WordTokenizer,
    anchor_word: str,
    positive_word: str,
    hard_words: list[str],
    margin: float = 0.3,
) -> tuple[float, dict]:
    # Project all candidate words once
    anchor_tid = tok.word_to_id[anchor_word]
    positive_tid = tok.word_to_id[positive_word]
    hard_tids = [tok.word_to_id[w] for w in hard_words if w in tok.word_to_id]
    if not hard_tids:
        return 0.0, {"violation": 0.0}

    a_emb = model.token_embed.weight.data[anchor_tid]
    p_emb = model.token_embed.weight.data[positive_tid]
    hard_embs = [model.token_embed.weight.data[tid] for tid in hard_tids]

    # Forward through encoder
    lat_a, z1_a, h1_a, z2_a = model._encoder_forward_full(a_emb)
    lat_p, z1_p, h1_p, z2_p = model._encoder_forward_full(p_emb)
    lats_h = [model._encoder_forward_full(h)[0] for h in hard_embs]

    # Distances in latent: squared Euclidean here (consistent with margin).
    d_pos = float(np.sum((lat_a - lat_p) ** 2))

    # Choose hardest negative: largest similarity => smallest latent distance.
    hard_penalties = []
    for lh in lats_h:
        d = float(np.sum((lat_a - lh) ** 2))
        hard_penalties.append(d)
    hard_dist = float(np.max(hard_penalties)) if hard_penalties else 0.0

    loss = d_pos - hard_dist + margin
    loss = max(0.0, loss)
    violation = int(loss > 1e-9)

    # Backprop toward positive, away from all hard negatives
    d_lat = (lat_a - lat_p) - (lat_a - lats_h[np.argmax(hard_penalties)])  # shape (latent,)
    d_lat = d_lat * (loss > 0)
    # Prop back into encoder weights
    dW1a, db1a, dW2a, db2a = model._encoder_backward(
        a_emb[np.newaxis, :], z1_a[np.newaxis, :], h1_a[np.newaxis, :], z2_a[np.newaxis, :],
        lat_a[np.newaxis, :], d_lat[np.newaxis, :],
    )
    # Use a small safe LR for encoder; only these params move.
    lr = 0.01
    model._enc_W1 -= lr * dW1a
    model._enc_b1 -= lr * db1a
    model._enc_W2 -= lr * dW2a
    model._enc_b2 -= lr * db2a

    detail = {
        "d_pos": round(d_pos, 5),
        "hard_dist": round(hard_dist, 5),
        "violation": violation,
    }
    return loss, detail


def run(
    epochs: int = 300,
    margin: float = 0.3,
    seed: int = 42,
    validate_every: int = 50,
    embed_dim: int = 64,
    concept_dim: int = 64,
) -> dict:
    tok = init_tokenizer()
    vocab = tok.vocab_size
    model = init_model(vocab=vocab, embed_dim=embed_dim, concept_dim=concept_dim, seed=seed)
    model._tokenizer = tok

    records: list[dict] = []

    for epoch in range(1, epochs + 1):
        total_margin_loss = 0.0
        total_violation = 0
        used = 0
        for anchor, positive, hards in TRIPLES:
            # Draw up to one fresh alternate negative from noise pool per epoch to keep hard set varying
            noise_pool = [
                "river", "mountain", "ocean", "sky", "cloud", "stone", "metal",
                "wood", "blood", "bone", "mind", "voice", "dream", "memory", "time",
            ]
            if epoch % 7 == 0:
                extra = noise_pool[(epoch + hash(anchor)) % len(noise_pool)]
                hards = list(hards)
                hards[0] = extra
            loss, detail = encoder_gradients(
                model=model,
                tok=tok,
                anchor_word=anchor,
                positive_word=positive,
                hard_words=hards,
                margin=margin,
            )
            total_margin_loss += loss
            total_violation += detail["violation"]
            used += 1

        surface_stats = {}
        for anchor, positive, hards in TRIPLES:
            surface_stats[anchor] = hard_gap_surface(model, tok, anchor, [positive], hards)

        pos_means = [v["mean_pos"] for v in surface_stats.values() if not np.isnan(v["mean_pos"])]
        gap_means = [v["mean_gap"] for v in surface_stats.values() if not np.isnan(v["mean_gap"])]
        pos_mean = round(float(np.mean(pos_means)), 4) if pos_means else float("nan")
        pos_std = round(float(np.std(pos_means)), 4) if pos_means else 0.0
        gap_mean = round(float(np.mean(gap_means)), 4) if gap_means else float("nan")
        gap_std = round(float(np.std(gap_means)), 4) if gap_means else 0.0
        collapse = bool(isinstance(pos_std, float) and isinstance(gap_mean, float) and (pos_std < 0.008) and (gap_mean < 0.15))

        rec = {
            "epoch": epoch,
            "margin_loss": round(total_margin_loss / max(1, used), 5),
            "violation_count": total_violation,
            "positive_similarity_mean": pos_mean,
            "positive_similarity_std": pos_std,
            "hard_negative_gap_mean": gap_mean,
            "hard_negative_gap_std": gap_std,
            "collapse_flag": collapse,
            "surface": surface_stats,
        }
        records.append(rec)

        if epoch == 1 or epoch % validate_every == 0 or epoch == epochs:
            print(
                f"  [Phase4] epoch={epoch:4d} margin_loss={rec['margin_loss']:.5f} "
                f"violations={total_violation}/{used} pos={pos_mean:.4f} ±{pos_std:.4f} "
                f"gap={gap_mean:.4f} ±{gap_std:.4f} collapse={collapse}"
            )
            if collapse:
                print(f"  [Phase4] WARNING: collapse risk at epoch {epoch}")

    out = {
        "config": {
            "epochs": epochs,
            "margin": margin,
            "seed": seed,
            "validate_every": validate_every,
            "embed_dim": embed_dim,
            "concept_dim": concept_dim,
            "vocab": vocab,
        },
        "records": records,
    }
    out_dir = os.path.join(SCRIPT_DIR, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "trajectory_phase4.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Saved] {out_path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validate-every", type=int, default=50)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--concept-dim", type=int, default=64)
    args = ap.parse_args()
    run(
        epochs=args.epochs,
        margin=args.margin,
        seed=args.seed,
        validate_every=args.validate_every,
        embed_dim=args.embed_dim,
        concept_dim=args.concept_dim,
    )
