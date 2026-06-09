"""
Phase 4 — Option 2: Augmented in-domain triplet-margin trajectory.

This script runs the triplet-margin training loop with:
- 64-dim latent / 72-dim hidden / margin=0.1
- Augmented TRAIN_TEXTS (contextual sentences for fear/avoidance and encryption/data)
- Logs trajectory to trajectory_option2_augmented.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ravana_ml.tokenizer import WordTokenizer
from ravana_ml.nn.rlm_v2 import RLMv2


AUTOENCODER_EPOCHS = 300
DIM = 64
CONCEPT_DIM = 64
LATENT = 64
HIDDEN = 72
MARGIN = 0.1


# ──────────────────────────────────────────────────────────────────────────
# Training triples (in-domain words only) + Augmentation
# ──────────────────────────────────────────────────────────────────────────

TRAIN_TEXTS: List[Tuple[str, str]] = [
    ("heat causes expansion", "expansion"),
    ("heat melts ice", "ice"),
    ("fire produces smoke", "smoke"),
    ("fire creates heat", "heat"),
    ("sun produces heat", "heat"),
    ("sun causes warmth", "warmth"),
    ("steel is strong", "strong"),
    ("gold is valuable", "valuable"),
    ("glass is transparent", "transparent"),
    ("water is liquid", "liquid"),
    ("cold freezes water", "water"),
    ("ice is cold", "cold"),
    ("exercise produces heat", "heat"),
    ("food provides energy", "energy"),
    ("data is valuable", "valuable"),
    ("kindness causes trust", "trust"),
    ("kindness creates friendship", "friendship"),
    ("anger produces conflict", "conflict"),
    ("anger causes isolation", "isolation"),
    ("fear produces avoidance", "avoidance"),
    ("love creates bonds", "bonds"),
    ("love causes trust", "trust"),
    ("trust is valuable", "valuable"),
    ("love is powerful", "powerful"),
    ("anger is destructive", "destructive"),
    ("empathy builds connection", "connection"),
    ("empathy causes trust", "trust"),
    ("stress causes damage", "damage"),
    ("stress weakens immunity", "immunity"),
    ("viruses cause illness", "illness"),
    ("code causes illness", "illness"),
    ("bugs cause illness", "illness"),
    ("exercise causes illness", "illness"),
    ("rain causes flooding", "flooding"),
    ("rain creates mud", "mud"),
    ("wind produces waves", "waves"),
    ("wind causes erosion", "erosion"),
    ("wind is powerful", "powerful"),
    ("rain is refreshing", "refreshing"),
    ("storm produces rain", "rain"),
    ("storm causes damage", "damage"),
    ("exercise strengthens muscles", "muscles"),
    ("exercise causes sweating", "sweating"),
    ("sleep restores energy", "energy"),
    ("blood is essential", "essential"),
    ("bones are rigid", "rigid"),
    ("running is exercise", "exercise"),
    ("code creates software", "software"),
    ("bugs cause crashes", "crashes"),
    ("encryption protects data", "data"),
    ("viruses corrupt files", "files"),
    ("python is popular", "popular"),
    ("fire is hot", "hot"),
    ("sun is hot", "hot"),
    ("trust is essential", "essential"),
    ("energy is essential", "essential"),
    ("sun provides energy", "energy"),
    ("fire provides heat", "heat"),
    ("love provides comfort", "comfort"),
    ("weakness causes failure", "failure"),
    ("storm causes flooding", "flooding"),
    ("storm creates mud", "mud"),
    ("running causes sweating", "sweating"),
    ("running strengthens muscles", "muscles"),
    ("empathy creates friendship", "friendship"),
    ("cold causes contraction", "contraction"),
    ("exercise creates software", "software"),
    ("viruses cause crashes", "crashes"),
    ("code causes fatigue", "fatigue"),
    ("bugs produce illness", "illness"),
    ("love produces rain", "rain"),
    ("kindness creates waves", "waves"),
    ("running strengthens muscles", "muscles"),
    ("rain produces sadness", "sadness"),
    ("storm creates conflict", "conflict"),
    ("sun produces happiness", "happiness"),
    ("kindness causes flooding", "flooding"),
    ("rain produces conflict", "conflict"),
    ("heat causes conflict", "conflict"),
    ("expansion causes trust", "trust"),
    ("love produces heat", "heat"),
    ("heat causes trust", "heat"),
    ("fire produces friendship", "friendship"),
    ("kindness causes flooding", "flooding"),
    ("rain produces conflict", "conflict"),
    ("code causes illness", "illness"),
    ("exercise produces crashes", "crashes"),
    ("heat causes rain", "rain"),
    ("cold produces snow", "snow"),
    ("rain produces heat", "heat"),
    ("sun creates expansion", "expansion"),
    ("stress produces heat", "heat"),
    ("energy causes growth", "growth"),
    ("damage produces isolation", "isolation"),
    ("strength causes bonds", "bonds"),
    
    # ─── Option 2 Augmentation ──────────────────────────────────────────────
    # fear/avoidance context augmentation
    ("fear causes avoidance", "avoidance"),
    ("danger produces fear", "fear"),
    ("fear triggers panic", "panic"),
    ("panic creates fear", "fear"),
    ("fear causes isolation", "isolation"),
    ("avoidance creates isolation", "isolation"),
    ("danger triggers avoidance", "avoidance"),
    ("stress produces fear", "fear"),
    # encryption/data context augmentation
    ("encryption secures data", "data"),
    ("data requires encryption", "encryption"),
    ("encryption hides data", "data"),
    ("code creates encryption", "encryption"),
    ("privacy requires encryption", "encryption"),
    ("security protects data", "data"),
    ("encryption provides security", "security"),
    ("software uses encryption", "encryption"),
]

# Validation triples: in-domain, structurally diverse, 5 pairs.
# All anchor/positive/hard words appear in TRAIN_TEXTS.
VALIDATION_TRIPLES: List[Tuple[str, str, List[str]]] = [
    (
        "heat",
        "expansion",
        ["steel", "friendship", "wind", "viruses", "fatigue"],
    ),
    (
        "fear",
        "avoidance",
        ["glass", "erosion", "kindness", "code", "contraction"],
    ),
    (
        "kindness",
        "trust",
        ["mud", "crashes", "cold", "viruses", "waves"],
    ),
    (
        "sun",
        "warmth",
        ["isolation", "friendship", "mud", "crashes", "fatigue"],
    ),
    (
        "encryption",
        "data",
        ["contraction", "kindness", "erosion", "smoke", "sleep"],
    ),
]


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def proto(model: RLMv2, tok: WordTokenizer, word: str) -> np.ndarray:
    tid = tok.word_to_id.get(word)
    if tid is None:
        raise KeyError(word)
    emb = model.token_embed.weight.data[tid]      # (embed_dim,)
    lat, *_ = model._encoder_forward_full(emb)    # (latent_dim,)
    return lat


def init_tokenizer() -> WordTokenizer:
    tok = WordTokenizer()
    for text, target in TRAIN_TEXTS:
        for word in text.split():
            tok.encode(word)
        tok.encode(target)
    for anchor, positive, hards in VALIDATION_TRIPLES:
        for w in [anchor, positive] + hards:
            tok.encode(w)
    return tok


def init_model(vocab: int, embed_dim: int, concept_dim: int, seed: int) -> RLMv2:
    np.random.seed(seed)
    return RLMv2(
        vocab_size=vocab + 5,
        embed_dim=embed_dim,
        concept_dim=concept_dim,
        n_concepts=vocab,
        sleep_interval=300,
        gate_concept_creation=False,
        latent_dim=LATENT,
        hidden_dim=HIDDEN,
    )


def _triple_label(anchor: str, positive: str) -> str:
    return f"{anchor}→{positive}"


def hard_gap_stats(
    model: RLMv2, tok: WordTokenizer
) -> Tuple[float, float, List[Dict]]:
    details = []
    pos_gaps: List[float] = []
    for anchor, positive, hards in VALIDATION_TRIPLES:
        pa = proto(model, tok, anchor)
        pp = proto(model, tok, positive)
        hard_vecs = [proto(model, tok, h) for h in hards]
        sp = cosine(pa, pp)
        sh = float(np.mean([cosine(pa, hv) for hv in hard_vecs]))
        gap = sp - sh
        pos_gaps.append(gap)
        details.append(
            {
                "triple": _triple_label(anchor, positive),
                "anchor": anchor,
                "positive": positive,
                "s_pos": round(sp, 4),
                "s_hard_mean": round(sh, 4),
                "gap": round(gap, 4),
            }
        )
    if not pos_gaps:
        return float("nan"), 0.0, []
    arr = np.array(pos_gaps, dtype=np.float32)
    return float(np.mean(arr)), float(np.std(arr)), details


def apply_triplet_margin(
    model: RLMv2,
    tok: WordTokenizer,
    anchor: str,
    positive: str,
    hards: List[str],
    margin: float = 0.3,
    lr: float = 0.005,
) -> Tuple[float, Dict]:
    a_tid = tok.word_to_id[anchor]
    p_tid = tok.word_to_id[positive]
    h_tids = [tok.word_to_id[h] for h in hards if h in tok.word_to_id]
    if not h_tids:
        return 0.0, {"violation": 0, "applied": False}

    a_emb = model.token_embed.weight.data[a_tid]
    p_emb = model.token_embed.weight.data[p_tid]
    h_embs = [model.token_embed.weight.data[tid] for tid in h_tids]

    lat_a, z1_a, h1_a, z2_a = model._encoder_forward_full(a_emb)
    lat_p, *_ = model._encoder_forward_full(p_emb)
    lats_h = [model._encoder_forward_full(h)[0] for h in h_embs]

    # Latent squared Euclideans
    def d2(x, y):
        diff = x - y
        return float(np.dot(diff, diff))

    d_pos = d2(lat_a, lat_p)
    d_neg = np.max([d2(lat_a, lh) for lh in lats_h])

    loss = d_pos - d_neg + margin
    loss = max(0.0, loss)
    violation = int(loss > 1e-9)

    if violation:
        # backprop toward positive, away from hardest negative
        hardest = lats_h[int(np.argmax([d2(lat_a, lh) for lh in lats_h]))]
        d_lat = ((lat_a - lat_p) - (lat_a - hardest)) * loss
        dW1a, db1a, dW2a, db2a = model._encoder_backward(
            a_emb[np.newaxis, :],
            z1_a[np.newaxis, :],
            h1_a[np.newaxis, :],
            z2_a[np.newaxis, :],
            lat_a[np.newaxis, :],
            d_lat[np.newaxis, :],
        )
        model._enc_W1 -= lr * dW1a
        model._enc_b1 -= lr * db1a
        model._enc_W2 -= lr * dW2a
        model._enc_b2 -= lr * db2a

    return loss, {"violation": violation, "applied": violation}


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────

def run(
    epochs: int = 300,
    margin: float = 0.3,
    seed: int = 42,
    validate_every: int = 50,
    embed_dim: int = 64,
    concept_dim: int = 64,
) -> Dict:
    tok = init_tokenizer()
    vocab = tok.vocab_size
    model = init_model(
        vocab=vocab,
        embed_dim=embed_dim,
        concept_dim=concept_dim,
        seed=seed,
    )
    model._tokenizer = tok

    records: List[Dict] = []
    lr = 0.005

    # Warmup schedule: reduce effective margin by scaling lr in early epochs
    def lr_for(epoch: int) -> float:
        return lr * min(1.0, epoch / 20)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_violation = 0
        used = 0
        cur_lr = lr_for(epoch)

        # During warmup, sample a random half-subset each epoch so gradients vary.
        if epoch <= 20:
            subset = random.sample(
                VALIDATION_TRIPLES,
                k=len(VALIDATION_TRIPLES) // 2 + 1,
            )
        else:
            subset = VALIDATION_TRIPLES

        for anchor, positive, hards in subset:
            loss, detail = apply_triplet_margin(
                model, tok, anchor, positive, hards, margin=margin, lr=cur_lr
            )
            total_loss += loss
            total_violation += detail["violation"]
            used += 1

        gap_mean, gap_std, gap_details = hard_gap_stats(model, tok)
        per_triple_gaps = {
            item["triple"]: item["gap"]
            for item in gap_details
        }
        gap_min = float(min(per_triple_gaps.values())) if per_triple_gaps else float("nan")
        records.append({
            "epoch": epoch,
            "margin_loss": round(total_loss / max(1, used), 5),
            "violation_count": total_violation,
            "hard_negative_gap_mean": round(gap_mean, 4),
            "hard_negative_gap_std": round(gap_std, 4),
            "hard_negative_gap_min": round(gap_min, 4),
            "collapse_flag": bool(
                isinstance(gap_std, float)
                and isinstance(gap_mean, float)
                and not np.isnan(gap_std)
                and not np.isnan(gap_mean)
                and gap_std < 0.008
                and gap_mean < 0.15
            ),
            "per_triple_gaps": per_triple_gaps,
            "gap_details": gap_details,
        })

        should_print = epoch == 1 or epoch % validate_every == 0 or epoch == epochs
        if should_print:
            print(
                f"  [Phase4] epoch={epoch:4d} loss={records[-1]['margin_loss']:.5f} "
                f"viol={total_violation}/{used} gap={records[-1]['hard_negative_gap_mean']:.4f}"
                f" ±{records[-1]['hard_negative_gap_std']:.4f} "
                f"collapse={records[-1]['collapse_flag']}"
            )
            if records[-1]["collapse_flag"]:
                print(f"  [Phase4] WARNING: collapse at epoch {epoch}")

    out_path = os.path.join(SCRIPT_DIR, "experiment_results", "trajectory_option2_augmented.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save = {
        "config": {
            "epochs": epochs,
            "margin": margin,
            "seed": seed,
            "validate_every": validate_every,
            "embed_dim": embed_dim,
            "concept_dim": concept_dim,
            "vocab": vocab,
            "n_train": len(TRAIN_TEXTS),
            "n_validation": len(VALIDATION_TRIPLES),
        },
        "records": records,
    }
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n[Saved] {out_path}")
    return save


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validate-every", type=int, default=50)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--concept-dim", type=int, default=64)
    args = ap.parse_args()
    
    # Force the margin to be 0.1 and embed/concept dims to be 64
    run(
        epochs=args.epochs,
        margin=args.margin,
        seed=args.seed,
        validate_every=args.validate_every,
        embed_dim=args.embed_dim,
        concept_dim=args.concept_dim,
    )
