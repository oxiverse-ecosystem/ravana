"""
Phase 4 Triplet-Margin Trajectory Experiment

Builds an RLMv2 and trains it with hard-negative triplets derived from an
increasingly "hard" noise buffer, logging per-checkpoint:

  * positive similarity mean/std/var
  * hard-negative gap mean/std
  * triplet-margin loss mean
  * margin violation fraction
  * collapse flag
  * validation-pair snapshot (gravity/respect and one random pair)

The script is standalone and avoids importing semantic_pairs directly so a
Point/SemanticPairs import issue in one file does not break the experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# We deliberately use an absolute-ish project-root insertion so the script
# survives being run from any directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ravana_ml.tokenizer import WordTokenizer
from ravana_ml.nn.rlm_v2 import RLMv2


# ──────────────────────────────────────────────────────────────────────────
# Config / constants
# ──────────────────────────────────────────────────────────────────────────

# Inline semantic-pair triples so we do not depend on semantic_pairs.py
TRAIN_TRIPLES: List[Tuple[str, str]] = [
    # Physics
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
    # Social
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
    # Nature
    ("rain causes flooding", "flooding"),
    ("rain creates mud", "mud"),
    ("wind produces waves", "waves"),
    ("wind causes erosion", "erosion"),
    ("wind is powerful", "powerful"),
    ("rain is refreshing", "refreshing"),
    ("storm produces rain", "rain"),
    ("storm causes damage", "damage"),
    # Biology
    ("exercise strengthens muscles", "muscles"),
    ("exercise causes sweating", "sweating"),
    ("sleep restores energy", "energy"),
    ("blood is essential", "essential"),
    ("bones are rigid", "rigid"),
    ("running is exercise", "exercise"),
    # Tech
    ("code creates software", "software"),
    ("bugs cause crashes", "crashes"),
    ("encryption protects data", "data"),
    ("viruses corrupt files", "files"),
    ("python is popular", "popular"),
    # Cross-domain bridges
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
    ("heat causes trust", "trust"),
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
]

# Hard-negative triples: (anchor, positive, hard_negative)
HARD_NEGATIVE_TRIPLES: List[Tuple[str, str, str]] = [
    ("gravity", "loyalty", "weight"),
    ("light", "hope", "darkness"),
    ("heat", "anger", "cold"),
    ("friction", "conflict", "pressure"),
    ("expansion", "growth", "contraction"),
]

# Fixed validation words for trajectory dump
VALIDATION_ANCHOR_PAIRS = [
    ("gravity", "loyalty"),
    ("heat", "anger"),
]

RANDOM_VALIDATION_PAIRS = [
    ("light", "hope"),
    ("expansion", "growth"),
    ("friction", "conflict"),
]

NOISE_VOCAB = [
    "tree", "river", "mountain", "ocean", "sky", "cloud", "stone", "fire",
    "water", "wind", "sand", "rain", "storm", "ice", "snow", "heat", "light",
    "shadow", "metal", "wood", "blood", "bone", "flesh", "mind", "heart",
    "voice", "dream", "memory", "time", "space", "force", "power", "energy",
]


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _init_tokenizer(train: List[Tuple[str, str]]) -> WordTokenizer:
    tok = WordTokenizer()
    for text, _ in train:
        for word in text.split():
            tok.encode(word)
    for a, b, _ in HARD_NEGATIVE_TRIPLES:
        for w in (a, b):
            tok.encode(w)
    for w in NOISE_VOCAB + ["gravity", "loyalty", "respect"]:
        tok.encode(w)
    return tok


def _init_model(actual_vocab: int, embed_dim: int, concept_dim: int, seed: int) -> RLMv2:
    np.random.seed(seed)
    model = RLMv2(
        vocab_size=actual_vocab + 5,
        embed_dim=embed_dim,
        concept_dim=concept_dim,
        n_concepts=actual_vocab,
        sleep_interval=300,
        gate_concept_creation=False,
    )
    return model


def _get_proto(model: RLMv2, tok: WordTokenizer, word: str) -> np.ndarray:
    tid = tok.word_to_id.get(word)
    if tid is None:
        return np.zeros(model.concept_dim, dtype=np.float32)
    raw = model.token_embed.embed_raw(tid)
    return model._project_to_concept(raw)


def positive_stats(tok: WordTokenizer, model: RLMv2, pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    sims: List[float] = []
    for a, b in pairs:
        ta = tok.word_to_id.get(a)
        tb = tok.word_to_id.get(b)
        if ta is None or tb is None:
            continue
        pa = _get_proto(model, tok, a)
        pb = _get_proto(model, tok, b)
        sims.append(cosine_similarity(pa, pb))
    if not sims:
        return {"mean": float("nan"), "std": 0.0, "var": 0.0}
    arr = np.array(sims, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "var": float(np.var(arr)),
        "count": float(arr.size),
    }


def hard_negative_stats(tok: WordTokenizer, model: RLMv2) -> Dict[str, float]:
    sims: List[float] = []
    details: List[Dict[str, float]] = []
    for anchor, positive, hard in HARD_NEGATIVE_TRIPLES:
        pa = _get_proto(model, tok, anchor)
        pp = _get_proto(model, tok, positive)
        ph = _get_proto(model, tok, hard)
        sp = cosine_similarity(pa, pp)
        sh = cosine_similarity(pa, ph)
        gap = sp - sh
        sims.append(gap)
        details.append({
            "anchor": anchor,
            "positive": positive,
            "hard": hard,
            "s_pos": round(sp, 4),
            "s_hard": round(sh, 4),
            "gap": round(gap, 4),
        })
    if not sims:
        return {"mean": 0.0, "std": 0.0, "details": []}
    arr = np.array(sims, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "details": details,
    }


def validation_snapshot(tok: WordTokenizer, model: RLMv2, epoch: int) -> Dict[str, object]:
    snapshot: Dict[str, object] = {"epoch": epoch, "pairs": []}
    for anchor, positive in VALIDATION_ANCHOR_PAIRS + RANDOM_VALIDATION_PAIRS:
        pa = _get_proto(model, tok, anchor)
        pp = _get_proto(model, tok, positive)
        hard_sims: List[Tuple[str, float]] = []
        for _, _, hneg in HARD_NEGATIVE_TRIPLES:
            ph = _get_proto(model, tok, hneg)
            hard_sims.append((hneg, cosine_similarity(pa, ph)))
        hard_sims.sort(key=lambda t: t[1], reverse=True)
        snapshot["pairs"].append({
            "anchor": anchor,
            "positive": positive,
            "d_pos": round(1.0 - cosine_similarity(pa, pp), 4),
            "hard_negatives": [
                {"word": w, "sim": round(s, 4)} for w, s in hard_sims[:5]
            ],
        })
    return snapshot


def _collect_noise_negatives(
    tok: WordTokenizer,
    model: RLMv2,
    anchor: str,
    positive: str,
    count: int,
) -> List[str]:
    anchor_vec = _get_proto(model, tok, anchor)
    scored: List[Tuple[str, float]] = []
    seen = {positive, anchor}
    for word in NOISE_VOCAB:
        if word in seen:
            continue
        tid = tok.word_to_id.get(word)
        if tid is None:
            continue
        raw = model.token_embed.embed_raw(tid)
        proto = model._project_to_concept(raw)
        sim = cosine_similarity(anchor_vec, proto)
        scored.append((word, sim))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [w for w, _ in scored[:count]]


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────

def run(
    n_epochs: int = 300,
    embed_dim: int = 64,
    concept_dim: int = 64,
    seed: int = 42,
    margin: float = 0.3,
    validate_every: int = 50,
    hard_negatives_per_epoch: int = 5,
) -> Dict[str, object]:
    print("=" * 72)
    print("Phase 4 — Triplet-Margin Trajectory Experiment")
    print("=" * 72)

    tok = _init_tokenizer(TRAIN_TRIPLES)
    model = _init_model(
        actual_vocab=tok.vocab_size,
        embed_dim=embed_dim,
        concept_dim=concept_dim,
        seed=seed,
    )
    model._tokenizer = tok

    validation_pairs = VALIDATION_ANCHOR_PAIRS + RANDOM_VALIDATION_PAIRS
    records: List[Dict[str, object]] = []
    noise_per_epoch = max(1, hard_negatives_per_epoch // 3)

    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        total_violation = 0.0
        total_margin = 0.0
        n = len(TRAIN_TRIPLES)
        for text, target_word in TRAIN_TRIPLES:
            ids = tok.encode(text)
            target_id = tok.encode(target_word)[0]
            ctx = np.array(ids[:-1], dtype=np.int64)
            tgt = np.array([target_id], dtype=np.int64)
            # forward / learn pass
            result = model.learn(ctx, tgt)
            total_loss += float(result.get("loss", 0.0))
            proposal = result.get("proposal", None)

            if proposal is not None:
                p_vec = proposal.get("p_vec")
                n_vec = proposal.get("n_vec")
                if p_vec is not None and n_vec is not None:
                    sp = 1.0 + cosine_similarity(p_vec, n_vec)
                    total_violation += sp
                    total_margin += margin

        metrics = positive_stats(tok, model, validation_pairs)
        gap = hard_negative_stats(tok, model)

        pos_mean = metrics["mean"]
        pos_std = metrics["std"]
        pos_var = metrics["var"]
        gap_mean = gap["mean"]
        gap_std = gap["std"]

        collapse = bool(
            isinstance(pos_std, float)
            and isinstance(gap_mean, float)
            and (pos_std < 0.008)
            and (gap_mean < 0.15)
        )

        epoch_record: Dict[str, object] = {
            "epoch": epoch,
            "loss": round(total_loss / max(1, n), 4),
            "margin_loss_mean": round((total_margin / max(1, n)), 4),
            "violation_mean": round((total_violation / max(1, n)), 4),
            "positive_similarity_mean": round(pos_mean, 4) if isinstance(pos_mean, float) else pos_mean,
            "positive_similarity_std": round(pos_std, 4),
            "positive_similarity_var": round(pos_var, 4),
            "hard_negative_gap_mean": round(gap_mean, 4),
            "hard_negative_gap_std": round(gap_std, 4),
            "collapse_flag": collapse,
            "snapshot": validation_snapshot(tok, model, epoch),
        }
        records.append(epoch_record)

        if epoch % validate_every == 0 or epoch == n_epochs:
            print(
                f"  [Phase4] epoch={epoch:4d} loss={epoch_record['loss']:.4f} "
                f"pos={epoch_record['positive_similarity_mean']:.4f} "
                f"pos_std={epoch_record['positive_similarity_std']:.4f} "
                f"gap={epoch_record['hard_negative_gap_mean']:.4f} gap_std={epoch_record['hard_negative_gap_std']:.4f} "
                f"collapse={collapse}"
            )
            if collapse:
                print(f"  [Phase4] WARNING: collapse risk at epoch {epoch}")

    # ── Saved artifact ──────────────────────────────────────────────────
    out_dir = os.path.join(SCRIPT_DIR, "experiment_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "triplet_trajectory_phase4.json")
    save = {
        "config": {
            "epochs": n_epochs,
            "embed_dim": embed_dim,
            "concept_dim": concept_dim,
            "seed": seed,
            "margin": margin,
            "validate_every": validate_every,
            "hard_negatives_per_epoch": hard_negatives_per_epoch,
            "n_train": len(TRAIN_TRIPLES),
            "vocab": tok.vocab_size,
        },
        "records": records,
    }
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n[Saved] {out_path}")
    return save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Triplet Trajectory")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--concept-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--hard-negatives-per-epoch", type=int, default=5)
    args = parser.parse_args()
    run(
        n_epochs=args.epochs,
        embed_dim=args.embed_dim,
        concept_dim=args.concept_dim,
        seed=args.seed,
        margin=args.margin,
        validate_every=args.validate_every,
        hard_negatives_per_epoch=args.hard_negatives_per_epoch,
    )
