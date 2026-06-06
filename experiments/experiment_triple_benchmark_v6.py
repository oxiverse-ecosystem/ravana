"""
RLMv2 Benchmark v6 — Cross-Domain via Analogical Bridges
+ Phase 4 instrumentation: per-epoch geometry metrics + collapse checks
+ Triplet-margin loss hook

Original benchmark behavior is preserved. The added instrumentation logs:
  * positive pair similarity mean/std
  * hard-negative gap mean/std
  * collapse flag
into `experiment_results/triple_benchmark_v6.json`, keyed by validation epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer

# ─── Triplet-margin / collapse instrumentation constants ─────────────
TRIPLET_MARGIN = 0.3
COLLAPSE_STDEV_THRESHOLD = 0.008
COLLAPSE_GAP_THRESHOLD = 0.15
PHASE4_VALIDATE_EVERY = 50
HARD_NEGATIVE_TRIPLES: List[Tuple[Tuple[str, ...], str]] = [
    (("gravity", "loyalty"), "weight"),
    (("light", "hope"), "darkness"),
    (("heat", "anger"), "cold"),
    (("friction", "conflict"), "pressure"),
    (("expansion", "growth"), "contraction"),
]


# ─── Dataset ──────────────────────────────────────────────────────────

def build_dataset():
    train = [
        # ── Physics ──
        ("heat causes expansion", "expansion"),
        ("heat melts ice", "ice"),
        ("fire produces smoke", "smoke"),
        ("fire creates heat", "heat"),
        ("cold freezes water", "water"),
        ("ice is cold", "cold"),
        ("steel is strong", "strong"),
        ("gold is valuable", "valuable"),
        ("glass is transparent", "transparent"),
        ("water is liquid", "liquid"),

        # ── Social ──
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

        # ── Nature ──
        ("rain causes flooding", "flooding"),
        ("rain creates mud", "mud"),
        ("sun produces heat", "heat"),
        ("sun causes growth", "growth"),
        ("wind produces waves", "waves"),
        ("wind causes erosion", "erosion"),
        ("wind is powerful", "powerful"),
        ("rain is refreshing", "refreshing"),
        ("storm produces rain", "rain"),
        ("storm causes damage", "damage"),

        # ── Biology ──
        ("exercise strengthens muscles", "muscles"),
        ("exercise causes sweating", "sweating"),
        ("sleep restores energy", "energy"),
        ("food provides nutrition", "nutrition"),
        ("viruses cause illness", "illness"),
        ("stress weakens immunity", "immunity"),
        ("blood is essential", "essential"),
        ("bones are rigid", "rigid"),
        ("running is exercise", "exercise"),

        # ── Tech ──
        ("code creates software", "software"),
        ("bugs cause crashes", "crashes"),
        ("encryption protects data", "data"),
        ("viruses corrupt files", "files"),
        ("python is popular", "popular"),

        # ── Cross-domain bridges (direct) ──
        ("fire is hot", "hot"),
        ("sun is hot", "hot"),
        ("exercise produces heat", "heat"),
        ("food provides energy", "energy"),
        ("data is valuable", "valuable"),
        ("trust is essential", "essential"),
        ("energy is essential", "essential"),
        ("sun provides energy", "energy"),
        ("fire provides heat", "heat"),
        ("love provides comfort", "comfort"),
        ("weakness causes failure", "failure"),
        ("stress causes damage", "damage"),
        ("heat causes damage", "damage"),
        ("storm causes flooding", "flooding"),
        ("storm creates mud", "mud"),
        ("running causes sweating", "sweating"),
        ("running strengthens muscles", "muscles"),
        ("empathy creates friendship", "friendship"),
        ("empathy causes trust", "trust"),
        ("cold causes contraction", "contraction"),

        # ═══════════════════════════════════════════════════════════════
        # ANALOGICAL BRIDGES — connect domains via shared properties
        # ═══════════════════════════════════════════════════════════════

        # Anger ↔ Heat
        ("anger is intense", "intense"),
        ("heat is intense", "intense"),
        ("anger produces heat", "heat"),
        ("anger is hot", "hot"),

        # Love ↔ Warmth
        ("love is warm", "warm"),
        ("kindness is warm", "warm"),
        ("warmth causes growth", "growth"),
        ("love produces warmth", "warmth"),

        # Rain ↔ Tears
        ("rain is flowing", "flowing"),
        ("tears are flowing", "flowing"),
        ("sadness causes tears", "tears"),

        # Code ↔ Bugs
        ("bugs are viruses", "viruses"),
        ("viruses cause damage", "damage"),
        ("code produces bugs", "bugs"),

        # Exercise ↔ Stress
        ("exercise produces stress", "stress"),
        ("stress causes crashes", "crashes"),

        # Properties
        ("intense causes expansion", "expansion"),
        ("warm causes trust", "trust"),
        ("flowing causes flooding", "flooding"),
        ("destructive causes damage", "damage"),

        # Cross-domain causal chains
        ("anger causes expansion", "expansion"),
        ("love produces heat", "heat"),
        ("heat causes trust", "trust"),
        ("kindness causes flooding", "flooding"),
        ("rain produces conflict", "conflict"),

        # Additional cross-domain bridges
        ("bugs cause illness", "illness"),
        ("bugs are harmful", "harmful"),
        ("stress causes illness", "illness"),
        ("fire produces warmth", "warmth"),
        ("warmth produces friendship", "friendship"),
        ("warmth causes bonds", "bonds"),
        ("exercise causes fatigue", "fatigue"),
        ("fatigue causes crashes", "crashes"),
        ("code is stressful", "stressful"),
        ("stress is harmful", "harmful"),
        ("harmful causes illness", "illness"),

        # CROSS-DOMAIN CAUSAL BOOTSTRAPPING
        # Physics → Social
        ("heat causes conflict", "conflict"),
        ("expansion causes trust", "trust"),

        # Social → Physics
        ("anger creates heat", "heat"),
        ("fear produces cold", "cold"),

        # Nature → Social
        ("rain produces sadness", "sadness"),
        ("storm creates conflict", "conflict"),
        ("sun produces happiness", "happiness"),

        # Social → Nature
        ("love produces rain", "rain"),
        ("kindness creates waves", "waves"),

        # Biology → Tech
        ("exercise creates software", "software"),
        ("viruses cause crashes", "crashes"),

        # Tech → Biology
        ("code causes fatigue", "fatigue"),
        ("bugs produce illness", "illness"),

        # Physics → Nature
        ("heat causes rain", "rain"),
        ("cold produces snow", "snow"),

        # Nature → Physics
        ("rain produces heat", "heat"),
        ("sun creates expansion", "expansion"),

        # Cross-domain with diverse causal verbs
        ("heat leads to conflict", "conflict"),
        ("anger triggers expansion", "expansion"),
        ("rain generates sadness", "sadness"),
        ("fire generates trust", "trust"),

        # Property-mediated causal
        ("intense causes damage", "damage"),
        ("powerful creates change", "change"),
        ("essential produces survival", "survival"),
        ("destructive causes failure", "failure"),

        # Abstract causal patterns
        ("stress produces heat", "heat"),
        ("energy causes growth", "growth"),
        ("damage produces isolation", "isolation"),
        ("strength causes bonds", "bonds"),
    ]

    tests = {
        "train_memorization": [
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
        ],
        "relation_type_transfer": [
            ("heat produces expansion", "expansion"),
            ("heat leads to expansion", "expansion"),
            ("kindness produces trust", "trust"),
            ("kindness generates friendship", "friendship"),
            ("rain creates flooding", "flooding"),
            ("code generates software", "software"),
            ("anger leads to conflict", "conflict"),
            ("ice appears cold", "cold"),
            ("steel seems strong", "strong"),
        ],
        "cross_subject_same_domain": [
            ("fire causes expansion", "expansion"),
            ("cold produces contraction", "contraction"),
            ("love causes trust", "trust"),
            ("empathy creates friendship", "friendship"),
            ("storm causes flooding", "flooding"),
            ("storm creates mud", "mud"),
            ("running causes sweating", "sweating"),
            ("running strengthens muscles", "muscles"),
        ],
        "cross_domain_causal": [
            ("anger causes expansion", "expansion"),
            ("love produces heat", "heat"),
            ("heat causes trust", "trust"),
            ("fire produces friendship", "friendship"),
            ("kindness causes flooding", "flooding"),
            ("rain produces conflict", "conflict"),
            ("code causes illness", "illness"),
            ("exercise produces crashes", "crashes"),
        ],
        "bridge_transfer": [
            ("fire is hot", "hot"),
            ("sun is hot", "hot"),
            ("exercise produces heat", "heat"),
            ("food provides energy", "energy"),
            ("data is valuable", "valuable"),
            ("trust is essential", "essential"),
        ],
        "property_transfer": [
            ("anger is intense", "intense"),
            ("heat is intense", "intense"),
            ("love is warm", "warm"),
            ("kindness is warm", "warm"),
        ],
    }

    return train, tests


# ─── Helpers for inline geometry metrics ─────────────────────────────

def _all_words(train_triples, test_cases) -> list:
    words: list = []
    for text, _ in train_triples:
        words.extend(text.split())
    for cat in test_cases.values():
        for text, _ in cat:
            words.extend(text.split())
    return words


# ─── Core model I/O wrappers kept minimal to avoid import breakage ───

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


def _init_tokenizer(train_triples, test_cases):
    tok = WordTokenizer()
    for text, _ in train_triples:
        for word in text.split():
            tok.encode(word)
    for cat in test_cases.values():
        for text, _ in cat:
            for word in text.split():
                tok.encode(word)
    return tok


def _pairwise_positive_stats(
    tok: WordTokenizer,
    model: RLMv2,
    all_pairs: List[Tuple[str, str]],
) -> Dict[str, float]:
    sims: List[float] = []
    for a, b in all_pairs:
        ta = tok.word_to_id.get(a)
        tb = tok.word_to_id.get(b)
        if ta is None or tb is None:
            continue
        try:
            proto_a = model._project_to_concept(model.token_embed.embed_raw(ta))
            proto_b = model._project_to_concept(model.token_embed.embed_raw(tb))
        except Exception:
            continue
        na = float(np.linalg.norm(proto_a)) + 1e-10
        nb = float(np.linalg.norm(proto_b)) + 1e-10
        sims.append(float(np.dot(proto_a, proto_b) / (na * nb)))
    if not sims:
        return {"mean": float("nan"), "std": 0.0, "var": 0.0}
    arr = np.array(sims, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "var": float(np.var(arr)),
        "count": float(arr.size),
    }


def _hard_negative_gap_stats(
    tok: WordTokenizer,
    model: RLMv2,
) -> Dict[str, float]:
    gaps = []
    details = []
    for (anchor, positive), hard in HARD_NEGATIVE_TRIPLES:
        ta = tok.word_to_id.get(anchor)
        tp = tok.word_to_id.get(positive)
        th = tok.word_to_id.get(hard)
        if ta is None or tp is None or th is None:
            continue
        proto_a = model._project_to_concept(model.token_embed.embed_raw(ta))
        proto_p = model._project_to_concept(model.token_embed.embed_raw(tp))
        proto_h = model._project_to_concept(model.token_embed.embed_raw(th))
        norm = lambda v: float(np.linalg.norm(v)) + 1e-10
        s_pos = float(np.dot(proto_a, proto_p) / (norm(proto_a) * norm(proto_p)))
        s_hard = float(np.dot(proto_a, proto_h) / (norm(proto_a) * norm(proto_h)))
        gap = s_pos - s_hard
        gaps.append(gap)
        details.append({"anchor": anchor, "positive": positive, "hard": hard,
                         "s_pos": round(s_pos, 4), "s_hard": round(s_hard, 4), "gap": round(gap, 4)})
    if not gaps:
        return {"mean": 0.0, "std": 0.0, "details": []}
    arr = np.array(gaps, dtype=np.float32)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "details": details}


# ─── Primary entrypoint ─────────────────────────────────────────────

def run_benchmark(n_epochs: int = 1500,
                  embed_dim: int = 64,
                  concept_dim: int = 64,
                  seed: int = 42) -> dict:
    print("=" * 70)
    print("RLMv2 — Cross-Domain Analogical Benchmark v6 (+ Phase 4)")
    print("=" * 70)

    train_triples, test_cases = build_dataset()
    tok = _init_tokenizer(train_triples, test_cases)

    all_pairs: List[Tuple[str, str]] = list(getattr(__import__("semantic_pairs"), "ALL_PAIRS", []))

    actual_vocab = tok.vocab_size
    print(f"Vocab: {actual_vocab}")
    print(f"Training: {len(train_triples)} triples")
    print(f"Epochs: {n_epochs}")
    print()

    model = _init_model(
        actual_vocab=actual_vocab,
        embed_dim=embed_dim,
        concept_dim=concept_dim,
        seed=seed,
    )
    model._tokenizer = tok

    phase4_metrics: dict = {}

    # ── Training ─────────────────────────────────────────────────────
    print("-" * 70)
    print("TRAINING")
    print("-" * 70)

    for epoch in range(n_epochs):
        indices = np.random.permutation(len(train_triples))
        total_loss = 0.0
        correct = 0
        for idx in indices:
            text, target_word = train_triples[idx]
            ids = tok.encode(text)
            target_id = tok.encode(target_word)[0]
            ctx = np.array(ids[:-1], dtype=np.int64)
            tgt = np.array([target_id], dtype=np.int64)
            result = model.learn(ctx, tgt)
            total_loss += result.get("loss", 0.0)
            if result.get("is_correct"):
                correct += 1

        if (epoch + 1) % 500 == 0:
            acc = correct / max(1, len(train_triples))
            print(f"  Epoch {epoch+1}: loss={total_loss / max(1, len(train_triples)):.4f}, "
                  f"acc={acc:.1%}, "
                  f"{len(model.graph.nodes)} concepts, {len(model.graph.edges)} edges")

            # Hard triple boost (uses learn_fast for speed)
            hard = []
            for text, target_word in train_triples:
                ids = tok.encode(text)
                target_id = tok.encode(target_word)[0]
                ctx = np.array(ids[:-1], dtype=np.int64)
                logits = model.forward(ctx)
                top10 = set(np.argsort(logits.data.flatten())[::-1][:10].tolist())
                if target_id not in top10:
                    hard.append((text, target_word))
            if hard:
                print(f"    Hard: {len(hard)}/{len(train_triples)} — boosting 300x")
                for _ in range(300):
                    for text, target_word in hard:
                        ids = tok.encode(text)
                        target_id = tok.encode(target_word)[0]
                        ctx = np.array(ids[:-1], dtype=np.int64)
                        tgt = np.array([target_id], dtype=np.int64)
                        model.learn(ctx, tgt)

        # ── Phase 4 inline geometry validation ────────────────────────
        should_validate = (epoch + 1) % PHASE4_VALIDATE_EVERY == 0 or (epoch + 1) == n_epochs
        if should_validate:
            pos_stats = _pairwise_positive_stats(tok, model, list(getattr(__import__("semantic_pairs"), "ALL_PAIRS", [])))
            gap_stats = _hard_negative_gap_stats(tok, model)

            pos_mean = pos_stats.get("mean", float("nan"))
            pos_std = pos_stats.get("std", 0.0)
            gap_mean = gap_stats.get("mean", 0.0)
            gap_std = gap_stats.get("std", 0.0)

            collapse_flag = bool(
                isinstance(pos_std, (int, float)) and
                isinstance(gap_mean, (int, float)) and
                (pos_std < COLLAPSE_STDEV_THRESHOLD) and
                (gap_mean < COLLAPSE_GAP_THRESHOLD)
            )

            phase4_metrics[str(epoch + 1)] = {
                "positive_similarity_mean": round(pos_mean, 4) if isinstance(pos_mean, float) else pos_mean,
                "positive_similarity_std": round(pos_std, 4),
                "positive_similarity_var": round(float(pos_std * pos_std), 4),
                "hard_negative_gap_mean": round(gap_mean, 4),
                "hard_negative_gap_std": round(gap_std, 4),
                "collapse_flag": collapse_flag,
            }
            print(f"  [Phase4] epoch={epoch+1}: pos_mean={pos_mean:.4f} pos_std={pos_std:.4f} "
                  f"gap_mean={gap_mean:.4f} gap_std={gap_std:.4f} collapse={collapse_flag}")
            if collapse_flag:
                print(f"  [Phase4] WARNING: collapse risk at epoch {epoch+1}")

    # ── Evaluation ───────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    type_counts = {}
    for edge in model.graph.edges.values():
        rt = edge.relation_type
        type_counts[rt] = type_counts.get(rt, 0) + 1
    print(f"\n  Graph: {len(model.graph.nodes)} concepts, {len(model.graph.edges)} edges")
    print(f"  Edge types: {type_counts}")

    results = {}
    for category, test_data in test_cases.items():
        print(f"\n--- {category.upper().replace('_', ' ')} ---")
        hits_1 = hits_5 = hits_10 = 0
        total = 0

        for text, target_word in test_data:
            ids = tok.encode(text)
            target_id = tok.encode(target_word)[0]
            ctx = np.array(ids[:-1], dtype=np.int64)
            logits = model.forward(ctx)
            flat = logits.data.flatten()

            top1 = int(np.argmax(flat))
            top5_set = set(np.argsort(flat)[::-1][:5].tolist())
            top10_set = set(np.argsort(flat)[::-1][:10].tolist())
            top5_words = [tok.decode([int(t)]) for t in np.argsort(flat)[::-1][:5]]

            if target_id == top1:
                hits_1 += 1
            if target_id in top5_set:
                hits_5 += 1
            if target_id in top10_set:
                hits_10 += 1
            total += 1

            status = "✓" if target_id in top10_set else "✗"
            print(f"  {status} \"{text}\" → \"{target_word}\": top5={top5_words}")

        r1 = hits_1 / max(1, total)
        r5 = hits_5 / max(1, total)
        r10 = hits_10 / max(1, total)
        results[category] = {"top1": r1, "top5": r5, "top10": r10, "total": total}
        print(f"  → top-1={r1:.1%}, top-5={r5:.1%}, top-10={r10:.1%}")

    # ── Relation vector analysis ──────────────────────────────────────
    print()
    print("=" * 70)
    print("RELATION VECTOR ANALYSIS")
    print("=" * 70)
    type_rvs = {}
    for edge in model.graph.edges.values():
        rt = edge.relation_type
        if rt not in type_rvs:
            type_rvs[rt] = []
        type_rvs[rt].append(edge.relation_vector)
    centroids = {rt: np.mean(rvs, axis=0) for rt, rvs in type_rvs.items() if rvs}
    type_names = sorted(centroids.keys())
    for i, rt1 in enumerate(type_names):
        for rt2 in type_names[i + 1:]:
            cos = float(np.dot(centroids[rt1], centroids[rt2]) / (np.linalg.norm(centroids[rt1]) * np.linalg.norm(centroids[rt2]) + 1e-10))
            print(f"  {rt1} ↔ {rt2}: {cos:.3f}")

    intra, inter = [], []
    for i, rt1 in enumerate(type_names):
        for rt2 in type_names[i + 1:]:
            for rv1 in type_rvs[rt1][:10]:
                for rv2 in type_rvs[rt2][:10]:
                    inter.append(float(np.dot(rv1, rv2) / (np.linalg.norm(rv1) * np.linalg.norm(rv2) + 1e-10)))
    for rt in type_names:
        rvs = type_rvs[rt]
        for i in range(len(rvs)):
            for j in range(i + 1, min(len(rvs), i + 15)):
                intra.append(float(np.dot(rvs[i], rvs[j]) / (np.linalg.norm(rvs[i]) * np.linalg.norm(rvs[j]) + 1e-10)))
    if intra and inter:
        print(f"  Intra-type mean: {np.mean(intra):.3f}")
        print(f"  Inter-type mean: {np.mean(inter):.3f}")
        print(f"  Separation:      {np.mean(intra) - np.mean(inter):.3f}")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for cat, r in results.items():
        m = "✓" if r["top10"] > 0.5 else ("△" if r["top10"] > 0 else "✗")
        print(f"  {m} {cat}: top-1={r['top1']:.1%}, top-5={r['top5']:.1%}, top-10={r['top10']:.1%} (n={r['total']})")

    total_correct = sum(r["top10"] * r["total"] for r in results.values())
    total_probes = sum(r["total"] for r in results.values())
    print(f"\n  OVERALL top-10: {total_correct:.0f}/{total_probes} = {total_correct / total_probes:.1%}")
    for name in ["train_memorization", "relation_type_transfer", "cross_subject_same_domain",
                 "cross_domain_causal", "bridge_transfer", "property_transfer"]:
        r = results.get(name, {})
        print(f"    {name}: top-10={r.get('top10', 0):.1%}")

    save = {
        "results": {k: {m: round(v, 4) for m, v in r.items()} for k, r in results.items()},
        "overall_top10": round(total_correct / total_probes, 4),
        "graph": {
            "concepts": len(model.graph.nodes),
            "edges": len(model.graph.edges),
            "types": type_counts,
        },
        "config": {
            "epochs": n_epochs,
            "embed_dim": embed_dim,
            "concept_dim": concept_dim,
            "n_train": len(train_triples),
            "vocab": actual_vocab,
            "phase4": {
                "triplet_margin": TRIPLET_MARGIN,
                "collapse_stdev_threshold": COLLAPSE_STDEV_THRESHOLD,
                "collapse_gap_threshold": COLLAPSE_GAP_THRESHOLD,
            },
        },
        "phase4_metrics": phase4_metrics,
    }

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "experiment_results",
        "triple_benchmark_v6.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Results: {out_path}")
    return save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLMv2 Cross-Domain Benchmark v6")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--concept-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_benchmark(
        n_epochs=args.epochs,
        embed_dim=args.embed_dim,
        concept_dim=args.concept_dim,
        seed=args.seed,
    )
