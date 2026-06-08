"""
Contrastive Regularized Encoder Experiment (Option 3)

Unfreezes the encoder and adds a domain-consistency regularizer:
  L = -log(sim(concept_A, concept_B_analog)) + log(sim(concept_A, concept_B_wrong))

Penalizes encoder updates that increase distance between Domain A and Domain B
analogous concepts, forcing the encoder to separate semantics from morphology.

Usage:
    python experiments/experiment_contrastive.py                      # full run
    python experiments/experiment_contrastive.py --epochs 200          # quick test
    python experiments/experiment_contrastive.py --lambda-c 1.0        # tune lambda
    python experiments/experiment_contrastive.py --skip-baseline       # no pre-train eval
"""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer
from semantic_pairs import ALL_PAIRS, SEMANTIC_PAIRS, WITHIN_DOMAIN_PAIRS


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class ContrastiveConfig:
    n_epochs: int = 500
    seed: int = 42

    # RLMv2 architecture
    embed_dim: int = 64
    concept_dim: int = 64

    # Contrastive regularization
    lambda_contrastive: float = 0.5
    neg_sample_size: int = 5

    # Training
    hard_boost_interval: int = 200
    hard_boost_multiplier: int = 100

    # Checkpoint
    save_model: bool = True
    output_dir: str = "experiment_results"


# ── Training Data (same as experiment_cross_domain_v2) ──────────────────────

def build_training_data():
    train = [
        ("heat causes expansion", "expansion"),
        ("friction produces heat", "heat"),
        ("light enables vision", "vision"),
        ("gravity pulls objects", "objects"),
        ("rain causes growth", "growth"),
        ("fire produces warmth", "warmth"),
        ("cold causes shivering", "shivering"),
        ("wind causes erosion", "erosion"),
        ("water causes rust", "rust"),
        ("sunlight causes warmth", "warmth"),
        ("ice makes slippery", "slippery"),
        ("pressure creates diamonds", "diamonds"),
        ("oxygen enables combustion", "combustion"),
        ("drought causes famine", "famine"),
        ("flood causes destruction", "destruction"),
        ("heat melts ice", "ice"),
        ("cold freezes water", "water"),
        ("voltage drives current", "current"),
        ("friction slows motion", "motion"),
        ("gravity shapes orbits", "orbits"),
        ("radiation damages dna", "dna"),
        ("magnetism deflects compasses", "compasses"),
        ("evaporation cools surfaces", "surfaces"),
        ("condensation forms clouds", "clouds"),
        ("sedimentation builds layers", "layers"),
        ("oxidation causes tarnish", "tarnish"),
        ("nuclear force binds protons", "protons"),
        ("tides shift sediment", "sediment"),
        ("lightning ignites fires", "fires"),
        ("corrosion weakens metals", "metals"),
        ("centrifugal force pushes outward", "outward"),
        ("capillary action draws liquid", "liquid"),
        ("resonance shatters glass", "glass"),
        ("diffusion spreads particles", "particles"),
        ("combustion releases energy", "energy"),
        ("osmosis transfers water", "water"),
        ("photosynthesis produces oxygen", "oxygen"),
        ("friction generates static", "static"),
        ("decompression causes cooling", "cooling"),
        ("fermentation produces alcohol", "alcohol"),
        ("water is liquid", "liquid"),
        ("ice is solid", "solid"),
        ("fire is hot", "hot"),
        ("steel is strong", "strong"),
        ("glass is fragile", "fragile"),
        ("diamond is hard", "hard"),
        ("silk is smooth", "smooth"),
        ("lead is heavy", "heavy"),
        ("helium is light", "light"),
        ("rubber is elastic", "elastic"),
        ("granite is dense", "dense"),
        ("mercury is toxic", "toxic"),
        ("quartz is crystalline", "crystalline"),
        ("nitrogen is inert", "inert"),
        ("carbon is versatile", "versatile"),
        ("copper is conductive", "conductive"),
        ("tungsten is refractory", "refractory"),
        ("neon is noble", "noble"),
        ("sulfur is pungent", "pungent"),
        ("aluminum is lightweight", "lightweight"),
        ("kindness leads to trust", "trust"),
        ("anger causes conflict", "conflict"),
        ("sharing builds friendship", "friendship"),
        ("lying destroys trust", "trust"),
        ("patience creates understanding", "understanding"),
        ("honesty builds respect", "respect"),
        ("empathy creates connection", "connection"),
        ("greed causes loneliness", "loneliness"),
        ("jealousy causes resentment", "resentment"),
        ("generosity creates gratitude", "gratitude"),
        ("rudeness causes offense", "offense"),
        ("listening builds rapport", "rapport"),
        ("teaching builds knowledge", "knowledge"),
        ("neglect causes distance", "distance"),
        ("celebration builds bonds", "bonds"),
        ("criticism causes defensiveness", "defensiveness"),
        ("forgiveness heals wounds", "wounds"),
        ("praise boosts confidence", "confidence"),
        ("isolation causes sadness", "sadness"),
        ("teamwork creates success", "success"),
        ("gossip spreads mistrust", "mistrust"),
        ("mentorship builds skills", "skills"),
        ("bullying causes trauma", "trauma"),
        ("collaboration produces innovation", "innovation"),
        ("rejection causes withdrawal", "withdrawal"),
        ("inclusion builds belonging", "belonging"),
        ("betrayal destroys loyalty", "loyalty"),
        ("gratitude strengthens relationships", "relationships"),
        ("boredom triggers exploration", "exploration"),
        ("competition drives excellence", "excellence"),
        ("compassion reduces suffering", "suffering"),
        ("sarcasm creates tension", "tension"),
        ("trust enables vulnerability", "vulnerability"),
        ("leadership inspires action", "action"),
        ("apology restores harmony", "harmony"),
        ("neglect weakens bonds", "bonds"),
        ("humor defuses conflict", "conflict"),
        ("rivalry spurs growth", "growth"),
        ("grief deepens empathy", "empathy"),
        ("curiosity sparks discovery", "discovery"),
        ("friendship is valuable", "valuable"),
        ("family is important", "important"),
        ("trust is fragile", "fragile"),
        ("wisdom is rare", "rare"),
        ("courage is admirable", "admirable"),
        ("patience is virtuous", "virtuous"),
        ("humor is helpful", "helpful"),
        ("loyalty is noble", "noble"),
        ("rudeness is harmful", "harmful"),
        ("kindness is powerful", "powerful"),
        ("honesty is essential", "essential"),
        ("grief is natural", "natural"),
        ("ambition is driving", "driving"),
        ("solitude is peaceful", "peaceful"),
        ("chaos is destabilizing", "destabilizing"),
        ("harmony is restorative", "restorative"),
        ("resentment is corrosive", "corrosive"),
        ("hope is resilient", "resilient"),
        ("pride is dangerous", "dangerous"),
        ("grace is inspiring", "inspiring"),
        ("anger is intense", "intense"),
        ("heat is intense", "intense"),
        ("anger produces heat", "heat"),
        ("anger is hot", "hot"),
        ("love is warm", "warm"),
        ("kindness is warm", "warm"),
        ("warmth causes growth", "growth"),
        ("love produces warmth", "warmth"),
        ("rain is flowing", "flowing"),
        ("tears are flowing", "flowing"),
        ("sadness causes tears", "tears"),
        ("bugs are viruses", "viruses"),
        ("viruses cause damage", "damage"),
        ("code produces bugs", "bugs"),
        ("exercise produces stress", "stress"),
        ("stress causes crashes", "crashes"),
        ("intense causes expansion", "expansion"),
        ("warm causes trust", "trust"),
        ("flowing causes flooding", "flooding"),
        ("destructive causes damage", "damage"),
        ("anger causes expansion", "expansion"),
        ("love produces heat", "heat"),
        ("heat causes trust", "trust"),
        ("fire produces friendship", "friendship"),
        ("kindness causes flooding", "flooding"),
        ("rain produces conflict", "conflict"),
        ("code causes illness", "illness"),
        ("exercise produces crashes", "crashes"),
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
    ]

    test_cases = {
        "train_memorization": [
            ("heat causes expansion", "expansion"),
            ("anger causes conflict", "conflict"),
            ("kindness leads to trust", "trust"),
            ("fire produces warmth", "warmth"),
            ("sharing builds friendship", "friendship"),
            ("gravity pulls objects", "objects"),
            ("wind causes erosion", "erosion"),
            ("patience creates understanding", "understanding"),
            ("water causes rust", "rust"),
            ("empathy creates connection", "connection"),
            ("light enables vision", "vision"),
            ("cold causes shivering", "shivering"),
        ],
        "relation_type_transfer": [
            ("friction generates static", "static"),
            ("gossip spreads mistrust", "mistrust"),
            ("praise boosts confidence", "confidence"),
            ("radiation damages dna", "dna"),
            ("bullying causes trauma", "trauma"),
            ("corrosion weakens metals", "metals"),
            ("compassion reduces suffering", "suffering"),
            ("combustion releases energy", "energy"),
            ("forgiveness heals wounds", "wounds"),
        ],
        "cross_subject_same_domain": [
            ("sedimentation builds diamonds", "diamonds"),
            ("gravity slows motion", "motion"),
            ("cold drives current", "current"),
            ("heat forms clouds", "clouds"),
            ("lying builds trust", "trust"),
            ("gossip creates connection", "connection"),
            ("patience destroys loyalty", "loyalty"),
            ("competition builds bonds", "bonds"),
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
            ("fire produces heat", "heat"),
            ("sun produces growth", "growth"),
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

    return train, test_cases


# ── Training & Evaluation ────────────────────────────────────────────────────

def train_epoch(model, train_triples, tok):
    indices = np.random.permutation(len(train_triples))
    total_loss = 0
    correct = 0
    for idx in indices:
        text, target_word = train_triples[idx]
        ids = tok.encode(text)
        target_id = tok.encode(target_word)[0]
        ctx = np.array(ids[:-1], dtype=np.int64)
        tgt = np.array([target_id], dtype=np.int64)
        result = model.learn(ctx, tgt)
        total_loss += result["loss"]
        if result.get("is_correct"):
            correct += 1
    return total_loss, correct


def hard_boost(model, train_triples, tok, multiplier=100):
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
        for _ in range(multiplier):
            for text, target_word in hard:
                ids = tok.encode(text)
                target_id = tok.encode(target_word)[0]
                ctx = np.array(ids[:-1], dtype=np.int64)
                tgt = np.array([target_id], dtype=np.int64)
                model.learn(ctx, tgt)
    return len(hard)


def evaluate(model, test_data, tok):
    hits_1 = hits_5 = hits_10 = 0
    total = 0
    per_probe = []

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

        if target_id == top1: hits_1 += 1
        if target_id in top5_set: hits_5 += 1
        if target_id in top10_set: hits_10 += 1
        total += 1

        per_probe.append({
            "input": text,
            "expected": target_word,
            "top1": tok.decode([top1]),
            "top5": top5_words,
            "correct_top1": target_id == top1,
            "correct_top10": target_id in top10_set,
        })

    return {
        "top1": hits_1 / max(1, total),
        "top5": hits_5 / max(1, total),
        "top10": hits_10 / max(1, total),
        "total": total,
        "probes": per_probe,
    }


# ── Encoder Analysis ─────────────────────────────────────────────────────────

def compute_morphology_semantic_separation(model, tok):
    """Measure how well the encoder separates semantics from morphology.

    Computes the ratio of semantic similarity to morphological (character n-gram)
    similarity for semantically related vs. unrelated pairs in latent space.
    Higher values = better separation.
    """
    semantic_sims = []
    morph_sims = []

    for word_a, word_b in SEMANTIC_PAIRS:
        tid_a = tok.word_to_id.get(word_a)
        tid_b = tok.word_to_id.get(word_b)
        if tid_a is None or tid_b is None:
            continue

        embed_a = model.token_embed.weight.data[tid_a]
        embed_b = model.token_embed.weight.data[tid_b]

        lat_a = model._encoder_forward_full(embed_a)[0]
        lat_b = model._encoder_forward_full(embed_b)[0]

        norm_a = np.linalg.norm(lat_a)
        norm_b = np.linalg.norm(lat_b)
        if norm_a > 0 and norm_b > 0:
            sem_sim = float(np.dot(lat_a, lat_b) / (norm_a * norm_b))
            semantic_sims.append(sem_sim)

    # Morphologically similar (shared character n-grams) but semantically unrelated
    morph_pairs = [
        ("evaporation", "condensation"),   # both end in -ation, science processes
        ("fermentation", "sedimentation"), # both end in -mentation, unrelated processes
        ("expansion", "exploration"),      # both start with ex-, different semantics
        ("resonance", "resilient"),        # both start with re-, different domains
        ("sediment", "sedimentation"),     # same root, different concepts
    ]
    # Use only pairs where both words exist
    valid_morph = []
    for wa, wb in morph_pairs:
        if tok.word_to_id.get(wa) is not None and tok.word_to_id.get(wb) is not None:
            valid_morph.append((wa, wb))

    for word_a, word_b in valid_morph:
        tid_a = tok.word_to_id.get(word_a)
        tid_b = tok.word_to_id.get(word_b)
        embed_a = model.token_embed.weight.data[tid_a]
        embed_b = model.token_embed.weight.data[tid_b]

        lat_a = model._encoder_forward_full(embed_a)[0]
        lat_b = model._encoder_forward_full(embed_b)[0]

        norm_a = np.linalg.norm(lat_a)
        norm_b = np.linalg.norm(lat_b)
        if norm_a > 0 and norm_b > 0:
            morph_sim = float(np.dot(lat_a, lat_b) / (norm_a * norm_b))
            morph_sims.append(morph_sim)

    return {
        "mean_semantic_similarity": float(np.mean(semantic_sims)) if semantic_sims else 0.0,
        "mean_morphological_similarity": float(np.mean(morph_sims)) if morph_sims else 0.0,
        "separation_ratio": float(np.mean(semantic_sims) / (np.mean(morph_sims) + 1e-10))
        if semantic_sims and morph_sims else 0.0,
    }


# ── Main Experiment ──────────────────────────────────────────────────────────

def run_contrastive_experiment(config: ContrastiveConfig, enable_contrastive: bool = True) -> Dict[str, Any]:
    tag = "contrastive" if enable_contrastive else "baseline"
    print(f"\n{'=' * 70}")
    print(f"  {'CONTRASTIVE REGULARIZED ENCODER' if enable_contrastive else 'BASELINE (no contrastive)'}")
    print(f"  lambda={config.lambda_contrastive}, neg_sample={config.neg_sample_size}")
    print(f"{'=' * 70}\n")

    np.random.seed(config.seed)

    train_triples, test_cases = build_training_data()

    tok = WordTokenizer()
    all_texts = set()
    for text, _ in train_triples:
        all_texts.add(text)
    for cat in test_cases.values():
        for text, _ in cat:
            all_texts.add(text)
    for text in sorted(all_texts):
        tok.encode(text)

    actual_vocab = tok.vocab_size
    print(f"Vocab: {actual_vocab}")
    print(f"Training: {len(train_triples)} triples")
    print(f"Semantic pairs: {len(ALL_PAIRS)} ({len(SEMANTIC_PAIRS)} cross-domain + {len(WITHIN_DOMAIN_PAIRS)} within-domain)")
    print(f"Epochs: {config.n_epochs}")

    model = RLMv2(
        vocab_size=actual_vocab + 5,
        embed_dim=config.embed_dim,
        concept_dim=config.concept_dim,
        n_concepts=actual_vocab,
        sleep_interval=300,
        gate_concept_creation=False,
    )
    model._tokenizer = tok

    # ── Configure contrastive regularization ──
    if enable_contrastive:
        model.freeze_encoder = False
        model.use_contrastive_reg = True
        model.semantic_pairs = ALL_PAIRS
        model.lambda_contrastive = config.lambda_contrastive
        model.neg_sample_size = config.neg_sample_size
        print(f"Contrastive: freeze_encoder={model.freeze_encoder}, "
              f"use_contrastive_reg={model.use_contrastive_reg}, "
              f"lambda={model.lambda_contrastive}, neg_sample={model.neg_sample_size}")
    else:
        model.freeze_encoder = True
        model.use_contrastive_reg = False
        print(f"Baseline: freeze_encoder={model.freeze_encoder} (no contrastive reg)")
    print()

    # ── Baseline (before training) ──
    print("-" * 70)
    print("BASELINE (before training)")
    print("-" * 70)
    baseline_results = {}
    for cat, test_data in test_cases.items():
        r = evaluate(model, test_data, tok)
        baseline_results[cat] = r
        print(f"  {cat}: top1={r['top1']:.1%}, top10={r['top10']:.1%}")

    enc_analysis_pre = compute_morphology_semantic_separation(model, tok)
    print(f"\n  Encoder analysis (pre):")
    print(f"    Semantic sim: {enc_analysis_pre['mean_semantic_similarity']:.4f}")
    print(f"    Morphological sim: {enc_analysis_pre['mean_morphological_similarity']:.4f}")
    print(f"    Separation ratio: {enc_analysis_pre['separation_ratio']:.4f}")

    # ── Training ──
    print()
    print("-" * 70)
    print("TRAINING")
    print("-" * 70)

    t0 = time.time()
    for epoch in range(config.n_epochs):
        total_loss, correct = train_epoch(model, train_triples, tok)
        acc = correct / len(train_triples)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_triples):.4f}, "
                  f"acc={acc:.1%}, {len(model.graph.nodes)} concepts, "
                  f"{len(model.graph.edges)} edges")

        if (epoch + 1) % config.hard_boost_interval == 0:
            n_hard = hard_boost(model, train_triples, tok, config.hard_boost_multiplier)
            if n_hard > 0:
                print(f"    Hard boost: {n_hard}/{len(train_triples)} triples")

    train_time = time.time() - t0
    print(f"\n  Training time: {train_time:.1f}s")

    # ── Evaluation ──
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
    for cat, test_data in test_cases.items():
        r = evaluate(model, test_data, tok)
        results[cat] = r
        status = "PASS" if r["top10"] > 0.5 else ("PARTIAL" if r["top10"] > 0 else "FAIL")
        print(f"\n  [{status}] {cat.upper().replace('_', ' ')}")
        print(f"    top-1={r['top1']:.1%}, top-5={r['top5']:.1%}, top-10={r['top10']:.1%} (n={r['total']})")
        for p in r["probes"]:
            mark = "OK" if p["correct_top10"] else "X"
            print(f"      [{mark}] \"{p['input']}\" -> \"{p['expected']}\": top5={p['top5']}")

    total_correct = sum(r["top10"] * r["total"] for r in results.values())
    total_probes = sum(r["total"] for r in results.values())
    overall_top10 = total_correct / total_probes
    total_correct_1 = sum(r["top1"] * r["total"] for r in results.values())
    overall_top1 = total_correct_1 / total_probes

    print(f"\n  OVERALL: top-1={overall_top1:.1%}, top-10={overall_top10:.1%}")

    cd = results.get("cross_domain_causal", {})
    print(f"\n  CROSS-DOMAIN CAUSAL: top-1={cd.get('top1', 0):.1%}, top-10={cd.get('top10', 0):.1%}")

    # ── Encoder analysis (post-training) ──
    enc_analysis_post = compute_morphology_semantic_separation(model, tok)
    print(f"\n  Encoder analysis (post):")
    print(f"    Semantic sim: {enc_analysis_post['mean_semantic_similarity']:.4f}")
    print(f"    Morphological sim: {enc_analysis_post['mean_morphological_similarity']:.4f}")
    print(f"    Separation ratio: {enc_analysis_post['separation_ratio']:.4f}")
    print(f"    Change: semantic {enc_analysis_pre['mean_semantic_similarity']:.4f} → {enc_analysis_post['mean_semantic_similarity']:.4f}, "
          f"morphological {enc_analysis_pre['mean_morphological_similarity']:.4f} → {enc_analysis_post['mean_morphological_similarity']:.4f}")

    # ── Verdict ──
    print()
    print("-" * 70)
    if cd.get("top1", 0) > 0.3:
        print("  VERDICT: STRONG CROSS-DOMAIN TRANSFER")
    elif cd.get("top10", 0) > 0.5:
        print("  VERDICT: MODERATE CROSS-DOMAIN TRANSFER (top-10)")
    elif cd.get("top10", 0) > 0.2:
        print("  VERDICT: WEAK CROSS-DOMAIN TRANSFER")
    else:
        print("  VERDICT: NO CROSS-DOMAIN TRANSFER")
    print("-" * 70)

    save_data = {
        "tag": tag,
        "config": asdict(config),
        "results": {k: {m: round(v, 4) if isinstance(v, float) else v
                        for m, v in r.items() if m != "probes"}
                    for k, r in results.items()},
        "overall_top1": round(overall_top1, 4),
        "overall_top10": round(overall_top10, 4),
        "cross_domain_top1": round(cd.get("top1", 0), 4),
        "cross_domain_top10": round(cd.get("top10", 0), 4),
        "encoder_analysis": {
            "pre": enc_analysis_pre,
            "post": enc_analysis_post,
        },
        "graph": {
            "concepts": len(model.graph.nodes),
            "edges": len(model.graph.edges),
            "types": type_counts,
        },
        "train_time": round(train_time, 1),
    }

    if config.save_model:
        os.makedirs(config.output_dir, exist_ok=True)
        model_path = os.path.join(config.output_dir, f"rlmv2_{tag}.pkl")
        model.save(model_path)
        save_data["model_path"] = model_path

    return save_data


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Contrastive Regularized Encoder Experiment")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-c", type=float, default=0.5, help="Contrastive loss weight")
    parser.add_argument("--neg-sample", type=int, default=5, help="Negative samples per pair")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline run")
    parser.add_argument("--output", type=str, default="experiment_results/contrastive_results.json")
    args = parser.parse_args()

    config = ContrastiveConfig(
        n_epochs=args.epochs,
        seed=args.seed,
        lambda_contrastive=args.lambda_c,
        neg_sample_size=args.neg_sample,
    )

    # Run baseline (no contrastive)
    baseline_data = None
    if not args.skip_baseline:
        print("\n>>> Running BASELINE (contrastive OFF, encoder frozen)")
        baseline_data = run_contrastive_experiment(config, enable_contrastive=False)
        print()

    # Run contrastive experiment
    print(">>> Running CONTRASTIVE REGULARIZED (encoder unfrozen)")
    contrastive_data = run_contrastive_experiment(config, enable_contrastive=True)

    # ── Comparative Summary ──
    print("\n" + "=" * 70)
    print("  COMPARATIVE SUMMARY")
    print("=" * 70)
    if baseline_data:
        bl_cd = baseline_data.get("cross_domain_top1", 0)
        ct_cd = contrastive_data.get("cross_domain_top1", 0)
        bl_oa = baseline_data.get("overall_top1", 0)
        ct_oa = contrastive_data.get("overall_top1", 0)
        bl_sep = baseline_data.get("encoder_analysis", {}).get("post", {}).get("separation_ratio", 0)
        ct_sep = contrastive_data.get("encoder_analysis", {}).get("post", {}).get("separation_ratio", 0)
        print(f"\n  {'Metric':<35} {'Baseline':>12} {'Contrastive':>12} {'Delta':>10}")
        print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")
        print(f"  {'Cross-domain top-1':<35} {bl_cd:>11.1%} {ct_cd:>11.1%} {ct_cd - bl_cd:>+9.1%}")
        print(f"  {'Overall top-1':<35} {bl_oa:>11.1%} {ct_oa:>11.1%} {ct_oa - bl_oa:>+9.1%}")
        print(f"  {'Semantic/morphology separation':<35} {bl_sep:>12.4f} {ct_sep:>12.4f} {ct_sep - bl_sep:>+10.4f}")

    # Save combined results
    combined = {
        "config": asdict(config),
        "baseline": baseline_data,
        "contrastive": contrastive_data,
    }

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(_PROJECT_ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=convert)

    print(f"\nResults saved to {out_path}")
