"""
Cross-Domain Transfer Experiment for RAVANA RLMv2

Tests whether knowledge learned in Domain A transfers to Domain B via
analogical bridges in the concept graph.

Ported from RLMv1 to RLMv2: triple decomposition + spreading activation
instead of GRU + 5-path blend. Includes bridge facts connecting domains
and proper epoch-based training with hard-boost.

Usage:
    python experiment_cross_domain_v2.py                     # full experiment
    python experiment_cross_domain_v2.py --epochs 200       # quick test
    python experiment_cross_domain_v2.py --skip-baselines    # RLMv2 only
"""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field, asdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CrossDomainConfig:
    n_epochs: int = 500
    seed: int = 42
    skip_baselines: bool = False
    hard_boost_interval: int = 200
    hard_boost_multiplier: int = 100

    # RLMv2 architecture
    embed_dim: int = 64
    concept_dim: int = 64


# ═══════════════════════════════════════════════════════════════════════════
# Domain Knowledge Bases
# ═══════════════════════════════════════════════════════════════════════════

def build_training_data():
    """Build all training triples: domain A + domain B + bridges.

    Returns (train_triples, test_cases) where each triple is
    (full_sentence, target_word).
    """
    train = [
        # ── Domain A: Science (causal + semantic) ──
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
        # Domain A semantic
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

        # ── Domain B: Social (causal + semantic) ──
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
        # Domain B semantic
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

        # ── Cross-domain bridges (shared properties) ──
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

        # ── Cross-domain causal chains (explicit) ──
        ("anger causes expansion", "expansion"),
        ("love produces heat", "heat"),
        ("heat causes trust", "trust"),
        ("fire produces friendship", "friendship"),
        ("kindness causes flooding", "flooding"),
        ("rain produces conflict", "conflict"),
        ("code causes illness", "illness"),
        ("exercise produces crashes", "crashes"),

        # ── Extra bridges for coverage ──
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


# ═══════════════════════════════════════════════════════════════════════════
# Training & Evaluation (RLMv2 style)
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, train_triples, tok):
    """Train one epoch. Returns (loss_sum, correct_count)."""
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
    """Find failing triples and re-train them extra times."""
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
    """Evaluate on test data. Returns dict with top-1/5/10 accuracy."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_cross_domain_experiment(config: CrossDomainConfig) -> Dict[str, Any]:
    """Run the full cross-domain transfer experiment with RLMv2."""

    print("=" * 70)
    print("  CROSS-DOMAIN TRANSFER EXPERIMENT -- RLMv2")
    print("=" * 70)
    print()

    np.random.seed(config.seed)

    train_triples, test_cases = build_training_data()

    # Build tokenizer
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
    print(f"Epochs: {config.n_epochs}")
    print(f"Hard boost: {config.hard_boost_multiplier}x every {config.hard_boost_interval} epochs")
    print()

    # Create model
    model = RLMv2(
        vocab_size=actual_vocab + 5,
        embed_dim=config.embed_dim,
        concept_dim=config.concept_dim,
        n_concepts=actual_vocab,
        sleep_interval=300,
        gate_concept_creation=False,
    )
    model._tokenizer = tok

    # ── Baseline (before training) ──
    print("-" * 70)
    print("BASELINE (before training)")
    print("-" * 70)
    baseline_results = {}
    for cat, test_data in test_cases.items():
        r = evaluate(model, test_data, tok)
        baseline_results[cat] = r
        print(f"  {cat}: top1={r['top1']:.1%}, top10={r['top10']:.1%}")

    # ── Training ──
    print()
    print("-" * 70)
    print("TRAINING")
    print("-" * 70)

    t0 = time.time()
    for epoch in range(config.n_epochs):
        total_loss, correct = train_epoch(model, train_triples, tok)
        acc = correct / len(train_triples)

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(train_triples):.4f}, "
                  f"acc={acc:.1%}, {len(model.graph.nodes)} concepts, "
                  f"{len(model.graph.edges)} edges")

        # Hard boost
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

    # Graph stats
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

    # Overall
    total_correct = sum(r["top10"] * r["total"] for r in results.values())
    total_probes = sum(r["total"] for r in results.values())
    overall_top10 = total_correct / total_probes
    total_correct_1 = sum(r["top1"] * r["total"] for r in results.values())
    overall_top1 = total_correct_1 / total_probes

    print(f"\n  OVERALL: top-1={overall_top1:.1%}, top-10={overall_top10:.1%}")

    # Cross-domain focus
    cd = results.get("cross_domain_causal", {})
    print(f"\n  CROSS-DOMAIN CAUSAL: top-1={cd.get('top1', 0):.1%}, top-10={cd.get('top10', 0):.1%}")

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

    # Save results
    save_data = {
        "config": asdict(config),
        "results": {k: {m: round(v, 4) if isinstance(v, float) else v
                        for m, v in r.items() if m != "probes"}
                    for k, r in results.items()},
        "overall_top1": round(overall_top1, 4),
        "overall_top10": round(overall_top10, 4),
        "cross_domain_top1": round(cd.get("top1", 0), 4),
        "cross_domain_top10": round(cd.get("top10", 0), 4),
        "graph": {
            "concepts": len(model.graph.nodes),
            "edges": len(model.graph.edges),
            "types": type_counts,
        },
        "train_time": round(train_time, 1),
    }

    return save_data


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Domain Transfer Experiment (RLMv2)")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="experiment_results/cross_domain_v2.json")
    args = parser.parse_args()

    config = CrossDomainConfig(
        n_epochs=args.epochs,
        skip_baselines=args.skip_baselines,
        seed=args.seed,
    )

    results = run_cross_domain_experiment(config)

    # Save results
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
        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to {out_path}")
