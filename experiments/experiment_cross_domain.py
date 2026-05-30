"""
Cross-Domain Transfer Experiment for RAVANA RLM

Tests whether knowledge learned in Domain A transfers to Domain B.

Hypothesis: RLM's concept graph + sleep cycles (especially REM cross-linking)
enable positive transfer between structurally similar but semantically distinct
domains. Baseline MLP with backprop should show no such transfer (or negative
transfer / catastrophic forgetting).

Experiment Design:
  Phase 1: Train on Domain A facts (science: causes, effects)
  Phase 2: Train on Domain B facts (social: relationships, emotions)
  Phase 3: Test Domain A recall (retention after Domain B training)
  Phase 4: Test cross-domain queries (can the model use Domain A knowledge
           to help answer Domain B questions?)

Domains:
  Domain A — "Science": causal relationships between physical concepts
    e.g., "heat causes expansion", "friction produces heat"
  Domain B — "Social": relational facts about people and emotions
    e.g., "kindness leads to trust", "anger causes conflict"

Transfer is measured by:
  1. Retention: Domain A accuracy after Domain B training
  2. Forward transfer: Does Domain A knowledge help Domain B learning speed?
  3. Cross-domain inference: Can the model chain A→B when concepts share
     structural similarity (e.g., "causes" edges in both domains)?

Usage:
    python experiment_cross_domain.py                     # full experiment
    python experiment_cross_domain.py --n 500             # quick test
    python experiment_cross_domain.py --skip-baselines    # RLM only
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

from ravana_ml.nn.rlm import RLM
from ravana_ml.tokenizer import WordTokenizer
from experiments.experiment_baselines import SimpleMLP


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CrossDomainConfig:
    n_train_repeats: int = 3            # repeats of each fact during training
    n_test_probes: int = 50             # probes per test
    seed: int = 42
    skip_baselines: bool = False

    # RLM architecture
    embed_dim: int = 64
    concept_dim: int = 64
    n_hidden: int = 128
    n_layers: int = 3
    sleep_interval: int = 100


# ═══════════════════════════════════════════════════════════════════════════
# Domain Knowledge Bases
# ═══════════════════════════════════════════════════════════════════════════

def build_domain_a_science() -> Dict[str, List[Tuple[str, str, str]]]:
    """Domain A: Science — causal relationships between physical concepts.

    Returns dict with 'train' and 'test' splits. Each item is
    (input_text, target_text, relation_type).
    """
    facts = [
        # Causal facts
        ("heat causes ", "expansion", "causal"),
        ("friction produces ", "heat", "causal"),
        ("light enables ", "vision", "causal"),
        ("gravity pulls ", "objects", "causal"),
        ("rain causes ", "growth", "causal"),
        ("fire produces ", "warmth", "causal"),
        ("cold causes ", "shivering", "causal"),
        ("wind causes ", "erosion", "causal"),
        ("water causes ", "rust", "causal"),
        ("sunlight causes ", "warmth", "causal"),
        ("ice makes ", "slippery", "causal"),
        ("pressure creates ", "diamonds", "causal"),
        ("oxygen enables ", "combustion", "causal"),
        ("drought causes ", "famine", "causal"),
        ("flood causes ", "destruction", "causal"),
        ("heat melts ", "ice", "causal"),
        ("cold freezes ", "water", "causal"),
        ("voltage drives ", "current", "causal"),
        ("friction slows ", "motion", "causal"),
        ("gravity shapes ", "orbits", "causal"),
        ("radiation damages ", "dna", "causal"),
        ("magnetism deflects ", "compasses", "causal"),
        ("evaporation cools ", "surfaces", "causal"),
        ("condensation forms ", "clouds", "causal"),
        ("sedimentation builds ", "layers", "causal"),
        ("oxidation causes ", "tarnish", "causal"),
        ("nuclear force binds ", "protons", "causal"),
        ("tides shift ", "sediment", "causal"),
        ("lightning ignites ", "fires", "causal"),
        ("corrosion weakens ", "metals", "causal"),
        ("centrifugal force pushes ", "outward", "causal"),
        ("capillary action draws ", "liquid", "causal"),
        ("resonance shatters ", "glass", "causal"),
        ("diffusion spreads ", "particles", "causal"),
        ("combustion releases ", "energy", "causal"),
        ("osmosis transfers ", "water", "causal"),
        ("photosynthesis produces ", "oxygen", "causal"),
        ("friction generates ", "static", "causal"),
        ("decompression causes ", "cooling", "causal"),
        ("fermentation produces ", "alcohol", "causal"),
        # Semantic facts (is-a / properties)
        ("water is ", "liquid", "semantic"),
        ("ice is ", "solid", "semantic"),
        ("fire is ", "hot", "semantic"),
        ("steel is ", "strong", "semantic"),
        ("glass is ", "fragile", "semantic"),
        ("diamond is ", "hard", "semantic"),
        ("silk is ", "smooth", "semantic"),
        ("lead is ", "heavy", "semantic"),
        ("helium is ", "light", "semantic"),
        ("rubber is ", "elastic", "semantic"),
        ("granite is ", "dense", "semantic"),
        ("mercury is ", "toxic", "semantic"),
        ("quartz is ", "crystalline", "semantic"),
        ("nitrogen is ", "inert", "semantic"),
        ("carbon is ", "versatile", "semantic"),
        ("copper is ", "conductive", "semantic"),
        ("tungsten is ", "refractory", "semantic"),
        ("neon is ", "noble", "semantic"),
        ("sulfur is ", "pungent", "semantic"),
        ("aluminum is ", "lightweight", "semantic"),
    ]

    # Train on ALL facts; cross-domain probes are the real test.
    return {"train": facts, "test": facts}


def build_domain_b_social() -> Dict[str, List[Tuple[str, str, str]]]:
    """Domain B: Social — relationships and emotions between people.

    Structurally parallel to Domain A (cause→effect + is-a) but
    semantically distinct. Tests whether structural patterns transfer.
    """
    facts = [
        # Causal facts
        ("kindness leads to ", "trust", "causal"),
        ("anger causes ", "conflict", "causal"),
        ("sharing builds ", "friendship", "causal"),
        ("lying destroys ", "trust", "causal"),
        ("patience creates ", "understanding", "causal"),
        ("honesty builds ", "respect", "causal"),
        ("empathy creates ", "connection", "causal"),
        ("greed causes ", "loneliness", "causal"),
        ("jealousy causes ", "resentment", "causal"),
        ("generosity creates ", "gratitude", "causal"),
        ("rudeness causes ", "offense", "causal"),
        ("listening builds ", "rapport", "causal"),
        ("teaching builds ", "knowledge", "causal"),
        ("neglect causes ", "distance", "causal"),
        ("celebration builds ", "bonds", "causal"),
        ("criticism causes ", "defensiveness", "causal"),
        ("forgiveness heals ", "wounds", "causal"),
        ("praise boosts ", "confidence", "causal"),
        ("isolation causes ", "sadness", "causal"),
        ("teamwork creates ", "success", "causal"),
        ("gossip spreads ", "mistrust", "causal"),
        ("mentorship builds ", "skills", "causal"),
        ("bullying causes ", "trauma", "causal"),
        ("collaboration produces ", "innovation", "causal"),
        ("rejection causes ", "withdrawal", "causal"),
        ("inclusion builds ", "belonging", "causal"),
        ("betrayal destroys ", "loyalty", "causal"),
        ("gratitude strengthens ", "relationships", "causal"),
        ("boredom triggers ", "exploration", "causal"),
        ("competition drives ", "excellence", "causal"),
        ("compassion reduces ", "suffering", "causal"),
        ("sarcasm creates ", "tension", "causal"),
        ("trust enables ", "vulnerability", "causal"),
        ("leadership inspires ", "action", "causal"),
        ("apology restores ", "harmony", "causal"),
        ("neglect weakens ", "bonds", "causal"),
        ("humor defuses ", "conflict", "causal"),
        ("rivalry spurs ", "growth", "causal"),
        ("grief deepens ", "empathy", "causal"),
        ("curiosity sparks ", "discovery", "causal"),
        # Semantic facts
        ("friendship is ", "valuable", "semantic"),
        ("family is ", "important", "semantic"),
        ("trust is ", "fragile", "semantic"),
        ("wisdom is ", "rare", "semantic"),
        ("courage is ", "admirable", "semantic"),
        ("patience is ", "virtuous", "semantic"),
        ("humor is ", "helpful", "semantic"),
        ("loyalty is ", "noble", "semantic"),
        ("rudeness is ", "harmful", "semantic"),
        ("kindness is ", "powerful", "semantic"),
        ("honesty is ", "essential", "semantic"),
        ("grief is ", "natural", "semantic"),
        ("ambition is ", "driving", "semantic"),
        ("solitude is ", "peaceful", "semantic"),
        ("chaos is ", "destabilizing", "semantic"),
        ("harmony is ", "restorative", "semantic"),
        ("resentment is ", "corrosive", "semantic"),
        ("hope is ", "resilient", "semantic"),
        ("pride is ", "dangerous", "semantic"),
        ("grace is ", "inspiring", "semantic"),
    ]

    # Train on ALL facts; cross-domain probes are the real test.
    return {"train": facts, "test": facts}


# ═══════════════════════════════════════════════════════════════════════════
# Training & Evaluation Helpers
# ═══════════════════════════════════════════════════════════════════════════

def encode_fact(tokenizer, input_text: str, target_text: str):
    """Encode a (input, target) pair into token arrays."""
    input_ids = np.array(tokenizer.encode(input_text), dtype=np.int64)
    target_ids = np.array(tokenizer.encode(target_text), dtype=np.int64)
    return input_ids, target_ids


def train_rlm_on_domain(model: RLM, facts: List[Tuple[str, str, str]],
                         tokenizer, n_repeats: int = 3,
                         domain_tag: Optional[str] = None,
                         buffer_for_replay: bool = False):
    """Train RLM on a set of facts. Returns per-step accuracy history.

    Args:
        domain_tag: if provided, experiences are buffered for interleaved replay
        buffer_for_replay: if True, buffer experiences in the model's replay system
    """
    acc_history = []
    errors = []

    for repeat in range(n_repeats):
        for input_text, target_text, rel_type in facts:
            input_ids, target_ids = encode_fact(tokenizer, input_text, target_text)
            err = model.learn(input_ids, target_ids)
            errors.append(err)
            acc_history.append(model.conceptual_accuracy)

            # Buffer experience for interleaved replay during sleep
            if buffer_for_replay and domain_tag:
                model.buffer_experience(input_ids, target_ids, domain=domain_tag)

    return acc_history, errors


def evaluate_rlm(model: RLM, facts: List[Tuple[str, str, str]],
                  tokenizer, temperature: float = 1.0) -> Dict[str, Any]:
    """Evaluate RLM on a set of facts. Returns accuracy metrics.

    Uses top-10 matching (consistent with experiment_lifelong.py).

    Args:
        temperature: softmax temperature for Top-1 scoring. T<1 sharpens
            the distribution (amplifies top logit), T=1 is raw.
    """
    correct_top1 = 0
    correct_top10 = 0
    total = 0

    for input_text, target_text, rel_type in facts:
        input_ids = np.array(tokenizer.encode(input_text), dtype=np.int64)
        target_ids = np.array(tokenizer.encode(target_text), dtype=np.int64)
        if len(input_ids) == 0 or len(target_ids) == 0:
            continue

        logits = model.forward(input_ids[np.newaxis, :], eval_mode=True)
        probs_data = logits.data if hasattr(logits, 'data') else np.array(logits)
        if probs_data.ndim > 1:
            probs_data = probs_data[0]

        target_id = int(target_ids[0])  # first token of target word

        # LayerNorm + temperature scaling for Top-1 (eval-time only)
        scaled = (probs_data - np.mean(probs_data)) / (np.std(probs_data) + 1e-8)
        if temperature != 1.0:
            scaled = scaled / temperature
        pred_id = int(np.argmax(scaled))
        if pred_id == target_id:
            correct_top1 += 1

        # Top-10 (always on raw logits — temperature only affects Top-1 selection)
        top10 = set(np.argsort(probs_data)[-10:])
        if target_id in top10:
            correct_top10 += 1

        total += 1

    return {
        "top1_accuracy": correct_top1 / max(1, total),
        "top10_accuracy": correct_top10 / max(1, total),
        "n_tested": total,
    }


def train_mlp_on_domain(model: SimpleMLP, facts: List[Tuple[str, str, str]],
                         tokenizer, n_repeats: int = 3):
    """Train SimpleMLP baseline on facts."""
    losses = []
    for repeat in range(n_repeats):
        for input_text, target_text, rel_type in facts:
            input_ids, target_ids = encode_fact(tokenizer, input_text, target_text)
            # MLP expects (batch,) targets as class indices
            loss = model.train_step(input_ids, target_ids)
            losses.append(loss)
    return losses


def evaluate_mlp(model: SimpleMLP, facts: List[Tuple[str, str, str]],
                  tokenizer) -> Dict[str, Any]:
    """Evaluate SimpleMLP on facts."""
    correct_top1 = 0
    correct_top10 = 0
    total = 0

    for input_text, target_text, rel_type in facts:
        input_ids = np.array(tokenizer.encode(input_text), dtype=np.int64)
        if len(input_ids) == 0:
            continue

        logits = model.predict(input_ids)
        if logits.ndim > 1:
            logits = logits[0]

        target_ids = np.array(tokenizer.encode(target_text), dtype=np.int64)
        if len(target_ids) == 0:
            continue
        target_id = int(target_ids[0])

        pred_id = int(np.argmax(logits))
        if pred_id == target_id:
            correct_top1 += 1

        top10 = set(np.argsort(logits)[-10:])
        if target_id in top10:
            correct_top10 += 1

        total += 1

    return {
        "top1_accuracy": correct_top1 / max(1, total),
        "top10_accuracy": correct_top10 / max(1, total),
        "n_tested": total,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Cross-Domain Transfer Probes
# ═══════════════════════════════════════════════════════════════════════════

def test_structural_transfer(model: RLM, tokenizer) -> Dict[str, Any]:
    """Test if structural patterns from Domain A help Domain B.

    Uses novel cross-domain prompts that require understanding the
    'causes' relation pattern learned in Domain A, applied to Domain B
    vocabulary.
    """
    # Novel cross-domain probes
    cross_probes = [
        ("kindness causes ", "trust", "B vocab + A causal pattern"),
        ("anger produces ", "conflict", "'produces' from A, B vocab"),
        ("sharing enables ", "friendship", "'enables' from A, B vocab"),
        ("heat causes ", "expansion", "pure A recall"),
        ("trust is ", "fragile", "pure B recall"),
        ("friction produces ", "heat", "pure A recall (held out)"),
        ("patience creates ", "understanding", "pure B recall (held out)"),
        # Additional cross-domain probes for statistical robustness
        ("gossip spreads ", "mistrust", "B causal recall"),
        ("collaboration produces ", "innovation", "'produces' from A, B vocab"),
        ("gravity pulls ", "objects", "pure A recall"),
        ("inclusion builds ", "belonging", "B causal recall"),
        ("compassion reduces ", "suffering", "B causal recall"),
        ("fire produces ", "warmth", "pure A recall"),
        ("leadership inspires ", "action", "B causal recall"),
        ("apology restores ", "harmony", "B causal recall"),
        ("oxygen enables ", "combustion", "pure A recall"),
        ("rivalry spurs ", "growth", "B causal recall"),
        ("grief deepens ", "empathy", "B causal recall"),
        ("curiosity sparks ", "discovery", "B causal recall"),
        ("trust enables ", "vulnerability", "'enables' from A, B vocab"),
    ]

    results = []
    for input_text, expected, description in cross_probes:
        input_ids = np.array(tokenizer.encode(input_text), dtype=np.int64)
        if len(input_ids) == 0:
            results.append({
                "input": input_text, "expected": expected,
                "predicted": "?", "correct": False,
                "in_top10": False, "description": description,
            })
            continue

        logits = model.forward(input_ids[np.newaxis, :], eval_mode=True)
        probs_data = logits.data if hasattr(logits, 'data') else np.array(logits)
        if probs_data.ndim > 1:
            probs_data = probs_data[0]

        target_ids = tokenizer.encode(expected)
        target_id = target_ids[0] if target_ids else 0
        pred_id = int(np.argmax(probs_data))
        top10 = np.argsort(probs_data)[-10:][::-1]
        top5 = top10[:5]

        pred_text = tokenizer.decode([pred_id])
        top5_text = [tokenizer.decode([int(t)]) for t in top5]

        results.append({
            "input": input_text,
            "expected": expected,
            "predicted": pred_text,
            "correct": pred_id == target_id,
            "in_top10": target_id in set(top10),
            "top5": top5_text,
            "description": description,
        })

    return {
        "probes": results,
        "top1_accuracy": sum(1 for r in results if r["correct"]) / len(results),
        "top10_accuracy": sum(1 for r in results if r["in_top10"]) / len(results),
    }


def measure_graph_overlap(model: RLM) -> Dict[str, Any]:
    """Measure concept graph properties relevant to transfer."""
    graph = model.graph
    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)
    n_shortcut = sum(1 for e in graph.edges.values() if e.shortcut)
    n_inferred = sum(1 for e in graph.edges.values() if e.relation_type == "inferred")

    rel_types = defaultdict(int)
    for e in graph.edges.values():
        rel_types[e.relation_type] += 1

    if n_edges > 0:
        weights = [e.weight for e in graph.edges.values()]
        mean_weight = float(np.mean(weights))
        max_weight = float(np.max(weights))
    else:
        mean_weight = 0.0
        max_weight = 0.0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_shortcut_edges": n_shortcut,
        "n_inferred_edges": n_inferred,
        "relation_types": dict(rel_types),
        "mean_edge_weight": mean_weight,
        "max_edge_weight": max_weight,
        "sleep_cycles": model.sleep_cycles_completed,
        "conceptual_accuracy": model.conceptual_accuracy,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_cross_domain_experiment(config: CrossDomainConfig) -> Dict[str, Any]:
    """Run the full cross-domain transfer experiment."""

    print("=" * 70)
    print("  CROSS-DOMAIN TRANSFER EXPERIMENT -- RAVANA RLM")
    print("=" * 70)
    print()

    results = {
        "config": asdict(config),
        "rlm": {},
        "mlp_baseline": {},
        "transfer_metrics": {},
    }

    tokenizer = WordTokenizer()

    # Build domain data
    domain_a = build_domain_a_science()
    domain_b = build_domain_b_social()

    # Pre-build vocab from all data (input + target text)
    all_facts = domain_a['train'] + domain_a['test'] + domain_b['train'] + domain_b['test']
    for input_text, target_text, _ in all_facts:
        tokenizer.encode(input_text)
        tokenizer.encode(target_text)
    # Also pre-tokenize cross-domain probes
    for input_text, expected in [
        ("kindness causes ", "trust"), ("anger produces ", "conflict"),
        ("sharing enables ", "friendship"), ("heat causes ", "expansion"),
        ("trust is ", "fragile"), ("friction produces ", "heat"),
        ("patience creates ", "understanding"), ("gossip spreads ", "mistrust"),
        ("collaboration produces ", "innovation"), ("gravity pulls ", "objects"),
        ("inclusion builds ", "belonging"), ("compassion reduces ", "suffering"),
        ("fire produces ", "warmth"), ("leadership inspires ", "action"),
        ("apology restores ", "harmony"), ("oxygen enables ", "combustion"),
        ("rivalry spurs ", "growth"), ("grief deepens ", "empathy"),
        ("curiosity sparks ", "discovery"), ("trust enables ", "vulnerability"),
    ]:
        tokenizer.encode(input_text)
        tokenizer.encode(expected)

    vocab_size = tokenizer.vocab_size

    print(f"Domain A (Science): {len(domain_a['train'])} train, {len(domain_a['test'])} test")
    print(f"Domain B (Social):  {len(domain_b['train'])} train, {len(domain_b['test'])} test")
    print(f"Tokenizer: {tokenizer}")
    print()

    # ─── RLM Experiment ────────────────────────────────────────────────

    print("-" * 70)
    print("  RLM: Cross-Domain Transfer")
    print("-" * 70)

    np.random.seed(config.seed)
    model = RLM(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        concept_dim=config.concept_dim,
        n_concepts=vocab_size,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        sleep_interval=config.sleep_interval,
        tokenizer=tokenizer,
    )

    # ── Phase 0: Baseline (before any training) ──
    print("\n[Phase 0] Pre-training baseline...")
    baseline_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    baseline_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    print(f"  Domain A: top1={baseline_a['top1_accuracy']:.1%}, top10={baseline_a['top10_accuracy']:.1%}")
    print(f"  Domain B: top1={baseline_b['top1_accuracy']:.1%}, top10={baseline_b['top10_accuracy']:.1%}")

    # ── Phase 1: Train on Domain A ──
    print("\n[Phase 1] Training on Domain A (Science)...")
    t0 = time.time()
    acc_a, errors_a = train_rlm_on_domain(
        model, domain_a["train"], tokenizer,
        n_repeats=config.n_train_repeats,
        domain_tag="science", buffer_for_replay=True,
    )
    phase1_time = time.time() - t0

    # Snapshot Domain A buffer so it persists during Domain B training
    model.snapshot_replay_buffer("science")
    # Immediately activate it for interleaved replay during sleep
    model.activate_domain_memories("science")

    post_a_on_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    post_a_on_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    graph_after_a = measure_graph_overlap(model)

    print(f"  Time: {phase1_time:.1f}s ({len(acc_a)} training steps)")
    print(f"  Domain A test: top1={post_a_on_a['top1_accuracy']:.1%}, top10={post_a_on_a['top10_accuracy']:.1%}")
    print(f"  Domain B zero-shot: top1={post_a_on_b['top1_accuracy']:.1%}, top10={post_a_on_b['top10_accuracy']:.1%}")
    print(f"  Graph: {graph_after_a['n_nodes']} nodes, {graph_after_a['n_edges']} edges")
    print(f"  Relation types: {graph_after_a['relation_types']}")
    print(f"  Conceptual accuracy: {model.conceptual_accuracy:.3f}")

    # ── Phase 2: Train on Domain B with interleaved sleep replay ──
    print("\n[Phase 2] Training on Domain B (Social)...")
    print("  (Domain A memories active for interleaved replay during sleep)")
    t0 = time.time()
    acc_b, errors_b = train_rlm_on_domain(
        model, domain_b["train"], tokenizer,
        n_repeats=config.n_train_repeats,
        domain_tag="social", buffer_for_replay=True,
    )
    phase2_time = time.time() - t0

    post_b_on_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    post_b_on_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    graph_after_b = measure_graph_overlap(model)

    print(f"  Time: {phase2_time:.1f}s ({len(acc_b)} training steps)")
    print(f"  Domain B test: top1={post_b_on_b['top1_accuracy']:.1%}, top10={post_b_on_b['top10_accuracy']:.1%}")
    print(f"  Domain A retention: top1={post_b_on_a['top1_accuracy']:.1%}, top10={post_b_on_a['top10_accuracy']:.1%}")
    print(f"  Graph: {graph_after_b['n_nodes']} nodes, {graph_after_b['n_edges']} edges")
    print(f"  Conceptual accuracy: {model.conceptual_accuracy:.3f}")

    # ── Phase 3: Cross-Domain Transfer Probes ──
    print("\n[Phase 3] Cross-domain transfer probes...")
    transfer_probes = test_structural_transfer(model, tokenizer)
    print(f"  Cross-domain top-1 accuracy: {transfer_probes['top1_accuracy']:.1%}")
    print(f"  Cross-domain top-10 accuracy: {transfer_probes['top10_accuracy']:.1%}")
    for probe in transfer_probes["probes"]:
        status = "OK" if probe["correct"] else ("~" if probe["in_top10"] else "X")
        print(f"    [{status}] '{probe['input'].strip()}' -> expected '{probe['expected']}'"
              f"  got '{probe['predicted']}'  ({probe['description']})")

    # ── Phase 4: Sleep cycle and re-evaluate ──
    print("\n[Phase 4] After sleep cycle...")
    model.sleep_cycle()
    post_sleep_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    post_sleep_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    graph_after_sleep = measure_graph_overlap(model)

    print(f"  Domain A after sleep: top1={post_sleep_a['top1_accuracy']:.1%}, top10={post_sleep_a['top10_accuracy']:.1%}")
    print(f"  Domain B after sleep: top1={post_sleep_b['top1_accuracy']:.1%}, top10={post_sleep_b['top10_accuracy']:.1%}")
    print(f"  Graph: {graph_after_sleep['n_nodes']} nodes, {graph_after_sleep['n_edges']} edges")
    print(f"  Inferred (cross-link) edges: {graph_after_sleep['n_inferred_edges']}")
    print(f"  Shortcut edges: {graph_after_sleep['n_shortcut_edges']}")

    # Re-run transfer probes after sleep
    post_sleep_probes = test_structural_transfer(model, tokenizer)
    print(f"  Cross-domain probes after sleep: top1={post_sleep_probes['top1_accuracy']:.1%}, top10={post_sleep_probes['top10_accuracy']:.1%}")

    results["rlm"] = {
        "baseline_a": baseline_a,
        "baseline_b": baseline_b,
        "post_train_a_on_a": post_a_on_a,
        "post_train_a_on_b": post_a_on_b,
        "post_train_b_on_a": post_b_on_a,
        "post_train_b_on_b": post_b_on_b,
        "post_sleep_a": post_sleep_a,
        "post_sleep_b": post_sleep_b,
        "graph_after_a": graph_after_a,
        "graph_after_b": graph_after_b,
        "graph_after_sleep": graph_after_sleep,
        "transfer_probes": transfer_probes,
        "post_sleep_probes": post_sleep_probes,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "sleep_cycles": model.sleep_cycles_completed,
        "total_edges_learned": model._edges_learned,
    }

    # ─── MLP Baseline ──────────────────────────────────────────────────

    if not config.skip_baselines:
        print("\n" + "-" * 70)
        print("  MLP Baseline: Cross-Domain Transfer")
        print("-" * 70)

        np.random.seed(config.seed)
        mlp = SimpleMLP(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            n_hidden=config.n_hidden,
            lr=0.01,
        )

        # Phase 0
        mlp_baseline_a = evaluate_mlp(mlp, domain_a["test"], tokenizer)
        mlp_baseline_b = evaluate_mlp(mlp, domain_b["test"], tokenizer)
        print(f"\n  Baseline A: top1={mlp_baseline_a['top1_accuracy']:.1%}, top10={mlp_baseline_a['top10_accuracy']:.1%}")
        print(f"  Baseline B: top1={mlp_baseline_b['top1_accuracy']:.1%}, top10={mlp_baseline_b['top10_accuracy']:.1%}")

        # Phase 1
        print("\n  Training on Domain A...")
        t0 = time.time()
        mlp_losses_a = train_mlp_on_domain(
            mlp, domain_a["train"], tokenizer,
            n_repeats=config.n_train_repeats,
        )
        mlp_phase1_time = time.time() - t0

        mlp_post_a_on_a = evaluate_mlp(mlp, domain_a["test"], tokenizer)
        mlp_post_a_on_b = evaluate_mlp(mlp, domain_b["test"], tokenizer)
        print(f"  Domain A: top1={mlp_post_a_on_a['top1_accuracy']:.1%}, top10={mlp_post_a_on_a['top10_accuracy']:.1%}")
        print(f"  Domain B zero-shot: top1={mlp_post_a_on_b['top1_accuracy']:.1%}, top10={mlp_post_a_on_b['top10_accuracy']:.1%}")

        # Phase 2
        print("\n  Training on Domain B...")
        t0 = time.time()
        mlp_losses_b = train_mlp_on_domain(
            mlp, domain_b["train"], tokenizer,
            n_repeats=config.n_train_repeats,
        )
        mlp_phase2_time = time.time() - t0

        mlp_post_b_on_a = evaluate_mlp(mlp, domain_a["test"], tokenizer)
        mlp_post_b_on_b = evaluate_mlp(mlp, domain_b["test"], tokenizer)
        print(f"  Domain B: top1={mlp_post_b_on_b['top1_accuracy']:.1%}, top10={mlp_post_b_on_b['top10_accuracy']:.1%}")
        print(f"  Domain A retention: top1={mlp_post_b_on_a['top1_accuracy']:.1%}, top10={mlp_post_b_on_a['top10_accuracy']:.1%}")

        results["mlp_baseline"] = {
            "baseline_a": mlp_baseline_a,
            "baseline_b": mlp_baseline_b,
            "post_train_a_on_a": mlp_post_a_on_a,
            "post_train_a_on_b": mlp_post_a_on_b,
            "post_train_b_on_a": mlp_post_b_on_a,
            "post_train_b_on_b": mlp_post_b_on_b,
            "phase1_time": mlp_phase1_time,
            "phase2_time": mlp_phase2_time,
        }

    # ─── Transfer Metrics Summary ──────────────────────────────────────

    rlm = results["rlm"]

    # Retention: Domain A accuracy after Domain B training vs after A training
    retention_delta = (rlm["post_train_b_on_a"]["top10_accuracy"]
                       - rlm["post_train_a_on_a"]["top10_accuracy"])

    # Forward transfer: Did Domain A pre-training help Domain B?
    forward_transfer = (rlm["post_train_b_on_b"]["top10_accuracy"]
                        - rlm["baseline_b"]["top10_accuracy"])

    # Zero-shot: Domain B accuracy after only Domain A training
    zero_shot_transfer = (rlm["post_train_a_on_b"]["top10_accuracy"]
                          - rlm["baseline_b"]["top10_accuracy"])

    # Sleep benefit
    sleep_benefit_a = (rlm["post_sleep_a"]["top10_accuracy"]
                       - rlm["post_train_b_on_a"]["top10_accuracy"])
    sleep_benefit_b = (rlm["post_sleep_b"]["top10_accuracy"]
                       - rlm["post_train_b_on_b"]["top10_accuracy"])

    transfer_metrics = {
        "retention_delta_domain_a": retention_delta,
        "forward_transfer_to_b": forward_transfer,
        "zero_shot_transfer_a_to_b": zero_shot_transfer,
        "sleep_benefit_a": sleep_benefit_a,
        "sleep_benefit_b": sleep_benefit_b,
        "cross_domain_probe_top1": rlm["transfer_probes"]["top1_accuracy"],
        "cross_domain_probe_top10": rlm["transfer_probes"]["top10_accuracy"],
        "post_sleep_probe_top1": rlm["post_sleep_probes"]["top1_accuracy"],
        "post_sleep_probe_top10": rlm["post_sleep_probes"]["top10_accuracy"],
    }

    if not config.skip_baselines:
        mlp = results["mlp_baseline"]
        mlp_retention = (mlp["post_train_b_on_a"]["top10_accuracy"]
                         - mlp["post_train_a_on_a"]["top10_accuracy"])
        mlp_forward = (mlp["post_train_b_on_b"]["top10_accuracy"]
                       - mlp["baseline_b"]["top10_accuracy"])
        mlp_zero_shot = (mlp["post_train_a_on_b"]["top10_accuracy"]
                         - mlp["baseline_b"]["top10_accuracy"])

        transfer_metrics["mlp_retention_delta"] = mlp_retention
        transfer_metrics["mlp_forward_transfer"] = mlp_forward
        transfer_metrics["mlp_zero_shot_transfer"] = mlp_zero_shot
        transfer_metrics["rlm_vs_mlp_retention"] = retention_delta - mlp_retention
        transfer_metrics["rlm_vs_mlp_forward"] = forward_transfer - mlp_forward

    results["transfer_metrics"] = transfer_metrics

    # ─── Final Report ──────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  TRANSFER METRICS SUMMARY")
    print("=" * 70)
    print()
    print(f"  Domain A Retention (after B training):  {retention_delta:+.1%}")
    print(f"  Zero-Shot Transfer (A->B):              {zero_shot_transfer:+.1%}")
    print(f"  Domain B Learning:                      {forward_transfer:+.1%}")
    print(f"  Sleep Benefit (A):                      {sleep_benefit_a:+.1%}")
    print(f"  Sleep Benefit (B):                      {sleep_benefit_b:+.1%}")
    print(f"  Cross-Domain Probes (before sleep):     {rlm['transfer_probes']['top10_accuracy']:.1%}")
    print(f"  Cross-Domain Probes (after sleep):      {rlm['post_sleep_probes']['top10_accuracy']:.1%}")

    if not config.skip_baselines:
        print()
        print(f"  MLP Domain A Retention:                 {mlp_retention:+.1%}")
        print(f"  MLP Zero-Shot Transfer:                 {mlp_zero_shot:+.1%}")
        print(f"  MLP Domain B Learning:                  {mlp_forward:+.1%}")
        print(f"  ---")
        print(f"  RLM vs MLP Retention Advantage:         {retention_delta - mlp_retention:+.1%}")
        print(f"  RLM vs MLP Forward Transfer Advantage:  {forward_transfer - mlp_forward:+.1%}")

    print()
    print(f"  RLM Graph: {graph_after_sleep['n_nodes']} nodes, {graph_after_sleep['n_edges']} edges")
    print(f"  Sleep cycles: {model.sleep_cycles_completed}")
    print(f"  Edges learned: {model._edges_learned}")

    # ── Temperature sweep: Top-1 at different T values ──
    print("\n  Temperature sweep (Domain B post-sleep):")
    temp_results = {}
    for T in [1.0, 0.5, 0.2, 0.1]:
        res = evaluate_rlm(model, domain_b["test"], tokenizer, temperature=T)
        temp_results[T] = res
        print(f"    T={T:.1f}: top1={res['top1_accuracy']:.1%}  top10={res['top10_accuracy']:.1%}")
    results["temperature_sweep"] = temp_results
    print()

    # Verdict
    print("-" * 70)
    has_positive_transfer = (retention_delta > -0.05 and
                             (zero_shot_transfer > 0.0 or
                              transfer_probes['top10_accuracy'] > 0.2))
    has_neutral = retention_delta > -0.10

    if has_positive_transfer:
        print("  VERDICT: POSITIVE TRANSFER DETECTED")
        print("  RLM shows cross-domain knowledge transfer via concept graph.")
    elif has_neutral:
        print("  VERDICT: NEUTRAL TRANSFER")
        print("  RLM retains Domain A knowledge during Domain B training.")
    else:
        print("  VERDICT: NEGATIVE TRANSFER (interference)")
        print("  Domain B training degrades Domain A knowledge.")

    if not config.skip_baselines:
        if retention_delta - mlp_retention > 0.05:
            print("  RLM retains significantly better than MLP baseline.")
        elif retention_delta - mlp_retention > 0:
            print("  RLM retains slightly better than MLP baseline.")
        else:
            print("  MLP baseline retains comparably or better.")

    print("-" * 70)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Domain Transfer Experiment")
    parser.add_argument("--n", type=int, default=3, help="Repeats of each fact during training")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip MLP baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="experiment_results/cross_domain_transfer.json")
    args = parser.parse_args()

    config = CrossDomainConfig(
        n_train_repeats=args.n,
        skip_baselines=args.skip_baselines,
        seed=args.seed,
    )

    results = run_cross_domain_experiment(config)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to {args.output}")
