"""
Cross-Domain Transfer Experiment for RAVANA RLMv2

Ported to RLMv2 (triple decomposition + spreading activation + Relation Predictor MLP).
Tests whether knowledge learned in Domain A transfers to Domain B.
Includes programmatically injected abstract cross-domain bridge nodes.
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

    # RLMv2 architecture
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

    # Split into train/test: hold out ~20% of each category for evaluation
    causal = [f for f in facts if f[2] == "causal"]
    semantic = [f for f in facts if f[2] == "semantic"]
    rng = np.random.RandomState(42)
    rng.shuffle(causal)
    rng.shuffle(semantic)
    n_causal_test = max(1, len(causal) // 5)
    n_semantic_test = max(1, len(semantic) // 5)
    train = causal[n_causal_test:] + semantic[n_semantic_test:]
    test = causal[:n_causal_test] + semantic[:n_semantic_test]
    return {"train": train, "test": test}


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

    # Split into train/test: hold out ~20% of each category for evaluation
    causal = [f for f in facts if f[2] == "causal"]
    semantic = [f for f in facts if f[2] == "semantic"]
    rng = np.random.RandomState(42)
    rng.shuffle(causal)
    rng.shuffle(semantic)
    n_causal_test = max(1, len(causal) // 5)
    n_semantic_test = max(1, len(semantic) // 5)
    train = causal[n_causal_test:] + semantic[n_semantic_test:]
    test = causal[:n_causal_test] + semantic[:n_semantic_test]
    return {"train": train, "test": test}


# ═══════════════════════════════════════════════════════════════════════════
# Training & Evaluation Helpers (RLMv2 style)
# ═══════════════════════════════════════════════════════════════════════════

def encode_fact(tokenizer, input_text: str, target_text: str):
    """Encode a (input, target) pair into token arrays."""
    input_ids = np.array(tokenizer.encode(input_text), dtype=np.int64)
    target_ids = np.array(tokenizer.encode(target_text), dtype=np.int64)
    return input_ids, target_ids


def train_rlm_on_domain(model: RLMv2, facts: List[Tuple[str, str, str]],
                         tokenizer, n_repeats: int = 3,
                         domain_tag: Optional[str] = None,
                         buffer_for_replay: bool = False):
    """Train RLMv2 on a set of facts."""
    acc_history = []
    errors = []

    for repeat in range(n_repeats):
        losses = []
        correct = 0
        total = 0
        for input_text, target_text, rel_type in facts:
            input_ids, target_ids = encode_fact(tokenizer, input_text, target_text)
            err = model.learn(input_ids, target_ids)
            errors.append(err)
            acc = err.get("accuracy", 0.0)
            acc_history.append(acc)
            losses.append(err.get("loss", 0.0))
            if err.get("is_correct", False):
                correct += 1
            total += 1
        if repeat % 5 == 0 or repeat == n_repeats - 1:
            avg_loss = np.mean(losses)
            epoch_acc = correct / total
            print(f"  [Train {domain_tag or ''}] Repeat {repeat:2d} Loss: {avg_loss:.6f} Acc: {epoch_acc:.1%}")

    return acc_history, errors


def evaluate_rlm(model: RLMv2, facts: List[Tuple[str, str, str]],
                  tokenizer, temperature: float = 1.0) -> Dict[str, Any]:
    """Evaluate RLMv2 on a set of facts."""
    correct_top1 = 0
    correct_top10 = 0
    total = 0

    for input_text, target_text, rel_type in facts:
        input_ids = np.array(tokenizer.encode(input_text), dtype=np.int64)
        target_ids = np.array(tokenizer.encode(target_text), dtype=np.int64)
        if len(input_ids) == 0 or len(target_ids) == 0:
            continue

        logits = model.forward(input_ids)
        probs_data = logits.data.flatten()
        target_id = int(target_ids[0])

        pred_id = int(np.argmax(probs_data))
        if pred_id == target_id:
            correct_top1 += 1

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
        for input_text, target_text, _ in facts:
            input_ids, target_ids = encode_fact(tokenizer, input_text, target_text)
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


def test_structural_transfer(model: RLMv2, tokenizer,
                             domain_a_test=None,
                             domain_b_test=None) -> Dict[str, Any]:
    """Test if structural patterns from Domain A help Domain B."""
    cross_probes = []

    if domain_a_test is not None:
        for input_text, target_text, rel_type in domain_a_test:
            cross_probes.append((input_text, target_text, f"held-out A ({rel_type})"))
    if domain_b_test is not None:
        for input_text, target_text, rel_type in domain_b_test:
            cross_probes.append((input_text, target_text, f"held-out B ({rel_type})"))

    if domain_b_test is not None:
        a_causal_verbs = ["causes ", "produces ", "enables ", "creates ",
                          "drives ", "shapes "]
        b_causal_facts = [(i, t) for i, t, r in domain_b_test if r == "causal"]
        for verb_idx, (orig_input, orig_target) in enumerate(
                b_causal_facts[:min(6, len(b_causal_facts))]):
            subject = orig_input.split()[0]
            verb = a_causal_verbs[verb_idx % len(a_causal_verbs)]
            new_input = f"{subject} {verb}"
            cross_probes.append((new_input, orig_target,
                                 f"cross-domain (A verb '{verb.strip()}' + B vocab)"))

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

        logits = model.forward(input_ids)
        probs_data = logits.data.flatten()

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

    n = max(1, len(results))
    return {
        "probes": results,
        "top1_accuracy": sum(1 for r in results if r["correct"]) / n,
        "top10_accuracy": sum(1 for r in results if r["in_top10"]) / n,
    }


def measure_graph_overlap(model: RLMv2) -> Dict[str, Any]:
    """Measure graph properties relevant to transfer."""
    graph = model.graph
    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)

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
        "n_shortcut_edges": 0,
        "n_inferred_edges": 0,
        "relation_types": dict(rel_types),
        "mean_edge_weight": mean_weight,
        "max_edge_weight": max_weight,
        "sleep_cycles": model.sleep_cycles_completed if hasattr(model, 'sleep_cycles_completed') else 0,
        "conceptual_accuracy": getattr(model, 'conceptual_accuracy', 0.0),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Abstract Bridge Injection
# ═══════════════════════════════════════════════════════════════════════════

def add_abstract_bridge(model: RLMv2, label: str, source_token: str, target_token: str, relation_type: str, weight: float = 0.8):
    """Add an abstract relation node linking source and target concepts."""
    tok = model._tokenizer
    src_tid = tok.encode(source_token)[0]
    tgt_tid = tok.encode(target_token)[0]
    src_cid = model._get_or_create_concept(src_tid, model.token_embed.weight.data[src_tid])
    tgt_cid = model._get_or_create_concept(tgt_tid, model.token_embed.weight.data[tgt_tid])
    
    src_node = model.graph.get_node(src_cid)
    tgt_node = model.graph.get_node(tgt_cid)
    
    # Geometrically blend the representations to sit between them in embedding space
    bridge_vec = 0.5 * (src_node.vector + tgt_node.vector)
    bridge_vec_norm = np.linalg.norm(bridge_vec)
    if bridge_vec_norm > 0:
        bridge_vec /= bridge_vec_norm
        
    bridge_cid = model.graph.add_node(bridge_vec, label=label)
    
    # Create the analogical bridge links: src -> bridge -> tgt
    model.graph.add_edge(src_cid, bridge_cid, weight=weight, relation_type="semantic")
    model.graph.add_edge(bridge_cid, tgt_cid, weight=weight, relation_type=relation_type)
    
    print(f"    Injected abstract node: '{label}' connecting '{source_token}' -> '{label}' -> '{target_token}'")


# ═══════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════

def run_cross_domain_experiment(config: CrossDomainConfig) -> Dict[str, Any]:
    """Run the full cross-domain transfer experiment."""

    print("=" * 70)
    print("  CROSS-DOMAIN TRANSFER EXPERIMENT -- RAVANA RLMv2")
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

    # Pre-build vocab
    all_facts = domain_a['train'] + domain_a['test'] + domain_b['train'] + domain_b['test']
    for input_text, target_text, _ in all_facts:
        tokenizer.encode(input_text)
        tokenizer.encode(target_text)
    # Pre-tokenize cross-domain probes
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
    print("  RLMv2: Cross-Domain Transfer")
    print("-" * 70)

    np.random.seed(config.seed)
    model = RLMv2(
        vocab_size=vocab_size + 5,
        embed_dim=config.embed_dim,
        concept_dim=config.concept_dim,
        n_concepts=vocab_size,
        sleep_interval=config.sleep_interval,
    )
    model._tokenizer = tokenizer

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
        domain_tag="science", buffer_for_replay=False,
    )
    phase1_time = time.time() - t0

    post_a_on_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    post_a_on_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    graph_after_a = measure_graph_overlap(model)

    print(f"  Time: {phase1_time:.1f}s ({len(acc_a)} training steps)")
    print(f"  Domain A test: top1={post_a_on_a['top1_accuracy']:.1%}, top10={post_a_on_a['top10_accuracy']:.1%}")
    print(f"  Domain B zero-shot: top1={post_a_on_b['top1_accuracy']:.1%}, top10={post_a_on_b['top10_accuracy']:.1%}")
    print(f"  Graph: {graph_after_a['n_nodes']} nodes, {graph_after_a['n_edges']} edges")

    # ── Phase 1.8: Inject Abstract Cross-Domain Bridge Nodes ──
    print("\n[Phase 1.8] Injecting abstract cross-domain bridge nodes...")
    # "anger" -> is -> "intense_bridge" -> causes -> "expansion"
    add_abstract_bridge(model, "intense_bridge", "anger", "expansion", "causal", weight=0.8)
    # "kindness" -> is -> "warm_bridge" -> causes -> "trust"
    add_abstract_bridge(model, "warm_bridge", "kindness", "trust", "causal", weight=0.8)

    # ── Phase 1.9: Zero-shot cross-domain probes (before Domain B training) ──
    print("\n[Phase 1.9] Zero-shot cross-domain probes (before Domain B training)...")
    zero_shot_probes = test_structural_transfer(
        model, tokenizer, domain_a["test"], domain_b["test"])
    print(f"  Zero-shot probe top-1: {zero_shot_probes['top1_accuracy']:.1%}")
    print(f"  Zero-shot probe top-10: {zero_shot_probes['top10_accuracy']:.1%}")
    for probe in zero_shot_probes["probes"]:
        status = "OK" if probe["correct"] else ("~" if probe["in_top10"] else "X")
        print(f"    [{status}] '{probe['input'].strip()}' -> expected '{probe['expected']}'"
              f"  got '{probe['predicted']}'  ({probe['description']})")

    # ── Phase 2: Train on Domain B ──
    print("\n[Phase 2] Training on Domain B (Social)...")
    t0 = time.time()
    acc_b, errors_b = train_rlm_on_domain(
        model, domain_b["train"], tokenizer,
        n_repeats=config.n_train_repeats,
        domain_tag="social", buffer_for_replay=False,
    )
    phase2_time = time.time() - t0

    post_b_on_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    post_b_on_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    graph_after_b = measure_graph_overlap(model)

    print(f"  Time: {phase2_time:.1f}s ({len(acc_b)} training steps)")
    print(f"  Domain B test: top1={post_b_on_b['top1_accuracy']:.1%}, top10={post_b_on_b['top10_accuracy']:.1%}")
    print(f"  Domain A retention: top1={post_b_on_a['top1_accuracy']:.1%}, top10={post_b_on_a['top10_accuracy']:.1%}")
    print(f"  Graph: {graph_after_b['n_nodes']} nodes, {graph_after_b['n_edges']} edges")

    # ── Phase 3: Cross-Domain Transfer Probes ──
    print("\n[Phase 3] Cross-domain transfer probes...")
    transfer_probes = test_structural_transfer(
        model, tokenizer, domain_a["test"], domain_b["test"])
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

    # Re-run transfer probes after sleep
    post_sleep_probes = test_structural_transfer(
        model, tokenizer, domain_a["test"], domain_b["test"])
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
        "zero_shot_probes": zero_shot_probes,
        "transfer_probes": transfer_probes,
        "post_sleep_probes": post_sleep_probes,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "sleep_cycles": model.sleep_cycles_completed if hasattr(model, 'sleep_cycles_completed') else 0,
        "total_edges_learned": len(model.graph.edges),
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
        )

        mlp_losses_a = []
        for repeat in range(config.n_train_repeats):
            for input_text, target_text, _ in domain_a["train"]:
                input_ids, target_ids = encode_fact(tokenizer, input_text, target_text)
                loss = mlp.train_step(input_ids, target_ids)
                mlp_losses_a.append(loss)

        mlp_post_a_on_a = evaluate_mlp(mlp, domain_a["test"], tokenizer)
        mlp_post_a_on_b = evaluate_mlp(mlp, domain_b["test"], tokenizer)

        mlp_losses_b = []
        for repeat in range(config.n_train_repeats):
            for input_text, target_text, _ in domain_b["train"]:
                input_ids, target_ids = encode_fact(tokenizer, input_text, target_text)
                loss = mlp.train_step(input_ids, target_ids)
                mlp_losses_b.append(loss)

        mlp_post_b_on_a = evaluate_mlp(mlp, domain_a["test"], tokenizer)
        mlp_post_b_on_b = evaluate_mlp(mlp, domain_b["test"], tokenizer)

        print(f"  MLP Domain A retention: top1={mlp_post_b_on_a['top1_accuracy']:.1%}, top10={mlp_post_b_on_a['top10_accuracy']:.1%}")
        print(f"  MLP Domain B test:      top1={mlp_post_b_on_b['top1_accuracy']:.1%}, top10={mlp_post_b_on_b['top10_accuracy']:.1%}")

        results["mlp_baseline"] = {
            "post_train_a_on_a": mlp_post_a_on_a,
            "post_train_a_on_b": mlp_post_a_on_b,
            "post_train_b_on_a": mlp_post_b_on_a,
            "post_train_b_on_b": mlp_post_b_on_b,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Domain Transfer Experiment (RLMv2)")
    parser.add_argument("--n-repeats", type=int, default=3, help="Training repeats per fact")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="experiment_results/cross_domain_v2.json")
    args = parser.parse_args()

    config = CrossDomainConfig(
        n_train_repeats=args.n_repeats,
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
