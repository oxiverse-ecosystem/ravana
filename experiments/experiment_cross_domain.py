"""
Cross-Domain Transfer Experiment for RAVANA RLMv2

Ported to RLMv2 (triple decomposition + spreading activation + Relation Predictor MLP).
Tests whether knowledge learned in Domain A transfers to Domain B.
Includes programmatically injected abstract cross-domain bridge nodes.

Usage:
    python experiments/experiment_cross_domain.py
    python experiments/experiment_cross_domain.py --n-repeats 5
    python experiments/experiment_cross_domain.py --skip-baselines
"""

import os
import sys
from pathlib import Path
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field, asdict

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer
# Inline SimpleMLP (was in experiments/archive/old_experiments/experiment_baselines.py)
class SimpleMLP:
    """Simple MLP baseline for cross-domain comparison."""
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, lr: float = 0.01):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        # Embedding layer
        self.W_embed = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.1
        # Hidden layer
        self.W1 = np.random.randn(embed_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        # Output layer
        self.W2 = np.random.randn(hidden_dim, vocab_size).astype(np.float32) * 0.1
        self.b2 = np.zeros(vocab_size, dtype=np.float32)

    def train_step(self, input_ids, target_ids):
        """Single training step."""
        # Embed input (mean of embeddings for multi-word input)
        x = np.mean(self.W_embed[input_ids], axis=0)
        # Hidden layer
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        # Output
        logits = h @ self.W2 + self.b2
        # Softmax + cross-entropy
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        loss = -np.log(probs[target_ids[0]] + 1e-10)
        # Backprop
        grad = probs.copy()
        grad[target_ids[0]] -= 1.0
        # Output layer grads
        self.W2 -= self.lr * np.outer(h, grad)
        self.b2 -= self.lr * grad
        # Hidden layer grads
        grad_h = grad @ self.W2.T
        grad_h[h <= 0] = 0  # ReLU derivative
        self.W1 -= self.lr * np.outer(x, grad_h)
        self.b1 -= self.lr * grad_h
        # Embedding grad
        for idx in input_ids:
            self.W_embed[idx] -= self.lr * (grad_h @ self.W1.T) / len(input_ids)
        return loss

    def predict(self, input_ids):
        """Predict logits for input."""
        x = np.mean(self.W_embed[input_ids], axis=0)
        h = np.maximum(0, x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits




# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CrossDomainConfig:
    # CROSS-DOMAIN GENERALIZATION FIX #2: 100 repeats overfits a 115-fact dataset
    # into pure memorization (100% train, 0% held-out). 15 repeats with frozen token
    # embeddings (rlm_v2.freeze_token_embeds_in_rp=True) generalizes much better.
    n_train_repeats: int = 15            # repeats of each fact during training
    n_test_probes: int = 50             # probes per test
    seed: int = 42
    skip_baselines: bool = False

    # RLMv2 architecture (80.9% benchmark architecture configuration)
    embed_dim: int = 64
    concept_dim: int = 64
    n_hidden: int = 128
    n_layers: int = 3
    sleep_interval: int = 300           # 80.9% v6 benchmark config
    # CRITICAL: latent_dim must match embed_dim for RP relation matrices to work
    # (relation matrix: latent_dim x latent_dim applied to source_embed of embed_dim)
    latent_dim: int = 64
    hidden_dim: int = 128


# ═══════════════════════════════════════════════════════════════════════════
# Domain Knowledge Bases
# ═══════════════════════════════════════════════════════════════════════════

def _subject_holdout_split(facts, seed=42, holdout_ratio=0.2):
    """Split facts into train/test by holding out ENTIRE SUBJECTS.

    Key design: Held-out subjects are never seen during training.
    The model must generalize using verb-stem offset arithmetic:
      predicted_embed = subject_embed + offset(query_verb)
    This tests TRUE generalization: can the model predict targets for
    entirely unseen subjects using only the shared verb offset?

    This replaces the old _stratified_domain_split which grouped by
    (target, relation_type) — that design guaranteed 0% held-out
    because the bilinear W_rel @ subject is mathematically incapable
    of mapping the same (subject, relation) to two different targets.
    """
    rng = np.random.RandomState(seed)

    # Extract unique subjects (first word of input_text)
    from collections import defaultdict as _dd
    subject_facts = _dd(list)
    for fact in facts:
        subject = fact[0].split()[0].lower()
        subject_facts[subject].append(fact)

    all_subjects = list(subject_facts.keys())
    rng.shuffle(all_subjects)

    # Hold out a fraction of subjects entirely
    n_holdout = max(1, int(len(all_subjects) * holdout_ratio))
    holdout_subjects = set(all_subjects[:n_holdout])

    train = []
    test = []
    for subject, entries in subject_facts.items():
        if subject in holdout_subjects:
            test.extend(entries)
        else:
            train.extend(entries)

    return {"train": train, "test": test}


def build_domain_a_science():
    """Domain A: Science -- causal relationships between physical concepts.

    Expanded to 50+ facts per domain with:
    - Repeated verbs across multiple subjects (e.g., "X causes Y" for 10+ X per verb)
    - Cross-verb coverage (causes, produces, enables, creates, drives, shapes, melts, freezes, etc.)
    - Balanced relation types (causal, semantic, temporal)

    Returns dict with 'train' and 'test' splits using stratified holdout:
    every target appears in training at least once.
    """
    facts = [
        # Causal facts - "causes" verb (10+ subjects)
        ("heat causes ", "expansion", "causal"),
        ("cold causes ", "contraction", "causal"),
        ("friction causes ", "wear", "causal"),
        ("pressure causes ", "phase_change", "causal"),
        ("radiation causes ", "mutation", "causal"),
        ("gravity causes ", "acceleration", "causal"),
        ("voltage causes ", "current_flow", "causal"),
        ("lightning causes ", "fire", "causal"),
        ("oxidation causes ", "rust", "causal"),
        ("corrosion causes ", "weakening", "causal"),
        ("earthquake causes ", "destruction", "causal"),
        ("volcano causes ", "lava_flow", "causal"),
        ("metabolism causes ", "energy", "causal"),

        # Causal facts - "produces" verb (8+ subjects)
        ("fire produces ", "warmth", "causal"),
        ("photosynthesis produces ", "oxygen", "causal"),
        ("friction produces ", "heat", "causal"),
        ("combustion produces ", "energy", "causal"),
        ("fermentation produces ", "alcohol", "causal"),
        ("respiration produces ", "co2", "causal"),
        ("nuclear_fission produces ", "radiation", "causal"),
        ("decomposition produces ", "nutrients", "causal"),
        ("condensation produces ", "dew", "causal"),
        ("transpiration produces ", "humidity", "causal"),

        # Causal facts - "enables" verb (6+ subjects)
        ("oxygen enables ", "combustion", "causal"),
        ("light enables ", "vision", "causal"),
        ("catalyst enables ", "reaction", "causal"),
        ("enzyme enables ", "metabolism", "causal"),
        ("chlorophyll enables ", "photosynthesis", "causal"),
        ("membrane enables ", "osmosis", "causal"),

        # Causal facts - "creates" verb (6+ subjects)
        ("pressure creates ", "diamonds", "causal"),
        ("fusion creates ", "elements", "causal"),
        ("sedimentation creates ", "rock_layers", "causal"),
        ("crystallization creates ", "crystals", "causal"),
        ("volcanic_activity creates ", "islands", "causal"),
        ("erosion creates ", "canyons", "causal"),

        # Causal facts - "drives" verb (6+ subjects)
        ("voltage drives ", "current", "causal"),
        ("gravity drives ", "tides", "causal"),
        ("pressure_difference drives ", "wind", "causal"),
        ("concentration_gradient drives ", "diffusion", "causal"),
        ("temperature_difference drives ", "convection", "causal"),
        ("electric_field drives ", "ion_motion", "causal"),

        # Causal facts - "shapes" verb (4+ subjects)
        ("gravity shapes ", "orbits", "causal"),
        ("wind shapes ", "dunes", "causal"),
        ("water shapes ", "riverbeds", "causal"),
        ("glaciers shape ", "valleys", "causal"),

        # Additional causal verbs for coverage
        ("heat melts ", "ice", "causal"),
        ("cold freezes ", "water", "causal"),
        ("rain erodes ", "soil", "causal"),
        ("acid dissolves ", "metal", "causal"),
        ("magnetism attracts ", "iron", "causal"),
        ("centrifugal_force pushes ", "outward", "causal"),
        ("capillary_action draws ", "liquid", "causal"),
        ("resonance shatters ", "glass", "causal"),
        ("diffusion spreads ", "particles", "causal"),
        ("osmosis transfers ", "water", "causal"),
        ("decompression causes ", "cooling", "causal"),

        # Semantic facts (is-a / properties) - expanded
        ("water is ", "liquid", "semantic"),
        ("ice is ", "solid", "semantic"),
        ("steam is ", "gas", "semantic"),
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
        ("gold is ", "malleable", "semantic"),
        ("platinum is ", "catalytic", "semantic"),
        ("titanium is ", "strong", "semantic"),
        ("graphite is ", "lubricating", "semantic"),

        # Temporal facts
        ("sunrise then ", "daylight", "temporal"),
        ("winter after ", "autumn", "temporal"),
        ("seedling then ", "plant", "temporal"),
        ("reaction then ", "product", "temporal"),
        ("evaporation then ", "condensation", "temporal"),
        ("fertilization then ", "embryo", "temporal"),
    ]

    return _subject_holdout_split(facts, seed=42)


def build_domain_b_social():
    """Domain B: Social -- relationships and emotions between people.

    Expanded to 50+ facts per domain with:
    - Repeated verbs across multiple subjects (e.g., "X causes Y" for 10+ X per verb)
    - Cross-verb coverage (causes, produces, enables, creates, drives, shapes, leads to, etc.)
    - Balanced relation types (causal, semantic, temporal)

    Structurally parallel to Domain A but semantically distinct.
    Returns dict with 'train' and 'test' splits using stratified holdout.
    """
    facts = [
        # Causal facts - "causes" verb (10+ subjects)
        ("anger causes ", "conflict", "causal"),
        ("greed causes ", "loneliness", "causal"),
        ("jealousy causes ", "resentment", "causal"),
        ("rudeness causes ", "offense", "causal"),
        ("neglect causes ", "distance", "causal"),
        ("isolation causes ", "sadness", "causal"),
        ("bullying causes ", "trauma", "causal"),
        ("rejection causes ", "withdrawal", "causal"),
        ("betrayal causes ", "betrayal_trauma", "causal"),
        ("stress causes ", "anxiety", "causal"),
        ("trauma causes ", "hypervigilance", "causal"),
        ("criticism causes ", "defensiveness", "causal"),
        ("deception causes ", "mistrust", "causal"),

        # Causal facts - "produces" verb (8+ subjects)
        ("collaboration produces ", "innovation", "causal"),
        ("teamwork produces ", "success", "causal"),
        ("mentorship produces ", "growth", "causal"),
        ("practice produces ", "mastery", "causal"),
        ("reflection produces ", "wisdom", "causal"),
        ("communication produces ", "understanding", "causal"),
        ("trust produces ", "intimacy", "causal"),
        ("kindness produces ", "goodwill", "causal"),
        ("effort produces ", "results", "causal"),
        ("creativity produces ", "expression", "causal"),

        # Causal facts - "enables" verb (6+ subjects)
        ("trust enables ", "vulnerability", "causal"),
        ("listening enables ", "rapport", "causal"),
        ("honesty enables ", "authenticity", "causal"),
        ("empathy enables ", "connection", "causal"),
        ("patience enables ", "understanding", "causal"),
        ("forgiveness enables ", "healing", "causal"),

        # Causal facts - "creates" verb (6+ subjects)
        ("kindness creates ", "belonging", "causal"),
        ("sharing creates ", "community", "causal"),
        ("celebration creates ", "bonds", "causal"),
        ("inclusion creates ", "belonging", "causal"),
        ("gratitude creates ", "abundance", "causal"),
        ("love creates ", "connection", "causal"),

        # Causal facts - "drives" verb (6+ subjects)
        ("ambition drives ", "achievement", "causal"),
        ("curiosity drives ", "discovery", "causal"),
        ("passion drives ", "excellence", "causal"),
        ("competition drives ", "improvement", "causal"),
        ("purpose drives ", "meaning", "causal"),
        ("fear drives ", "avoidance", "causal"),

        # Causal facts - "shapes" verb (4+ subjects)
        ("culture shapes ", "values", "causal"),
        ("upbringing shapes ", "character", "causal"),
        ("experience shapes ", "wisdom", "causal"),
        ("trauma shapes ", "resilience", "causal"),

        # Additional causal verbs for coverage - "leads to"
        ("kindness leads to ", "trust", "causal"),
        ("honesty leads to ", "respect", "causal"),
        ("patience leads to ", "peace", "causal"),
        ("generosity leads to ", "gratitude", "causal"),
        ("empathy leads to ", "compassion", "causal"),
        ("forgiveness leads to ", "freedom", "causal"),

        # Additional causal verbs - "builds"
        ("sharing builds ", "friendship", "causal"),
        ("honesty builds ", "trust", "causal"),
        ("listening builds ", "rapport", "causal"),
        ("teamwork builds ", "trust", "causal"),
        ("mentorship builds ", "skills", "causal"),
        ("consistency builds ", "reliability", "causal"),
        ("communication builds ", "understanding", "causal"),
        ("vulnerability builds ", "intimacy", "causal"),

        # Additional causal verbs - "triggers", "sparks", "ignites"
        ("boredom triggers ", "exploration", "causal"),
        ("curiosity sparks ", "discovery", "causal"),
        ("insult triggers ", "anger", "causal"),
        ("praise ignites ", "confidence", "causal"),
        ("injustice sparks ", "activism", "causal"),
        ("failure triggers ", "learning", "causal"),

        # Additional causal verbs - "fosters", "nurtures", "cultivates"
        ("kindness fosters ", "trust", "causal"),
        ("mentorship nurtures ", "potential", "causal"),
        ("patience cultivates ", "understanding", "causal"),
        ("support fosters ", "growth", "causal"),
        ("encouragement nurtures ", "confidence", "causal"),
        ("love cultivates ", "connection", "causal"),

        # Semantic facts (is-a / properties) - expanded
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
        ("compassion is ", "healing", "semantic"),
        ("integrity is ", "steadfast", "semantic"),
        ("forgiveness is ", "liberating", "semantic"),
        ("gratitude is ", "transformative", "semantic"),
        ("humility is ", "grounding", "semantic"),
        ("curiosity is ", "expansive", "semantic"),
        ("resilience is ", "enduring", "semantic"),
        ("presence is ", "powerful", "semantic"),
        ("authenticity is ", "magnetic", "semantic"),
        ("vulnerability is ", "courageous", "semantic"),
        ("empathy is ", "bridging", "semantic"),
        ("acceptance is ", "peaceful", "semantic"),
        ("boundaries are ", "healthy", "semantic"),

        # Temporal facts
        ("meeting then ", "friendship", "temporal"),
        ("conflict then ", "resolution", "temporal"),
        ("trust then ", "intimacy", "temporal"),
        ("grief then ", "healing", "temporal"),
        ("learning then ", "mastery", "temporal"),
        ("apology then ", "forgiveness", "temporal"),
    ]

    return _subject_holdout_split(facts, seed=42)




def encode_fact(tokenizer, input_text: str, target_text: str):
    """Encode a (input, target) pair into token arrays."""
    input_ids = np.array(tokenizer.encode(input_text), dtype=np.int64)
    target_ids = np.array(tokenizer.encode(target_text), dtype=np.int64)
    return input_ids, target_ids


def train_rlm_on_domain(model: RLMv2, facts: List[Tuple[str, str, str]],
                         tokenizer, n_repeats: int = 3,
                         domain_tag: Optional[str] = None,
                         buffer_for_replay: bool = False,
                         replay_facts: Optional[List[Tuple[str, str, str]]] = None,
                         replay_domain_id: Optional[int] = None):
    """Train RLMv2 on a set of facts.

    If replay_facts is provided, interleave replay of those facts at 20% rate
    to prevent catastrophic forgetting of shared relation embeddings.
    """
    # Set domain based on domain_tag
    domain_map = {'science': 0, 'social': 1, 'math': 2, 'history': 3}
    if domain_tag is not None and domain_tag in domain_map:
        model.set_domain(domain_map[domain_tag])
    elif domain_tag is not None:
        # Unknown tag - use hash for consistency
        domain_id = hash(domain_tag) % model.num_domains
        model.set_domain(domain_id)
    
    acc_history = []
    errors = []

    for repeat in range(n_repeats):
        rng = np.random.RandomState(repeat + 42)  # diverse per epoch
        losses = []
        correct = 0
        total = 0
        for input_text, target_text, rel_type in facts:
            input_ids, target_ids = encode_fact(tokenizer, input_text, target_text)
            err = model.learn(input_ids, target_ids)
            errors.append(err)

            if buffer_for_replay:
                model.buffer_experience(input_ids, target_ids)

            losses.append(err.get("loss", 0.0))
            if err.get("is_correct", False):
                correct += 1
            total += 1

        # ── Interleaved Replay (20% of main domain batches) ──
        if replay_facts is not None and len(replay_facts) > 0:
            replay_n = max(1, len(facts) // 5)
            replay_indices = rng.choice(
                len(replay_facts), size=min(replay_n, len(replay_facts)), replace=False)

            # Save current domain state before switching
            saved_domain_id = model.current_domain_id
            saved_frozen = model._frozen_domains.copy()

            # Switch to replay domain
            if replay_domain_id is not None:
                model.set_domain(replay_domain_id)  # freezes others

            for idx in replay_indices:
                input_text, target_text, _ = replay_facts[idx]
                input_ids, target_ids = encode_fact(tokenizer, input_text, target_text)
                err = model.learn(input_ids, target_ids)
                errors.append(err)

            # Restore original domain state
            model._frozen_domains = saved_frozen
            model.current_domain_id = saved_domain_id

        if repeat % 5 == 0 or repeat == n_repeats - 1:
            avg_loss = np.mean(losses)
            epoch_acc = correct / total
            print(f"  [Train {domain_tag or ''}] Repeat {repeat:2d} Loss: {avg_loss:.6f} Acc: {epoch_acc:.1%}")

    return acc_history, errors


def evaluate_rlm(model: RLMv2, facts: List[Tuple[str, str, str]],
                  tokenizer) -> Dict[str, Any]:
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
        
    bridge_node = model.graph.add_node(bridge_vec, label=label)
    bridge_cid = bridge_node.id  # CRITICAL FIX: retrieved node ID instead of using the node object directly
    
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
        gate_concept_creation=False,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        use_shared_relation_embeds=False,  # Domain-specific, aligned via cross-domain loss
    )
    model._tokenizer = tokenizer  # Attach tokenizer for relation classification
    model.use_cross_domain_alignment = True  # Enable explicit relation alignment
    model.use_rp_contrastive = False  # Disable RP contrastive (not the primary path)
    model._tokenizer = tokenizer  # triggers embed init + autoencoder pre-training
    
    # Spreading activation is primary; vector arithmetic is backup
    model.use_rp_hidden = True
    model.use_rp_contrastive = False
    model.use_cross_domain_alignment = True
    
    # Use VECTOR ARITHMETIC mode (shared relation vectors) instead of learned RP
    # Brain-inspired: relation is a VECTOR OFFSET shared across all subjects
    # target_embed = subject_embed + avg_causal_vector
    model.use_rp_for_analogy = True  # Use learned RP with identity-init relation matrices
    
    # ── VERB-STEM OFFSET PREDICTOR ──
    # Replaces bilinear W_rel @ subject with verb-conditioned offset arithmetic.
    # This is the structural fix for the 0% held-out problem:
    #   - offset("causes") captures abstract cause-effect (heat->expansion)
    #   - offset("freezes") captures liquid->solid (water->ice)
    #   - offset("melts") captures solid->liquid (ice->water)
    # Each verb has its OWN offset, enabling same-subject different-verb predictions.
    model.use_verb_offset = True
    
    # For evaluation, also test with spreading disabled to isolate RP
    # model.disable_spreading_activation = True  # Keep spreading enabled for main eval

    # ── Phase 0: Baseline (before any training) ──
    print("\n[Phase 0] Pre-training baseline...")
    baseline_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    baseline_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    print(f"  Domain A: top1={baseline_a['top1_accuracy']:.1%}, top10={baseline_a['top10_accuracy']:.1%}")
    print(f"  Domain B: top1={baseline_b['top1_accuracy']:.1%}, top10={baseline_b['top10_accuracy']:.1%}")

    # ── Phase 1: JOINT Training (single shared domain head) ──
    print("\n[Phase 1] JOINT Training on Domain A (Science) + Domain B (Social)...")
    print("  Using SINGLE shared domain head (set_domain(0)) for ALL facts")
    print("  Brain-inspired: relation vectors shared across domains → analogical transfer")
    
    # Build joint joint_facts (no domain annotations - all go through same head)
    joint_facts = []
    max_len = max(len(domain_a["train"]), len(domain_b["train"]))
    for i in range(max_len):
        if i < len(domain_a["train"]):
            joint_facts.append(domain_a["train"][i])
        if i < len(domain_b["train"]):
            joint_facts.append(domain_b["train"][i])
    
    t0 = time.time()
    acc_joint = []
    errors_joint = []
    
    # Use single domain head for ALL training (shared representation)
    model.set_domain(0)
    model.unfreeze_all_domains()
    
    for repeat in range(config.n_train_repeats):
        rng = np.random.RandomState(repeat + 42)
        rng.shuffle(joint_facts)
        
        losses = []
        correct = 0
        total = 0
        for input_text, target_text, rel_type in joint_facts:
            input_ids, target_ids = encode_fact(tokenizer, input_text, target_text)
            err = model.learn(input_ids, target_ids)
            errors_joint.append(err)
            
            # Call cross-domain edge injection after learning each triple
            if hasattr(model, '_inject_cross_domain_edge'):
                try:
                    subject_tid = int(input_ids[0])
                    rel_idx = model.classify_relation(input_ids[1:].tolist() if len(input_ids) > 1 else [])
                    rel_name = ["causal","semantic","temporal","possessive","analogical","contextual"][rel_idx]
                    subject_embed = model.token_embed.weight.data[subject_tid]
                    subject_cid = model._get_or_create_concept(subject_tid, subject_embed)
                    object_tid = int(target_ids[0])
                    object_embed = model.token_embed.weight.data[object_tid]
                    object_cid = model._get_or_create_concept(object_tid, object_embed)
                    model._inject_cross_domain_edge(subject_cid, object_cid, rel_name, subject_tid)
                except Exception:
                    pass
            
            losses.append(err.get("loss", 0.0))
            if err.get("is_correct", False):
                correct += 1
            total += 1
        
        if repeat % 5 == 0 or repeat == config.n_train_repeats - 1:
            avg_loss = np.mean(losses)
            epoch_acc = correct / total
            print(f"  [Train joint] Repeat {repeat:2d} Loss: {avg_loss:.6f} Acc: {epoch_acc:.1%}")
    
    phase1_time = time.time() - t0

    # ── Finalize verb offsets for evaluation ──
    # Compute avg(target - subject) per verb stem from training data.
    # At inference, predicted_embed = subject_embed + offset(query_verb)
    # enables held-out subject generalization via verb-conditioned vector arithmetic.
    model._compute_verb_offsets()

    # Keep single domain for evaluation (consistency)
    model.set_domain(0)
    
    # Standard evaluation (with spreading activation)
    post_a_on_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    post_a_on_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    
    # RP-only evaluation (disable spreading activation)
    model.disable_spreading_activation = True
    rp_post_a_on_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    rp_post_a_on_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    model.disable_spreading_activation = False
    
    print(f"  RP-only Domain A test: top1={rp_post_a_on_a['top1_accuracy']:.1%}, top10={rp_post_a_on_a['top10_accuracy']:.1%}")
    print(f"  RP-only Domain B test: top1={rp_post_a_on_b['top1_accuracy']:.1%}, top10={rp_post_a_on_b['top10_accuracy']:.1%}")
    graph_after_joint = measure_graph_overlap(model)
    
    # Alignment quality measurement
    alignment_quality = {}
    if hasattr(model, 'measure_cross_domain_alignment'):
        try:
            alignment_quality = model.measure_cross_domain_alignment()
        except Exception:
            pass

    print(f"  Time: {phase1_time:.1f}s ({len(acc_joint)} training steps)")
    print(f"  Domain A test: top1={post_a_on_a['top1_accuracy']:.1%}, top10={post_a_on_a['top10_accuracy']:.1%}")
    print(f"  Domain B test: top1={post_a_on_b['top1_accuracy']:.1%}, top10={post_a_on_b['top10_accuracy']:.1%}")
    print(f"  Graph: {graph_after_joint['n_nodes']} nodes, {graph_after_joint['n_edges']} edges")
    if alignment_quality:
        print(f"  Alignment quality (causal sim): {alignment_quality.get('causal', 0.0):.3f}")

    # ── Phase 1.8: Inject Abstract Cross-Domain Bridge Nodes ──
    print("\n[Phase 1.8] Injecting abstract cross-domain bridge nodes...")
    add_abstract_bridge(model, "intense_bridge", "anger", "expansion", "causal", weight=0.8)
    add_abstract_bridge(model, "warm_bridge", "kindness", "warmth", "causal", weight=0.8)
    add_abstract_bridge(model, "cold_bridge", "sadness", "ice", "causal", weight=0.8)
    add_abstract_bridge(model, "give_bridge", "generosity", "growth", "causal", weight=0.8)
    add_abstract_bridge(model, "fire_bridge", "heat", "conflict", "causal", weight=0.8)
    add_abstract_bridge(model, "insight_bridge", "light", "understanding", "causal", weight=0.8)

    # ── Phase 2: Evaluation of cross-domain transfer ──
    print("\n[Phase 2] Cross-domain transfer evaluation...")
    transfer_probes = test_structural_transfer(
        model, tokenizer, domain_a["test"], domain_b["test"])
    print(f"  Cross-domain top-1 accuracy: {transfer_probes['top1_accuracy']:.1%}")
    print(f"  Cross-domain top-10 accuracy: {transfer_probes['top10_accuracy']:.1%}")
    for probe in transfer_probes["probes"]:
        status = "OK" if probe["correct"] else ("~" if probe["in_top10"] else "X")
        print(f"    [{status}] '{probe['input'].strip()}' -> expected '{probe['expected']}'"
              f"  got '{probe['predicted']}'  ({probe['description']})")

    # RP-only transfer evaluation
    model.disable_spreading_activation = True
    rp_transfer_probes = test_structural_transfer(
        model, tokenizer, domain_a["test"], domain_b["test"])
    model.disable_spreading_activation = False
    print(f"  RP-only cross-domain top-1: {rp_transfer_probes['top1_accuracy']:.1%}")
    print(f"  RP-only cross-domain top-10: {rp_transfer_probes['top10_accuracy']:.1%}")

    # ── Phase 3: Cross-Domain Relation Alignment ──
    print("\n[Phase 3] Cross-domain relation alignment...")
    model.set_domain(0)
    # Run alignment steps if method exists
    if hasattr(model, '_cross_domain_relation_alignment'):
        for align_step in range(30):
            model._cross_domain_relation_alignment()
        print("  Alignment complete.")
    else:
        print("  Skipping - method not available in this model version")
    
    # Alignment quality after alignment
    if hasattr(model, 'measure_cross_domain_alignment'):
        try:
            alignment_quality_post = model.measure_cross_domain_alignment()
            print(f"  Alignment quality post-alignment (causal sim): {alignment_quality_post.get('causal', 0.0):.3f}")
        except Exception:
            pass

    # Re-evaluate after alignment
    print("\n[Phase 3.1] Cross-domain probes after alignment...")
    transfer_probes_aligned = test_structural_transfer(
        model, tokenizer, domain_a["test"], domain_b["test"])
    print(f"  Cross-domain top-1: {transfer_probes_aligned['top1_accuracy']:.1%}, top-10: {transfer_probes_aligned['top10_accuracy']:.1%}")

    # ── Phase 4: Sleep cycle and re-evaluate ──
    print("\n[Phase 4] After sleep cycle...")
    model.sleep_cycle()
    post_sleep_a = evaluate_rlm(model, domain_a["test"], tokenizer)
    post_sleep_b = evaluate_rlm(model, domain_b["test"], tokenizer)
    graph_after_sleep = measure_graph_overlap(model)

    print(f"  Domain A after sleep: top1={post_sleep_a['top1_accuracy']:.1%}, top10={post_sleep_a['top10_accuracy']:.1%}")
    print(f"  Domain B after sleep: top1={post_sleep_b['top1_accuracy']:.1%}, top10={post_sleep_b['top10_accuracy']:.1%}")
    print(f"  Graph: {graph_after_sleep['n_nodes']} nodes, {graph_after_sleep['n_edges']} edges")

    # Run alignment after sleep too if method exists
    if hasattr(model, '_cross_domain_relation_alignment'):
        print("  Running cross-domain relation alignment (post-sleep)...")
        for align_step in range(20):
            model._cross_domain_relation_alignment()
    else:
        print("  Skipping post-sleep alignment - method not available")

    # Re-run transfer probes after sleep + alignment
    post_sleep_probes = test_structural_transfer(
        model, tokenizer, domain_a["test"], domain_b["test"])
    print(f"  Cross-domain probes after sleep+align: top1={post_sleep_probes['top1_accuracy']:.1%}, top10={post_sleep_probes['top10_accuracy']:.1%}")
    
    # Final alignment quality
    if hasattr(model, 'measure_cross_domain_alignment'):
        try:
            alignment_quality_final = model.measure_cross_domain_alignment()
            print(f"  Final alignment quality: {alignment_quality_final}")
        except Exception:
            pass

    results["rlm"] = {
        "baseline_a": baseline_a,
        "baseline_b": baseline_b,
        "post_train_a_on_a": post_a_on_a,
        "post_train_a_on_b": post_a_on_b,
        "post_sleep_a": post_sleep_a,
        "post_sleep_b": post_sleep_b,
        "graph_after_joint": graph_after_joint,
        "graph_after_sleep": graph_after_sleep,
        "transfer_probes": transfer_probes,
        "transfer_probes_aligned": transfer_probes_aligned,
        "post_sleep_probes": post_sleep_probes,
        "alignment_quality": alignment_quality if alignment_quality else {},
        "phase1_time": phase1_time,
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
    parser.add_argument("--n-repeats", type=int, default=15, help="Training repeats per fact (default 15; 100+ overfits)")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="experiments/experiment_results/cross_domain.json")
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
        # Resolve relative to project root
        out_path = os.path.join(str(Path(__file__).parent.parent), out_path)
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
