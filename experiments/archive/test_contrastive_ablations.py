"""
Contrastive Regularizer Ablation Sweep with Publication-Grade Diagnostics

Phased approach:
  Phase 1: lambda sensitivity  (fixed neg_sample=5, sweep lambda)
  Phase 2: neg_sample sweep    (only around best lambda from Phase 1)
  Phase 3: Validation pairs    (held-out generalization test)

Each run measures:
  - Accuracy (in-domain, cross-domain)
  - Encoder drift (semantic vs. morphological separation)
  - Error categorization (reasoning/coverage/leakage/relation_type)
  - Validation pair generalization

Usage:
    python experiments/test_contrastive_ablations.py --phase 1   # lambda sweep
    python experiments/test_contrastive_ablations.py --phase 2   # neg_sample sweep
    python experiments/test_contrastive_ablations.py --phase 3   # validation gen
    python experiments/test_contrastive_ablations.py --phase all # full sweep (3+ hours)
    python experiments/test_contrastive_ablations.py --plot-only # re-plot from saved
"""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import copy
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Any, List, Optional

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer
from semantic_pairs import SEMANTIC_PAIRS, WITHIN_DOMAIN_PAIRS, ALL_PAIRS


# ═══════════════════════════════════════════════════════════════════════════
# Domain Word Lexicons (for error categorization)
# ═══════════════════════════════════════════════════════════════════════════

SCIENCE_WORDS = {
    # Subjects and objects from Domain A causal
    "heat", "friction", "expansion", "light", "vision", "gravity", "objects",
    "rain", "growth", "fire", "warmth", "cold", "shivering", "wind", "erosion",
    "water", "rust", "sunlight", "ice", "slippery", "pressure", "diamonds",
    "oxygen", "combustion", "drought", "famine", "flood", "destruction",
    "melts", "freezes", "voltage", "current", "slows", "motion", "shapes",
    "orbits", "radiation", "dna", "magnetism", "compasses", "evaporation",
    "surfaces", "condensation", "clouds", "sedimentation", "layers",
    "oxidation", "tarnish", "nuclear", "force", "binds", "protons", "tides",
    "sediment", "lightning", "fires", "corrosion", "metals", "centrifugal",
    "outward", "capillary", "liquid", "resonance", "shatters", "glass",
    "diffusion", "particles", "releases", "energy", "osmosis", "transfers",
    "photosynthesis", "produces", "static", "decompression", "cooling",
    "fermentation", "alcohol",
    # Domain A semantic subjects
    "steel", "strong", "diamond", "hard", "silk", "smooth", "lead", "heavy",
    "helium", "rubber", "elastic", "granite", "dense", "mercury", "toxic",
    "quartz", "crystalline", "nitrogen", "inert", "carbon", "versatile",
    "copper", "conductive", "tungsten", "refractory", "neon", "noble",
    "sulfur", "pungent", "aluminum", "lightweight", "solid", "hot",
}

SOCIAL_WORDS = {
    # Subjects and objects from Domain B causal
    "kindness", "trust", "anger", "conflict", "sharing", "friendship",
    "lying", "destroys", "patience", "understanding", "honesty", "respect",
    "empathy", "connection", "greed", "loneliness", "jealousy", "resentment",
    "generosity", "gratitude", "rudeness", "offense", "listening", "rapport",
    "teaching", "knowledge", "neglect", "distance", "celebration", "bonds",
    "criticism", "defensiveness", "forgiveness", "wounds", "praise",
    "confidence", "isolation", "sadness", "teamwork", "success", "gossip",
    "mistrust", "mentorship", "skills", "bullying", "trauma", "collaboration",
    "innovation", "rejection", "withdrawal", "inclusion", "belonging",
    "betrayal", "loyalty", "relationships", "boredom", "exploration",
    "competition", "excellence", "compassion", "suffering", "sarcasm",
    "tension", "vulnerability", "leadership", "action", "apology", "harmony",
    "humor", "rivalry", "grief", "curiosity", "discovery",
    # Domain B semantic
    "friendship", "family", "important", "wisdom", "rare", "courage",
    "admirable", "virtuous", "helpful", "noble", "harmful", "powerful",
    "essential", "natural", "ambition", "driving", "solitude", "peaceful",
    "chaos", "destabilizing", "restorative", "corrosive", "hope", "resilient",
    "pride", "dangerous", "grace", "inspiring", "valuable", "fragile",
}

BRIDGE_WORDS = {
    "intense", "hot", "warm", "love", "flowing", "tears", "bugs", "viruses",
    "damage", "code", "exercise", "stress", "crashes", "destructive",
    "provides", "comfort", "weakness", "failure", "storm", "mud", "sweating",
    "muscles", "contraction", "sun", "tears",
}

ALL_SCIENCE = SCIENCE_WORDS | BRIDGE_WORDS
ALL_SOCIAL = SOCIAL_WORDS | BRIDGE_WORDS


# ═══════════════════════════════════════════════════════════════════════════
# Validation Pairs (held-out — NOT in SEMANTIC_PAIRS)
# ═══════════════════════════════════════════════════════════════════════════

VALIDATION_PAIRS = [
    ("gravity", "respect"),       # binding forces
    ("light", "curiosity"),       # illumination/revelation
    ("rain", "tears"),            # flowing water
    ("diffusion", "gossip"),      # propagation through medium
    ("erosion", "criticism"),     # gradual wearing down
]


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SweepConfig:
    n_epochs: int = 500
    seed: int = 42
    embed_dim: int = 64
    concept_dim: int = 64
    hard_boost_interval: int = 200
    hard_boost_multiplier: int = 100

    lambda_values: Tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 2.0)
    neg_sample_sizes: Tuple[int, ...] = (3, 5, 10)

    output_dir: str = "experiment_results/ablations"

    # Phase control
    phase: str = "all"  # "1", "2", "3", or "all"


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Training Utilities
# ═══════════════════════════════════════════════════════════════════════════

def tokenize_all(tok, train_triples, test_cases):
    all_texts = set()
    for text, _ in train_triples:
        all_texts.add(text)
    for cat in test_cases.values():
        for text, _ in cat:
            all_texts.add(text)
    for text in sorted(all_texts):
        tok.encode(text)


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
    for text, target_word in test_data:
        ids = tok.encode(text)
        target_id = tok.encode(target_word)[0]
        ctx = np.array(ids[:-1], dtype=np.int64)
        logits = model.forward(ctx)
        flat = logits.data.flatten()
        top1 = int(np.argmax(flat))
        top5_set = set(np.argsort(flat)[::-1][:5].tolist())
        top10_set = set(np.argsort(flat)[::-1][:10].tolist())
        if target_id == top1: hits_1 += 1
        if target_id in top5_set: hits_5 += 1
        if target_id in top10_set: hits_10 += 1
        total += 1
    return hits_1 / max(1, total), hits_5 / max(1, total), hits_10 / max(1, total)


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic: Encoder Drift Measurement
# ═══════════════════════════════════════════════════════════════════════════

def _cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na > 0 and nb > 0:
        return float(np.dot(a, b) / (na * nb))
    return 0.0


def _encode_word(model, word, tok):
    """Get latent vector for a word through the encoder."""
    tid = tok.word_to_id.get(word)
    if tid is None:
        return None
    embed = model.token_embed.weight.data[tid]
    latent = model._encoder_forward_full(embed)[0]
    return latent


def measure_encoder_drift(model, baseline_model, tok):
    """Compare latent space of model vs. baseline encoder.

    Returns dict with:
      - semantic_sim_improvement: how much closer semantic pairs got
      - morphological_drift: how much farther morph false positives got
      - per-pair breakdowns
    """
    semantic_before, semantic_after = [], []
    morph_before, morph_after = [], []

    # Semantic analogues (should get CLOSER)
    for wa, wb in SEMANTIC_PAIRS:
        za_b = _encode_word(baseline_model, wa, tok)
        zb_b = _encode_word(baseline_model, wb, tok)
        za_a = _encode_word(model, wa, tok)
        zb_a = _encode_word(model, wb, tok)
        if any(x is None for x in (za_b, zb_b, za_a, zb_a)):
            continue
        semantic_before.append(_cosine_sim(za_b, zb_b))
        semantic_after.append(_cosine_sim(za_a, zb_a))

    # Morphological false positives (should get FARTHER)
    morph_pairs = [
        ("evaporation", "condensation"),
        ("fermentation", "sedimentation"),
        ("expansion", "exploration"),
        ("resonance", "resilient"),
        ("sediment", "sedimentation"),
    ]
    for wa, wb in morph_pairs:
        za_b = _encode_word(baseline_model, wa, tok)
        zb_b = _encode_word(baseline_model, wb, tok)
        za_a = _encode_word(model, wa, tok)
        zb_a = _encode_word(model, wb, tok)
        if any(x is None for x in (za_b, zb_b, za_a, zb_a)):
            continue
        morph_before.append(_cosine_sim(za_b, zb_b))
        morph_after.append(_cosine_sim(za_a, zb_a))

    mean_sem_before = float(np.mean(semantic_before)) if semantic_before else 0.0
    mean_sem_after = float(np.mean(semantic_after)) if semantic_after else 0.0
    mean_morph_before = float(np.mean(morph_before)) if morph_before else 0.0
    mean_morph_after = float(np.mean(morph_after)) if morph_after else 0.0

    return {
        "semantic_sim_improvement": mean_sem_after - mean_sem_before,
        "morphological_drift": mean_morph_before - mean_morph_after,
        "semantic_sims_before": semantic_before,
        "semantic_sims_after": semantic_after,
        "morph_sims_before": morph_before,
        "morph_sims_after": morph_after,
        "mean_semantic_before": mean_sem_before,
        "mean_semantic_after": mean_sem_after,
        "mean_morphological_before": mean_morph_before,
        "mean_morphological_after": mean_morph_after,
    }


def measure_validation_generalization(model, baseline_model, tok):
    """Measure encoder drift on held-out validation pairs."""
    results = {"pairs": []}
    before_sims, after_sims = [], []

    for wa, wb in VALIDATION_PAIRS:
        za_b = _encode_word(baseline_model, wa, tok)
        zb_b = _encode_word(baseline_model, wb, tok)
        za_a = _encode_word(model, wa, tok)
        zb_a = _encode_word(model, wb, tok)
        if any(x is None for x in (za_b, zb_b, za_a, zb_a)):
            continue
        sim_before = _cosine_sim(za_b, zb_b)
        sim_after = _cosine_sim(za_a, zb_a)
        before_sims.append(sim_before)
        after_sims.append(sim_after)
        results["pairs"].append({
            "word_a": wa, "word_b": wb,
            "sim_before": sim_before,
            "sim_after": sim_after,
            "improvement": sim_after - sim_before,
        })

    mean_before = float(np.mean(before_sims)) if before_sims else 0.0
    mean_after = float(np.mean(after_sims)) if after_sims else 0.0
    results["mean_improvement"] = mean_after - mean_before
    results["mean_before"] = mean_before
    results["mean_after"] = mean_after
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic: Error Categorization
# ═══════════════════════════════════════════════════════════════════════════

def _word_domain(word: str) -> Optional[str]:
    """Classify a word into 'science', 'social', or None (bridge/unknown)."""
    in_sci = word in SCIENCE_WORDS
    in_soc = word in SOCIAL_WORDS
    if in_sci and not in_soc:
        return "science"
    if in_soc and not in_sci:
        return "social"
    return None  # bridge or unknown


def _get_query_relation(query_text: str) -> Optional[str]:
    """Extract relation type keyword from a query string."""
    causal_keywords = {"causes", "cause", "produces", "produce", "creates",
                       "create", "generates", "generate", "drives", "drive",
                       "builds", "build", "destroys", "destroy", "triggers",
                       "trigger", "strengthens", "strengthen", "weakens",
                       "weaken", "reduces", "reduce", "heals", "heal",
                       "sparks", "spark", "defuses", "defuse", "spurs",
                       "spur", "deepens", "deepen", "releases", "release",
                       "makes", "make", "melts", "melt", "freezes", "freeze"}
    semantic_keywords = {"is", "are", "represents", "represent"}
    words = query_text.lower().split()
    for w in words:
        if w in causal_keywords:
            return "causal"
        if w in semantic_keywords:
            return "semantic"
    return None


def _get_concept_relations(model, word: str, tok) -> set:
    """Get set of relation types associated with a word's concept."""
    tid = tok.word_to_id.get(word)
    if tid is None:
        return set()
    bindings = model.binding_map.get_concepts(tid, min_confidence=0.0)
    if not bindings:
        return set()
    cid = bindings[0].concept_id
    rel_types = set()
    for tgt_id, edge in model.graph.get_outgoing(cid):
        rel_types.add(edge.relation_type)
    for src_id, edge in model.graph.get_incoming(cid):
        rel_types.add(edge.relation_type)
    return rel_types


def diagnose_failure(model, query_text: str, target_word: str, tok) -> str:
    """Classify a prediction failure into a category."""
    ids = tok.encode(query_text)
    target_id = tok.encode(target_word)[0]
    ctx = np.array(ids[:-1], dtype=np.int64)
    logits = model.forward(ctx)
    flat = logits.data.flatten()
    pred_id = int(np.argmax(flat))
    pred_word = tok.decode([pred_id])

    if pred_id == target_id:
        return "correct"

    # 1. Coverage: target concept not in graph
    target_bindings = model.binding_map.get_concepts(target_id, min_confidence=0.0)
    if not target_bindings:
        return "coverage"
    target_cid = target_bindings[0].concept_id
    if model.graph.get_node(target_cid) is None:
        return "coverage"

    # 2. Leakage: predicted word is from wrong domain
    target_domain = _word_domain(target_word)
    pred_domain = _word_domain(pred_word)
    if target_domain is not None and pred_domain is not None:
        if target_domain != pred_domain:
            return "leakage"

    # 3. Relation-type confusion: predicted word's edges mismatch query relation
    query_rel = _get_query_relation(query_text)
    if query_rel is not None:
        pred_rels = _get_concept_relations(model, pred_word, tok)
        if pred_rels and query_rel not in pred_rels:
            return "relation_type"

    # 4. Default: reasoning failure
    return "reasoning"


def categorize_failures(model, test_cases, tok) -> Dict[str, float]:
    """Compute per-category failure rates across all test data."""
    counts = {"correct": 0, "reasoning": 0, "coverage": 0,
              "leakage": 0, "relation_type": 0}
    total = 0
    for cat_name, cat_data in test_cases.items():
        for query_text, target_word in cat_data:
            cat = diagnose_failure(model, query_text, target_word, tok)
            counts[cat] = counts.get(cat, 0) + 1
            total += 1

    rates = {k: v / max(1, total) * 100 for k, v in counts.items()}
    rates["total_queries"] = total
    return rates


# ═══════════════════════════════════════════════════════════════════════════
# Model Factory
# ═══════════════════════════════════════════════════════════════════════════

def create_model(actual_vocab: int, config: SweepConfig, tok,
                 lambda_c: float = 0.0, neg_size: int = 5) -> RLMv2:
    """Create and configure an RLMv2 model."""
    model = RLMv2(
        vocab_size=actual_vocab + 5,
        embed_dim=config.embed_dim,
        concept_dim=config.concept_dim,
        n_concepts=actual_vocab,
        sleep_interval=300,
        gate_concept_creation=False,
    )
    model._tokenizer = tok

    if lambda_c > 0:
        model.freeze_encoder = False
        model.use_contrastive_reg = True
        model.semantic_pairs = ALL_PAIRS
        model.lambda_contrastive = lambda_c
        model.neg_sample_size = neg_size

    return model


def train_model(model, train_triples, tok, config: SweepConfig):
    """Run full training loop."""
    for epoch in range(config.n_epochs):
        train_epoch(model, train_triples, tok)
        if (epoch + 1) % config.hard_boost_interval == 0:
            hard_boost(model, train_triples, tok, config.hard_boost_multiplier)


def save_encoder_snapshot(model) -> dict:
    """Save encoder weights for later comparison."""
    return {
        "_enc_W1": model._enc_W1.copy(),
        "_enc_b1": model._enc_b1.copy(),
        "_enc_W2": model._enc_W2.copy(),
        "_enc_b2": model._enc_b2.copy(),
    }


def load_encoder_snapshot(model, snapshot: dict):
    """Restore encoder weights from snapshot."""
    model._enc_W1 = snapshot["_enc_W1"].copy()
    model._enc_b1 = snapshot["_enc_b1"].copy()
    model._enc_W2 = snapshot["_enc_W2"].copy()
    model._enc_b2 = snapshot["_enc_b2"].copy()
    # Also reset momentum buffers
    model._enc_mW1 = np.zeros_like(model._enc_W1)
    model._enc_mb1 = np.zeros_like(model._enc_b1)
    model._enc_mW2 = np.zeros_like(model._enc_W2)
    model._enc_mb2 = np.zeros_like(model._enc_b2)


# ═══════════════════════════════════════════════════════════════════════════
# Single Run (enhanced with diagnostics)
# ═══════════════════════════════════════════════════════════════════════════

def train_and_eval(lambda_c: float, neg_size: int, config: SweepConfig,
                   tok, train_triples, test_cases, actual_vocab,
                   baseline_model: Optional[object] = None):
    """Train a model and collect all diagnostics.

    Args:
        baseline_model: A trained RLMv2 with frozen/lambda=0 encoder to compare against.
            Pass None to skip encoder drift measurements.
    """
    print(f"  Training lambda={lambda_c}, neg_sample={neg_size}...", end=" ", flush=True)

    model = create_model(actual_vocab, config, tok, lambda_c, neg_size)
    train_model(model, train_triples, tok, config)

    # ── Accuracy ──
    in_top1, in_top5, in_top10 = evaluate(model, test_cases["train_memorization"], tok)
    cd_top1, cd_top5, cd_top10 = evaluate(model, test_cases["cross_domain_causal"], tok)
    _, _, rt_top10 = evaluate(model, test_cases["relation_type_transfer"], tok)

    # ── Error categorization ──
    error_breakdown = categorize_failures(model, test_cases, tok)

    # ── Encoder drift (if baseline provided) ──
    encoder_drift = None
    val_generalization = None
    if baseline_model is not None:
        encoder_drift = measure_encoder_drift(model, baseline_model, tok)
        val_generalization = measure_validation_generalization(model, baseline_model, tok)

    print(f"in_domain@1={in_top1:.1%}, x_domain@1={cd_top1:.1%}, "
          f"reasoning={error_breakdown.get('reasoning', 0):.1f}%")

    result = {
        "lambda": lambda_c,
        "neg_sample_size": neg_size,
        "in_domain_top1": in_top1,
        "in_domain_top5": in_top5,
        "in_domain_top10": in_top10,
        "cross_domain_top1": cd_top1,
        "cross_domain_top5": cd_top5,
        "cross_domain_top10": cd_top10,
        "relation_type_transfer_top10": rt_top10,
        "error_breakdown": error_breakdown,
    }

    if encoder_drift is not None:
        result["encoder_drift"] = {
            "semantic_sim_improvement": encoder_drift["semantic_sim_improvement"],
            "morphological_drift": encoder_drift["morphological_drift"],
            "mean_semantic_before": encoder_drift["mean_semantic_before"],
            "mean_semantic_after": encoder_drift["mean_semantic_after"],
            "mean_morphological_before": encoder_drift["mean_morphological_before"],
            "mean_morphological_after": encoder_drift["mean_morphological_after"],
            "semantic_pairs": [
                {"before": s1, "after": s2}
                for s1, s2 in zip(encoder_drift["semantic_sims_before"],
                                  encoder_drift["semantic_sims_after"])
            ],
            "morph_pairs": [
                {"before": s1, "after": s2}
                for s1, s2 in zip(encoder_drift["morph_sims_before"],
                                  encoder_drift["morph_sims_after"])
            ],
        }

    if val_generalization is not None:
        result["validation_generalization"] = {
            "mean_improvement": val_generalization["mean_improvement"],
            "mean_before": val_generalization["mean_before"],
            "mean_after": val_generalization["mean_after"],
            "pairs": val_generalization["pairs"],
        }

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Sweep Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def run_sweep_phase1(config: SweepConfig) -> Tuple[Dict, dict]:
    """Phase 1: lambda sensitivity, fixed neg_sample=5."""
    print("=" * 70)
    print("  PHASE 1: Lambda Sensitivity (neg_sample=5 fixed)")
    print("=" * 70)
    print(f"  Lambda values: {config.lambda_values}")
    print(f"  Epochs: {config.n_epochs}")
    print()

    np.random.seed(config.seed)
    train_triples, test_cases = build_training_data()
    tok = WordTokenizer()
    tokenize_all(tok, train_triples, test_cases)
    actual_vocab = tok.vocab_size
    print(f"  Vocab: {actual_vocab}, Training: {len(train_triples)} triples\n")

    # Train baseline first (lambda=0) and save encoder snapshot
    print("  Training baseline (lambda=0.0)...")
    baseline_model = create_model(actual_vocab, config, tok, lambda_c=0.0, neg_size=5)
    train_model(baseline_model, train_triples, tok, config)
    baseline_eval = {
        "in_domain_top1": evaluate(baseline_model, test_cases["train_memorization"], tok)[0],
        "cross_domain_top1": evaluate(baseline_model, test_cases["cross_domain_causal"], tok)[0],
        "error_breakdown": categorize_failures(baseline_model, test_cases, tok),
    }
    print(f"  Baseline: in_domain@1={baseline_eval['in_domain_top1']:.1%}, "
          f"x_domain@1={baseline_eval['cross_domain_top1']:.1%}")

    results = {}
    for lambda_c in config.lambda_values:
        if lambda_c == 0.0:
            continue  # skip re-run; we have baseline
        result = train_and_eval(
            lambda_c, 5, config, tok, train_triples, test_cases, actual_vocab,
            baseline_model=baseline_model
        )
        results[(lambda_c, 5)] = result

    return results, baseline_eval


def run_sweep_phase2(config: SweepConfig, best_lambda: float,
                     baseline_model: object) -> Dict:
    """Phase 2: neg_sample sensitivity around best lambda."""
    print("=" * 70)
    print(f"  PHASE 2: Negative Sample Sensitivity (lambda={best_lambda} fixed)")
    print("=" * 70)
    print(f"  neg_sample values: {config.neg_sample_sizes}")
    print()

    np.random.seed(config.seed)
    train_triples, test_cases = build_training_data()
    tok = WordTokenizer()
    tokenize_all(tok, train_triples, test_cases)
    actual_vocab = tok.vocab_size
    print(f"  Vocab: {actual_vocab}, Training: {len(train_triples)} triples\n")

    results = {}
    for neg_size in config.neg_sample_sizes:
        result = train_and_eval(
            best_lambda, neg_size, config, tok, train_triples, test_cases, actual_vocab,
            baseline_model=baseline_model
        )
        results[(best_lambda, neg_size)] = result

    return results


def run_sweep_phase3(config: SweepConfig, best_lambda: float, best_neg: int,
                     baseline_model: object) -> Dict:
    """Phase 3: validation generalization with best config."""
    print("=" * 70)
    print(f"  PHASE 3: Validation Generalization (lambda={best_lambda}, neg={best_neg})")
    print("=" * 70)
    print(f"  Validation pairs: {len(VALIDATION_PAIRS)}")
    print()

    np.random.seed(config.seed)
    train_triples, test_cases = build_training_data()
    tok = WordTokenizer()
    tokenize_all(tok, train_triples, test_cases)
    actual_vocab = tok.vocab_size

    result = train_and_eval(
        best_lambda, best_neg, config, tok, train_triples, test_cases, actual_vocab,
        baseline_model=baseline_model
    )

    # Print validation detail
    val = result.get("validation_generalization", {})
    if val:
        print(f"\n  Validation generalization:")
        print(f"    Mean similarity before: {val.get('mean_before', 0):.4f}")
        print(f"    Mean similarity after:  {val.get('mean_after', 0):.4f}")
        print(f"    Improvement:            {val.get('mean_improvement', 0):+.4f}")
        for p in val.get("pairs", []):
            arrow = "+" if p["improvement"] > 0 else ""
            print(f"    {p['word_a']:>15} <-> {p['word_b']:<15}: "
                  f"{p['sim_before']:.4f} -> {p['sim_after']:.4f} "
                  f"({arrow}{p['improvement']:.4f})")

    return {(best_lambda, best_neg): result}


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def generate_diagnostic_plots(all_results: Dict, phase1_results: Dict,
                              phase2_results: Dict, phase3_results: Dict,
                              baseline_eval: dict, output_dir: str):
    """Generate publication-quality diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── Plot 1: Cross-domain accuracy vs. lambda (Phase 1) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    lambdas = sorted([k[0] for k in phase1_results.keys()])
    cross_domain = [phase1_results[(l, 5)]["cross_domain_top1"] * 100 for l in lambdas]
    in_domain = [phase1_results[(l, 5)]["in_domain_top1"] * 100 for l in lambdas]

    bar_width = 0.35
    x = np.arange(len(lambdas))
    axes[0, 0].bar(x - bar_width/2, in_domain, bar_width, label="In-domain", alpha=0.8)
    axes[0, 0].bar(x + bar_width/2, cross_domain, bar_width, label="Cross-domain", alpha=0.8)
    axes[0, 0].axhline(y=baseline_eval.get("cross_domain_top1", 0) * 100,
                        color="gray", linestyle="--", alpha=0.7,
                        label=f"Baseline CD={baseline_eval.get('cross_domain_top1', 0):.1%}")
    axes[0, 0].set_xlabel("lambda (contrastive weight)")
    axes[0, 0].set_ylabel("Top-1 Accuracy (%)")
    axes[0, 0].set_title("Effect of Contrastive Loss Weight")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([str(l) for l in lambdas])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # ── Plot 2: Error breakdown (baseline vs. best config) ──
    categories = ["reasoning", "coverage", "leakage", "relation_type"]
    phase1_items = sorted(phase1_results.items(), key=lambda x: x[1]["cross_domain_top1"], reverse=True)
    best_key = phase1_items[0][0] if phase1_items else (0.5, 5)

    before_errors = [baseline_eval.get("error_breakdown", {}).get(c, 0) for c in categories]
    after_errors = [phase1_results.get(best_key, {}).get("error_breakdown", {}).get(c, 0) for c in categories]

    x2 = np.arange(len(categories))
    axes[0, 1].bar(x2 - 0.2, before_errors, 0.4, label="Baseline (frozen)", alpha=0.8)
    axes[0, 1].bar(x2 + 0.2, after_errors, 0.4, label=f"Contrastive (lambda={best_key[0]})", alpha=0.8)
    axes[0, 1].set_xticks(x2)
    axes[0, 1].set_xticklabels(categories)
    axes[0, 1].set_ylabel("Error Rate (% of queries)")
    axes[0, 1].set_title(f"Error Category Improvement (best: lambda={best_key[0]})")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ── Plot 3: Encoder learning scatter (semantic pairs before → after) ──
    best_result = phase1_results.get(best_key, {})
    drift = best_result.get("encoder_drift", {})
    sem_before = drift.get("semantic_pairs", [])
    if sem_before:
        sem_before_vals = [p["before"] for p in sem_before]
        sem_after_vals = [p["after"] for p in sem_before]
        axes[1, 0].scatter(sem_before_vals, sem_after_vals, s=80, alpha=0.7, c="green", edgecolors="darkgreen")
        axes[1, 0].plot([-1, 1], [-1, 1], "k--", alpha=0.5, label="no change")
        axes[1, 0].set_xlabel("Semantic Similarity (before)")
        axes[1, 0].set_ylabel("Semantic Similarity (after)")
        axes[1, 0].set_title(f"Encoder Learning (lambda={best_key[0]}, neg={best_key[1]})")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        # Add arrow showing mean shift
        mean_b = drift.get("mean_semantic_before", 0)
        mean_a = drift.get("mean_semantic_after", 0)
        axes[1, 0].annotate(f"mean: {mean_b:.3f} -> {mean_a:.3f}",
                            xy=(0.6, 0.1), xycoords="axes fraction",
                            fontsize=9, bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

    # ── Plot 4: Morphological drift bar chart ──
    morph_before_vals = drift.get("morph_pairs", [])
    if morph_before_vals:
        morph_b = [p["before"] for p in morph_before_vals]
        morph_a = [p["after"] for p in morph_before_vals]
        labels = ["evaporation\ncondensation", "fermentation\nsedimentation",
                  "expansion\nexploration", "resonance\nresilient",
                  "sediment\nsedimentation"]
        x4 = np.arange(len(morph_b))
        width = 0.35
        axes[1, 1].bar(x4 - width/2, morph_b, width, label="Before", alpha=0.8)
        axes[1, 1].bar(x4 + width/2, morph_a, width, label="After", alpha=0.8)
        axes[1, 1].set_xticks(x4)
        axes[1, 1].set_xticklabels(labels, fontsize=7)
        axes[1, 1].set_ylabel("Cosine Similarity")
        axes[1, 1].set_title("Morphological False Positives (should decrease)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "phase1_diagnostics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Phase 1 diagnostics saved to {path}")

    # ── If Phase 2 exists, plot heatmap ──
    if phase2_results:
        fig2, ax = plt.subplots(figsize=(8, 6))
        lambdas_list = sorted(set(k[0] for k in phase2_results.keys()))
        negs_list = sorted(set(k[1] for k in phase2_results.keys()))
        heatmap_data = np.zeros((len(lambdas_list), len(negs_list)))
        for i, l in enumerate(lambdas_list):
            for j, n in enumerate(negs_list):
                key = (l, n)
                if key in phase2_results:
                    heatmap_data[i, j] = phase2_results[key]["cross_domain_top1"] * 100
                else:
                    heatmap_data[i, j] = np.nan
        im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(negs_list)))
        ax.set_xticklabels([str(n) for n in negs_list])
        ax.set_yticks(range(len(lambdas_list)))
        ax.set_yticklabels([str(l) for l in lambdas_list])
        ax.set_xlabel("Negative Samples")
        ax.set_ylabel("Lambda")
        ax.set_title("Cross-Domain Top-1 Accuracy (%)")
        for i in range(len(lambdas_list)):
            for j in range(len(negs_list)):
                val = heatmap_data[i, j]
                if not np.isnan(val):
                    color = "white" if val < 50 else "black"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=9)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        path2 = os.path.join(output_dir, "phase2_heatmap.png")
        plt.savefig(path2, dpi=150)
        plt.close()
        print(f"  Phase 2 heatmap saved to {path2}")

    # ── Validation generalization plot ──
    if phase3_results:
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        key = list(phase3_results.keys())[0]
        val = phase3_results[key].get("validation_generalization", {})
        pairs = val.get("pairs", [])
        if pairs:
            names = [f"{p['word_a']}\n{p['word_b']}" for p in pairs]
            before = [p["sim_before"] for p in pairs]
            after = [p["sim_after"] for p in pairs]
            x3 = np.arange(len(names))
            width = 0.35
            ax3.bar(x3 - width/2, before, width, label="Before", alpha=0.8)
            ax3.bar(x3 + width/2, after, width, label="After", alpha=0.8)
            ax3.set_xticks(x3)
            ax3.set_xticklabels(names, fontsize=8)
            ax3.set_ylabel("Cosine Similarity")
            ax3.set_title("Validation Pair Generalization (held-out)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.annotate(f"Mean improvement: {val.get('mean_improvement', 0):+.4f}",
                        xy=(0.6, 0.95), xycoords="axes fraction", fontsize=10,
                        bbox=dict(boxstyle="round", fc="lightblue", alpha=0.5))
        plt.tight_layout()
        path3 = os.path.join(output_dir, "phase3_validation.png")
        plt.savefig(path3, dpi=150)
        plt.close()
        print(f"  Phase 3 validation plot saved to {path3}")


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_phase1_table(phase1_results: Dict, baseline_eval: dict):
    print("\n" + "=" * 70)
    print("  PHASE 1: LAMBDA SENSITIVITY RESULTS")
    print("=" * 70)
    h = f"{'lambda':>6} {'in-dom@1':>10} {'x-dom@1':>10} {'x-dom@10':>10} "
    h += f"{'reasoning':>10} {'coverage':>10} {'leakage':>10} {'rel_type':>10}"
    h += f"{'sem-imp':>9} {'morph-drift':>11}"
    print(f"  {h}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*9} {'-'*11}")

    # Baseline row
    bl_eb = baseline_eval.get("error_breakdown", {})
    print(f"  {'0.0':>6} "
          f"{baseline_eval.get('in_domain_top1', 0):>9.1%} "
          f"{baseline_eval.get('cross_domain_top1', 0):>9.1%} "
          f"{'':>10} "
          f"{bl_eb.get('reasoning', 0):>9.1f}% "
          f"{bl_eb.get('coverage', 0):>9.1f}% "
          f"{bl_eb.get('leakage', 0):>9.1f}% "
          f"{bl_eb.get('relation_type', 0):>9.1f}% "
          f"{'':>8} {'':>10} [baseline]")

    for (lambda_c, neg), res in sorted(phase1_results.items()):
        eb = res.get("error_breakdown", {})
        drift = res.get("encoder_drift", {})
        sem_imp = drift.get("semantic_sim_improvement", 0)
        morph_drift = drift.get("morphological_drift", 0)
        print(f"  {lambda_c:>6.1f} "
              f"{res['in_domain_top1']:>9.1%} "
              f"{res['cross_domain_top1']:>9.1%} "
              f"{res['cross_domain_top10']:>9.1%} "
              f"{eb.get('reasoning', 0):>9.1f}% "
              f"{eb.get('coverage', 0):>9.1f}% "
              f"{eb.get('leakage', 0):>9.1f}% "
              f"{eb.get('relation_type', 0):>9.1f}% "
              f"{sem_imp:>+8.4f} {morph_drift:>+10.4f}")

    # Best finders
    best = max(phase1_results.items(), key=lambda x: x[1]["cross_domain_top1"])
    print(f"\n  Best cross-domain: lambda={best[0][0]:.1f}, "
          f"cd@1={best[1]['cross_domain_top1']:.1%}, "
          f"cd@10={best[1]['cross_domain_top10']:.1%}")
    print(f"  Best reasoning: lambda={min(phase1_results.items(), key=lambda x: x[1]['error_breakdown']['reasoning'])[0][0]:.1f}")
    return best[0][0]


def print_phase2_table(phase2_results: Dict):
    print("\n" + "=" * 70)
    print("  PHASE 2: NEGATIVE SAMPLE SENSITIVITY RESULTS")
    print("=" * 70)
    h = f"{'neg':>5} {'in-dom@1':>10} {'x-dom@1':>10} {'x-dom@10':>10} "
    h += f"{'reasoning':>10} {'sem-imp':>9} {'morph-drift':>11}"
    print(f"  {h}")
    print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*9} {'-'*11}")

    for (lam, neg), res in sorted(phase2_results.items()):
        eb = res.get("error_breakdown", {})
        drift = res.get("encoder_drift", {})
        print(f"  {neg:>5d} "
              f"{res['in_domain_top1']:>9.1%} "
              f"{res['cross_domain_top1']:>9.1%} "
              f"{res['cross_domain_top10']:>9.1%} "
              f"{eb.get('reasoning', 0):>9.1f}% "
              f"{drift.get('semantic_sim_improvement', 0):>+8.4f} "
              f"{drift.get('morphological_drift', 0):>+10.4f}")

    best = max(phase2_results.items(), key=lambda x: x[1]["cross_domain_top1"])
    print(f"\n  Best: lambda={best[0][0]:.1f}, neg={best[0][1]}, "
          f"cd@1={best[1]['cross_domain_top1']:.1%}")
    return best[0]


# ═══════════════════════════════════════════════════════════════════════════
# Save / Load
# ═══════════════════════════════════════════════════════════════════════════

def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_convert)
    print(f"\nResults saved to {path}")


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Contrastive Regularizer Ablation Sweep")
    parser.add_argument("--phase", type=str, default="1",
                        choices=["1", "2", "3", "all"],
                        help="Which phase to run (default: 1)")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Training epochs per config")
    parser.add_argument("--best-lambda", type=float, default=None,
                        help="Best lambda from Phase 1 (for Phase 2/3)")
    parser.add_argument("--best-neg", type=int, default=None,
                        help="Best neg_sample from Phase 2 (for Phase 3)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-plot from saved results")
    parser.add_argument("--results-path", type=str,
                        default="experiment_results/ablations/sweep_results.json",
                        help="Path to load/save results")
    args = parser.parse_args()

    out_dir = os.path.join(_PROJECT_ROOT, "experiment_results", "ablations")
    os.makedirs(out_dir, exist_ok=True)
    results_path = args.results_path
    if not os.path.isabs(results_path):
        results_path = os.path.join(_PROJECT_ROOT, results_path)

    # ── Plot-only mode ──
    if args.plot_only:
        if os.path.exists(results_path):
            data = load_results(results_path)
            print_results_table(
                {(r["lambda"], r.get("neg_sample_size", 5)): r
                 for r in data.get("runs", []) if "lambda" in r}
            )
        else:
            print(f"No results found at {results_path}")
        sys.exit(0)

    # ── Phase selection ──
    phase = args.phase

    if phase == "1" or phase == "all":
        config = SweepConfig(
            n_epochs=args.epochs,
            lambda_values=SweepConfig.lambda_values,
            neg_sample_sizes=(5,),
        )
        phase1_results, baseline_eval = run_sweep_phase1(config)
        print_phase1_table(phase1_results, baseline_eval)

        # Determine best lambda
        best = max(phase1_results.items(), key=lambda x: x[1]["cross_domain_top1"])
        best_lambda = best[0][0]
        best_neg = best[0][1]
        print(f"\n  >>> Best lambda = {best_lambda} <<<")

        save_results(results_path, {
            "phase": 1,
            "config": {
                "n_epochs": config.n_epochs,
                "lambda_values": list(config.lambda_values),
            },
            "baseline": baseline_eval,
            "runs": [v for k, v in phase1_results.items()],
        })

        # Generate plots
        generate_diagnostic_plots(
            phase1_results, phase1_results, {}, {},
            baseline_eval, out_dir
        )

        if phase == "1":
            sys.exit(0)
    else:
        # Load Phase 1 results to get best lambda
        try:
            data = load_results(results_path)
            phase1_runs = [r for r in data.get("runs", []) if r.get("lambda", 0) > 0]
            best = max(phase1_runs, key=lambda r: r["cross_domain_top1"])
            best_lambda = best["lambda"]
            best_neg = best.get("neg_sample_size", 5)
            print(f"Loaded Phase 1 results: best lambda={best_lambda}")
        except (FileNotFoundError, KeyError, ValueError):
            best_lambda = args.best_lambda or 0.5
            best_neg = args.best_neg or 5
            print(f"No Phase 1 results found, using default: lambda={best_lambda}")

    # ── Phase 2: neg_sample sensitivity ──
    if phase == "2" or phase == "all":
        config = SweepConfig(
            n_epochs=args.epochs,
            lambda_values=(best_lambda,),
            neg_sample_sizes=SweepConfig.neg_sample_sizes,
        )

        # Need baseline encoder from a trained baseline model
        np.random.seed(config.seed)
        train_triples, test_cases = build_training_data()
        tok = WordTokenizer()
        tokenize_all(tok, train_triples, test_cases)
        actual_vocab = tok.vocab_size
        baseline_model = create_model(actual_vocab, config, tok, 0.0, 5)
        train_model(baseline_model, train_triples, tok, config)

        phase2_results = run_sweep_phase2(config, best_lambda, baseline_model)
        best_key = print_phase2_table(phase2_results)
        best_lambda, best_neg = best_key

        # Save
        save_results(results_path.replace(".json", "_phase2.json"), {
            "phase": 2,
            "best_lambda": best_lambda,
            "runs": [v for k, v in phase2_results.items()],
        })

        if phase == "2":
            sys.exit(0)

    # ── Phase 3: Validation generalization ──
    if phase == "3" or phase == "all":
        config = SweepConfig(n_epochs=args.epochs)

        # Need baseline encoder
        np.random.seed(config.seed)
        train_triples, test_cases = build_training_data()
        tok = WordTokenizer()
        tokenize_all(tok, train_triples, test_cases)
        actual_vocab = tok.vocab_size
        baseline_model = create_model(actual_vocab, config, tok, 0.0, 5)
        train_model(baseline_model, train_triples, tok, config)

        phase3_results = run_sweep_phase3(config, best_lambda, best_neg, baseline_model)

        # Generate all plots
        generate_diagnostic_plots(
            phase3_results, {}, {},
            phase3_results, {"cross_domain_top1": 0}, out_dir
        )

        save_results(results_path.replace(".json", "_phase3.json"), {
            "phase": 3,
            "best_lambda": best_lambda,
            "best_neg": best_neg,
            "runs": [v for k, v in phase3_results.items()],
        })
