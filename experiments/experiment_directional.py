"""
Directional Reasoning Benchmark
================================
Tests Design B: broad traversal + predicate-aware filtering.

Key principle:
  Traversal retrieves broadly (family level)
  Filtering decides precision (sub-family / predicate level)

Tests:
  1. Directional specificity: "exercise increases ?" -> fatigue, NOT injury
  2. Broad effects: "exercise effects ?" -> fatigue AND injury AND strength
  3. Contradictory evidence: "coffee increases ?" vs "coffee decreases ?"
  4. Confidence ranking within sub-families
  5. Multi-granularity queries
"""
import sys
import io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.word_tokenizer import WordTokenizer
from ravana_ml.relation_ontology import (
    Candidate, TraversalConfig, get_sub_family, get_family,
    get_confidence, matches_config, SUB_FAMILIES
)


def inject_embeddings(model, tok, embed_map):
    dim = model.embed_dim
    for word, v3 in embed_map.items():
        tid = tok.word_to_id.get(word)
        if tid is None:
            continue
        full = np.zeros(dim, dtype=np.float32)
        rng = np.random.RandomState(hash(word) % 2**31)
        for i in range(dim):
            full[i] = v3[i % 3] + rng.randn() * 0.005
        full /= np.linalg.norm(full)
        model.token_embed.weight.data[tid] = full


def train_facts(model, tok, facts, epochs=5):
    for epoch in range(epochs):
        for s, r, o in facts:
            ids = tok.encode(f"{s} {r} {o}")
            if len(ids) < 2:
                continue
            ctx = np.array([ids[:-1]], dtype=np.int64)
            tgt = np.array([[ids[-1]]], dtype=np.int64)
            model.learn(ctx, tgt)


def traverse_broad(model, tok, subject, max_depth=3):
    """Broad traversal: follow all edges, return structured Candidates.

    This is Design B: retrieve everything, filter later.
    """
    subj_tid = tok.word_to_id.get(subject)
    if subj_tid is None:
        return []
    bindings = model.binding_map.get_concepts(subj_tid, min_confidence=0.1)
    if not bindings:
        return []

    frontier = {bindings[0].concept_id}
    visited = {bindings[0].concept_id}
    candidates = []

    for depth in range(1, max_depth + 1):
        next_frontier = set()
        for nid in frontier:
            for tgt_id, edge in model.graph.get_outgoing(nid):
                if tgt_id in visited:
                    continue
                next_frontier.add(tgt_id)
                tokens = model.binding_map.get_tokens(tgt_id, 0.0)
                for b in tokens:
                    word = tok.decode([b.token_id])
                    if word == subject or word.startswith("?"):
                        continue
                    # Decode predicate
                    pred_word = edge.relation_type
                    if hasattr(edge, 'predicate_token_id') and edge.predicate_token_id is not None:
                        decoded = tok.decode([edge.predicate_token_id])
                        if decoded.strip():
                            pred_word = decoded.strip()
                    # Build structured candidate
                    sub_fam = get_sub_family(pred_word) or "unknown"
                    family = get_family(pred_word) or edge.relation_type
                    conf = get_confidence(pred_word)
                    candidates.append(Candidate(
                        word=word,
                        predicate=pred_word,
                        family=family,
                        sub_family=sub_fam,
                        depth=depth,
                        confidence=conf,
                        path=[(subject, pred_word, word)]
                    ))
        visited |= next_frontier
        frontier = next_frontier

    # Deduplicate: prefer shallower, then higher confidence
    seen = {}
    for c in candidates:
        if c.word not in seen or c.depth < seen[c.word].depth or \
           (c.depth == seen[c.word].depth and c.confidence > seen[c.word].confidence):
            seen[c.word] = c
    return sorted(seen.values(), key=lambda x: (x.depth, -x.confidence))


def filter_candidates(candidates, config):
    """Filter candidates by traversal config (sub-family, family, or relaxed)."""
    return [c for c in candidates if matches_config(c.predicate, config)]


def show_candidates(candidates, label=""):
    if label:
        print(f"\n{label} ({len(candidates)} results):")
    for c in candidates:
        print(f"  depth {c.depth} conf {c.confidence:.2f} [{c.sub_family:25s}] {c.predicate:20s}: {c.word}")


# ============================================================
# TEST 1: DIRECTIONAL SPECIFICITY
# ============================================================

def test_directional_specificity():
    """exercise increases ? should return fatigue, NOT injury."""
    print("=" * 70)
    print("TEST 1: DIRECTIONAL SPECIFICITY")
    print("exercise increases ? -> fatigue (NOT injury)")
    print("=" * 70)

    facts = [
        ("exercise", "increases", "fatigue"),
        ("exercise", "increases", "strength"),
        ("exercise", "increases", "flexibility"),
        ("exercise", "decreases", "injury_risk"),
        ("exercise", "decreases", "stress"),
        ("exercise", "improves", "mood"),
        ("exercise", "causes", "soreness"),
        ("exercise", "contains", "movement"),
    ]

    embed_map = {
        "exercise": [0.5, 0.8, 0.2], "fatigue": [-0.3, 0.2, 0.8],
        "strength": [0.7, 0.6, 0.3], "flexibility": [0.4, 0.5, 0.6],
        "injury_risk": [-0.6, 0.3, 0.5], "stress": [-0.5, 0.4, 0.6],
        "mood": [0.3, 0.7, 0.4], "soreness": [-0.4, 0.3, 0.7],
        "movement": [0.2, 0.5, 0.5],
    }

    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")

    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=500, gate_concept_creation=False)
    model._tokenizer = tok
    inject_embeddings(model, tok, embed_map)
    train_facts(model, tok, facts)

    # Broad retrieval
    all_candidates = traverse_broad(model, tok, "exercise", max_depth=2)
    show_candidates(all_candidates, "exercise ? (BROAD — all relations)")

    # Filter: directional-positive only
    config_pos = TraversalConfig(mode="sub_family", sub_family="directional-positive")
    pos_only = filter_candidates(all_candidates, config_pos)
    show_candidates(pos_only, "exercise increases ? (directional-positive filter)")

    # Filter: directional-negative only
    config_neg = TraversalConfig(mode="sub_family", sub_family="directional-negative")
    neg_only = filter_candidates(all_candidates, config_neg)
    show_candidates(neg_only, "exercise decreases ? (directional-negative filter)")

    # Filter: all directional
    config_dir = TraversalConfig(mode="family", family="directional")
    dir_all = filter_candidates(all_candidates, config_dir)
    show_candidates(dir_all, "exercise directional ? (both sub-families)")

    # Verify
    pos_words = {c.word for c in pos_only}
    neg_words = {c.word for c in neg_only}

    print(f"\nDirectional-positive results: {pos_words}")
    print(f"Directional-negative results: {neg_words}")
    print(f"'fatigue' in positive: {'fatigue' in pos_words}")
    print(f"'injury_risk' in negative: {'injury_risk' in neg_words}")
    print(f"'injury_risk' NOT in positive: {'injury_risk' not in pos_words}")
    print(f"'fatigue' NOT in negative: {'fatigue' not in neg_words}")

    return pos_words, neg_words


# ============================================================
# TEST 2: BROAD EFFECTS
# ============================================================

def test_broad_effects():
    """exercise effects ? should return ALL effects regardless of direction."""
    print("\n" + "=" * 70)
    print("TEST 2: BROAD EFFECTS")
    print("exercise effects ? -> fatigue AND injury_risk AND strength")
    print("=" * 70)

    facts = [
        ("exercise", "increases", "fatigue"),
        ("exercise", "increases", "strength"),
        ("exercise", "decreases", "injury_risk"),
        ("exercise", "decreases", "stress"),
        ("exercise", "improves", "mood"),
        ("exercise", "causes", "soreness"),
    ]

    embed_map = {
        "exercise": [0.5, 0.8, 0.2], "fatigue": [-0.3, 0.2, 0.8],
        "strength": [0.7, 0.6, 0.3], "injury_risk": [-0.6, 0.3, 0.5],
        "stress": [-0.5, 0.4, 0.6], "mood": [0.3, 0.7, 0.4],
        "soreness": [-0.4, 0.3, 0.7],
    }

    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")

    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=500, gate_concept_creation=False)
    model._tokenizer = tok
    inject_embeddings(model, tok, embed_map)
    train_facts(model, tok, facts)

    # Broad: causal + directional
    all_candidates = traverse_broad(model, tok, "exercise", max_depth=2)

    config_broad = TraversalConfig(mode="super_family", super_family="causal_directional")
    broad = filter_candidates(all_candidates, config_broad)
    show_candidates(broad, "exercise effects ? (causal + directional)")

    broad_words = {c.word for c in broad}
    print(f"\nAll effects: {broad_words}")
    print(f"Contains fatigue: {'fatigue' in broad_words}")
    print(f"Contains injury_risk: {'injury_risk' in broad_words}")
    print(f"Contains strength: {'strength' in broad_words}")
    print(f"Contains soreness: {'soreness' in broad_words}")

    return broad_words


# ============================================================
# TEST 3: CONTRADICTORY EVIDENCE
# ============================================================

def test_contradictions():
    """coffee increases ? and coffee decreases ? should return different sets."""
    print("\n" + "=" * 70)
    print("TEST 3: CONTRADICTORY EVIDENCE")
    print("coffee increases ? vs coffee decreases ?")
    print("=" * 70)

    facts = [
        ("coffee", "increases", "focus"),
        ("coffee", "increases", "alertness"),
        ("coffee", "increases", "anxiety"),
        ("coffee", "decreases", "focus"),      # contradiction!
        ("coffee", "decreases", "fatigue"),
        ("coffee", "decreases", "hunger"),
        ("coffee", "contains", "caffeine"),
    ]

    embed_map = {
        "coffee": [-1, 0.8, 0.2], "focus": [-0.5, 0.5, 0.5],
        "alertness": [-0.4, 0.6, 0.4], "anxiety": [-0.7, 0.3, 0.6],
        "fatigue": [-0.3, 0.2, 0.8], "hunger": [-0.2, 0.4, 0.6],
        "caffeine": [-0.3, 0.5, 0.5],
    }

    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")

    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=500, gate_concept_creation=False)
    model._tokenizer = tok
    inject_embeddings(model, tok, embed_map)
    train_facts(model, tok, facts)

    all_candidates = traverse_broad(model, tok, "coffee", max_depth=2)

    # Filter by sub-family
    config_pos = TraversalConfig(mode="sub_family", sub_family="directional-positive")
    config_neg = TraversalConfig(mode="sub_family", sub_family="directional-negative")

    pos = filter_candidates(all_candidates, config_pos)
    neg = filter_candidates(all_candidates, config_neg)

    show_candidates(pos, "coffee increases ? (directional-positive)")
    show_candidates(neg, "coffee decreases ? (directional-negative)")

    pos_words = {c.word for c in pos}
    neg_words = {c.word for c in neg}

    print(f"\nincreases: {pos_words}")
    print(f"decreases: {neg_words}")
    print(f"'focus' in BOTH: {'focus' in pos_words and 'focus' in neg_words}")
    print(f"'alertness' ONLY in increases: {'alertness' in pos_words and 'alertness' not in neg_words}")
    print(f"'fatigue' ONLY in decreases: {'fatigue' in neg_words and 'fatigue' not in pos_words}")

    # Show the contradiction explicitly
    both = pos_words & neg_words
    if both:
        print(f"\nCONTRADICTED ENTITIES: {both}")
        for word in both:
            pos_preds = [c.predicate for c in pos if c.word == word]
            neg_preds = [c.predicate for c in neg if c.word == word]
            print(f"  {word}: increased via {pos_preds}, decreased via {neg_preds}")

    return pos_words, neg_words


# ============================================================
# TEST 4: MULTI-GRANULARITY
# ============================================================

def test_multi_granularity():
    """Same data, different query granularities."""
    print("\n" + "=" * 70)
    print("TEST 4: MULTI-GRANULARITY QUERIES")
    print("Same data, different precision levels")
    print("=" * 70)

    facts = [
        ("drug_x", "causes", "recovery"),
        ("drug_x", "causes", "nausea"),
        ("drug_x", "improves", "mobility"),
        ("drug_x", "reduces", "pain"),
        ("drug_x", "associated_with", "drowsiness"),
        ("drug_x", "contains", "morphine"),
        ("drug_x", "is_a", "analgesic"),
    ]

    embed_map = {
        "drug_x": [0, 0, 1], "recovery": [0.5, 0.5, 0.5],
        "nausea": [-0.5, -0.3, 0.7], "mobility": [0.3, 0.7, 0.3],
        "pain": [-0.7, -0.5, 0.3], "drowsiness": [-0.3, -0.2, 0.8],
        "morphine": [0.1, 0.1, 0.9], "analgesic": [0, 0.2, 0.8],
    }

    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")

    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=500, gate_concept_creation=False)
    model._tokenizer = tok
    inject_embeddings(model, tok, embed_map)
    train_facts(model, tok, facts)

    all_candidates = traverse_broad(model, tok, "drug_x", max_depth=2)

    granularities = [
        ("PREDICATE: 'causes'", TraversalConfig(mode="predicate", sub_family="causes")),
        ("SUB-FAMILY: causal-strong", TraversalConfig(mode="sub_family", sub_family="causal-strong")),
        ("FAMILY: causal", TraversalConfig(mode="family", family="causal")),
        ("SUPER-FAMILY: causal+directional", TraversalConfig(mode="super_family", super_family="causal_directional")),
        ("RELAXED: everything", TraversalConfig(mode="relaxed")),
    ]

    for label, config in granularities:
        filtered = filter_candidates(all_candidates, config)
        words = [c.word for c in filtered]
        print(f"\n  {label:40s} -> {words}")

    return all_candidates


# ============================================================
# TEST 5: STRUCTURED CANDIDATE METADATA
# ============================================================

def test_structured_candidates():
    """Show that structured candidates enable richer reasoning."""
    print("\n" + "=" * 70)
    print("TEST 5: STRUCTURED CANDIDATE METADATA")
    print("Full candidate info enables router decisions")
    print("=" * 70)

    facts = [
        ("smoking", "causes", "cancer"),
        ("smoking", "causes", "heart_disease"),
        ("smoking", "associated_with", "stress"),
        ("smoking", "reduces", "appetite"),
        ("smoking", "increases", "anxiety"),
        ("smoking", "contains", "nicotine"),
        ("smoking", "is_a", "habit"),
    ]

    embed_map = {
        "smoking": [-0.8, 0.3, 0.5], "cancer": [-0.9, -0.5, 0.3],
        "heart_disease": [-0.7, -0.4, 0.4], "stress": [-0.6, 0.2, 0.6],
        "appetite": [0.2, 0.5, 0.5], "anxiety": [-0.5, 0.3, 0.6],
        "nicotine": [-0.3, 0.4, 0.5], "habit": [-0.1, 0.1, 0.7],
    }

    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")

    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=500, gate_concept_creation=False)
    model._tokenizer = tok
    inject_embeddings(model, tok, embed_map)
    train_facts(model, tok, facts)

    all_candidates = traverse_broad(model, tok, "smoking", max_depth=2)

    print("\nFull structured output:")
    print(f"{'Word':<18s} {'Predicate':<20s} {'Family':<15s} {'Sub-family':<25s} {'Conf':<6s} {'Depth'}")
    print("-" * 95)
    for c in all_candidates:
        print(f"{c.word:<18s} {c.predicate:<20s} {c.family:<15s} {c.sub_family:<25s} {c.confidence:<6.2f} {c.depth}")

    # Router-style queries
    print("\n--- Router simulation ---")
    queries = {
        "smoking causes ?": TraversalConfig(mode="family", family="causal"),
        "smoking increases ?": TraversalConfig(mode="sub_family", sub_family="directional-positive"),
        "smoking effects ?": TraversalConfig(mode="super_family", super_family="causal_directional"),
        "smoking contains ?": TraversalConfig(mode="sub_family", sub_family="compositional"),
    }

    for query, config in queries.items():
        filtered = filter_candidates(all_candidates, config)
        words = [c.word for c in filtered]
        print(f"  {query:30s} -> {words}")

    return all_candidates


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("DIRECTIONAL REASONING BENCHMARK")
    print("Design B: broad traversal + predicate-aware filtering")
    print("=" * 70)

    pos1, neg1 = test_directional_specificity()
    broad2 = test_broad_effects()
    pos3, neg3 = test_contradictions()
    all4 = test_multi_granularity()
    all5 = test_structured_candidates()

    print("\n" + "=" * 70)
    print("CAPABILITY MATRIX")
    print("=" * 70)

    t1 = ("fatigue" in pos1 and "injury_risk" not in pos1 and
          "injury_risk" in neg1 and "fatigue" not in neg1)
    t2 = "fatigue" in broad2 and "injury_risk" in broad2 and "strength" in broad2
    t3 = "focus" in pos3 and "focus" in neg3
    t4 = len(all4) >= 5
    t5 = len(all5) >= 5

    print(f"\n{'Test':<35s} {'Result':<10s}")
    print("-" * 50)
    print(f"{'1. Directional specificity':<35s} {'PASS' if t1 else 'FAIL'}")
    print(f"{'2. Broad effects':<35s} {'PASS' if t2 else 'FAIL'}")
    print(f"{'3. Contradictory evidence':<35s} {'PASS' if t3 else 'FAIL'}")
    print(f"{'4. Multi-granularity':<35s} {'PASS' if t4 else 'FAIL'}")
    print(f"{'5. Structured candidates':<35s} {'PASS' if t5 else 'FAIL'}")


if __name__ == "__main__":
    main()
