"""
Noisy Corpus Benchmark
=======================
Tests reasoning under realistic conditions:
  Tier 1: Linguistic variation (causes, contributes_to, associated_with, ...)
  Tier 2: Cross-relation chains (contains -> affects)
  Tier 3: Confidence ranking (strong vs weak evidence)
  Tier 4: Contradictions (increases vs decreases)
"""
import sys
import io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ravana_ml.nn.rlm_v2 import RLMv2, RELATION_TYPES
from ravana_ml.word_tokenizer import WordTokenizer


RELATION_FAMILY = {
    "causes": "causal", "produces": "causal", "creates": "causal",
    "leads_to": "causal", "triggers": "causal", "results_in": "causal",
    "contributes_to": "causal", "associated_with": "causal",
    "linked_to": "causal", "may_cause": "causal",
    "affects": "causal", "influences": "causal",
    "increases": "causal", "decreases": "causal",
    "improves": "causal", "worsens": "causal",
    "reduces": "causal", "enhances": "causal",
    "prevents": "causal", "inhibits": "causal",
    "contains": "possessive", "includes": "possessive",
    "has": "possessive", "comprises": "possessive",
    "is_a": "semantic", "is_type_of": "semantic",
    "means": "semantic", "defines": "semantic",
    "precedes": "temporal", "follows": "temporal",
}

RELATION_CONFIDENCE = {
    "causes": 0.95, "produces": 0.90, "creates": 0.90,
    "leads_to": 0.85, "triggers": 0.85, "results_in": 0.85,
    "contributes_to": 0.60, "associated_with": 0.45,
    "linked_to": 0.50, "may_cause": 0.55,
    "affects": 0.70, "influences": 0.65,
    "increases": 0.75, "decreases": 0.75,
    "improves": 0.80, "worsens": 0.80,
    "reduces": 0.70, "enhances": 0.75,
    "prevents": 0.80, "inhibits": 0.75,
    "contains": 0.95, "includes": 0.90, "has": 0.95,
    "comprises": 0.85, "is_a": 0.90, "is_type_of": 0.85,
}


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


def traverse(model, tok, query, max_depth=3, strict=True):
    """Relation-aware traversal with family matching.
    strict=True: only follow edges matching query relation family
    strict=False: follow any edge (for cross-relation chains)
    """
    parts = query.split()
    if len(parts) < 2:
        return []
    subj_word, rel_word = parts[0], parts[1]

    subj_tid = tok.word_to_id.get(subj_word)
    if subj_tid is None:
        return []
    bindings = model.binding_map.get_concepts(subj_tid, min_confidence=0.1)
    if not bindings:
        return []

    query_family = RELATION_FAMILY.get(rel_word, None)

    frontier = {bindings[0].concept_id}
    visited = {bindings[0].concept_id}
    candidates = []

    for depth in range(1, max_depth + 1):
        next_frontier = set()
        for nid in frontier:
            for tgt_id, edge in model.graph.get_outgoing(nid):
                if tgt_id in visited:
                    continue
                if strict and query_family:
                    edge_family = RELATION_FAMILY.get(edge.relation_type, edge.relation_type)
                    if edge_family != query_family:
                        continue
                next_frontier.add(tgt_id)
                tokens = model.binding_map.get_tokens(tgt_id, 0.0)
                for b in tokens:
                    word = tok.decode([b.token_id])
                    if word not in (subj_word, rel_word) and not word.startswith("?"):
                        pred_word = edge.relation_type
                        if hasattr(edge, 'predicate_token_id') and edge.predicate_token_id is not None:
                            decoded = tok.decode([edge.predicate_token_id])
                            if decoded.strip():
                                pred_word = decoded.strip()
                        conf = RELATION_CONFIDENCE.get(pred_word, 0.5)
                        candidates.append((word, depth, edge.relation_type, conf, pred_word))
        visited |= next_frontier
        frontier = next_frontier

    seen = {}
    for word, depth, rel, conf, pred in candidates:
        if word not in seen or depth < seen[word][0] or (depth == seen[word][0] and conf > seen[word][1]):
            seen[word] = (depth, conf, rel, pred)
    return sorted(seen.items(), key=lambda x: (x[1][0], -x[1][1]))


def show_results(results, label=""):
    if label:
        print(f"\n{label} ({len(results)} results):")
    for word, (depth, conf, rel, pred) in results:
        print(f"  depth {depth} conf {conf:.2f} via {pred:20s}: {word}")


# ============================================================
# TIER 1: LINGUISTIC VARIATION
# ============================================================

def tier1():
    print("=" * 70)
    print("TIER 1: LINGUISTIC VARIATION")
    print("Same relation expressed with different words")
    print("=" * 70)

    facts = [
        ("coffee", "causes", "alertness"),
        ("coffee", "contributes_to", "anxiety"),
        ("coffee", "associated_with", "insomnia"),
        ("coffee", "affects", "metabolism"),
        ("coffee", "triggers", "acid_reflux"),
        ("coffee", "produces", "jitteriness"),
        ("tea", "causes", "calmness"),
        ("tea", "leads_to", "hydration"),
        ("tea", "associated_with", "antioxidants"),
        ("coffee", "contains", "caffeine"),
        ("coffee", "has", "flavor"),
        ("tea", "contains", "tannin"),
        ("tea", "is_a", "beverage"),
    ]

    embed_map = {
        "coffee": [-1, 0.8, 0.2], "tea": [-0.5, 0.7, 0.3],
        "alertness": [-0.8, 0.6, 0.3], "anxiety": [-0.9, 0.4, 0.5],
        "insomnia": [-0.7, 0.3, 0.6], "metabolism": [-0.6, 0.5, 0.4],
        "acid_reflux": [-0.85, 0.35, 0.55], "jitteriness": [-0.75, 0.45, 0.5],
        "calmness": [0.5, 0.8, 0.2], "hydration": [0.4, 0.7, 0.3],
        "antioxidants": [0.3, 0.6, 0.4],
        "caffeine": [-0.3, 0.5, 0.5], "flavor": [-0.2, 0.4, 0.6],
        "tannin": [0.2, 0.5, 0.5], "beverage": [0, 0.3, 0.7],
    }

    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")

    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=500, gate_concept_creation=False)
    model._tokenizer = tok
    inject_embeddings(model, tok, embed_map)
    train_facts(model, tok, facts)

    print(f"\nGraph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

    # Check what relation types the classifier assigned
    print("\n--- Edge relation types ---")
    for (src, tgt), edge in list(model.graph.edges.items())[:15]:
        src_toks = model.binding_map.get_tokens(src, 0.0)
        tgt_toks = model.binding_map.get_tokens(tgt, 0.0)
        sl = tok.decode([src_toks[0].token_id]) if src_toks else f"c{src}"
        tl = tok.decode([tgt_toks[0].token_id]) if tgt_toks else f"c{tgt}"
        pred = edge.relation_type
        if hasattr(edge, 'predicate_token_id') and edge.predicate_token_id is not None:
            decoded = tok.decode([edge.predicate_token_id])
            if decoded.strip():
                pred = decoded.strip()
        print(f"  {sl:15s} --[{pred:20s}]--> {tl:15s}  (type={edge.relation_type})")

    # Test: coffee causes ? (strict causal family)
    results = traverse(model, tok, "coffee causes", max_depth=2)
    show_results(results, "coffee causes ? (strict)")

    # Test: coffee causes ? (relaxed — any edge)
    results_relaxed = traverse(model, tok, "coffee causes", max_depth=2, strict=False)
    show_results(results_relaxed, "coffee causes ? (relaxed)")

    # Check: are all causal effects found?
    expected_causal = {"alertness", "anxiety", "insomnia", "metabolism", "acid_reflux", "jitteriness"}
    found_strict = {w for w, _ in results}
    found_relaxed = {w for w, _ in results_relaxed}

    print(f"\nStrict causal found: {found_strict & expected_causal} ({len(found_strict & expected_causal)}/6)")
    print(f"Relaxed found: {found_relaxed & expected_causal} ({len(found_relaxed & expected_causal)}/6)")

    return results


# ============================================================
# TIER 2: CROSS-RELATION CHAINS
# ============================================================

def tier2():
    print("\n" + "=" * 70)
    print("TIER 2: CROSS-RELATION CHAINS")
    print("coffee contains caffeine, caffeine affects sleep")
    print("Query: coffee affects ? (must cross contains -> affects)")
    print("=" * 70)

    facts = [
        ("coffee", "contains", "caffeine"),
        ("caffeine", "affects", "sleep"),
        ("caffeine", "affects", "heart_rate"),
        ("caffeine", "triggers", "anxiety"),
        ("battery", "has", "lithium"),
        ("lithium", "causes", "pollution"),
        ("energy_drink", "contains", "coffee"),
        ("espresso", "is_a", "coffee"),
        ("coffee", "has", "flavor"),
    ]

    embed_map = {
        "coffee": [-1, 0.8, 0.2], "caffeine": [-0.8, 0.6, 0.4],
        "sleep": [-0.3, 0.2, 0.8], "heart_rate": [-0.5, 0.5, 0.5],
        "anxiety": [-0.9, 0.4, 0.5], "battery": [0.5, 0.7, 0.1],
        "lithium": [0.3, 0.5, 0.3], "pollution": [0.1, 0.3, 0.5],
        "energy_drink": [-0.5, 0.9, 0.1], "espresso": [-0.9, 0.75, 0.25],
        "flavor": [-0.2, 0.4, 0.6],
    }

    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")

    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=500, gate_concept_creation=False)
    model._tokenizer = tok
    inject_embeddings(model, tok, embed_map)
    train_facts(model, tok, facts)

    print(f"\nGraph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

    # Strict: coffee affects ? (won't cross contains edge)
    results_strict = traverse(model, tok, "coffee affects", max_depth=3, strict=True)
    show_results(results_strict, "coffee affects ? (strict)")

    # Relaxed: coffee affects ? (follows any edge)
    results_relaxed = traverse(model, tok, "coffee affects", max_depth=3, strict=False)
    show_results(results_relaxed, "coffee affects ? (relaxed)")

    found_relaxed = {w for w, _ in results_relaxed}
    print(f"\nReached sleep via cross-relation: {'sleep' in found_relaxed}")
    print(f"Reached heart_rate: {'heart_rate' in found_relaxed}")

    # energy_drink affects ? (2-hop: contains -> contains -> affects)
    results_energy = traverse(model, tok, "energy_drink affects", max_depth=4, strict=False)
    show_results(results_energy, "energy_drink affects ? (relaxed)")
    found_energy = {w for w, _ in results_energy}
    print(f"Reached sleep via 2-hop contains bridge: {'sleep' in found_energy}")

    return results_relaxed


# ============================================================
# TIER 3: CONFIDENCE RANKING
# ============================================================

def tier3():
    print("\n" + "=" * 70)
    print("TIER 3: CONFIDENCE RANKING")
    print("Strong evidence should rank above weak evidence")
    print("=" * 70)

    facts = [
        ("chocolate", "causes", "happiness"),        # 0.95
        ("chocolate", "produces", "energy"),         # 0.90
        ("chocolate", "contributes_to", "obesity"),  # 0.60
        ("chocolate", "associated_with", "acne"),    # 0.45
        ("chocolate", "may_cause", "migraine"),      # 0.55
        ("chocolate", "contains", "sugar"),
        ("chocolate", "contains", "cocoa"),
        ("exercise", "causes", "fitness"),
        ("exercise", "leads_to", "endurance"),
        ("exercise", "associated_with", "longevity"),
        ("exercise", "may_cause", "injury"),
        ("exercise", "triggers", "endorphins"),
    ]

    embed_map = {
        "chocolate": [-1, 0.8, 0.2], "happiness": [-0.8, 0.7, 0.3],
        "energy": [-0.7, 0.6, 0.4], "obesity": [-0.9, 0.3, 0.5],
        "acne": [-0.6, 0.4, 0.6], "migraine": [-0.85, 0.35, 0.55],
        "sugar": [-0.5, 0.5, 0.5], "cocoa": [-0.4, 0.6, 0.4],
        "exercise": [0.5, 0.8, 0.2], "fitness": [0.7, 0.7, 0.3],
        "endurance": [0.6, 0.6, 0.4], "longevity": [0.4, 0.5, 0.5],
        "injury": [0.3, 0.4, 0.6], "endorphins": [0.8, 0.6, 0.3],
    }

    tok = WordTokenizer()
    for s, r, o in facts:
        tok.encode(f"{s} {r} {o}")

    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=500, gate_concept_creation=False)
    model._tokenizer = tok
    inject_embeddings(model, tok, embed_map)
    train_facts(model, tok, facts)

    print(f"\nGraph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

    results = traverse(model, tok, "chocolate causes", max_depth=2)
    show_results(results, "chocolate causes ? (sorted by confidence)")

    if len(results) >= 2:
        confs = [conf for _, (_, conf, _, _) in results]
        is_sorted = all(confs[i] >= confs[i+1] for i in range(len(confs)-1))
        print(f"\nConfidence-sorted: {'YES' if is_sorted else 'NO'}")
        print(f"Confidence values: {[f'{c:.2f}' for c in confs]}")

    return results


# ============================================================
# TIER 4: CONTRADICTIONS
# ============================================================

def tier4():
    print("\n" + "=" * 70)
    print("TIER 4: CONTRADICTIONS")
    print("X increases Y AND X decreases Y")
    print("=" * 70)

    facts = [
        ("coffee", "increases", "focus"),
        ("coffee", "decreases", "focus"),
        ("stress", "increases", "cortisol"),
        ("stress", "decreases", "immunity"),
        ("stress", "increases", "alertness"),
        ("sunlight", "increases", "vitamin_d"),
        ("sunlight", "increases", "mood"),
        ("rain", "decreases", "temperature"),
        ("sleep", "improves", "memory"),
        ("sleep_deprivation", "worsens", "memory"),
        ("coffee", "contains", "caffeine"),
    ]

    embed_map = {
        "coffee": [-1, 0.8, 0.2], "focus": [-0.5, 0.5, 0.5],
        "stress": [-0.8, 0.3, 0.6], "cortisol": [-0.7, 0.4, 0.5],
        "immunity": [-0.6, 0.6, 0.4], "alertness": [-0.5, 0.7, 0.3],
        "sunlight": [1, 0.9, 0.1], "vitamin_d": [0.8, 0.7, 0.3],
        "mood": [0.7, 0.8, 0.2], "rain": [0.3, 0.2, 0.8],
        "temperature": [0.4, 0.3, 0.7],
        "sleep": [0.2, 0.1, 0.9], "memory": [0, 0.5, 0.5],
        "sleep_deprivation": [-0.2, 0.2, 0.8],
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

    print(f"\nGraph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

    r_inc = traverse(model, tok, "coffee increases", max_depth=2)
    r_dec = traverse(model, tok, "coffee decreases", max_depth=2)
    show_results(r_inc, "coffee increases ?")
    show_results(r_dec, "coffee decreases ?")

    inc_words = {w for w, _ in r_inc}
    dec_words = {w for w, _ in r_dec}
    print(f"\nFocus in increases: {'focus' in inc_words}")
    print(f"Focus in decreases: {'focus' in dec_words}")
    print(f"Both sides discoverable: {'focus' in inc_words and 'focus' in dec_words}")

    # Stress
    r_s_inc = traverse(model, tok, "stress increases", max_depth=2)
    r_s_dec = traverse(model, tok, "stress decreases", max_depth=2)
    show_results(r_s_inc, "stress increases ?")
    show_results(r_s_dec, "stress decreases ?")

    return r_inc, r_dec


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("NOISY CORPUS BENCHMARK")
    print("=" * 70)

    r1 = tier1()
    r2 = tier2()
    r3 = tier3()
    r4_inc, r4_dec = tier4()

    print("\n" + "=" * 70)
    print("CAPABILITY MATRIX")
    print("=" * 70)

    t1_expected = {"alertness", "anxiety", "insomnia", "metabolism", "acid_reflux", "jitteriness"}
    t1_found = {w for w, _ in r1}
    t1_hit = len(t1_found & t1_expected) >= 4

    t2_found = {w for w, _ in r2}
    t2_hit = "sleep" in t2_found

    if len(r3) >= 2:
        t3_confs = [conf for _, (_, conf, _, _) in r3]
        t3_sorted = all(t3_confs[i] >= t3_confs[i+1] for i in range(len(t3_confs)-1))
    else:
        t3_sorted = False

    t4_inc = {w for w, _ in r4_inc}
    t4_dec = {w for w, _ in r4_dec}
    t4_hit = "focus" in t4_inc and "focus" in t4_dec

    print(f"\n{'Tier':<25s} {'Result':<10s} {'Detail'}")
    print("-" * 65)
    print(f"{'1. Linguistic variation':<25s} {'PASS' if t1_hit else 'FAIL':<10s} "
          f"{len(t1_found & t1_expected)}/6 causal effects found")
    print(f"{'2. Cross-relation chains':<25s} {'PASS' if t2_hit else 'FAIL':<10s} "
          f"sleep via contains->affects: {t2_hit}")
    print(f"{'3. Confidence ranking':<25s} {'PASS' if t3_sorted else 'FAIL':<10s} "
          f"sorted: {t3_sorted}")
    print(f"{'4. Contradictions':<25s} {'PASS' if t4_hit else 'FAIL':<10s} "
          f"focus in both inc/dec: {t4_hit}")


if __name__ == "__main__":
    main()
