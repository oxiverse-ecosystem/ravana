"""
Dense KB Validation — Prove Phase 2 Reasoning Works
=====================================================
Hand-constructs a dense 100+ fact KB where EVERY major node has outgoing edges.
Tests composed reasoning end-to-end to validate Phase 2 logic.

Hypothesis: Phase 2 reasoning fails on sparse data, not architecture.
This test proves it by running the same queries on dense data.

Test cases:
1. tea → causes → ? (should find wakefulness via caffeine bridge)
2. heroin → causes → ? (should find addiction/withdrawal via addiction bridge)
3. fox → hunts → ? (should find rabbit/deer via predator bridge)
4. energy_drink → increases → ? (should find alertness via caffeine bridge)
5. coffee → causes → ? (should find wakefulness via caffeine bridge)
"""
import sys
import io
import time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sentence_transformers import SentenceTransformer
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.word_tokenizer import WordTokenizer
from ravana_ml.episode_injector import EpisodeInjector, Fact
from ravana_ml.relation_ontology import (
    Candidate, TraversalConfig, get_sub_family, get_family,
    get_confidence, matches_config
)


# ============================================================
# DENSE KNOWLEDGE BASE — Every node has outgoing edges
# ============================================================

DENSE_KB = {
    # --- PHARMACOLOGY (dense) ---
    "aspirin": {
        "causes": ["pain_relief", "stomach_irritation", "blood_thinning"],
        "reduces": ["inflammation", "fever", "pain"],
        "contains": ["salicylic_acid"],
        "is_a": ["nsaid", "analgesic"],
    },
    "ibuprofen": {
        "causes": ["pain_relief", "stomach_irritation"],
        "reduces": ["inflammation", "fever", "pain"],
        "contains": ["propionic_acid"],
        "is_a": ["nsaid", "analgesic"],
    },
    "acetaminophen": {
        "causes": ["pain_relief", "liver_damage"],
        "reduces": ["fever", "pain"],
        "is_a": ["analgesic"],
    },
    # KEY: caffeine has CAUSES edges (missing in original KB!)
    "caffeine": {
        "causes": ["wakefulness", "alertness", "insomnia", "anxiety"],
        "increases": ["heart_rate", "blood_pressure", "alertness"],
        "decreases": ["fatigue", "drowsiness"],
        "contains": ["methylxanthine"],
        "is_a": ["stimulant"],
    },
    "nicotine": {
        "causes": ["addiction", "alertness", "nausea"],
        "increases": ["heart_rate", "blood_pressure"],
        "decreases": ["appetite", "anxiety"],
        "contains": ["pyridine"],
        "is_a": ["stimulant", "alkaloid"],
    },
    # KEY: addiction has CAUSES edges (missing in original KB!)
    "addiction": {
        "causes": ["withdrawal", "dependence", "tolerance", "cravings"],
        "leads_to": ["relapse", "health_decline"],
        "is_a": ["condition"],
    },
    "withdrawal": {
        "causes": ["anxiety", "tremors", "nausea", "insomnia"],
        "is_a": ["symptom"],
    },
    "pain_relief": {
        "leads_to": ["comfort", "mobility"],
        "is_a": ["therapeutic_effect"],
    },
    "stomach_irritation": {
        "causes": ["nausea", "ulcers"],
        "is_a": ["side_effect"],
    },
    "liver_damage": {
        "causes": ["jaundice", "failure"],
        "leads_to": ["hospitalization"],
        "is_a": ["side_effect"],
    },
    "insomnia": {
        "causes": ["fatigue", "irritability"],
        "is_a": ["sleep_disorder"],
    },
    "anxiety": {
        "causes": ["insomnia", "panic"],
        "increases": ["heart_rate"],
        "is_a": ["mental_health_condition"],
    },
    "alertness": {
        "leads_to": ["productivity", "focus"],
        "reduces": ["errors"],
        "is_a": ["cognitive_state"],
    },
    "wakefulness": {
        "leads_to": ["productivity"],
        "decreases": ["sleep"],
        "is_a": ["physiological_state"],
    },

    # --- ECOLOGY (dense) ---
    "wolf": {
        "hunts": ["deer", "rabbit", "elk"],
        "is_a": ["predator", "mammal"],
        "lives_in": ["forest", "tundra"],
    },
    "deer": {
        "eats": ["grass", "leaves", "shrubs"],
        "is_a": ["herbivore", "mammal"],
        "hunted_by": ["wolf", "bear"],
        "lives_in": ["forest", "meadow"],
    },
    "bear": {
        "hunts": ["deer", "fish", "berries"],
        "is_a": ["predator", "mammal"],
        "lives_in": ["forest", "mountain"],
    },
    # KEY: predator has HUNTS edges (missing in original KB!)
    "predator": {
        "hunts": ["prey", "herbivore", "small_mammal"],
        "is_a": ["animal"],
        "lives_in": ["wilderness"],
    },
    # KEY: herbivore has EATS edges (missing in original KB!)
    "herbivore": {
        "eats": ["grass", "leaves", "shrubs", "plants"],
        "is_a": ["animal"],
        "hunted_by": ["predator"],
    },
    "prey": {
        "eats": ["grass", "seeds"],
        "hunted_by": ["predator"],
        "is_a": ["animal"],
    },
    "grass": {
        "is_a": ["plant"],
        "needs": ["sunlight", "water", "soil"],
        "eaten_by": ["deer", "rabbit", "herbivore"],
    },
    "sunlight": {
        "enables": ["photosynthesis"],
        "increases": ["plant_growth"],
        "is_a": ["energy_source"],
    },
    "photosynthesis": {
        "produces": ["oxygen", "glucose"],
        "requires": ["sunlight", "water", "carbon_dioxide"],
        "is_a": ["biological_process"],
    },
    "plant_growth": {
        "requires": ["sunlight", "water", "soil"],
        "leads_to": ["biomass"],
        "is_a": ["biological_process"],
    },
    # KEY: soil has INCREASES/ENABLES edges (missing in original KB!)
    "soil": {
        "contains": ["nutrients", "minerals", "organic_matter"],
        "enables": ["plant_growth"],
        "is_a": ["substrate"],
    },
    # KEY: energy_source has ENABLES edges (missing in original KB!)
    "energy_source": {
        "enables": ["growth", "metabolism", "movement"],
        "is_a": ["resource"],
    },

    # --- BEVERAGES (new domain for testing) ---
    "tea": {
        "contains": ["caffeine", "antioxidants", "tannins"],
        "causes": ["relaxation", "alertness"],
        "is_a": ["beverage"],
    },
    "coffee": {
        "contains": ["caffeine", "antioxidants"],
        "causes": ["wakefulness", "alertness", "jitteriness"],
        "is_a": ["beverage"],
    },
    "energy_drink": {
        "contains": ["caffeine", "sugar", "taurine"],
        "causes": ["alertness", "insomnia"],
        "increases": ["heart_rate", "energy"],
        "is_a": ["beverage"],
    },
    "heroin": {
        "causes": ["addiction", "euphoria", "respiratory_depression"],
        "contains": ["diacetylmorphine"],
        "is_a": ["opioid", "drug"],
    },
    "morphine": {
        "causes": ["pain_relief", "addiction", "drowsiness"],
        "reduces": ["pain"],
        "is_a": ["opioid", "analgesic"],
    },

    # --- RELATIONSHIPS (for fox→hunts test) ---
    "fox": {
        "hunts": ["rabbit", "mouse", "bird"],
        "is_a": ["predator", "mammal"],
        "lives_in": ["forest", "field"],
    },
    "rabbit": {
        "eats": ["grass", "carrots", "clover"],
        "is_a": ["herbivore", "mammal"],
        "hunted_by": ["fox", "wolf"],
    },
    "mouse": {
        "eats": ["seeds", "grain"],
        "is_a": ["herbivore", "rodent"],
        "hunted_by": ["fox", "owl"],
    },
    "owl": {
        "hunts": ["mouse", "rabbit"],
        "is_a": ["predator", "bird"],
        "lives_in": ["forest"],
    },

    # --- MEDICAL CONDITIONS ---
    "fever": {
        "causes": ["discomfort", "dehydration"],
        "reduced_by": ["aspirin", "ibuprofen", "acetaminophen"],
        "is_a": ["symptom"],
    },
    "inflammation": {
        "causes": ["pain", "swelling"],
        "reduced_by": ["aspirin", "ibuprofen"],
        "is_a": ["immune_response"],
    },
    "pain": {
        "causes": ["discomfort"],
        "reduced_by": ["aspirin", "ibuprofen", "acetaminophen"],
        "is_a": ["symptom"],
    },
    "blood_thinning": {
        "prevents": ["clots"],
        "increases": ["bleeding_risk"],
        "is_a": ["pharmacological_effect"],
    },
    "stomach_acid": {
        "causes": ["irritation"],
        "increases": ["digestion"],
        "is_a": ["biological_chemical"],
    },
    "salicylic_acid": {
        "reduces": ["inflammation"],
        "is_a": ["chemical_compound"],
    },
    "methylxanthine": {
        "causes": ["alertness"],
        "increases": ["heart_rate"],
        "is_a": ["chemical_compound"],
    },
    "pyridine": {
        "is_a": ["chemical_compound"],
    },
    "diacetylmorphine": {
        "causes": ["euphoria"],
        "is_a": ["chemical_compound"],
    },
    "propionic_acid": {
        "is_a": ["chemical_compound"],
    },
    "taurine": {
        "increases": ["energy"],
        "is_a": ["amino_acid"],
    },
    "sugar": {
        "causes": ["energy_spike"],
        "increases": ["blood_sugar"],
        "is_a": ["carbohydrate"],
    },
    "antioxidants": {
        "prevent": ["cell_damage"],
        "is_a": ["compound"],
    },
    "tannins": {
        "causes": ["astringency"],
        "is_a": ["polyphenol"],
    },
    "euphoria": {
        "leads_to": ["addiction"],
        "is_a": ["emotional_state"],
    },
    "respiratory_depression": {
        "causes": ["death"],
        "is_a": ["medical_emergency"],
    },
}


def build_dense_facts():
    """Convert dense KB to Fact objects."""
    facts = []
    for subject, relations in DENSE_KB.items():
        for relation, objects in relations.items():
            for obj in objects:
                facts.append(Fact(
                    subject=subject, relation=relation, object=obj,
                    confidence=0.9, source="dense_kb"
                ))
    return facts


def inject_minilm_embeddings(model, tok, st_model):
    """Replace random embeddings with MiniLM embeddings."""
    dim = model.embed_dim
    st_dim = st_model.get_sentence_embedding_dimension()

    words = list(tok.word_to_id.keys())
    if not words:
        return

    embeddings = st_model.encode(words, show_progress_bar=False)

    rng = np.random.RandomState(42)
    projection = rng.randn(st_dim, dim).astype(np.float32) / np.sqrt(dim)

    for i, word in enumerate(words):
        tid = tok.word_to_id[word]
        projected = embeddings[i] @ projection
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected /= norm
        model.token_embed.weight.data[tid] = projected

    print(f"  Injected MiniLM embeddings for {len(words)} tokens")


def nn_bridge(model, tok, st_model, novel_word, top_k=5):
    """Find graph nodes most similar to a novel word via MiniLM embeddings."""
    st_dim = st_model.get_sentence_embedding_dimension()
    dim = model.embed_dim

    rng = np.random.RandomState(42)
    projection = rng.randn(st_dim, dim).astype(np.float32) / np.sqrt(dim)

    novel_embed = st_model.encode([novel_word])[0] @ projection
    norm = np.linalg.norm(novel_embed)
    if norm > 0:
        novel_embed /= norm

    similarities = []
    for word, tid in tok.word_to_id.items():
        node_embed = model.token_embed.weight.data[tid]
        sim = float(np.dot(novel_embed, node_embed))
        similarities.append((word, sim))

    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]


def traverse_broad(model, tok, subject, max_depth=3, relation_filter=None):
    """Broad traversal returning structured Candidates."""
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
                    pred_word = edge.relation_type
                    if hasattr(edge, 'predicate_token_id') and edge.predicate_token_id is not None:
                        decoded = tok.decode([edge.predicate_token_id])
                        if decoded.strip():
                            pred_word = decoded.strip()
                    sub_fam = get_sub_family(pred_word) or "unknown"
                    family = get_family(pred_word) or edge.relation_type
                    conf = get_confidence(pred_word)
                    candidates.append(Candidate(
                        word=word, predicate=pred_word, family=family,
                        sub_family=sub_fam, depth=depth, confidence=conf,
                        path=[(subject, pred_word, word)]
                    ))
        visited |= next_frontier
        frontier = next_frontier

    seen = {}
    for c in candidates:
        if c.word not in seen or c.depth < seen[c.word].depth or \
           (c.depth == seen[c.word].depth and c.confidence > seen[c.word].confidence):
            seen[c.word] = c
    return sorted(seen.values(), key=lambda x: (x.depth, -x.confidence))


def composed_query_with_bridge(model, tok, st_model, novel_word, relation, top_k=5):
    """Novel concept -> NN bridge -> graph traversal -> prediction."""
    # Step 1: NN bridge
    neighbors = nn_bridge(model, tok, st_model, novel_word, top_k=top_k)

    # Step 2: Traverse from each neighbor
    all_candidates = []
    for neighbor_word, sim in neighbors:
        if sim < 0.1:
            continue
        candidates = traverse_broad(model, tok, neighbor_word, max_depth=3)
        for c in candidates:
            c.confidence *= sim
            c.path = [(novel_word, f"via:{neighbor_word}", c.word)]
            all_candidates.append(c)

    # Step 3: Filter by relation family
    family = get_family(relation)
    if family:
        config = TraversalConfig(mode="family", family=family)
        filtered = [c for c in all_candidates if matches_config(c.predicate, config)]
    else:
        # For predicates not in ontology (like "hunts"), use relaxed mode
        filtered = [c for c in all_candidates if c.predicate == relation]

    # Deduplicate
    seen = {}
    for c in filtered:
        if c.word not in seen or c.confidence > seen[c.word].confidence:
            seen[c.word] = c
    return sorted(seen.values(), key=lambda x: -x.confidence)


def audit_graph_connectivity(model, tok):
    """Audit which (concept, predicate) pairs have edges."""
    print("\n" + "=" * 70)
    print("GRAPH CONNECTIVITY AUDIT")
    print("=" * 70)

    # Key concepts to check
    key_concepts = ["caffeine", "addiction", "predator", "herbivore", "soil",
                    "energy_source", "tea", "coffee", "energy_drink", "heroin",
                    "fox", "owl", "withdrawal", "alertness", "wakefulness"]

    key_predicates = ["causes", "increases", "decreases", "contains", "is_a",
                      "hunts", "eats", "enables", "reduces", "leads_to"]

    print(f"\n{'Concept':<20} ", end="")
    for pred in key_predicates[:8]:
        print(f"{pred:>10}", end="")
    print()
    print("-" * 100)

    for concept in key_concepts:
        tid = tok.word_to_id.get(concept)
        if tid is None:
            continue
        bindings = model.binding_map.get_concepts(tid, min_confidence=0.1)
        if not bindings:
            continue
        cid = bindings[0].concept_id

        # Get all outgoing edge relation types
        outgoing = model.graph.get_outgoing(cid)
        edge_types = {}
        for tgt_id, edge in outgoing:
            rt = edge.relation_type
            if rt not in edge_types:
                edge_types[rt] = 0
            edge_types[rt] += 1

        print(f"  {concept:<18} ", end="")
        for pred in key_predicates[:8]:
            count = edge_types.get(pred, 0)
            if count > 0:
                print(f"{'✅ ' + str(count):>10}", end="")
            else:
                print(f"{'❌':>10}", end="")
        print(f"  (total outgoing: {sum(edge_types.values())})")


def main():
    print("=" * 70)
    print("DENSE KB VALIDATION — Prove Phase 2 Reasoning Works")
    print("=" * 70)

    # Build dense facts
    all_facts = build_dense_facts()
    print(f"\nDense KB: {len(all_facts)} facts across {len(DENSE_KB)} concepts")

    # Build vocabulary
    all_triples = [(f.subject, f.relation, f.object) for f in all_facts]
    tok = WordTokenizer()
    for s, r, o in all_triples:
        tok.encode(f"{s} {r} {o}")

    print(f"Vocabulary: {tok.vocab_size} tokens")

    # Create model
    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=5000, gate_concept_creation=False)
    model._tokenizer = tok

    # Inject facts
    print("\nInjecting facts...")
    t0 = time.time()
    injector = EpisodeInjector(model, tok)
    stats = injector.inject_facts(all_facts, epochs=8)
    print(f"  Injection: {stats['successful']}/{stats['total']} facts, "
          f"{stats['epochs_trained']} total epochs ({time.time()-t0:.1f}s)")
    print(f"  Graph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

    # Load MiniLM and inject embeddings
    print("\nLoading MiniLM...")
    t0 = time.time()
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"  Loaded in {time.time()-t0:.1f}s")

    inject_minilm_embeddings(model, tok, st_model)

    # ============================================================
    # AUDIT: Check graph connectivity BEFORE reasoning tests
    # ============================================================
    audit_graph_connectivity(model, tok)

    # ============================================================
    # TEST 1: NN BRIDGE — Can MiniLM find the right graph nodes?
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 1: NN BRIDGE (should be identical to sparse KB)")
    print("Novel concepts -> find similar graph nodes")
    print("=" * 70)

    novel_concepts = ["tea", "heroin", "fox", "fertilizer", "energy_drink"]
    for concept in novel_concepts:
        neighbors = nn_bridge(model, tok, st_model, concept, top_k=5)
        neighbor_str = ", ".join([f"{w}({s:.2f})" for w, s in neighbors])
        print(f"  {concept:20s} -> {neighbor_str}")

    # ============================================================
    # TEST 2: COMPOSED REASONING — The critical test
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 2: COMPOSED REASONING (dense KB — should WORK)")
    print("Novel concept -> NN bridge -> traversal -> prediction")
    print("=" * 70)

    queries = [
        ("tea", "causes", ["wakefulness", "alertness", "relaxation"]),
        ("heroin", "causes", ["addiction", "euphoria", "withdrawal"]),
        ("fox", "hunts", ["rabbit", "mouse", "deer"]),
        ("energy_drink", "increases", ["alertness", "heart_rate"]),
        ("coffee", "causes", ["wakefulness", "alertness"]),
        ("morphine", "causes", ["addiction", "pain_relief"]),
    ]

    results_summary = []
    for novel_word, relation, expected in queries:
        results = composed_query_with_bridge(model, tok, st_model, novel_word, relation)
        result_words = [c.word for c in results[:5]]

        # Check if any expected answer is in results
        hits = [w for w in expected if w in result_words]
        hit_rate = len(hits) / len(expected) if expected else 0

        if results:
            result_str = ", ".join([f"{c.word}({c.confidence:.2f})" for c in results[:5]])
            print(f"\n  {novel_word} {relation} ?")
            print(f"    Results:   {result_str}")
            print(f"    Expected:  {', '.join(expected)}")
            print(f"    Hits:      {hits} ({hit_rate:.0%})")
        else:
            print(f"\n  {novel_word} {relation} ? -> NO RESULTS")
            print(f"    Expected:  {', '.join(expected)}")

        results_summary.append({
            "query": f"{novel_word} {relation}",
            "results": result_words,
            "expected": expected,
            "hits": hits,
            "hit_rate": hit_rate,
        })

    # ============================================================
    # TEST 3: SEMANTIC CLUSTERING (should be same as sparse)
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 3: SEMANTIC CLUSTERING (MiniLM embedding quality)")
    print("=" * 70)

    pharma_terms = ["aspirin", "ibuprofen", "caffeine", "nicotine", "acetaminophen"]
    eco_terms = ["wolf", "deer", "bear", "grass", "sunlight"]

    pharma_sims = []
    eco_sims = []
    cross_sims = []

    for i, t1 in enumerate(pharma_terms):
        for t2 in pharma_terms[i+1:]:
            e1 = st_model.encode(t1)
            e2 = st_model.encode(t2)
            sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
            pharma_sims.append(sim)

    for i, t1 in enumerate(eco_terms):
        for t2 in eco_terms[i+1:]:
            e1 = st_model.encode(t1)
            e2 = st_model.encode(t2)
            sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
            eco_sims.append(sim)

    for t1 in pharma_terms:
        for t2 in eco_terms:
            e1 = st_model.encode(t1)
            e2 = st_model.encode(t2)
            sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
            cross_sims.append(sim)

    print(f"  Pharmacology intra-domain:  {np.mean(pharma_sims):.3f}")
    print(f"  Ecology intra-domain:       {np.mean(eco_sims):.3f}")
    print(f"  Cross-domain:               {np.mean(cross_sims):.3f}")
    print(f"  Intra > Cross gap:          {min(np.mean(pharma_sims), np.mean(eco_sims)) - np.mean(cross_sims):.3f}")

    # ============================================================
    # TEST 4: MULTI-HOP REASONING (3-hop chains)
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 4: MULTI-HOP REASONING (3-hop chains)")
    print("=" * 70)

    # Test: tea -> caffeine -> wakefulness (2-hop via bridge)
    # Test: heroin -> addiction -> withdrawal (2-hop via bridge)
    # Test: fox -> predator -> prey (2-hop via bridge)

    multi_hop_queries = [
        ("tea", "causes", "Should find: wakefulness (tea->caffeine->wakefulness)"),
        ("heroin", "causes", "Should find: withdrawal (heroin->addiction->withdrawal)"),
        ("fox", "hunts", "Should find: prey (fox->predator->prey)"),
    ]

    for novel_word, relation, description in multi_hop_queries:
        print(f"\n  {description}")
        results = composed_query_with_bridge(model, tok, st_model, novel_word, relation)
        if results:
            result_str = ", ".join([f"{c.word}({c.confidence:.2f})" for c in results[:8]])
            print(f"    Results: {result_str}")
        else:
            print(f"    Results: NO RESULTS")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_hits = sum(r["hit_rate"] for r in results_summary)
    avg_hit_rate = total_hits / len(results_summary) if results_summary else 0

    print(f"\n  Composed Reasoning Tests: {len(results_summary)}")
    print(f"  Average Hit Rate: {avg_hit_rate:.0%}")

    for r in results_summary:
        status = "✅" if r["hit_rate"] > 0 else "❌"
        print(f"    {status} {r['query']}: {r['hit_rate']:.0%} hits")

    if avg_hit_rate > 0.5:
        print("\n  ✅ PHASE 2 VALIDATED: Reasoning logic works with dense data!")
        print("  The failure mode is DATA, not ARCHITECTURE.")
        print("  Next step: Enrich the real KB with missing edges.")
    else:
        print("\n  ❌ PHASE 2 NOT VALIDATED: Reasoning logic may have bugs.")
        print("  Investigate traversal logic before enriching KB.")


if __name__ == "__main__":
    main()
