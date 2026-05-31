"""
Held-Out Term Transfer Test — Prove RAVANA Reasons About Novel Concepts
========================================================================
Tests whether RAVANA can reason about concepts it has NEVER been trained on,
using only the NN bridge (MiniLM similarity) + graph traversal.

Method:
1. Build dense KB (248 facts, 51 concepts)
2. Hold out 12 terms from injection (never trained)
3. But they're in vocabulary (get MiniLM embeddings)
4. Run phase 2 queries on held-out terms
5. Measure: can RAVANA infer correct relations via bridge?

This is the ACTUAL RAVANA use case: reasoning about novel concepts.
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
# DENSE KB (same as experiment_dense_kb_validation.py)
# ============================================================

DENSE_KB = {
    # --- PHARMACOLOGY ---
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

    # --- ECOLOGY ---
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
    "predator": {
        "hunts": ["prey", "herbivore", "small_mammal"],
        "is_a": ["animal"],
        "lives_in": ["wilderness"],
    },
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
    "soil": {
        "contains": ["nutrients", "minerals", "organic_matter"],
        "enables": ["plant_growth"],
        "is_a": ["substrate"],
    },
    "energy_source": {
        "enables": ["growth", "metabolism", "movement"],
        "is_a": ["resource"],
    },

    # --- BEVERAGES ---
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

    # --- ANIMALS ---
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


# ============================================================
# HELD-OUT TERMS — These are NEVER injected into the model
# ============================================================

HELD_OUT = {
    # Beverages (similar to tea/coffee/energy_drink)
    "matcha": {
        "bridge_to": ["tea", "coffee"],
        "expected": {
            "causes": ["wakefulness", "alertness"],
            "contains": ["caffeine"],
        },
        "description": "Japanese green tea — should bridge to tea/coffee",
    },
    "espresso": {
        "bridge_to": ["coffee", "tea"],
        "expected": {
            "causes": ["wakefulness", "alertness"],
            "contains": ["caffeine"],
        },
        "description": "Concentrated coffee — should bridge to coffee",
    },
    "yerba_mate": {
        "bridge_to": ["tea", "coffee"],
        "expected": {
            "causes": ["alertness"],
            "contains": ["caffeine"],
        },
        "description": "South American tea — should bridge to tea",
    },

    # Drugs (similar to heroin/morphine)
    "cocaine": {
        "bridge_to": ["heroin", "morphine"],
        "expected": {
            "causes": ["addiction", "euphoria"],
        },
        "description": "Stimulant drug — should bridge to heroin/morphine",
    },
    "fentanyl": {
        "bridge_to": ["heroin", "morphine"],
        "expected": {
            "causes": ["addiction", "respiratory_depression"],
        },
        "description": "Synthetic opioid — should bridge to heroin",
    },
    "amphetamines": {
        "bridge_to": ["nicotine", "caffeine"],
        "expected": {
            "causes": ["addiction", "alertness"],
            "increases": ["heart_rate"],
        },
        "description": "Stimulant — should bridge to nicotine/caffeine",
    },

    # Animals (similar to wolf/fox/bear)
    "coyote": {
        "bridge_to": ["wolf", "fox"],
        "expected": {
            "hunts": ["rabbit", "deer"],
            "is_a": ["predator"],
        },
        "description": "Canid predator — should bridge to wolf/fox",
    },
    "lynx": {
        "bridge_to": ["wolf", "fox"],
        "expected": {
            "hunts": ["rabbit", "deer"],
            "is_a": ["predator"],
        },
        "description": "Feline predator — should bridge to wolf/fox",
    },
    "squirrel": {
        "bridge_to": ["rabbit", "mouse"],
        "expected": {
            "eats": ["grass", "seeds"],
            "is_a": ["herbivore"],
        },
        "description": "Rodent — should bridge to rabbit/mouse",
    },

    # Plants (similar to grass)
    "fern": {
        "bridge_to": ["grass"],
        "expected": {
            "is_a": ["plant"],
            "needs": ["sunlight", "water"],
        },
        "description": "Plant — should bridge to grass",
    },

    # Stimulants (similar to caffeine)
    "theobromine": {
        "bridge_to": ["caffeine", "methylxanthine"],
        "expected": {
            "increases": ["heart_rate"],
            "is_a": ["stimulant"],
        },
        "description": "Chocolate stimulant — should bridge to caffeine",
    },

    # Medical ( similar to pain_relief)
    "massage": {
        "bridge_to": ["pain_relief"],
        "expected": {
            "leads_to": ["comfort"],
            "is_a": ["therapeutic_effect"],
        },
        "description": "Therapy — should bridge to pain_relief",
    },
}


def build_facts_excluding(exclude_terms):
    """Build facts from DENSE_KB, excluding any that mention excluded terms."""
    exclude = set(exclude_terms)
    facts = []
    for subject, relations in DENSE_KB.items():
        if subject in exclude:
            continue
        for relation, objects in relations.items():
            for obj in objects:
                if obj in exclude:
                    continue
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


def traverse_broad(model, tok, subject, max_depth=3):
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


def composed_query(model, tok, st_model, novel_word, relation, top_k=5):
    """Novel concept -> NN bridge -> graph traversal -> prediction."""
    neighbors = nn_bridge(model, tok, st_model, novel_word, top_k=top_k)

    all_candidates = []
    for neighbor_word, sim in neighbors:
        if sim < 0.1:
            continue
        candidates = traverse_broad(model, tok, neighbor_word, max_depth=3)
        for c in candidates:
            c.confidence *= sim
            c.path = [(novel_word, f"via:{neighbor_word}", c.word)]
            all_candidates.append(c)

    family = get_family(relation)
    if family:
        config = TraversalConfig(mode="family", family=family)
        filtered = [c for c in all_candidates if matches_config(c.predicate, config)]
    else:
        filtered = [c for c in all_candidates if c.predicate == relation]

    seen = {}
    for c in filtered:
        if c.word not in seen or c.confidence > seen[c.word].confidence:
            seen[c.word] = c
    return sorted(seen.values(), key=lambda x: -x.confidence)


def main():
    print("=" * 70)
    print("HELD-OUT TERM TRANSFER TEST")
    print("Can RAVANA reason about concepts it has NEVER been trained on?")
    print("=" * 70)

    # Build held-out set
    held_out_terms = list(HELD_OUT.keys())
    print(f"\nHeld-out terms ({len(held_out_terms)}): {', '.join(held_out_terms)}")

    # Build facts EXCLUDING held-out terms
    all_facts = build_facts_excluding(held_out_terms)
    print(f"Facts to inject: {len(all_facts)} (excluding held-out terms)")

    # Build vocabulary (includes held-out terms for embedding)
    all_triples = [(f.subject, f.relation, f.object) for f in all_facts]
    # Add held-out terms to vocabulary (but not their facts)
    for term in held_out_terms:
        all_triples.append((term, "is_a", "concept"))

    tok = WordTokenizer()
    for s, r, o in all_triples:
        tok.encode(f"{s} {r} {o}")

    print(f"Vocabulary: {tok.vocab_size} tokens (includes held-out terms)")

    # Create model
    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=5000, gate_concept_creation=False)
    model._tokenizer = tok

    # Inject facts (EXCLUDING held-out terms)
    print("\nInjecting facts (held-out terms excluded)...")
    t0 = time.time()
    injector = EpisodeInjector(model, tok)
    stats = injector.inject_facts(all_facts, epochs=8)
    print(f"  Injection: {stats['successful']}/{stats['total']} facts ({time.time()-t0:.1f}s)")
    print(f"  Graph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

    # Load MiniLM and inject embeddings
    print("\nLoading MiniLM...")
    t0 = time.time()
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"  Loaded in {time.time()-t0:.1f}s")

    inject_minilm_embeddings(model, tok, st_model)

    # ============================================================
    # TEST: Held-out term transfer
    # ============================================================
    print("\n" + "=" * 70)
    print("TRANSFER TEST: Novel concept -> NN bridge -> reasoning")
    print("=" * 70)

    results = []
    for term, info in HELD_OUT.items():
        print(f"\n--- {term} ---")
        print(f"  Description: {info['description']}")
        print(f"  Expected bridge: {', '.join(info['bridge_to'])}")

        # Check NN bridge
        neighbors = nn_bridge(model, tok, st_model, term, top_k=5)
        bridge_words = [w for w, _ in neighbors]
        bridge_hits = [w for w in info['bridge_to'] if w in bridge_words[:5]]

        print(f"  NN bridge: {', '.join([f'{w}({s:.2f})' for w, s in neighbors[:5]])}")
        print(f"  Bridge hits: {bridge_hits}")

        # Run composed queries for each expected relation
        term_results = {"term": term, "bridge_hits": bridge_hits, "queries": []}

        for relation, expected_objects in info['expected'].items():
            query_results = composed_query(model, tok, st_model, term, relation)
            result_words = [c.word for c in query_results[:5]]
            hits = [w for w in expected_objects if w in result_words]

            status = "✅" if hits else "❌"
            print(f"  {status} {term} {relation} ?")
            print(f"      Results:  {', '.join(result_words[:5])}")
            print(f"      Expected: {', '.join(expected_objects)}")
            print(f"      Hits:     {hits}")

            term_results["queries"].append({
                "relation": relation,
                "results": result_words,
                "expected": expected_objects,
                "hits": hits,
                "hit_rate": len(hits) / len(expected_objects) if expected_objects else 0,
            })

        results.append(term_results)

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_queries = sum(len(r["queries"]) for r in results)
    queries_with_hits = sum(1 for r in results for q in r["queries"] if q["hits"])
    total_expected = sum(len(q["expected"]) for r in results for q in r["queries"])
    total_hits = sum(len(q["hits"]) for r in results for q in r["queries"])

    bridge_correct = sum(1 for r in results if r["bridge_hits"])
    bridge_total = len(results)

    print(f"\n  Held-out terms: {len(results)}")
    print(f"  Bridge accuracy: {bridge_correct}/{bridge_total} ({bridge_correct/bridge_total:.0%})")
    print(f"  Query success: {queries_with_hits}/{total_queries} ({queries_with_hits/total_queries:.0%})")
    print(f"  Object hit rate: {total_hits}/{total_expected} ({total_hits/total_expected:.0%})")

    print(f"\n  Per-term breakdown:")
    for r in results:
        term_hits = sum(len(q["hits"]) for q in r["queries"])
        term_expected = sum(len(q["expected"]) for q in r["queries"])
        bridge = "✅" if r["bridge_hits"] else "❌"
        query = "✅" if term_hits > 0 else "❌"
        print(f"    {bridge}{query} {r['term']:20s} bridge={r['bridge_hits']} "
              f"objects={term_hits}/{term_expected}")

    # Final verdict
    print(f"\n  {'='*50}")
    if queries_with_hits / total_queries >= 0.7:
        print(f"  ✅ TRANSFER VALIDATED: {queries_with_hits/total_queries:.0%} query success")
        print(f"  RAVANA can reason about novel concepts via NN bridge.")
        print(f"  This is the real milestone — not enrichment.")
    elif queries_with_hits / total_queries >= 0.5:
        print(f"  ⚠️  PARTIAL TRANSFER: {queries_with_hits/total_queries:.0%} query success")
        print(f"  NN bridge works but some relations don't transfer.")
        print(f"  Investigate: which relation types fail?")
    else:
        print(f"  ❌ TRANSFER FAILED: {queries_with_hits/total_queries:.0%} query success")
        print(f"  NN bridge may be finding wrong nodes, or traversal is broken.")
        print(f"  Investigate: bridge accuracy + traversal logic.")


if __name__ == "__main__":
    main()
