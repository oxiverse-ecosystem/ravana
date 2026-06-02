"""
Held-Out Transfer — Fixed Multi-Bridge (Independent Traversals)
================================================================
Fix: each bridge candidate gets its OWN traversal (no shared visited set).
This prevents earlier candidates from blocking later ones.
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
# DENSE KB (same as before)
# ============================================================

DENSE_KB = {
    "aspirin": {"causes": ["pain_relief", "stomach_irritation", "blood_thinning"], "reduces": ["inflammation", "fever", "pain"], "contains": ["salicylic_acid"], "is_a": ["nsaid", "analgesic"]},
    "ibuprofen": {"causes": ["pain_relief", "stomach_irritation"], "reduces": ["inflammation", "fever", "pain"], "contains": ["propionic_acid"], "is_a": ["nsaid", "analgesic"]},
    "acetaminophen": {"causes": ["pain_relief", "liver_damage"], "reduces": ["fever", "pain"], "is_a": ["analgesic"]},
    "caffeine": {"causes": ["wakefulness", "alertness", "insomnia", "anxiety"], "increases": ["heart_rate", "blood_pressure", "alertness"], "decreases": ["fatigue", "drowsiness"], "contains": ["methylxanthine"], "is_a": ["stimulant"]},
    "nicotine": {"causes": ["addiction", "alertness", "nausea"], "increases": ["heart_rate", "blood_pressure"], "decreases": ["appetite", "anxiety"], "contains": ["pyridine"], "is_a": ["stimulant", "alkaloid"]},
    "addiction": {"causes": ["withdrawal", "dependence", "tolerance", "cravings"], "leads_to": ["relapse", "health_decline"], "is_a": ["condition"]},
    "withdrawal": {"causes": ["anxiety", "tremors", "nausea", "insomnia"], "is_a": ["symptom"]},
    "pain_relief": {"leads_to": ["comfort", "mobility"], "is_a": ["therapeutic_effect"]},
    "stomach_irritation": {"causes": ["nausea", "ulcers"], "is_a": ["side_effect"]},
    "liver_damage": {"causes": ["jaundice", "failure"], "leads_to": ["hospitalization"], "is_a": ["side_effect"]},
    "insomnia": {"causes": ["fatigue", "irritability"], "is_a": ["sleep_disorder"]},
    "anxiety": {"causes": ["insomnia", "panic"], "increases": ["heart_rate"], "is_a": ["mental_health_condition"]},
    "alertness": {"leads_to": ["productivity", "focus"], "reduces": ["errors"], "is_a": ["cognitive_state"]},
    "wakefulness": {"leads_to": ["productivity"], "decreases": ["sleep"], "is_a": ["physiological_state"]},
    "wolf": {"hunts": ["deer", "rabbit", "elk"], "is_a": ["predator", "mammal"], "lives_in": ["forest", "tundra"]},
    "deer": {"eats": ["grass", "leaves", "shrubs"], "is_a": ["herbivore", "mammal"], "hunted_by": ["wolf", "bear"], "lives_in": ["forest", "meadow"]},
    "bear": {"hunts": ["deer", "fish", "berries"], "is_a": ["predator", "mammal"], "lives_in": ["forest", "mountain"]},
    "predator": {"hunts": ["prey", "herbivore", "small_mammal"], "is_a": ["animal"], "lives_in": ["wilderness"]},
    "herbivore": {"eats": ["grass", "leaves", "shrubs", "plants"], "is_a": ["animal"], "hunted_by": ["predator"]},
    "prey": {"eats": ["grass", "seeds"], "hunted_by": ["predator"], "is_a": ["animal"]},
    "grass": {"is_a": ["plant"], "needs": ["sunlight", "water", "soil"], "eaten_by": ["deer", "rabbit", "herbivore"]},
    "sunlight": {"enables": ["photosynthesis"], "increases": ["plant_growth"], "is_a": ["energy_source"]},
    "photosynthesis": {"produces": ["oxygen", "glucose"], "requires": ["sunlight", "water", "carbon_dioxide"], "is_a": ["biological_process"]},
    "plant_growth": {"requires": ["sunlight", "water", "soil"], "leads_to": ["biomass"], "is_a": ["biological_process"]},
    "soil": {"contains": ["nutrients", "minerals", "organic_matter"], "enables": ["plant_growth"], "is_a": ["substrate"]},
    "energy_source": {"enables": ["growth", "metabolism", "movement"], "is_a": ["resource"]},
    "tea": {"contains": ["caffeine", "antioxidants", "tannins"], "causes": ["relaxation", "alertness"], "is_a": ["beverage"]},
    "coffee": {"contains": ["caffeine", "antioxidants"], "causes": ["wakefulness", "alertness", "jitteriness"], "is_a": ["beverage"]},
    "energy_drink": {"contains": ["caffeine", "sugar", "taurine"], "causes": ["alertness", "insomnia"], "increases": ["heart_rate", "energy"], "is_a": ["beverage"]},
    "heroin": {"causes": ["addiction", "euphoria", "respiratory_depression"], "contains": ["diacetylmorphine"], "is_a": ["opioid", "drug"]},
    "morphine": {"causes": ["pain_relief", "addiction", "drowsiness"], "reduces": ["pain"], "is_a": ["opioid", "analgesic"]},
    "fox": {"hunts": ["rabbit", "mouse", "bird"], "is_a": ["predator", "mammal"], "lives_in": ["forest", "field"]},
    "rabbit": {"eats": ["grass", "carrots", "clover"], "is_a": ["herbivore", "mammal"], "hunted_by": ["fox", "wolf"]},
    "mouse": {"eats": ["seeds", "grain"], "is_a": ["herbivore", "rodent"], "hunted_by": ["fox", "owl"]},
    "owl": {"hunts": ["mouse", "rabbit"], "is_a": ["predator", "bird"], "lives_in": ["forest"]},
    "fever": {"causes": ["discomfort", "dehydration"], "reduced_by": ["aspirin", "ibuprofen", "acetaminophen"], "is_a": ["symptom"]},
    "inflammation": {"causes": ["pain", "swelling"], "reduced_by": ["aspirin", "ibuprofen"], "is_a": ["immune_response"]},
    "pain": {"causes": ["discomfort"], "reduced_by": ["aspirin", "ibuprofen", "acetaminophen"], "is_a": ["symptom"]},
    "blood_thinning": {"prevents": ["clots"], "increases": ["bleeding_risk"], "is_a": ["pharmacological_effect"]},
    "stomach_acid": {"causes": ["irritation"], "increases": ["digestion"], "is_a": ["biological_chemical"]},
    "salicylic_acid": {"reduces": ["inflammation"], "is_a": ["chemical_compound"]},
    "methylxanthine": {"causes": ["alertness"], "increases": ["heart_rate"], "is_a": ["chemical_compound"]},
    "pyridine": {"is_a": ["chemical_compound"]},
    "diacetylmorphine": {"causes": ["euphoria"], "is_a": ["chemical_compound"]},
    "propionic_acid": {"is_a": ["chemical_compound"]},
    "taurine": {"increases": ["energy"], "is_a": ["amino_acid"]},
    "sugar": {"causes": ["energy_spike"], "increases": ["blood_sugar"], "is_a": ["carbohydrate"]},
    "antioxidants": {"prevent": ["cell_damage"], "is_a": ["compound"]},
    "tannins": {"causes": ["astringency"], "is_a": ["polyphenol"]},
    "euphoria": {"leads_to": ["addiction"], "is_a": ["emotional_state"]},
    "respiratory_depression": {"causes": ["death"], "is_a": ["medical_emergency"]},
}

HELD_OUT = {
    "matcha": {"bridge_to": ["tea", "coffee"], "expected": {"causes": ["wakefulness", "alertness"], "contains": ["caffeine"]}, "desc": "Japanese green tea"},
    "espresso": {"bridge_to": ["coffee", "tea"], "expected": {"causes": ["wakefulness", "alertness"], "contains": ["caffeine"]}, "desc": "Concentrated coffee"},
    "yerba_mate": {"bridge_to": ["tea", "coffee"], "expected": {"causes": ["alertness"], "contains": ["caffeine"]}, "desc": "South American tea"},
    "cocaine": {"bridge_to": ["heroin", "morphine"], "expected": {"causes": ["addiction", "euphoria"]}, "desc": "Stimulant drug"},
    "fentanyl": {"bridge_to": ["heroin", "morphine"], "expected": {"causes": ["addiction", "respiratory_depression"]}, "desc": "Synthetic opioid"},
    "amphetamines": {"bridge_to": ["nicotine", "caffeine"], "expected": {"causes": ["addiction", "alertness"], "increases": ["heart_rate"]}, "desc": "Stimulant"},
    "coyote": {"bridge_to": ["wolf", "fox"], "expected": {"hunts": ["rabbit", "deer"], "is_a": ["predator"]}, "desc": "Canid predator"},
    "lynx": {"bridge_to": ["wolf", "fox"], "expected": {"hunts": ["rabbit", "deer"], "is_a": ["predator"]}, "desc": "Feline predator"},
    "squirrel": {"bridge_to": ["rabbit", "mouse"], "expected": {"eats": ["grass", "seeds"], "is_a": ["herbivore"]}, "desc": "Rodent"},
    "fern": {"bridge_to": ["grass"], "expected": {"is_a": ["plant"], "needs": ["sunlight", "water"]}, "desc": "Plant"},
    "theobromine": {"bridge_to": ["caffeine", "methylxanthine"], "expected": {"increases": ["heart_rate"], "is_a": ["stimulant"]}, "desc": "Chocolate stimulant"},
    "massage": {"bridge_to": ["pain_relief"], "expected": {"leads_to": ["comfort"], "is_a": ["therapeutic_effect"]}, "desc": "Therapy"},
}


def build_facts_excluding(exclude_terms):
    exclude = set(exclude_terms)
    facts = []
    for subject, relations in DENSE_KB.items():
        if subject in exclude:
            continue
        for relation, objects in relations.items():
            for obj in objects:
                if obj in exclude:
                    continue
                facts.append(Fact(subject=subject, relation=relation, object=obj, confidence=0.9, source="dense_kb"))
    return facts


def main():
    print("=" * 70)
    print("HELD-OUT TRANSFER — FIXED MULTI-BRIDGE")
    print("Each bridge candidate gets independent traversal")
    print("=" * 70)

    held_out_terms = list(HELD_OUT.keys())
    all_facts = build_facts_excluding(held_out_terms)

    all_triples = [(f.subject, f.relation, f.object) for f in all_facts]
    for term in held_out_terms:
        all_triples.append((term, "is_a", "concept"))

    tok = WordTokenizer()
    for s, r, o in all_triples:
        tok.encode(f"{s} {r} {o}")

    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=5000, gate_concept_creation=False)
    model._tokenizer = tok

    injector = EpisodeInjector(model, tok)
    injector.inject_facts(all_facts, epochs=8)

    t0 = time.time()
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    st_dim = st_model.get_embedding_dimension()
    print(f"\nLoaded MiniLM ({st_dim}-dim) in {time.time()-t0:.1f}s")

    # Inject projected embeddings for model internals
    words = list(tok.word_to_id.keys())
    embeddings = st_model.encode(words, show_progress_bar=False)
    rng = np.random.RandomState(42)
    projection = rng.randn(st_dim, 32).astype(np.float32) / np.sqrt(32)
    for i, word in enumerate(words):
        tid = tok.word_to_id[word]
        projected = embeddings[i] @ projection
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected /= norm
        model.token_embed.weight.data[tid] = projected

    # Build full-dim embedding index for bridge
    word_embeds = {}
    for i, word in enumerate(words):
        word_embeds[word] = embeddings[i]

    def nn_bridge_full(novel_word, top_k=5):
        novel_embed = st_model.encode([novel_word])[0]
        similarities = []
        for word, embed in word_embeds.items():
            sim = float(np.dot(novel_embed, embed) /
                       (np.linalg.norm(novel_embed) * np.linalg.norm(embed)))
            similarities.append((word, sim))
        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]

    # FIXED: Independent traversal per bridge candidate
    def traverse_single(subject, max_depth=3):
        """Traverse from a SINGLE subject. Returns candidates."""
        subj_tid = tok.word_to_id.get(subject)
        if subj_tid is None:
            return []
        bindings = model.binding_map.get_concepts(subj_tid, min_confidence=0.1)
        if not bindings:
            return []

        frontier = {bindings[0].concept_id}
        visited = {bindings[0].concept_id}  # LOCAL visited set
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

        return candidates

    def composed_query_fixed(novel_word, relation, top_k=5, bridge_k=5):
        """FIXED: Each bridge candidate gets independent traversal."""
        neighbors = nn_bridge_full(novel_word, top_k=bridge_k)

        all_candidates = []
        for neighbor_word, sim in neighbors:
            if sim < 0.1:
                continue
            # INDEPENDENT traversal per candidate
            candidates = traverse_single(neighbor_word, max_depth=3)
            for c in candidates:
                c.confidence *= sim
                c.path = [(novel_word, f"via:{neighbor_word}", c.word)]
                all_candidates.append(c)

        # Filter by relation family
        family = get_family(relation)
        if family:
            config = TraversalConfig(mode="family", family=family)
            filtered = [c for c in all_candidates if matches_config(c.predicate, config)]
        else:
            filtered = [c for c in all_candidates if c.predicate == relation]

        # Deduplicate (keep highest confidence)
        seen = {}
        for c in filtered:
            if c.word not in seen or c.confidence > seen[c.word].confidence:
                seen[c.word] = c
        return sorted(seen.values(), key=lambda x: -x.confidence)

    # ============================================================
    # TEST: Held-out term transfer with fixed multi-bridge
    # ============================================================
    print("\n" + "=" * 70)
    print("TRANSFER TEST: Fixed multi-bridge (independent traversals)")
    print("=" * 70)

    results = []
    for term, info in HELD_OUT.items():
        neighbors = nn_bridge_full(term, top_k=5)
        bridge_words = [w for w, _ in neighbors]
        bridge_hits = [w for w in info['bridge_to'] if w in bridge_words[:5]]

        print(f"\n--- {term} ---")
        print(f"  Bridge: {', '.join([f'{w}({s:.2f})' for w, s in neighbors[:5]])}")

        term_results = {"term": term, "bridge_hits": bridge_hits, "queries": []}
        for relation, expected_objects in info['expected'].items():
            query_results = composed_query_fixed(term, relation, bridge_k=5)
            result_words = [c.word for c in query_results[:8]]
            hits = [w for w in expected_objects if w in result_words]

            status = "✅" if hits else "❌"
            print(f"  {status} {term} {relation} ?")
            print(f"      Results:  {', '.join(result_words[:8])}")
            print(f"      Expected: {', '.join(expected_objects)}")
            print(f"      Hits:     {hits}")

            term_results["queries"].append({
                "relation": relation,
                "results": result_words,
                "expected": expected_objects,
                "hits": hits,
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

    print(f"\n  Held-out terms: {len(results)}")
    print(f"  Bridge accuracy: {bridge_correct}/{len(results)} ({bridge_correct/len(results):.0%})")
    print(f"  Query success: {queries_with_hits}/{total_queries} ({queries_with_hits/total_queries:.0%})")
    print(f"  Object hit rate: {total_hits}/{total_expected} ({total_hits/total_expected:.0%})")

    print(f"\n  Per-term breakdown:")
    for r in results:
        term_hits = sum(len(q["hits"]) for q in r["queries"])
        term_expected = sum(len(q["expected"]) for q in r["queries"])
        bridge = "✅" if r["bridge_hits"] else "❌"
        query = "✅" if term_hits > 0 else "❌"
        print(f"    {bridge}{query} {r['term']:20s} objects={term_hits}/{term_expected}")

    # Compare with previous
    print(f"\n  {'='*50}")
    print(f"  COMPARISON WITH PREVIOUS RUNS:")
    print(f"  {'='*50}")
    print(f"  MiniLM + random projection:    42% bridge, 45% query")
    print(f"  MiniLM + full-dim bridge:      67% bridge, 59% query")
    print(f"  MiniLM + fixed multi-bridge:   {bridge_correct/len(results):.0%} bridge, {queries_with_hits/total_queries:.0%} query  ← NEW")


if __name__ == "__main__":
    main()
