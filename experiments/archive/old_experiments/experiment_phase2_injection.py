"""
Phase 2 Injection Test
=======================
Tests synthetic episode injection with real-world-style knowledge bases.
Feeds pharmacology and ecology KBs, then runs directional reasoning.
"""
import sys
import io
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.word_tokenizer import WordTokenizer
from ravana_ml.episode_injector import (
    EpisodeInjector, Fact,
    load_pharmacology_kb, load_ecology_kb,
    PHARMACOLOGY_KB, ECOLOGY_KB
)
from ravana_ml.relation_ontology import (
    Candidate, TraversalConfig, get_sub_family, get_family,
    get_confidence, matches_config
)


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


def filter_candidates(candidates, config):
    return [c for c in candidates if matches_config(c.predicate, config)]


def show(candidates, label=""):
    if label:
        print(f"\n{label} ({len(candidates)} results):")
    for c in candidates:
        print(f"  depth {c.depth} conf {c.confidence:.2f} [{c.sub_family:20s}] {c.predicate:20s}: {c.word}")


def main():
    print("=" * 70)
    print("PHASE 2 INJECTION TEST")
    print("Real-world-style knowledge bases -> graph -> reasoning")
    print("=" * 70)

    # Build vocabulary from both KBs
    all_facts = load_pharmacology_kb() + load_ecology_kb()
    all_triples = [(f.subject, f.relation, f.object) for f in all_facts]

    tok = WordTokenizer()
    for s, r, o in all_triples:
        tok.encode(f"{s} {r} {o}")

    print(f"\nVocabulary: {tok.vocab_size} tokens")
    print(f"Facts to inject: {len(all_facts)}")

    # Create model
    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=2000, gate_concept_creation=False)
    model._tokenizer = tok

    # Inject via EpisodeInjector
    injector = EpisodeInjector(model, tok)

    print("\n--- Injecting Pharmacology KB ---")
    pharma_facts = load_pharmacology_kb()
    stats = injector.inject_facts(pharma_facts, epochs=5)
    print(f"  Injected: {stats['successful']}/{stats['total']}")
    print(f"  Epochs: {stats['epochs_trained']}")

    print("\n--- Injecting Ecology KB ---")
    eco_facts = load_ecology_kb()
    stats = injector.inject_facts(eco_facts, epochs=5)
    print(f"  Injected: {stats['successful']}/{stats['total']}")

    print(f"\n{injector.summary()}")

    # ============================================================
    # QUERY TESTS
    # ============================================================

    print("\n" + "=" * 70)
    print("QUERY TESTS")
    print("=" * 70)

    # Test 1: Pharmacology — directional
    print("\n--- aspirin reduces ? ---")
    all_c = traverse_broad(model, tok, "aspirin", max_depth=2)
    config_neg = TraversalConfig(mode="sub_family", sub_family="directional-negative")
    reduced = filter_candidates(all_c, config_neg)
    show(reduced, "aspirin reduces ? (directional-negative)")

    # Test 2: Pharmacology — broad effects
    print("\n--- caffeine effects ? ---")
    all_c = traverse_broad(model, tok, "caffeine", max_depth=2)
    config_causal = TraversalConfig(mode="super_family", super_family="causal_directional")
    effects = filter_candidates(all_c, config_causal)
    show(effects, "caffeine effects ? (causal+directional)")

    # Test 3: Cross-domain — does the graph connect pharmacology and ecology?
    print("\n--- Cross-domain query ---")
    print("Does 'stimulant' connect caffeine to nicotine?")
    all_c = traverse_broad(model, tok, "caffeine", max_depth=3)
    show(all_c, "caffeine ? (all, depth 3)")

    # Test 4: Ecology — predator-prey chains
    print("\n--- wolf hunts ? ---")
    all_c = traverse_broad(model, tok, "wolf", max_depth=2)
    show(all_c, "wolf ? (all)")

    # Test 5: Multi-hop — wolf -> deer -> eats -> grass
    print("\n--- Multi-hop: wolf -> deer -> eats -> ? ---")
    all_c = traverse_broad(model, tok, "wolf", max_depth=3)
    show(all_c, "wolf ? (depth 3)")

    # Test 6: Structured candidate metadata
    print("\n--- Full candidate metadata for caffeine ---")
    all_c = traverse_broad(model, tok, "caffeine", max_depth=2)
    print(f"\n{'Word':<18s} {'Predicate':<20s} {'Family':<15s} {'Sub-family':<20s} {'Conf':<6s} {'Depth'}")
    print("-" * 85)
    for c in all_c:
        print(f"{c.word:<18s} {c.predicate:<20s} {c.family:<15s} {c.sub_family:<20s} {c.confidence:<6.2f} {c.depth}")

    # Test 7: Similarity-based queries (NN)
    print("\n--- NN similarity tests ---")
    for query in ["aspirin", "caffeine", "wolf"]:
        tid = tok.word_to_id.get(query)
        if tid is None:
            continue
        embed = model.token_embed.weight.data[tid]
        sims = []
        for other_word in tok.word_to_id:
            if other_word == query:
                continue
            other_tid = tok.word_to_id[other_word]
            other_embed = model.token_embed.weight.data[other_tid]
            sim = float(np.dot(embed, other_embed) / (np.linalg.norm(embed) * np.linalg.norm(other_embed) + 1e-8))
            sims.append((other_word, sim))
        sims.sort(key=lambda x: -x[1])
        top5 = sims[:5]
        print(f"  {query:15s} similar to: {[(w, f'{s:.3f}') for w, s in top5]}")


if __name__ == "__main__":
    main()
