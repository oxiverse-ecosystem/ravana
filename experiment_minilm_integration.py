"""
MiniLM Integration Test
========================
Wires pretrained MiniLM embeddings into RLMv2's graph.
Tests: NN bridge, composed reasoning, transfer on novel concepts.
"""
import sys
import io
import time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sentence_transformers import SentenceTransformer
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.word_tokenizer import WordTokenizer
from ravana_ml.episode_injector import EpisodeInjector, load_pharmacology_kb, load_ecology_kb
from ravana_ml.relation_ontology import (
    Candidate, TraversalConfig, get_sub_family, get_family,
    get_confidence, matches_config
)


def inject_minilm_embeddings(model, tok, st_model):
    """Replace random embeddings with MiniLM embeddings."""
    dim = model.embed_dim  # 32
    st_dim = st_model.get_sentence_embedding_dimension()  # 384

    # Get all words in vocabulary
    words = list(tok.word_to_id.keys())
    if not words:
        return

    # Batch encode all words
    embeddings = st_model.encode(words, show_progress_bar=False)

    # Project to model dimension via random projection (deterministic per word)
    # This preserves semantic structure while fitting the model's embed_dim
    rng = np.random.RandomState(42)
    projection = rng.randn(st_dim, dim).astype(np.float32) / np.sqrt(dim)

    for i, word in enumerate(words):
        tid = tok.word_to_id[word]
        # Project 384-dim -> 32-dim
        projected = embeddings[i] @ projection
        # Normalize
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected /= norm
        model.token_embed.weight.data[tid] = projected

    print(f"  Injected MiniLM embeddings for {len(words)} tokens")


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


def nn_bridge(model, tok, st_model, novel_word, top_k=5):
    """Find graph nodes most similar to a novel word via MiniLM embeddings."""
    st_dim = st_model.get_sentence_embedding_dimension()
    dim = model.embed_dim

    # Get projection matrix (same as injection)
    rng = np.random.RandomState(42)
    projection = rng.randn(st_dim, dim).astype(np.float32) / np.sqrt(dim)

    # Embed novel word
    novel_embed = st_model.encode([novel_word])[0] @ projection
    norm = np.linalg.norm(novel_embed)
    if norm > 0:
        novel_embed /= norm

    # Compare to all graph node embeddings
    similarities = []
    for word, tid in tok.word_to_id.items():
        node_embed = model.token_embed.weight.data[tid]
        sim = float(np.dot(novel_embed, node_embed))
        similarities.append((word, sim))

    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]


def composed_query_with_bridge(model, tok, st_model, novel_word, relation, top_k=3):
    """Novel concept -> NN bridge -> graph traversal -> prediction.

    This is the full composed reasoning pipeline:
    1. Embed novel concept via MiniLM
    2. Find top-k similar graph nodes
    3. Traverse from those nodes
    4. Return predictions
    """
    # Step 1: NN bridge
    neighbors = nn_bridge(model, tok, st_model, novel_word, top_k=top_k)

    # Step 2: Traverse from each neighbor
    all_candidates = []
    for neighbor_word, sim in neighbors:
        if sim < 0.1:  # skip very low similarity
            continue
        candidates = traverse_broad(model, tok, neighbor_word, max_depth=2)
        for c in candidates:
            # Weight by bridge similarity
            c.confidence *= sim
            c.path = [(novel_word, f"via:{neighbor_word}", c.word)]
            all_candidates.append(c)

    # Step 3: Filter by relation family
    config = TraversalConfig(mode="family", family=get_family(relation) or "causal")
    filtered = [c for c in all_candidates if matches_config(c.predicate, config)]

    # Deduplicate
    seen = {}
    for c in filtered:
        if c.word not in seen or c.confidence > seen[c.word].confidence:
            seen[c.word] = c
    return sorted(seen.values(), key=lambda x: -x.confidence)


def main():
    print("=" * 70)
    print("MINILM INTEGRATION TEST")
    print("Pretrained embeddings -> graph -> composed reasoning")
    print("=" * 70)

    # Build vocabulary
    all_facts = load_pharmacology_kb() + load_ecology_kb()
    all_triples = [(f.subject, f.relation, f.object) for f in all_facts]

    tok = WordTokenizer()
    for s, r, o in all_triples:
        tok.encode(f"{s} {r} {o}")

    print(f"\nVocabulary: {tok.vocab_size} tokens")

    # Create model
    model = RLMv2(vocab_size=tok.vocab_size, embed_dim=32, concept_dim=32,
                  n_concepts=2000, gate_concept_creation=False)
    model._tokenizer = tok

    # Inject facts
    injector = EpisodeInjector(model, tok)
    injector.inject_facts(load_pharmacology_kb(), epochs=5)
    injector.inject_facts(load_ecology_kb(), epochs=5)
    print(f"Graph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

    # Load MiniLM and inject embeddings
    print("\nLoading MiniLM...")
    t0 = time.time()
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Loaded in {time.time()-t0:.1f}s")

    inject_minilm_embeddings(model, tok, st_model)

    # ============================================================
    # TEST 1: NN BRIDGE — Can MiniLM find the right graph nodes?
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 1: NN BRIDGE")
    print("Novel concepts -> find similar graph nodes")
    print("=" * 70)

    novel_concepts = ["tea", "heroin", "fox", "fertilizer", "energy_drink"]
    for concept in novel_concepts:
        neighbors = nn_bridge(model, tok, st_model, concept, top_k=5)
        neighbor_str = ", ".join([f"{w}({s:.2f})" for w, s in neighbors])
        print(f"  {concept:20s} -> {neighbor_str}")

    # ============================================================
    # TEST 2: COMPOSED REASONING — Novel concept -> graph -> prediction
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 2: COMPOSED REASONING")
    print("Novel concept -> NN bridge -> traversal -> prediction")
    print("=" * 70)

    queries = [
        ("tea", "causes"),
        ("heroin", "causes"),
        ("fox", "hunts"),
        ("energy_drink", "increases"),
        ("fertilizer", "increases"),
    ]

    for novel_word, relation in queries:
        results = composed_query_with_bridge(model, tok, st_model, novel_word, relation)
        if results:
            result_str = ", ".join([f"{c.word}({c.confidence:.2f})" for c in results[:5]])
            print(f"  {novel_word} {relation} ? -> {result_str}")
        else:
            print(f"  {novel_word} {relation} ? -> NO RESULTS")

    # ============================================================
    # TEST 3: SEMANTIC CLUSTERING — Do similar concepts cluster?
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 3: SEMANTIC CLUSTERING")
    print("Are pharmacology terms closer to each other than to ecology?")
    print("=" * 70)

    pharma_terms = ["aspirin", "ibuprofen", "caffeine", "nicotine", "acetaminophen"]
    eco_terms = ["wolf", "deer", "bear", "grass", "sunlight"]

    # Compute pairwise similarities within and across domains
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

    print(f"  Pharmacology intra-domain:  {np.mean(pharma_sims):.3f} (std={np.std(pharma_sims):.3f})")
    print(f"  Ecology intra-domain:       {np.mean(eco_sims):.3f} (std={np.std(eco_sims):.3f})")
    print(f"  Cross-domain:               {np.mean(cross_sims):.3f} (std={np.std(cross_sims):.3f})")
    print(f"  Intra > Cross gap:          {min(np.mean(pharma_sims), np.mean(eco_sims)) - np.mean(cross_sims):.3f}")


if __name__ == "__main__":
    main()
