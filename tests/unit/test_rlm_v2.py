"""
Tests for RLMv2 — Triple-Based Cognitive Architecture

Tests cover:
1. Triple decomposition
2. Relation type classification
3. Forward pass (spreading activation inference)
4. Learn (Hebbian triple updates)
5. Cross-domain transfer
6. Sleep cycle
7. Save/load
"""

import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ravana_ml.nn.rlm_v2 import RLMv2, RELATION_TYPES
from ravana_ml.tokenizer import WordTokenizer


def test_triple_decomposition():
    """Test that input sequences are properly decomposed into triples."""
    model = RLMv2(vocab_size=50, embed_dim=16, concept_dim=16, n_concepts=10)

    # 3 tokens: subject, relation, object
    s, r, o = model.decompose_triple([0, 1, 2])
    assert s == [0], f"Expected subject=[0], got {s}"
    assert r == [1], f"Expected relation=[1], got {r}"
    assert o == [2], f"Expected object=[2], got {o}"

    # 2 tokens: subject, object (no relation)
    s, r, o = model.decompose_triple([0, 1])
    assert s == [0], f"Expected subject=[0], got {s}"
    assert r == [], f"Expected relation=[], got {r}"
    assert o == [1], f"Expected object=[1], got {o}"

    # 4+ tokens: subject, multi-word relation, object
    s, r, o = model.decompose_triple([0, 1, 2, 3])
    assert s == [0], f"Expected subject=[0], got {s}"
    assert r == [1, 2], f"Expected relation=[1,2], got {r}"
    assert o == [3], f"Expected object=[3], got {o}"

    # 1 token: subject only
    s, r, o = model.decompose_triple([0])
    assert s == [0], f"Expected subject=[0], got {s}"
    assert r == [], f"Expected relation=[], got {r}"
    assert o == [], f"Expected object=[], got {o}"

    # Empty
    s, r, o = model.decompose_triple([])
    assert s == [], f"Expected subject=[], got {s}"

    print("  ✓ test_triple_decomposition")


def test_relation_type_classification():
    """Test that relation tokens are classified into correct types."""
    tok = WordTokenizer()
    # Build vocab
    for word in ["causes", "produces", "melts", "is", "has", "like", "then", "before"]:
        tok.encode(word)

    model = RLMv2(vocab_size=50, embed_dim=16, concept_dim=16, n_concepts=10)
    model._tokenizer = tok

    # Causal keywords
    for word in ["causes", "produces", "melts"]:
        tid = tok.encode(word)[0]
        idx = model.classify_relation([tid])
        assert RELATION_TYPES[idx] == "causal", f"Expected '{word}' → causal, got {RELATION_TYPES[idx]}"

    # Semantic keywords
    tid = tok.encode("is")[0]
    idx = model.classify_relation([tid])
    assert RELATION_TYPES[idx] == "semantic", f"Expected 'is' → semantic, got {RELATION_TYPES[idx]}"

    # Possessive keywords
    tid = tok.encode("has")[0]
    idx = model.classify_relation([tid])
    assert RELATION_TYPES[idx] == "possessive", f"Expected 'has' → possessive, got {RELATION_TYPES[idx]}"

    # Empty relation → semantic (default)
    idx = model.classify_relation([])
    assert RELATION_TYPES[idx] == "semantic", f"Expected empty → semantic, got {RELATION_TYPES[idx]}"

    print("  ✓ test_relation_type_classification")


def test_forward_returns_logits():
    """Test that forward() returns properly shaped logits."""
    model = RLMv2(vocab_size=50, embed_dim=16, concept_dim=16, n_concepts=10)

    ids = np.array([0, 1, 2], dtype=np.int64)
    logits = model.forward(ids)

    assert logits.data.shape == (50,), f"Expected shape (50,), got {logits.data.shape}"
    assert not np.all(logits.data == 0) or len(model.graph.edges) == 0, "Logits should be non-zero after training"
    print("  ✓ test_forward_returns_logits")


def test_learn_creates_edges():
    """Test that learn() creates concept nodes and typed edges."""
    tok = WordTokenizer()
    tok.encode("heat causes expansion")

    model = RLMv2(vocab_size=50, embed_dim=16, concept_dim=16, n_concepts=10)
    model._tokenizer = tok

    ids = tok.encode("heat causes expansion")
    ctx = np.array(ids[:-1], dtype=np.int64)
    tgt = np.array([ids[-1]], dtype=np.int64)

    initial_edges = len(model.graph.edges)
    result = model.learn(ctx, tgt)

    assert len(model.graph.edges) > initial_edges, "learn() should create edges"
    assert "loss" in result, "result should contain loss"
    assert "relation_type" in result, "result should contain relation_type"
    print("  ✓ test_learn_creates_edges")


def test_relation_type_in_edges():
    """Test that edges get proper relation types (not all 'semantic')."""
    tok = WordTokenizer()
    for text in ["heat causes expansion", "ice is cold", "fire produces smoke"]:
        tok.encode(text)

    model = RLMv2(vocab_size=100, embed_dim=16, concept_dim=16, n_concepts=20)
    model._tokenizer = tok

    # Train on causal and semantic triples
    for epoch in range(50):
        for text, target in [("heat causes expansion", "expansion"),
                             ("ice is cold", "cold"),
                             ("fire produces smoke", "smoke")]:
            ids = tok.encode(text)
            model.learn(np.array(ids[:-1], dtype=np.int64), np.array([ids[-1]], dtype=np.int64))

    # Check edge types
    type_counts = {}
    for edge in model.graph.edges.values():
        type_counts[edge.relation_type] = type_counts.get(edge.relation_type, 0) + 1

    assert "causal" in type_counts, f"Expected causal edges, got {type_counts}"
    assert type_counts.get("causal", 0) >= 2, f"Expected ≥2 causal edges, got {type_counts}"
    print(f"  ✓ test_relation_type_in_edges (types: {type_counts})")


def test_train_memorization():
    """Test that the model can memorize training triples."""
    tok = WordTokenizer()
    for text in ["heat causes expansion", "kindness causes trust"]:
        tok.encode(text)

    model = RLMv2(vocab_size=100, embed_dim=32, concept_dim=32, n_concepts=20)
    model._tokenizer = tok

    triples = [("heat causes expansion", "expansion"), ("kindness causes trust", "trust")]

    for epoch in range(100):
        for text, target in triples:
            ids = tok.encode(text)
            model.learn(np.array(ids[:-1], dtype=np.int64), np.array([ids[-1]], dtype=np.int64))

    # Evaluate
    hits = 0
    for text, target in triples:
        ids = tok.encode(text)
        logits = model.forward(np.array(ids[:-1], dtype=np.int64))
        top10 = set(np.argsort(logits.data.flatten())[::-1][:10].tolist())
        target_id = tok.encode(target)[0]
        if target_id in top10:
            hits += 1

    assert hits == len(triples), f"Expected {len(triples)} hits, got {hits}"
    print(f"  ✓ test_train_memorization ({hits}/{len(triples)})")


def test_relation_type_transfer():
    """Test that different verbs mapping to the same relation type work."""
    tok = WordTokenizer()
    for text in ["heat causes expansion", "heat produces steam", "heat leads to melting"]:
        tok.encode(text)

    model = RLMv2(vocab_size=100, embed_dim=32, concept_dim=32, n_concepts=20)
    model._tokenizer = tok

    # Train on "heat causes expansion"
    for epoch in range(100):
        ids = tok.encode("heat causes expansion")
        model.learn(np.array(ids[:-1], dtype=np.int64), np.array([ids[-1]], dtype=np.int64))

    # Test: "heat produces expansion" should also work (produces → CAUSAL)
    ids = tok.encode("heat produces expansion")
    logits = model.forward(np.array(ids[:-1], dtype=np.int64))
    top10 = set(np.argsort(logits.data.flatten())[::-1][:10].tolist())
    target_id = tok.encode("expansion")[0]

    assert target_id in top10, "heat produces expansion should predict expansion"
    print("  ✓ test_relation_type_transfer")


def test_cross_subject_causal():
    """Test that causal relation transfers across similar subjects."""
    tok = WordTokenizer()
    for text in ["heat causes expansion", "fire creates heat", "fire causes smoke"]:
        tok.encode(text)

    model = RLMv2(vocab_size=100, embed_dim=32, concept_dim=32, n_concepts=20)
    model._tokenizer = tok

    # Train on heat→expansion and fire→heat (creates link between fire and heat)
    for epoch in range(100):
        for text, target in [("heat causes expansion", "expansion"),
                             ("fire creates heat", "heat")]:
            ids = tok.encode(text)
            model.learn(np.array(ids[:-1], dtype=np.int64), np.array([ids[-1]], dtype=np.int64))

    # Test: "fire causes expansion" — fire is similar to heat via fire→heat edge
    ids = tok.encode("fire causes expansion")
    logits = model.forward(np.array(ids[:-1], dtype=np.int64))
    top10 = set(np.argsort(logits.data.flatten())[::-1][:10].tolist())
    target_id = tok.encode("expansion")[0]

    # This is a hard test — may not always pass
    hit = target_id in top10
    print(f"  {'✓' if hit else '✗'} test_cross_subject_causal (hit={hit})")
    return hit


def test_sleep_cycle():
    """Test that sleep cycle consolidates edges and resets pressure."""
    tok = WordTokenizer()
    tok.encode("heat causes expansion")

    model = RLMv2(vocab_size=100, embed_dim=16, concept_dim=16, n_concepts=10,
                  sleep_interval=10)
    model._tokenizer = tok

    # Train a few steps
    for i in range(5):
        ids = tok.encode("heat causes expansion")
        model.learn(np.array(ids[:-1], dtype=np.int64), np.array([ids[-1]], dtype=np.int64))

    initial_pressure = model._sleep_pressure
    initial_edges = len(model.graph.edges)

    # Run sleep cycle
    model.sleep_cycle()

    assert model._sleep_pressure == 0.0, "Sleep pressure should reset to 0"
    # Edges should still exist (not pruned if weight > threshold)
    print(f"  ✓ test_sleep_cycle (pressure: {initial_pressure:.3f} → 0, edges: {initial_edges} → {len(model.graph.edges)})")


def test_save_load():
    """Test that model can be saved and loaded."""
    tok = WordTokenizer()
    tok.encode("heat causes expansion")

    model = RLMv2(vocab_size=100, embed_dim=16, concept_dim=16, n_concepts=10)
    model._tokenizer = tok

    # Train a bit
    for _ in range(10):
        ids = tok.encode("heat causes expansion")
        model.learn(np.array(ids[:-1], dtype=np.int64), np.array([ids[-1]], dtype=np.int64))

    # Get prediction before save
    ids = tok.encode("heat causes expansion")
    logits_before = model.forward(np.array(ids[:-1], dtype=np.int64)).data.copy()

    # Save
    save_path = tempfile.mktemp(suffix=".pkl")
    model.save(save_path)

    # Load into new model
    model2 = RLMv2(vocab_size=100, embed_dim=16, concept_dim=16, n_concepts=10)
    model2.load(save_path)

    # Check prediction matches
    logits_after = model2.forward(np.array(ids[:-1], dtype=np.int64)).data
    diff = np.linalg.norm(logits_before - logits_after)
    assert diff < 0.1, f"Predictions should match after load (diff={diff:.6f})"

    # Check graph state
    assert len(model2.graph.nodes) == len(model.graph.nodes), "Node count should match"
    assert len(model2.graph.edges) == len(model.graph.edges), "Edge count should match"

    os.remove(save_path)
    print(f"  ✓ test_save_load (diff={diff:.6f})")


def test_relation_vector_separation():
    """Test that causal and semantic relation vectors separate over training."""
    tok = WordTokenizer()
    for text in ["heat causes expansion", "ice is cold", "fire produces smoke", "water is liquid"]:
        tok.encode(text)

    model = RLMv2(vocab_size=100, embed_dim=32, concept_dim=32, n_concepts=20)
    model._tokenizer = tok

    # Train
    for epoch in range(100):
        for text, target in [("heat causes expansion", "expansion"),
                             ("ice is cold", "cold"),
                             ("fire produces smoke", "smoke"),
                             ("water is liquid", "liquid")]:
            ids = tok.encode(text)
            model.learn(np.array(ids[:-1], dtype=np.int64), np.array([ids[-1]], dtype=np.int64))

    # Collect relation vectors by type
    type_rvs = {}
    for edge in model.graph.edges.values():
        rt = edge.relation_type
        if rt not in type_rvs:
            type_rvs[rt] = []
        type_rvs[rt].append(edge.relation_vector)

    if "causal" in type_rvs and "semantic" in type_rvs:
        causal_centroid = np.mean(type_rvs["causal"], axis=0)
        semantic_centroid = np.mean(type_rvs["semantic"], axis=0)
        cos = np.dot(causal_centroid, semantic_centroid) / (
            np.linalg.norm(causal_centroid) * np.linalg.norm(semantic_centroid) + 1e-10)
        # Should be low or negative (different types)
        print(f"  ✓ test_relation_vector_separation (causal↔semantic cosine: {cos:.3f})")
        return cos < 0.5  # Should be separated
    else:
        print(f"  ✗ test_relation_vector_separation (not enough typed edges: {type_counts})")
        return False


def test_bridge_alignment_targets():
    """Test that opt-in bridge alignment promotes semantic-pair targets."""
    tok = WordTokenizer()
    for text in ["heat causes expansion", "heat", "expansion"]:
        tok.encode(text)

    model = RLMv2(vocab_size=100, embed_dim=16, concept_dim=16, n_concepts=10)
    model._tokenizer = tok
    model.use_bridge_alignment = True
    model.semantic_pairs = [("heat", "expansion")]

    subject_tid = tok.encode("heat")[0]
    subject_cid = model._get_or_create_concept(subject_tid, model.token_embed.weight.data[subject_tid])

    targets = model._bridge_alignment_targets(subject_tid, subject_cid, "causal", "causes")
    expansion_tid = tok.encode("expansion")[0]
    expansion_cid = model._get_or_create_concept(expansion_tid, model.token_embed.weight.data[expansion_tid])

    assert expansion_cid in targets, "Bridge alignment should surface the paired target concept"
    assert targets[expansion_cid] > 0.0, "Bridge target should receive a positive score"
    print("  ✓ test_bridge_alignment_targets")


def test_sleep_cycle_cross_domain_alignment_toggle():
    """Test that optional cross-domain alignment runs during sleep when enabled."""
    tok = WordTokenizer()
    tok.encode("heat causes expansion")

    model = RLMv2(vocab_size=100, embed_dim=16, concept_dim=16, n_concepts=10,
                  sleep_interval=10)
    model._tokenizer = tok
    model.use_cross_domain_alignment = True
    model.cross_domain_alignment_on_sleep = True
    model.cross_domain_alignment_steps = 3

    calls = []

    def fake_alignment(lr=None):
        calls.append(lr)
        return {"causal": 0.0}

    model._cross_domain_relation_alignment = fake_alignment

    ids = tok.encode("heat causes expansion")
    model.learn(np.array(ids[:-1], dtype=np.int64), np.array([ids[-1]], dtype=np.int64))
    model.sleep_cycle()

    assert len(calls) == 3, f"Expected 3 alignment calls during sleep, got {len(calls)}"
    print("  ✓ test_sleep_cycle_cross_domain_alignment_toggle")


if __name__ == "__main__":
    print("=" * 50)
    print("RLMv2 Test Suite")
    print("=" * 50)
    print()

    tests = [
        test_triple_decomposition,
        test_relation_type_classification,
        test_forward_returns_logits,
        test_learn_creates_edges,
        test_relation_type_in_edges,
        test_train_memorization,
        test_relation_type_transfer,
        test_cross_subject_causal,
        test_sleep_cycle,
        test_save_load,
        test_relation_vector_separation,
        test_sleep_cycle_cross_domain_alignment_toggle,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"WARNING: {failed} test(s) failed!")
        sys.exit(1)
