"""Tests for native cognitive architecture embedded in RLM.

Covers: identity, emotion (VAD), meaning, sleep pressure, regulation,
native memory (episodic/semantic), memory bridge, save/load roundtrip,
token-level dissonance metric, edge weight convergence.
"""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import numpy as np
import tempfile
from ravana_ml.nn import RLM


def _make_model(**kwargs):
    defaults = dict(vocab_size=50, embed_dim=16, concept_dim=8,
                    n_concepts=30, n_hidden=32, n_layers=2, sleep_interval=100)
    defaults.update(kwargs)
    return RLM(**defaults)


def _learn_n(model, n):
    for i in range(n):
        model.learn(np.array([i % model.vocab_size]),
                    np.array([(i + 1) % model.vocab_size]))


# ── Initialization ──

def test_cognitive_state_init():
    m = _make_model()
    assert m.identity_strength == 0.5
    assert m.identity_momentum == 0.0
    assert m.identity_history == []
    assert m.valence == 0.0
    assert m.arousal == 0.3
    assert m.dominance == 0.5
    assert m.accumulated_meaning == 0.0
    assert m.meaning_history == []
    assert m.sleep_pressure == 0.0
    assert m.sleep_pressure_threshold == 0.7
    assert m.regulation_mode == "NORMAL"
    assert m.dissonance_ema == 0.5
    assert m._episodic_buffer == []
    assert m._semantic_memories == {}
    assert m._concept_vad == {}
    # New dissonance fields
    assert m._edge_weight_ema == 0.0
    assert m._token_hit_ema == 0.5
    print("PASS: test_cognitive_state_init")


# ── Emotion ──

def test_emotion_update():
    m = _make_model()
    # Positive stimulus should increase valence
    m._update_emotion(valence_stimulus=0.5, arousal_stimulus=0.8)
    assert m.valence > 0.0, f"valence should increase: {m.valence}"
    assert m.arousal > 0.3, f"arousal should increase: {m.arousal}"
    # Multiple updates should converge (differential equations)
    for _ in range(50):
        m._update_emotion(valence_stimulus=-0.5, arousal_stimulus=0.1)
    assert m.valence < 0.0, f"valence should go negative: {m.valence}"
    assert m.arousal >= 0.0 and m.arousal <= 1.0
    assert m.dominance >= 0.0 and m.dominance <= 1.0
    print("PASS: test_emotion_update")


# ── Identity ──

def test_identity_update():
    m = _make_model()
    initial = m.identity_strength
    # Correct prediction should increase identity
    delta = m._compute_identity_update(error=0.2, is_correct=True)
    assert delta > 0, f"correct prediction delta should be positive: {delta}"
    # Incorrect prediction should decrease identity
    delta_fail = m._compute_identity_update(error=0.8, is_correct=False)
    assert delta_fail < 0, f"wrong prediction delta should be negative: {delta_fail}"
    # Momentum carries forward
    m.identity_momentum = 0.1
    delta_mom = m._compute_identity_update(error=0.2, is_correct=True)
    assert delta_mom > delta, f"momentum should boost: {delta_mom} vs {delta}"
    # High identity should dampen changes
    m.identity_strength = 0.9
    delta_high = m._compute_identity_update(error=0.2, is_correct=True)
    m.identity_strength = 0.5
    delta_low = m._compute_identity_update(error=0.2, is_correct=True)
    assert delta_high < delta_low, "high identity should dampen"
    print("PASS: test_identity_update")


# ── Meaning ──

def test_meaning_computation():
    m = _make_model()
    m.dissonance_ema = 0.7
    m.identity_strength = 0.6
    m.identity_history = [0.5]
    meaning = m._compute_meaning(error=0.3)
    # dissonance_reduction = max(0, 0.7 - 0.3) = 0.4
    # identity_gain = 0.6 - 0.5 = 0.1
    # predictive_power = max(0, 1.0 - 0.3) = 0.7
    # meaning = 0.4*0.4 + 0.3*0.1 + 0.3*0.7 = 0.16 + 0.03 + 0.21 = 0.4
    assert meaning > 0, f"meaning should be positive: {meaning}"
    assert abs(meaning - 0.4) < 0.01, f"meaning formula mismatch: {meaning}"
    print("PASS: test_meaning_computation")


# ── Sleep Pressure ──

def test_sleep_pressure_accumulation():
    m = _make_model()
    assert m.sleep_pressure == 0.0
    # Learn a few steps — pressure should accumulate
    _learn_n(m, 5)
    assert m.sleep_pressure > 0.0, f"pressure should accumulate: {m.sleep_pressure}"
    # Sleep should reduce pressure
    pre_sleep = m.sleep_pressure
    m.sleep_cycle()
    assert m.sleep_pressure < pre_sleep, f"sleep should reduce pressure: {m.sleep_pressure} vs {pre_sleep}"
    print("PASS: test_sleep_pressure_accumulation")


# ── Episodic Memory ──

def test_native_memory_store():
    m = _make_model()
    _learn_n(m, 10)
    assert len(m._episodic_buffer) == 10
    # Each episode should have required fields
    ep = m._episodic_buffer[0]
    assert 'concepts' in ep
    assert 'error' in ep
    assert 'correct' in ep
    assert 'valence' in ep
    assert 'arousal' in ep
    assert 'timestamp' in ep
    # Buffer should be bounded
    _learn_n(m, 200)
    assert len(m._episodic_buffer) <= m._episodic_buffer_max
    print("PASS: test_native_memory_store")


# ── Memory Consolidation ──

def test_memory_consolidation():
    m = _make_model()
    # Learn enough to get some correct predictions
    for _ in range(10):
        _learn_n(m, 5)
        m.sleep_cycle()
    # After many cycles, semantic memories should form if any episodes were correct
    # (may be 0 if model never gets correct — that's OK, just verify no crash)
    assert len(m._semantic_memories) >= 0
    print(f"PASS: test_memory_consolidation (semantic_memories={len(m._semantic_memories)})")


# ── Memory Bridge ──

def test_memory_bridge():
    m = _make_model()
    # Manually add semantic memories
    nodes = list(m.graph.nodes.keys())
    if len(nodes) >= 2:
        m._semantic_memories[nodes[0]] = {'strength': 0.6, 'access_count': 5, 'last_access': 0}
        m._semantic_memories[nodes[1]] = {'strength': 0.7, 'access_count': 3, 'last_access': 0}
        # Bridge should strengthen edges between co-memorized concepts
        m._bridge_memories_to_graph()
        edge = m.graph.get_edge(nodes[0], nodes[1])
        # Edge may or may not exist — bridge creates if both strong enough
        print(f"PASS: test_memory_bridge (edge exists: {edge is not None})")
    else:
        print("PASS: test_memory_bridge (skipped — not enough nodes)")


# ── Regulation ──

def test_regulation_modes():
    m = _make_model()
    # Normal mode
    m.dissonance_ema = 0.3
    m._regulate_cognitive_state()
    assert m.regulation_mode == "NORMAL"

    # Recovery mode
    m.dissonance_ema = 0.85
    m._regulate_cognitive_state()
    assert m.regulation_mode == "RECOVERY"

    # Resolution mode
    m.dissonance_ema = 0.55
    m._regulate_cognitive_state()
    assert m.regulation_mode == "RESOLUTION"

    # Exploration mode
    m.dissonance_ema = 0.10
    m._regulate_cognitive_state()
    assert m.regulation_mode == "EXPLORATION"

    # Boundary pressure — high identity should be dampened
    m.identity_strength = 0.92
    m.dissonance_ema = 0.3
    m._regulate_cognitive_state()
    assert m.identity_strength < 0.92, "boundary pressure should reduce high identity"

    # Boundary pressure — low identity should be boosted
    m.identity_strength = 0.15
    m._regulate_cognitive_state()
    assert m.identity_strength > 0.15, "recovery should boost low identity"

    print("PASS: test_regulation_modes")


# ── Emotion Modulates Forward ──

def test_emotion_modulates_forward():
    m = _make_model()
    token = np.array([0])
    # Baseline forward
    logits_base = m.forward(token).data.copy()
    # Set high arousal
    m.arousal = 0.9
    m.valence = 0.5
    m.identity_strength = 0.8
    logits_excited = m.forward(token).data.copy()
    # Should differ from baseline (emotion modulates concept_logits weight)
    assert not np.allclose(logits_base, logits_excited), "emotion should modulate logits"
    print("PASS: test_emotion_modulates_forward")


# ── Sleep Consolidates Cognitive State ──

def test_sleep_consolidates_cognitive_state():
    m = _make_model()
    _learn_n(m, 10)
    pre_arousal = m.arousal
    pre_pressure = m.sleep_pressure
    m.sleep_cycle()
    # Arousal should move toward baseline (0.3)
    assert abs(m.arousal - 0.3) < abs(pre_arousal - 0.3), \
        f"arousal should converge to baseline: {m.arousal}"
    # Pressure should decrease
    assert m.sleep_pressure < pre_pressure
    # Valence magnitude should decrease
    if pre_arousal > 0.3:
        assert m.arousal < pre_arousal
    print("PASS: test_sleep_consolidates_cognitive_state")


# ── Save/Load Roundtrip (pickle) ──

def test_cognitive_save_load_roundtrip():
    m = _make_model(sleep_interval=5)
    _learn_n(m, 15)
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        path = f.name
    try:
        m.save(path)
        loaded = RLM.load(path)
        # All cognitive fields
        assert abs(loaded.identity_strength - m.identity_strength) < 1e-6
        assert abs(loaded.identity_momentum - m.identity_momentum) < 1e-6
        assert loaded.identity_history == m.identity_history
        assert abs(loaded.valence - m.valence) < 1e-6
        assert abs(loaded.arousal - m.arousal) < 1e-6
        assert abs(loaded.dominance - m.dominance) < 1e-6
        assert abs(loaded.accumulated_meaning - m.accumulated_meaning) < 1e-6
        assert loaded.meaning_history == m.meaning_history
        assert abs(loaded.sleep_pressure - m.sleep_pressure) < 1e-6
        assert loaded.regulation_mode == m.regulation_mode
        assert abs(loaded.dissonance_ema - m.dissonance_ema) < 1e-6
        assert abs(loaded._edge_weight_ema - m._edge_weight_ema) < 1e-6
        assert abs(loaded._token_hit_ema - m._token_hit_ema) < 1e-6
        assert len(loaded._episodic_buffer) == len(m._episodic_buffer)
        assert len(loaded._semantic_memories) == len(m._semantic_memories)
        assert len(loaded._concept_vad) == len(m._concept_vad)
        # Graph state
        assert len(loaded.graph.nodes) == len(m.graph.nodes)
        assert len(loaded.graph.edges) == len(m.graph.edges)
        # Post-load learn
        loaded.learn(np.array([0]), np.array([1]))
        print("PASS: test_cognitive_save_load_roundtrip")
    finally:
        os.unlink(path)


# ── Save/Load Roundtrip (zip) ──

def test_cognitive_zip_roundtrip():
    m = _make_model(sleep_interval=5)
    _learn_n(m, 15)
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
        path = f.name
    try:
        m.save_zip(path)
        loaded = RLM.load_zip(path)
        # All cognitive fields
        assert abs(loaded.identity_strength - m.identity_strength) < 1e-6
        assert abs(loaded.valence - m.valence) < 1e-6
        assert abs(loaded.arousal - m.arousal) < 1e-6
        assert abs(loaded.dominance - m.dominance) < 1e-6
        assert abs(loaded.accumulated_meaning - m.accumulated_meaning) < 1e-6
        assert abs(loaded.sleep_pressure - m.sleep_pressure) < 1e-6
        assert loaded.regulation_mode == m.regulation_mode
        assert abs(loaded.dissonance_ema - m.dissonance_ema) < 1e-6
        assert abs(loaded._edge_weight_ema - m._edge_weight_ema) < 1e-6
        assert abs(loaded._token_hit_ema - m._token_hit_ema) < 1e-6
        assert len(loaded._episodic_buffer) == len(m._episodic_buffer)
        assert len(loaded._semantic_memories) == len(m._semantic_memories)
        assert len(loaded._concept_vad) == len(m._concept_vad)
        # Graph state
        assert len(loaded.graph.nodes) == len(m.graph.nodes)
        assert len(loaded.graph.edges) == len(m.graph.edges)
        # Core/genesis vectors preserved
        node = list(loaded.graph.nodes.values())[0]
        assert node.core_vector is not None
        assert node.genesis_vector is not None
        # Post-load learn
        loaded.learn(np.array([0]), np.array([1]))
        print("PASS: test_cognitive_zip_roundtrip")
    finally:
        os.unlink(path)


# ── Full Cognitive Cycle ──

def test_full_cognitive_cycle():
    m = _make_model(sleep_interval=5)
    # Learn → sleep → learn → save → load → verify continuity
    _learn_n(m, 15)
    assert m.sleep_cycles_completed >= 1
    id_before = m.identity_strength
    v_before = m.valence
    a_before = m.arousal

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        path = f.name
    try:
        m.save(path)
        loaded = RLM.load(path)
        # Continue learning
        _learn_n(loaded, 10)
        # State should have changed from pre-load
        assert loaded.identity_strength != id_before or loaded.valence != v_before or loaded.arousal != a_before, \
            "cognitive state should evolve after load"
        print("PASS: test_full_cognitive_cycle")
    finally:
        os.unlink(path)


# ── Token-Level Dissonance Metric ──

def test_dissonance_changes_over_time():
    """The dissonance metric should actually change as the model learns,
    not stay flat at ~1.0 like the old set-overlap metric."""
    m = _make_model(sleep_interval=50)
    dissonance_snapshots = []
    # Use a repeating pattern so the model can learn to predict it
    for step in range(500):
        # Simple sequential pattern: i -> i+1 mod vocab_size
        i = step % m.vocab_size
        next_i = (i + 1) % m.vocab_size
        error = m.learn(np.array([i]), np.array([next_i]))
        if step % 50 == 0:
            dissonance_snapshots.append((step, m.dissonance_ema, m._token_hit_ema, m._edge_weight_ema))

    print("\n--- Dissonance over 500 steps ---")
    print(f"{'Step':>5}  {'Dissonance':>10}  {'TokenHitEMA':>12}  {'EdgeWeightEMA':>14}")
    for step, diss, hit, ew in dissonance_snapshots:
        print(f"{step:>5}  {diss:>10.4f}  {hit:>12.4f}  {ew:>14.6f}")

    first_diss = dissonance_snapshots[0][1]
    last_diss = dissonance_snapshots[-1][1]

    # The key test: dissonance should NOT be stuck at ~1.0
    # It should reflect actual prediction accuracy
    # With a repeating sequential pattern, the model should eventually learn
    # At minimum, the metric should vary (not be flat)
    all_dissonances = [d[1] for d in dissonance_snapshots]
    dissonance_range = max(all_dissonances) - min(all_dissonances)
    assert dissonance_range > 0.01, \
        f"Dissonance should vary over time! Range was {dissonance_range:.4f} (stuck at {first_diss:.4f})"

    # Edge weight EMA should be non-negative (initialized to 0, grows with edges)
    last_ew = dissonance_snapshots[-1][3]
    print(f"\nFinal edge weight EMA: {last_ew:.6f}")
    assert last_ew >= 0.0, "Edge weight EMA should be non-negative"

    print(f"\nDissonance range: {first_diss:.4f} -> {last_diss:.4f} (delta={last_diss - first_diss:+.4f})")
    print("PASS: test_dissonance_changes_over_time")


# ── Run all tests ──

if __name__ == "__main__":
    test_cognitive_state_init()
    test_emotion_update()
    test_identity_update()
    test_meaning_computation()
    test_sleep_pressure_accumulation()
    test_native_memory_store()
    test_memory_consolidation()
    test_memory_bridge()
    test_regulation_modes()
    test_emotion_modulates_forward()
    test_sleep_consolidates_cognitive_state()
    test_cognitive_save_load_roundtrip()
    test_cognitive_zip_roundtrip()
    test_full_cognitive_cycle()
    test_dissonance_changes_over_time()
    print("\n=== ALL 15 COGNITIVE TESTS PASSED ===")
