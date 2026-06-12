#!/usr/bin/env python3
"""Test ravana_chat.py interactively by sending prompts and capturing responses."""
import sys, os, tempfile, time

# Fix path setup - same as ravana_learn.py does
_proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))

# Try importing with scripts prefix first, fall back to direct import
try:
    from scripts.ravana_chat import CognitiveChatEngine
except ImportError:
    sys.path.insert(0, os.path.join(_proj_root, "scripts"))
    from ravana_chat import CognitiveChatEngine

# Use a temp dir so we don't interfere with the saved weights
tmpdir = tempfile.mkdtemp()
print(f"[Test] Using temp dir: {tmpdir}")

engine = CognitiveChatEngine(seed=42, data_dir=tmpdir, baby_mode=False)
engine._trace_enabled = False
engine._recall_mode = False
print(f"[Test] Loaded {len(engine.graph.nodes)} concepts, {len(engine.graph.edges)} edges")
print(f"[Test] Dopamine tone: {engine._dopamine_tone:.3f}")
print(f"[Test] Mean prediction error: {engine._mean_prediction_error:.3f}")
print()

# Test 1: Simple query
print("=" * 60)
print("USER: what is trust")
print("=" * 60)
resp1 = engine.process_turn("what is trust")
print(f"RAVANA: {resp1}")
print()

# Test 2: Knowledge query
print("=" * 60)
print("USER: tell me about knowledge")
print("=" * 60)
resp2 = engine.process_turn("tell me about knowledge")
print(f"RAVANA: {resp2}")
print()

# Test 3: Contrastive - good vs bad
print("=" * 60)
print("USER: what is the difference between good and bad")
print("=" * 60)
resp3 = engine.process_turn("what is the difference between good and bad")
print(f"RAVANA: {resp3}")
print()

# Test 4: Epistemic hedging
print("=" * 60)
print("USER: explain complex")
print("=" * 60)
conf = engine._get_concept_confidence("complex")
print(f"[Confidence in 'complex': {conf:.3f}]")
resp4 = engine.process_turn("explain complex")
print(f"RAVANA: {resp4}")
print()

# Test 5: Love and hate (contrastive)
print("=" * 60)
print("USER: tell me about love and hate")
print("=" * 60)
resp5 = engine.process_turn("tell me about love and hate")
print(f"RAVANA: {resp5}")
print()

# Test 6: Epistemic hedge sampling
print("=" * 60)
print("[TEST 6: Epistemic hedge sampling for 'complex']")
print("=" * 60)
hedges = []
for _ in range(10):
    h = engine._get_epistemic_hedge("complex")
    if h:
        hedges.append(h)
print(f"  Hedges (10 samples): {hedges[:6]}{'...' if len(hedges) > 6 else ''}")
print()

# Test 7: Dopamine modulation
print("=" * 60)
print("[TEST 7: Dopamine tone effect on tail probability]")
print("=" * 60)
for da in [0.0, 0.5, 1.0]:
    engine._dopamine_tone = da
    tail_prob = max(0.15, min(0.65, 0.35 + (da - 0.5) * 0.5))
    print(f"  DA={da:.1f} -> tail_prob={tail_prob:.3f}")
engine._dopamine_tone = 0.5
print()

# Test 8: Recall mode
print("=" * 60)
print("USER: remember when we talked about trust")
print("=" * 60)
engine._topic_list = ["trust", "knowledge"]
engine._recall_mode = True
resp8 = engine.process_turn("remember when we talked about trust")
print(f"RAVANA: {resp8}")
engine._recall_mode = False
print()

# Test 9: Freedom query
print("=" * 60)
print("USER: what is freedom")
print("=" * 60)
resp9 = engine.process_turn("what is freedom")
print(f"RAVANA: {resp9}")
print()

# Test 10: Follow-up
print("=" * 60)
print("USER: tell me more")
print("=" * 60)
resp10 = engine.process_turn("tell me more")
print(f"RAVANA: {resp10}")
print()

# Test 11: Contrastive connector detection
print("=" * 60)
print("[TEST 11: has_contrastive_connector]")
print("=" * 60)
for chain in ["good is like bad but different", "love leads to joy", "freedom but responsibility"]:
    result = engine._has_contrastive_connector(chain)
    print(f"  '{chain}' -> contrastive={result}")
print()

# Test 12: Cerebellar discourse
print("=" * 60)
print("[TEST 12: Cerebellar discourse format]")
print("=" * 60)
for subj in ["trust", "knowledge", "complex", "freedom"]:
    fmt = engine._get_cerebellar_discourse(subj, 0)
    ngram = engine._cerebellar_ngram.get(subj, {})
    depth = engine._cerebellar_depth.get(subj, 0.0)
    print(f"  {subj}: format={fmt}, ngram_entries={len(ngram)}, depth={depth:.2f}")
print()

print("[Test] All tests completed successfully!")
