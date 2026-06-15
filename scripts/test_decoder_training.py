#!/usr/bin/env python3
"""Test whether the decoder trains on real article text in _learn_from_text."""

import sys
import os

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))
sys.path.insert(0, _proj_root)

try:
    from scripts.ravana_chat import CognitiveChatEngine
except ImportError:
    from ravana_chat import CognitiveChatEngine

import time

print("Loading engine...")
start = time.time()
engine = CognitiveChatEngine(dim=64, baby_mode=True)
print(f"Engine loaded in {time.time()-start:.1f}s")

print("\n=== INITIAL STATE ===")
print(f"Decoder training count (total): {engine._decoder_training_count}")
print(f"Decoder training count (web):   {engine._decoder_web_training_count}")
print(f"Decoder vocab size: {len(engine._decoder_word_to_idx)}")
print(f"Decoder exists: {engine.neural_decoder is not None}")
print(f"Decoder vocab built: {engine._decoder_vocab_built}")
if engine.neural_decoder is not None:
    print(f"NeuralDecoder total training examples: {engine.neural_decoder._total_training_examples}")

print("\n=== TEST 1: _learn_from_text with real article text ===")
sample_text = (
    "The truth is often more complex than people realize. "
    "Scientists study truth through careful observation and experimentation. "
    "There are many different theories about what truth means in philosophy. "
    "The search for truth drives human knowledge forward. "
    "Understanding truth requires both logic and evidence. "
    "Truth can be found through honest inquiry and open minded debate."
)

old_total = engine._decoder_training_count
old_web = engine._decoder_web_training_count
old_decoder_examples = engine.neural_decoder._total_training_examples if engine.neural_decoder else 0

new_concepts = engine._learn_from_text(sample_text, "truth", source_url="test")

delta_total = engine._decoder_training_count - old_total
delta_web = engine._decoder_web_training_count - old_web
delta_decoder = engine.neural_decoder._total_training_examples - old_decoder_examples if engine.neural_decoder else 0

print(f"New concepts added: {new_concepts}")
print(f"Decoder total count delta: {delta_total} ({old_total} -> {engine._decoder_training_count})")
print(f"Decoder web count delta:   {delta_web}")
print(f"NeuralDecoder internal examples delta: {delta_decoder}")

if delta_decoder > 0:
    print("✅ DECODER IS TRAINING ON REAL TEXT!")
else:
    print("❌ Decoder did NOT train on real text")

print("\n=== TEST 2: Call _learn_from_text again with different text ===")
sample_text2 = (
    "Freedom is the power to act and speak without hindrance. "
    "People value freedom as a fundamental human right. "
    "The concept of freedom has evolved throughout history. "
    "Philosophers debate the limits of freedom in society. "
    "Freedom requires responsibility and respect for others. "
    "True freedom means being able to make choices about ones own life."
)

old_total = engine._decoder_training_count
new_concepts2 = engine._learn_from_text(sample_text2, "freedom", source_url="test")
delta_total2 = engine._decoder_training_count - old_total
print(f"New concepts added: {new_concepts2}")
print(f"Decoder total count delta: {delta_total2} ({old_total} -> {engine._decoder_training_count})")

if delta_total2 > 0:
    print("✅ DECODER TRAINING PERSISTS ON SUBSEQUENT CALLS!")
else:
    print("❌ Decoder did NOT train on second call")

print("\n=== DONE ===")
engine.save()
print("State saved.")
