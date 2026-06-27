import sys
import os
import time

_proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))
sys.path.insert(0, _proj_root)

# Remove the old scrambled weights so we start fresh
weights_path = os.path.join(_proj_root, "data", "ravana_weights.pkl")
if os.path.exists(weights_path):
    print(f"Removing old scrambled weights at {weights_path}...")
    os.remove(weights_path)

from scripts.ravana_chat import CognitiveChatEngine, CognitiveResponseContext

print("Initializing fresh engine...")
start = time.time()
engine = CognitiveChatEngine(dim=64, baby_mode=True)
print(f"Fresh engine initialized in {time.time()-start:.1f}s")

# Let's verify the initial state of the decoder
print(f"Decoder exists: {engine.neural_decoder is not None}")
print(f"Decoder training count: {engine._decoder_training_count}")

# Process a dummy first turn to trigger the deferred seed & synthetic training
print("\n=== Running first dummy turn to trigger training ===")
t0 = time.time()
res = engine.process_turn("hello")
print(f"Turn response: {res}")
print(f"Training took {time.time()-t0:.1f}s")
print(f"Decoder training count: {engine._decoder_training_count}")
if engine.neural_decoder is not None:
    print(f"NeuralDecoder total training examples: {engine.neural_decoder._total_training_examples}")
    print(f"Decoder average cross entropy: {engine.neural_decoder._avg_cross_entropy:.4f}")
    print(f"Decoder top1 accuracy: {engine.neural_decoder._avg_top1_acc:.4f}")

# Let's save the fresh, clean trained state
print("\nSaving clean weights...")
engine.save()
print("Saved.")

# Reload the engine to verify that the loaded state preserves the vocabulary and doesn't scramble indices!
print("\n=== Reloading engine from save ===")
engine_reload = CognitiveChatEngine(dim=64, baby_mode=True)
print("Reloaded.")

# Let's test a few queries to see if they produce natural responses now!
print("\n=== Testing responses after reload ===")
queries = ["what is ravana", "what is oxiverse", "hello", "tell me about trust"]
for q in queries:
    t0 = time.time()
    resp = engine_reload.process_turn(q)
    elapsed = time.time() - t0
    # Determine the route used
    route = engine_reload._last_strategy
    print(f"Q: {q}")
    print(f"A: {resp} (route={route}, took {elapsed:.1f}s)")
