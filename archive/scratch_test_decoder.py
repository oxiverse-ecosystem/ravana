import sys
import os

_proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))
sys.path.insert(0, _proj_root)

from scripts.ravana_chat import CognitiveChatEngine, CognitiveResponseContext

print("Loading engine...")
engine = CognitiveChatEngine(dim=64, baby_mode=True)
print("Engine loaded.")

# Let's test what the neural decoder generates
print("\n--- Testing Neural Decoder Directly ---")
topics = ["oxiverse", "ravana", "trust", "mind", "learning"]
for topic in topics:
    # Build a mock context
    ctx = CognitiveResponseContext(
        raw_input=f"what is {topic}",
        subject=topic,
        associated_concepts=[(topic, 1.0)],
        past_topics=[]
    )
    # Check if decoder is ready
    nd = engine.neural_decoder
    ce_ok = nd._avg_cross_entropy < 4.0 if nd._metric_examples > 10 else False
    t1_ok = nd._avg_top1_acc > 0.25 if nd._metric_examples > 10 else False
    trained_enough = engine._decoder_training_count >= 2000
    decoder_ready = ce_ok and t1_ok and trained_enough
    print(f"\nTopic: {topic}")
    print(f"Decoder ready check: ce_ok={ce_ok} ({nd._avg_cross_entropy:.4f}), t1_ok={t1_ok} ({nd._avg_top1_acc:.4f}), trained={engine._decoder_training_count} (ready={decoder_ready})")
    
    try:
        response_dec = engine._generate_with_decoder(ctx)
        print(f"Neural Decoder Output: {response_dec}")
    except Exception as e:
        print(f"Neural Decoder Error: {e}")
        
    try:
        response_syntax = engine._generate_with_decoder_and_syntax(ctx)
        print(f"Syntactic Pipeline Output: {response_syntax}")
    except Exception as e:
        print(f"Syntactic Pipeline Error: {e}")
