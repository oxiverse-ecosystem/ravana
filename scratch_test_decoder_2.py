import sys
import os
import traceback

_proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))
sys.path.insert(0, _proj_root)

from scripts.ravana_chat import CognitiveChatEngine, CognitiveResponseContext, _is_word_salad

print("Loading engine...")
engine = CognitiveChatEngine(dim=64, baby_mode=True)
print("Engine loaded.")

nd = engine.neural_decoder
ce_ok = nd._avg_cross_entropy < 7.0 if nd._metric_examples > 10 else False
t1_ok = nd._avg_top1_acc > 0.12 if nd._metric_examples > 10 else False
trained_enough = engine._decoder_training_count >= 2000
decoder_ready = ce_ok and t1_ok and trained_enough

print(f"Decoder ready: {decoder_ready} (CE={nd._avg_cross_entropy:.4f}, Top1={nd._avg_top1_acc:.4f}, Trained={engine._decoder_training_count})")

topics = ["ravana", "oxiverse", "hello", "trust"]
for topic in topics:
    print(f"\n--- Testing Topic: {topic} ---")
    ctx = CognitiveResponseContext(
        raw_input=f"what is {topic}",
        subject=topic,
        associated_concepts=[(topic, 1.0)],
        past_topics=[]
    )
    # Set required fields
    engine._activation_boost = {}
    
    try:
        raw_resp = engine._generate_with_decoder(ctx)
        print(f"Raw Decoder Response: {raw_resp}")
        if raw_resp:
            words = raw_resp.split()
            print(f"  Length: {len(raw_resp)}")
            print(f"  Word count: {len(words)}")
            print(f"  Is word salad: {_is_word_salad(raw_resp)}")
            
            # Replicate the quality checks
            # 1. >60% function words check
            func_set = {"a","an","the","is","are","was","were","be","been",
                "being","have","has","had","do","does","did","will",
                "would","could","should","may","might","shall","can",
                "not","no","nor","so","if","then","than","too","very",
                "just","about","also","into","over","after","before",
                "between","through","during","because","while","which",
                "who","whom","what","when","where","why","how","all",
                "each","every","both","few","more", "most", "some", "any",
                "this", "that", "these", "those", "it", "its", "they", "them",
                "their", "we", "our", "you", "your", "he", "she", "him", "her",
                "his", "i", "me", "my", "myself", "am",
                "of", "to", "for", "with", "from", "at", "by", "as", "on"}
            func_count = sum(1 for w in words if w.lower() in func_set)
            func_ratio = func_count / len(words) if len(words) > 0 else 0.0
            print(f"  Function word ratio: {func_ratio:.2f} (gate is >0.70)")
            
            # 2. Template-pattern check
            template_verbs = {"connects", "connect", "relates", "relate", "links",
                               "link", "associated"}
            template_preps = {"with", "into"}
            text_lower = raw_resp.lower()
            template_rejected = False
            for tv in template_verbs:
                for tp in template_preps:
                    if import_re := __import__('re').search(rf'\b\w+\s+' + tv + r'\s+' + tp + r'\s+\w+', text_lower):
                        template_rejected = True
                        break
                if template_rejected:
                    break
            print(f"  Template rejected: {template_rejected}")
            
    except Exception as e:
        print(f"Exception during decoder run:")
        traceback.print_exc()
