import sys
import os
import traceback
import numpy as np

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
        # Build conditioning embs manually so we can print the raw output
        bos_idx = engine._decoder_word_to_idx.get("<bos>", 1)
        eos_idx = engine._decoder_word_to_idx.get("<eos>", 2)
        pad_idx = engine._decoder_word_to_idx.get("<pad>", 0)
        unk_idx = engine._decoder_word_to_idx.get("<unk>", 3)
        
        # Build conditioning
        concept_embs = []
        subj_lower = topic.lower()
        if subj_lower in engine._concept_keywords:
            nids = engine._concept_keywords[subj_lower]
            node = engine.graph.get_node(nids[0])
            if node and node.vector is not None:
                concept_embs.append(node.vector.copy())
        if len(concept_embs) < 1:
            if subj_lower in engine._decoder_word_to_embed:
                concept_embs.append(engine._decoder_word_to_embed[subj_lower].copy())
        
        if len(concept_embs) > 0:
            cond_embs = np.stack(concept_embs, axis=0).astype(np.float32)
            
            # Print raw logits top 5 at step 0
            h = np.zeros(nd.hidden_dim, dtype=np.float32)
            cond_proj = nd.condition_proj.forward_raw(cond_embs)
            word_emb = nd.word_embedding.embed_raw(bos_idx)
            projected_word = nd.condition_proj.forward_raw(word_emb[np.newaxis, :])[0]
            h = nd.gru(projected_word, h)
            attn_logits = nd.attention.forward_raw(cond_proj)
            combined = h * 0.5 + attn_logits * 0.5
            logits = nd.output_proj.forward_raw(combined[np.newaxis, :])[0]
            
            top_indices = np.argsort(logits)[-5:][::-1]
            print("  Top 5 logits at step 0:")
            for idx in top_indices:
                print(f"    {engine._decoder_idx_to_word.get(idx, '<unk>')}: {logits[idx]:.4f}")
                
            # Block pad, bos, unk
            for idx in [pad_idx, bos_idx, unk_idx]:
                if idx < len(logits):
                    logits[idx] = -1e9
                    
            # Let's run generate with special tokens blocked
            generated = nd.generate(
                conditioning_embs=cond_embs,
                max_steps=28,
                bos_idx=bos_idx,
                eos_idx=eos_idx,
                temperature=0.35,
                cerebellar_ngram=engine.cerebellar_ngram,
                idx_to_word=engine._decoder_idx_to_word,
                basal_ganglia=engine.basal_ganglia
            )
            print(f"  Raw Generated Indices: {generated}")
            raw_words = [engine._decoder_idx_to_word.get(idx, "") for idx in generated]
            print(f"  Raw Generated Words: {raw_words}")
            
            # Let's see if engine's _generate_with_decoder returns None
            engine_resp = engine._generate_with_decoder(ctx)
            print(f"  Engine Response: {engine_resp}")
            
            # Let's see the syntactic pipeline response
            try:
                syntax_resp = engine._generate_with_decoder_and_syntax(ctx)
                print(f"  Syntactic Pipeline Response: {syntax_resp}")
            except Exception as e:
                print(f"  Syntactic Pipeline Error: {e}")
            
    except Exception as e:
        print(f"Exception during decoder run:")
        traceback.print_exc()


