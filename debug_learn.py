"""Debug: trace learn() calls to see why only 1 edge is created"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer

tok = WordTokenizer()
train = [
    ("heat causes expansion", "expansion"),
    ("fire produces smoke", "smoke"),
    ("fire creates heat", "heat"),
]
all_texts = set()
for t, _ in train: all_texts.add(t)
for t in sorted(all_texts): tok.encode(t)

model = RLMv2(vocab_size=tok.vocab_size+5, embed_dim=32, concept_dim=32, n_concepts=tok.vocab_size, sleep_interval=100)
model._tokenizer = tok

# Trace each learn call
for text, target_word in train:
    ids = tok.encode(text)
    target_id = tok.encode(target_word)[0]
    ctx = np.array(ids[:-1], dtype=np.int64)
    tgt = np.array([target_id], dtype=np.int64)
    
    # Manual decompose
    full_ids = ids[:-1] + [target_id]
    subject_ids, relation_ids, object_ids = model.decompose_triple(full_ids)
    
    print(f"\n--- \"{text}\" → \"{target_word}\" ---")
    print(f"  token_ids: {ids}")
    print(f"  full_ids: {full_ids}")
    print(f"  subject_ids: {subject_ids}")
    print(f"  relation_ids: {relation_ids}")
    print(f"  object_ids: {object_ids}")
    
    if subject_ids:
        subject_tid = subject_ids[0]
        subject_embed = model.token_embed.weight.data[subject_tid]
        subject_cid = model._get_or_create_concept(subject_tid, subject_embed)
        object_cid = model._get_or_create_concept(target_id, model.token_embed.weight.data[target_id])
        print(f"  subject_tid={subject_tid} → subject_cid={subject_cid}")
        print(f"  target_id={target_id} → object_cid={object_cid}")
        
        # Get relation type
        rel_type_idx, _ = model._classify_relation_learned(relation_ids)
        from ravana_ml.nn.rlm_v2 import RELATION_TYPES
        print(f"  relation_type: {RELATION_TYPES[rel_type_idx]}")
    
    result = model.learn(ctx, tgt)
    print(f"  edges after: {len(model.graph.edges)}")
    for (src, tgt_id), edge in model.graph.edges.items():
        print(f"    {src} → {tgt_id}  type={edge.relation_type}")
