"""Diagnostic: trace forward pass scoring for 'anger causes expansion'."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer
from experiments.experiment_triple_benchmark_v6 import build_dataset

# Monkey-patch forward to trace scoring
_original_forward = RLMv2.forward

def traced_forward(self, token_ids):
    """Forward with detailed scoring trace."""
    from ravana_ml.tensor import tensor as make_tensor

    if token_ids.ndim > 1:
        token_ids = token_ids.flatten()
    token_ids = token_ids.tolist()

    subject_ids, relation_ids, object_ids = self.decompose_triple(token_ids)
    if not subject_ids:
        return make_tensor(np.zeros(self.vocab_size, dtype=np.float32))

    subject_tid = subject_ids[0]
    subject_embed = self.token_embed.weight.data[subject_tid]
    subject_cid = self._get_or_create_concept(subject_tid, subject_embed)
    rel_type_idx, rel_type_embed = self._classify_relation_learned(relation_ids)
    rel_type_name = self.RELATION_TYPES[rel_type_idx] if hasattr(self, 'RELATION_TYPES') else ['causal', 'semantic', 'temporal', 'possessive', 'analogical', 'contextual'][rel_type_idx]

    # Decode tokens
    tok = self._tokenizer
    query_words = [tok.decode([t]) for t in token_ids]
    subject_word = tok.decode([subject_tid])
    print(f"\n  FORWARD: [{' '.join(query_words)}]")
    print(f"  Subject: '{subject_word}' cid={subject_cid}, rel_type='{rel_type_name}'")

    # Run the real forward
    result = _original_forward(self, np.array(token_ids, dtype=np.int64))

    # Now trace what happened
    flat = result.data.flatten()
    top10 = np.argsort(flat)[::-1][:10]

    print(f"  Top-10:")
    for rank, tid in enumerate(top10):
        word = tok.decode([int(tid)])
        print(f"    {rank+1}. '{word}' (tid={tid}): {flat[tid]:.4f}")

    return result

RLMv2.forward = traced_forward

def run():
    train, tests = build_dataset()
    tok = WordTokenizer()
    all_texts = set()
    for text, _ in train: all_texts.add(text)
    for cat in tests.values():
        for text, _ in cat: all_texts.add(text)
    for text in sorted(all_texts): tok.encode(text)

    model = RLMv2(
        vocab_size=tok.vocab_size + 5,
        embed_dim=64, concept_dim=64,
        n_concepts=tok.vocab_size,
        sleep_interval=300,
        gate_concept_creation=False,
    )
    model._tokenizer = tok

    print(f"Vocab: {tok.vocab_size}, Training: {len(train)} triples")

    for epoch in range(1500):
        indices = np.random.permutation(len(train))
        for idx in indices:
            text, target_word = train[idx]
            ids = tok.encode(text)
            target_id = tok.encode(target_word)[0]
            ctx = np.array(ids[:-1], dtype=np.int64)
            tgt = np.array([target_id], dtype=np.int64)
            model.learn(ctx, tgt)

    print(f"\nGraph: {len(model.graph.nodes)} concepts, {len(model.graph.edges)} edges")

    # Check heat→expansion edge
    heat_tid = tok.encode("heat")[0]
    exp_tid = tok.encode("expansion")[0]
    heat_b = model.binding_map.get_concepts(heat_tid, min_confidence=0.0)
    exp_b = model.binding_map.get_concepts(exp_tid, min_confidence=0.0)
    if heat_b and exp_b:
        edge = model.graph.get_edge(heat_b[0].concept_id, exp_b[0].concept_id)
        if edge:
            print(f"heat→expansion: w={edge.weight:.3f}, c={edge.confidence:.3f}")
        else:
            print(f"heat→expansion: MISSING")

    # Trace cross-domain causal queries
    print(f"\n{'='*70}")
    print("CROSS-DOMAIN CAUSAL TRACES")
    print(f"{'='*70}")
    for query, target_word in tests["cross_domain_causal"]:
        ids = tok.encode(query)
        target_id = tok.encode(target_word)[0]
        ctx = np.array(ids[:-1], dtype=np.int64)
        logits = model.forward(ctx)
        flat = logits.data.flatten()
        top10_set = set(np.argsort(flat)[::-1][:10].tolist())
        status = "✓" if target_id in top10_set else "✗"
        print(f"  {status} target='{target_word}' (tid={target_id})")

if __name__ == "__main__":
    run()
