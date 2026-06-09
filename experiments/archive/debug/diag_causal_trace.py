"""Diagnostic: trace activation flow for cross-domain causal queries."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer

# Build same dataset
from experiments.experiment_triple_benchmark_v6 import build_dataset

def _resolve_name(model, tok, cid):
    """Resolve concept_id to word name via binding map reverse lookup."""
    node = model.graph.get_node(cid)
    if node is None:
        return f"??(cid={cid})"
    # Scan binding_map for any token bound to this concept
    for tok_id in range(tok.vocab_size):
        bindings = model.binding_map.get_tokens(tok_id, min_confidence=0.0)
        for b in bindings:
            if b.concept_id == cid:
                return tok.decode([tok_id])
    return f"??(cid={cid})"

def run_diag():
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

    # Train (short run for diagnostic)
    for epoch in range(500):
        indices = np.random.permutation(len(train))
        for idx in indices:
            text, target_word = train[idx]
            ids = tok.encode(text)
            target_id = tok.encode(target_word)[0]
            ctx = np.array(ids[:-1], dtype=np.int64)
            tgt = np.array([target_id], dtype=np.int64)
            model.learn(ctx, tgt)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Cross-Domain Causal Activation Trace")
    print("=" * 70)

    # Test case: "anger causes expansion" → expansion
    query = "anger causes expansion"
    target_word = "expansion"
    ids = tok.encode(query)
    target_id = tok.encode(target_word)[0]
    ctx = np.array(ids[:-1], dtype=np.int64)

    print(f"\nQuery: '{query}' → '{target_word}'")
    print(f"Token IDs: {ids}, target_id: {target_id}")

    # Decode each token
    for i, tid in enumerate(ids):
        print(f"  Token {i}: id={tid} → '{tok.decode([tid])}'")

    # Find concepts
    subject_word = "anger"
    subject_tid = tok.encode(subject_word)[0]
    print(f"\nSubject: '{subject_word}' → tid={subject_tid}")

    # Check what concepts are bound to 'anger'
    bindings = model.binding_map.get_tokens(subject_tid, min_confidence=0.0)
    print(f"  anger bindings: {[(b.concept_id, b.confidence) for b in bindings]}")

    # Check what concept 'expansion' is bound to
    expansion_tid = tok.encode("expansion")[0]
    exp_bindings = model.binding_map.get_tokens(expansion_tid, min_confidence=0.0)
    print(f"  expansion bindings: {[(b.concept_id, b.confidence) for b in exp_bindings]}")

    # Check edges from anger's concept
    if bindings:
        anger_cid = bindings[0].concept_id
        print(f"\n  anger concept id: {anger_cid}")
        outgoing = model.graph.get_outgoing(anger_cid)
        print(f"  anger outgoing edges ({len(outgoing)}):")
        for tgt_id, edge in outgoing:
            tgt_node = model.graph.get_node(tgt_id)
            tgt_name = _resolve_name(model, tok, tgt_id)
            print(f"    → {tgt_name} (cid={tgt_id}): type={edge.relation_type}, weight={edge.weight:.3f}, conf={edge.confidence:.3f}")

        # Check if heat exists and its edges
        heat_tid = tok.encode("heat")[0]
        heat_bindings = model.binding_map.get_tokens(heat_tid, min_confidence=0.0)
        if heat_bindings:
            heat_cid = heat_bindings[0].concept_id
            print(f"\n  heat concept id: {heat_cid}")
            heat_outgoing = model.graph.get_outgoing(heat_cid)
            print(f"  heat outgoing edges ({len(heat_outgoing)}):")
            for tgt_id, edge in heat_outgoing:
                tgt_name = _resolve_name(model, tok, tgt_id)
                print(f"    → {tgt_name} (cid={tgt_id}): type={edge.relation_type}, weight={edge.weight:.3f}, conf={edge.confidence:.3f}")

    # Now run forward and trace activation
    print(f"\n{'='*70}")
    print("ACTIVATION TRACE")
    print(f"{'='*70}")

    logits = model.forward(ctx)
    flat = logits.data.flatten()

    top10 = np.argsort(flat)[::-1][:10]
    print(f"\nTop-10 predictions:")
    for rank, tid in enumerate(top10):
        word = tok.decode([int(tid)])
        marker = " ← TARGET" if int(tid) == target_id else ""
        print(f"  {rank+1}. '{word}' (tid={tid}): {flat[tid]:.4f}{marker}")

    # Check if expansion is in top-10
    if target_id in top10:
        print(f"\n✓ SUCCESS: expansion is in top-10")
    else:
        print(f"\n✗ FAIL: expansion NOT in top-10")
        # Find where expansion ranks
        all_sorted = np.argsort(flat)[::-1]
        exp_rank = np.where(all_sorted == target_id)[0]
        if len(exp_rank) > 0:
            print(f"  expansion rank: {exp_rank[0]+1}/{len(all_sorted)}")
            print(f"  expansion score: {flat[target_id]:.4f}")

if __name__ == "__main__":
    run_diag()
