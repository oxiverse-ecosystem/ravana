"""Diagnostic: trace ALL cross-domain causal test cases with full training."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer
from experiments.experiment_triple_benchmark_v6 import build_dataset

def _resolve_name(model, tok, cid):
    for tid in range(tok.vocab_size):
        for b in model.binding_map.get_tokens(tid, min_confidence=0.0):
            if b.concept_id == cid:
                return tok.decode([tid])
    return f"??({cid})"

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

    # Check heat→expansion specifically
    heat_tid = tok.encode("heat")[0]
    expansion_tid = tok.encode("expansion")[0]
    heat_bindings = model.binding_map.get_tokens(heat_tid, min_confidence=0.0)
    exp_bindings = model.binding_map.get_tokens(expansion_tid, min_confidence=0.0)

    print(f"\nheat tid={heat_tid}, bindings={[(b.concept_id, b.confidence) for b in heat_bindings]}")
    print(f"expansion tid={expansion_tid}, bindings={[(b.concept_id, b.confidence) for b in exp_bindings]}")

    if heat_bindings:
        heat_cid = heat_bindings[0].concept_id
        print(f"\nheat outgoing ({len(model.graph.get_outgoing(heat_cid))}):")
        for tgt_id, edge in model.graph.get_outgoing(heat_cid):
            name = _resolve_name(model, tok, tgt_id)
            print(f"  → {name}: type={edge.relation_type}, w={edge.weight:.3f}, c={edge.confidence:.3f}")

        print(f"\nheat incoming ({len(model.graph.get_incoming(heat_cid))}):")
        for src_id, edge in model.graph.get_incoming(heat_cid):
            name = _resolve_name(model, tok, src_id)
            print(f"  ← {name}: type={edge.relation_type}, w={edge.weight:.3f}, c={edge.confidence:.3f}")

    # Now trace ALL cross-domain causal queries
    print(f"\n{'='*70}")
    print("CROSS-DOMAIN CAUSAL ACTIVATION TRACES")
    print(f"{'='*70}")

    for query, target_word in tests["cross_domain_causal"]:
        ids = tok.encode(query)
        target_id = tok.encode(target_word)[0]
        ctx = np.array(ids[:-1], dtype=np.int64)
        logits = model.forward(ctx)
        flat = logits.data.flatten()

        top5 = np.argsort(flat)[::-1][:5]
        top5_words = [tok.decode([int(t)]) for t in top5]
        in_top10 = target_id in set(np.argsort(flat)[::-1][:10])

        # Find subject concept edges
        subject_word = query.split()[0]
        subject_tid = tok.encode(subject_word)[0]
        subject_b = model.binding_map.get_tokens(subject_tid, min_confidence=0.0)
        subject_cid = subject_b[0].concept_id if subject_b else -1

        # Count how many intermediate nodes have causal edges to target
        target_tid = tok.encode(target_word)[0]
        target_b = model.binding_map.get_tokens(target_tid, min_confidence=0.0)
        target_cid = target_b[0].concept_id if target_b else -1

        # Check 2-hop paths: subject→mid→target where mid→target is causal
        hop2_paths = []
        if subject_cid >= 0:
            for mid_cid, mid_edge in model.graph.get_outgoing(subject_cid):
                mid_node = model.graph.get_node(mid_cid)
                if mid_node is None:
                    continue
                for tgt_cid, tgt_edge in model.graph.get_outgoing(mid_cid):
                    if tgt_cid == target_cid and tgt_edge.relation_type == "causal":
                        mid_name = _resolve_name(model, tok, mid_cid)
                        hop2_paths.append(f"{subject_word}→{mid_name}→{target_word}")

        status = "✓" if in_top10 else "✗"
        print(f"\n  {status} '{query}' → '{target_word}'")
        print(f"     top5: {top5_words}")
        print(f"     subject_cid={subject_cid}, target_cid={target_cid}")
        print(f"     2-hop paths: {hop2_paths if hop2_paths else 'NONE'}")

if __name__ == "__main__":
    run()
