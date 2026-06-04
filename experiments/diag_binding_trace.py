"""Diagnostic: trace concept ID assignments during training."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer
from experiments.experiment_triple_benchmark_v6 import build_dataset

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

    # Key tokens to track
    track_words = ["heat", "expansion", "anger", "trust", "friendship", "flooding",
                   "conflict", "illness", "crashes", "running", "hot", "intense"]
    track_tids = {}
    for w in track_words:
        tids = tok.encode(w)
        if len(tids) == 1:
            track_tids[w] = tids[0]

    print(f"Vocab: {tok.vocab_size}")
    print(f"Tracked tokens: {track_tids}")
    print(f"Max concepts: {model._max_concepts}")
    print()

    # Track concept assignments at key epochs
    snapshots = {}

    for epoch in range(1500):
        indices = np.random.permutation(len(train))
        for idx in indices:
            text, target_word = train[idx]
            ids = tok.encode(text)
            target_id = tok.encode(target_word)[0]
            ctx = np.array(ids[:-1], dtype=np.int64)
            tgt = np.array([target_id], dtype=np.int64)
            model.learn(ctx, tgt)

        # Snapshot at key epochs
        if epoch in [0, 99, 499, 999, 1499]:
            snap = {}
            for word, tid in track_tids.items():
                bindings = model.binding_map.get_concepts(tid, min_confidence=0.0)
                snap[word] = [(b.concept_id, round(b.strength, 3)) for b in bindings]
            snapshots[epoch] = snap
            print(f"Epoch {epoch}:")
            for word in ["heat", "expansion", "anger", "trust", "running"]:
                if word in snap:
                    print(f"  {word} (tid={track_tids[word]}): {snap[word]}")
            print()

    # Final graph analysis
    print(f"{'='*70}")
    print("FINAL GRAPH ANALYSIS")
    print(f"{'='*70}")
    print(f"Nodes: {len(model.graph.nodes)}, Edges: {len(model.graph.edges)}")

    # Check heat→expansion edge directly
    heat_tid = track_tids["heat"]
    exp_tid = track_tids["expansion"]
    heat_b = model.binding_map.get_concepts(heat_tid, min_confidence=0.0)
    exp_b = model.binding_map.get_concepts(exp_tid, min_confidence=0.0)

    if heat_b and exp_b:
        heat_cid = heat_b[0].concept_id
        exp_cid = exp_b[0].concept_id
        print(f"\nheat_cid={heat_cid}, exp_cid={exp_cid}")

        edge = model.graph.get_edge(heat_cid, exp_cid)
        if edge:
            print(f"heat→expansion edge EXISTS: weight={edge.weight:.3f}, conf={edge.confidence:.3f}, type={edge.relation_type}")
        else:
            print(f"heat→expansion edge MISSING!")

        # Check ALL edges from heat
        print(f"\nAll outgoing from heat_cid={heat_cid}:")
        for tgt_id, edge in model.graph.get_outgoing(heat_cid):
            # Reverse-lookup: what tokens are bound to this target concept?
            tgt_tokens = model.binding_map.get_tokens(tgt_id, min_confidence=0.0)
            tgt_names = [tok.decode([t.token_id]) for t in tgt_tokens[:3]]
            print(f"  → cid={tgt_id} ({tgt_names}): type={edge.relation_type}, w={edge.weight:.3f}")

        # Check if expansion has ANY incoming edges
        print(f"\nAll incoming to exp_cid={exp_cid}:")
        for src_id, edge in model.graph.get_incoming(exp_cid):
            src_tokens = model.binding_map.get_tokens(src_id, min_confidence=0.0)
            src_names = [tok.decode([t.token_id]) for t in src_tokens[:3]]
            print(f"  ← cid={src_id} ({src_names}): type={edge.relation_type}, w={edge.weight:.3f}")

    # Check for many-to-one bindings (multiple tokens → same concept)
    print(f"\n{'='*70}")
    print("MANY-TO-ONE BINDING CHECK")
    print(f"{'='*70}")
    concept_to_tokens = {}
    for tid in range(tok.vocab_size):
        bindings = model.binding_map.get_concepts(tid, min_confidence=0.0)
        for b in bindings:
            if b.concept_id not in concept_to_tokens:
                concept_to_tokens[b.concept_id] = []
            concept_to_tokens[b.concept_id].append((tid, tok.decode([tid]), round(b.strength, 3)))

    # Show concepts with multiple tokens
    for cid, tokens in sorted(concept_to_tokens.items()):
        if len(tokens) > 1:
            print(f"  concept {cid}: {tokens}")

if __name__ == "__main__":
    run()
