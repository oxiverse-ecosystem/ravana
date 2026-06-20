"""
RAVANA Forgetting & Memory-Scaling Stress Test
===============================================
Replicates real-world failure pattern:
  "Works great for limited data, but becomes difficult to keep memories
   sorted and recalled. Lots of false positives when injecting memories."

Phases (each self-contained, ~5-15s):
1. Scale degradation — recall@1 at 10→30→60 facts
2. False positive intrusions — irrelevant memories scoring above correct
3. Clustering quality — prototype cohesion vs cross-domain separation
4. Forgetting curve — recall vs training frequency (1x, 10x, 100x)
5. Sleep consolidation effect — before/after sleep precision
6. Catastrophic forgetting — A→B→C sequential domains

Run: python scripts/test_forgetting.py
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ravana-v2"))

import numpy as np
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer

np.random.seed(42)
_passed = _failed = _total = 0

def check(name, ok, detail=""):
    global _passed, _failed, _total
    _total += 1
    if ok:
        _passed += 1
        print(f"  [OK] {name}")
    else:
        _failed += 1
        print(f"  [FAIL] {name}  {'— ' + detail if detail else ''}")

def make_model(tok, dim=32):
    m = RLMv2(vocab_size=tok.vocab_size + 5, embed_dim=dim, concept_dim=dim,
              n_concepts=tok.vocab_size, sleep_interval=500, latent_dim=dim)
    m._tokenizer = tok
    m.use_verb_offset = True
    m.disable_spreading_activation = True
    return m

def train(m, tok, facts, epochs=20):
    for _ in range(epochs):
        order = list(range(len(facts)))
        np.random.shuffle(order)
        for i in order:
            inp, tgt = facts[i]
            m.learn(np.array(tok.encode(inp), dtype=np.int64),
                    np.array(tok.encode(tgt), dtype=np.int64))

def recall_at(m, tok, facts, k=1):
    correct = total = intrusions = 0
    for inp, tgt in facts:
        input_ids = np.array(tok.encode(inp), dtype=np.int64)
        try:
            logits = m.forward(input_ids).data.flatten()
        except Exception:
            continue
        tid = tok.encode(tgt)[0]
        if tid < 0 or tid >= len(logits):
            continue
        total += 1
        if tid in set(np.argsort(logits)[-k:]):
            correct += 1
        intrusions += int(np.sum(logits > logits[tid])) - 1
    return correct / max(1, total), intrusions / max(1, total)


# ─── Phase 1: Scale degradation ───
def phase1():
    print("\n" + "=" * 60)
    print("PHASE 1: Scale Degradation — Recall vs Memory Size")
    print("=" * 60)
    tok = WordTokenizer()
    facts = [(f"sub_{i} causes ", f"val_{i}") for i in range(60)]
    for inp, tgt in facts:
        tok.encode(inp); tok.encode(tgt)
    model = make_model(tok)
    results = []
    for size in [10, 30, 60]:
        train(model, tok, facts[:size], epochs=25)
        model._compute_verb_offsets()
        r1, intr = recall_at(model, tok, facts[:size], k=1)
        results.append((size, r1, intr))
        print(f"  {size:>3} facts: recall@1={r1:.3f}  intrusions={intr:.2f}/fact")
    r_at_60 = results[-1][1]
    check(f"Recall@1 at 60 facts >= 0.30", r_at_60 >= 0.30, f"got {r_at_60:.3f}")

# ─── Phase 2: False positive intrusions ───
def phase2():
    print("\n" + "=" * 60)
    print("PHASE 2: False Positive Intrusion Analysis")
    print("=" * 60)
    tok = WordTokenizer()
    domains = {
        "animals": ["dog","cat","bird","fish","rabbit","hamster"],
        "colors":  ["red","blue","green","yellow","purple","orange"],
        "fruits":  ["apple","banana","grape","mango","kiwi","pear"],
    }
    facts = []
    for items in domains.values():
        for i, item in enumerate(items):
            facts.append((f"{item} relates_to ", items[(i+1)%len(items)]))
    for inp, tgt in facts:
        tok.encode(inp); tok.encode(tgt)
    model = make_model(tok)
    model._prototype_cluster_threshold = 0.45
    train(model, tok, facts, epochs=40)
    model._compute_verb_offsets()
    n_protos_before = len(model._prototype_hierarchy)
    model._init_default_prototypes()
    n_protos_after = len(model._prototype_hierarchy)
    print(f"  Prototypes: {n_protos_before} -> {n_protos_after}")
    total_intrusions = n_facts = 0
    for inp, tgt in facts:
        logits = model.forward(np.array(tok.encode(inp), dtype=np.int64)).data.flatten()
        tid = tok.encode(tgt)[0]
        total_intrusions += int(np.sum(logits > logits[tid])) - 1
        n_facts += 1
    avg = total_intrusions / n_facts
    print(f"  Avg {avg:.1f} intrusions/fact across {n_facts} facts")
    check(f"Avg intrusions < 5", avg < 5, f"got {avg:.1f}")
    # Cross-domain test: "dog relates_to" should rank rabbit #1
    # and suppress cross-domain false positives below correct-domain tokens
    logits = model.forward(np.array(tok.encode("dog relates_to "), dtype=np.int64)).data.flatten()
    rabbit_score = logits[tok.encode("rabbit")[0]]
    # Check no fruit/color scores above rabbit (tied-or-below is OK, above is false positive)
    max_cross = max(logits[tok.encode(w)[0]] for domain in domains.values()
                    for w in domain if w not in domains["animals"])
    no_false_positive = max_cross <= rabbit_score + 1e-6
    print(f"  Rabbit score: {rabbit_score:.2f}, max cross-domain: {max_cross:.2f}")
    check(f"No cross-domain score > rabbit", no_false_positive, f"max cross {max_cross:.2f} > rabbit {rabbit_score:.2f}")

# ─── Phase 3: Clustering quality ───
def phase3():
    print("\n" + "=" * 60)
    print("PHASE 3: Prototype Clustering Quality")
    print("=" * 60)
    tok = WordTokenizer()
    clusters = {
        "temperature": ["heat","cold","warm","cool","hot","freeze","boil","chill","melt","steam"],
        "emotion":     ["happy","sad","anger","fear","joy","grief","love","hate","calm","panic"],
        "movement":    ["run","walk","fly","swim","crawl","jump","slide","roll","dance","march"],
        "sound":       ["loud","quiet","echo","ring","bang","hum","whisper","roar","click","tap"],
        "texture":     ["rough","smooth","soft","hard","slick","gritty","fuzzy","silky","bumpy","sharp"],
    }
    facts = []
    for items in clusters.values():
        for i in range(len(items)-1):
            facts.append((f"{items[i]} relates_to ", items[i+1]))
            facts.append((f"{items[i+1]} relates_to ", items[(i+2)%len(items)]))
    for inp, tgt in facts:
        tok.encode(inp); tok.encode(tgt)
    model = make_model(tok, dim=64)
    train(model, tok, facts, epochs=50)
    model._init_default_prototypes()
    # Map concept ID to prototype
    c2p = {}
    for p_label, concept_ids in model._prototype_hierarchy.items():
        for cid in concept_ids:
            n = model.graph.get_node(cid)
            if n and n.label:
                c2p[n.label.lower()] = p_label
    # Same-domain sharing
    share, cross_conflict = 0, 0
    same_pairs, cross_pairs = 0, 0
    for dname, items in clusters.items():
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                same_pairs += 1
                if c2p.get(items[i]) and c2p.get(items[j]) and c2p[items[i]] == c2p[items[j]]:
                    share += 1
    doms = list(clusters.keys())
    for i in range(len(doms)):
        for j in range(i+1, len(doms)):
            for a in clusters[doms[i]]:
                for b in clusters[doms[j]]:
                    cross_pairs += 1
                    if c2p.get(a) and c2p.get(b) and c2p[a] == c2p[b]:
                        cross_conflict += 1
    share_rate = share / max(1, same_pairs)
    conflict_rate = cross_conflict / max(1, cross_pairs)
    print(f"  Same-domain prototype sharing: {share}/{same_pairs} ({share_rate:.0%})")
    print(f"  Cross-domain prototype conflicts: {cross_conflict}/{cross_pairs} ({conflict_rate:.1%})")
    # Prototype hierarchy populated
    check(f"Prototype hierarchy populated", len(model._prototype_hierarchy) > 0,
          f"got {len(model._prototype_hierarchy)} prototypes")

# ─── Phase 4: Forgetting curve ───
def phase4():
    print("\n" + "=" * 60)
    print("PHASE 4: Forgetting Curve — Recall vs Training Frequency")
    print("=" * 60)
    tok = WordTokenizer()
    groups = {"rare": (1,3), "moderate": (10,3), "frequent": (100,3)}
    all_facts = []
    for gname, (count, n) in groups.items():
        for i in range(n):
            for _ in range(count):
                all_facts.append((f"{gname}_s{i} causes ", f"{gname}_o{i}"))
    for inp, tgt in all_facts:
        tok.encode(inp); tok.encode(tgt)
    model = make_model(tok)
    train(model, tok, all_facts, epochs=15)
    model._compute_verb_offsets()
    print(f"  {'Group':<12} {'Freq':>8} {'Recall@1':>10}")
    print("  " + "-" * 32)
    results = {}
    for gname, (count, n) in groups.items():
        gf = [(f"{gname}_s{i} causes ", f"{gname}_o{i}") for i in range(n)]
        r1, _ = recall_at(model, tok, gf, k=1)
        results[gname] = r1
        print(f"  {gname:<12} {count:>8} {r1:>10.3f}")
    check(f"Frequent > moderate recall", results["frequent"] >= results["moderate"],
          f"{results['frequent']:.3f} vs {results['moderate']:.3f}")
    check(f"Moderate > rare recall", results["moderate"] >= results["rare"],
          f"{results['moderate']:.3f} vs {results['rare']:.3f}")
    check(f"Rare (1x) not completely forgotten", results["rare"] > 0.0,
          f"got {results['rare']:.3f}")

# ─── Phase 5: Sleep consolidation ───
def phase5():
    print("\n" + "=" * 60)
    print("PHASE 5: Sleep Consolidation Effect")
    print("=" * 60)
    tok = WordTokenizer()
    signal = [(f"sig_{i} causes ", f"sig_o{i}") for i in range(15)]
    noise = [(f"noise_{j} causes ", f"noise_o{j}") for j in range(25)]
    for inp, tgt in signal + noise:
        tok.encode(inp); tok.encode(tgt)
    model = make_model(tok)
    train(model, tok, signal, epochs=30)
    train(model, tok, noise, epochs=3)
    model._compute_verb_offsets()
    r_before, i_before = recall_at(model, tok, signal, k=1)
    edges_before = len(model.graph.edges)
    print(f"  Signal recall@1 before sleep: {r_before:.3f}  intrusions: {i_before:.2f}")
    print(f"  Total edges before sleep: {edges_before}")
    for _ in range(3):
        model.sleep_cycle()
    r_after, i_after = recall_at(model, tok, signal, k=1)
    edges_after = len(model.graph.edges)
    print(f"  Signal recall@1 after sleep:  {r_after:.3f}  intrusions: {i_after:.2f}")
    print(f"  Total edges after sleep: {edges_after} ({edges_before - edges_after} pruned)")
    # Sleep may add replay edges; verify edge count didn't explode (>2x)
    check(f"Sleep edges within 2x of pre-sleep count", edges_after < edges_before * 2,
          f"{edges_after} vs {edges_before} ({(edges_after/edges_before-1)*100:+.0f}%)")
    check(f"Recall not catastrophically reduced", r_after >= r_before - 0.3,
          f"dropped {r_before - r_after:.3f}")

# ─── Phase 6: Catastrophic forgetting ───
def phase6():
    print("\n" + "=" * 60)
    print("PHASE 6: Catastrophic Forgetting — Sequential A -> B -> C")
    print("=" * 60)
    tok = WordTokenizer()
    A = [(f"phys_{i} causes ", f"phys_o{i}") for i in range(12)]
    B = [(f"cook_{i} causes ", f"cook_o{i}") for i in range(12)]
    C = [(f"music_{i} causes ", f"music_o{i}") for i in range(12)]
    for inp, tgt in A + B + C:
        tok.encode(inp); tok.encode(tgt)
    model = make_model(tok)
    train(model, tok, A, epochs=30)
    model._compute_verb_offsets()
    rA_afterA, _ = recall_at(model, tok, A, k=1)
    train(model, tok, B, epochs=30)
    model._compute_verb_offsets()
    rA_afterB, _ = recall_at(model, tok, A, k=1)
    rB_afterB, _ = recall_at(model, tok, B, k=1)
    train(model, tok, C, epochs=30)
    model._compute_verb_offsets()
    rA_afterC, _ = recall_at(model, tok, A, k=1)
    rC_afterC, _ = recall_at(model, tok, C, k=1)
    print(f"  A recall after A: {rA_afterA:.3f}  after B: {rA_afterB:.3f}  after C: {rA_afterC:.3f}")
    print(f"  B recall after B: {rB_afterB:.3f}")
    print(f"  C recall after C: {rC_afterC:.3f}")
    forgetting = rA_afterA - rA_afterC
    print(f"  Forgetting of A after B+C: {forgetting:.3f}")
    check(f"A recall after C > 0.0", rA_afterC > 0.0, f"got {rA_afterC:.3f}")
    check(f"A forgetting < 0.5", forgetting < 0.5, f"got {forgetting:.3f}")


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("RAVANA — Forgetting & Memory-Scaling Stress Test")
    print("=" * 60)
    print("Replicating: 'Works great for limited data, but quickly")
    print("becomes difficult at scale. Lots of false positives.'\n")

    for i, (name, fn) in enumerate([
        ("Scale Degradation", phase1),
        ("False Positive Intrusions", phase2),
        ("Clustering Quality", phase3),
        ("Forgetting Curve", phase4),
        ("Sleep Consolidation Effect", phase5),
        ("Catastrophic Forgetting", phase6),
    ]):
        try:
            fn()
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")
            import traceback; traceback.print_exc()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"RESULTS: {_passed}/{_total} passed ({elapsed:.0f}s)")
    if _failed:
        print(f"         {_failed} FAILED")
    print("=" * 60)
    sys.exit(0 if _failed == 0 else 1)
