#!/usr/bin/env python3
"""
Experiment: Adversarial Ambiguity — competing contextual attractors

Tests whether RLM can handle conflicting long-range contexts without
identity collapse or edge explosion:

  Phase 1: Multi-ambiguity — bat→animal AND bat→baseball
  Phase 2: Context priming — scientist→bat→animal vs programmer→bat→baseball
  Phase 3: Adversarial — conflicting contexts compete for same ambiguous node
  Phase 4: Edge competition — does inhibition prevent edge explosion?
"""

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
np.random.seed(42)

from ravana.lab import ConceptLab, ExperimentPhase, Snapshot


def main():
    TOKENS = {
        'bat': 0, 'animal': 1, 'baseball': 2,
        'mouse': 3, 'rodent': 4, 'device': 5,
        'virus': 6, 'biology': 7, 'computer': 8,
        'scientist': 9, 'laboratory': 10,
        'programmer': 11, 'terminal': 12,
    }
    REV = {v: k for k, v in TOKENS.items()}

    def tok(name):
        return TOKENS[name]

    def concept_name(lab, cid):
        tid = lab.concept_token_map(cid)
        return REV.get(tid, f"c{cid}")

    print("=" * 70)
    print("CONCEPT PHYSICS LAB — Experiment: Adversarial Ambiguity")
    print("=" * 70)

    config = dict(
        vocab_size=13, embed_dim=32, concept_dim=32, n_concepts=26,
        n_hidden=32, n_layers=1, max_seq_len=8,
        sleep_interval=5,
    )

    lab = ConceptLab(config, name="adversarial_ambiguity")
    rlm = lab.rlm

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 1: Multi-Ambiguity
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PHASE 1: Multi-Ambiguity")
    print("─" * 70)

    lab.run_phase(ExperimentPhase("multi_ambiguous", [
        (tok('bat'), tok('animal')),
        (tok('bat'), tok('baseball')),
        (tok('mouse'), tok('rodent')),
        (tok('mouse'), tok('device')),
        (tok('virus'), tok('biology')),
        (tok('virus'), tok('computer')),
    ], n_repeats=20))

    # Check edges for each ambiguous node
    for ambig_name, ambig_tok in [('bat', tok('bat')), ('mouse', tok('mouse')), ('virus', tok('virus'))]:
        ac = lab.token_concept_map(ambig_tok)
        print(f"\n  {ambig_name} concept: c{ac}")
        print(f"  Edges from c{ac}:")
        for (s, t), e in rlm.graph.edges.items():
            if s == ac:
                target_name = REV.get(lab.concept_token_map(t), f"c{t}")
                print(f"    → c{t} ({target_name}) w={e.weight:.3f} conf={e.confidence:.3f}")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 2: Context Priming
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PHASE 2: Context Priming")
    print("─" * 70)

    for rep in range(80):
        # scientist → bat → animal
        rlm.learn(np.array([tok('scientist'), tok('bat')], dtype=np.int64),
                  np.array([tok('animal')], dtype=np.int64))
        rlm.learn(np.array([tok('scientist')], dtype=np.int64),
                  np.array([tok('bat')], dtype=np.int64))
        rlm.learn(np.array([tok('bat')], dtype=np.int64),
                  np.array([tok('animal')], dtype=np.int64))
        # programmer → bat → baseball
        rlm.learn(np.array([tok('programmer'), tok('bat')], dtype=np.int64),
                  np.array([tok('baseball')], dtype=np.int64))
        rlm.learn(np.array([tok('programmer')], dtype=np.int64),
                  np.array([tok('bat')], dtype=np.int64))
        rlm.learn(np.array([tok('bat')], dtype=np.int64),
                  np.array([tok('baseball')], dtype=np.int64))
        # laboratory → virus → biology
        rlm.learn(np.array([tok('laboratory'), tok('virus')], dtype=np.int64),
                  np.array([tok('biology')], dtype=np.int64))
        rlm.learn(np.array([tok('laboratory')], dtype=np.int64),
                  np.array([tok('virus')], dtype=np.int64))
        rlm.learn(np.array([tok('virus')], dtype=np.int64),
                  np.array([tok('biology')], dtype=np.int64))
        # terminal → virus → computer
        rlm.learn(np.array([tok('terminal'), tok('virus')], dtype=np.int64),
                  np.array([tok('computer')], dtype=np.int64))
        rlm.learn(np.array([tok('terminal')], dtype=np.int64),
                  np.array([tok('virus')], dtype=np.int64))
        rlm.learn(np.array([tok('virus')], dtype=np.int64),
                  np.array([tok('computer')], dtype=np.int64))

    lab.snapshots.append(Snapshot(rlm, "after_context_priming"))

    rlm.context_scale = 3.0

    context_tests = [
        ([tok('scientist'), tok('bat')], 'animal', 'bat'),
        ([tok('programmer'), tok('bat')], 'baseball', 'bat'),
        ([tok('laboratory'), tok('virus')], 'biology', 'virus'),
        ([tok('terminal'), tok('virus')], 'computer', 'virus'),
    ]

    all_pass = True
    for ctx, expected, ambig_name in context_tests:
        result = lab.probe_with_context(ctx)
        pred_name = REV.get(result['predicted'], f"c{result['predicted']}")
        correct = pred_name == expected
        status = "PASS" if correct else "FAIL"
        if not correct:
            all_pass = False
        print(f"  Context {[REV[c] for c in ctx]} → {pred_name} (expected {expected}) [{status}]")

    print(f"\n  Phase 2 overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    # Direct probe (no context)
    rlm.context_scale = 0.0
    for ambig_name, ambig_tok in [('bat', 0), ('virus', 6)]:
        probe = lab.probe(ambig_tok)
        print(f"  {ambig_name} alone → {REV.get(probe['predicted'], probe['predicted'])}")
    rlm.context_scale = 3.0

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 3: Adversarial — Conflicting Long-Range Contexts
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PHASE 3: Adversarial Conflicts")
    print("─" * 70)

    for rep in range(80):
        # programmer → laboratory → virus → biology
        # (programmer suggests computer, but laboratory → biology dominates)
        rlm.learn(np.array([tok('programmer'), tok('laboratory'), tok('virus')], dtype=np.int64),
                  np.array([tok('biology')], dtype=np.int64))
        rlm.learn(np.array([tok('programmer')], dtype=np.int64),
                  np.array([tok('laboratory')], dtype=np.int64))
        rlm.learn(np.array([tok('laboratory')], dtype=np.int64),
                  np.array([tok('virus')], dtype=np.int64))
        # scientist → terminal → virus → computer
        rlm.learn(np.array([tok('scientist'), tok('terminal'), tok('virus')], dtype=np.int64),
                  np.array([tok('computer')], dtype=np.int64))
        rlm.learn(np.array([tok('scientist')], dtype=np.int64),
                  np.array([tok('terminal')], dtype=np.int64))
        rlm.learn(np.array([tok('terminal')], dtype=np.int64),
                  np.array([tok('virus')], dtype=np.int64))

    adversarial_tests = [
        # T-2 context = laboratory → should bias toward biology (laboratory→biology shortcut)
        ([tok('programmer'), tok('laboratory'), tok('virus')], 'biology',
         "programmer→lab→virus → biology (lab overrides programmer)"),
        # T-2 context = terminal → should bias toward computer (terminal→computer shortcut)
        ([tok('scientist'), tok('terminal'), tok('virus')], 'computer',
         "scientist→terminal→virus → computer (terminal overrides scientist)"),
    ]

    all_pass = True
    for ctx, expected, desc in adversarial_tests:
        result = lab.probe_with_context(ctx)
        pred_name = REV.get(result['predicted'], f"c{result['predicted']}")
        correct = pred_name == expected
        status = "PASS" if correct else "FAIL"
        if not correct:
            all_pass = False
        print(f"  {desc}")
        print(f"    → {pred_name} (expected {expected}) [{status}]")

    print(f"\n  Phase 3 overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 4: Edge Competition Analysis
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PHASE 4: Edge Competition Analysis")
    print("─" * 70)

    topo = lab.edge_topology_summary(-1)
    print(f"\n  Total edges: {topo['n_edges']}")
    print(f"  Mean weight: {topo['mean_weight']:.3f}")
    print(f"  Mean confidence: {topo['mean_confidence']:.3f}")
    print(f"  Edges at weight=1.0: {topo['n_edges_weight_1']}")

    # Check edge distribution per source node
    source_edge_counts = {}
    source_outgoing = {}
    for (s, t), e in rlm.graph.edges.items():
        source_edge_counts[s] = source_edge_counts.get(s, 0) + 1
        source_outgoing[s] = source_outgoing.get(s, 0.0) + e.weight

    print(f"\n  Outgoing edge distribution:")
    for s in sorted(source_outgoing.keys()):
        label = concept_name(lab, s)
        n_edges = source_edge_counts[s]
        total_w = source_outgoing[s]
        print(f"    c{s} ({label}): {n_edges} edges, total weight={total_w:.3f}")

    # Edge explosion check
    print(f"\n  Edge explosion check:")
    n_vocab = config['vocab_size']
    max_theoretical = n_vocab * (n_vocab - 1)  # all possible pairs
    density = topo['n_edges'] / max_theoretical * 100
    print(f"    Density: {topo['n_edges']}/{max_theoretical} ({density:.1f}%)")
    if topo['n_edges'] > n_vocab * 3:
        print(f"    WARNING EDGE EXPLOSION: {topo['n_edges']} > {n_vocab * 3} threshold")
    else:
        print(f"    OK Edges contained below {n_vocab * 3} threshold")

    loc = lab.free_energy_localization(-1)
    print(f"\n  Pressure localization: entropy={loc['normalized_entropy']:.3f}, hotspots={loc['hotspots']}")

    print(f"\nFinal: {rlm}")
    print("=" * 70)


if __name__ == "__main__":
    main()
