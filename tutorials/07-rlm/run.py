"""Tutorial 07: RLMv2 instantiation and usage.

Final tutorial in the progression. Shows the ML model behind relation learning.

Usage:
    python tutorials/07-rlm/run.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "ravana_ml", "src"))

from ravana_ml.nn.rlm_v2 import RLMv2


def main() -> None:
    print("=== RLMv2 — Relation Learner Model ===\n")

    # 1. Configure the model
    vocab_size = 500  # number of unique words in vocabulary
    embed_dim = 64    # embedding dimension (matches graph dimension)
    concept_dim = 64  # concept vector dimension

    # 2. Instantiate RLMv2 — the triple decomposition model
    # (subject, relation, object) → vector arithmetic
    model = RLMv2(
        vocab_size=vocab_size + 10,  # +10 for special tokens
        embed_dim=embed_dim,
        concept_dim=concept_dim,
        n_concepts=200,
    )
    print(f"  Model type:       {type(model).__name__}")
    print(f"  Embed dimension:  {embed_dim}")
    print(f"  Concept dimension:{concept_dim}")
    print(f"  Vocab size:       {vocab_size}")
    print()

    # 3. What RLMv2 does internally
    print("  RLMv2 decomposes sentences into (subject, relation, object) triples:")
    print('    "heat causes expansion"  ->  (heat, causes, expansion)')
    print()
    print("  This enables analogy via vector arithmetic:")
    print("    subject_embed + offset(verb) ~= target_embed")
    print('    heat + offset("causes") ~= expansion')
    print()

    # 4. Where RLMv2 is used in the codebase
    print("  Used by:")
    print("    - scripts/benchmark_vs_transformers.py  - held-out benchmarks")
    print("    - scripts/validate_held_out_generalization.py  - verb-offset checks")
    print("    - scripts/external_benchmark.py  - cross-domain profiling")
    print("    - scripts/diagnose_transfer.py  - cross-domain transfer analysis")


if __name__ == "__main__":
    main()
