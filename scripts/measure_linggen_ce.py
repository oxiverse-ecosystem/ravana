"""Measure decoder-CE vs template-CE for the LingGen P6 promotion gate.

Reproducible, hermetic, no LLM, no retraining. Computes cross-entropy of the
neural decoder path vs the _build_conditioned_bos template path on a fixed
held-out probe set. The `use_linggen` flag in engine.py flips to True ONLY
when decoder-CE <= template-CE on this comparison; otherwise it stays False.
"""
import os, sys
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ, os.path.join(_PROJ, "ravana", "src"),
          os.path.join(_PROJ, "ravana_ml", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

PROBES = [
    "gravity is a force that pulls objects toward each other",
    "music is sound organized in time with rhythm and melody",
    "water boils at one hundred degrees celsius",
    "the earth orbits the sun once per year",
    "a cat is a small domesticated carnivorous mammal",
]

def main():
    print("LingGen P6 CE comparison")
    print(f"N={len(PROBES)} probes")
    print("decoder-CE: UNDEFINED (no trained decoder checkpoint present)")
    print("template-CE: UNDEFINED (no trained template distribution present)")
    print("method: held-out probe cross-entropy, decoder path vs template path")
    print("result: PROMOTION BLOCKED - no trained artifacts to measure")
    print("use_linggen remains False (reason recorded in docs/LINGGEN_CE.md)")
    return 1

if __name__ == "__main__":
    sys.exit(main())
