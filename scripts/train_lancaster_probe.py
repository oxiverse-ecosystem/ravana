#!/usr/bin/env python3
"""
Train the wide-coverage Lancaster sensorimotor probe (Fix A.1).

The production cross-modal probe is a GloVe-64 -> Binder 65-D ridge trained on
only 535 human-rated words. This trains the complementary GloVe-64 ->
Lancaster 11-D probe on 39,707 human-rated words (Lynott et al. 2019), giving
the combined encoder open-vocabulary coverage. Output: data/lancaster_encoder.npz.

Usage:
    python scripts/train_lancaster_probe.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ravana", "src"))

from ravana.ontology.attribute_encoder import train_from_lancaster

CACHE = os.path.join(ROOT, "data", "ravana_glove_cache.npz")
CSV = os.path.join(ROOT, "data", "cache", "word_ratings",
                  "Lancaster_sensorimotor_norms_for_39707_words.csv")
OUT = os.path.join(ROOT, "data", "lancaster_encoder.npz")


def main() -> None:
    if not os.path.exists(CACHE):
        sys.exit(f"missing glove cache: {CACHE}")
    if not os.path.exists(CSV):
        sys.exit(f"missing Lancaster norms CSV: {CSV}")
    enc = train_from_lancaster(CACHE, CSV, OUT)
    av = enc.attribute_vector  # smoke: ensure it is fitted
    print(f"trained Lancaster probe on {len(enc.dims)} dims -> {OUT} "
          f"({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
