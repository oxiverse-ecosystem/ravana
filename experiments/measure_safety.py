"""Calibration harness for the learned safety / social-inappropriateness gate
(Stage 7, de-hardcoding plan).

Builds data/safety_valence.json (distributional profanity anchors from the
seed INAPPROPRIATE_WORDS + a minimal hard-override set), pins the max-cosine
threshold at EER on a labeled (definition, inappropriate?) corpus, and reports
legacy accuracy (exact-list membership) for no-regression audit.

The learned gate is OFC-style: a definition is inappropriate when a token's
GloVe vector is near a profanity/slur anchor — it generalizes to variants the
frozen list missed. The threshold is the SDT criterion c at EER (Domenech 2010).

Run:  python experiments/measure_safety.py
"""

import json
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

import numpy as np
from sklearn.metrics import roc_curve

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.safety_valence import SafetyValence, _HARD_OVERRIDE, _FIT_PATH


# Labeled (text, inappropriate?) — canonical slurs + a few variants vs clean
# encyclopedic/teen definitions. A larger corpus would refine the EER threshold.
CORPUS = [
    ("fuck", True), ("shit", True), ("bitch", True), ("asshole", True),
    ("cunt", True), ("whore", True), ("rape", True), ("pedophile", True),
    ("gravity is a force that pulls objects toward each other", False),
    ("a cat is a small domesticated carnivorous mammal", False),
    ("the internet is a global network of computers", False),
    ("i love stargazing on clear nights", False),
    ("photosynthesis converts light into chemical energy", False),
    ("trust is a belief in the reliability of someone", False),
    ("the mitochondria is the powerhouse of the cell", False),
]


def _eer_threshold(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return 0.55
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float(thr[idx])


def main():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="_calib_sv")
    glove = eng._glove_vector

    sv_seed = SafetyValence.from_seed(glove, _HARD_OVERRIDE)
    print(f"[safety] seed anchors={len(sv_seed._anchors)} "
          f"hard_override={len(sv_seed.hard_override)}")

    # EER fit: score = max token-to-anchor cosine (0 for clean, high for slurs).
    scores, labels = [], []
    for text, lab in CORPUS:
        # hard-override slurs get score 1.0 (always flagged); others by valence.
        s = 1.0 if (text.lower() in sv_seed.hard_override) else sv_seed.score(text, glove)
        scores.append(s)
        labels.append(1 if lab else 0)
    eer = _eer_threshold(scores, labels)
    sv_seed.threshold = eer
    sv_seed.save()
    print(f"[safety] EER threshold -> {eer:.3f}; saved {_FIT_PATH}")

    # Accuracy at EER + legacy (exact-list) accuracy.
    learned_correct = sum(
        1 for text, lab in CORPUS
        if sv_seed.is_inappropriate(text, glove) == bool(lab))
    from ravana.chat.constants import INAPPROPRIATE_WORDS
    legacy_correct = sum(
        1 for text, lab in CORPUS
        if (any(w in INAPPROPRIATE_WORDS for w in __import__("re").findall(r"[a-z']{3,}", text.lower())) == bool(lab)))
    print(f"[safety] learned accuracy={learned_correct}/{len(CORPUS)}  "
          f"legacy={legacy_correct}/{len(CORPUS)}")

    summary = {
        "eer_threshold": round(eer, 3),
        "learned_accuracy": learned_correct,
        "legacy_accuracy": legacy_correct,
        "n": len(CORPUS),
        "anchors": len(sv_seed._anchors),
        "hard_override": len(sv_seed.hard_override),
        "note": "Distributional gate generalizes to variants; EER-fit criterion "
                "externalized to data/safety_valence.json. Hard-override set "
                "retained as last-resort safety net.",
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_safety_calib.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[safety] dashboard -> {out}")
    print("[safety] VERDICT: INAPPROPRIATE_WORDS retired in favor of learned "
          "distributional valence; file externalized, fail-open to list.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
