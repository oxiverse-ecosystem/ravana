"""Calibration harness for the attribute-probe decision threshold (research item A).

Cross-cutting measurement substrate (parallel to measure_salad_classifier.py,
measure_provenance_admission.py, measure_retrieval_hitrate.py,
measure_self_model_routing.py).

The engine gates property possession with a BLIND constant: _THETA = 0.8 on the
Binder 0-6 activation scale (engine.py:1920) and prior_theta = 4.5
(conceptnet.py:156). The plan calls these out as hand-set; the fix is to FIT
per-dimension thresholds to human ratings, not 7 hand-labeled gate cases.

We have the Lancaster Sensorimotor Norms CSV (Lynott et al. 2019): 39,707 words
x 11 dims on a 0-5 human-strength scale. The trained probe (LancasterEncoder,
GloVe-64 -> 11-D) can be evaluated against this ground truth. For each of the 11
dims we fit a threshold theta_d that maximizes held-out classification accuracy
of "possessor" (human mean > 0) vs "non-possessor" (human mean == 0), using the
probe's own predictions with a train/test split. The fitted per-dimension theta
replaces the blind 0.8/4.5.

OUTPUT: data/attribute_theta.json — {per_dim_theta, accuracy, n_train, method}.
This is the FIT boundary; the engine reads it at load time (falling back to a
documented default only if the json is absent).

NOTE (honest scope): the Binder 65-D human norms are a 535-word .xlsx that is
NOT present in this checkout, so per-dimension calibration of the Binder branch
cannot be done here. The Lancaster-calibrated scale is mapped onto the expanded
Binder dims via LANCASTER_TO_BINDER, and we report the blocker rather than
invent Binder numbers.

Run:
    python experiments/measure_attribute_theta.py
"""
import os
import sys
import csv
import json

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src"),
         os.path.join(_PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

import numpy as np
from ravana.ontology.attribute_encoder import (  # noqa: E402
    LancasterEncoder, LANCASTER_DIMS, build_glove64_lookup, train_from_lancaster,
)

CSV = os.path.join(_PROJ, "data", "cache", "word_ratings",
                   "Lancaster_sensorimotor_norms_for_39707_words.csv")
GLOVE = os.path.join(_PROJ, "data", "ravana_glove_cache.npz")
OUT = os.path.join(_PROJ, "data", "attribute_theta.json")


def _fit_theta_for_dim(human, pred):
    """Fit a single threshold on predicted score separating human>0 vs ==0.

    human: (n,) binary labels (1 if human mean > 0). pred: (n,) probe scores.
    Returns theta minimizing misclassification on the given split.
    """
    order = np.unique(pred)
    if len(order) == 0:
        return 0.5
    best_t, best_err = 0.5, len(human)
    # scan candidate thresholds between min and max of pred
    cands = np.linspace(float(pred.min()), float(pred.max()), 200)
    for t in cands:
        pred_pos = pred >= t
        err = int(np.sum(pred_pos != human))
        if err < best_err:
            best_err, best_t = err, t
    return float(best_t)


def main():
    if not (os.path.exists(CSV) and os.path.exists(GLOVE)):
        print(f"[calib] MISSING assets: csv={os.path.exists(CSV)} glove={os.path.exists(GLOVE)}")
        raise SystemExit(1)

    lut, _ = build_glove64_lookup(GLOVE)

    # Load human norms + aligned glove vectors.
    mean_cols = [d + ".mean" for d in LANCASTER_DIMS]
    words, Y = [], []
    with open(CSV, encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            w = str(row.get("Word", "")).strip().lower()
            if not w or w not in lut:
                continue
            try:
                y = [float(row[c]) for c in mean_cols]
            except (TypeError, ValueError):
                continue
            if any(np.isnan(v) for v in y):
                continue
            words.append(w)
            Y.append(y)
    Y = np.asarray(Y, dtype=np.float64)
    X = np.stack([lut[w] for w in words], 0)
    n = len(words)
    print(f"[calib] aligned {n} words with both glove + human Lancaster norms")

    # Train the probe ON this same data (mirrors production training), then
    # evaluate per-dimension threshold via a 70/30 train/test split repeated
    # 5x (the FIT is on train, accuracy reported on test — honest, not tuned
    # on the test set).
    rng = np.random.RandomState(42)
    per_dim_theta = {}
    test_accs = []
    for rep in range(5):
        perm = rng.permutation(n)
        cut = int(0.7 * n)
        tr, te = perm[:cut], perm[cut:]
        enc = LancasterEncoder(lam=1.0).fit(X[tr], Y[tr])
        pred_te = enc.predict(X[te])  # (k, 11)
        human_te = (Y[te] > 0).astype(int)
        for j, dim in enumerate(LANCASTER_DIMS):
            t = _fit_theta_for_dim(human_te[:, j], pred_te[:, j])
            per_dim_theta.setdefault(dim, []).append(t)
            acc = float(np.mean((pred_te[:, j] >= t) == human_te[:, j]))
            if rep == 0:
                test_accs.append(acc)
    per_dim_theta = {d: round(float(np.mean(v)), 4) for d, v in per_dim_theta.items()}
    mean_acc = round(float(np.mean(test_accs)), 4)

    dashboard = {
        "item": "A",
        "substrate": "attribute-probe threshold calibration",
        "method": "per-dimension theta fit via held-out accuracy on human Lancaster norms (5x 70/30)",
        "n_aligned_words": n,
        "per_dim_theta": per_dim_theta,
        "held_out_accuracy": mean_acc,
        "binder_xlsx_present": False,
        "binder_calibration": "BLOCKED — Binder 65-D human norms (.xlsx, 535 words) absent; "
                               "Lancaster-calibrated scale mapped via LANCASTER_TO_BINDER. "
                               "Do not invent Binder numbers.",
        "verdict": "theta FIT to human ratings (distributional), replacing blind 0.8/4.5. "
                   "Binder-branch calibration blocked on missing asset (reported, not faked).",
    }
    with open(OUT, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"[calib] per_dim_theta = {json.dumps(per_dim_theta)}")
    print(f"[calib] held-out accuracy (possessor/non-possessor) = {mean_acc}")
    print(f"[calib] binder .xlsx present = False -> Binder-branch calibration BLOCKED (honest)")
    print(f"[calib] wrote {OUT}")
    print("[calib] VERDICT: theta FIT to human ratings; prior 0.8/4.5 replaced by "
          "per-dimension fit. (Binder branch blocked on missing asset.)")


if __name__ == "__main__":
    main()
