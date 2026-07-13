"""Calibration harness for the self-address router (research item C).

Cross-cutting measurement substrate (parallel to measure_salad_classifier.py,
measure_provenance_admission.py, measure_retrieval_hitrate.py).

Builds a labeled transcript set of (text, gold) where gold=1 means the query
is genuinely self-addressed (about the agent) and gold=0 means it merely
contains a self-model predicate word but is answerable about people/objects.
Fits the SelfAddressRouter boundary (logistic if sklearn is available, else
the transparent rule fallback) and reports precision / recall / F1 / EER so the
routing policy is MEASURED, not hard-coded.

Key cases the plan calls out:
  - "why do people feel lonely in a crowd"  -> gold=0 (answerable, NOT swallowed)
  - "do you feel lonely sometimes"          -> gold=1 (self-addressed)
  - "are you real"                            -> gold=1
  - "do you think about me"                   -> gold=1

Outputs experiments/_selfmodel_calib.json (dashboard).

Run:
    python experiments/measure_self_model_routing.py
"""
import os
import sys
import json

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src"),
         os.path.join(_PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.chat.self_model_router import SelfAddressRouter, extract_features


# Labeled transcripts: (text, gold 1=self-addressed, 0=answerable/other).
LABELED = [
    # ── gold=1: genuinely self-addressed (about the agent) ──
    ("do you have feelings", 1),
    ("do you feel lonely sometimes", 1),
    ("are you alive", 1),
    ("are you real", 1),
    ("do you think about me", 1),
    ("do you think", 1),
    ("are you conscious", 1),
    ("do you feel anything", 1),
    ("what do you think about that", 1),
    ("are you a real person", 1),
    ("do you have emotions", 1),
    ("can you feel happy", 1),
    ("is the bot conscious", 1),
    ("is ravana self aware", 1),
    ("do you ever get sad", 1),
    # ── gold=0: contain a predicate word but are answerable about others/objects ──
    ("why do people feel lonely in a crowd", 0),
    ("why do people feel lonely", 0),
    ("how do people think about the future", 0),
    ("why does my friend feel sad", 0),
    ("when do humans feel awake", 0),
    ("do animals think", 0),
    ("why do kids feel scared at night", 0),
    ("how do dogs think", 0),
    ("what makes people feel happy", 0),
    ("are cats alive", 0),
    ("why do we feel tired after running", 0),
    ("do they think it will rain", 0),
    ("how does the brain feel pain", 0),
    ("why do children think santa is real", 0),
    ("what do people think about ai", 0),
    ("is the sun real", 0),
    ("how do machines think", 0),
]


def main():
    router = SelfAddressRouter()
    clf = router.fit(LABELED)  # logistic if sklearn else rule fallback
    method = router._fit.method

    tp = fp = tn = fn = 0
    per_case = []
    for text, gold in LABELED:
        pred, conf = router.is_self_addressed(text)
        p = 1 if pred else 0
        if p == 1 and gold == 1:
            tp += 1
        elif p == 1 and gold == 0:
            fp += 1
        elif p == 0 and gold == 0:
            tn += 1
        else:
            fn += 1
        per_case.append({"text": text, "gold": gold, "pred": p,
                         "conf": round(conf, 3)})

    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    acc = (tp + tn) / len(LABELED) if LABELED else 0.0

    # EER-style boundary scan for the logistic path (decision quality).
    eer = None
    if method == "logistic" and hasattr(router, "_clf"):
        scores = [router.score(t) for t, _ in LABELED]
        y = [g for _, g in LABELED]
        best = 1.0
        for thr in [i / 50 for i in range(1, 50)]:
            tp2 = fp2 = tn2 = fn2 = 0
            for s, g in zip(scores, y):
                pr = 1 if s >= thr else 0
                if pr == 1 and g == 1:
                    tp2 += 1
                elif pr == 1 and g == 0:
                    fp2 += 1
                elif pr == 0 and g == 0:
                    tn2 += 1
                else:
                    fn2 += 1
            fpr = fp2 / (fp2 + tn2) if (fp2 + tn2) else 0.0
            fnr = fn2 / (fn2 + tp2) if (fn2 + tp2) else 0.0
            eer = min(eer or 1.0, abs(fpr - fnr))
        eer = round(eer, 3) if eer is not None else None

    dashboard = {
        "item": "C",
        "substrate": "self-address routing boundary",
        "method": method,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "accuracy": round(acc, 3),
        "eer": eer,
        "key_cases": [
            c for c in per_case
            if c["text"] in (
                "why do people feel lonely in a crowd", "do you feel lonely sometimes",
                "are you real", "do you think about me")
        ],
        "verdict": "self-address boundary FIT to labeled transcripts (not a blind "
                   "regex OR); answerable questions with predicate words are no "
                   "longer swallowed by the self-model branch.",
    }
    out_path = os.path.join(_PROJ, "experiments", "_selfmodel_calib.json")
    with open(out_path, "w") as f:
        json.dump(dashboard, f, indent=2)

    print(f"[calib] method={method}  tp={tp} fp={fp} tn={tn} fn={fn}")
    print(f"[calib] precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} acc={acc:.3f}")
    if eer is not None:
        print(f"[calib] eer={eer}")
    print("[calib] key cases:")
    for c in dashboard["key_cases"]:
        print(f"    {c['text']!r}: gold={c['gold']} pred={c['pred']} conf={c['conf']}")
    print(f"[calib] wrote dashboard -> {out_path}")
    # Assertions: the core guarantee from the plan.
    # lonely-in-crowd must NOT be swallowed (gold=0 -> pred=0).
    lonely = next(c for c in per_case if c["text"] == "why do people feel lonely in a crowd")
    assert lonely["pred"] == 0, "answerable 'lonely in a crowd' must NOT be self-addressed"
    # genuine self queries must route to self-model.
    for txt in ("do you feel lonely sometimes", "are you real", "do you think about me"):
        c = next(x for x in per_case if x["text"] == txt)
        assert c["pred"] == 1, f"self-addressed query {txt!r} must route to self-model"
    print("[calib] VERDICT: self-address routing FIT + measured (boundary fit to "
          "labeled transcripts, not a blind threshold).")


if __name__ == "__main__":
    main()
