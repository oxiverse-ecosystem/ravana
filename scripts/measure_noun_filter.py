#!/usr/bin/env python3
"""Measure the noun-admissibility filter before wiring it in.

Two questions:
  1. ACCURACY -- on an independent gold set, does `looks_like_noun` agree with
     dictionary knowledge? (Same methodology as
     `scripts/evaluate_noun_suffix.py`: gold never read from `_concept_pos`.)
  2. EFFECT -- on the live store, which plan-route targets does the filter
     reject, and does the realized output improve?

READ-ONLY. Usage:
    uv run python scripts/measure_noun_filter.py
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GOLD_NOUN = set("""
action activity animal behavior body book brain carbon cat century climate
computer culture discovery dog earth energy environment ethics evolution
force freedom gravity history journal learning liquid mammal mind music
natural ocean oxygen plant sea salt sound sun theory time trust unit water
years
""".split())

GOLD_NON = set("""
only however now there rather further just also very often always never
sometimes usually really quite even still yet already almost enough back away
together therefore thus hence instead again different ancient primary global
mental moral ethical physical practical individual original social
environmental mathematical psychological
""".split())


def main():
    os.environ["RAVANA_OFFLINE"] = "1"
    from ravana.chat.engine import CognitiveChatEngine
    from ravana.learn.noun_admissibility import looks_like_noun

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    pm = getattr(eng, "_concept_pos", {}) or {}

    # ── 1. accuracy on gold ────────────────────────────────────────────────
    tp = fp = fn = tn = 0
    wrong = []
    for w in GOLD_NOUN | GOLD_NON:
        if w not in pm:
            continue
        pred = looks_like_noun(w, pos_map=pm)
        gold = w in GOLD_NOUN
        if pred and gold:
            tp += 1
        elif pred:
            fp += 1
            wrong.append(('FP', w))
        elif gold:
            fn += 1
            wrong.append(('FN', w))
        else:
            tn += 1
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print(f"[nf] gold n={n}  acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}")
    if wrong:
        print(f"[nf] errors ({len(wrong)}):")
        for k, w in wrong[:20]:
            print(f"       {k}  {w}")

    # ── 2. effect on live plan targets ─────────────────────────────────────
    from eval_decoder_groundedness import build_ctx, discover_subjects
    kept, dropped = [], []
    for s in discover_subjects(eng)[:40]:
        ctx = build_ctx(eng, s, f"what is {s}")
        nids = eng._concept_keywords.get(s.lower(), [])
        if not nids:
            continue
        try:
            edges = eng.graph.get_outgoing(nids[0]) or []
        except Exception:
            continue
        for tgt, ed in edges:
            tn_ = eng.graph.get_node(tgt)
            lab = (getattr(tn_, 'label', '') or '').strip()
            if not lab:
                continue
            rel = getattr(ed, 'relation_type', 'related_to') or 'related_to'
            rec = (rel, lab, float(getattr(ed, 'weight', 0.0) or 0.0),
                   float(getattr(ed, 'confidence', 0.0) or 0.0))
            (kept if looks_like_noun(lab, pos_map=pm) else dropped).append(
                (s, rel, lab))

    print(f"\n[nf] live edges: kept={len(kept)} dropped={len(dropped)}")
    seen = set()
    print("[nf] dropped targets (sample):")
    for s, rel, lab in dropped:
        if lab in seen:
            continue
        seen.add(lab)
        print(f"       {lab:<16} (rel={rel}, subj={s})")
        if len(seen) >= 20:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
