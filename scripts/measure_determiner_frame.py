#!/usr/bin/env python3
"""Independent replication of the determiner-frame measurement.

docs/BRAIN_LEXICAL_CATEGORY.md reports the corpus determiner frame reaching
accuracy 0.787 on a 75-word gold set. That gold set was built by the same
subagent that chose the mechanism. This script re-measures with a gold set
assigned from dictionary knowledge only, and also reports the circular version
(gold = the current POS map) so the gap is visible.

READ-ONLY. Usage:
    uv run python scripts/measure_determiner_frame.py
"""
import collections
import os
import re
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CORPUS = os.path.join(PROJ, "data", "corpora", "grounded_descriptions.txt")
DET = re.compile(r"\b(a|an|the|this|that|these|those)\s+([a-z]+)")

# Assigned from dictionary knowledge. Deliberately does NOT read _concept_pos.
GOLD_NOUN = set("""
climate learning development environment history theory plant force sound liquid
earth years gravity ocean water light time music air analysis sun sea salt dog
brain evolution book anger carbon body unit trust memory freedom mind life
people energy data example travel
""".split())
GOLD_NONNOUN = set("""
only however now there rather further just also very often always never
sometimes usually really quite even still yet already almost enough back away
together therefore thus hence instead again furthermore moreover nevertheless
nonetheless otherwise meanwhile indeed perhaps maybe probably possibly
certainly definitely clearly obviously different ancient primary
""".split())


def counts(path):
    with open(path, encoding="utf-8") as fh:
        low = fh.read().lower()
    return collections.Counter(m.group(2) for m in DET.finditer(low)), len(low)


def score(gold_noun, gold_non, cnt, threshold):
    tp = fp = fn = tn = 0
    for w in gold_noun | gold_non:
        pred = cnt.get(w, 0) >= threshold
        gold = w in gold_noun
        if pred and gold:
            tp += 1
        elif pred:
            fp += 1
        elif gold:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    return ((tp + tn) / n if n else 0.0,
            tp / max(1, tp + fp), tp / max(1, tp + fn))


def main():
    cnt, nchar = counts(CORPUS)
    print(f"[det] corpus chars={nchar}  distinct words after a determiner={len(cnt)}")
    print()
    print("[det] INDEPENDENT gold (dictionary knowledge, n=%d)"
          % len(GOLD_NOUN | GOLD_NONNOUN))
    for t in (1, 2, 3):
        a, p, r = score(GOLD_NOUN, GOLD_NONNOUN, cnt, t)
        print(f"       det>={t}  acc={a:.3f}  prec={p:.3f}  rec={r:.3f}")
    print()
    print("[det] CIRCULAR gold (gold = the current POS map) -- for contrast only")
    os.environ["RAVANA_OFFLINE"] = "1"
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    pm = getattr(eng, "_concept_pos", {}) or {}
    gn = {w for w, p in pm.items() if p == "noun"}
    gnon = {w for w, p in pm.items() if p != "noun"}
    for t in (1, 2, 3):
        a, p, r = score(gn, gnon, cnt, t)
        print(f"       det>={t}  acc={a:.3f}  prec={p:.3f}  rec={r:.3f}")
    print()
    print("[det] cue counts on the words that matter")
    for w in ("only", "however", "now", "there", "rather", "further",
              "climate", "learning", "development", "history", "sound",
              "years", "earth", "force"):
        print(f"       {w:<14} {cnt.get(w, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
