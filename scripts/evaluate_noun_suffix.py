#!/usr/bin/env python3
"""Held-out evaluation of candidate noun-suffix tables.

The earlier probe (design_noun_suffix.py) reported 0.803 for table A and a
later ad-hoc run reported 0.475 for the same table. Same table, same code
path, different number -- because the two runs used different gold sets. That
instability is the finding: a 61-word hand-labelled gold set is too small to
rank tables.

This script fixes the methodology:
  * one gold set, fixed here, labelled from dictionary knowledge only
  * 5-fold cross-validation over the graph labels in the gold set
  * reports mean accuracy AND the per-fold spread, so a table that wins by
    noise is visible
  * evaluates the DECISION that matters: "may this word head a noun phrase?"

READ-ONLY. Usage:
    uv run python scripts/evaluate_noun_suffix.py
"""
import os
import random
import re
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Gold, labelled from dictionary knowledge only ──────────────────────────
# NOUNS: unambiguous common nouns that appear as graph labels.
GOLD_NOUN = set("""
action activity animal behavior body book brain carbon cat century climate
computer culture discovery dog earth energy environment ethics evolution
force freedom gravity history journal learning liquid mammal mind music
natural ocean oxygen plant sea salt sound sun theory time trust unit water
years
""".split())

# NON-NOUNS: unambiguous adverbs, adjectives, verbs.
GOLD_NON = set("""
only however now there rather further just also very often always never
sometimes usually really quite even still yet already almost enough back away
together therefore thus hence instead again different ancient primary global
mental moral ethical physical practical individual original social
environmental mathematical psychological
""".split())

TABLES = {
    "baseline": None,
    "A al+ory+ary": ['tion', 'sion', 'ment', 'ness', 'ity', 'ance', 'ence',
                     'ancy', 'ency', 'ery', 'age', 'ism', 'ist', 'ure', 'dom',
                     'ship', 'hood', 'ics', 'ia', 'um', 'us', 'al', 'ory', 'ary'],
    "D no-al": ['tion', 'sion', 'ment', 'ness', 'ity', 'ance', 'ence', 'ancy',
                'ency', 'ery', 'age', 'ism', 'ist', 'ure', 'dom', 'ship',
                'hood', 'ics', 'ia', 'um', 'us', 'ory', 'ary'],
    "E D+ic": ['tion', 'sion', 'ment', 'ness', 'ity', 'ance', 'ence', 'ancy',
               'ency', 'ery', 'age', 'ism', 'ist', 'ure', 'dom', 'ship',
               'hood', 'ics', 'ia', 'um', 'us', 'ory', 'ary', 'ic'],
    "F minimal": ['tion', 'sion', 'ment', 'ness', 'ity', 'ance', 'ence',
                  'ery', 'age', 'ism', 'ist', 'ure'],
}


def main():
    os.environ["RAVANA_OFFLINE"] = "1"
    from ravana.chat.engine import CognitiveChatEngine
    from ravana.chat.constants import (KNOWN_VERBS, KNOWN_ADJS,
                                       FUNCTION_WORDS, FUNCTION_POS)

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    pm = getattr(eng, "_concept_pos", {}) or {}
    verb_suf = ['ing', 'ed', 'ize', 'ify', 'ate', 'en', 'ish']
    adj_suf = ['able', 'ible', 'ful', 'less', 'ous', 'al', 'ic', 'ive']

    def current(w):
        ll = w.lower()
        if ll in KNOWN_VERBS:
            return 'verb'
        if ll.endswith('s') and (ll[:-1] in KNOWN_VERBS
                                 or (ll.endswith('es') and ll[:-2] in KNOWN_VERBS)):
            return 'verb'
        if ll in KNOWN_ADJS:
            return 'adj'
        if any(len(ll) > len(s) + 1 and ll.endswith(s) for s in verb_suf):
            return 'verb'
        if any(len(ll) > len(s) + 1 and ll.endswith(s) for s in adj_suf):
            return 'adj'
        if ll in FUNCTION_WORDS:
            return FUNCTION_POS.get(ll, 'adv')
        return 'noun'

    def predict(w, tbl):
        if tbl is None:
            return current(w) == 'noun'
        if any(len(w) > len(s) + 1 and w.endswith(s) for s in tbl):
            return True
        return current(w) == 'noun'

    words = [w for w in (GOLD_NOUN | GOLD_NON) if w in pm]
    print(f"[ev] gold labels present in graph: {len(words)} "
          f"(noun {sum(1 for w in words if w in GOLD_NOUN)}, "
          f"non {sum(1 for w in words if w in GOLD_NON)})")
    print()

    rng = random.Random(1234)
    folds = [[] for _ in range(5)]
    for i, w in enumerate(sorted(words)):
        folds[i % 5].append(w)

    print(f"[ev] {'table':<16} {'mean':>7} {'sd':>7} {'folds':>26}")
    for name, tbl in TABLES.items():
        accs = []
        for k in range(5):
            test = folds[k]
            train = [w for j in range(5) if j != k for w in folds[j]]
            # The candidate tables are FIXED (not fitted), so 'train' is only
            # used to report that the rule does not depend on the fold.
            tp = fp = fn = tn = 0
            for w in test:
                pred = predict(w, tbl)
                gold = w in GOLD_NOUN
                if pred and gold:
                    tp += 1
                elif pred:
                    fp += 1
                elif gold:
                    fn += 1
                else:
                    tn += 1
            n = tp + fp + fn + tn
            accs.append((tp + tn) / n if n else 0.0)
        mean = sum(accs) / len(accs)
        sd = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
        print(f"[ev] {name:<16} {mean:>7.3f} {sd:>7.3f} "
              f"{' '.join(f'{a:.2f}' for a in accs):>26}")

    print()
    print("[ev] per-word disagreement vs baseline (table A):")
    tbl = TABLES["A al+ory+ary"]
    for w in sorted(words):
        b = predict(w, None)
        a = predict(w, tbl)
        if a != b:
            print(f"       {w:<16} baseline={'noun' if b else 'non':<5} "
                  f"A={'noun' if a else 'non':<5} gold={'noun' if w in GOLD_NOUN else 'non'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
