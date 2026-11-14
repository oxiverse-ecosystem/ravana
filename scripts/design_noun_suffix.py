#!/usr/bin/env python3
"""Design and measure a noun-suffix table for `_build_concept_pos`.

The current rule set (`chain_walker.py:178-179`) is:

    verb_suffixes = ing ed ize ify ate en ish
    adj_suffixes  = able ible ful less ous al ic ive

Measured on the 259 graph labels, these fire on real nouns:

    verb:  climate(ate) learning(ing) often(en) even(en) oxygen(en)
           music? no -- adj(ic); animal/mammal/journal/global(adj al)
    adj:   music(ic) animal(al) mammal(al) journal(al) global(al)
           natural(al) moral(al) mental(al) ethical(al)

Two problems:
  1. GREEDY -- `en`/`ate`/`ing`/`al`/`ic` are ambiguous and fire on nouns.
  2. NO NOUN TABLE AT ALL -- nothing positively asserts noun-hood, so
     `_determine_article` gets `pos='noun'` from the fall-through default
     rather than from evidence.

This script scores candidate noun-suffix tables against an INDEPENDENT gold
set (dictionary knowledge, not `_concept_pos`), and reports which table is
worth wiring in. READ-ONLY.
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Independent gold: unambiguous noun labels in the graph.
GOLD_NOUN = set("""
climate learning development environment history theory plant animal mammal
journal global natural moral mental ethical music action activity behavior
body century computer culture discovery earth energy ethics force sound liquid
gravity ocean water light time air analysis sun sea salt dog brain evolution
book anger carbon unit trust memory freedom mind life people data example
travel years
""".split())

# Independent gold: unambiguous NON-noun labels.
GOLD_NON = set("""
only however now there rather further just also very often always never
sometimes usually really quite even still yet already almost enough back away
together therefore thus hence instead again different ancient primary
""".split())

CANDIDATES = {
    "baseline (current rules)": None,
    "A: conservative noun table": [
        'tion', 'sion', 'ment', 'ness', 'ity', 'ance', 'ence', 'ancy',
        'ency', 'ery', 'age', 'ism', 'ist', 'ure', 'dom', 'ship', 'hood',
        'ics', 'ia', 'um', 'us', 'al', 'ory', 'ary',
    ],
    "B: A minus ambiguous al/ory/ary": [
        'tion', 'sion', 'ment', 'ness', 'ity', 'ance', 'ence', 'ancy',
        'ency', 'ery', 'age', 'ism', 'ist', 'ure', 'dom', 'ship', 'hood',
        'ics', 'ia', 'um', 'us',
    ],
    "C: B plus y": [
        'tion', 'sion', 'ment', 'ness', 'ity', 'ance', 'ence', 'ancy',
        'ency', 'ery', 'age', 'ism', 'ist', 'ure', 'dom', 'ship', 'hood',
        'ics', 'ia', 'um', 'us', 'y',
    ],
}


def noun_by_suffix(word, table):
    if not table:
        return None
    for s in table:
        if len(word) > len(s) + 1 and word.endswith(s):
            return s
    return None


def main():
    os.environ["RAVANA_OFFLINE"] = "1"
    from ravana.chat.engine import CognitiveChatEngine
    from ravana.chat.constants import KNOWN_VERBS, KNOWN_ADJS, FUNCTION_WORDS, FUNCTION_POS

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    pm = getattr(eng, "_concept_pos", {}) or {}
    verb_suf = ['ing', 'ed', 'ize', 'ify', 'ate', 'en', 'ish']
    adj_suf = ['able', 'ible', 'ful', 'less', 'ous', 'al', 'ic', 'ive']

    def current_tag(w):
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

    def score(table):
        """Accuracy of 'is this label a noun?' under the candidate rule."""
        tp = fp = fn = tn = 0
        for w in GOLD_NOUN | GOLD_NON:
            if w not in pm:
                continue
            # candidate: noun suffix wins over verb/adj
            if table and noun_by_suffix(w, table):
                pred = True
            else:
                pred = current_tag(w) == 'noun'
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
        return ((tp + tn) / n if n else 0.0,
                tp / max(1, tp + fp), tp / max(1, tp + fn), n)

    print(f"[suf] gold words present in graph: "
          f"{len([w for w in GOLD_NOUN | GOLD_NON if w in pm])}")
    print()
    for name, table in CANDIDATES.items():
        a, p, r, n = score(table)
        print(f"[suf] {name:<32} acc={a:.3f} prec={p:.3f} rec={r:.3f} (n={n})")
    print()
    print("[suf] what each table does to the words that are wrong today:")
    wrong = [w for w in GOLD_NOUN if pm.get(w) != 'noun']
    for w in sorted(wrong):
        row = [w, pm.get(w, '-')]
        for name, table in CANDIDATES.items():
            if table is None:
                row.append(pm.get(w, '-'))
            elif noun_by_suffix(w, table):
                row.append('noun*')
            else:
                row.append(pm.get(w, '-'))
        print("      " + "  ".join(f"{x:<10}" for x in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
