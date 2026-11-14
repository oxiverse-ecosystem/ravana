#!/usr/bin/env python3
"""Quantify the POS mislabelling that produces 'an only'.

Reports, over the persisted graph's concept labels:
  * how many fall through to the 'noun' default
  * how many of those are adverbs / adjectives by independent evidence
  * which plan-route outputs are affected

READ-ONLY. Usage:
    uv run python scripts/measure_pos_mislabel.py
"""
import os
import re
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Independent evidence: these are unambiguous closed-class / adverbial words.
# Not a hand-tuned exception list for the fix -- a measurement instrument.
ADVERBS = set("""
only very often always never sometimes usually really quite rather just also
too now then here there when where how why still yet already almost enough
even back away together however therefore thus hence instead again further
furthermore moreover nevertheless nonetheless otherwise meanwhile indeed
perhaps maybe probably possibly certainly definitely clearly obviously
""".split())

ADJ_SUFFIXES = ('able', 'ible', 'ful', 'less', 'ous', 'ive', 'ish', 'like',
                'some', 'ent', 'ant', 'ary', 'ory', 'ical')


def main():
    os.environ["RAVANA_OFFLINE"] = "1"
    from ravana.chat.engine import CognitiveChatEngine

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    pm = getattr(eng, "_concept_pos", {}) or {}
    nouns = [w for w, p in pm.items() if p == "noun"]
    adv = sorted(w for w in nouns if w in ADVERBS)
    adj = sorted(w for w in nouns
                 if any(w.endswith(s) for s in ADJ_SUFFIXES) and len(w) > 4)

    print(f"[pos] concept labels tagged : {len(pm)}")
    print(f"[pos] tagged 'noun'         : {len(nouns)} "
          f"({100 * len(nouns) / max(1, len(pm)):.0f}%)")
    print(f"[pos] noun-tagged ADVERBS   : {len(adv)}")
    print(f"       {adv}")
    print(f"[pos] noun-tagged ADJ-suffix: {len(adj)}")
    print(f"       {adj}")

    # Which live plan outputs are affected?
    from eval_decoder_groundedness import build_ctx, discover_subjects
    bad = []
    for s in discover_subjects(eng)[:40]:
        ctx = build_ctx(eng, s, f"what is {s}")
        txt = eng._plan_narrative_sentence(ctx)
        if not txt:
            continue
        if re.search(r"\b(a|an) (only|very|often|always|never|rather|just|"
                     r"also|too|now|then|here|there|however|therefore|thus|"
                     r"different|ancient|primary)\b", txt, re.I):
            bad.append((s, txt))
    print(f"\n[pos] plan outputs with a mislabelled target: {len(bad)}")
    for s, t in bad[:10]:
        print(f"       {s}: {t[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
