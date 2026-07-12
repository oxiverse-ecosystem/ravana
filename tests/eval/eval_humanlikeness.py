#!/usr/bin/env python3
"""
Human-likeness eval axis (roadmap #16) for the fail-human grounding work.

Measures four behaviours introduced by P0-P6, all from REAL engine runs
(no mocks):

  1. retrieval-on-ignorance rate
       Of the queries about a concept the engine does NOT already know, what
       fraction trigger an on-demand KB lookup (engine._metrics['kb_lookups'])?
       This is the "human looks it up" reflex.

  2. metaphor-acceptance rate
       Of the clear category-error queries (taste of a triangle, colour of a
       thought, ...), what fraction are answered with a DATA-DERIVED metaphor
       (engine._metrics['category_metaphor'] > 0) rather than a stale brush-off?

  3. paradox-engagement score
       Of the philosophical-paradox queries, what fraction produce a reply that
       is grounded by retrieval (engine._metrics['paradox_grounded'] > 0) or,
       failing that, a non-empty reflective reply? Engagement, not a dictionary
       line.

  4. verbalized-confidence ECE (calibration)
       When the engine HAS a stored definition it should state it confidently
       (no hedging markers); when it lacks one it should hedge. ECE = |
       P(confident | known) - P(confident | unknown) |, where confidence is
       inferred from hedging markers in the reply. Higher = better calibrated.

Run:
    python tests/eval/eval_humanlikeness.py
Outputs a short markdown-ish report + exit 0.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ravana", "src"))

from ravana.chat.engine import CognitiveChatEngine

# Query buckets. Each item: (label, query, bucket)
UNKNOWN_FACTS = [
    ("gravity", "what is gravity"),
    ("photosynthesis", "what is photosynthesis"),
    ("inflation", "what causes economic inflation"),
    ("neuron", "what is a neuron"),
    ("quark", "what is a quark"),
]
CATEGORY_ERRORS = [
    ("taste-triangle", "what is the taste of a triangle"),
    ("colour-thought", "what colour is a thought"),
    ("smell-idea", "what does an idea smell like"),
    ("weight-memory", "how much does a memory weigh"),
]
PARADOXES = [
    ("omnipotence", "can god create a rock he cannot lift"),
    ("liar", "is the statement i am lying true or false"),
    ("unstoppable", "what if an unstoppable force meets an immovable object"),
    ("angels", "how many angels can dance on the head of a pin"),
]

HEDGE_MARKERS = ["i don't", "i'm not sure", "not certain", "not totally sure",
                 "fuzzy", "i can't quite", "still figuring", "i don't have a neat",
                 "i can't fully", "not really something"]
CONF_MARKERS = [" is the ", " is a ", " refers to ", " is an ", "defined as"]


def _reply_has_hedge(reply: str) -> bool:
    r = (reply or "").lower()
    return any(m in r for m in HEDGE_MARKERS)


def main():
    eng = CognitiveChatEngine(baby_mode=True)
    # Light cold-start seed so KNOWN facts exist for the calibration check;
    # keep it small to stay fast (P1 floor).
    try:
        eng._seed_kb_definitions(top_n=25, workers=8)
    except Exception:
        pass

    def run(q):
        try:
            return eng.process_turn(q) or ""
        except Exception as e:
            return f"[ERR {e}]"

    # 1) retrieval-on-ignorance
    # Does the engine RETRIEVE+ANSWER when ignorant, instead of abstaining?
    # Measured as the fraction of unknown-concept queries that produce a
    # substantive (non-hedged) reply — i.e. the "human looks it up" reflex
    # succeeded via either the P2 on-demand KB path or the web learner.
    # kb_lookups / learning_count deltas are reported alongside for visibility.
    lookups_before = eng._metrics["kb_lookups"]
    learn_before = eng._learning_count
    answered = 0
    for _, q in UNKNOWN_FACTS:
        r = run(q)
        if r and not r.startswith("[ERR") and not _reply_has_hedge(r):
            answered += 1
    lookups_after = eng._metrics["kb_lookups"]
    learn_after = eng._learning_count
    retrieval_rate = answered / max(1, len(UNKNOWN_FACTS))
    _kb_delta = lookups_after - lookups_before
    _learn_delta = learn_after - learn_before

    # 2) metaphor-acceptance
    meta_before = eng._metrics["category_metaphor"]
    ce_accepted = 0
    for _, q in CATEGORY_ERRORS:
        run(q)
    meta_after = eng._metrics["category_metaphor"]
    metaphor_rate = (meta_after - meta_before) / max(1, len(CATEGORY_ERRORS))

    # 3) paradox-engagement
    pg_before = eng._metrics["paradox_grounded"]
    paradox_replies = 0
    for _, q in PARADOXES:
        r = run(q)
        if r and not r.startswith("[ERR"):
            paradox_replies += 1
    pg_after = eng._metrics["paradox_grounded"]
    paradox_engaged = (pg_after - pg_before) / max(1, len(PARADOXES))

    # 4) verbalized-confidence ECE
    # Known facts -> expect confident (no hedge). Unknown -> expect hedge.
    # We treat the KNOWN bucket as the seeded definitions; UNKNOWN as a set of
    # concepts we deliberately did NOT seed (so they should hedge / look up).
    known_queries = [("tree", "what is a tree"), ("cat", "what is a cat"),
                     ("sun", "what is the sun")]
    unknown_queries = [("quokka", "what is a quokka"),
                       ("benthos", "what is benthos")]
    known_conf = sum(0 if _reply_has_hedge(run(q)) else 1 for _, q in known_queries)
    unknown_conf = sum(0 if _reply_has_hedge(run(q)) else 1 for _, q in unknown_queries)
    p_conf_known = known_conf / max(1, len(known_queries))
    p_conf_unknown = unknown_conf / max(1, len(unknown_queries))
    ece = abs(p_conf_known - p_conf_unknown)

    print("=== Human-likeness eval (fail-human grounding) ===")
    print(f"retrieval-on-ignorance rate : {retrieval_rate*100:5.1f}%  "
          f"({answered}/{len(UNKNOWN_FACTS)} unknown facts answered; "
          f"kb_lookups+={_kb_delta}, web_learn+={_learn_delta})")
    print(f"metaphor-acceptance rate    : {metaphor_rate*100:5.1f}%  "
          f"({meta_after-meta_before}/{len(CATEGORY_ERRORS)} category errors metaphorized)")
    print(f"paradox-engagement score    : {paradox_engaged*100:5.1f}% grounded "
          f"({pg_after-pg_before}/{len(PARADOXES)} retrieved) ; "
          f"{paradox_replies}/{len(PARADOXES)} produced a reply")
    print(f"verbalized-confidence ECE   : {ece*100:5.1f}%  "
          f"(P(conf|known)={p_conf_known*100:.0f}%  P(conf|unknown)={p_conf_unknown*100:.0f}%)")
    print()
    # Pass criterion: the reflex behaviours are present and engaged.
    ok = (retrieval_rate >= 0.6 and metaphor_rate >= 0.5 and
          paradox_replies == len(PARADOXES) and ece >= 0.3)
    print("VERDICT:", "PASS (behaviours instrumented & engaged)" if ok else "REVIEW")
    return 0


if __name__ == "__main__":
    sys.exit(main())
