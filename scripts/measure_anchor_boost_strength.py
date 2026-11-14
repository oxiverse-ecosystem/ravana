#!/usr/bin/env python3
"""Sweep the anchor-boost strength on the production decoder path.

The 9.0 knee was measured in the TRAINING harness (scripts/train.py), whose
decoder call does NOT pass cerebellar_ngram and uses a different temperature
schedule from the production path. This sweep re-measures the dose-response
curve on the PRODUCTION path, so the strength is fitted to the mechanism that
actually runs.

READ-ONLY. Usage:
    uv run python scripts/measure_anchor_boost_strength.py [subject ...]
"""
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from eval_decoder_groundedness import (  # noqa: E402
    build_engine, build_ctx, score_fluency, score_on_topic,
    score_unsupported_assertions, _content_words, _subject_neighbours,
    _definition_words,
)

_STRENGTHS = [0.0, 2.0, 4.0, 6.0, 9.0, 12.0]


def _run(eng, ctx, subj, strength):
    """Generate with the anchor boost forced to ``strength``."""
    import ravana.chat.response_gen as rg
    from ravana.learn import anchor_boost as ab

    saved_const = ab.ANCHOR_TOKEN_BOOST
    saved_fn = ab.anchor_token_boost
    try:
        if strength == 0.0:
            ab.anchor_token_boost = lambda *a, **k: None
        else:
            def _forced(desc, w2i, *, strength=strength, _f=saved_fn):
                return _f(desc, w2i, strength=strength)
            ab.anchor_token_boost = _forced
        out = eng._generate_with_situation_model(ctx)
    finally:
        ab.ANCHOR_TOKEN_BOOST = saved_const
        ab.anchor_token_boost = saved_fn
    return out


def main(subjects):
    eng, label = build_engine("persisted")
    if eng is None:
        return 1
    print(f"[sweep] engine={label}")

    for subj in subjects:
        ctx = build_ctx(eng, subj, f"what is {subj}")
        neigh = _subject_neighbours(eng, subj)
        defw = _definition_words(eng, subj)
        print(f"\n=== {subj} ===  neighbours={len(neigh)} defwords={len(defw)}")
        for s in _STRENGTHS:
            out = _run(eng, ctx, subj, s)
            if not out:
                print(f"  boost={s:>4}: <no output>")
                continue
            text, strat = out
            cw = _content_words(text)
            on_topic = score_on_topic(eng, subj, text)
            unsup = score_unsupported_assertions(eng, subj, text)
            flu = score_fluency(text, subj)
            n_on = on_topic.get("overlap_count", 0)
            n_cw = on_topic.get("content_words", len(cw)) or 1
            print(f"  boost={s:>4}: on={n_on}/{n_cw} "
                  f"({on_topic.get('on_topic_frac', 0.0):.2f}) "
                  f"cos={on_topic.get('mean_cosine', 0.0):.3f} "
                  f"unsup={unsup.get('unsupported_count', 0)} "
                  f"salad={flu.get('is_word_salad')} "
                  f"strat={strat}")
            print(f"           {text[:150]}")
    return 0


if __name__ == "__main__":
    subs = sys.argv[1:] or ["gravity", "ocean", "species", "time"]
    raise SystemExit(main(subs))
