#!/usr/bin/env python3
"""Probe the anchor-boost wiring on the production decoder path.

READ-ONLY: builds a persisted engine, builds a ctx exactly as the eval
instrument does, and calls ``_generate_with_situation_model`` with the anchor
boost ON and OFF. Prints both outputs so the effect is visible.

Usage:
    uv run python scripts/probe_anchor_boost.py [subject ...]
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from eval_decoder_groundedness import build_engine, build_ctx  # noqa: E402


def main(subjects):
    eng, label = build_engine("persisted")
    if eng is None:
        print("[probe] no persisted engine")
        return 1
    print(f"[probe] engine={label} use_linggen={getattr(eng, 'use_linggen', None)} "
          f"decoder_built={getattr(eng, '_decoder_vocab_built', None)} "
          f"defs={len(getattr(eng, '_definitions', {}) or {})}")
    print(f"[probe] cpg_enabled={getattr(eng, '_cpg_enabled', None)} "
          f"cpg_escalate={getattr(eng, '_cpg_escalate', None)}")

    for subj in subjects:
        ctx = build_ctx(eng, subj, f"what is {subj}")
        # ON: production config as it now stands.
        out_on = eng._generate_with_situation_model(ctx)
        # OFF: disable the anchor boost by emptying the definition lookup the
        # wiring reads. This is the pre-change behaviour for a subject with a
        # definition, and it is the honest counterfactual.
        saved = None
        try:
            defs = getattr(eng, "_definitions", {}) or {}
            saved = defs.get(subj.lower())
            if saved is not None:
                defs.pop(subj.lower(), None)
            out_off = eng._generate_with_situation_model(ctx)
        finally:
            if saved is not None:
                (getattr(eng, "_definitions", {}) or {})[subj.lower()] = saved

        print(f"\n=== {subj} ===")
        print(f"  defn : {str(saved)[:110]}")
        print(f"  ON  : {out_on}")
        print(f"  OFF : {out_off}")
    return 0


if __name__ == "__main__":
    subs = sys.argv[1:] or ["gravity", "ocean", "species", "time"]
    raise SystemExit(main(subs))
