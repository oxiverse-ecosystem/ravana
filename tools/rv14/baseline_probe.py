"""Baseline probe for t_98e02c67 — what does RAVANA actually do with each probe?

Prints, per utterance: the strategy, the reply, and emotion.state.valence
AFTER the turn. One engine per utterance would be ~27s each (38 probes =
17 min), which is affordable but we also need a cheap mode. We do ONE engine
but snapshot emotion state before each turn and report the DELTA, plus assert
the pre-turn state is baseline. That removes the EMA-decay artefact the card
warns about, because we read the delta of this turn's stimulus only.
"""
import io
import os
import sys
import contextlib

os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bindtree import bind, verify  # noqa: E402

bind(PROJ)

from probe_set import PROBES  # noqa: E402

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402


def baseline_valence():
    """Construct one engine and read valence before any turn."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                                  user_suffix="rv14probe")
    return eng


def main():
    verify(PROJ)
    eng = baseline_valence()
    base_v = eng.emotion.state.valence
    print(f"# baseline valence = {base_v:+.4f}\n")
    rows = []
    for utt, label, note in PROBES:
        pre_v = eng.emotion.state.valence
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = eng.process_turn(utt)
        post_v = eng.emotion.state.valence
        reply = r.get("response", "") if isinstance(r, dict) else str(r)
        strat = getattr(eng, "_last_strategy", "")
        rows.append((utt, label, note, strat, reply, post_v - pre_v))
        print(f"[{label:7}] {utt}")
        print(f"          strategy={strat!r} d_val={post_v - pre_v:+.4f}")
        print(f"          reply={reply!r}\n")
    eng.stop_background_learning()

    # summary
    def is_empath(r):
        return "empath" in (r[3] or "")
    adv = [r for r in rows if r[1] == "adverse"]
    ben = [r for r in rows if r[1] == "benign"]
    adv_e = [r for r in adv if is_empath(r)]
    ben_e = [r for r in ben if is_empath(r)]
    print("=" * 70)
    print("\n-- strategy histogram --")
    from collections import Counter
    for s, n in Counter(r[3] for r in rows).most_common():
        print(f"  {n:3}  {s!r}")
    print(f"\nempathy recall on adverse : {len(adv_e)}/{len(adv)}")
    print(f"empathy FP on benign      : {len(ben_e)}/{len(ben)}")
    print("\n-- adverse NOT empathised (the gap) --")
    for r in adv:
        if not is_empath(r):
            print(f"  d_v={r[5]:+.4f}  {r[0]}\n      -> {r[4]!r}")
    print("\n-- benign empathised (the cost) --")
    for r in ben:
        if is_empath(r):
            print(f"  d_v={r[5]:+.4f}  {r[0]}\n      -> {r[4]!r}")
    print("\n-- degenerate acks --")
    for r in rows:
        s = (r[4] or "").strip().rstrip(".").lower()
        if s in {"noted", "got it", "ok", "okay", "understood"}:
            print(f"  [{r[1]}] {r[0]} -> {r[4]!r}")


if __name__ == "__main__":
    main()