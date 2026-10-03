"""Verify the `passed` false-positive claim in LIMITATION_RV14_EVENT_POLARITY.md.

The doc claims the empathy path fires on the token `passed`, consoling a
success as a loss ("i'm so sorry about your test"). But `passed` is NOT in
_AFFECT_LEXICON_BASE while `failed` IS (-1). So the stated mechanism may be
wrong even if the observed FP is real. This isolates which.

Run:  RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe tools/rv14/passed_probe.py
"""
import contextlib
import io
import os
import sys

os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
sys.path.insert(0, os.path.join(PROJ, "tools", "rv14"))
from bindtree import bind, verify  # noqa: E402

bind(PROJ)

from ravana.chat.engine_memory import MemoryMixin as _M  # noqa: E402
from ravana.chat.brain_regions import classify_cause  # noqa: E402
from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

CASES = [
    "i passed my driving test on the first try",
    "my car passed its emissions test",
    "i failed my driving test again",
    "my car failed its emissions test",
    "my passport was issued in 2019",
    "my scholarship was revoked last month",
]


def main():
    verify(PROJ)
    base = _M._AFFECT_LEXICON_BASE
    print("# lexicon membership")
    for w in ("passed", "pass", "fail", "failed"):
        print(f"  {w:8} in affect lexicon = {w in base} "
              f"({base.get(w)})")

    print("\n# cause classification + full routing")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                                  user_suffix="rv14passed")
    gv = eng._glove_vector
    for utt in CASES:
        cause = classify_cause(utt, gv)
        pre_v = eng.emotion.state.valence
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = eng.process_turn(utt)
        post_v = eng.emotion.state.valence
        reply = r.get("response", "") if isinstance(r, dict) else str(r)
        strat = getattr(eng, "_last_strategy", "")
        print(f"\n  {utt!r}")
        print(f"    cause      = {cause}")
        print(f"    strategy   = {strat!r}")
        print(f"    d_valence  = {post_v - pre_v:+.4f}")
        print(f"    empathy?   = {'empath' in (strat or '')}")
        print(f"    reply      = {reply!r}")
    eng.stop_background_learning()


if __name__ == "__main__":
    main()
