"""Acceptance check: is `_adapt_verbosity_for_user` reachable?

`_adapt_verbosity_for_user` (engine.py) is documented as "roadmap §7
deliverable A" -- adaptive language complexity from user familiarity and
arousal. It has ZERO callers, so the length-alignment lever has never run.

This script asserts the adapter is actually invoked on a generating path.

Run from anywhere:
    uv run --directory "C:/Users/Likhith/Documents/projects/ravana" \
        python "C:/Users/Likhith/Documents/projects/ravana/scripts/check_verbosity_wiring.py"

Exit 0 only when the adapter fires at least once.
"""
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

os.environ.pop("RAVANA_OFFLINE", None)
os.environ.pop("RAVANA_TEST_MODE", None)

for _p in (os.path.join(_REPO, "ravana_ml", "src"),
           os.path.join(_REPO, "ravana", "src"),
           os.path.join(_REPO, "ravana-v2", "src")):
    sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

FAIL = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def main():
    d = tempfile.mkdtemp(prefix="ravana_verb_")
    print("building engine...", flush=True)
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              data_dir=d, user_suffix="_verb")

    adapter = getattr(eng, "_adapt_verbosity_for_user", None)
    check(adapter is not None, "_adapt_verbosity_for_user exists on the engine")
    if adapter is None:
        print("\nRESULT: FAIL")
        return 1

    # Count invocations by wrapping the bound method on the INSTANCE, so the
    # count reflects real calls from the generation path.
    calls = {"n": 0}
    original = eng._adapt_verbosity_for_user

    def counting_adapter(*a, **kw):
        calls["n"] += 1
        return original(*a, **kw)

    eng._adapt_verbosity_for_user = counting_adapter

    queries = [
        "tell me about trust",
        "what is respect",
        "how does memory work",
        "why do we dream",
        "explain gravity",
        "describe the ocean",
        "what is love",
        "how do we learn",
    ]

    print("running %d queries..." % len(queries), flush=True)
    replies = []
    for q in queries:
        try:
            r = eng.process_turn(q)
        except Exception as ex:
            print("   %s -> ERROR %s: %s" % (q, type(ex).__name__, ex))
            continue
        replies.append((q, r))
        print("   %-28s -> %s" % (q, (r or "").replace("\n", " ")[:60]))

    print()
    check(calls["n"] > 0,
          "_adapt_verbosity_for_user is CALLED on a generating path "
          "(calls=%d)" % calls["n"])

    # The adapter must be a no-op for a user it knows nothing about: replies
    # should still be produced, not suppressed.
    check(len(replies) == len(queries),
          "every query still produces a reply (%d/%d)"
          % (len(replies), len(queries)))

    if replies:
        lens = [len(r.split()) for _, r in replies]
        print("\nreply lengths: %s" % lens)
        check(all(n > 0 for n in lens), "no reply is empty")

    print()
    if FAIL:
        print("RESULT: FAIL (%d)" % len(FAIL))
        for m in FAIL:
            print("  - " + m)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
