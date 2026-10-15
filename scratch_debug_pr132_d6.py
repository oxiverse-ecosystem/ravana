import os, sys, io, contextlib
os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src", f"{PROJ}\\ravana-v2\\src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(
    dim=64, seed=42, baby_mode=True,
    user_suffix="d6_test_" + str(os.getpid()))
eng.process_turn("i live on a houseboat moored in a quiet lagoon.")
eng.process_turn("i have a pet crow named mica.")

queries = [
    ("what do you make of my lutefisk habit now?", "my"),
    ("how do you feel about that, me leaving the water?", "me"),
    ("do you think i'm contradictory, or just changing?", "i'm"),
    ("what's your take on my brother tomas's motorcycles?", "my"),
    ("how do you feel about us moving away?", "us"),
]

for query, bad_tail in queries:
    r = eng.process_turn(query)
    r_l = r.lower() if r else ""
    _parts = r_l.split("view on")[1:]
    ok = True
    if not _parts:
        print(f"[FAIL] no 'view on' found for {query!r}: {r!r}")
        continue
    for _slot in _parts:
        _words = _slot.strip().split()
        if _words:
            _topic = _words[0].rstrip(".,;:!?'\"")
            if bad_tail == _topic:
                print(f"[FAIL] pronoun {bad_tail!r} leaked for {query!r}: {r!r}")
                ok = False
    if ok:
        print(f"[OK] {query!r}")
