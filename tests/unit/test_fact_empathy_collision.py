"""RED regression: benign self-descriptions must NOT be stolen by empathy.

A first-person 'X is Y' disclosure with no suffering word (e.g. 'my favorite
color is ochre', 'i keep a quail named pip') must be stored as a personal fact,
NOT answered with 'feeling lonely is hard'.

This is the rotating-probe regression for round v-aug06d: prior rounds used
dev/soren/mira personas; this one uses wren, which exposed the collision.
"""
import os, sys, io, contextlib
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

def run():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_fact_empathy")
    cases = [
        ("my favorite color is ochre", "ochre"),
        ("i keep a quail named pip", "pip"),
        ("i work as a bookbinder", "bookbinder"),
    ]
    fails = 0
    for q, expect_substr in cases:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = eng.process_turn(q)
        reply = r if isinstance(r, str) else r.get("reply", "")
        facts = {str(k): getattr(v, "value", v) for k, v in eng.user_model.personal_facts.facts.items()}
        flat = " ".join(facts.values()).lower()
        ok = expect_substr.lower() in flat
        empathized = "feeling lonely is hard" in reply or "here for it" in reply
        status = "OK" if (ok and not empathized) else "FAIL"
        if status == "FAIL":
            fails += 1
        print(f"[{status}] U={q!r} reply='{reply[:70]}' facts_have_{expect_substr}={ok}")
    if fails:
        print(f"RED: {fails} benign disclosures were stolen by empathy")
        return 1
    print("GREEN: benign self-descriptions stored as facts, not empathized")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
