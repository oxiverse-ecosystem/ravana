"""RED/GREEN regression for stance mining (round v-aug06d).

Exposes two defects the wren persona surfaced:
1. Opinion mining on a QUESTION creates a garbage stance ("letterpress given"
   from "do you still think i love letterpress given the wrist thing?").
2. A retraction naming only part of a multiword stance key does not reverse
   it ("letterpress" should relax "letterpress printing" which stayed +0.95).
"""
import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

def run():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_stance")
    # seed a positive letterpress printing stance
    eng.user_model.mine_personal_facts("i love letterpress printing, the bite of the type is everything")
    st = eng.user_model.opinions.stances
    before = st.get("letterpress printing")
    print(f"seeded letterpress printing polarity={getattr(before,'polarity',None)}")

    # a QUESTION that mentions the stance must NOT create a new garbage stance
    eng.user_model.mine_personal_facts("do you still think i love letterpress given the wrist thing?")
    keys = list(st.keys())
    junk = [k for k in keys if "given" in k or k == "letterpress"]
    print(f"stance keys after question: {keys}")
    fail1 = bool(junk)

    # a retraction naming only 'letterpress' must relax 'letterpress printing'
    eng.user_model.mine_personal_facts("actually, i take it back — i've come to prefer linocut, letterpress is too slow and too precious")
    after = st.get("letterpress printing")
    print(f"letterpress printing polarity after retraction={getattr(after,'polarity',None)}")
    fail2 = not (after is not None and getattr(after, "polarity", 1) < 0.5)

    if fail1 or fail2:
        print(f"RED: fail1(question junk stance)={fail1} fail2(reversal not applied)={fail2}")
        return 1
    print("GREEN: questions don't mine stances; partial-key retraction reverses")
    return 0

if __name__ == "__main__":
    raise SystemExit(run())
