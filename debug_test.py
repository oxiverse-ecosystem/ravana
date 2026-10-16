import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)

from ravana.chat.user_model import UserModel
from ravana.chat.relation_attrs import relation_of

# Test 1: leading modifier "first" should be skipped, "mentor" should be found
um = UserModel()
um.personal_facts.facts.clear()
um.mine_personal_facts("my first mentor Priya taught me astronomy", run_correction=True)
caps = {(a, b): f.value for (a, b, c), f in um.personal_facts.facts.items() if not getattr(f, "superseded", False)}
print("Test 1 - leading modifier:")
print("  facts:", caps)
print("  relation_of(first):", relation_of("first"))
print("  relation_of(mentor):", relation_of("mentor"))

# Test 2: false positive pet on "my pet rock collection"
um2 = UserModel()
um2.personal_facts.facts.clear()
for s in ["my pet rock collection is huge.", "i have a question about the router.", "my dog likes the park."]:
    um2.mine_personal_facts(s, run_correction=True)
pet_attrs = {k[1] for k, f in um2.personal_facts.facts.items()
             if isinstance(k, tuple) and len(k) == 3 and k[0] == "i"
             and k[1] in ("rock", "question", "park", "pet", "dog")}
print("\nTest 2 - false positive pet:")
print("  pet_attrs:", pet_attrs)
print("  all facts:", {(a,b):f.value for (a,b,c),f in um2.personal_facts.facts.items() if not getattr(f,'superseded',False)})

# Test 3: possession_attr recall
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_debug_002")
eng.process_turn("my sword is forged from meteorite iron")
ans = eng.process_turn("what's my sword made of").strip().lower()
print("\nTest 3 - sword recall:")
print("  ans:", repr(ans))

# Check the stored fact
for (s, a, v), f in eng.user_model.personal_facts.facts.items():
    print(f"  fact: ({s}, {a}, {v}) = {f.value}")
