import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.retirement import RetirementLedger, looks_like_retraction

led = RetirementLedger()
print("all_markers:", led.all_markers())
print("learned_markers:", led.learned_markers)
result = looks_like_retraction("patna, franks he lives in delhi", led.all_markers())
print("looks_like_retraction:", result)
