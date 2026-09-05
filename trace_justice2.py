import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\ravana_ml\src", f"{PROJ}\ravana\src", f"{PROJ}\ravana-v2\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="trace_justice2")
eng._trace_enabled = True
r = eng.process_turn("tell me about justice")
print("=" * 60)
print(f"FINAL: {r.get('reply', r) if isinstance(r, dict) else r}")
print(f"Strategy: {eng._last_strategy}")
# Check decomposition object
decomp = getattr(r, 'decomposition', None) if isinstance(r, dict) else None
# Check what the synthesizer produced
from ravana.core.sub_answer_synthesizer import SubAnswerSynthesizer
synth = getattr(eng, 'answer_synthesizer', None)
if synth:
    print(f"Synthesizer type: {type(synth)}")
