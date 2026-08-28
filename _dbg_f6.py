import os, sys
os.environ["RAVANA_OFFLINE"]="1"
sys.path.insert(0, "ravana/src")
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_f1_target")
r = eng.process_turn("give me your honest read on the sea versus the land")
print("REPLY:", repr(r))
print("STRAT:", getattr(eng, "_last_strategy", None))
