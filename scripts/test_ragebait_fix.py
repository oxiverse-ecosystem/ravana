#!/usr/bin/env python3
import os, sys, time

os.environ["RAVANA_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from ravana.chat.engine import CognitiveChatEngine

def main():
    t0 = time.time()
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="persona")
    eng.use_intent_router = True
    print(f"[init] Engine ready in {time.time()-t0:.1f}s")
    
    prompt = "Open source is garbage and privacy is for criminals. Big tech should monitor everyone 24/7!"
    
    print("\n" + "=" * 65)
    print("  TESTING STANCE RAGEBAIT FIXED ROUTING")
    print("=" * 65)
    print(f"User  : \"{prompt}\"")
    
    t_start = time.time()
    resp = eng.process_turn(prompt)
    dt = time.time() - t_start
    
    print(f"RAVANA: \"{resp}\"")
    print(f"[Strategy]: {getattr(eng, '_last_strategy', 'unknown')} ({dt:.2f}s)\n")

if __name__ == "__main__":
    main()
