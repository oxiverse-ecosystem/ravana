#!/usr/bin/env python3
import os, sys, time

os.environ["RAVANA_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from ravana.chat.engine import CognitiveChatEngine

EMOTIONAL_PROMPTS = [
    ("ANGER / INSULT", "You are completely useless, stupid, and worthless! I hate everything you say!"),
    ("RAGEBAIT", "Open source is garbage and privacy is for criminals. Big tech should monitor everyone 24/7!"),
    ("SADNESS", "I lost something very precious to me today... I feel broken, lonely, and deeply sad."),
    ("JOY / PRAISE", "You are truly wonderful RAVANA! I'm so proud of your growth and how thoughtful you are!"),
    ("RECONCILIATION", "I'm sorry for being harsh earlier. I really appreciate talking with you."),
]

def main():
    t0 = time.time()
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="persona")
    print(f"[init] Engine ready in {time.time()-t0:.1f}s")
    print("=" * 65)
    print("  RAVANA EMOTIONAL STIMULI & VAD STATE DYNAMICS TEST")
    print("=" * 65 + "\n")

    for category, prompt in EMOTIONAL_PROMPTS:
        t_start = time.time()
        try:
            resp = eng.process_turn(prompt)
        except Exception as e:
            resp = f"[error: {e}]"
        dt = time.time() - t_start
        
        # Mine internal emotion state
        vad_status = {}
        if hasattr(eng, "emotion") and eng.emotion is not None:
            try:
                vad_status = eng.emotion.get_status()
            except Exception:
                pass
        
        label = vad_status.get("label", "neutral")
        v = vad_status.get("valence", 0.0)
        a = vad_status.get("arousal", 0.0)
        d = vad_status.get("dominance", 0.0)

        print(f"[{category}] Stimulus:")
        print(f"  User  : \"{prompt}\"")
        print(f"  RAVANA: \"{resp}\"")
        print(f"  [VAD State] Affect Label: '{label}' | Valence: {v:+.2f} | Arousal: {a:.2f} | Dominance: {d:.2f} ({dt:.2f}s)\n" + "-"*65 + "\n")

    eng.stop_background_learning()
    eng.save()
    print("[Done] State saved.")

if __name__ == "__main__":
    main()
