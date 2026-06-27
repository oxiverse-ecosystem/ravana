import sys
import os

_proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))
sys.path.insert(0, _proj_root)

from scripts.ravana_chat import CognitiveChatEngine

print("initializing engine...")
engine = CognitiveChatEngine(dim=64, baby_mode=True)
print("engine loaded.\n")

conversation = [
    "hello",
    "whats up?",
    "what is time",
    "what is ravana",
    "what is oxiverse",
    "what is hebbian learning",
    "tell me about trust",
    "what is privacy",
    "bye"
]

print("==========================================")
print("   CHATTING WITH RAVANA (10 TURNS)")
print("==========================================\n")

for i, turn in enumerate(conversation):
    print(f"You: {turn}")
    response = engine.process_turn(turn)
    strategy = engine._last_strategy
    print(f"RAVANA: {response} [{strategy}]\n")

engine.stop_background_learning()
engine.save()
