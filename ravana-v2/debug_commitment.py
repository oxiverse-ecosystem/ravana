import sys, os, numpy as np
sys.path.insert(0, '/home/workspace/Projects/ravana-v2')
sys.path.insert(0, '/home/workspace/Projects/ravana-v2/core')

from core.governor import Governor, GovernorConfig, CognitiveSignals
from core.identity import IdentityEngine
from core.resolution import ResolutionEngine
from core.state import StateManager

gov = Governor(GovernorConfig())
res = ResolutionEngine()
ident = IdentityEngine()
sm = StateManager(gov, res, ident)

# Start identity at 0.7 (high commitment state)
sm.state.identity = 0.7
print(f"Initial identity: {sm.state.identity}")
print(f"Initial dissonance: {sm.state.dissonance}")

# Step with correctness=False, difficulty=0.3
step = sm.step(correctness=False, difficulty=0.3)
print(f"\nAfter step(correctness=False, difficulty=0.3):")
print(f"  Final identity: {sm.state.identity:.4f}")
print(f"  Final dissonance: {sm.state.dissonance:.4f}")
print(f"  Delta identity: {sm.state.identity - 0.7:.4f}")
print(f"  Resolution: {step['resolution']}")
