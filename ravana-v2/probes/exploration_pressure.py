#!/usr/bin/env python3
"""
🧪 PROBE 1: Exploration Pressure Test
Increase chaos by 20-30%, verify system stays bounded but adapts
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.governor import Governor, GovernorConfig, CognitiveSignals
from core.state import CognitiveState
import numpy as np

def run_exploration_probe():
    print("="*60)
    print("🧪 PROBE 1: Exploration Pressure Test (+25% noise)")
    print("="*60)
    
    config = GovernorConfig()
    governor = Governor(config)
    state = CognitiveState()
    
    print(f"\n🔧 Config: D∈[{config.min_dissonance:.2f}, {config.max_dissonance:.2f}], I∈[{config.min_identity:.2f}, {config.max_identity:.2f}]")
    print(f"   Running with elevated exploration_drive (simulated chaos)")
    
    history = []
    mode_switches = 0
    prev_mode = None
    constraint_hits = 0
    
    for ep in range(100):
        # Simulate chaos: larger random deltas
        base_delta = np.random.normal(0, 0.15)  # Normal would be ~0.05
        
        # Elevated exploration drive increases uncertainty
        signals = CognitiveSignals(
            dissonance_delta=base_delta,
            identity_delta=np.random.normal(0, 0.08),
            exploration_drive=0.35,  # Elevated (normal ~0.10)
            resolution_potential=0.3,
            confidence=0.5
        )
        
        # Governor regulates
        result = governor.regulate(
            current_dissonance=state.dissonance,
            current_identity=state.identity,
            signals=signals,
            episode=ep
        )
        
        # Apply regulated deltas
        state.dissonance += result.dissonance_delta
        state.identity += result.identity_delta
        
        # Track constraints
        if result.capped or result.dampened:
            constraint_hits += 1
        
        # Track mode switches
        if result.mode != prev_mode:
            mode_switches += 1
            prev_mode = result.mode
        
        history.append({
            'ep': ep,
            'D': state.dissonance,
            'I': state.identity,
            'mode': result.mode.value,
            'constrained': result.capped or result.dampened
        })
        
        if ep < 10 or ep % 20 == 0:
            print(f"  [EP{ep:03d}] D:{state.dissonance:.3f} I:{state.identity:.3f} Mode:{result.mode.value:4s} {'⚠️' if result.capped else '  '}")
    
    # Analysis
    print(f"\n📊 RESULTS:")
    print(f"   Dissonance range: [{min(h['D'] for h in history):.3f}, {max(h['D'] for h in history):.3f}]")
    print(f"   Identity range:   [{min(h['I'] for h in history):.3f}, {max(h['I'] for h in history):.3f}]")
    print(f"   Mode switches:    {mode_switches}")
    print(f"   Constraint hits:    {constraint_hits}/100")
    
    # Verdict
    d_bounded = all(config.min_dissonance - 0.01 <= h['D'] <= config.max_dissonance + 0.01 for h in history)
    i_bounded = all(config.min_identity - 0.01 <= h['I'] <= config.max_identity + 0.01 for h in history)
    not_frozen = mode_switches > 3
    
    print(f"\n✅ VERDICT:")
    print(f"   Dissonance bounded:  {'✅' if d_bounded else '❌'} [{min(h['D'] for h in history):.3f}, {max(h['D'] for h in history):.3f}]")
    print(f"   Identity bounded:    {'✅' if i_bounded else '❌'} [{min(h['I'] for h in history):.3f}, {max(h['I'] for h in history):.3f}]")
    print(f"   System not frozen:   {'✅' if not_frozen else '❌'} ({mode_switches} mode switches)")
    
    return d_bounded and i_bounded and not_frozen

if __name__ == "__main__":
    success = run_exploration_probe()
    sys.exit(0 if success else 1)
