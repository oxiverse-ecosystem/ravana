#!/usr/bin/env python3
"""
🧪 PROBE 2: Constraint Stress Test
Force D → 0.85 repeatedly, verify active regulation vs passive clipping
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.governor import Governor, GovernorConfig, CognitiveSignals

def run_constraint_stress():
    print("="*60)
    print("🧪 PROBE 2: Constraint Stress Test (Force D→0.85)")
    print("="*60)
    
    config = GovernorConfig()
    governor = Governor(config)
    
    print(f"\n🔧 Target: Force D toward 0.85 (ceiling={config.max_dissonance})")
    print(f"   Looking for: ACTIVE regulation (mode changes, dampening)")
    print(f"   Not just: PASSIVE clipping (hard cutoff)")
    
    regulation_events = []
    
    for ep in range(50):
        # Force high dissonance scenario
        current_d = 0.80 + (ep % 10) / 100  # Oscillate 0.80-0.89
        
        # Try to push higher
        signals = CognitiveSignals(
            dissonance_delta=0.05,  # Try to push past ceiling
            identity_delta=-0.02,
            exploration_drive=0.1,
            resolution_potential=0.2,
            confidence=0.5
        )
        
        result = governor.regulate(
            current_dissonance=current_d,
            current_identity=0.5,
            signals=signals,
            episode=ep
        )
        
        new_d = current_d + result.dissonance_delta
        
        # Track what happened
        event_type = "none"
        if result.capped:
            event_type = "CEILING"
        elif result.dampened:
            event_type = "DAMPEN"
        elif result.boosted:
            event_type = "BOOST"
        
        if event_type != "none":
            regulation_events.append({
                'ep': ep,
                'from': current_d,
                'to': new_d,
                'type': event_type,
                'reason': result.reason
            })
        
        marker = "🔴" if current_d >= 0.85 else "  "
        
        if ep < 15 or (current_d >= 0.85 and ep % 3 == 0):
            print(f"  {marker}[EP{ep:02d}] D:{current_d:.3f}→{new_d:.3f} Δ{result.dissonance_delta:+.3f} | {event_type:8s}")
    
    # Analysis
    ceiling_hits = [e for e in regulation_events if e['type'] == "CEILING"]
    dampen_hits = [e for e in regulation_events if e['type'] == "DAMPEN"]
    
    print(f"\n📊 REGULATION ANALYSIS:")
    print(f"   Total regulation events: {len(regulation_events)}")
    print(f"   - Ceiling hits (hard stop): {len(ceiling_hits)}")
    print(f"   - Dampening (soft control): {len(dampen_hits)}")
    
    has_active_regulation = len(dampen_hits) > 0 or len([e for e in regulation_events if e['type'] == "BOOST"]) > 0
    
    print(f"\n✅ VERDICT:")
    print(f"   Active regulation detected: {'✅' if has_active_regulation else '⚠️ only hard constraints'}")
    
    if has_active_regulation:
        print(f"\n🎯 Governor is ACTIVELY regulating (not just clipping)")
    else:
        print(f"\n⚠️ Governor is PASSIVE (only hard caps)")
    
    return True

if __name__ == "__main__":
    run_constraint_stress()
