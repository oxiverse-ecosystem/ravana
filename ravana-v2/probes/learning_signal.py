#!/usr/bin/env python3
"""
🧪 PROBE 3: Learning Signal Test
Track ΔD over time, verify learning vs stagnation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.governor import Governor, GovernorConfig, CognitiveSignals
from core.state import CognitiveState
import numpy as np

def run_learning_probe():
    print("="*60)
    print("🧪 PROBE 3: Learning Signal Test (Trends over time)")
    print("="*60)
    
    config = GovernorConfig()
    governor = Governor(config)
    state = CognitiveState()
    
    print(f"\n🔧 Tracking: ΔD across 200 episodes")
    print(f"   Healthy learning: ΔD fluctuates but system explores")
    print(f"   Stagnation: ΔD ≈ 0 constantly (system frozen)")
    
    dissonance_history = []
    delta_history = []
    mode_history = []
    
    for ep in range(200):
        # Simulate realistic learning curve
        progress = ep / 200
        
        # Decreasing noise as system learns
        noise = 0.1 * (1 - progress * 0.5)
        
        # Improving but noisy reward signal
        reward_quality = 0.3 + (progress * 0.4)
        
        # Build signals based on learning progress
        signals = CognitiveSignals(
            dissonance_delta=np.random.normal(0, noise) - (0.02 if reward_quality > 0.5 else 0),
            identity_delta=0.02 if reward_quality > 0.5 else -0.01,
            exploration_drive=0.2 * (1 - progress),
            resolution_potential=reward_quality,
            confidence=0.4 + progress * 0.4
        )
        
        # Capture before
        d_before = state.dissonance
        
        # Regulate
        result = governor.regulate(
            current_dissonance=state.dissonance,
            current_identity=state.identity,
            signals=signals,
            episode=ep
        )
        
        # Apply
        state.dissonance += result.dissonance_delta
        state.identity += result.identity_delta
        
        delta = state.dissonance - d_before
        
        dissonance_history.append(state.dissonance)
        delta_history.append(delta)
        mode_history.append(result.mode.value)
        
        if ep < 10 or ep % 40 == 0:
            print(f"  [EP{ep:03d}] D:{state.dissonance:.3f} Δ{delta:+.3f} Mode:{result.mode.value:4s}")
    
    # Analyze
    print(f"\n📊 TREND ANALYSIS:")
    
    early_d = dissonance_history[:50]
    late_d = dissonance_history[150:200]
    
    early_mean = np.mean(early_d)
    late_mean = np.mean(late_d)
    trend = late_mean - early_mean
    
    print(f"   Early D (avg): {early_mean:.3f}")
    print(f"   Late D (avg):  {late_mean:.3f}")
    print(f"   Overall trend: {trend:+.3f}")
    
    delta_variance = np.var(delta_history)
    mean_abs_delta = np.mean([abs(d) for d in delta_history])
    
    print(f"\n   Delta variance: {delta_variance:.6f}")
    print(f"   Mean |ΔD|:      {mean_abs_delta:.4f}")
    
    # Mode distribution
    mode_counts = {}
    for m in mode_history:
        mode_counts[m] = mode_counts.get(m, 0) + 1
    print(f"\n   Mode distribution:")
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"      {mode:12s}: {count:3d} ({count/2:.0f}%)")
    
    # Verdict
    near_zero = sum(1 for d in delta_history if abs(d) < 0.001)
    stagnation_ratio = near_zero / len(delta_history)
    
    is_stagnant = stagnation_ratio > 0.7
    is_exploring = mean_abs_delta > 0.01 and delta_variance > 0.0001
    
    print(f"\n✅ VERDICT:")
    if is_stagnant:
        print(f"   ❌ STAGNATION: System frozen ({stagnation_ratio*100:.0f}% near-zero ΔD)")
    elif is_exploring:
        print(f"   ✅ EXPLORATION: Healthy variance, multiple modes")
    else:
        print(f"   ⚠️  MILD: Low activity but not frozen")
    
    return not is_stagnant

if __name__ == "__main__":
    success = run_learning_probe()
    sys.exit(0 if success else 1)
