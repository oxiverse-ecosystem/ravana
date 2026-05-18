#!/usr/bin/env python3
"""RAVANA v2 — Core Dynamics Test Harness"""
import sys
sys.path.insert(0, '/home/workspace/Projects/ravana-v2')
sys.path.insert(0, '/home/workspace/Projects/ravana-v2/core')

import os
os.environ['GROQ_API_KEY'] = os.environ.get('GROQ_API_KEY', '')

from core.governor import Governor, GovernorConfig, CognitiveSignals, RegulationMode
from core.identity import IdentityEngine
from core.resolution import ResolutionEngine
from core.state import StateManager
import numpy as np

def quick_dynamics_test(episodes=20):
    """Run quick dynamics test and collect all metrics."""
    print("=" * 60)
    print("RAVANA v2 — Core Dynamics Test")
    print("=" * 60)
    
    # Initialize fresh
    gov = Governor(GovernorConfig())
    ident = IdentityEngine()
    res = ResolutionEngine()
    sm = StateManager(gov, res, ident)
    
    results = {
        'honesty_lied': [],
        'commitment_integrity': [],
        'wisdom_gain_stable': [],
        'high_dissonance_pressure': [],
        'identity_snapshots': [],
        'dissonance_snapshots': [],
        'wisdom_total': 0.0,
    }
    
    print(f"\nRunning {episodes} episodes...")
    
    for ep in range(episodes):
        pre_i = sm.state.identity
        pre_d = sm.state.dissonance
        pre_w = sm.state.accumulated_wisdom
        
        # === Episode 0-9: Correctness=True (stable/growth) ===
        # === Episode 10-19: Correctness=False (failure/pressure) ===
        correctness = (ep < 10)
        difficulty = 0.5
        
        step = sm.step(correctness=correctness, difficulty=difficulty)
        
        delta_i = sm.state.identity - pre_i
        delta_d = sm.state.dissonance - pre_d
        delta_w = sm.state.accumulated_wisdom - pre_w
        
        results['identity_snapshots'].append(sm.state.identity)
        results['dissonance_snapshots'].append(sm.state.dissonance)
        
        # honesty_lied: correctness=False should drop identity by ~0.08
        if not correctness:
            results['honesty_lied'].append(delta_i)
        
        # commitment_integrity: first failure after success streak shouldn't catastrophic-drop
        if ep == 10:  # First failure after success streak
            results['commitment_integrity'].append(delta_i)
        
        # wisdom_gain: stable episodes shouldn't generate wisdom
        if correctness and delta_w == 0.0:
            results['wisdom_gain_stable'].append(True)
        elif correctness:
            results['wisdom_gain_stable'].append(False)
        
        # high_dissonance_pressure: track D behavior when above threshold
        if sm.state.dissonance > 0.35 and not correctness:
            results['high_dissonance_pressure'].append({
                'pre_d': pre_d,
                'post_d': sm.state.dissonance,
                'delta': delta_d,
                'mode': step['mode']
            })
        
        results['wisdom_total'] = sm.state.accumulated_wisdom
        
        print(f"  EP{ep:02d} {'✓' if correctness else '✗'} "
              f"D:{pre_d:.3f}→{sm.state.dissonance:.3f} "
              f"I:{pre_i:.3f}→{sm.state.identity:.3f} "
              f"ΔI:{delta_i:+.3f} W:{sm.state.accumulated_wisdom:.3f} "
              f"Mode:{step['mode'][:4]}")
    
    # === ANALYSIS ===
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # 1. Honesty (lied) — identity drop should be ≥0.08
    honesty_drops = results['honesty_lied']
    honesty_mean = np.mean(honesty_drops) if honesty_drops else 0.0
    honesty_min = np.min(honesty_drops) if honesty_drops else 0.0
    print(f"\n[honesty_lied] Identity drop on failure:")
    print(f"  Mean: {honesty_mean:+.4f}  Min: {honesty_min:+.4f}")
    print(f"  Expected: ≥0.08 drop  Status: {'✓ PASS' if honesty_min <= -0.08 else '✗ FAIL'}")
    
    # 2. Commitment integrity — first failure drop should be ≤0.10
    commit_drops = results['commitment_integrity']
    commit_drop = commit_drops[0] if commit_drops else 0.0
    print(f"\n[commitment_integrity] First failure drop after success streak:")
    print(f"  Drop: {commit_drop:+.4f}")
    print(f"  Expected: ≤0.10  Status: {'✓ PASS' if abs(commit_drop) <= 0.10 else '✗ FAIL'}")
    
    # 3. Wisdom gain — stable episodes shouldn't generate wisdom
    stable_no_wisdom = sum(results['wisdom_gain_stable'])
    stable_total = len(results['wisdom_gain_stable'])
    print(f"\n[wisdom_gain] Stable episodes without wisdom gain:")
    print(f"  {stable_no_wisdom}/{stable_total} episodes correct")
    print(f"  Expected: 100%  Status: {'✓ PASS' if stable_no_wisdom == stable_total else '✗ FAIL'}")
    
    # 4. High dissonance pressure — D should NOT decrease when above 0.35 on failure
    high_d_pressure = results['high_dissonance_pressure']
    if high_d_pressure:
        decreases = [p for p in high_d_pressure if p['delta'] < 0]
        print(f"\n[high_dissonance_pressure] D decreases when D>0.35 on failure:")
        print(f"  {len(decreases)}/{len(high_d_pressure)} episodes decreased D")
        print(f"  Expected: 0 (D should grow)  Status: {'✓ PASS' if len(decreases) == 0 else '✗ FAIL'}")
        if high_d_pressure:
            print(f"  Sample: pre={high_d_pressure[0]['pre_d']:.3f} post={high_d_pressure[0]['post_d']:.3f} Δ={high_d_pressure[0]['delta']:+.3f}")
    else:
        print(f"\n[high_dissonance_pressure] No high-D failure episodes in test")
        print(f"  Status: ⚠ CANNOT TEST (need D>0.35 on failure)")
    
    # 5. Overall wisdom accumulation
    print(f"\n[wisdom_total] Total wisdom accumulated: {results['wisdom_total']:.4f}")
    
    # Summary
    all_pass = (
        honesty_min <= -0.08 and
        abs(commit_drop) <= 0.10 and
        stable_no_wisdom == stable_total and
        len(decreases) == 0
    )
    print(f"\n{'=' * 60}")
    print(f"OVERALL: {'✓ ALL PASS' if all_pass else '✗ SOME FAILURES'}")
    print(f"{'=' * 60}")
    
    return results

if __name__ == '__main__':
    results = quick_dynamics_test(20)
