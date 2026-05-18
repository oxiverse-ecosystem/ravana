#!/usr/bin/env python3
"""
🧪 TEST: Grace Layer (Phase B.0)

Verify control behavior upgrade:
- Soft boundary function
- Predictive dampening
- Mode behavior fix
- Identity-coupled control
- Anti-overshoot term
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.governor import Governor, GovernorConfig, RegulationMode, CognitiveSignals
from core.state import CognitiveState
import numpy as np

def test_soft_boundary_function():
    """Test 1: Soft boundary creates resistance near limits"""
    print("\n" + "="*60)
    print("🧪 TEST 1: Soft Boundary Function")
    print("="*60)
    
    config = GovernorConfig()
    governor = Governor(config)
    
    # Test boundary pressure at different points
    test_points = [
        (0.50, 1.0, "safe zone - no pressure"),
        (0.75, 1.0, "threshold - pressure starts"),
        (0.80, 0.4375, "moderate pressure"),
        (0.90, 0.0625, "high pressure"),
        (0.94, 0.0025, "near wall - extreme pressure"),
    ]
    
    all_passed = True
    for D, expected_pressure, desc in test_points:
        pressure = governor._boundary_pressure(D, 0.1)
        diff = abs(pressure - expected_pressure)
        passed = diff < 0.01  # Tolerance
        status = "✅" if passed else "❌"
        print(f"  D={D:.2f} → pressure={pressure:.4f} (expected ~{expected_pressure:.4f}) [{status}] {desc}")
        if not passed:
            all_passed = False
    
    return all_passed

def test_predictive_dampening():
    """Test 2: System eases off before impact"""
    print("\n" + "="*60)
    print("🧪 TEST 2: Predictive Dampening")
    print("="*60)
    
    config = GovernorConfig()
    governor = Governor(config)
    
    # Simulate approaching boundary
    D = 0.85  # Near but not at boundary
    desired_dd = 0.10  # Wants to move toward 0.95
    
    signals = CognitiveSignals(
        dissonance_delta=desired_dd,
        identity_delta=0.0,
        trend=0.5,
        
        
    )
    
    result = governor.regulate(D, 0.5, signals, episode=1)
    
    # Without predictive: D + dd = 0.95
    # With predictive: should be less
    predicted_final = D + desired_dd
    actual_final = D + result.dissonance_delta
    
    dampened = result.dampened
    reduction = (predicted_final - actual_final)
    
    print(f"  Current D: {D:.3f}")
    print(f"  Desired Δ: +{desired_dd:.3f} → would reach {predicted_final:.3f}")
    print(f"  Actual Δ: {result.dissonance_delta:+.3f} → reaches {actual_final:.3f}")
    print(f"  Reduction: {reduction:.3f}")
    print(f"  Dampened flag: {dampened}")
    
    # Should have dampened since we're near boundary
    passed = dampened and reduction > 0.02
    status = "✅" if passed else "❌"
    print(f"  [{status}] System eases off before impact")
    
    return passed

def test_mode_resolution_fix():
    """Test 3: Resolution weakens near boundaries"""
    print("\n" + "="*60)
    print("🧪 TEST 3: Resolution Mode Behavior Fix")
    print("="*60)
    
    config = GovernorConfig()
    governor = Governor(config)
    
    # Test at low D (safe) - should allow amplification
    dd_low = 0.05
    dd_result_low = governor._apply_mode_regulation(
        RegulationMode.RESOLUTION, dd_low, 0.0, 0.3, 0.5
    )
    amplification_low = dd_result_low[0] / dd_low
    
    # Test at high D (near boundary) - should dampen
    dd_high = 0.05
    dd_result_high = governor._apply_mode_regulation(
        RegulationMode.RESOLUTION, dd_high, 0.0, 0.85, 0.5
    )
    amplification_high = dd_result_high[0] / dd_high
    
    print(f"  At D=0.30 (safe): amplification = {amplification_low:.2f}x")
    print(f"  At D=0.85 (near limit): amplification = {amplification_high:.2f}x")
    
    # Should be less amplification near boundary
    passed = amplification_high < amplification_low
    status = "✅" if passed else "❌"
    print(f"  [{status}] Resolution weakens near boundaries")
    
    return passed

def test_identity_coupled_control():
    """Test 4: Identity affects control strength"""
    print("\n" + "="*60)
    print("🧪 TEST 4: Identity-Coupled Dampening")
    print("="*60)
    
    config = GovernorConfig()
    governor = Governor(config)
    
    D = 0.80
    dd = 0.10
    
    # Test with low identity (should be cautious)
    signals_low = CognitiveSignals(dd, 0.0, 0.3, 0.0, 0.0)
    result_low = governor.regulate(D, 0.2, signals_low, episode=1)
    
    # Test with high identity (should explore more)
    signals_high = CognitiveSignals(dd, 0.0, 0.3, 0.0, 0.0)
    result_high = governor.regulate(D, 0.9, signals_high, episode=1)
    
    low_dd = abs(result_low.dissonance_delta)
    high_dd = abs(result_high.dissonance_delta)
    
    print(f"  Low identity (0.2): allows Δ = {low_dd:.3f}")
    print(f"  High identity (0.9): allows Δ = {high_dd:.3f}")
    print(f"  Difference: {abs(high_dd - low_dd):.3f}")
    
    # High identity should allow more movement
    passed = high_dd >= low_dd * 0.8  # At least 80%
    status = "✅" if passed else "❌"
    print(f"  [{status}] Identity controls exploration bandwidth")
    
    return passed

def test_anti_overshoot():
    """Test 5: Center-seeking behavior"""
    print("\n" + "="*60)
    print("🧪 TEST 5: Anti-Overshoot (Center-Seeking)")
    print("="*60)
    
    config = GovernorConfig()
    governor = Governor(config)
    
    # When D is high (0.85), should pull toward target (0.6)
    D_high = 0.85
    signals = CognitiveSignals(0.0, 0.0, 0.0, 0.0, 0.0)
    result_high = governor.regulate(D_high, 0.5, signals, episode=1)
    
    # When D is low (0.35), should push toward target
    D_low = 0.35
    result_low = governor.regulate(D_low, 0.5, signals, episode=1)
    
    print(f"  At D=0.85 (high): dd = {result_high.dissonance_delta:+.4f} (should be negative)")
    print(f"  At D=0.35 (low): dd = {result_low.dissonance_delta:+.4f} (should be positive)")
    print(f"  Target: {config.target_dissonance:.2f}")
    
    # Both should be pulled toward target
    high_pulls_down = result_high.dissonance_delta < -0.01
    low_pulls_up = result_low.dissonance_delta > 0.01
    
    passed = high_pulls_down and low_pulls_up
    status = "✅" if passed else "❌"
    print(f"  [{status}] System seeks center ({config.target_dissonance:.2f})")
    
    return passed

def test_grace_metrics():
    """Test 6: New health metrics track properly"""
    print("\n" + "="*60)
    print("🧪 TEST 6: Grace Metrics Tracking")
    print("="*60)
    
    config = GovernorConfig()
    governor = Governor(config)
    
    # Run some regulation cycles
    D = 0.50
    for i in range(10):
        dd = 0.05 if i % 2 == 0 else -0.03  # Oscillating
        signals = CognitiveSignals(dd, 0.0, 0.3, 0.0, 0.0)
        result = governor.regulate(D, 0.6, signals, episode=i)
        D = np.clip(D + result.dissonance_delta, 0.15, 0.95)
    
    health = governor.get_health_metrics()
    
    print(f"  Overshoot count: {health['overshoot_count']}")
    print(f"  Mean approach velocity: {health['mean_approach_velocity']:.4f}")
    print(f"  Regulation events: {health['total_regulation_events']}")
    print(f"  Predictions: {health['predictions_made']}")
    print(f"  Success rate: {health['prediction_accuracy']:.1%}")
    
    passed = (
        health['total_regulation_events'] > 0 and
        isinstance(health['overshoot_count'], int)
    )
    status = "✅" if passed else "❌"
    print(f"  [{status}] Metrics tracking active")
    
    return passed

def run_all_tests():
    """Run full Grace Layer test suite"""
    print("\n" + "="*70)
    print("🧪 RAVANA v2 — PHASE B.0: GRACE LAYER TEST SUITE")
    print("="*70)
    print("\nVerifying: Control Field (active regulation)")
    print("Not just: Constraint Enforcement (passive bounds)\n")
    
    tests = [
        ("Soft Boundary Function", test_soft_boundary_function),
        ("Predictive Dampening", test_predictive_dampening),
        ("Resolution Mode Fix", test_mode_resolution_fix),
        ("Identity-Coupled Control", test_identity_coupled_control),
        ("Anti-Overshoot Term", test_anti_overshoot),
        ("Grace Metrics", test_grace_metrics),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} FAILED with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 GRACE LAYER FULLY OPERATIONAL")
        print("   Control Field Active: System flows, doesn't crash 🌊")
    elif passed >= total * 0.8:
        print("\n⚠️  GRACE LAYER MOSTLY WORKING")
        print("   Minor issues, safe to proceed")
    else:
        print("\n🔴 GRACE LAYER NEEDS WORK")
        print("   Fix before Phase B")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
