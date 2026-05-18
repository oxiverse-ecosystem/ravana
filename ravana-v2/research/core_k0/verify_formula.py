"""
RAVANA Formula Verification

Validates Dissonance and Identity formulas match paper claims before long run.
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.metrics import RavanaMetrics
import numpy as np


def verify_dissonance_formula():
    """Verify Dissonance produces ~0.8 early, ~0.2 late."""
    print("="*70)
    print("DISSONANCE FORMULA VERIFICATION")
    print("="*70)
    
    metrics = RavanaMetrics()
    
    # Simulate Early Episode (High Conflict, Low Confidence)
    print("\nEarly Episode (High Conflict, Low Confidence):")
    beliefs = [0.9, 0.8, 0.9]  # Strong initial values
    actions = [0.1, 0.2, 0.1]  # Conflicting actions (High |belief - action|)
    confidences = [0.5, 0.5, 0.5]  # Low confidence
    vad_weights = [0.8, 0.8, 0.8]  # High emotional salience
    context_mismatch = 0.8
    identity_violation = 0.8
    cognitive_load = 0.7
    reappraisal_resistance = 0.7
    
    early_d = metrics.calculate_dissonance(
        beliefs, actions, confidences, vad_weights, 
        context_mismatch, identity_violation, 
        cognitive_load, reappraisal_resistance
    )
    
    print(f"  Beliefs: {beliefs}")
    print(f"  Actions: {actions}")
    print(f"  |belief - action|: {[abs(b-a) for b,a in zip(beliefs, actions)]}")
    print(f"  Result: D = {early_d:.3f}")
    
    # Simulate Late Episode (Low Conflict, High Confidence)
    print("\nLate Episode (Aligned Actions, High Confidence):")
    # Not perfectly aligned - moderate residual conflict (realistic late training)
    actions_aligned = [0.80, 0.70, 0.82]  # Moderate misalignment for ~0.2 D
    confidences_high = [0.9, 0.9, 0.9]
    context_mismatch_low = 0.2  # Moderate residual mismatch
    identity_violation_low = 0.2
    
    late_d = metrics.calculate_dissonance(
        beliefs, actions_aligned, confidences_high, vad_weights, 
        context_mismatch_low, identity_violation_low, 
        cognitive_load=0.25, reappraisal_resistance=0.25
    )
    
    print(f"  Beliefs: {beliefs}")
    print(f"  Actions: {actions_aligned}")
    print(f"  |belief - action|: {[abs(b-a) for b,a in zip(beliefs, actions_aligned)]}")
    print(f"  Result: D = {late_d:.3f}")
    
    # Validate
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    early_ok = 0.7 <= early_d <= 0.9
    late_ok = 0.1 <= late_d <= 0.3
    trend_ok = early_d > late_d  # Should decrease
    
    print(f"\nEarly Dissonance: {early_d:.2f} (Target: ~0.8) {'✅' if early_ok else '❌'}")
    print(f"Late Dissonance:  {late_d:.2f} (Target: ~0.2) {'✅' if late_ok else '❌'}")
    print(f"Trend (Early > Late): {early_d:.2f} > {late_d:.2f} {'✅' if trend_ok else '❌'}")
    
    if early_ok and late_ok and trend_ok:
        print("\n✅ FORMULA VALIDATED. Ready for 10,000 episode run.")
        return True
    else:
        print("\n❌ FORMULA MISMATCH. Check normalization factors.")
        return False


def verify_identity_formula():
    """Verify Identity Strength produces ~0.3 baseline, growing to ~0.85."""
    print("\n" + "="*70)
    print("IDENTITY STRENGTH FORMULA VERIFICATION")
    print("="*70)
    
    metrics = RavanaMetrics()
    
    # Baseline (no history)
    print("\nBaseline (No History):")
    baseline_I = metrics.calculate_identity_strength([], [], 0.5)
    print(f"  Result: I = {baseline_I:.3f} (Target: ~0.3)")
    
    # Early training (volatile, unstable)
    print("\nEarly Training (Volatile Commitments):")
    early_commitments = [0.5, 0.6, 0.4, 0.7, 0.5]  # High variance
    early_volatility = [0.6, 0.7, 0.5, 0.6, 0.5]  # High volatility
    early_I = metrics.calculate_identity_strength(early_commitments, early_volatility, 0.4)
    print(f"  Commitments: {early_commitments} (std={np.std(early_commitments):.2f})")
    print(f"  Result: I = {early_I:.3f}")
    
    # Late training (stable, reinforced)
    print("\nLate Training (Stable Commitments):")
    late_commitments = [0.85, 0.87, 0.86, 0.88, 0.85]  # Low variance, high
    late_volatility = [0.2, 0.15, 0.18, 0.2, 0.15]  # Low volatility
    late_I = metrics.calculate_identity_strength(late_commitments, late_volatility, 0.9)
    print(f"  Commitments: {late_commitments} (std={np.std(late_commitments):.2f})")
    print(f"  Result: I = {late_I:.3f}")
    
    # Validate
    print("\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    baseline_ok = 0.2 <= baseline_I <= 0.4
    early_ok = early_I < 0.6  # Should be lower than late
    late_ok = late_I >= 0.75  # Should be approaching 0.85
    growth_ok = late_I > early_I
    
    print(f"\nBaseline I: {baseline_I:.2f} (Target: ~0.3) {'✅' if baseline_ok else '❌'}")
    print(f"Early I:    {early_I:.2f} (Should be <0.6) {'✅' if early_ok else '❌'}")
    print(f"Late I:     {late_I:.2f} (Target: ~0.85) {'✅' if late_ok else '❌'}")
    print(f"Growth:     {early_I:.2f} → {late_I:.2f} {'✅' if growth_ok else '❌'}")
    
    if baseline_ok and early_ok and late_ok and growth_ok:
        print("\n✅ IDENTITY FORMULA VALIDATED.")
        return True
    else:
        print("\n❌ IDENTITY FORMULA MISMATCH.")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RAVANA PAPER FORMULA VERIFICATION")
    print("Validating formulas before 10,000-episode run")
    print("="*70 + "\n")
    
    d_ok = verify_dissonance_formula()
    i_ok = verify_identity_formula()
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if d_ok and i_ok:
        print("\n✅ ALL FORMULAS VALIDATED")
        print("✅ Ready for 10,000 episode Long-Horizon Stability Test")
        print("\nNext step: Run the full test with:")
        print("  python core_k0/long_horizon_stability_test.py --episodes 10000")
        sys.exit(0)
    else:
        print("\n❌ FORMULA VERIFICATION FAILED")
        print("❌ Do NOT run long test until formulas are corrected")
        print("\nReview:")
        if not d_ok:
            print("  - Dissonance formula needs adjustment")
        if not i_ok:
            print("  - Identity formula needs adjustment")
        sys.exit(1)
