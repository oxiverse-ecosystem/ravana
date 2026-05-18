"""
RAVANA v2 — Phase A Tests
Verify core architecture before full training.
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import Governor, ResolutionEngine, IdentityEngine, StateManager


def test_governor_hard_constraints():
    """Test that governor enforces hard constraints."""
    print("\n🧪 Testing Governor Hard Constraints...")
    
    governor = Governor()
    from core.governor import CognitiveSignals
    
    # Test 1: Dissonance ceiling
    signals = CognitiveSignals(dissonance_delta=0.5, identity_delta=0.0)
    result = governor.regulate(
        current_dissonance=0.8,  # Close to ceiling
        current_identity=0.5,
        signals=signals,
        episode=0
    )
    
    # Should be capped
    projected = 0.8 + result.dissonance_delta
    assert projected <= 0.96, f"Ceiling failed: {projected}"
    print(f"  ✓ Dissonance ceiling: {projected:.3f} <= 0.96")
    
    # Test 2: Identity floor
    signals = CognitiveSignals(dissonance_delta=0.0, identity_delta=-0.5)
    result = governor.regulate(
        current_dissonance=0.5,
        current_identity=0.12,  # Close to floor
        signals=signals,
        episode=0
    )
    
    projected = 0.12 + result.identity_delta
    assert projected >= 0.09, f"Floor failed: {projected}"
    print(f"  ✓ Identity floor: {projected:.3f} >= 0.09")
    
    print("  ✅ All hard constraints working!")


def test_resolution_partial_credit():
    """Test resolution engine partial credit accumulation."""
    print("\n🧪 Testing Resolution Partial Credit...")
    
    engine = ResolutionEngine(partial_threshold=0.15)
    
    # Add several small resolution events
    for i in range(5):
        result = engine.compute(
            episode=i,
            prev_dissonance=0.5,
            current_dissonance=0.48,  # Small reduction
            correctness=True,
            difficulty=0.5,
        )
        print(f"  EP{i}: partial={result['partial_credit']:.4f}, accumulated={result['total_partial_credit']:.4f}")
    
    # Check for wisdom generation or accumulation
    status = engine.get_memory_status()
    assert status['accumulated_partial'] > 0, "No partial credit accumulated"
    print(f"  ✓ Accumulated partial credit: {status['accumulated_partial']:.4f}")
    print("  ✅ Partial credit system working!")


def test_identity_momentum():
    """Test identity momentum and recovery."""
    print("\n🧪 Testing Identity Momentum...")
    
    engine = IdentityEngine(initial_strength=0.2)
    
    # Simulate growth
    strengths = [0.2]
    for i in range(10):
        new_s = engine.compute_update(
            resolution_delta=0.05,
            resolution_success=True,
            regulated_identity_delta=0.02,
            current_dissonance=0.5
        )
        engine.apply_update(new_s)
        strengths.append(new_s)
    
    # Should show growth (with possible momentum effects)
    print(f"  Start: {strengths[0]:.3f}, End: {strengths[-1]:.3f}")
    assert strengths[-1] > strengths[0], "Identity should grow"
    print("  ✅ Identity dynamics working!")


def test_integration():
    """Test full integration via StateManager."""
    print("\n🧪 Testing Full Integration...")
    
    governor = Governor()
    resolution = ResolutionEngine()
    identity = IdentityEngine()
    manager = StateManager(governor, resolution, identity)
    
    # Run 10 steps
    print("  Running 10 integration steps...")
    for i in range(10):
        record = manager.step(
            correctness=(i % 3 != 0),  # 2/3 success rate
            difficulty=0.5,
            debug=True
        )
    
    # Check state is healthy
    status = manager.get_status()
    print(f"  Final: D={status['state']['dissonance']:.3f}, I={status['state']['identity']:.3f}")
    
    # Assertions
    assert 0.1 < status['state']['dissonance'] < 1.0, "Dissonance in sane range"
    assert 0.1 < status['state']['identity'] < 1.0, "Identity in sane range"
    
    print("  ✅ Full integration working!")


def main():
    """Run all Phase A tests."""
    print("=" * 60)
    print("RAVANA v2 — Phase A Architecture Tests")
    print("=" * 60)
    
    try:
        test_governor_hard_constraints()
        test_resolution_partial_credit()
        test_identity_momentum()
        test_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED — Phase A Ready!")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
