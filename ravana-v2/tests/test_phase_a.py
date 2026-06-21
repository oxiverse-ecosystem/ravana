"""RAVANA v2 — Phase A Tests
Verify core architecture before full training.
"""
import sys

try:
    from .conftest import import_core
except ImportError:
    from conftest import import_core

Governor, CognitiveSignals = import_core(
    "governor", "Governor", "CognitiveSignals"
)
ResolutionEngine = import_core("resolution", "ResolutionEngine")
IdentityEngine = import_core("identity", "IdentityEngine")
StateManager = import_core("state", "StateManager")


def test_governor_hard_constraints():
    """Test that governor enforces hard constraints."""
    print("\n[TEST] Testing Governor Hard Constraints...")

    governor = Governor()

    # Test 1: Dissonance ceiling
    signals = CognitiveSignals(dissonance_delta=0.5, identity_delta=0.0)
    result = governor.regulate(
        current_dissonance=0.8,
        current_identity=0.5,
        signals=signals,
        episode=0,
    )

    projected = 0.8 + result.dissonance_delta
    assert projected <= 0.96, f"Ceiling failed: {projected}"
    print(f"  [OK] Dissonance ceiling: {projected:.3f} <= 0.96")

    # Test 2: Identity floor
    signals = CognitiveSignals(dissonance_delta=0.0, identity_delta=-0.5)
    result = governor.regulate(
        current_dissonance=0.5,
        current_identity=0.12,
        signals=signals,
        episode=0,
    )

    projected = 0.12 + result.identity_delta
    assert projected >= 0.09, f"Floor failed: {projected}"
    print(f"  [OK] Identity floor: {projected:.3f} >= 0.09")

    print("  [PASS] All hard constraints working!")


def test_resolution_partial_credit():
    """Test resolution engine partial credit accumulation."""
    print("\n[TEST] Testing Resolution Partial Credit...")

    engine = ResolutionEngine(partial_threshold=0.15)

    for i in range(5):
        result = engine.compute(
            episode=i,
            prev_dissonance=0.5,
            current_dissonance=0.40,
            correctness=True,
            difficulty=0.5,
        )
        print(
            f"  EP{i}: partial={result['partial_credit']:.4f}, "
            f"accumulated={result['total_partial_credit']:.4f}"
        )

    status = engine.get_memory_status()
    assert status["accumulated_partial"] > 0, "No partial credit accumulated"
    print(f"  [OK] Accumulated partial credit: {status['accumulated_partial']:.4f}")
    print("  [PASS] Partial credit system working!")


def test_identity_momentum():
    """Test identity momentum and recovery."""
    print("\n[TEST] Testing Identity Momentum...")

    engine = IdentityEngine(initial_strength=0.2)
    strengths = [0.2]

    for i in range(10):
        new_s = engine.compute_update(
            resolution_delta=0.05,
            resolution_success=True,
            regulated_identity_delta=0.02,
            current_dissonance=0.5,
        )
        engine.apply_update(new_s)
        strengths.append(new_s)

    print(f"  Start: {strengths[0]:.3f}, End: {strengths[-1]:.3f}")
    assert strengths[-1] > strengths[0], "Identity should grow"
    print("  [PASS] Identity dynamics working!")


def test_integration():
    """Test full integration via StateManager."""
    print("\n[TEST] Testing Full Integration...")

    governor = Governor()
    resolution = ResolutionEngine()
    identity = IdentityEngine()
    manager = StateManager(governor, resolution, identity)

    print("  Running 10 integration steps...")
    for i in range(10):
        record = manager.step(
            correctness=(i % 3 != 0),
            difficulty=0.5,
            debug=True,
        )

    status = manager.get_status()
    print(f"  Final: D={status['state']['dissonance']:.3f}, I={status['state']['identity']:.3f}")

    assert 0.1 < status["state"]["dissonance"] < 1.0, "Dissonance in sane range"
    assert 0.1 < status["state"]["identity"] < 1.0, "Identity in sane range"
    print("  [PASS] Full integration working!")


def main():
    """Run all Phase A tests."""
    print("=" * 60)
    print("RAVANA v2 — Phase A Architecture Tests")
    print("=" * 60)

    tests = [
        ("Governor Hard Constraints", test_governor_hard_constraints),
        ("Resolution Partial Credit", test_resolution_partial_credit),
        ("Identity Momentum", test_identity_momentum),
        ("Full Integration", test_integration),
    ]

    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, True))
        except Exception as e:
            print(f"\n[FAIL] {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"[DONE] {passed}/{total} tests passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)