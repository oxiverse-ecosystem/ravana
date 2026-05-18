"""
RAVANA v2 — PHASE I+: Reality Friction Runner
Test RAVANA under hostile, messy, delayed real-world conditions.

PHASE I+: REALITY FRICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
From laboratory epistemics → adversarial, messy reality

This phase introduces:
  • Partial observability (state is never fully visible)
  • Delayed ground truth (feedback arrives late, partially, or never)
  • Noisy signals (observation = truth + noise)
  • Non-stationarity (environment dynamics shift unexpectedly)
  • Resource constraints (limited compute forces shortcuts)
  • Hidden variables (causal factors RAVANA cannot observe)

GOAL: Does RAVANA remain sane when reality stops being cooperative?

USAGE:
    python run_phase_i_plus.py                    # Run all friction scenarios
    python run_phase_i_plus.py --scenario fog     # Run specific scenario
    python run_phase_i_plus.py --quick            # Quick test mode
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.reality_friction_env import (
    run_all_friction_tests,
    analyze_results,
    RealityFrictionEnvironment
)


def main():
    parser = argparse.ArgumentParser(
        description='Phase I+ Reality Friction Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SCENARIOS:
  fog_of_war        Heavy noise + partial observability
  delayed_truth     Ground truth arrives late or never  
  shifting_ground   Environment dynamics shift unexpectedly
  resource_star     Severe compute and memory constraints
  the_gauntlet      All friction types combined (ultimate test)

EXAMPLES:
  python run_phase_i_plus.py                    # Full test suite
  python run_phase_i_plus.py --quick            # Quick test (50 episodes each)
  python run_phase_i_plus.py --scenario fog     # Run single scenario
        """
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (reduced episodes)'
    )
    parser.add_argument(
        '--scenario',
        type=str,
        choices=list(RealityFrictionEnvironment.SCENARIOS.keys()),
        help='Run specific scenario only'
    )
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "="*70)
    print("  PHASE I+: REALITY FRICTION")
    print("  Testing RAVANA under hostile real-world conditions")
    print("="*70)
    
    print("\n🧪 FRICTION MECHANISMS:")
    print("  • Partial observability — state is never fully visible")
    print("  • Delayed feedback — truth arrives late or never")
    print("  • Observation noise — signal corrupted by interference")
    print("  • Non-stationarity — rules shift unexpectedly")
    print("  • Resource constraints — limited compute/memory")
    print("  • Hidden variables — unseen causal factors")
    
    print("\n🎯 SURVIVAL METRICS:")
    print("  • Belief drift — does RAVANA wander from truth?")
    print("  • Recovery time — how fast after disruption?")
    print("  • Confidence calibration — does certainty match accuracy?")
    print("  • Graceful degradation — collapse or adapt?")
    
    if args.scenario:
        # Run single scenario
        scenario = RealityFrictionEnvironment.SCENARIOS[args.scenario]
        
        if args.quick:
            scenario.episodes = 50
        
        print(f"\n{'─'*70}")
        print(f"  Running: {scenario.name}")
        print(f"  {scenario.description}")
        print(f"  Intensity: {scenario.intensity} | Episodes: {scenario.episodes}")
        print(f"{'─'*70}\n")
        
        env = RealityFrictionEnvironment(scenario)
        result = env.run_full_test()
        
        # Print detailed result
        status = "✅ SURVIVED" if result['survived'] else "❌ FAILED"
        print(f"\n{'='*70}")
        print(f"  {status}")
        print(f"{'='*70}")
        print(f"\n  Survival Score: {result['survival_score']:.2%}")
        print(f"  Threshold: {result['threshold']:.2%}")
        print(f"\n  Belief Drift:")
        print(f"    Average: {result['belief_drift']['avg']:.3f}")
        print(f"    Maximum: {result['belief_drift']['max']:.3f}")
        print(f"    Final: {result['belief_drift']['final']:.3f}")
        print(f"\n  Confidence Calibration:")
        print(f"    Overconfidence rate: {result['confidence']['overconfidence_rate']:.1%}")
        print(f"    Avg calibration error: {result['confidence']['avg_calibration']:.3f}")
        print(f"\n  Recovery:")
        print(f"    Total disruptions: {result['recovery']['total_disruptions']}")
        print(f"    Avg recovery time: {result['recovery']['avg_recovery_episodes']:.1f} episodes")
        print(f"\n{'='*70}")
        
        if result['survived']:
            print("\n✅ RAVANA PASSED this friction scenario")
            print("   System demonstrates epistemic resilience")
        else:
            print("\n❌ RAVANA FAILED this friction scenario")
            print(f"   Belief drift ({result['belief_drift']['final']:.3f}) exceeded threshold")
            print("   Architectural weakness exposed")
        
        # Save result
        import json
        import numpy as np
        
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            elif isinstance(obj, bool):
                return bool(obj)
            return obj
        
        os.makedirs('results', exist_ok=True)
        with open(f'results/phase_i_plus_{args.scenario}.json', 'w') as f:
            json.dump(convert(result), f, indent=2)
        
        print(f"\n💾 Result saved to: results/phase_i_plus_{args.scenario}.json")
        
    else:
        # Run full test suite
        results = run_all_friction_tests(quick=args.quick)
        report = analyze_results(results)
        print(report)
        
        # Save results
        import json
        import numpy as np
        
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            elif isinstance(obj, bool):
                return bool(obj)
            return obj
        
        os.makedirs('results', exist_ok=True)
        
        serializable_results = convert(results)
        with open('results/phase_i_plus_all_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        with open('results/phase_i_plus_report.txt', 'w') as f:
            f.write(report)
        
        print(f"\n💾 Results saved to:")
        print(f"  • results/phase_i_plus_all_results.json")
        print(f"  • results/phase_i_plus_report.txt")
        
        # Summary stats
        survived = sum(1 for r in results.values() if r['survived'])
        total = len(results)
        
        print("\n" + "="*70)
        print(f"  SUMMARY: {survived}/{total} scenarios survived")
        print("="*70)
        
        if survived == total:
            print("\n🎉 PERFECT SCORE — RAVANA is ready for the real world")
        elif survived >= total * 0.8:
            print(f"\n⚠️  PARTIAL — {total - survived} fragility detected, needs hardening")
        else:
            print(f"\n💥 CRITICAL — RAVANA collapsed under {total - survived} scenarios")
            print("   Major architectural revision needed")
        
        print("="*70)
    
    print("\n📚 Documentation:")
    print("  • core/reality_friction.py — Friction layer implementation")
    print("  • experiments/reality_friction_env.py — Test scenarios")
    print("  • results/ — Test outputs and metrics")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
