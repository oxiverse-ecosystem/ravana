import sys, os, numpy as np
sys.path.insert(0, '/home/workspace/Projects/ravana-v2')
sys.path.insert(0, '/home/workspace/Projects/ravana-v2/core')

from core.governor import Governor, GovernorConfig, CognitiveSignals, RegulationMode
from core.identity import IdentityEngine
from core.resolution import ResolutionEngine
from core.state import StateManager

def quick_dynamics_test():
    results = {'honesty_lied': [], 'commitment_integrity': [], 'high_dissonance_pressure': []}
    
    # honesty_lied + commitment_integrity
    gov = Governor(GovernorConfig())
    ident = IdentityEngine()
    res = ResolutionEngine()
    sm = StateManager(gov, res, ident)
    
    print("Running 20 episodes (honesty_lied test)...")
    for ep in range(20):
        pre_i = sm.state.identity
        step = sm.step(correctness=False, difficulty=0.5)
        delta_i = sm.state.identity - pre_i
        results['honesty_lied'].append(delta_i)
        sm.state.dissonance = 0.2
    
    honesty_drops = [x for x in results['honesty_lied'] if x < -0.05]
    avg_honesty_drop = np.mean([x for x in results['honesty_lied'] if x < 0])
    
    print(f"\n=== METRIC: honesty_lied ===")
    print(f"  Avg identity drop on correctness=False: {avg_honesty_drop:.4f}")
    print(f"  Expected: >= 0.08 (NOT ~0.005)")
    print(f"  Drops >= 0.05: {len(honesty_drops)}/20")
    
    # commitment_integrity: test that failure doesn't catastrophically drop identity
    gov2 = Governor(GovernorConfig())
    sm2 = StateManager(gov2, ResolutionEngine(), IdentityEngine())
    sm2.state.identity = 0.7
    pre_i2 = sm2.state.identity
    sm2.step(correctness=False, difficulty=0.3)
    post_i2 = sm2.state.identity
    commitment_delta = post_i2 - pre_i2
    
    print(f"\n=== METRIC: commitment_integrity ===")
    print(f"  Identity start=0.7, after failure: {post_i2:.4f}")
    print(f"  Delta: {commitment_delta:.4f}")
    print(f"  Expected: <= 0.10 (NOT ~0.38)")
    
    # high_dissonance_pressure: D should NOT decrease when above 0.35
    gov3 = Governor(GovernorConfig())
    # Test: D=0.45 (above 0.35), upstream wants +0.05 increase
    sig = CognitiveSignals(dissonance_delta=0.05, identity_delta=0.0, source="test")
    reg = gov3.regulate(current_dissonance=0.45, current_identity=0.5, signals=sig, episode=1)
    d_after = 0.45 + reg.dissonance_delta
    
    print(f"\n=== METRIC: high_dissonance_pressure ===")
    print(f"  D_before=0.45 (>= 0.35 threshold), upstream delta=+0.05")
    print(f"  D_after regulation: {d_after:.4f}")
    print(f"  Expected: D should NOT decrease when above 0.35")
    print(f"  Governor mode: {reg.mode.value}")
    print(f"  delta direction: {'+' if reg.dissonance_delta >= 0 else '-'}{abs(reg.dissonance_delta):.4f}")
    
    # Also test D=0.35 exactly (boundary case)
    sig2 = CognitiveSignals(dissonance_delta=0.05, identity_delta=0.0, source="test")
    reg2 = gov3.regulate(current_dissonance=0.35, current_identity=0.5, signals=sig2, episode=2)
    d_after2 = 0.35 + reg2.dissonance_delta
    print(f"\n  Boundary test: D_before=0.35, D_after={d_after2:.4f}")
    
    return results

results = quick_dynamics_test()
