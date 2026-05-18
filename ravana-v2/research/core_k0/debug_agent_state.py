"""
Debug Inspector: Inspect K2's internal attributes for metric extraction.
"""

import sys
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.core_k0.agent_loop_k2 import K2_Agent
from research.experiments_k0.resource_env import ResourceSurvivalEnv

def inspect_agent_state():
    env = ResourceSurvivalEnv(seed=42)
    agent = K2_Agent()
    
    # Perform one step
    obs = env._generate_observation()
    action = agent.select_action(obs)
    result = env.execute_action(action)
    
    # Record outcome
    if hasattr(agent, '_record_outcome'):
        agent._record_outcome(env, action, result)
    
    print("=" * 70)
    print("AGENT STATE INSPECTION (K2_Agent)")
    print("=" * 70)
    
    print(f"\nAction Output: {action}")
    print(f"Result: {result}")
    
    # Inspect internal attributes
    print("\n" + "=" * 70)
    print("INTERNAL ATTRIBUTES")
    print("=" * 70)
    
    for attr in dir(agent):
        if not attr.startswith('_'):
            try:
                val = getattr(agent, attr)
                if isinstance(val, (list, dict, float, int)) and not callable(val):
                    if isinstance(val, (list, dict)):
                        print(f"{attr}: {type(val).__name__} (len={len(val)})")
                    else:
                        print(f"{attr}: {val}")
            except:
                pass
    
    # Check agent.state
    if hasattr(agent, 'state'):
        print("\n" + "=" * 70)
        print("AGENT.STATE ATTRIBUTES")
        print("=" * 70)
        
        for attr in dir(agent.state):
            if not attr.startswith('_'):
                try:
                    val = getattr(agent.state, attr)
                    if isinstance(val, (list, dict, float, int)) and not callable(val):
                        if isinstance(val, (list, dict)):
                            print(f"state.{attr}: {type(val).__name__} (len={len(val)})")
                            if isinstance(val, list) and len(val) > 0 and len(str(val)) < 200:
                                print(f"  Sample: {val[:3] if len(val) > 3 else val}")
                        else:
                            print(f"state.{attr}: {val}")
                except:
                    pass
    
    # Check for specific metric inputs
    print("\n" + "=" * 70)
    print("METRIC INPUT CHECK")
    print("=" * 70)
    
    print(f"Has outcome_history? {hasattr(agent.state, 'outcome_history') if hasattr(agent, 'state') else False}")
    print(f"Has action_history? {hasattr(agent.state, 'action_history') if hasattr(agent, 'state') else False}")
    print(f"Has energy_history? {hasattr(agent.state, 'energy_history') if hasattr(agent, 'state') else False}")
    print(f"Has context_weights? {hasattr(agent, 'context_weights')}")
    print(f"Has policy? {hasattr(agent, 'policy')}")
    
    # Inspect outcome_history if exists
    if hasattr(agent.state, 'outcome_history') and agent.state.outcome_history:
        print("\n" + "=" * 70)
        print("OUTCOME_HISTORY SAMPLE (first 3 entries)")
        print("=" * 70)
        
        for i, outcome in enumerate(agent.state.outcome_history[:3]):
            print(f"\nOutcome {i}:")
            for attr in dir(outcome):
                if not attr.startswith('_'):
                    try:
                        val = getattr(outcome, attr)
                        if not callable(val):
                            print(f"  {attr}: {val}")
                    except:
                        pass

if __name__ == "__main__":
    inspect_agent_state()
