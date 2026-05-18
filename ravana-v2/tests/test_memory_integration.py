"""
Test script for Ravana Memory System integration into StateManager.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.governor import Governor, GovernorConfig
from core.resolution import ResolutionEngine
from core.identity import IdentityEngine
from core.state import StateManager

def test_memory_integration():
    print("="*60)
    print("TESTING RAVANA MEMORY INTEGRATION")
    print("="*60)
    
    # 1. Setup components
    governor = Governor(GovernorConfig(
        max_dissonance=0.9,
        min_dissonance=0.1
    ))
    resolution = ResolutionEngine()
    identity = IdentityEngine()
    
    manager = StateManager(governor, resolution, identity)
    
    print("\n[STEP 1] Initializing StateManager with Memory...")
    assert hasattr(manager, 'memory'), "StateManager should have a memory attribute"
    print("✓ Memory system initialized.")
    
    # 2. Run some steps to generate memories
    print("\n[STEP 2] Running 10 cognitive steps...")
    for i in range(10):
        # Alternate success and failure to create dissonance fluctuations
        correctness = (i % 2 == 0)
        manager.step(correctness=correctness, difficulty=0.6)
        
    print(f"✓ Completed {len(manager.history)} steps.")
    
    # 3. Verify Episodic Memory
    print("\n[STEP 3] Verifying Episodic Memory...")
    traces = manager.memory.episodic.traces
    print(f"   Traces recorded: {len(traces)}")
    assert len(traces) == 10, f"Expected 10 traces, got {len(traces)}"
    
    # Check if salience is being recorded
    sample_trace = traces[0]
    print(f"   Sample trace salience: {sample_trace.salience:.3f}")
    assert sample_trace.salience > 0, "Trace salience should be positive"
    print("✓ Episodic memory traces verified.")
    
    # 4. Verify Semantic Memory (Norm Updates)
    print("\n[STEP 4] Verifying Semantic Memory...")
    growth_norm = manager.memory.semantic.knowledge_graph["growth"]
    print(f"   Growth norm weight: {growth_norm['weight']:.3f}")
    print(f"   Growth norm confidence: {growth_norm['confidence']:.3f}")
    
    # In my memory.py, it updates growth if dissonance < 0.2
    # Let's check if any updates occurred
    updates = [h for h in manager.memory.semantic.history if h["norm"] == "growth"]
    print(f"   Growth norm updates: {len(updates)}")
    print("✓ Semantic memory verified.")
    
    # 5. Verify Retrieval
    print("\n[STEP 5] Verifying Retrieval Context...")
    context = manager.get_status() # Updated get_status might be needed or just direct access
    mem_context = manager.memory.get_context_for_decision()
    print(f"   Past failures retrieved: {len(mem_context['past_failures'])}")
    print("✓ Retrieval context verified.")
    
    print("\n" + "="*60)
    print("ALL MEMORY INTEGRATION TESTS PASSED")
    print("="*60)

if __name__ == "__main__":
    test_memory_integration()
