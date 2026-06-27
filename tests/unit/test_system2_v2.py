import pytest
import numpy as np
from ravana_ml.graph import ConceptGraph
from ravana.core.causal_schema import CausalSchemaLearner
from ravana.core.system2 import System2Simulator

def test_system2_subgraph_and_simulation():
    graph = ConceptGraph(dim=8, max_nodes=50)
    
    n1 = graph.add_node(vector=np.random.randn(8)*0.1, label="ice")
    n2 = graph.add_node(vector=np.random.randn(8)*0.1, label="water")
    n3 = graph.add_node(vector=np.random.randn(8)*0.1, label="steam")
    
    # Add edges
    graph.add_edge(n1.id, n2.id, weight=0.7)
    graph.add_edge(n2.id, n3.id, weight=0.7)
    
    # Set relation types
    edge1 = graph.get_edge(n1.id, n2.id)
    edge1.relation_type = "causal"
    edge2 = graph.get_edge(n2.id, n3.id)
    edge2.relation_type = "causal"
    
    schema_learner = CausalSchemaLearner()
    # Teach the learner transitions: ice + heat -> water, water + heat -> steam
    schema_learner.learn("ice", "heat", "water")
    schema_learner.learn("water", "heat", "steam")
    
    simulator = System2Simulator(graph, schema_learner)
    
    # Extract causal subgraph
    subgraph = simulator.extract_causal_subgraph("ice")
    assert n1.id in subgraph
    assert n2.id in subgraph
    assert n3.id in subgraph
    
    # Forward simulation
    trace = simulator.simulate_forward("ice", steps=3)
    assert len(trace) >= 2
    assert trace[0][0] == "ice"
    assert trace[0][1] == "heat"
    assert trace[0][2] > 0.3
    
    # Counterfactual simulation (suppress ice + heat transition)
    cf_trace = simulator.simulate_counterfactual("ice", ("ice", "heat"), steps=3)
    # The trace should be empty because we blocked the transition
    assert len(cf_trace) == 0
