import pytest
import numpy as np
from ravana_ml.graph import ConceptGraph
from ravana.core.system1 import System1Attractor

def test_system1_settle():
    graph = ConceptGraph(dim=8, max_nodes=50)
    
    # Add nodes
    n1 = graph.add_node(vector=np.random.randn(8)*0.1, label="cause")
    n2 = graph.add_node(vector=np.random.randn(8)*0.1, label="effect")
    n3 = graph.add_node(vector=np.random.randn(8)*0.1, label="unrelated")
    
    # Add edge from cause to effect
    graph.add_edge(n1.id, n2.id, weight=0.8)
    
    attractor = System1Attractor(graph, decay=0.1, threshold=0.5)
    
    # Settle starting from n1 (cause)
    settled, confidence = attractor.settle([n1.id], max_iter=20)
    
    assert n1.id in settled
    assert settled[n1.id] == 1.0  # Clamped seed
    assert n2.id in settled
    assert settled[n2.id] > 0.05  # Propagated activation
    
    # Confidence score should be computed
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
