import numpy as np
import pytest
from ravana_ml.graph import ConceptGraph
from ravana.core.predictive_coding import PredictiveCodingLearner

def test_predictive_coding_init():
    graph = ConceptGraph(dim=16, max_nodes=100)
    learner = PredictiveCodingLearner(graph, lr=0.01)
    
    assert learner.graph == graph
    assert learner.lr == 0.01
    assert len(learner.predictors) == 0

def test_predictive_coding_predict_and_learn():
    graph = ConceptGraph(dim=8, max_nodes=100)
    # Initialize a node
    vec = np.random.randn(8) * 0.1
    node = graph.add_node(vector=vec, label="concept1")
    nid = node.id
    
    learner = PredictiveCodingLearner(graph, lr=0.05)
    
    # Predict using random context
    context = np.random.randn(8) * 0.1
    pred = learner.predict(nid, context)
    assert pred.shape == (8,)
    
    # Learn
    actual = np.random.randn(8) * 0.1
    old_vector = node.vector.copy()
    old_predictor = learner.get_predictor(nid).copy()
    
    error, error_norm = learner.learn_node(nid, context, actual)
    
    # Check that error is shape (8,) and error_norm is scalar
    assert error.shape == (8,)
    assert isinstance(error_norm, float)
    
    # Check node vector and predictor have changed
    assert not np.array_equal(node.vector, old_vector)
    assert not np.array_equal(learner.get_predictor(nid), old_predictor)
    
    # Check free energy fields
    assert node.prediction_free_energy > 0.0
    assert len(node.free_energy_history) == 1

def test_predictive_coding_propagate_errors():
    graph = ConceptGraph(dim=8, max_nodes=100)
    n1 = graph.add_node(vector=np.random.randn(8)*0.1, label="a")
    n2 = graph.add_node(vector=np.random.randn(8)*0.1, label="b")
    n3 = graph.add_node(vector=np.random.randn(8)*0.1, label="c")
    
    learner = PredictiveCodingLearner(graph, lr=0.01)
    context = np.random.randn(8) * 0.1
    
    total_error = learner.propagate_errors([n1.id, n2.id, n3.id], context)
    assert isinstance(total_error, float)
    assert total_error >= 0.0
