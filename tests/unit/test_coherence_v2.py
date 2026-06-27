import pytest
from ravana.core.coherence import CoherenceNetwork

def test_coherence_init():
    net = CoherenceNetwork(decay=0.05)
    assert net.decay == 0.05
    assert len(net.propositions) == 0

def test_coherence_settling():
    net = CoherenceNetwork(decay=0.05, upper_bound=1.0, lower_bound=-1.0)
    
    # 2 competing hypotheses explaining a fact (evidence)
    # Evidence: Fact A
    # Hypothesis 1: H1 (coheres with A)
    # Hypothesis 2: H2 (coheres with A, contradicts H1)
    net.add_proposition("FactA", is_evidence=True)
    net.add_proposition("H1", initial_activation=0.1)
    net.add_proposition("H2", initial_activation=0.1)
    
    net.add_coherence("FactA", "H1", weight=0.4)
    net.add_coherence("FactA", "H2", weight=0.2) # H1 has stronger evidence coherence
    net.add_contradiction("H1", "H2", weight=-0.5) # Direct contradiction
    
    activations = net.settle(max_iter=100)
    
    # FactA should be clamped at 1.0
    assert activations["FactA"] == 1.0
    
    # H1 should win and be accepted, H2 should be rejected/suppressed due to contradiction and weaker coherence
    assert activations["H1"] > 0.4
    assert activations["H2"] < 0.0
    
    accepted = net.get_accepted(threshold=0.3)
    assert "H1" in accepted
    assert "H2" not in accepted
    
    rejected = net.get_rejected(threshold=0.0)
    assert "H2" in rejected
