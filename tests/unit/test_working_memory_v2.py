import numpy as np
import pytest
from ravana.core.working_memory import WorkingMemory, hrr_bind, hrr_unbind, cosine_sim

def test_vsa_binding():
    # Simple vector bindings
    a = np.random.randn(512)
    a /= np.linalg.norm(a)
    b = np.random.randn(512)
    b /= np.linalg.norm(b)
    
    bound = hrr_bind(a, b)
    assert bound.shape == (512,)
    
    decoded = hrr_unbind(bound, b)
    assert decoded.shape == (512,)
    
    # Check that cosine similarity of decoded vector with original 'a' is high
    assert cosine_sim(decoded, a) > 0.5

def test_working_memory_push_pop_eviction():
    wm = WorkingMemory(capacity=3, decay_rate=0.1)
    
    v1 = np.ones(8)
    v2 = np.ones(8) * 2
    v3 = np.ones(8) * 3
    v4 = np.ones(8) * 4
    
    wm.push(v1, tag="t1")
    wm.push(v2, tag="t2")
    wm.push(v3, tag="t3")
    
    assert len(wm.slots) == 3
    assert wm.slots[0].tag == "t1"
    
    # Push 4th item, should evict t1 (FIFO)
    wm.push(v4, tag="t4")
    assert len(wm.slots) == 3
    assert wm.slots[0].tag == "t2"
    assert wm.slots[2].tag == "t4"

def test_working_memory_retrieve_and_decay():
    wm = WorkingMemory(capacity=5, decay_rate=0.1)
    
    # Create orthogonal-ish vectors
    v1 = np.array([1.0, 0.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0, 0.0])
    v3 = np.array([0.0, 0.0, 1.0, 0.0])
    
    wm.push(v1, tag="x")
    wm.push(v2, tag="y")
    wm.push(v3, tag="z")
    
    # Retrieve with query similar to v2
    query = np.array([0.1, 0.9, 0.0, 0.0])
    match = wm.retrieve(query)
    
    assert match is not None
    assert match.tag == "y"
    
    # Retrieve triggers decay on the slot vectors
    # Check that slot vectors are decayed (norm < 1.0)
    for slot in wm.slots:
        assert np.linalg.norm(slot.vector) < 1.0

def test_working_memory_interference():
    wm = WorkingMemory(capacity=5, interference_scale=0.2)
    
    # Push two identical vectors in sequence
    v1 = np.array([1.0, 0.0, 0.0, 0.0])
    wm.push(v1, tag="first")
    wm.push(v1, tag="second") # Trigger interference
    
    # Both vectors should be inhibited (magnitude reduced)
    assert np.linalg.norm(wm.slots[0].vector) < 1.0
    assert np.linalg.norm(wm.slots[1].vector) < 1.0
