import numpy as np
import pytest
from ravana.core.vsa import VSAManager, VSASchema

def test_vsa_manager_init():
    mgr = VSAManager(dim=128)
    assert mgr.dim == 128
    assert "subject" in mgr.role_vectors
    assert "verb" in mgr.role_vectors
    assert "object" in mgr.role_vectors

def test_vsa_binding_unbinding():
    mgr = VSAManager(dim=256)
    
    filler = mgr.generate_vector()
    role = "subject"
    
    bound = mgr.bind_role_filler(role, filler)
    assert bound.shape == (256,)
    
    unbound = mgr.unbind_role(bound, role)
    assert unbound.shape == (256,)
    
    # Verify unbinding recovers the filler vector
    from ravana.core.vsa import cosine_sim
    assert cosine_sim(unbound, filler) > 0.6

def test_vsa_bundling():
    mgr = VSAManager(dim=128)
    v1 = mgr.generate_vector()
    v2 = mgr.generate_vector()
    
    bundled = mgr.bundle([v1, v2])
    assert bundled.shape == (128,)
    # Verify it is normalized
    assert np.allclose(np.linalg.norm(bundled), 1.0)

def test_vsa_schema():
    mgr = VSAManager(dim=512)
    schema = VSASchema(mgr, "S-V-O", ["subject", "verb", "object"])
    
    fillers = {
        "subject": mgr.generate_vector(),
        "verb": mgr.generate_vector(),
        "object": mgr.generate_vector()
    }
    
    # Construct schema vector
    schema_vec = schema.construct_schema_vector(fillers)
    assert schema_vec.shape == (512,)
    
    # Reconstruct fillers
    reconstructed = schema.reconstruct_fillers(schema_vec)
    assert "subject" in reconstructed
    assert "verb" in reconstructed
    assert "object" in reconstructed
    
    # Check matching
    from ravana.core.vsa import cosine_sim
    subj_decoded = reconstructed["subject"]
    best_candidate, score = schema.score_matching_filler(
        subj_decoded, 
        {"candidate1": fillers["subject"], "candidate2": mgr.generate_vector()}
    )
    assert best_candidate == "candidate1"
    assert score > 0.35
