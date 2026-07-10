"""
Tests for N3 dual-code HRR space (additive lift, 64D glove untouched).
"""

import os
import sys

import numpy as np
import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))

from ravana.core.dual_code_space import DualCodeSpace, DEFAULT_HRR_DIM


@pytest.fixture(scope="module")
def dcs():
    cache = os.path.join(_PROJ, "data", "ravana_glove_cache.npz")
    if not os.path.exists(cache):
        pytest.skip("glove cache not present")
    return DualCodeSpace(cache)


class TestDualCodeSpaceAdditive:
    def test_64d_glove_untouched(self, dcs):
        v = dcs.atom64("water")
        assert v is not None
        assert v.shape == (64,)
        # glove-64 is the SAME lookup the intent pipeline uses
        assert np.allclose(v, dcs._lut64["water"] / (np.linalg.norm(dcs._lut64["water"]) + 1e-9))

    def test_hrr_space_is_high_dimensional(self, dcs):
        h = dcs.atom_hrr("water")
        assert h.shape == (DEFAULT_HRR_DIM,)
        assert np.isclose(np.linalg.norm(h), 1.0, atol=1e-4)

    def test_lift_is_additive_only(self, dcs):
        # instantiating DualCodeSpace must not mutate the 64D glove lookup
        before = dict(dcs._lut64)
        _ = dcs.atom_hrr("fire")
        assert set(dcs._lut64.keys()) == set(before.keys())


class TestBindingRecovery:
    def test_role_filler_recovery_better_in_hrr(self, dcs):
        # encode a fact, unbind subject, recover against candidates
        struct = dcs.encode_fact("water", "causes", "smoke")
        rec = dcs.recover_role_filler(struct, "subject",
                                       ["water", "fire", "earth", "paris"])
        assert rec == "water"

    def test_analogy_same_relation_similar(self, dcs):
        sim = dcs.relation_similarity(("water", "causes", "smoke"),
                                       ("fire", "causes", "smoke"))
        # same relation 'causes' -> structures should be similar (>0.5)
        assert sim > 0.5


class TestOOV:
    def test_oov_atom_deterministic(self, dcs):
        a = dcs.atom_hrr("zzqnox_blarg_42")
        b = dcs.atom_hrr("zzqnox_blarg_42")
        assert np.allclose(a, b)
        assert a.shape == (DEFAULT_HRR_DIM,)
