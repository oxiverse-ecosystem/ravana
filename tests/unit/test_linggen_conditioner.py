"""The LingGen conditioner must separate the subjects it conditions on.

MEASURED BUG (2026-11-14, docs/CONDITIONING_COLLAPSES_TO_A_POINT.md): fitted on
the 65-D Binder signature alone, the projection collapsed every subject to the
same point -- pairwise cosine of the projected conditioning vectors was 0.9990
across unrelated subjects (gravity vs ocean = 0.9996), and `W_sm` had effective
rank 9 of 65. The decoder therefore received no subject signal and emitted the
same filler skeleton for every subject.

`fit(..., identity_vecs=...)` concatenates an identity-bearing component (the
subject's own GloVe vector) after the binder signature. These tests pin that the
identity component actually increases separation, and that the forward pass
fails soft when it is missing.
"""
import numpy as np
import pytest

from ravana.ontology.linggen import LingGenConditioner


def _make_data(n=20, seed=0):
    """Binder signatures near-identical (typicality), GloVe vectors distinct."""
    rng = np.random.default_rng(seed)
    binder = (np.tile(np.array([3.0] * 65, dtype=np.float64), (n, 1))
              + rng.normal(0, 0.05, (n, 65)))
    glove = rng.normal(0, 1, (n, 64))
    target = rng.normal(0, 1, (n, 75)) + np.repeat(glove[:, :1], 75, axis=1) * 2.0
    return binder, glove, target


def _separation(cond, binder, glove=None):
    outs = [cond.condition(binder[i], identity_vec=glove[i] if glove is not None else None)
            for i in range(len(binder))]
    O = np.stack([o for o in outs if o is not None])
    if len(O) < 2:
        return 0.0
    return float(np.linalg.norm(O[0] - O[1]))


def test_identity_component_increases_separation():
    binder, glove, target = _make_data()
    without = LingGenConditioner.fit(binder, target)
    with_id = LingGenConditioner.fit(binder, target, identity_vecs=glove)
    assert with_id.trained and without.trained
    d_without = _separation(without, binder)
    d_with = _separation(with_id, binder, glove)
    assert d_with > d_without, (
        f"identity component must separate subjects: "
        f"with={d_with:.4f} without={d_without:.4f}")


def test_wider_matrix_is_produced():
    binder, glove, target = _make_data()
    c = LingGenConditioner.fit(binder, target, identity_vecs=glove)
    assert c._W.shape == (75, 65 + 64)


def test_condition_fails_soft_without_identity():
    """A projection fitted WITH an identity component must not silently produce
    a wrong-width matmul when the caller omits it."""
    binder, glove, target = _make_data()
    c = LingGenConditioner.fit(binder, target, identity_vecs=glove)
    assert c.condition(binder[0]) is None
    assert c.condition(binder[0], identity_vec=glove[0]) is not None


def test_condition_fails_soft_on_wrong_identity_width():
    binder, glove, target = _make_data()
    c = LingGenConditioner.fit(binder, target, identity_vecs=glove)
    assert c.condition(binder[0], identity_vec=glove[0][:10]) is None


def test_backward_compatible_fit_without_identity():
    """The original call signature still works and produces a (75, 65) matrix."""
    binder, glove, target = _make_data()
    c = LingGenConditioner.fit(binder, target)
    assert c._W.shape == (75, 65)
    assert c.condition(binder[0]) is not None


def test_save_load_roundtrip_with_identity():
    import os
    import tempfile
    binder, glove, target = _make_data()
    c = LingGenConditioner.fit(binder, target, identity_vecs=glove)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "wsm.npz")
        c.save(p)
        loaded = LingGenConditioner.load(p)
        assert loaded is not None and loaded.trained
        assert loaded._W.shape == c._W.shape
        a = c.condition(binder[3], identity_vec=glove[3])
        b = loaded.condition(binder[3], identity_vec=glove[3])
        assert np.allclose(a, b)
