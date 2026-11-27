"""Regression tests for the default-OFF GloVe boot memo.

The memo is a measurement-harness optimisation only: it must change NOTHING
when RAVANA_GLOVE_BOOT_MEMO is unset/0, and when enabled it may share only the
immutable boot payload -- never per-engine mutable state.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from ravana.chat.engine_graph import (
    GraphMixin,
    _GLOVE_BOOT_MEMO,
    _glove_projected_digest,
)

FLAG = "RAVANA_GLOVE_BOOT_MEMO"


@pytest.fixture(autouse=True)
def _clean_memo(monkeypatch):
    _GLOVE_BOOT_MEMO.clear()
    monkeypatch.delenv(FLAG, raising=False)
    yield
    _GLOVE_BOOT_MEMO.clear()


def _npz(tmp_path, n=12, gdim=5):
    rng = np.random.RandomState(7)
    words = ["w%02d" % i for i in range(n)]
    vecs = rng.randn(n, gdim).astype(np.float32)
    proj = rng.randn(4, gdim).astype(np.float32)
    path = tmp_path / "glove.npz"
    np.savez(path, words=words, vecs=vecs, proj=proj, glove_dim=gdim)
    return str(path)


def _boot(path, dim=4):
    host = SimpleNamespace(dim=dim, _glove_cache_path=path)
    GraphMixin._init_glove(host)
    return host


def test_flag_off_or_zero_is_unmodified_path(tmp_path, monkeypatch):
    path = _npz(tmp_path)
    a = _boot(path)
    monkeypatch.setenv(FLAG, "0")
    b = _boot(path)
    assert not _GLOVE_BOOT_MEMO
    assert a._glove_vecs_proj is not b._glove_vecs_proj
    np.testing.assert_array_equal(a._glove_vecs_proj, b._glove_vecs_proj)


def test_memo_shares_payload_but_not_mutable_state(tmp_path, monkeypatch):
    path = _npz(tmp_path)
    monkeypatch.setenv(FLAG, "1")
    a = _boot(path)
    b = _boot(path)
    assert len(_GLOVE_BOOT_MEMO) == 1
    assert b._glove_vecs_proj is a._glove_vecs_proj
    assert b._glove_word_index is a._glove_word_index
    assert b._glove_vecs == {} and b._glove_vecs is not a._glove_vecs
    assert b._glove_vector_cache == {} and b._glove_vector_cache is not a._glove_vector_cache
    assert _glove_projected_digest(a._glove_vecs_proj) == _glove_projected_digest(b._glove_vecs_proj)
    np.testing.assert_array_equal(
        GraphMixin._glove_vector(a, "w01"),
        GraphMixin._glove_vector(b, "w01"))
    b._glove_vector_cache["sentinel"] = 1
    assert "sentinel" not in a._glove_vector_cache


def test_digest_guard_rejects_corrupted_shared_matrix(tmp_path, monkeypatch):
    path = _npz(tmp_path)
    monkeypatch.setenv(FLAG, "1")
    a = _boot(path)
    expected = a._glove_vecs_proj.copy()
    entry = next(iter(_GLOVE_BOOT_MEMO.values()))
    entry[4][0, 0] += 100.0  # simulate in-place corruption after first boot
    b = _boot(path)
    assert b._glove_vecs_proj is not a._glove_vecs_proj
    np.testing.assert_array_equal(b._glove_vecs_proj, expected)


def test_key_includes_dim_and_npz_mtime(tmp_path, monkeypatch):
    path = _npz(tmp_path)
    monkeypatch.setenv(FLAG, "1")
    a = _boot(path, dim=4)
    b = _boot(path, dim=3)
    assert b._glove_vecs_proj is not a._glove_vecs_proj
    assert len(_GLOVE_BOOT_MEMO) == 2
    assert _boot(path, dim=4)._glove_vecs_proj is a._glove_vecs_proj
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    d = _boot(path, dim=4)
    assert d._glove_vecs_proj is not a._glove_vecs_proj
    assert len(_GLOVE_BOOT_MEMO) == 3
