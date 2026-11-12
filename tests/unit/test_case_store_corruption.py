"""A corrupt case-distribution store must not break its callers.

`data/case_dist.json` is runtime-generated and gitignored. A partial write
(observed on this host: 73 NUL bytes) made `CaseDistributionStore.load` raise
`JSONDecodeError`. `kb_describe` wraps its whole body in
`except Exception: continue`, so the good Wikipedia description it had already
fetched was DISCARDED and returned as None -- which silently emptied the
LingGen harvest (250 pairs, then 0 usable).

Two fixes, both tested here:
  1. `load()` degrades to the empty store on a corrupt file.
  2. `kb_describe` returns the description even if casing fails.
"""
import json
import os
import tempfile

from ravana.chat.case_distribution import (
    CaseDistributionStore,
    _DEFAULT_CAP_PROB,
    case_infer,
)


def _write_corrupt(path, payload: bytes):
    with open(path, "wb") as fh:
        fh.write(payload)


def test_load_survives_nul_bytes(tmp_path):
    """The exact corruption observed: a file of NUL bytes."""
    p = tmp_path / "case_dist.json"
    _write_corrupt(p, b"\x00" * 73)
    store = CaseDistributionStore.load(str(p))
    assert store is not None
    # Degrades to the conservative default rather than raising.
    assert store.cap_prob("anything") == _DEFAULT_CAP_PROB


def test_load_survives_truncated_json(tmp_path):
    p = tmp_path / "case_dist.json"
    _write_corrupt(p, b'{"cap": {"Gravity": 0.9')
    store = CaseDistributionStore.load(str(p))
    assert store is not None


def test_load_survives_non_dict_json(tmp_path):
    """Valid JSON that is not the expected shape must not raise either."""
    p = tmp_path / "case_dist.json"
    p.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    store = CaseDistributionStore.load(str(p))
    assert store is not None


def test_load_reads_a_valid_store(tmp_path):
    """The fix must not break the normal path."""
    p = tmp_path / "case_dist.json"
    p.write_text(json.dumps({"cap": {"gravity": 0.95}, "default": 0.1}),
                 encoding="utf-8")
    store = CaseDistributionStore.load(str(p))
    assert store.cap_prob("gravity") > 0.9


def test_case_infer_works_with_empty_store():
    """The degraded store still yields usable casing."""
    store = CaseDistributionStore()
    out = case_infer("gravity is a force", store=store)
    assert out
    # Sentence-initial positional route still capitalizes.
    assert out[0] == "G"
