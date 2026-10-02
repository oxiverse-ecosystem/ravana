"""Simulate CI's WARM GloVe cache for a local pytest run.

`glove_ready` is a read-only property derived from the GloVe backing stores,
so a warm environment is reproduced by populating the file-read fallback store
(`_glove_vecs`) plus its projection -- exactly what the warm-glove-cache CI job
provides -- NOT by assigning to the property.

Enable with:  PYTHONPATH=tools pytest ... -p warmglove_plugin

This is a DIAGNOSTIC harness, not a shipped test dependency: it exists so a
routing assertion can be exercised in the warm-cache environment CI actually
uses, which a developer box with no GloVe data cannot otherwise reproduce.
"""
import numpy as np

from ravana.chat.engine import CognitiveChatEngine

_orig_init = CognitiveChatEngine.__init__

# Real English words, so the gibberish guard can see them. Keyboard-mash tokens
# ("asdf", "qwer", "zxcv") are deliberately absent, so genuine letter-salad is
# still rejected in the warm environment too.
_REAL = {
    "tea", "coffee", "trains", "planes", "mountains", "coast", "small",
    "towns", "big", "cities", "humans", "honestly", "think", "believe",
    "are", "better", "than", "finer", "make", "daily", "work", "remote",
    "office", "sea", "hiking", "letterpress",
}


def _warm_init(self, *a, **k):
    _orig_init(self, *a, **k)
    # Real GloVe vectors are 100d; the store holds RAW vectors and a
    # projection down to the engine's dim, exactly as the cache path does.
    raw_dim, proj_dim = 100, getattr(self, "dim", 64)
    self._glove_vecs = {w: np.ones(raw_dim, dtype=np.float32) for w in _REAL}
    self._glove_proj = np.eye(proj_dim, raw_dim, dtype=np.float32)


CognitiveChatEngine.__init__ = _warm_init