#!/usr/bin/env python3
"""SCRATCH ONLY: bounded WFA A/B with larger re cache + memoized GloVe boot.

Caches the immutable GloVe load result (words, index map, projection, projected
matrix) at module level, keyed by cache-file identity + dim, and reuses it for
subsequent engine boots. No production file edited. Outputs copied for diff.
"""
import json
import os
import re
import shutil
import sys
import time

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "scripts"))
sys.path.insert(0, os.path.join(WT, "ravana", "src"))
os.environ["RAVANA_OFFLINE"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["WFA_BOUNDED_DIAGNOSTIC"] = "1"
os.environ["WFA_IS_MAX_PER_POOL"] = "2"
os.environ["WFA_IS_FRESH_ENGINE"] = "1"
re._MAXCACHE = 8192
re._MAXCACHE2 = 8192

import numpy as np  # noqa: E402
import measure_learn5_wfa as wfa  # noqa: E402
from ravana.chat.engine_graph import GraphMixin  # noqa: E402

_orig_glove = GraphMixin._init_glove
_CACHE = {}
_hits = [0]


def _patched_glove(self):
    repo = os.path.join(WT, "data", "ravana_glove_cache.npz")
    path = getattr(self, "_glove_cache_path", None)
    if not path or not os.path.exists(path):
        path = repo
    if not os.path.exists(path):
        return _orig_glove(self)
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns, int(self.dim))
    hit = _CACHE.get(key)
    if hit is not None:
        _hits[0] += 1
        (gd, proj, words, widx, vecs_proj) = hit
        self._glove_dim = gd
        self._glove_proj = proj
        self._glove_words = words
        self._glove_word_index = widx
        self._glove_vecs_proj = vecs_proj
        self._glove_vecs = {}
        self._glove_raw_vecs = None
        self._glove_vector_cache = {}
        print("  [GloVe] Loaded %d projected vectors from cache (%dD -> %dD)" %
              (len(words), gd, self.dim))
        return
    _orig_glove(self)
    if getattr(self, "_glove_vecs_proj", None) is not None:
        _CACHE[key] = (self._glove_dim, self._glove_proj, self._glove_words,
                       self._glove_word_index, self._glove_vecs_proj)


GraphMixin._init_glove = _patched_glove

_real = re._compiler.compile
_n = [0]


def _counted(pattern, flags):
    _n[0] += 1
    return _real(pattern, flags)


re._compiler.compile = _counted
t0 = time.time()
rc = wfa.main([])
elapsed = time.time() - t0
print("SIDE=B2_GLOVE WFA_EXIT=%s ELAPSED=%.1fs REAL_COMPILES=%d GLOVE_HITS=%d" %
      (rc, elapsed, _n[0], _hits[0]), flush=True)
for src, suffix in (("learn5_wfa_summary.json", "summary.json"),
                    ("learn5_wfa_probes.jsonl", "probes.jsonl")):
    shutil.copyfile(os.path.join(WT, "scratch", src),
                    os.path.join(WT, "scratch", "bounded_B2_" + suffix))
with open(os.path.join(WT, "scratch", "bounded_B2_meta.json"), "w",
          encoding="utf-8") as fh:
    json.dump({"side": "B2", "rc": rc, "elapsed_s": round(elapsed, 3),
               "real_compiles": _n[0], "glove_reuse_hits": _hits[0]},
              fh, indent=2)
