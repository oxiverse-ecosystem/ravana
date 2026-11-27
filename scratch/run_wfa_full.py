#!/usr/bin/env python3
"""SCRATCH ONLY: full-bank WFA harness with the candidate larger re cache.

No bounded env, no gate/item/threshold change. Only the scratch driver sets
re._MAXCACHE/_MAXCACHE2 before the unmodified harness main() runs.
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
SIDE = os.environ.get("SIDE", "B")
if SIDE == "B":
    re._MAXCACHE = 8192
    re._MAXCACHE2 = 8192

import measure_learn5_wfa as wfa  # noqa: E402

if os.environ.get("GLOVE_MEMO") == "1":
    from ravana.chat.engine_graph import GraphMixin  # noqa: E402
    _orig_glove = GraphMixin._init_glove
    _CACHE, _hits = {}, [0]

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
            self._glove_dim, self._glove_proj = gd, proj
            self._glove_words, self._glove_word_index = words, widx
            self._glove_vecs_proj = vecs_proj
            self._glove_vecs, self._glove_raw_vecs = {}, None
            self._glove_vector_cache = {}
            print("  [GloVe] Loaded %d projected vectors from cache (%dD -> %dD)" %
                  (len(words), gd, self.dim))
            return
        _orig_glove(self)
        if getattr(self, "_glove_vecs_proj", None) is not None:
            _CACHE[key] = (self._glove_dim, self._glove_proj,
                           self._glove_words, self._glove_word_index,
                           self._glove_vecs_proj)
        return None

    GraphMixin._init_glove = _patched_glove
    print("GLOVE_MEMO=1 installed", flush=True)

t0 = time.time()
rc = wfa.main([])
elapsed = time.time() - t0
print("FULL_SIDE=%s FULL_EXIT=%s ELAPSED=%.1fs" % (SIDE, rc, elapsed), flush=True)
shutil.copyfile(os.path.join(WT, "scratch", "learn5_wfa_summary.json"),
                os.path.join(WT, "scratch", "full_%s_summary.json" % SIDE))
shutil.copyfile(os.path.join(WT, "scratch", "learn5_wfa_probes.jsonl"),
                os.path.join(WT, "scratch", "full_%s_probes.jsonl" % SIDE))
with open(os.path.join(WT, "scratch", "full_%s_meta.json" % SIDE), "w",
          encoding="utf-8") as fh:
    json.dump({"side": SIDE, "rc": rc, "elapsed_s": round(elapsed, 3)},
              fh, indent=2)
