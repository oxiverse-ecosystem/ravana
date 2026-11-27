#!/usr/bin/env python3
"""SCRATCH ONLY: equivalence control for the GloVe boot memo.

Boot one engine normally, capture 300 GloVe vectors + the projected matrix
hash; install the memo patch, boot six more engines sequentially, and assert
every sampled vector and the base matrix are byte-identical across all boots.
No production file edited.
"""
import hashlib
import os
import sys

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "scripts"))
sys.path.insert(0, os.path.join(WT, "ravana", "src"))
os.environ["RAVANA_OFFLINE"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
from ravana.chat.engine_graph import GraphMixin  # noqa: E402

WORDS = ("water rain frost tide stomach climate storm cloud heat moon sea "
         "oxygen enzyme acid food durable material moonlight garden river "
         "evidence source credible unreliable memory learning").split()
WORDS += ["word%03d" % i for i in range(100)]


def digest(eng):
    h = hashlib.sha256()
    base = eng._glove_vecs_proj
    h.update(str(base.shape).encode() + base.tobytes()[:200000])
    for w in WORDS:
        v = eng._glove_vector(w)
        if v is None:
            h.update(("None:" + w).encode())
        else:
            h.update(w.encode() + v.tobytes())
    return h.hexdigest()


engines = []
for i in range(2):
    e = CognitiveChatEngine(data_dir=os.path.join(WT, "weights"), user_suffix="")
    engines.append(e)
base = digest(engines[0])
print("baseline digest", base[:24])

_orig = GraphMixin._init_glove
_CACHE = {}
_hits = [0]


def patched(self):
    path = os.path.join(WT, "data", "ravana_glove_cache.npz")
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns, int(self.dim))
    hit = _CACHE.get(key)
    if hit is not None:
        _hits[0] += 1
        gd, proj, words, widx, vecs = hit
        self._glove_dim, self._glove_proj = gd, proj
        self._glove_words, self._glove_word_index = words, widx
        self._glove_vecs_proj = vecs
        self._glove_vecs, self._glove_raw_vecs = {}, None
        self._glove_vector_cache = {}
        return
    _orig(self)
    if getattr(self, "_glove_vecs_proj", None) is not None:
        _CACHE[key] = (self._glove_dim, self._glove_proj, self._glove_words,
                       self._glove_word_index, self._glove_vecs_proj)


GraphMixin._init_glove = patched
digests = []
for i in range(6):
    e = CognitiveChatEngine(data_dir=os.path.join(WT, "weights"), user_suffix="")
    digests.append(digest(e))
print("memo hits", _hits[0], "digests", len(set(digests)),
      "all_equal_to_baseline", all(d == base for d in digests))
assert _hits[0] == 5 and all(d == base for d in digests)
print("GLOVE_MEMO_EQUIVALENCE=PASS")
