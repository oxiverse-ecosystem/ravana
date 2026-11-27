#!/usr/bin/env python3
"""SCRATCH control: production GloVe boot memo must be vector-identical.

Sequential boots of the isolated worktree store; flag off/on/on/off.
No production file is edited and no monkeypatch is installed.
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
from ravana.chat.engine_graph import _GLOVE_BOOT_MEMO  # noqa: E402

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


rows = []
for flag in ("0", "1", "1", "0"):
    os.environ["RAVANA_GLOVE_BOOT_MEMO"] = flag
    e = CognitiveChatEngine(data_dir=os.path.join(WT, "weights"), user_suffix="")
    rows.append((flag, digest(e)))
    print("FLAG=%s DIGEST=%s MEMO_KEYS=%d" % (flag, rows[-1][1][:24], len(_GLOVE_BOOT_MEMO)), flush=True)

ok = len(set(d for _, d in rows)) == 1
print("PROD_GLOVE_MEMO_EQUIVALENCE=%s" % ("PASS" if ok else "FAIL"))
assert ok
