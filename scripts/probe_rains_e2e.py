"""Turn-level before/after on IDENTICAL state, with the internal-knowledge
early return (engine.py:9026) neutralised so the generation path is isolated.

Same process, same graph: OLD resolution vs the REAL patched source.
"""
import os
import sys
import types

sys.path.insert(0, 'ravana/src')

from ravana.chat.engine import CognitiveChatEngine

DATA = os.environ.get('RAVANA_PROBE_DATA', 'weights')
print("=== BOOT (%s) ===" % DATA, flush=True)
eng = CognitiveChatEngine(data_dir=DATA, user_suffix='')
print("=== BOOTED ===", flush=True)

# Neutralise the internal-knowledge early return so we measure the
# generation path the task is about.
type(eng)._consult_internal_knowledge = lambda self, ui: None
eng._web_direct_answer = lambda ctx: None

import ravana.chat.response_gen as rg
import ravana.learn.explanation as expl
from ravana.chat.brain_regions import _singular_candidates

M = rg.ResponseGenMixin
REAL = M.__dict__['_why_how_causal_chain']      # the patched source


def _old_whc(self, ctx):
    import re as _re
    if not ctx or not ctx.subject:
        return None
    try:
        if not _re.search(r"\b(why|how|what\s+causes|what\s+makes|explain)\b",
                          (ctx.raw_input or ""), _re.IGNORECASE):
            return None
        _ckk = getattr(self, "_concept_keywords", {}) or {}
        _nids = [n for n in _ckk.get(ctx.subject, [])][:1]
        if not _nids:
            for _cand in _singular_candidates(ctx.subject):
                if _cand in _ckk:
                    _nids = [_ckk[_cand][0]]
                    break
        if not _nids:
            _n2 = self._graph_node_id_for_label(ctx.subject)
            if _n2 is not None:
                _nids = [_n2]
        if not _nids:
            return None
        _bw = bool(_re.search(r"\b(why|what\s+causes|what\s+makes)\b",
                              (ctx.raw_input or ""), _re.IGNORECASE))
        _ch = expl.enumerate_chains(_nids, self.graph,
                                    direction=("backward" if _bw else "forward"),
                                    min_steps=2, max_steps=4, limit=3)
        _ok = [(c, expl.grade_chain(c)) for c in _ch]
        _ok = [(c, g) for c, g in _ok
               if g["verdict"] == "assert" and expl.assertable_chain(c)]
        if not _ok:
            return None
        _b, _g = _ok[0]
        _l = _b.labels
        return (", and ".join(f"{_l[i]} leads to {_l[i+1]}" for i in range(len(_l) - 1)),
                "gated_explanation_chain")
    except Exception:
        return None


QS = ['why does it rain', 'why do you think it rains']
out = {}

for label, fn in (('OLD', _old_whc), ('NEW', REAL)):
    M._why_how_causal_chain = fn
    for q in QS:
        r = eng.process_turn(q)
        out[(label, q)] = r
        print("  [%s] %-28r -> %r" % (label, q, (r or '')[:120]), flush=True)

print("\n===== TURN-LEVEL BEFORE/AFTER =====", flush=True)
for q in QS:
    print("Q=%r\n  OLD: %r\n  NEW: %r"
          % (q, (out[('OLD', q)] or '')[:120], (out[('NEW', q)] or '')[:120]), flush=True)
print("DONE", flush=True)
