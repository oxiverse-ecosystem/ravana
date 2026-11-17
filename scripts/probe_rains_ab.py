"""A/B the REAL patched source against the OLD first-hit-wins logic, on
IDENTICAL graph state (no learning, no save, offline).

Directly calls `_why_how_causal_chain` with a synthetic ctx, so this measures
exactly the function under test and nothing else.
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

import ravana.learn.explanation as expl
from ravana.chat.brain_regions import _singular_candidates

_ck = getattr(eng, '_concept_keywords', {}) or {}
g = eng.graph
print("_ck['rains']=%r  _ck['rain']=%r" % (_ck.get('rains'), _ck.get('rain')), flush=True)
for lab in ('rains', 'rain', 'clouds', 'water vapour'):
    for nid, n in g.nodes.items():
        if (getattr(n, 'label', '') or '').strip().lower() == lab:
            print("  node %-4d label=%-14r in=%d out=%d"
                  % (nid, lab, len(g.get_incoming(nid) or []), len(g.get_outgoing(nid) or [])), flush=True)
            break


def _ctx(raw, subject):
    return types.SimpleNamespace(raw_input=raw, subject=subject,
                                 associated_concepts=[])


def _old_whc(self, ctx):
    """Verbatim pre-fix resolution (response_gen.py @HEAD, lines 8270-8298)."""
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
            _nid2 = self._graph_node_id_for_label(ctx.subject)
            if _nid2 is not None:
                _nids = [_nid2]
        if not _nids:
            return None
        print("    [OLD] nids=%r" % (_nids,), flush=True)
        _backward = bool(_re.search(r"\b(why|what\s+causes|what\s+makes)\b",
                                    (ctx.raw_input or ""), _re.IGNORECASE))
        _chains = expl.enumerate_chains(
            _nids, self.graph, direction=("backward" if _backward else "forward"),
            min_steps=2, max_steps=4, limit=3)
        _ok = [(c, expl.grade_chain(c)) for c in _chains]
        _ok = [(c, gg) for c, gg in _ok
               if gg["verdict"] == "assert" and expl.assertable_chain(c)]
        if not _ok:
            return None
        _best, _gg = _ok[0]
        _labels = _best.labels
        return (", and ".join(f"{_labels[i]} leads to {_labels[i+1]}"
                              for i in range(len(_labels) - 1)),
                "gated_explanation_chain")
    except Exception:
        import traceback
        traceback.print_exc()
        return None


QS = [("why does it rain", "rain"),
      ("why do you think it rains", "rains")]

print("\n########## A/B on identical state ##########", flush=True)
for raw, subj in QS:
    c = _ctx(raw, subj)
    print("\n--- %r  (subject=%r) ---" % (raw, subj), flush=True)
    old = _old_whc(eng, c)
    print("  OLD (HEAD)  -> %r" % (old,), flush=True)
    new = eng._why_how_causal_chain(c)          # REAL patched source
    print("  NEW (PATCH) -> %r" % (new,), flush=True)

# also prove the graph-label resolver is now live
print("\n--- _graph_node_id_for_label (real, patched) ---", flush=True)
for s in ("rains", "rain", "clouds", "water vapour"):
    print("  %-14r -> %r" % (s, eng._graph_node_id_for_label(s)), flush=True)

print("\nDONE", flush=True)
