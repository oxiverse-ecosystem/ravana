"""Confirm the root cause and validate the fix, in ONE process (boot is slow).

Phase A: measure why `_why_how_causal_chain('rains')` returns None.
Phase B: apply the corrected resolution in-process, re-run, compare.
"""
import os
import sys

sys.path.insert(0, 'ravana/src')

from ravana.chat.engine import CognitiveChatEngine

DATA = os.environ.get('RAVANA_PROBE_DATA', 'weights')
print("=== BOOT (data_dir=%s) ===" % DATA, flush=True)
eng = CognitiveChatEngine(data_dir=DATA, user_suffix='')
eng._web_direct_answer = lambda ctx: None
print("=== BOOTED ===", flush=True)

import ravana.chat.response_gen as rg
import ravana.learn.explanation as expl
from ravana.chat.brain_regions import _singular_candidates

M = rg.ResponseGenMixin

w = eng._get_web_to_graph()
for u in ('http://c/rain', 'http://d/rain'):
    w.learn_text('Water evaporates from the ocean. Water vapour condenses into clouds. Clouds release rain.',
                 source_url=u)
eng.save()
print("=== LEARNED ===", flush=True)

# ------------------------------------------------------------------ PHASE A
print("\n########## PHASE A: BEFORE ##########", flush=True)

SEEN = {}


def _dump(subject, tag):
    ck = getattr(eng, '_concept_keywords', {}) or {}
    g = eng.graph
    print("  [%s] subject=%r" % (tag, subject), flush=True)
    print("  [%s] _ck.get(subject)      = %r" % (tag, ck.get(subject)), flush=True)
    for c in _singular_candidates(subject):
        print("  [%s]   cand %-8r -> _ck %r" % (tag, c, ck.get(c)), flush=True)
    for c in _singular_candidates(subject):
        nids = ck.get(c) or []
        for nid in nids:
            n = g.nodes.get(nid)
            lab = getattr(n, 'label', None)
            inc = g.get_incoming(nid) or []
            out = g.get_outgoing(nid) or []
            print("  [%s]   node %-4d label=%-14r in=%d out=%d" % (
                tag, nid, lab, len(inc), len(out)), flush=True)
            for o, e in inc[:6]:
                on = g.nodes.get(o)
                print("  [%s]      IN  %-16r -%s-> %r" % (
                    tag, getattr(on, 'label', None),
                    getattr(e, 'relation_type', '?'), lab), flush=True)


_orig_whc = M._why_how_causal_chain


def _traced_whc(self, ctx):
    subj = getattr(ctx, 'subject', None)
    _dump(subj, 'A')
    r = _orig_whc(self, ctx)
    print("  [A] _why_how_causal_chain(%r) -> %r" % (subj, r), flush=True)
    SEEN['nid'] = subj
    return r


M._why_how_causal_chain = _traced_whc

_orig_enum = expl.enumerate_chains


def _traced_enum(subject_ids, graph, **kw):
    print("  [A] enumerate_chains(nids=%r, dir=%r)" % (list(subject_ids), kw.get('direction')), flush=True)
    r = _orig_enum(subject_ids, graph, **kw)
    print("  [A]   -> %d chains %r" % (len(r), [c.labels for c in r]), flush=True)
    return r


expl.enumerate_chains = _traced_enum

a1 = eng.process_turn('why does it rain')
print("  [A] Q1 'why does it rain'      -> %r" % (a1 or '')[:160], flush=True)
a2 = eng.process_turn('why do you think it rains')
print("  [A] Q2 'why do you think it rains' -> %r" % (a2 or '')[:160], flush=True)

# ------------------------------------------------------------------ PHASE B
print("\n########## PHASE B: AFTER (corrected resolution) ##########", flush=True)


def _fixed_gnidl(self, subject):
    """Same as response_gen.py:8219 but iterating node OBJECTS, not dict keys."""
    g = getattr(self, "graph", None)
    if g is None or not subject:
        return None
    try:
        nodes = list(g.nodes.values())
    except Exception:
        return None
    for _cand in _singular_candidates(subject):
        for n in nodes:
            try:
                lab = (getattr(n, "label", None) or "").lower().strip()
            except Exception:
                continue
            if lab == _cand:
                return getattr(n, "id", None)
    return None


M._graph_node_id_for_label = _fixed_gnidl


def _fixed_whc(self, ctx):
    """Resolution collects EVERY candidate nid (deduped, ordered) instead of
    stopping at the first `_ck` hit."""
    import re as _re
    if not ctx or not ctx.subject:
        return None
    try:
        if not _re.search(r"\b(why|how|what\s+causes|what\s+makes|explain)\b",
                          (ctx.raw_input or ""), _re.IGNORECASE):
            return None
        _ck = getattr(self, "_concept_keywords", {}) or {}
        _nids, _seen = [], set()

        def _add(x):
            if x is not None and x not in _seen:
                _seen.add(x)
                _nids.append(x)

        for _n in (_ck.get(ctx.subject) or []):
            _add(_n)
        for _cand in _singular_candidates(ctx.subject):
            for _n in (_ck.get(_cand) or []):
                _add(_n)
        _add(self._graph_node_id_for_label(ctx.subject))
        if not _nids:
            return None
        print("  [B] candidate nids = %r" % (_nids,), flush=True)
        _backward = bool(_re.search(r"\b(why|what\s+causes|what\s+makes)\b",
                                    (ctx.raw_input or ""), _re.IGNORECASE))
        _chains = expl.enumerate_chains(
            _nids, self.graph,
            direction=("backward" if _backward else "forward"),
            min_steps=2, max_steps=4, limit=3)
        _ok = [(c, expl.grade_chain(c)) for c in _chains]
        _ok = [(c, g) for c, g in _ok
               if g["verdict"] == "assert" and expl.assertable_chain(c)]
        if not _ok:
            return None
        _best, _g = _ok[0]
        _labels = _best.labels
        _pairs = [f"{_labels[i]} leads to {_labels[i + 1]}"
                  for i in range(len(_labels) - 1)]
        _narr = ", and ".join(_pairs)
        print("  [B] chain=%r adequacy=%.3f" % (_narr, _g['adequacy']), flush=True)
        return (_narr, "gated_explanation_chain")
    except Exception:
        import traceback
        traceback.print_exc()
        return None


M._why_how_causal_chain = _fixed_whc

b1 = eng.process_turn('why does it rain')
print("  [B] Q1 'why does it rain'      -> %r" % (b1 or '')[:160], flush=True)
b2 = eng.process_turn('why do you think it rains')
print("  [B] Q2 'why do you think it rains' -> %r" % (b2 or '')[:160], flush=True)

print("\n===== SUMMARY =====", flush=True)
print("BEFORE Q1: %r" % (a1 or '')[:120], flush=True)
print("BEFORE Q2: %r" % (a2 or '')[:120], flush=True)
print("AFTER  Q1: %r" % (b1 or '')[:120], flush=True)
print("AFTER  Q2: %r" % (b2 or '')[:120], flush=True)
print("DONE", flush=True)
