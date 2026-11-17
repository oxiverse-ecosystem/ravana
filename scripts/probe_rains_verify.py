"""Final verification: exact repro, plus the emitting call chain.

Phase 0: OLD (first-hit-wins) resolution restored in-process -> capture the
         stack that emits the definition.
Phase 1: the real, patched source -> confirm both queries give the mechanism.
"""
import os
import sys
import traceback

sys.path.insert(0, 'ravana/src')

from ravana.chat.engine import CognitiveChatEngine

DATA = os.environ.get('RAVANA_PROBE_DATA', 'weights')
print("=== BOOT (%s) ===" % DATA, flush=True)
eng = CognitiveChatEngine(data_dir=DATA, user_suffix='')
eng._web_direct_answer = lambda ctx: None
print("=== BOOTED ===", flush=True)

import ravana.chat.response_gen as rg
import ravana.learn.explanation as expl
from ravana.chat.brain_regions import _singular_candidates

M = rg.ResponseGenMixin

w = eng._get_web_to_graph()
for u in ('http://e/rain', 'http://f/rain'):
    w.learn_text('Water evaporates from the ocean. Water vapour condenses into clouds. Clouds release rain.',
                 source_url=u)
eng.save()
print("=== LEARNED ===", flush=True)


def _rv_stack():
    out = []
    for fr in traceback.extract_stack()[:-1]:
        if 'ravana' not in (fr.filename or ''):
            continue
        out.append("      %s:%d  %s" % (os.path.basename(fr.filename), fr.lineno, fr.name))
    return out[-14:]


# ── instrument the definition emitter ────────────────────────────────────────
_orig_cik = type(eng)._consult_internal_knowledge


def _traced_cik(self, user_input):
    r = _orig_cik(self, user_input)
    print("  [cik] _consult_internal_knowledge(%r) -> %r" % (user_input[:40], (r or '')[:70]), flush=True)
    if r:
        print("  [cik]   emitting stack:", flush=True)
        for ln in _rv_stack():
            print(ln, flush=True)
    return r


type(eng)._consult_internal_knowledge = _traced_cik

_orig_gen = M._generate_response


def _traced_gen(self, ctx):
    r = _orig_gen(self, ctx)
    print("  [gen] _generate_response -> %r" % ((r[0] if isinstance(r, tuple) else r),), flush=True)
    return r


M._generate_response = _traced_gen

# ── PHASE 0: restore the OLD first-hit-wins resolution ───────────────────────
print("\n########## PHASE 0: OLD behaviour ##########", flush=True)


def _old_whc(self, ctx):
    import re as _re
    if not ctx or not ctx.subject:
        return None
    try:
        if not _re.search(r"\b(why|how|what\s+causes|what\s+makes|explain)\b",
                          (ctx.raw_input or ""), _re.IGNORECASE):
            return None
        _ck = getattr(self, "_concept_keywords", {}) or {}
        _nids = [n for n in _ck.get(ctx.subject, [])][:1]
        if not _nids:
            for _c in _singular_candidates(ctx.subject):
                if _c in _ck:
                    _nids = [_ck[_c][0]]
                    break
        if not _nids:
            _n2 = self._graph_node_id_for_label(ctx.subject)
            if _n2 is not None:
                _nids = [_n2]
        if not _nids:
            return None
        print("  [OLD] resolved nids=%r (first-hit-wins)" % (_nids,), flush=True)
        _bw = bool(_re.search(r"\b(why|what\s+causes|what\s+makes)\b",
                              (ctx.raw_input or ""), _re.IGNORECASE))
        _ch = expl.enumerate_chains(_nids, self.graph,
                                    direction=("backward" if _bw else "forward"),
                                    min_steps=2, max_steps=4, limit=3)
        _ok = [(c, expl.grade_chain(c)) for c in _ch]
        _ok = [(c, g) for c, g in _ok if g["verdict"] == "assert" and expl.assertable_chain(c)]
        if not _ok:
            return None
        _b, _g = _ok[0]
        _l = _b.labels
        return (", and ".join(f"{_l[i]} leads to {_l[i+1]}" for i in range(len(_l) - 1)),
                "gated_explanation_chain")
    except Exception:
        return None


M._why_how_causal_chain = _old_whc

o1 = eng.process_turn('why does it rain')
print("  [OLD] Q1 -> %r" % (o1 or '')[:150], flush=True)
o2 = eng.process_turn('why do you think it rains')
print("  [OLD] Q2 -> %r" % (o2 or '')[:150], flush=True)

# ── PHASE 1: the real patched source ─────────────────────────────────────────
print("\n########## PHASE 1: PATCHED source ##########", flush=True)
M._why_how_causal_chain = rg.ResponseGenMixin.__dict__['_why_how_causal_chain']

_orig_enum = expl.enumerate_chains


def _traced_enum(subject_ids, graph, **kw):
    print("  [NEW] enumerate_chains(nids=%r, dir=%r)" % (list(subject_ids), kw.get('direction')), flush=True)
    r = _orig_enum(subject_ids, graph, **kw)
    print("  [NEW]   -> %d chains %r" % (len(r), [c.labels for c in r]), flush=True)
    return r


expl.enumerate_chains = _traced_enum

n1 = eng.process_turn('why does it rain')
print("  [NEW] Q1 -> %r" % (n1 or '')[:150], flush=True)
n2 = eng.process_turn('why do you think it rains')
print("  [NEW] Q2 -> %r" % (n2 or '')[:150], flush=True)

print("\n===== SUMMARY =====", flush=True)
print("OLD Q1: %r" % (o1 or '')[:130], flush=True)
print("OLD Q2: %r" % (o2 or '')[:130], flush=True)
print("NEW Q1: %r" % (n1 or '')[:130], flush=True)
print("NEW Q2: %r" % (n2 or '')[:130], flush=True)
print("DONE", flush=True)
