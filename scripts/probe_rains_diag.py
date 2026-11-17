"""Dense diagnostic probe for `why do you think it rains` vs `why does it rain`.

One process, many measurements. Monkeypatches are installed BEFORE any turn is
processed so every query is traced.
"""
import os
import sys
import traceback

sys.path.insert(0, 'ravana/src')

from ravana.chat.engine import CognitiveChatEngine

DATA = os.environ.get('RAVANA_PROBE_DATA', 'weights')
OFF = os.environ.get('RAVANA_OFFLINE', '')
if OFF:
    os.environ['RAVANA_OFFLINE'] = OFF

print("=== BOOT (data_dir=%s) ===" % DATA, flush=True)
eng = CognitiveChatEngine(data_dir=DATA, user_suffix='')
eng._web_direct_answer = lambda ctx: None
print("=== BOOTED ===", flush=True)

# ---------------------------------------------------------------- instrumentation
import ravana.chat.response_gen as rg
import ravana.learn.explanation as expl

M = rg.ResponseGenMixin

CALLS = {'gnidl': [], 'enum': [], 'grade': 0, 'assert': 0}

_orig_gnidl = M._graph_node_id_for_label


def _patched_gnidl(self, subject):
    r = _orig_gnidl(self, subject)
    CALLS['gnidl'].append((subject, r))
    return r


M._graph_node_id_for_label = _patched_gnidl

_orig_enum = expl.enumerate_chains


def _patched_enum(subject_ids, graph, **kw):
    CALLS['enum'].append((list(subject_ids), kw.get('direction')))
    return _orig_enum(subject_ids, graph, **kw)


expl.enumerate_chains = _patched_enum

_orig_grade = expl.grade_chain


def _patched_grade(chain, **kw):
    CALLS['grade'] += 1
    return _orig_grade(chain, **kw)


expl.grade_chain = _patched_grade

_orig_assert = expl.assertable_chain


def _patched_assert(chain):
    CALLS['assert'] += 1
    return _orig_assert(chain)


expl.assertable_chain = _patched_assert

# Trace the producing branch.
_orig_refl = M._reflective_response


def _patched_refl(self, ctx):
    r = _orig_refl(self, ctx)
    print("    [branch] _reflective_response -> %r" % (r,), flush=True)
    return r


M._reflective_response = _patched_refl

for _nm in ('_human_like_uncertainty', '_final_emit_guard', '_graph_fallback_response'):
    if hasattr(M, _nm):
        _o = getattr(M, _nm)

        def _mk(_nm=_nm, _o=_o):
            def _p(self, *a, **kw):
                r = _o(self, *a, **kw)
                print("    [branch] %s -> %r" % (_nm, (r[0] if isinstance(r, tuple) else r),), flush=True)
                return r
            return _p
        setattr(M, _nm, _mk())

_orig_whc = M._why_how_causal_chain


def _patched_whc(self, ctx):
    r = _orig_whc(self, ctx)
    print("    [branch] _why_how_causal_chain(%r) -> %r" % (getattr(ctx, 'subject', None), r), flush=True)
    return r


M._why_how_causal_chain = _patched_whc

# ---------------------------------------------------------------- graph census
g = getattr(eng, 'graph', None)
print("graph type=%r" % type(g).__name__, flush=True)
print("graph.nodes type=%r  len=%d" % (type(g.nodes).__name__, len(g.nodes)), flush=True)
_first = list(g.nodes)[:1]
print("list(g.nodes)[0] = %r (type %s)" % (_first[0], type(_first[0]).__name__), flush=True)
print("getattr(list(g.nodes)[0],'label',None) = %r" % getattr(_first[0], 'label', None), flush=True)

# correct iteration
labels = {}
for nid, node in g.nodes.items():
    lab = (getattr(node, 'label', '') or '').strip().lower()
    labels.setdefault(lab, nid)
print("distinct labels via .items() = %d" % len(labels), flush=True)
for probe in ('rain', 'rains', 'clouds', 'cloud', 'water vapour', 'water'):
    print("  label %-14r -> nid %r" % (probe, labels.get(probe)), flush=True)

# ---------------------------------------------------------------- learn
w = eng._get_web_to_graph()
for u in ('http://a/rain', 'http://b/rain'):
    w.learn_text('Water evaporates from the ocean. Water vapour condenses into clouds. Clouds release rain.',
                 source_url=u)
eng.save()
print("=== LEARNED ===", flush=True)

_ck = getattr(eng, '_concept_keywords', {}) or {}
print("_concept_keywords size=%d" % len(_ck), flush=True)
for k in ('rain', 'rains', 'clouds', 'cloud', 'water vapour', 'water'):
    print("  _ck.get(%-14r) = %r" % (k, _ck.get(k)), flush=True)

labels2 = {}
for nid, node in g.nodes.items():
    lab = (getattr(node, 'label', '') or '').strip().lower()
    labels2.setdefault(lab, nid)
for k in ('rain', 'rains', 'clouds', 'cloud', 'water vapour'):
    print("  post-learn label %-14r -> nid %r" % (k, labels2.get(k)), flush=True)

# ---------------------------------------------------------------- grounding
from ravana.chat.brain_regions import _singular_candidates, _singularize_subject

QS = ['why does it rain', 'why do you think it rains']
for q in QS:
    print("\n########## Q = %r ##########" % q, flush=True)
    try:
        gr = eng._ground_query(q)
        print("  _ground_query -> %r" % (gr,), flush=True)
        subj = None
        if isinstance(gr, dict):
            subj = gr.get('subject')
        else:
            subj = getattr(gr, 'subject', None)
        print("  subject=%r  singularize=%r  candidates=%r" % (
            subj, _singularize_subject(subj or ''), _singular_candidates(subj or '')), flush=True)
        for c in _singular_candidates(subj or ''):
            print("     cand %-10r in _ck? %s   in labels? %r" % (c, c in _ck, labels2.get(c)), flush=True)
    except Exception:
        traceback.print_exc()

    CALLS['gnidl'] = []
    CALLS['enum'] = []
    CALLS['grade'] = 0
    CALLS['assert'] = 0
    try:
        out = eng.process_turn(q)
    except Exception:
        traceback.print_exc()
        out = None
    print("  ANSWER = %r" % ((out or '')[:200],), flush=True)
    print("  _graph_node_id_for_label calls = %r" % (CALLS['gnidl'],), flush=True)
    print("  enumerate_chains calls = %r" % (CALLS['enum'],), flush=True)
    print("  grade_chain=%d assertable_chain=%d" % (CALLS['grade'], CALLS['assert']), flush=True)

print("\n=== DONE ===", flush=True)
