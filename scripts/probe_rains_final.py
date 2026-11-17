"""CLEAN verification of the patched source. No monkeypatching of
_why_how_causal_chain -- this exercises the real shipped code.

Runs the exact repro from the task description.
"""
import os
import sys

sys.path.insert(0, 'ravana/src')

from ravana.chat.engine import CognitiveChatEngine

DATA = os.environ.get('RAVANA_PROBE_DATA', 'weights')
print("=== BOOT (%s) ===" % DATA, flush=True)
eng = CognitiveChatEngine(data_dir=DATA, user_suffix='')
eng._web_direct_answer = lambda ctx: None
print("=== BOOTED ===", flush=True)

import ravana.chat.response_gen as rg
import ravana.learn.explanation as expl

M = rg.ResponseGenMixin
print("sanity: _why_how_causal_chain owner = %s"
      % [k.__name__ for k in type(eng).__mro__
         if '_why_how_causal_chain' in k.__dict__], flush=True)

w = eng._get_web_to_graph()
for u in ('http://g/rain', 'http://h/rain'):
    w.learn_text('Water evaporates from the ocean. Water vapour condenses into clouds. Clouds release rain.',
                 source_url=u)
eng.save()
print("=== LEARNED ===", flush=True)

_orig_enum = expl.enumerate_chains


def _traced_enum(subject_ids, graph, **kw):
    print("  [enum] nids=%r dir=%r" % (list(subject_ids), kw.get('direction')), flush=True)
    r = _orig_enum(subject_ids, graph, **kw)
    print("  [enum]   -> %d chains %r" % (len(r), [c.labels for c in r]), flush=True)
    return r


expl.enumerate_chains = _traced_enum

_orig_gnidl = M._graph_node_id_for_label


def _traced_gnidl(self, subject):
    r = _orig_gnidl(self, subject)
    print("  [gnidl] %r -> %r" % (subject, r), flush=True)
    return r


M._graph_node_id_for_label = _traced_gnidl

_ck = getattr(eng, '_concept_keywords', {}) or {}
print("_ck['rains']=%r  _ck['rain']=%r" % (_ck.get('rains'), _ck.get('rain')), flush=True)

res = {}
for q in ('why does it rain', 'why do you think it rains'):
    print("\n--- Q = %r ---" % q, flush=True)
    res[q] = eng.process_turn(q)
    print("  ANSWER = %r" % (res[q] or '')[:160], flush=True)

print("\n===== FINAL =====", flush=True)
for q, a in res.items():
    print("Q=%s\n  -> %r" % (q, (a or '')[:160]), flush=True)
print("DONE", flush=True)
