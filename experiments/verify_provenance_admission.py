"""Verify provenance-favoring bare-edge admission on the LIVE path (research item E).

Calls CognitiveChatEngine._graph_fallback_response (chain_walker.py, the real
runtime path — strategy 'graph_fallback') with an injected edge in
ctx.associated_concepts + the graph, bypassing the definition short-circuit.
Asserts:
  - UNPROVENANCED auto_expand edge (the Q9/Q15 hollow class) is HEDGED, never
    stated as established fact;
  - VERIFIED web_fact edge keeps a confident surface form + "(per source)".
Mirrors scripts/ravana_chat.py import order.
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src"),
         os.path.join(_PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.models import CognitiveResponseContext
from ravana.chat.provenance import populate_provenance, provenance_class

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                          data_dir="/tmp/ravana_prov_test")
g = eng.graph


def _mk_edge(subj, obj, kind, **prov):
    nids = eng._concept_keywords.get(subj.lower())
    s = nids[0] if nids else g.add_node(label=subj)
    if subj.lower() not in eng._concept_keywords:
        eng._concept_keywords[subj.lower()] = [s]
    nido = eng._concept_keywords.get(obj.lower())
    o = nido[0] if nido else g.add_node(label=obj)
    if obj.lower() not in eng._concept_keywords:
        eng._concept_keywords[obj.lower()] = [o]
    e = g.add_edge(s, o, weight=0.6, relation_type="causal")
    populate_provenance(e, edge_kind=kind, **prov)
    if kind != "web_fact":
        e.source_metadata["source_url"] = None
        e.source_metadata["retrieval_conf"] = None
    return e


# Case 1: UNPROVENANCED auto_expand (Q9/Q15 hollow class).
e1 = _mk_edge("sleep", "depression", "auto_expand", method="auto_expand_glove")
eng._definitions.pop("sleep", None)  # bypass definition short-circuit
ctx1 = CognitiveResponseContext(subject="sleep", raw_input="tell me about sleep",
                                 associated_concepts=[("depression", 0.8)])

# Case 2: VERIFIED web_fact.
e2 = _mk_edge("gravity", "falling", "web_fact", source="wikipedia",
              source_url="https://en.wikipedia.org/wiki/Gravity",
              method="web_fact", retrieval_conf=0.9)
eng._definitions.pop("gravity", None)
ctx2 = CognitiveResponseContext(subject="gravity", raw_input="tell me about gravity",
                                 associated_concepts=[("falling", 0.8)])

out1, strat1 = eng._graph_fallback_response(ctx1)
out2, strat2 = eng._graph_fallback_response(ctx2)

print(f"[debug] e1 class={provenance_class(e1)}  e2 class={provenance_class(e2)}")
HEDGE = ("suggest", "link", "associat", "may", "maybe", "perhaps",
         "vaguely", "patterns", "connections", "i've seen", "my associations")
hedged = any(m in out1.lower() for m in HEDGE)
confident_web = ("per source" in out2.lower()) and ("gravity" in out2.lower())

print(f"[E] unprovenanced auto_expand -> {out1!r}  [{strat1}]")
print(f"    hedged={hedged}  (expect True: never 'Sleep leads to depression.')")
print(f"[E] verified web_fact      -> {out2!r}  [{strat2}]")
print(f"    cites_source={confident_web}  (expect True: '(per source)')")

ok = hedged and confident_web
print("\nVERDICT:", "CONFIRMED — hollow bare-edge hedged, verified web fact cited."
      if ok else "CHECK — a provenance assertion failed.")
eng.stop_background_learning()
raise SystemExit(0 if ok else 1)
