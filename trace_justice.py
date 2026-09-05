import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\ravana_ml\src", f"{PROJ}\ravana\src", f"{PROJ}\ravana-v2\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="trace_justice")
eng._trace_enabled = True
r = eng.process_turn("tell me about justice")
print("=" * 60)
print(f"FINAL: {r.get('reply', r) if isinstance(r, dict) else r}")
print(f"Strategy: {eng._last_strategy}")
# Now examine what associations justice had
subj = "justice"
assocs = r.get('associated_concepts', []) if isinstance(r, dict) else []
print(f"Associations: {assocs}")
# Check graph edges for justice
nid_j = eng._concept_keywords.get(subj.lower(), [None])[0]
if nid_j is not None:
    print(f"Node ID for justice: {nid_j}")
    node = eng.graph.get_node(nid_j)
    if node:
        print(f"Node label: {node.label}")
    for tgt, e in eng.graph.get_outgoing(nid_j):
        tn = eng.graph.get_node(tgt)
        if tn:
            print(f"  OUT: {node.label} --[{e.relation_type} w={e.weight:.3f}]--> {tn.label}")
    for src, e in eng.graph.get_incoming(nid_j):
        sn = eng.graph.get_node(src)
        if sn:
            print(f"  IN: {sn.label} --[{e.relation_type} w={e.weight:.3f}]--> {node.label}")
else:
    print("justice not in graph")
# Also check all concepts that contain justice
for kw, nids in eng._concept_keywords.items():
    if "justice" in kw or kw in "justice":
        print(f"  concept_keyword: {kw} -> {nids}")
