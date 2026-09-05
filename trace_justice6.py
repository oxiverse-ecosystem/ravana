import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\ravana_ml\src", f"{PROJ}\ravana\src", f"{PROJ}\ravana-v2\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="trace_justice6")

# Trace hrr_query_chain for justice+causal
print("=== hrr_query_chain('justice', 'causal') ===")
chain = eng.hrr_query_chain("justice", "causal", max_hops=2)
print(f"Chain: {chain}")

# Also try a known concept like gravity
print("\n=== hrr_query_chain('gravity', 'causal') ===")
chain = eng.hrr_query_chain("gravity", "causal", max_hops=2)
print(f"Chain: {chain}")

# Check edge weight for justice->force
print("\n=== Edge weight check ===")
nid_j = eng._concept_keywords.get("justice", [None])[0]
if nid_j is not None:
    for tgt, e in eng.graph.get_outgoing(nid_j):
        if e.relation_type == "causal":
            tn = eng.graph.get_node(tgt)
            if tn:
                print(f"  justice --[causal w={e.weight:.3f}]--> {tn.label}")

# Check what the decomposition path does with confidence
print("\n=== Patching _structured_fact_answer to show confidence ===")
original = eng._structured_fact_answer
def patched(target, relation):
    result = original(target, relation)
    if result:
        print(f"  _structured_fact_answer('{target}', '{relation}') = '{result}'")
    return result
eng._structured_fact_answer = patched

# Run the full query
r = eng.process_turn("tell me about justice")
print(f"\nFINAL: {r.get('reply', r) if isinstance(r, dict) else r}")
