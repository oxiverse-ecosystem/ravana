import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\ravana_ml\src", f"{PROJ}\ravana\src", f"{PROJ}\ravana-v2\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="trace_justice7")

# Check what _structured_fact_answer does step by step for justice+causal
# by reading the source

# Manually trace _structured_fact_answer
target = "justice"
relation = "causal"

print(f"=== Tracing _structured_fact_answer('{target}', '{relation}') ===")

# Step 1: hrr_query_chain
chain = eng.hrr_query_chain(target, relation, max_hops=2)
print(f"hrr_query_chain result: {chain}")

# If hrr_query_chain returns nothing, maybe the answer comes from a different path
# Let's check if there's a graph.get_edge fallback in _structured_fact_answer

# Actually, looking at the code, _structured_fact_answer only uses hrr_query_chain.
# But it returned "Justice leads to force." for justice+causal...
# This is confusing. Let me check if the HRR store has different data than the graph.

# Check HRR store directly
print("\n=== HRR store check ===")
if hasattr(eng, 'hrr_store'):
    store = eng.hrr_store
    print(f"HRR store type: {type(store)}")
    # Check what justice maps to
    try:
        result = store.query("justice", "causal", max_hops=2)
        print(f"HRR query result: {result}")
    except Exception as e:
        print(f"HRR query error: {e}")
else:
    print("No HRR store")

# Check graph edges for justice
print("\n=== Graph edges for justice ===")
nid_j = eng._concept_keywords.get("justice", [])
print(f"Node IDs for justice: {nid_j}")
if nid_j:
    n = nid_j[0]
    print(f"Node: {eng.graph.get_node(n)}")
    print("Outgoing:")
    for tgt, e in eng.graph.get_outgoing(n):
        tn = eng.graph.get_node(tgt)
        print(f"  -> [{e.relation_type} w={e.weight:.3f}] {tn.label if tn else '?'}")
    print("Incoming:")
    for src, e in eng.graph.get_incoming(n):
        sn = eng.graph.get_node(src)
        print(f"  <- [{e.relation_type} w={e.weight:.3f}] {sn.label if sn else '?'}")

# Check infer_chain
print("\n=== infer_chain check ===")
if hasattr(eng, 'infer_chain'):
    try:
        chain = eng.infer_chain("justice", "causal", max_hops=2)
        print(f"infer_chain result: {chain}")
    except Exception as e:
        print(f"infer_chain error: {e}")
