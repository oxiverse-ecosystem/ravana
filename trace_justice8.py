import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\ravana_ml\src", f"{PROJ}\ravana\src", f"{PROJ}\ravana-v2\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="trace_justice8")
eng._trace_enabled = True

# Check what happens inside _structured_fact_answer for justice+causal
# by tracing hrr_query_chain behavior

# First, check if justice is in the HRR store
print("=== HRR store keys ===")
if hasattr(eng, 'hrr_store'):
    store = eng.hrr_store
    print(f"Type: {type(store)}")
    # Try to find justice
    try:
        # Check if store has keys method
        if hasattr(store, 'keys'):
            keys = list(store.keys())
            print(f"Total keys: {len(keys)}")
            justice_keys = [k for k in keys if 'justice' in str(k).lower()]
            print(f"Justice-related keys: {justice_keys[:10]}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("No hrr_store attribute")

# Check hrr_query_chain directly
print("\n=== hrr_query_chain('justice', 'causal', max_hops=2) ===")
result = eng.hrr_query_chain("justice", "causal", max_hops=2)
print(f"Result: {result}")

# Check if justice is in _concept_keywords
print(f"\njustice in _concept_keywords: {'justice' in eng._concept_keywords}")
print(f"justice in _concept_labels: {'justice' in eng._concept_labels}")

# Check graph
print(f"\nGraph nodes: {len(eng.graph.nodes)}")
print(f"Graph edges: {len(eng.graph.edges)}")

# Check if justice is in the graph
for nid, node in eng.graph.nodes.items():
    if 'justice' in node.label.lower():
        print(f"Found: nid={nid} label={node.label}")

# Now let's manually trace _structured_fact_answer
print("\n=== Manual trace of _structured_fact_answer ===")
target = "justice"
relation = "causal"

# Step 1: hrr_query_chain
chain = eng.hrr_query_chain(target, relation, max_hops=2)
print(f"Step 1 - hrr_query_chain: {chain}")

# If chain is empty, _structured_fact_answer returns None
# But we saw it return "Justice leads to force." - so something is different

# Maybe the decomposition path activates the graph first?
# Let's check what _spread_and_collect does
print("\n=== _spread_and_collect for justice ===")
# First, we need to find justice in the graph
nid_j = None
for nid, node in eng.graph.nodes.items():
    if 'justice' in node.label.lower():
        nid_j = nid
        break

if nid_j is not None:
    print(f"Found justice at nid={nid_j}")
    # Activate it
    eng.graph.activate(nid_j, 0.8)
    # Now try hrr_query_chain again
    chain = eng.hrr_query_chain(target, relation, max_hops=2)
    print(f"hrr_query_chain after activate: {chain}")
else:
    print("justice not in graph")
