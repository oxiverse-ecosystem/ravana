import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\ravana_ml\src", f"{PROJ}\ravana\src", f"{PROJ}\ravana-v2\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="trace_justice5")
eng._trace_enabled = True

# Patch _structured_fact_answer to trace
original_sfa = eng._structured_fact_answer
def patched_sfa(target, relation):
    result = original_sfa(target, relation)
    if result:
        print(f"  [TRACE] _structured_fact_answer('{target}', '{relation}') = '{result[:60]}'")
    return result
eng._structured_fact_answer = patched_sfa

# Patch _web_direct_answer to trace
original_wda = eng._web_direct_answer
def patched_wda(ctx):
    result = original_wda(ctx)
    if result:
        text = result[0] if isinstance(result, tuple) else result
        print(f"  [TRACE] _web_direct_answer('{ctx.raw_input[:40]}', subj='{ctx.subject}') = '{text[:60]}'")
    return result
eng._web_direct_answer = patched_wda

# Patch _causal_chain_from_graph to trace
original_ccfg = eng._causal_chain_from_graph
def patched_ccfg(target):
    result = original_ccfg(target)
    if result:
        print(f"  [TRACE] _causal_chain_from_graph('{target}') = '{result[:60]}'")
    return result
eng._causal_chain_from_graph = patched_ccfg

# Patch _causal_restructure to trace
original_cr = eng._causal_restructure
def patched_cr(text, rel):
    result = original_cr(text, rel)
    if result:
        print(f"  [TRACE] _causal_restructure('{text[:40]}', '{rel}') = '{result[:60]}'")
    return result
eng._causal_restructure = patched_cr

r = eng.process_turn("tell me about justice")
print(f"\nFINAL: {r.get('reply', r) if isinstance(r, dict) else r}")
