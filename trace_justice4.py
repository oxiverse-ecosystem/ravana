import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\ravana_ml\src", f"{PROJ}\ravana\src", f"{PROJ}\ravana-v2\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="trace_justice4")
eng._trace_enabled = True

# Manually trace what happens in decomposition for "justice"
# The key question: where does "Justice leads to force." come from?

# Check _web_direct_answer for the causal sub-question
from ravana.chat.models import CognitiveResponseContext

# Simulate what happens for sq3: "why does justice matter (causal)"
# sq_target would be None (no explicit target), so ctx.subject = "justice"
print("=== Simulating causal sub-question ===")
ctx = CognitiveResponseContext(
    subject="justice",
    raw_input="why does justice matter",
    associated_concepts=[],
    valence=0.0, arousal=0.5, dominance=0.5,
    emotional_label=None,
    identity_strength=0.5,
    processing_route="dorsal",
    turn_count=1,
)

# Try A: web_direct_answer
wa = eng._web_direct_answer(ctx)
print(f"web_direct_answer: {wa}")

# Try B: stored definition
print(f"justice in _definitions: {'justice' in eng._definitions}")

# Try C: _tc_known
_tc_known = bool(
    ("justice" in getattr(eng, '_definitions', {}))
    or ("justice" in getattr(eng, '_concept_sources', {})))
print(f"_tc_known for justice: {_tc_known}")

# Since _tc_known is False, Try C is skipped. So answer_text stays empty.
# But then how does "Justice leads to force." appear?

# Let me check if there's a Try D or something after Try C
# Actually, looking at the code again, after Try C there's no more code that sets answer_text
# So answer_text should be empty for this sub-question

# BUT WAIT: the trace shows "Justice leads to force." as the final synthesis.
# Let me check if the synthesis can produce text even when answer_text is empty

# Actually, maybe the issue is that the causal sub-question DID get an answer
# from the web, but it was a junk answer that passed the web_direct_answer filter

# Let me trace more carefully by patching the method
import ravana.chat.response_gen as rg
original_method = rg.ResponseGenMixin._decomposition_generation_path

def patched_method(self, ctx):
    result = original_method(self, ctx)
    # After the method runs, check what sub-answers exist
    if result:
        decomp = getattr(ctx, 'decomposition', None)
        if decomp:
            for sq in decomp.sub_questions:
                ans = getattr(sq, 'answer', '')
                conf = getattr(sq, 'confidence', 0.0)
                if ans:
                    print(f"  [PATCH] sq[{sq.id}] {sq.text[:40]} -> ans='{ans[:60]}' conf={conf}")
    return result

rg.ResponseGenMixin._decomposition_generation_path = patched_method

r = eng.process_turn("tell me about justice")
print(f"\nFINAL: {r.get('reply', r) if isinstance(r, dict) else r}")
