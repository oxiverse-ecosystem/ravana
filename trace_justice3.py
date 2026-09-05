import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\ravana_ml\src", f"{PROJ}\ravana\src", f"{PROJ}\ravana-v2\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="trace_justice3")

# Check causal chain
print("--- _causal_chain_from_graph('justice') ---")
cc = eng._causal_chain_from_graph("justice")
print(f"Result: {cc}")

# Check surface realizer for causal
print("\n--- Surface realizer test ---")
if hasattr(eng, 'syntactic_assembly') and hasattr(eng, 'surface_realizer'):
    frame = eng.syntactic_assembly.bind_to_sentence(
        subject="justice", relation="causal", target="force",
        pos_map=getattr(eng, '_concept_pos', {}),
    )
    from ravana.language.surface_realizer import DiscourseState
    disc = DiscourseState(sentence_index=0, discourse_type="explain", total_sentences=1, free_energy=0.3)
    sent = eng.surface_realizer.realize(frame=frame, discourse_context=disc, dopamine_tone=0.5)
    print(f"Realized: {sent}")

# Check what _web_direct_answer gives for "why does justice matter"
print("\n--- _web_direct_answer for 'why does justice matter' ---")
from ravana.chat.models import CognitiveResponseContext
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
wa = eng._web_direct_answer(ctx)
print(f"Web answer: {wa}")

# Check what _structured_fact_answer gives
print("\n--- _structured_fact_answer('justice', 'causal') ---")
sf = eng._structured_fact_answer("justice", "causal")
print(f"Structured: {sf}")

# Check _spread_and_collect for justice
print("\n--- _spread_and_collect for justice ---")
nid_j = eng._concept_keywords.get("justice", [None])[0]
if nid_j is not None:
    assocs = eng._spread_and_collect([nid_j], primary_ids={nid_j}, relation_preference=1.0)
    print(f"Associations: {assocs[:5]}")
