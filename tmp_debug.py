"""Debug relation type inference."""
from scripts.ravana_chat import CognitiveChatEngine
engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)

pairs = [
    ("good", "bad"),
    ("life", "death"),
    ("learn", "knowledge"),
    ("sun", "hot"),
    ("love", "hate"),
    ("trust", "hypocrisy"),
    ("question", "answer"),
]

for src, tgt in pairs:
    src_ids = engine._concept_keywords.get(src, [])
    tgt_ids = engine._concept_keywords.get(tgt, [])
    if src_ids and tgt_ids:
        e = engine.graph.get_edge(src_ids[0], tgt_ids[0])
        if not e:
            e = engine.graph.get_edge(tgt_ids[0], src_ids[0])
        if e:
            sn = engine.graph.get_node(src_ids[0])
            tn = engine.graph.get_node(tgt_ids[0])
            print(f"{sn.label:15s} -> {tn.label:15s} : {e.relation_type:15s} w={e.weight:.3f} conf={e.confidence:.3f}")
        else:
            sn = engine.graph.get_node(src_ids[0])
            tn = engine.graph.get_node(tgt_ids[0])
            print(f"{sn.label:15s} -> {tn.label:15s} : NO EDGE")
    else:
        print(f"{src:15s} -> {tgt:15s} : MISSING NODE")

# Check inference result directly
print("\nDirect inference test:")
src_lbl, tgt_lbl = "learn", "knowledge"
inf_type, inf_conf = engine._infer_relation_type(src_lbl, tgt_lbl, "semantic")
print(f"  {src_lbl} -> {tgt_lbl} : inferred={inf_type}, conf={inf_conf:.3f}")

src_lbl, tgt_lbl = "good", "bad"
inf_type, inf_conf = engine._infer_relation_type(src_lbl, tgt_lbl, "semantic")
print(f"  {src_lbl} -> {tgt_lbl} : inferred={inf_type}, conf={inf_conf:.3f}")

# Check if learn has any outgoing edges
lid = engine._concept_keywords.get("learn", [None])[0]
if lid is not None:
    print(f"\nAll edges from 'learn' (id={lid}):")
    for tid, e in engine.graph.get_outgoing(lid):
        tn = engine.graph.get_node(tid)
        lbl = tn.label if tn else "?"
        print(f"  -> {lbl}({tid}): type={e.relation_type}, w={e.weight:.3f}, conf={e.confidence:.3f}")
