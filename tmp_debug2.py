"""Deep debug: trace learn-knowledge edge creation."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Patch _apply_edges to log
from scripts.ravana_chat import CognitiveChatEngine

orig_apply = CognitiveChatEngine._apply_edges

def debug_apply(self, label_to_id, edge_list, rel_type, base_weight):
    for src, tgt in edge_list:
        if src == "learn" and tgt == "knowledge":
            print(f"[DEBUG] _apply_edges: {src} -> {tgt} with type={rel_type}")
            sid = label_to_id.get(src)
            tid = label_to_id.get(tgt)
            existing = self.graph.get_edge(sid, tid) if sid is not None and tid is not None else None
            if existing:
                print(f"  EXISTING: type={existing.relation_type}, w={existing.weight:.3f}")
            else:
                print(f"  NEW edge")
    orig_apply(self, label_to_id, edge_list, rel_type, base_weight)

CognitiveChatEngine._apply_edges = debug_apply

# Also patch the hub edge creation
orig_hub = CognitiveChatEngine._seed_concepts
def debug_seed(self):
    orig_hub(self)
    # After all seeding, check
    src_ids = self._concept_keywords.get("learn", [])
    tgt_ids = self._concept_keywords.get("knowledge", [])
    if src_ids and tgt_ids:
        e = self.graph.get_edge(src_ids[0], tgt_ids[0])
        if not e:
            e = self.graph.get_edge(tgt_ids[0], src_ids[0])
        if e:
            print(f"[DEBUG POST-SEED] learn -> knowledge: type={e.relation_type}, w={e.weight:.3f}, conf={e.confidence:.3f}")
CognitiveChatEngine._seed_concepts = debug_seed

engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
