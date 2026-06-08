path = r"C:\Users\Likhith\Documents\projects\ravana\ravana_ml\nn\rlm_v2.py"
with open(path, 'r') as f:
    content = f.read()

# Replace the heuristic domain inference with explicit domain tracking on edges
# Add domain_id to edges when created
old_add_edge = '''        edge_direct = self.graph.get_edge(subject_cid, object_cid)
        if edge_direct is None:
            edge_direct = self.graph.add_edge(
                source=subject_cid, target=object_cid,
                weight=0.3, relation_type=rel_type_name,
            )'''

new_add_edge = '''        edge_direct = self.graph.get_edge(subject_cid, object_cid)
        if edge_direct is None:
            edge_direct = self.graph.add_edge(
                source=subject_cid, target=object_cid,
                weight=0.3, relation_type=rel_type_name,
            )
        # Store domain id on edge for cross-domain alignment
        edge_direct.domain_id = self.current_domain_id if self.current_domain_id is not None else 0'''

content = content.replace(old_add_edge, new_add_edge)

# Also for other edges
old_sr = '''        edge_sr = self.graph.get_edge(subject_cid, rel_obj_cid)
        if edge_sr is None:
            edge_sr = self.graph.add_edge(
                source=subject_cid, target=rel_obj_cid,
                weight=0.3, relation_type=rel_type_name,
            )'''

new_sr = '''        edge_sr = self.graph.get_edge(subject_cid, rel_obj_cid)
        if edge_sr is None:
            edge_sr = self.graph.add_edge(
                source=subject_cid, target=rel_obj_cid,
                weight=0.3, relation_type=rel_type_name,
            )
        edge_sr.domain_id = self.current_domain_id if self.current_domain_id is not None else 0'''

content = content.replace(old_sr, new_sr)

old_ro = '''        edge_ro = self.graph.get_edge(rel_obj_cid, object_cid)
        if edge_ro is None:
            edge_ro = self.graph.add_edge(
                source=rel_obj_cid, target=object_cid,
                weight=0.3, relation_type=rel_type_name,
            )'''

new_ro = '''        edge_ro = self.graph.get_edge(rel_obj_cid, object_cid)
        if edge_ro is None:
            edge_ro = self.graph.add_edge(
                source=rel_obj_cid, target=object_cid,
                weight=0.3, relation_type=rel_type_name,
            )
        edge_ro.domain_id = self.current_domain_id if self.current_domain_id is not None else 0'''

content = content.replace(old_ro, new_ro)

with open(path, 'w') as f:
    f.write(content)

print("Added domain_id to edges")