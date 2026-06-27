import os
import numpy as np
import pytest
import tempfile
from ravana_ml.graph import ConceptGraph
from ravana.storage.db import CognitiveDB, migrate_pickle_to_sqlite

def test_db_init_and_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_cognitive.db")
        db = CognitiveDB(db_path)
        
        # Test concept save
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        db.save_concept("1", "test_concept", "concrete", vec)
        
        # Verify in DB
        cursor = db.conn.execute("SELECT id, label, node_type, vector FROM concepts WHERE id = '1'")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "1"
        assert row[1] == "test_concept"
        assert row[2] == "concrete"
        
        # Check vector decoding
        decoded_vec = db._blob_to_vector(row[3])
        assert np.allclose(decoded_vec, vec)
        
        # Test edge save
        db.save_edge("1", "2", 0.8, "excitatory", "semantic")
        cursor = db.conn.execute("SELECT source_id, target_id, weight FROM edges WHERE source_id = '1'")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "1"
        assert row[1] == "2"
        assert row[2] == 0.8
        
        # Test episode save
        ep_id = db.save_episode(123456.7, "hello world", "1,2")
        assert ep_id > 0
        
        # Test metadata
        meta_val = {"some_key": [1, 2, 3]}
        db.save_metadata("config", meta_val)
        loaded_meta = db.load_metadata("config")
        assert loaded_meta == meta_val
        
        db.close()

def test_db_save_load_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_graph.db")
        
        graph = ConceptGraph(dim=4, max_nodes=50)
        n1 = graph.add_node(vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), label="apple")
        n2 = graph.add_node(vector=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), label="fruit")
        graph.add_edge(n1.id, n2.id, weight=0.7)
        
        db = CognitiveDB(db_path)
        db.save_graph(graph)
        
        # Load into a clean graph
        new_graph = ConceptGraph(dim=4, max_nodes=50)
        db.load_graph(new_graph)
        
        assert n1.id in new_graph.nodes
        assert n2.id in new_graph.nodes
        assert new_graph.nodes[n1.id].label == "apple"
        assert new_graph.nodes[n2.id].label == "fruit"
        
        edge = new_graph.get_edge(n1.id, n2.id)
        assert edge is not None
        assert edge.weight == 0.7
        
        db.close()
