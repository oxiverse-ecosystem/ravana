import sqlite3
import numpy as np
import pickle
import os
import time
from typing import Dict, List, Tuple, Optional, Any
from ravana_ml.graph import ConceptGraph, ConceptNode, ConceptEdge, ConceptNodeType

class CognitiveDB:
    """SQLite database manager for persistent storage of the Cognitive Graph.
    
    Replaces the fragile pickle-based storage with an ACID-compliant, cross-platform
    SQLite database in WAL (Write-Ahead Logging) mode.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.init_db()
        
    def init_db(self):
        """Initialize the database schema and enable WAL mode."""
        self.conn = sqlite3.connect(self.db_path, timeout=10.0)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        
        # Create Tables
        with self.conn:
            # Concepts Table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS concepts (
                    id TEXT PRIMARY KEY,
                    label TEXT,
                    node_type TEXT,
                    vector BLOB,
                    sensorimotor BLOB,
                    stability REAL,
                    confidence REAL,
                    created_at REAL
                )
            """)
            
            # Edges Table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    source_id TEXT,
                    target_id TEXT,
                    weight REAL,
                    edge_type TEXT,
                    relation_type TEXT,
                    confidence REAL,
                    stability REAL,
                    PRIMARY KEY (source_id, target_id)
                )
            """)
            
            # Episodes Table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    content TEXT,
                    concepts TEXT
                )
            """)
            
            # Metadata Table (to store general key-value configs, RNG states, etc.)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value BLOB
                )
            """)

    def close(self):
        """Close connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _vector_to_blob(self, vec: Optional[np.ndarray]) -> Optional[bytes]:
        if vec is None:
            return None
        return vec.astype(np.float32).tobytes()

    def _blob_to_vector(self, blob: Optional[bytes]) -> Optional[np.ndarray]:
        if blob is None:
            return None
        return np.frombuffer(blob, dtype=np.float32).copy()

    def save_concept(self, node_id: str, label: str, node_type: str,
                     vector: np.ndarray, sensorimotor: Optional[np.ndarray] = None,
                     stability: float = 0.5, confidence: float = 0.1):
        """Upsert a concept node in the database."""
        vector_blob = self._vector_to_blob(vector)
        sm_blob = self._vector_to_blob(sensorimotor)
        
        with self.conn:
            self.conn.execute("""
                INSERT INTO concepts (id, label, node_type, vector, sensorimotor, stability, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label = excluded.label,
                    node_type = excluded.node_type,
                    vector = excluded.vector,
                    sensorimotor = excluded.sensorimotor,
                    stability = excluded.stability,
                    confidence = excluded.confidence
            """, (node_id, label, node_type, vector_blob, sm_blob, stability, confidence, time.time()))

    def save_edge(self, source_id: str, target_id: str, weight: float,
                  edge_type: str = "excitatory", relation_type: str = "semantic",
                  confidence: float = 0.5, stability: float = 0.3):
        """Upsert a concept edge in the database."""
        with self.conn:
            self.conn.execute("""
                INSERT INTO edges (source_id, target_id, weight, edge_type, relation_type, confidence, stability)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, target_id) DO UPDATE SET
                    weight = excluded.weight,
                    edge_type = excluded.edge_type,
                    relation_type = excluded.relation_type,
                    confidence = excluded.confidence,
                    stability = excluded.stability
            """, (source_id, target_id, weight, edge_type, relation_type, confidence, stability))

    def save_episode(self, timestamp: float, content: str, concepts: str) -> int:
        """Insert a new episode/memory log. Returns the generated ID."""
        with self.conn:
            cursor = self.conn.execute("""
                INSERT INTO episodes (timestamp, content, concepts)
                VALUES (?, ?, ?)
            """, (timestamp, content, concepts))
            return cursor.lastrowid

    def save_metadata(self, key: str, value: Any):
        """Store arbitrary python objects serialized in metadata table."""
        blob = pickle.dumps(value)
        with self.conn:
            self.conn.execute("""
                INSERT INTO metadata (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """, (key, blob))

    def load_metadata(self, key: str) -> Optional[Any]:
        """Load and deserialize value from metadata table."""
        cursor = self.conn.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return pickle.loads(row[0])
        return None

    def save_graph(self, graph: ConceptGraph):
        """Save the entire ConceptGraph to SQLite database."""
        with self.conn:
            # First clean up old entries
            self.conn.execute("DELETE FROM concepts")
            self.conn.execute("DELETE FROM edges")
            
        # Bulk save nodes
        for node_id, node in graph.nodes.items():
            self.save_concept(
                node_id=str(node_id),
                label=node.label,
                node_type=getattr(node, 'node_type', ConceptNodeType.CONCRETE),
                vector=node.vector,
                sensorimotor=getattr(node, 'sensorimotor_vector', None),
                stability=node.stability,
                confidence=node.confidence
            )
            
        # Bulk save edges
        for edge_key, edge in graph.edges.items():
            self.save_edge(
                source_id=str(edge.source),
                target_id=str(edge.target),
                weight=edge.weight,
                edge_type=getattr(edge, 'edge_type', 'excitatory'),
                relation_type=getattr(edge, 'relation_type', 'semantic'),
                confidence=edge.confidence,
                stability=edge.stability
            )

    def load_graph(self, graph: ConceptGraph):
        """Load all nodes and edges from SQLite into the given ConceptGraph."""
        # 1. Load concepts
        cursor = self.conn.execute("SELECT id, label, node_type, vector, sensorimotor, stability, confidence FROM concepts")
        for row in cursor.fetchall():
            node_id_str, label, node_type, vec_blob, sm_blob, stability, confidence = row
            node_id = int(node_id_str)
            vector = self._blob_to_vector(vec_blob)
            
            # Make sure we don't duplicate nodes
            if node_id in graph.nodes:
                node = graph.nodes[node_id]
                if vector is not None:
                    node.vector = vector.copy()
            else:
                if vector is None:
                    vector = np.zeros(graph.dim, dtype=np.float32)
                node = ConceptNode(node_id, vector, label=label, node_type=node_type)
                graph.nodes[node_id] = node
                graph.next_id = max(graph.next_id, node_id + 1)
                graph._vectors_dirty = True
                graph._adj_dirty = True
                if hasattr(graph, 'version'):
                    graph.version += 1
                
            node.node_type = node_type
            node.stability = stability
            node.confidence = confidence
            if sm_blob is not None:
                node.sensorimotor_vector = self._blob_to_vector(sm_blob)
                
        # 2. Load edges
        cursor = self.conn.execute("SELECT source_id, target_id, weight, edge_type, relation_type, confidence, stability FROM edges")
        for row in cursor.fetchall():
            s_id_str, t_id_str, weight, edge_type, relation_type, confidence, stability = row
            s_id, t_id = int(s_id_str), int(t_id_str)
            
            # Add or update edge
            edge = graph.get_edge(s_id, t_id)
            if not edge:
                # Add edge to graph (handles bi-directional if configured in graph.py)
                graph.add_edge(s_id, t_id, weight=weight)
                edge = graph.get_edge(s_id, t_id)
                
            if edge:
                edge.weight = weight
                edge.edge_type = edge_type
                edge.relation_type = relation_type
                edge.confidence = confidence
                edge.stability = stability


def migrate_pickle_to_sqlite(pickle_path: str, db_path: str) -> bool:
    """Migrate cognitive state from a legacy pickle file to SQLite.
    
    Extracts the graph, episodes, user model and other variables.
    """
    if not os.path.exists(pickle_path):
        return False
        
    try:
        with open(pickle_path, 'rb') as f:
            state = pickle.load(f)
    except Exception as e:
        print(f"Failed to read legacy pickle file: {e}")
        return False
        
    db = CognitiveDB(db_path)
    
    # 1. Migrate Graph
    graph = state.get('graph')
    if graph:
        db.save_graph(graph)
        
    # 2. Migrate Metadata and other state keys
    for key, value in state.items():
        if key == 'graph':
            continue
        try:
            db.save_metadata(key, value)
        except Exception as e:
            print(f"Failed to migrate key {key}: {e}")
            
    db.close()
    return True
