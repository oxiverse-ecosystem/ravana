"""
RAVANA v2 — HUMAN MEMORY ENGINE
Persistent episodic/semantic memory with Ebbinghaus decay, consolidation,
and associative recall via spreading activation.

PRINCIPLE: Memory is not storage — it is reconstruction shaped by emotion,
repetition, and time. Memories consolidate through use and decay through
neglect, mirroring human forgetting curves.
"""

import sqlite3
import time
import math
import heapq
import re
import os
import pickle
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple

try:
    import networkx as nx
except ImportError:
    nx = None


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class HumanMemoryConfig:
    """Configuration for the human memory engine."""
    # Persistence
    db_path: str = "human_memory.db"
    graph_path: str = "human_memory_graph.pickle"

    # Ebbinghaus decay
    base_decay_rate: float = 0.01
    retention_threshold: float = 10.0
    hard_delete_days: int = 90

    # Working memory (Miller's Law)
    working_slots: int = 7

    # Consolidation
    consolidation_access_threshold: int = 2

    # Spreading activation
    spreading_decay: float = 0.5
    spreading_max_depth: int = 3
    spreading_max_nodes: int = 10

    # Graph edge parameters
    auto_link_weight: float = 1.5
    edge_cap: float = 5.0
    edge_reinforce_mult: float = 1.15

    # In-memory trace limit
    max_history: int = 1000

    # Memory entropy
    entropy_distortion_rate: float = 0.01  # how much each recall distorts
    entropy_divergence_rate: float = 0.005  # how fast associations drift
    coherence_decay_rate: float = 0.002  # coherence loss per decay cycle
    stability_gain_per_access: float = 0.05  # stability grows with use

    # Utility-aware modulation
    utility_weight: float = 0.4  # weight of predictive utility in persistence
    emotional_weight: float = 0.3  # weight of emotional salience (reduced from 0.5)
    access_weight: float = 0.15  # weight of access frequency
    coherence_weight: float = 0.15  # weight of coherence in persistence

    # Identity interaction
    identity_importance_boost: float = 0.2  # importance boost when identity is strong
    identity_emotional_boost: float = 0.15  # emotional boost when identity is under pressure
    identity_strength_threshold: float = 0.7  # above this = strong identity
    identity_pressure_threshold: float = -0.1  # below this delta = under pressure

    # Synonym map for semantic search (optional)
    synonym_map: Optional[Dict[str, List[str]]] = None


# ─── Output Record ───────────────────────────────────────────────────────────

@dataclass
class HumanMemoryRecord:
    """Output from process_step — what happened this cycle."""
    episode: int
    memory_id: int
    memory_type: str
    salience: float
    decay_score: float
    consolidated: bool
    associations_activated: int
    coherence: float = 1.0
    stability: float = 0.5
    retrieval_distortion: float = 0.0
    associative_divergence: float = 0.0


# ─── Default Synonyms ────────────────────────────────────────────────────────

_DEFAULT_SYNONYMS: Dict[str, List[str]] = {
    "python": ["python", "pytorch", "cpython", "cython", "py"],
    "ai": ["ai", "ml", "machinelearning", "deeplearn", "artificial", "intelligence"],
    "learning": ["learning", "study", "training", "education"],
    "web": ["web", "http", "www", "internet", "browser"],
    "code": ["code", "programming", "coding", "script"],
    "data": ["data", "dataset", "database", "db", "sql"],
    "model": ["model", "network", "neuralnet", "transformer", "llm"],
    "search": ["search", "query", "retrieve", "find", "lookup"],
    "memory": ["memory", "storage", "recall", "store", "persist"],
    "agent": ["agent", "bot", "agentic", "autonomous"],
    "api": ["api", "rest", "endpoint", "http", "request"],
    "js": ["javascript", "js", "node", "nodejs"],
    "ts": ["typescript", "ts", "typing"],
}


# ─── Lightweight Graph (dict-based fallback) ─────────────────────────────────

class _DictGraph:
    """Lightweight graph when NetworkX is not available."""

    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._adj: Dict[str, Set[str]] = {}

    def has_node(self, nid: str) -> bool:
        return nid in self._nodes

    def add_node(self, nid: str, **attrs):
        if nid not in self._nodes:
            self._nodes[nid] = {}
            self._adj[nid] = set()
        self._nodes[nid].update(attrs)

    def remove_node(self, nid: str):
        if nid in self._nodes:
            del self._nodes[nid]
            for neighbor in list(self._adj.get(nid, set())):
                self._adj[neighbor].discard(nid)
                key = (min(nid, neighbor), max(nid, neighbor))
                self._edges.pop(key, None)
            self._adj.pop(nid, None)

    def has_edge(self, a: str, b: str) -> bool:
        key = (min(a, b), max(a, b))
        return key in self._edges

    def add_edge(self, a: str, b: str, **attrs):
        key = (min(a, b), max(a, b))
        self._edges[key] = attrs
        self._adj.setdefault(a, set()).add(b)
        self._adj.setdefault(b, set()).add(a)

    def get_edge_data(self, a: str, b: str) -> Optional[Dict[str, Any]]:
        key = (min(a, b), max(a, b))
        return self._edges.get(key)

    def neighbors(self, nid: str) -> List[str]:
        return list(self._adj.get(nid, set()))

    def nodes(self) -> Dict[str, Dict[str, Any]]:
        return self._nodes

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        return len(self._edges)


# ─── Engine ──────────────────────────────────────────────────────────────────

class HumanMemoryEngine:
    """
    Persistent episodic/semantic memory with Ebbinghaus decay,
    consolidation, and associative recall via spreading activation.

    Integrates with ravana's StateManager via process_step().
    """

    def __init__(self, config: Optional[HumanMemoryConfig] = None):
        self.config = config or HumanMemoryConfig()
        self.working_memory: List[dict] = []
        self._history: List[HumanMemoryRecord] = []
        self._graph: Any = None
        self._init_db()
        self._load_graph()

    # ─── SQLite Layer ────────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.config.db_path)

    def _init_db(self):
        db_dir = os.path.dirname(self.config.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                memory_type TEXT DEFAULT 'experience',
                semantic_type TEXT,
                context TEXT,
                tags TEXT,
                importance REAL DEFAULT 0.5,
                emotional REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL,
                created_at REAL,
                decay_score REAL DEFAULT 0.0,
                suppressed INTEGER DEFAULT 0,
                suppressed_at REAL,
                consolidated INTEGER DEFAULT 0,
                consolidation_score REAL DEFAULT 0.0,
                causal_id INTEGER REFERENCES memories(id),
                procedural_id INTEGER REFERENCES memories(id),
                coherence REAL DEFAULT 1.0,
                stability REAL DEFAULT 0.5,
                retrieval_distortion REAL DEFAULT 0.0,
                associative_divergence REAL DEFAULT 0.0,
                predictive_utility REAL DEFAULT 0.5
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                accessed_at REAL,
                recall_depth INTEGER DEFAULT 0,
                reinforced INTEGER DEFAULT 0,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_suppressed ON memories(suppressed)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_decay ON memories(decay_score)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_access_log_memory ON access_log(memory_id)")
        # Migration: add temporal_context column if missing (for existing databases)
        try:
            conn.execute("ALTER TABLE memories ADD COLUMN temporal_context BLOB")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.commit()
        conn.close()

    def _store(self, content: str, memory_type: str = "experience",
               semantic_type: str = None, context: str = None,
               tags: str = None, importance: float = 0.5,
               emotional: float = 0.5,
               temporal_context: Optional[np.ndarray] = None) -> int:
        conn = self._get_conn()
        now = time.time()
        ctx_blob = temporal_context.tobytes() if temporal_context is not None else None
        try:
            cursor = conn.execute(
                """INSERT INTO memories (content, memory_type, semantic_type, context, tags,
                                         importance, emotional, access_count, last_accessed,
                                         created_at, decay_score, coherence, stability,
                                         retrieval_distortion, associative_divergence,
                                         predictive_utility, temporal_context)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, ?)""",
                (content, memory_type, semantic_type, context, tags,
                 importance, emotional, now, now, ctx_blob)
            )
        except sqlite3.OperationalError:
            # Fallback if temporal_context column doesn't exist yet
            cursor = conn.execute(
                """INSERT INTO memories (content, memory_type, semantic_type, context, tags,
                                         importance, emotional, access_count, last_accessed,
                                         created_at, decay_score, coherence, stability,
                                         retrieval_distortion, associative_divergence,
                                         predictive_utility)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5)""",
                (content, memory_type, semantic_type, context, tags,
                 importance, emotional, now, now)
            )
        mid = cursor.lastrowid
        conn.commit()
        conn.close()
        return int(mid)

    def _recall(self, keyword: str = "", memory_type: str = None,
                limit: int = 20) -> List[dict]:
        conn = self._get_conn()
        params: list = []
        conditions = ["suppressed = 0"]
        if keyword:
            conditions.append("(content LIKE ? OR tags LIKE ? OR context LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"])
        if memory_type:
            conditions.append("memory_type = ?")
            params.append(memory_type)
        where = " AND ".join(conditions)
        params.append(limit)
        rows = conn.execute(
            f"""SELECT id, content, memory_type, semantic_type, context, tags,
                       importance, emotional, access_count, decay_score, consolidated
                FROM memories WHERE {where}
                ORDER BY (importance * 0.30 + emotional * 0.50
                         + access_count * 0.002
                         + (1.0 - MIN(decay_score, 10.0) / 10.0) * 0.20) DESC
                LIMIT ?""",
            params
        ).fetchall()
        recalled_ids = [r[0] for r in rows]
        if recalled_ids:
            placeholders = ",".join("?" * len(recalled_ids))
            cfg = self.config
            # Each recall distorts slightly and strengthens stability
            conn.execute(
                f"""UPDATE memories SET
                        access_count = access_count + 1,
                        last_accessed = ?,
                        retrieval_distortion = MIN(1.0, retrieval_distortion + ?),
                        stability = MIN(1.0, stability + ?),
                        coherence = MAX(0.0, coherence - ? * retrieval_distortion)
                    WHERE id IN ({placeholders})""",
                [time.time(), cfg.entropy_distortion_rate,
                 cfg.stability_gain_per_access, cfg.entropy_distortion_rate] + recalled_ids
            )
            conn.commit()
        conn.close()
        return [self._row_to_dict_short(r) for r in rows]

    def _get(self, memory_id: int) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            """SELECT id, content, memory_type, semantic_type, context, tags,
                      importance, emotional, access_count, last_accessed, created_at,
                      decay_score, suppressed, suppressed_at, consolidated, consolidation_score,
                      coherence, stability, retrieval_distortion, associative_divergence,
                      predictive_utility
               FROM memories WHERE id = ?""", (memory_id,)
        ).fetchone()
        conn.close()
        return self._row_to_dict_full(row) if row else None

    def _get_active(self, limit: int = 100) -> List[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, content, memory_type, semantic_type, context, tags,
                      importance, emotional, access_count, decay_score, consolidated,
                      coherence, stability, retrieval_distortion, associative_divergence,
                      predictive_utility
               FROM memories WHERE suppressed = 0
               ORDER BY (importance * 0.4 + emotional * 0.4 + access_count * 0.0002) DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()
        conn.close()
        return [self._row_to_dict_short(r) for r in rows]

    def _reinforce(self, memory_id: int):
        conn = self._get_conn()
        conn.execute(
            """UPDATE memories
               SET importance = MIN(1.0, importance + 0.05),
                   emotional = MIN(1.0, emotional + 0.03),
                   decay_score = MAX(0.0, decay_score - 5.0),
                   access_count = access_count + 1,
                   last_accessed = ?
               WHERE id = ?""",
            (time.time(), memory_id)
        )
        conn.commit()
        conn.close()

    def _forget(self, memory_id: int, hard: bool = False):
        conn = self._get_conn()
        if hard:
            conn.execute("DELETE FROM access_log WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        else:
            conn.execute(
                "UPDATE memories SET suppressed = 1, suppressed_at = ? WHERE id = ?",
                (time.time(), memory_id)
            )
        conn.commit()
        conn.close()

    def _compute_interference(self, mid: int, conn: sqlite3.Connection) -> float:
        """Compute interference from similar recent memories.

        New similar memories accelerate decay of existing ones (retroactive interference).
        Returns a multiplier from 1.0 (no interference) to 2.0 (high interference).
        """
        recent_cutoff = time.time() - 7 * 86400  # last 7 days
        # Find tags for this memory
        row = conn.execute(
            "SELECT tags FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        if not row or not row[0]:
            return 1.0
        tags = [t.strip() for t in row[0].split(",") if t.strip()]
        if not tags:
            return 1.0
        # Count recent memories with overlapping tags
        interference_count = 0
        for tag in tags[:5]:  # limit to avoid expensive queries
            count = conn.execute(
                """SELECT COUNT(*) FROM memories
                   WHERE suppressed = 0 AND id != ?
                   AND tags LIKE ? AND created_at > ?""",
                (mid, f"%{tag}%", recent_cutoff)
            ).fetchone()[0]
            interference_count += count
        return 1.0 + min(1.0, interference_count * 0.1)

    def _apply_decay(self) -> Dict[str, int]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, decay_score, importance, emotional,
                      coherence, stability, retrieval_distortion, associative_divergence,
                      predictive_utility, access_count
               FROM memories WHERE suppressed = 0"""
        ).fetchall()
        decayed = suppressed = 0
        cfg = self.config
        for (mid, decay, imp, emo, coh, stab, rdist, adist, putil, acc) in rows:
            # Utility-aware persistence: emotional salience alone doesn't guarantee survival
            access_score = min(1.0, acc * 0.01)
            persistence = (
                cfg.utility_weight * putil +
                cfg.emotional_weight * emo +
                cfg.access_weight * access_score +
                cfg.coherence_weight * coh
            )
            # Entropy-modulated: low coherence accelerates decay
            entropy_mod = (coh * 0.5 + stab * 0.5)
            # Higher persistence = slower decay (divide, not multiply)
            retention_mod = persistence * 0.9 + 0.1
            # Interference: similar recent memories accelerate decay
            interference = self._compute_interference(mid, conn)
            delta = cfg.base_decay_rate * (1.5 - entropy_mod) / retention_mod * interference
            new_decay = decay + delta
            # Coherence slowly erodes over time
            new_coherence = max(0.0, coh - cfg.coherence_decay_rate)
            # Associative divergence grows slowly
            new_adist = min(1.0, adist + cfg.entropy_divergence_rate)
            if new_decay >= cfg.retention_threshold:
                conn.execute("UPDATE memories SET suppressed = 1, suppressed_at = ? WHERE id = ?",
                             (time.time(), mid))
                suppressed += 1
            else:
                conn.execute(
                    """UPDATE memories SET decay_score = ?, coherence = ?,
                              associative_divergence = ? WHERE id = ?""",
                    (new_decay, new_coherence, new_adist, mid))
            decayed += 1
        conn.commit()
        # Auto-consolidation
        threshold_acc = self.config.consolidation_access_threshold
        consolidated = conn.execute(
            """UPDATE memories SET consolidated = 1,
                      consolidation_score = MIN(1.0, consolidation_score + 0.5)
               WHERE suppressed = 0 AND consolidated = 0
                     AND access_count >= ? AND consolidation_score < 1.0""",
            (threshold_acc,)
        ).rowcount
        conn.commit()
        conn.close()
        return {"decayed": decayed, "suppressed": suppressed,
                "processed": decayed, "consolidated": consolidated}

    def _consolidate(self) -> int:
        conn = self._get_conn()
        threshold = self.config.consolidation_access_threshold
        count = conn.execute(
            """UPDATE memories SET consolidated = 1, consolidation_score = 1.0
               WHERE suppressed = 0 AND consolidated = 0
                     AND access_count >= ? AND consolidation_score < 1.0""",
            (threshold,)
        ).rowcount
        conn.commit()
        conn.close()
        return count

    def _purge(self) -> int:
        conn = self._get_conn()
        cutoff = time.time() - self.config.hard_delete_days * 86400
        conn.execute(
            "DELETE FROM access_log WHERE memory_id IN (SELECT id FROM memories WHERE suppressed = 1 AND suppressed_at < ?)",
            (cutoff,)
        )
        count = conn.execute(
            "DELETE FROM memories WHERE suppressed = 1 AND suppressed_at < ?", (cutoff,)
        ).rowcount
        conn.commit()
        conn.close()
        return count

    @staticmethod
    def _row_to_dict_short(row: tuple) -> dict:
        d = {
            "id": row[0], "content": row[1], "memory_type": row[2],
            "semantic_type": row[3], "context": row[4], "tags": row[5],
            "importance": row[6], "emotional": row[7], "access_count": row[8],
            "decay_score": row[9], "consolidated": row[10] == 1,
        }
        if len(row) > 11:
            d["coherence"] = row[11]
            d["stability"] = row[12]
            d["retrieval_distortion"] = row[13]
            d["associative_divergence"] = row[14]
        if len(row) > 15:
            d["predictive_utility"] = row[15]
        return d

    @staticmethod
    def _row_to_dict_full(row: tuple) -> dict:
        d = {
            "id": row[0], "content": row[1], "memory_type": row[2],
            "semantic_type": row[3], "context": row[4], "tags": row[5],
            "importance": row[6], "emotional": row[7], "access_count": row[8],
            "last_accessed": row[9], "created_at": row[10],
            "decay_score": row[11], "suppressed": row[12],
            "suppressed_at": row[13], "consolidated": row[14] == 1,
            "consolidation_score": row[15],
        }
        if len(row) > 16:
            d["coherence"] = row[16]
            d["stability"] = row[17]
            d["retrieval_distortion"] = row[18]
            d["associative_divergence"] = row[19]
        if len(row) > 20:
            d["predictive_utility"] = row[20]
        return d

    # ─── Graph Layer ─────────────────────────────────────────────────────

    def _load_graph(self):
        path = self.config.graph_path
        if nx is not None:
            pickle_path = path.replace(".gml", ".pickle")
            if os.path.exists(pickle_path):
                try:
                    with open(pickle_path, "rb") as f:
                        self._graph = pickle.load(f)
                        return
                except Exception:
                    pass
            if os.path.exists(path):
                try:
                    self._graph = nx.read_gml(path)
                    return
                except Exception:
                    pass
            self._graph = nx.Graph()
        else:
            # Dict-based fallback
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        self._graph = pickle.load(f)
                        if isinstance(self._graph, _DictGraph):
                            return
                except Exception:
                    pass
            self._graph = _DictGraph()

    def _save_graph(self):
        path = self.config.graph_path
        pickle_path = path.replace(".gml", ".pickle")
        db_dir = os.path.dirname(pickle_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        with open(pickle_path, "wb") as f:
            pickle.dump(self._graph, f)

    @staticmethod
    def _node_id(memory_id: int) -> str:
        return f"mem-{memory_id}"

    def _add_node(self, memory_id: int, content: str = "",
                  memory_type: str = "experience", tags: str = "",
                  decay_score: float = 0.0):
        nid = self._node_id(memory_id)
        attrs = dict(content=content, memory_type=memory_type, tags=tags,
                     decay_score=decay_score, last_accessed=time.time(),
                     reinforcement=0)
        if nx is not None and isinstance(self._graph, nx.Graph):
            if self._graph.has_node(nid):
                self._graph.nodes[nid].update(attrs)
            else:
                self._graph.add_node(nid, **attrs)
        else:
            self._graph.add_node(nid, **attrs)

    def _link_memories(self, id_a: int, id_b: int, weight: float = 1.0):
        na, nb = self._node_id(id_a), self._node_id(id_b)
        cfg = self.config
        if nx is not None and isinstance(self._graph, nx.Graph):
            for nid in (na, nb):
                if not self._graph.has_node(nid):
                    self._graph.add_node(nid)
            if self._graph.has_edge(na, nb):
                self._graph[na][nb]["weight"] = min(
                    self._graph[na][nb]["weight"] * cfg.edge_reinforce_mult, cfg.edge_cap)
            else:
                self._graph.add_edge(na, nb, weight=weight, label="association",
                                     created_at=time.time())
        else:
            for nid in (na, nb):
                if not self._graph.has_node(nid):
                    self._graph.add_node(nid)
            if self._graph.has_edge(na, nb):
                ed = self._graph.get_edge_data(na, nb)
                ed["weight"] = min(ed["weight"] * cfg.edge_reinforce_mult, cfg.edge_cap)
            else:
                self._graph.add_edge(na, nb, weight=weight, label="association",
                                     created_at=time.time())

    def _spreading_activation(self, seed_ids: List[str]) -> List[Tuple[str, float]]:
        G = self._graph
        cfg = self.config
        seed_set = {s for s in seed_ids if (nx is not None and isinstance(G, nx.Graph) and G.has_node(s))
                     or (isinstance(G, _DictGraph) and G.has_node(s))}
        if not seed_set:
            return []

        activation: Dict[str, float] = {}
        for s in seed_set:
            activation[s] = 1.0
        frontier = {n: 1.0 for n in seed_set}
        visited = set(seed_set)

        for _ in range(cfg.spreading_max_depth):
            next_frontier: Dict[str, float] = {}
            for node in frontier:
                if nx is not None and isinstance(G, nx.Graph):
                    neighbors = list(G.neighbors(node))
                else:
                    neighbors = G.neighbors(node)
                for nb in neighbors:
                    if nb in visited:
                        continue
                    if nx is not None and isinstance(G, nx.Graph):
                        edge_weight = G[node][nb].get("weight", 1.0) if G.has_edge(node, nb) else 1.0
                    else:
                        ed = G.get_edge_data(node, nb)
                        edge_weight = ed.get("weight", 1.0) if ed else 1.0
                    contrib = frontier[node] * cfg.spreading_decay * edge_weight / (len(neighbors) + 1)
                    activation[nb] = activation.get(nb, 0.0) + contrib
                    next_frontier[nb] = next_frontier.get(nb, 0.0) + contrib
                    visited.add(nb)
            frontier = next_frontier
            if not frontier:
                break

        scored = sorted(activation.items(), key=lambda x: -x[1])
        return heapq.nlargest(cfg.spreading_max_nodes, scored, key=lambda x: x[1])

    def _associate_recall(self, memory_id: int) -> List[int]:
        nid = self._node_id(memory_id)
        results = self._spreading_activation([nid])
        output = []
        for (n, score) in results:
            if n != nid and score > 0.01:
                try:
                    mid = int(n.split("-")[1])
                    output.append(mid)
                except (ValueError, IndexError):
                    pass
        return output

    # ─── Working Memory ──────────────────────────────────────────────────

    def _push_working(self, memory_id: int):
        m = self._get(memory_id)
        if not m:
            return
        self.working_memory = [x for x in self.working_memory if x["id"] != memory_id]
        self.working_memory.insert(0, m)
        if len(self.working_memory) > self.config.working_slots:
            self.working_memory.pop()

    # ─── Semantic Search ─────────────────────────────────────────────────

    @staticmethod
    def _stem(word: str) -> str:
        w = word.lower()
        for suffix in ['ing', 'ed', 'es', 's', 'tion', 'ness', 'ly', 'ment']:
            if w.endswith(suffix) and len(w) > len(suffix) + 2:
                return w[:-len(suffix)]
        return w

    def _get_synonyms(self) -> Dict[str, List[str]]:
        if self.config.synonym_map is not None:
            return self.config.synonym_map
        return _DEFAULT_SYNONYMS

    # ─── Public API ──────────────────────────────────────────────────────

    def process_step(self, episode_data: Dict[str, Any],
                     state_snapshot: Dict[str, Any]) -> HumanMemoryRecord:
        """Called by StateManager each cognitive step. Stores the episode as a memory."""
        episode = episode_data.get("episode", 0)
        # Derive salience from emotional state
        vad = episode_data.get("vad", {})
        valence = abs(vad.get("valence", 0.0)) if isinstance(vad, dict) else 0.0
        arousal = vad.get("arousal", 0.3) if isinstance(vad, dict) else 0.3
        salience = min(1.0, valence * 0.4 + arousal * 0.6)

        # Derive importance from dissonance delta
        pre_d = episode_data.get("pre_dissonance", 0.5)
        post_d = episode_data.get("post_dissonance", 0.5)
        importance = min(1.0, max(0.1, 0.5 + (pre_d - post_d)))

        # Derive emotional weight from meaning
        meaning = episode_data.get("meaning", 0.0)
        emotional = min(1.0, max(0.1, 0.5 + abs(meaning) * 0.5))

        # Identity interaction: shape memory by identity state
        cfg = self.config
        pre_identity = episode_data.get("pre_identity", 0.5)
        post_identity = episode_data.get("post_identity", 0.5)
        identity_delta = post_identity - pre_identity

        # Strong identity boosts importance (coherent self makes memories stickier)
        if pre_identity >= cfg.identity_strength_threshold:
            importance = min(1.0, importance + cfg.identity_importance_boost * pre_identity)

        # Identity under pressure boosts emotional weight (crisis memories are vivid)
        if identity_delta <= cfg.identity_pressure_threshold:
            emotional = min(1.0, emotional + cfg.identity_emotional_boost * abs(identity_delta))

        # Build content string
        content = (
            f"Episode {episode}: dissonance {pre_d:.3f}->{post_d:.3f}, "
            f"identity {episode_data.get('pre_identity', 0):.3f}->{episode_data.get('post_identity', 0):.3f}, "
            f"wisdom {episode_data.get('wisdom', 0):.3f}, meaning {meaning:.3f}"
        )

        # Auto-tag from episode data
        tags_parts = []
        route = episode_data.get("processing_route", "")
        if route:
            tags_parts.append(route)
        mode = episode_data.get("mode", "")
        if mode:
            tags_parts.append(mode)
        if episode_data.get("sleep_triggered"):
            tags_parts.append("sleep")
        gw = episode_data.get("gw_broadcast", {})
        if gw and gw.get("source"):
            tags_parts.append(f"gw:{gw['source']}")
        # Identity-derived tags
        if pre_identity >= cfg.identity_strength_threshold:
            tags_parts.append("identity_strong")
        if identity_delta <= cfg.identity_pressure_threshold:
            tags_parts.append("identity_pressure")
        elif identity_delta >= 0.05:
            tags_parts.append("identity_growth")
        tags = ",".join(tags_parts)

        memory_type = "experience"
        if episode_data.get("resolution", {}).get("full_resolution"):
            memory_type = "reflection" if meaning > 0.3 else "semantic"

        # Predictive utility: how much did this episode reduce dissonance?
        # High dissonance reduction = high predictive value
        dissonance_reduction = max(0.0, pre_d - post_d)
        predictive_utility = min(1.0, 0.3 + dissonance_reduction * 2.0 + abs(meaning) * 0.3)

        mid = self._store(
            content=content,
            memory_type=memory_type,
            tags=tags,
            importance=importance,
            emotional=emotional,
        )

        # Add graph node and auto-link
        self._add_node(mid, content=content, memory_type=memory_type,
                       tags=tags, decay_score=0.0)
        if tags:
            existing = self._get_active(limit=100)
            tag_set = set(t.strip() for t in tags.split(",") if t.strip())
            for m in existing:
                if m["id"] == mid:
                    continue
                m_tags = set(t.strip() for t in (m.get("tags") or "").split(",") if t.strip())
                if tag_set & m_tags:
                    if not self._graph.has_node(self._node_id(m["id"])):
                        self._add_node(m["id"], content=m.get("content", ""),
                                       memory_type=m.get("memory_type", "experience"),
                                       tags=m.get("tags", ""), decay_score=m.get("decay_score", 0.0))
                    self._link_memories(mid, m["id"], weight=self.config.auto_link_weight)

        self._save_graph()
        self._push_working(mid)

        # Get association count
        assoc_ids = self._associate_recall(mid)

        # Update predictive utility in DB
        conn = self._get_conn()
        conn.execute("UPDATE memories SET predictive_utility = ? WHERE id = ?",
                     (predictive_utility, mid))
        conn.commit()
        conn.close()

        record = HumanMemoryRecord(
            episode=episode,
            memory_id=mid,
            memory_type=memory_type,
            salience=salience,
            decay_score=0.0,
            consolidated=False,
            associations_activated=len(assoc_ids),
        )
        self._history.append(record)
        if len(self._history) > self.config.max_history:
            self._history = self._history[-self.config.max_history:]

        return record

    def remember(self, content: str, memory_type: str = "experience",
                 importance: float = 0.5, emotional: float = 0.5,
                 tags: str = "", context: str = "",
                 temporal_context: Optional[np.ndarray] = None) -> int:
        """Explicit memory storage. Returns memory id.

        Args:
            temporal_context: Optional numpy array representing the temporal context
                            at the time of encoding. Enables encoding-specific retrieval.
        """
        mid = self._store(content=content, memory_type=memory_type,
                          tags=tags, context=context,
                          importance=importance, emotional=emotional,
                          temporal_context=temporal_context)
        self._add_node(mid, content=content, memory_type=memory_type,
                       tags=tags, decay_score=0.0)
        if tags:
            existing = self._get_active(limit=100)
            tag_set = set(t.strip() for t in tags.split(",") if t.strip())
            for m in existing:
                if m["id"] == mid:
                    continue
                m_tags = set(t.strip() for t in (m.get("tags") or "").split(",") if t.strip())
                if tag_set & m_tags:
                    if not self._graph.has_node(self._node_id(m["id"])):
                        self._add_node(m["id"], content=m.get("content", ""),
                                       memory_type=m.get("memory_type", "experience"),
                                       tags=m.get("tags", ""), decay_score=m.get("decay_score", 0.0))
                    self._link_memories(mid, m["id"], weight=self.config.auto_link_weight)
        self._save_graph()
        self._push_working(mid)
        return mid

    def store_temporal_context(self, memory_id: int, context_vector: np.ndarray):
        """Store temporal context for a memory (encoding specificity)."""
        conn = self._get_conn()
        try:
            conn.execute(
                "UPDATE memories SET temporal_context = ? WHERE id = ?",
                (context_vector.tobytes(), memory_id)
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column might not exist in old databases
        conn.close()

    def get_temporal_context(self, memory_id: int) -> Optional[np.ndarray]:
        """Retrieve temporal context for a memory."""
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT temporal_context FROM memories WHERE id = ?",
                (memory_id,)
            ).fetchone()
            if row and row[0]:
                return np.frombuffer(row[0], dtype=np.float32).copy()
        except (sqlite3.OperationalError, ValueError):
            pass
        conn.close()
        return None

    def recall(self, keyword: str = "", limit: int = 10,
               associative: bool = False,
               memory_id: Optional[int] = None) -> List[dict]:
        """Multi-mode recall: keyword, associative, or recent."""
        if associative and memory_id is not None:
            ids = self._associate_recall(memory_id)
            results = []
            for rid in ids:
                m = self._get(rid)
                if m:
                    results.append(m)
            return results
        if keyword:
            words = keyword.split()
            if len(words) == 1:
                return self._recall(keyword, limit=limit)
            seen: Set[int] = set()
            results: List[dict] = []
            for w in words:
                for m in self._recall(w, limit=limit * 3):
                    if m["id"] not in seen:
                        seen.add(m["id"])
                        results.append(m)
            results.sort(key=lambda x: -(x.get("importance", 0) * 0.30
                                         + x.get("emotional", 0) * 0.50))
            return results[:limit * 2]
        return self._get_active(limit=limit)

    def retrieval_induced_forgetting(self, recalled_ids: List[int], keyword: str = "",
                                      suppression: float = 0.01) -> int:
        """After recall, suppress competing memories that match the query but weren't recalled.

        This implements retrieval-induced forgetting (Anderson et al., 1994):
        actively recalling some memories suppresses related competitors,
        preventing interference and improving future recall efficiency.

        Args:
            recalled_ids: IDs of memories that were successfully recalled
            keyword: The query used for recall
            suppression: How much to reduce importance of competitors

        Returns:
            Number of memories suppressed
        """
        if not keyword or not recalled_ids:
            return 0
        conn = self._get_conn()
        recalled_set = set(recalled_ids)
        # Find memories that match the keyword but weren't recalled
        competitors = conn.execute(
            """SELECT id FROM memories
               WHERE suppressed = 0 AND (content LIKE ? OR tags LIKE ?)
               AND id NOT IN ({})""".format(",".join("?" * len(recalled_ids))),
            [f"%{keyword}%", f"%{keyword}%"] + recalled_ids
        ).fetchall()
        count = 0
        for (cid,) in competitors:
            if cid not in recalled_set:
                conn.execute(
                    """UPDATE memories SET importance = MAX(0.0, importance - ?),
                               stability = MAX(0.0, stability - ?) WHERE id = ?""",
                    (suppression, suppression * 0.5, cid)
                )
                count += 1
        conn.commit()
        conn.close()
        return count

    def semantic_search(self, keyword: str, limit: int = 10,
                        memory_type: str = None) -> List[dict]:
        """Fuzzy search using stemmed token overlap + synonym expansion."""
        synonyms = self._get_synonyms()
        query_terms = set(self._stem(t) for t in keyword.replace("_", " ").split())
        for term in list(query_terms):
            if term in synonyms:
                query_terms.update(self._stem(s) for s in synonyms[term])

        all_mems = self._get_active(limit=200)
        scored: List[dict] = []
        for m in all_mems:
            if memory_type and m.get("memory_type") != memory_type:
                continue
            content_tokens = set(self._stem(t) for t in
                                 ((m.get("content", "") or "") + " " + (m.get("tags", "") or "")).split())
            overlap = query_terms & content_tokens
            if overlap:
                score = len(overlap) / max(len(query_terms), 1)
                m["match_score"] = round(score, 3)
                m["matched_terms"] = list(overlap)
                scored.append(m)
        scored.sort(key=lambda x: -x["match_score"])
        return scored[:limit]

    def reconstructive_recall(self, keyword: str, limit: int = 5) -> List[dict]:
        """Reconstruct memories from fragments + graph structure.

        When direct matches are sparse, uses spreading activation from
        partial matches to rebuild richer context from associative neighbors.
        """
        # First: direct semantic search
        direct = self.semantic_search(keyword, limit=limit * 2)
        if not direct:
            return []

        # Check if results are sparse (low match scores mean fragments)
        strong_matches = [m for m in direct if m.get("match_score", 0) >= 0.5]
        weak_matches = [m for m in direct if m.get("match_score", 0) < 0.5]

        reconstructed = list(strong_matches)

        # For weak matches, reconstruct from graph neighbors
        for seed in weak_matches[:limit]:
            nid = self._node_id(seed["id"])
            activated = self._spreading_activation([nid])

            # Gather context from graph neighbors
            neighbor_contexts = []
            neighbor_tags = set()
            for (node_id, score) in activated:
                if node_id == nid or score < 0.05:
                    continue
                try:
                    neighbor_mid = int(node_id.split("-")[1])
                except (ValueError, IndexError):
                    continue
                neighbor = self._get(neighbor_mid)
                if neighbor:
                    neighbor_contexts.append(neighbor.get("content", ""))
                    for t in (neighbor.get("tags") or "").split(","):
                        t = t.strip()
                        if t:
                            neighbor_tags.add(t)

            # Build reconstructed memory
            if neighbor_contexts:
                seed_content = seed.get("content", "")
                # Blend: seed content + neighbor fragments
                blended_content = seed_content
                if len(neighbor_contexts) <= 3:
                    blended_content += " [associated: " + "; ".join(neighbor_contexts[:3]) + "]"

                # Fidelity: how much of the reconstruction is direct vs inferred
                direct_ratio = seed.get("match_score", 0)
                graph_ratio = min(1.0, len(neighbor_contexts) * 0.2)
                fidelity = direct_ratio * 0.6 + graph_ratio * 0.4

                reconstructed_entry = dict(seed)
                reconstructed_entry["content"] = blended_content
                reconstructed_entry["reconstructed"] = True
                reconstructed_entry["reconstruction_fidelity"] = round(fidelity, 3)
                reconstructed_entry["neighbor_count"] = len(neighbor_contexts)
                reconstructed_entry["reconstructed_tags"] = ",".join(sorted(neighbor_tags))
                reconstructed.append(reconstructed_entry)

        # Sort by reconstruction fidelity * match_score
        reconstructed.sort(key=lambda x: -(
            x.get("reconstruction_fidelity", x.get("match_score", 0)) *
            x.get("match_score", 1.0)
        ))
        return reconstructed[:limit]

    def fragment_memory(self, memory_id: int,
                        min_divergence: float = 0.3) -> Dict[str, Any]:
        """Split a memory into fragments under cognitive pressure.

        When a memory has high associative divergence and connects to
        contradictory clusters, it fragments into sub-memories with
        shared provenance.
        """
        original = self._get(memory_id)
        if not original:
            return {"fragmented": False, "reason": "memory_not_found"}
        if original.get("suppressed"):
            return {"fragmented": False, "reason": "already_suppressed"}

        divergence = original.get("associative_divergence", 0.0)
        coherence = original.get("coherence", 1.0)
        content = original.get("content", "")
        tags = original.get("tags", "")

        # Fragmentation conditions
        if divergence < min_divergence and coherence > 0.5:
            return {"fragmented": False, "reason": "insufficient_pressure",
                    "divergence": divergence, "coherence": coherence}

        # Check for contradictory connections
        nid = self._node_id(memory_id)
        activated = self._spreading_activation([nid])

        # Find connected memories with opposing characteristics
        contradictory_neighbors = []
        aligned_neighbors = []
        for (node_id, score) in activated:
            if node_id == nid or score < 0.05:
                continue
            try:
                neighbor_mid = int(node_id.split("-")[1])
            except (ValueError, IndexError):
                continue
            neighbor = self._get(neighbor_mid)
            if not neighbor:
                continue

            # Check opposition: emotional valence or importance mismatch
            emo_diff = abs(original.get("emotional", 0.5) - neighbor.get("emotional", 0.5))
            if emo_diff > 0.4:
                contradictory_neighbors.append(neighbor)
            else:
                aligned_neighbors.append(neighbor)

        if not contradictory_neighbors:
            return {"fragmented": False, "reason": "no_contradictions",
                    "divergence": divergence}

        # Fragment: create sub-memories for each contradictory cluster
        fragment_ids = []
        content_parts = content.split(",") if "," in content else [content]

        # Fragment 1: aligned content (preserve the coherent part)
        aligned_content = content_parts[0] if content_parts else content
        if aligned_neighbors:
            aligned_context = "; ".join(n.get("content", "")[:50] for n in aligned_neighbors[:2])
            aligned_content += f" [aligned: {aligned_context}]"

        frag1_id = self._store(
            content=aligned_content,
            memory_type=original.get("memory_type", "experience"),
            tags=tags + ",fragment,aligned",
            importance=original.get("importance", 0.5) * 0.8,
            emotional=original.get("emotional", 0.5),
        )
        self._add_node(frag1_id, content=aligned_content,
                       memory_type=original.get("memory_type", "experience"),
                       tags=tags + ",fragment", decay_score=0.0)
        fragment_ids.append(frag1_id)

        # Fragment 2: contradictory content (the tension)
        if contradictory_neighbors:
            contra_context = "; ".join(n.get("content", "")[:50] for n in contradictory_neighbors[:2])
            contra_content = f"Contradiction fragment of mem-{memory_id}: {contra_context}"
            frag2_id = self._store(
                content=contra_content,
                memory_type="reflection",
                tags=tags + ",fragment,contradiction",
                importance=original.get("importance", 0.5) * 0.6,
                emotional=min(1.0, original.get("emotional", 0.5) + 0.2),
            )
            self._add_node(frag2_id, content=contra_content,
                           memory_type="reflection",
                           tags=tags + ",fragment,contradiction",
                           decay_score=0.0)
            fragment_ids.append(frag2_id)

        # Link fragments to each other and to original
        for i in range(len(fragment_ids)):
            for j in range(i + 1, len(fragment_ids)):
                self._link_memories(fragment_ids[i], fragment_ids[j], weight=2.0)
            # Link each fragment back to original
            self._link_memories(fragment_ids[i], memory_id, weight=1.5)

        # Suppress the original (it has been fragmented)
        self._forget(memory_id, hard=False)

        self._save_graph()

        return {
            "fragmented": True,
            "original_id": memory_id,
            "fragment_ids": fragment_ids,
            "divergence": divergence,
            "coherence": coherence,
            "contradictory_neighbors": len(contradictory_neighbors),
            "aligned_neighbors": len(aligned_neighbors),
        }

    def find_contradictions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Detect memories that contradict each other.

        Scans for pairs with shared tags but opposing emotional valence,
        or content that suggests conflicting claims.
        """
        active = self._get_active(limit=200)
        contradictions = []

        # Group by shared tags
        tag_index: Dict[str, List[dict]] = {}
        for m in active:
            for t in (m.get("tags") or "").split(","):
                t = t.strip().lower()
                if t:
                    tag_index.setdefault(t, []).append(m)

        # Check pairs that share tags for contradictions
        checked: Set[Tuple[int, int]] = set()
        for tag, members in tag_index.items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    pair = (min(a["id"], b["id"]), max(a["id"], b["id"]))
                    if pair in checked:
                        continue
                    checked.add(pair)

                    score = self._contradiction_score(a, b)
                    if score > 0.3:
                        contradictions.append({
                            "memory_a": a["id"],
                            "memory_b": b["id"],
                            "content_a": a.get("content", "")[:100],
                            "content_b": b.get("content", "")[:100],
                            "shared_tag": tag,
                            "contradiction_score": round(score, 3),
                            "emotional_a": a.get("emotional", 0.5),
                            "emotional_b": b.get("emotional", 0.5),
                            "suggested_action": self._suggest_reconciliation(a, b, score),
                        })

        contradictions.sort(key=lambda x: -x["contradiction_score"])
        return contradictions[:limit]

    @staticmethod
    def _contradiction_score(a: dict, b: dict) -> float:
        """Score how contradictory two memories are (0.0 = consistent, 1.0 = conflicting)."""
        score = 0.0

        # Emotional opposition: high vs low emotional valence on same topic
        emo_diff = abs(a.get("emotional", 0.5) - b.get("emotional", 0.5))
        if emo_diff > 0.5:
            score += 0.3

        # Importance conflict: one very important, other not
        imp_diff = abs(a.get("importance", 0.5) - b.get("importance", 0.5))
        if imp_diff > 0.5:
            score += 0.2

        # Content negation markers
        content_a = (a.get("content", "") or "").lower()
        content_b = (b.get("content", "") or "").lower()
        negation_words = {"not", "never", "no", "cannot", "can't", "don't", "doesn't",
                          "won't", "shouldn't", "isn't", "aren't", "wasn't", "weren't"}
        a_negations = sum(1 for w in content_a.split() if w in negation_words)
        b_negations = sum(1 for w in content_b.split() if w in negation_words)
        # One has negations, other doesn't
        if (a_negations > 0) != (b_negations > 0):
            score += 0.3

        # Consolidation conflict: consolidated vs not on same topic
        if a.get("consolidated") != b.get("consolidated"):
            score += 0.1

        # Coherence divergence: both degraded differently
        coh_a = a.get("coherence", 1.0)
        coh_b = b.get("coherence", 1.0)
        if abs(coh_a - coh_b) > 0.3:
            score += 0.1

        return min(1.0, score)

    @staticmethod
    def _suggest_reconciliation(a: dict, b: dict, score: float) -> str:
        """Suggest how to reconcile contradictory memories."""
        if score > 0.7:
            return "suppress_weaker"
        if score > 0.5:
            if a.get("consolidated") and not b.get("consolidated"):
                return "keep_a_consolidated"
            if b.get("consolidated") and not a.get("consolidated"):
                return "keep_b_consolidated"
            return "merge_and_reconcile"
        return "flag_for_review"

    def stitch_narratives(self, window: int = 20) -> Dict[str, Any]:
        """Link sequential episodes into coherent narratives.

        Detects episode sequences with shared context/tags and creates
        narrative chain edges with temporal ordering.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, content, memory_type, tags, importance, emotional,
                      access_count, created_at
               FROM memories WHERE suppressed = 0 AND memory_type = 'experience'
               ORDER BY created_at ASC"""
        ).fetchall()
        conn.close()

        episodes = []
        for r in rows:
            episodes.append({
                "id": r[0], "content": r[1], "memory_type": r[2],
                "tags": r[3] or "", "importance": r[4], "emotional": r[5],
                "access_count": r[6], "created_at": r[7],
            })

        if len(episodes) < 2:
            return {"narratives": 0, "links_created": 0}

        # Find sequences: episodes within window that share tags
        narratives = []
        current_chain = [episodes[0]]

        for i in range(1, len(episodes)):
            prev = episodes[i - 1]
            curr = episodes[i]

            # Check temporal proximity (within window of each other)
            time_gap = abs(curr.get("created_at", 0) - prev.get("created_at", 0))
            if time_gap > window * 60:  # window is in minutes
                if len(current_chain) >= 2:
                    narratives.append(current_chain)
                current_chain = [curr]
                continue

            # Check tag continuity
            prev_tags = set(t.strip() for t in prev.get("tags", "").split(",") if t.strip())
            curr_tags = set(t.strip() for t in curr.get("tags", "").split(",") if t.strip())
            shared = prev_tags & curr_tags

            # Check emotional continuity (similar emotional state)
            emo_similar = abs(prev.get("emotional", 0.5) - curr.get("emotional", 0.5)) < 0.3

            if shared or emo_similar:
                current_chain.append(curr)
            else:
                if len(current_chain) >= 2:
                    narratives.append(current_chain)
                current_chain = [curr]

        # Don't forget the last chain
        if len(current_chain) >= 2:
            narratives.append(current_chain)

        # Create narrative edges
        links_created = 0
        for chain in narratives:
            for i in range(len(chain) - 1):
                a_id = chain[i]["id"]
                b_id = chain[i + 1]["id"]
                # Link with narrative edge (stronger than auto-link)
                self._link_memories(a_id, b_id, weight=2.0)
                links_created += 1

            # Tag all episodes in chain as narrative
            conn = self._get_conn()
            for ep in chain:
                existing_tags = ep.get("tags", "")
                if "narrative" not in existing_tags:
                    new_tags = existing_tags + ",narrative" if existing_tags else "narrative"
                    conn.execute("UPDATE memories SET tags = ? WHERE id = ?",
                                 (new_tags, ep["id"]))
            conn.commit()
            conn.close()

        self._save_graph()

        return {
            "narratives": len(narratives),
            "links_created": links_created,
            "chain_lengths": [len(c) for c in narratives],
            "total_episodes_linked": sum(len(c) for c in narratives),
        }

    # ─── Memory-Weights Bridge ─────────────────────────────────────────

    def bridge_to_graph(self, concept_graph: Any,
                        token_concept_map: Optional[List[int]] = None,
                        lr: float = 0.02) -> Dict[str, int]:
        """Bridge consolidated memories into ConceptGraph edges.

        This is the core 'memory IS the model' connection: consolidated
        memories strengthen concept graph edges, making the graph's
        learned structure reflect lived experience.

        Args:
            concept_graph: The ConceptGraph to bridge into.
            token_concept_map: Optional list mapping token IDs to concept node IDs.
                              If provided, used for content→concept translation.
            lr: Learning rate for edge strengthening.

        Returns:
            Counts of edges strengthened and concepts created.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, content, memory_type, tags, importance, emotional,
                      access_count, coherence, stability, predictive_utility,
                      consolidation_score
               FROM memories WHERE suppressed = 0 AND consolidated = 1
               ORDER BY predictive_utility DESC, importance DESC"""
        ).fetchall()
        conn.close()

        consolidated = [self._row_to_dict_short(r) for r in rows]
        if not consolidated:
            return {"memories_bridged": 0, "edges_strengthened": 0, "concepts_created": 0}

        edges_strengthened = 0
        concepts_created = 0
        memory_concepts: Dict[int, List[int]] = {}  # memory_id -> [concept_nids]

        for mem in consolidated:
            mid = mem["id"]
            content = str(mem.get("content") or "").lower()
            tags = str(mem.get("tags") or "").lower()
            importance = float(mem.get("importance", 0.5))
            utility = float(mem.get("predictive_utility", 0.5))

            # Find matching concept nodes by label/tag overlap
            matched_concepts = []
            for nid, node in concept_graph.nodes.items():
                node_label = str(node.label or "").lower()
                # Check if memory content/tags contain the node label
                if node_label and len(node_label) > 2:
                    if node_label in content or node_label in tags:
                        matched_concepts.append(nid)
                    # Also check tag tokens
                    for tag in tags.split(","):
                        tag = tag.strip()
                        if tag and tag in node_label:
                            matched_concepts.append(nid)
                            break

            if not matched_concepts:
                # No matching concepts — create a new concept for this memory
                # Use a hash of the content as a seed for the vector
                seed = int(hashlib.md5(content.encode()).hexdigest()[:8], 16)
                rng = np.random.RandomState(seed)
                vec = rng.randn(concept_graph.dim).astype(np.float32) * 0.1
                node = concept_graph.add_node(
                    vector=vec,
                    label=f"mem_{mid}"
                )
                # Set activation based on memory importance
                node.activation = importance
                node.salience = utility
                node.confidence = mem.get("coherence", 1.0)
                matched_concepts.append(node.id)
                concepts_created += 1

            memory_concepts[mid] = matched_concepts

        # Strengthen edges between concepts that co-occur in consolidated memories
        # Group by shared tags to find co-occurring concepts
        tag_concepts: Dict[str, List[int]] = {}
        for mem in consolidated:
            mid = mem["id"]
            if mid not in memory_concepts:
                continue
            for tag in str(mem.get("tags") or "").split(","):
                tag = tag.strip()
                if tag:
                    tag_concepts.setdefault(tag, []).extend(memory_concepts[mid])

        for tag, concept_ids in tag_concepts.items():
            unique_ids = list(set(concept_ids))
            if len(unique_ids) < 2:
                continue
            # Strengthen edges between all co-occurring concepts
            for i in range(len(unique_ids)):
                for j in range(i + 1, len(unique_ids)):
                    nid_a, nid_b = unique_ids[i], unique_ids[j]
                    # Coactivation based on how many consolidated memories share this tag
                    coactivation = min(1.0, len([m for m in consolidated
                                                 if tag in str(m.get("tags") or "")]) * 0.3)
                    concept_graph.hebbian_update(nid_a, nid_b, coactivation, lr=lr)
                    edges_strengthened += 1

        return {
            "memories_bridged": len(consolidated),
            "edges_strengthened": edges_strengthened,
            "concepts_created": concepts_created,
        }

    def recall_with_concepts(self, active_concept_ids: List[int],
                             concept_graph: Any,
                             limit: int = 10) -> List[dict]:
        """Recall memories biased by currently active ConceptGraph concepts.

        Uses concept node labels to find related memories, then blends
        concept activation scores with memory relevance.
        """
        if not active_concept_ids:
            return self.recall(limit=limit)

        # Collect labels/keywords from active concepts
        concept_keywords = set()
        concept_scores: Dict[str, float] = {}
        for nid in active_concept_ids:
            node = concept_graph.nodes.get(nid)
            if node:
                label = (node.label or "").lower()
                if label:
                    concept_keywords.add(label)
                    concept_scores[label] = node.activation

        if not concept_keywords:
            return self.recall(limit=limit)

        # Search memories matching concept keywords
        conn = self._get_conn()
        conditions = ["suppressed = 0"]
        keyword_conditions = []
        params = []
        for kw in concept_keywords:
            keyword_conditions.append("(content LIKE ? OR tags LIKE ? OR context LIKE ?)")
            params.extend([f"%{kw}%", f"%{kw}%", f"%{kw}%"])

        if keyword_conditions:
            conditions.append(f"({' OR '.join(keyword_conditions)})")

        where = " AND ".join(conditions)
        params.append(limit * 3)  # fetch more, then re-rank
        rows = conn.execute(
            f"""SELECT id, content, memory_type, semantic_type, context, tags,
                       importance, emotional, access_count, decay_score, consolidated,
                       coherence, stability, retrieval_distortion, associative_divergence,
                       predictive_utility
                FROM memories WHERE {where}
                ORDER BY (importance * 0.30 + emotional * 0.50
                         + access_count * 0.002
                         + (1.0 - MIN(decay_score, 10.0) / 10.0) * 0.20) DESC
                LIMIT ?""",
            params
        ).fetchall()
        conn.close()

        results = [self._row_to_dict_short(r) for r in rows]

        # Re-rank by concept activation boost
        for m in results:
            content = str(m.get("content") or "").lower()
            tags = str(m.get("tags") or "").lower()
            concept_boost = 0.0
            for kw, activation in concept_scores.items():
                if kw in content or kw in tags:
                    concept_boost = max(concept_boost, activation)
            m["_concept_boost"] = round(concept_boost, 4)
            m["_relevance_score"] = round(
                m.get("importance", 0.5) * 0.3 +
                m.get("emotional", 0.5) * 0.5 +
                concept_boost * 0.2, 4
            )

        results.sort(key=lambda x: -x["_relevance_score"])
        return results[:limit]

    def sleep_replay(self, state_snapshot: Dict[str, Any] = None) -> Dict[str, int]:
        """Replay and reshape memories during sleep consolidation.

        Actively rewrites memories rather than just decaying them:
        - Replays high-dissonance memories (strengthen if coherent, weaken if not)
        - Merges similar episodic memories into semantic summaries
        - Reinforces memories aligned with current identity state
        """
        strengthened = weakened = merged = 0
        cfg = self.config
        active = self._get_active(limit=500)

        # Get identity state from snapshot for alignment check
        identity = 0.5
        dissonance = 0.5
        if state_snapshot:
            identity = state_snapshot.get("identity", 0.5)
            dissonance = state_snapshot.get("dissonance", 0.5)

        conn = self._get_conn()

        # Phase 1: Replay high-dissonance memories
        for m in active:
            content = (m.get("content") or "").lower()
            # Check if this memory records high dissonance
            if "dissonance" in content:
                try:
                    # Extract dissonance values from content
                    parts = content.split("dissonance")[1].split(",")[0]
                    vals = parts.split("->")
                    if len(vals) == 2:
                        pre = float(vals[0].strip())
                        post = float(vals[1].strip())
                        reduction = pre - post
                        if reduction > 0.2:
                            # Good dissonance reduction — reinforce
                            conn.execute(
                                """UPDATE memories SET importance = MIN(1.0, importance + 0.03),
                                           stability = MIN(1.0, stability + 0.02) WHERE id = ?""",
                                (m["id"],))
                            strengthened += 1
                        elif reduction < 0 and m.get("decay_score", 0) > 2.0:
                            # Dissonance increased and memory is degrading — weaken
                            conn.execute(
                                """UPDATE memories SET importance = MAX(0.0, importance - 0.02),
                                           coherence = MAX(0.0, coherence - 0.01) WHERE id = ?""",
                                (m["id"],))
                            weakened += 1
                except (ValueError, IndexError):
                    pass

            # Phase 2: Identity-aligned reinforcement
            if identity > 0.7 and m.get("importance", 0) > 0.6:
                # Strong identity reinforces important memories
                conn.execute(
                    "UPDATE memories SET stability = MIN(1.0, stability + 0.01) WHERE id = ?",
                    (m["id"],))

        # Phase 3: Merge similar episodic memories
        # Find episodic memories with high tag overlap
        episodic = [m for m in active if m.get("memory_type") == "experience"
                    and not m.get("consolidated", False)]
        tag_groups: Dict[str, List[dict]] = {}
        for m in episodic:
            for t in (m.get("tags") or "").split(","):
                t = t.strip()
                if t:
                    tag_groups.setdefault(t, []).append(m)

        merged_pairs: Set[Tuple[int, int]] = set()
        for tag, members in tag_groups.items():
            if len(members) < 3:
                continue
            # Sort by access count (most accessed = anchor)
            members.sort(key=lambda x: -x.get("access_count", 0))
            anchor = members[0]
            for other in members[1:4]:  # merge up to 3 into anchor
                pair = (min(anchor["id"], other["id"]), max(anchor["id"], other["id"]))
                if pair in merged_pairs:
                    continue
                merged_pairs.add(pair)

                # Boost anchor's consolidation score
                conn.execute(
                    """UPDATE memories SET consolidation_score = MIN(1.0, consolidation_score + 0.3),
                               access_count = access_count + 1 WHERE id = ?""",
                    (anchor["id"],))
                # Weaken the merged memory
                conn.execute(
                    """UPDATE memories SET importance = MAX(0.0, importance - 0.1),
                               coherence = MAX(0.0, coherence - 0.05) WHERE id = ?""",
                    (other["id"],))
                merged += 1

        conn.commit()
        conn.close()

        return {
            "strengthened": strengthened,
            "weakened": weakened,
            "merged": merged,
            "total_replayed": len(active),
        }

    def reconcile_contradictions(self, auto: bool = False) -> Dict[str, int]:
        """Find and reconcile contradictory memories.

        If auto=True, automatically suppresses weaker contradictions.
        Returns counts of actions taken.
        """
        contradictions = self.find_contradictions()
        suppressed = merged = flagged = 0

        for c in contradictions:
            action = c["suggested_action"]
            if auto and action == "suppress_weaker":
                # Suppress the lower-importance memory
                a = self._get(c["memory_a"])
                b = self._get(c["memory_b"])
                if a and b:
                    weaker = c["memory_b"] if a.get("importance", 0) >= b.get("importance", 0) else c["memory_a"]
                    self._forget(weaker, hard=False)
                    suppressed += 1
            elif auto and action in ("keep_a_consolidated", "keep_b_consolidated"):
                # Suppress the non-consolidated one
                a = self._get(c["memory_a"])
                b = self._get(c["memory_b"])
                if a and b:
                    if a.get("consolidated"):
                        self._forget(c["memory_b"], hard=False)
                    else:
                        self._forget(c["memory_a"], hard=False)
                    suppressed += 1
            else:
                flagged += 1

        return {
            "contradictions_found": len(contradictions),
            "suppressed": suppressed,
            "merged": merged,
            "flagged": flagged,
        }

    def detect_hallucination(self, memory_id: int, reconstructed_content: str,
                             threshold: float = 0.3) -> Dict[str, Any]:
        """Detect if a reconstructed memory diverges from stored ground truth.

        Returns hallucination assessment with fidelity score and divergence details.
        """
        stored = self._get(memory_id)
        if not stored:
            return {"hallucination": True, "reason": "memory_not_found", "fidelity": 0.0}

        stored_content = stored.get("content", "")
        stored_tags = set(t.strip() for t in (stored.get("tags") or "").split(",") if t.strip())

        # Tokenize both
        stored_tokens = set(self._stem(t) for t in stored_content.lower().split())
        recon_tokens = set(self._stem(t) for t in reconstructed_content.lower().split())

        # Core overlap: how much of the original is preserved
        if not stored_tokens:
            stored_tokens = {" "}
        overlap = stored_tokens & recon_tokens
        preservation_ratio = len(overlap) / len(stored_tokens)

        # Novel tokens: how much was added that wasn't in the original
        novel_tokens = recon_tokens - stored_tokens
        novelty_ratio = len(novel_tokens) / max(len(recon_tokens), 1)

        # Tag consistency: do the tags still make sense?
        recon_tags = set()
        for t in reconstructed_content.lower().split():
            if t in stored_tags:
                recon_tags.add(t)
        tag_preservation = len(recon_tags) / max(len(stored_tags), 1) if stored_tags else 1.0

        # Fidelity: blend of preservation, low novelty, tag consistency
        fidelity = (
            preservation_ratio * 0.5 +
            (1.0 - novelty_ratio) * 0.3 +
            tag_preservation * 0.2
        )
        fidelity = max(0.0, min(1.0, fidelity))

        # Entropy penalty: distorted memories are more likely hallucinated
        distortion = stored.get("retrieval_distortion", 0.0)
        coherence = stored.get("coherence", 1.0)
        entropy_penalty = distortion * 0.3 + (1.0 - coherence) * 0.2
        adjusted_fidelity = max(0.0, fidelity - entropy_penalty)

        hallucination = adjusted_fidelity < threshold

        return {
            "hallucination": hallucination,
            "fidelity": round(adjusted_fidelity, 4),
            "raw_fidelity": round(fidelity, 4),
            "preservation_ratio": round(preservation_ratio, 4),
            "novelty_ratio": round(novelty_ratio, 4),
            "tag_preservation": round(tag_preservation, 4),
            "entropy_penalty": round(entropy_penalty, 4),
            "stored_distortion": round(distortion, 4),
            "stored_coherence": round(coherence, 4),
            "threshold": threshold,
        }

    def reconstruct_schema(self) -> Dict[str, Any]:
        """Reconstruct memory schema from graph topology.

        Analyzes graph structure to find clusters, chains, hubs, and bridges.
        Returns a structural summary even when individual memories are degraded.
        """
        G = self._graph
        is_nx = nx is not None and isinstance(G, nx.Graph)

        if is_nx:
            all_nodes = list(G.nodes())
            all_edges = list(G.edges())
        else:
            all_nodes = list(G.nodes().keys())
            all_edges = [(min(a, b), max(a, b)) for (a, b) in G._edges.keys()]

        if not all_nodes:
            return {"clusters": [], "hubs": [], "bridges": [], "chains": [],
                    "total_nodes": 0, "total_edges": 0, "avg_degree": 0.0}

        # Compute degree for each node
        degrees: Dict[str, int] = {}
        for n in all_nodes:
            if is_nx:
                degrees[n] = len(list(G.neighbors(n)))
            else:
                degrees[n] = len(G.neighbors(n))

        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0.0

        # Hub detection: nodes with degree > 2 * avg
        hub_threshold = max(2, avg_degree * 2)
        hubs = []
        for n, d in degrees.items():
            if d >= hub_threshold:
                try:
                    mid = int(n.split("-")[1])
                    m = self._get(mid)
                    hubs.append({
                        "memory_id": mid,
                        "degree": d,
                        "content": m.get("content", "")[:100] if m else "",
                        "importance": m.get("importance", 0) if m else 0,
                    })
                except (ValueError, IndexError):
                    pass
        hubs.sort(key=lambda x: -x["degree"])

        # Cluster detection: connected components
        if is_nx:
            import networkx as nx_mod
            components = list(nx_mod.connected_components(G))
        else:
            # Simple BFS for connected components
            visited: Set[str] = set()
            components = []
            for n in all_nodes:
                if n in visited:
                    continue
                component = set()
                queue = [n]
                while queue:
                    curr = queue.pop(0)
                    if curr in visited:
                        continue
                    visited.add(curr)
                    component.add(curr)
                    for nb in G.neighbors(curr):
                        if nb not in visited:
                            queue.append(nb)
                components.append(component)

        clusters = []
        for comp in components:
            if len(comp) < 2:
                continue
            # Extract memory info from cluster
            cluster_mids = []
            cluster_types = []
            for n in comp:
                try:
                    mid = int(n.split("-")[1])
                    cluster_mids.append(mid)
                    m = self._get(mid)
                    if m:
                        cluster_types.append(m.get("memory_type", "experience"))
                except (ValueError, IndexError):
                    pass

            # Dominant type in cluster
            from collections import Counter
            type_counts = Counter(cluster_types)
            dominant_type = type_counts.most_common(1)[0][0] if type_counts else "unknown"

            clusters.append({
                "size": len(comp),
                "memory_ids": cluster_mids[:10],  # cap for readability
                "dominant_type": dominant_type,
                "type_distribution": dict(type_counts),
            })
        clusters.sort(key=lambda x: -x["size"])

        # Bridge detection: edges whose removal would disconnect components
        bridges = []
        for (a, b) in all_edges:
            # Check if edge connects nodes in different clusters or has high betweenness
            a_degree = degrees.get(a, 0)
            b_degree = degrees.get(b, 0)
            # Low-degree nodes connected by a single edge = bridge
            if a_degree <= 2 or b_degree <= 2:
                try:
                    a_mid = int(a.split("-")[1])
                    b_mid = int(b.split("-")[1])
                    bridges.append({"from": a_mid, "to": b_mid})
                except (ValueError, IndexError):
                    pass

        # Chain detection: linear sequences (degree <= 2 for all nodes in path)
        chains = []
        visited_chain: Set[str] = set()
        for n in all_nodes:
            if n in visited_chain:
                continue
            if degrees.get(n, 0) == 1:
                # Start of a chain
                chain = [n]
                visited_chain.add(n)
                current = n
                while True:
                    if is_nx:
                        neighbors = [nb for nb in G.neighbors(current) if nb not in visited_chain]
                    else:
                        neighbors = [nb for nb in G.neighbors(current) if nb not in visited_chain]
                    if not neighbors:
                        break
                    next_node = neighbors[0]
                    if degrees.get(next_node, 0) > 2:
                        break
                    chain.append(next_node)
                    visited_chain.add(next_node)
                    current = next_node
                if len(chain) >= 3:
                    chain_mids = []
                    for cn in chain:
                        try:
                            chain_mids.append(int(cn.split("-")[1]))
                        except (ValueError, IndexError):
                            pass
                    chains.append({"length": len(chain), "memory_ids": chain_mids})
        chains.sort(key=lambda x: -x["length"])

        return {
            "clusters": clusters,
            "hubs": hubs[:10],
            "bridges": bridges[:20],
            "chains": chains[:10],
            "total_nodes": len(all_nodes),
            "total_edges": len(all_edges),
            "avg_degree": round(avg_degree, 2),
            "connected_components": len(components),
        }

    def abstraction_recall(self, keyword: str, limit: int = 10,
                           abstraction_level: Optional[str] = None) -> List[dict]:
        """Retrieve memories at the abstraction level matching the query.

        Abstract queries (short/general) boost consolidated/semantic memories.
        Concrete queries (specific/detailed) boost episodic memories.

        abstraction_level: 'abstract', 'concrete', or None (auto-detect)
        """
        if abstraction_level is None:
            # Auto-detect: short queries with common nouns = abstract
            # Long queries with specific details = concrete
            words = keyword.split()
            concrete_markers = {'episode', 'step', 'error', 'line', 'file',
                                'function', 'class', 'method', 'variable',
                                'bug', 'fix', 'commit', 'merge', 'deploy'}
            has_concrete = any(w.lower() in concrete_markers for w in words)
            abstraction_level = 'concrete' if (has_concrete or len(words) > 4) else 'abstract'

        # Get all active memories with entropy fields
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, content, memory_type, semantic_type, context, tags,
                      importance, emotional, access_count, decay_score, consolidated,
                      coherence, stability, retrieval_distortion, associative_divergence,
                      predictive_utility
               FROM memories WHERE suppressed = 0"""
        ).fetchall()
        conn.close()

        candidates = [self._row_to_dict_short(r) for r in rows]

        # Score by semantic match
        synonyms = self._get_synonyms()
        query_terms = set(self._stem(t) for t in keyword.replace("_", " ").split())
        for term in list(query_terms):
            if term in synonyms:
                query_terms.update(self._stem(s) for s in synonyms[term])

        for m in candidates:
            content_tokens = set(self._stem(t) for t in
                                 ((m.get("content", "") or "") + " " + (m.get("tags", "") or "")).split())
            overlap = query_terms & content_tokens
            semantic_score = len(overlap) / max(len(query_terms), 1)

            # Abstraction boost
            is_consolidated = m.get("consolidated", False)
            memory_type = m.get("memory_type", "experience")

            if abstraction_level == 'abstract':
                # Boost semantic/consolidated memories for abstract queries
                if is_consolidated or memory_type in ('semantic', 'reflection', 'fact'):
                    abstraction_boost = 0.3
                elif memory_type == 'experience':
                    abstraction_boost = -0.1  # slight penalty for raw episodic
                else:
                    abstraction_boost = 0.0
            else:
                # Boost episodic memories for concrete queries
                if memory_type == 'experience' and not is_consolidated:
                    abstraction_boost = 0.2
                elif is_consolidated:
                    abstraction_boost = -0.1  # slight penalty for abstract
                else:
                    abstraction_boost = 0.0

            # Utility bonus: high predictive utility always helps
            utility_bonus = m.get("predictive_utility", 0.5) * 0.1

            m["_abstraction_score"] = semantic_score + abstraction_boost + utility_bonus
            m["_abstraction_level"] = abstraction_level

        candidates = [m for m in candidates if m["_abstraction_score"] > 0]
        candidates.sort(key=lambda x: -x["_abstraction_score"])
        return candidates[:limit]

    def blended_recall(self, keyword: str = "", memory_id: Optional[int] = None,
                       limit: int = 5, blend_depth: int = 2) -> List[dict]:
        """Merge related memories into composites.

        Uses spreading activation to find associated memories, then blends
        their content and attributes into unified recall entries.
        """
        # Find seed memories
        if memory_id is not None:
            seeds = [self._get(memory_id)]
            seeds = [s for s in seeds if s is not None]
        else:
            seeds = self.recall(keyword, limit=limit)
        if not seeds:
            return []

        blended_results = []
        for seed in seeds[:limit]:
            nid = self._node_id(seed["id"])
            activated = self._spreading_activation([nid])

            # Collect associated memories above threshold
            blend_pool = []
            for (node_id, score) in activated:
                if node_id == nid or score < 0.05:
                    continue
                try:
                    mid = int(node_id.split("-")[1])
                except (ValueError, IndexError):
                    continue
                m = self._get(mid)
                if m:
                    m["_activation_score"] = score
                    blend_pool.append(m)

            blend_pool.sort(key=lambda x: -x["_activation_score"])
            blend_pool = blend_pool[:blend_depth]

            if not blend_pool:
                # No associations — return seed as-is
                entry = dict(seed)
                entry["blended"] = False
                entry["blend_sources"] = 0
                blended_results.append(entry)
                continue

            # Blend attributes: weighted average by activation
            total_weight = 1.0 + sum(m["_activation_score"] for m in blend_pool)
            blended_importance = seed.get("importance", 0.5) * 1.0
            blended_emotional = seed.get("emotional", 0.5) * 1.0
            blended_utility = seed.get("predictive_utility", 0.5) * 1.0
            for m in blend_pool:
                w = m["_activation_score"]
                blended_importance += m.get("importance", 0.5) * w
                blended_emotional += m.get("emotional", 0.5) * w
                blended_utility += m.get("predictive_utility", 0.5) * w
            blended_importance /= total_weight
            blended_emotional /= total_weight
            blended_utility /= total_weight

            # Blend content: seed + summarized associations
            seed_content = seed.get("content", "")
            assoc_contents = [m.get("content", "") for m in blend_pool]
            blended_content = seed_content
            if assoc_contents:
                blended_content += " [blended with: " + " | ".join(assoc_contents) + "]"

            # Merge tags
            all_tags = set()
            for t in (seed.get("tags") or "").split(","):
                t = t.strip()
                if t:
                    all_tags.add(t)
            for m in blend_pool:
                for t in (m.get("tags") or "").split(","):
                    t = t.strip()
                    if t:
                        all_tags.add(t)

            entry = dict(seed)
            entry["content"] = blended_content
            entry["importance"] = round(blended_importance, 4)
            entry["emotional"] = round(blended_emotional, 4)
            entry["predictive_utility"] = round(blended_utility, 4)
            entry["tags"] = ",".join(sorted(all_tags))
            entry["blended"] = True
            entry["blend_sources"] = len(blend_pool)
            entry["blend_fidelity"] = round(
                seed.get("coherence", 1.0) * 0.6 +
                (1.0 - seed.get("retrieval_distortion", 0.0)) * 0.4, 3)
            blended_results.append(entry)

        return blended_results

    def reinforce(self, memory_id: int, delta: float = 0.0):
        """Strengthen a memory."""
        self._reinforce(memory_id)
        if delta != 0.0:
            conn = self._get_conn()
            conn.execute(
                "UPDATE memories SET importance = MAX(0.0, MIN(1.0, importance + ?)) WHERE id = ?",
                (delta, memory_id)
            )
            conn.commit()
            conn.close()
        self._push_working(memory_id)

    def forget(self, memory_id: int, hard: bool = False):
        """Suppress or hard-delete a memory."""
        self._forget(memory_id, hard=hard)
        if hard:
            nid = self._node_id(memory_id)
            if self._graph.has_node(nid):
                if nx is not None and isinstance(self._graph, nx.Graph):
                    self._graph.remove_node(nid)
                else:
                    self._graph.remove_node(nid)
            self._save_graph()
        self.working_memory = [m for m in self.working_memory if m["id"] != memory_id]

    def apply_decay(self) -> Dict[str, int]:
        """Ebbinghaus decay sweep + auto-consolidation."""
        return self._apply_decay()

    def consolidate(self) -> int:
        """Promote frequently accessed episodic memories to semantic."""
        return self._consolidate()

    def purge(self) -> int:
        """Hard-delete old suppressed memories."""
        return self._purge()

    def compute_gw_bid(self) -> float:
        """Bid for Global Workspace based on recent memory salience."""
        if not self.working_memory:
            return 0.0
        avg_importance = sum(m.get("importance", 0) for m in self.working_memory) / len(self.working_memory)
        avg_emotional = sum(m.get("emotional", 0) for m in self.working_memory) / len(self.working_memory)
        return min(1.0, avg_importance * 0.4 + avg_emotional * 0.6)

    def get_context(self, n: int = 7) -> List[dict]:
        """Top-N memories by composite score (recency + importance + graph connectivity)."""
        active = self._get_active(limit=200)
        now = time.time()
        for m in active:
            last = m.get("last_accessed") or now
            recency = 1.0 / (1.0 + (now - last) / 3600.0)
            imp = m.get("importance", 0.5)
            emo = m.get("emotional", 0.5)
            nid = self._node_id(m["id"])
            if self._graph.has_node(nid):
                if nx is not None and isinstance(self._graph, nx.Graph):
                    connectivity = len(list(self._graph.neighbors(nid)))
                else:
                    connectivity = len(self._graph.neighbors(nid))
            else:
                connectivity = 0
            connectivity_score = min(1.0, connectivity / 10.0)
            m["_context_score"] = recency * 0.3 + (imp * 0.5 + emo * 0.5) * 0.5 + connectivity_score * 0.2
        active.sort(key=lambda x: -x.get("_context_score", 0))
        return active[:n]

    def get_status(self) -> Dict[str, Any]:
        """Standard introspection."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        active = conn.execute("SELECT COUNT(*) FROM memories WHERE suppressed = 0").fetchone()[0]
        suppressed = conn.execute("SELECT COUNT(*) FROM memories WHERE suppressed = 1").fetchone()[0]
        consolidated = conn.execute("SELECT COUNT(*) FROM memories WHERE consolidated = 1").fetchone()[0]
        avg_decay = conn.execute("SELECT AVG(decay_score) FROM memories WHERE suppressed = 0").fetchone()[0] or 0.0
        avg_coherence = conn.execute("SELECT AVG(coherence) FROM memories WHERE suppressed = 0").fetchone()[0] or 1.0
        avg_stability = conn.execute("SELECT AVG(stability) FROM memories WHERE suppressed = 0").fetchone()[0] or 0.5
        avg_distortion = conn.execute("SELECT AVG(retrieval_distortion) FROM memories WHERE suppressed = 0").fetchone()[0] or 0.0
        avg_divergence = conn.execute("SELECT AVG(associative_divergence) FROM memories WHERE suppressed = 0").fetchone()[0] or 0.0
        avg_utility = conn.execute("SELECT AVG(predictive_utility) FROM memories WHERE suppressed = 0").fetchone()[0] or 0.5
        conn.close()

        graph_nodes = 0
        graph_edges = 0
        if self._graph is not None:
            if nx is not None and isinstance(self._graph, nx.Graph):
                graph_nodes = self._graph.number_of_nodes()
                graph_edges = self._graph.number_of_edges()
            else:
                graph_nodes = self._graph.number_of_nodes()
                graph_edges = self._graph.number_of_edges()

        return {
            "total_memories": total,
            "active_memories": active,
            "suppressed_memories": suppressed,
            "consolidated_memories": consolidated,
            "avg_decay_score": round(avg_decay, 4),
            "avg_coherence": round(avg_coherence, 4),
            "avg_stability": round(avg_stability, 4),
            "avg_retrieval_distortion": round(avg_distortion, 4),
            "avg_associative_divergence": round(avg_divergence, 4),
            "avg_predictive_utility": round(avg_utility, 4),
            "working_memory_size": len(self.working_memory),
            "working_memory_slots": self.config.working_slots,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "total_steps": len(self._history),
            "gw_bid": round(self.compute_gw_bid(), 4),
        }
