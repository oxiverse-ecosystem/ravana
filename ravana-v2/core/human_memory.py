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
                procedural_id INTEGER REFERENCES memories(id)
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
        conn.commit()
        conn.close()

    def _store(self, content: str, memory_type: str = "experience",
               semantic_type: str = None, context: str = None,
               tags: str = None, importance: float = 0.5,
               emotional: float = 0.5) -> int:
        conn = self._get_conn()
        now = time.time()
        cursor = conn.execute(
            """INSERT INTO memories (content, memory_type, semantic_type, context, tags,
                                     importance, emotional, access_count, last_accessed,
                                     created_at, decay_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, 0.0)""",
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
            conn.execute(
                f"UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id IN ({placeholders})",
                [time.time()] + recalled_ids
            )
            conn.commit()
        conn.close()
        return [self._row_to_dict_short(r) for r in rows]

    def _get(self, memory_id: int) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        conn.close()
        return self._row_to_dict_full(row) if row else None

    def _get_active(self, limit: int = 100) -> List[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, content, memory_type, semantic_type, context, tags,
                      importance, emotional, access_count, decay_score, consolidated
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

    def _apply_decay(self) -> Dict[str, int]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, decay_score, importance, emotional FROM memories WHERE suppressed = 0"
        ).fetchall()
        decayed = suppressed = 0
        base = self.config.base_decay_rate
        threshold = self.config.retention_threshold
        for (mid, decay, imp, emo) in rows:
            retention_mod = (imp * 0.5 + emo * 0.5) * 0.9 + 0.1
            delta = base * retention_mod
            new_decay = decay + delta
            if new_decay >= threshold:
                conn.execute("UPDATE memories SET suppressed = 1, suppressed_at = ? WHERE id = ?",
                             (time.time(), mid))
                suppressed += 1
            else:
                conn.execute("UPDATE memories SET decay_score = ? WHERE id = ?", (new_decay, mid))
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
        return {
            "id": row[0], "content": row[1], "memory_type": row[2],
            "semantic_type": row[3], "context": row[4], "tags": row[5],
            "importance": row[6], "emotional": row[7], "access_count": row[8],
            "decay_score": row[9], "consolidated": row[10] == 1,
        }

    @staticmethod
    def _row_to_dict_full(row: tuple) -> dict:
        return {
            "id": row[0], "content": row[1], "memory_type": row[2],
            "semantic_type": row[3], "context": row[4], "tags": row[5],
            "importance": row[6], "emotional": row[7], "access_count": row[8],
            "last_accessed": row[9], "created_at": row[10],
            "decay_score": row[11], "suppressed": row[12],
            "suppressed_at": row[13], "consolidated": row[14] == 1,
            "consolidation_score": row[15],
        }

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
        tags = ",".join(tags_parts)

        memory_type = "experience"
        if episode_data.get("resolution", {}).get("full_resolution"):
            memory_type = "reflection" if meaning > 0.3 else "semantic"

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
                 tags: str = "", context: str = "") -> int:
        """Explicit memory storage. Returns memory id."""
        mid = self._store(content=content, memory_type=memory_type,
                          tags=tags, context=context,
                          importance=importance, emotional=emotional)
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
            "working_memory_size": len(self.working_memory),
            "working_memory_slots": self.config.working_slots,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "total_steps": len(self._history),
            "gw_bid": round(self.compute_gw_bid(), 4),
        }
