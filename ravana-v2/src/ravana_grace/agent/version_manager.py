"""
RAVANA Agent — Version & Context Manager
Maintains version history, experiment state, and improvement tracking.
Uses SQLite for persistent context across agent runs.
"""

import sqlite3
import json
import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict


DB_PATH = os.environ.get("CONTEXT_DB", "/home/workspace/Skills/ravana-interface/context.db")


@dataclass
class ScriptVersion:
    name: str
    version: str
    checksum: str
    last_updated: str
    status: str  # 'active', 'modified', 'testing'


@dataclass
class VersionEntry:
    agent_version: str
    script_versions: List[Dict]
    last_updated: str
    changelog: List[Dict]


@dataclass
class ChangeEntry:
    timestamp: str
    component: str
    change_type: str  # 'added', 'improved', 'fixed', 'researched'
    description: str
    tested: bool
    notes: str


class VersionManager:
    """
    Manages version history and context for the RAVANA Interface Agent.
    
    Tracks:
    - Script versions + checksums for change detection
    - Improvement changelog
    - Active experiments
    - Pending improvements queue
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._ensure_db()

    def _ensure_db(self):
        """Create DB and tables if they don't exist."""
        os.makedirs(Path(self.db_path).parent, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_version TEXT NOT NULL,
                script_versions TEXT,  -- JSON
                last_updated TEXT,
                changelog TEXT  -- JSON array
            );
            
            CREATE TABLE IF NOT EXISTS context (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT,  -- JSON
                updated_at TEXT
            );
            
            CREATE TABLE IF NOT EXISTS changelog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                component TEXT NOT NULL,
                change_type TEXT NOT NULL,
                description TEXT,
                tested INTEGER DEFAULT 0,
                notes TEXT
            );
            
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',  -- pending, running, completed, abandoned
                created_at TEXT,
                updated_at TEXT,
                results TEXT  -- JSON
            );
            
            CREATE TABLE IF NOT EXISTS improvements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                source TEXT,  -- web_search, brainstorm, user, self
                priority INTEGER DEFAULT 5,  -- 1-10
                status TEXT DEFAULT 'pending',  -- pending, approved, rejected, implemented
                created_at TEXT,
                implemented_at TEXT,
                notes TEXT
            );
            
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                status TEXT,  -- pass, fail, error
                output TEXT,
                duration_ms INTEGER,
                ran_at TEXT
            );
        """)
        conn.commit()
        conn.close()

    # ─── Versions ───────────────────────────────────────────────

    def get_current_versions(self) -> Dict[str, Any]:
        """Get latest version entry."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM versions ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row:
            return {
                "agent_version": row["agent_version"],
                "script_versions": json.loads(row["script_versions"] or "[]"),
                "last_updated": row["last_updated"],
                "changelog": json.loads(row["changelog"] or "[]"),
            }
        return {"agent_version": "0.0.0", "script_versions": [], "last_updated": "", "changelog": []}

    def detect_changed_scripts(self, scripts_dir: str) -> List[Dict]:
        """Scan scripts directory and find files that changed since last record."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        current = self.get_current_versions()
        last_versions = {v['name']: v for v in current.get('script_versions', [])}
        
        detected = []
        scripts_path = Path(scripts_dir)
        
        for py_file in scripts_path.glob("*.py"):
            content = py_file.read_text()
            checksum = hashlib.md5(content.encode()).hexdigest()
            name = py_file.name
            
            if name not in last_versions or last_versions[name]['checksum'] != checksum:
                last_ver = last_versions.get(name, {})
                detected.append({
                    "name": name,
                    "version": self._bump_version(last_ver.get('version', '1.0.0')),
                    "checksum": checksum,
                    "last_updated": datetime.now().isoformat(),
                    "status": "modified" if name in last_versions else "new",
                    "changed": True
                })
            else:
                detected.append({**last_versions[name], "changed": False})
        
        conn.close()
        return detected

    def save_version(self, agent_version: str, script_versions: List[Dict], changelog: List[Dict]):
        """Save new version entry."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO versions (agent_version, script_versions, last_updated, changelog)
            VALUES (?, ?, ?, ?)
        """, (agent_version, json.dumps(script_versions), datetime.now().isoformat(), json.dumps(changelog)))
        conn.commit()
        conn.close()

    def _bump_version(self, version: str, bump_type: str = "patch") -> str:
        """Bump semantic version. bump_type: major, minor, patch."""
        try:
            major, minor, patch = version.split('.')
            major, minor, patch = int(major), int(minor), int(patch)
            if bump_type == "major":
                return f"{major+1}.0.0"
            elif bump_type == "minor":
                return f"{major}.{minor+1}.0"
            else:
                return f"{major}.{minor}.{patch+1}"
        except:
            return "1.0.0"

    # ─── Changelog ──────────────────────────────────────────────

    def add_changelog(self, component: str, change_type: str, description: str, notes: str = "", tested: bool = False):
        """Add entry to changelog."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO changelog (timestamp, component, change_type, description, tested, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), component, change_type, description, int(tested), notes))
        conn.commit()
        conn.close()

    def get_recent_changelog(self, limit: int = 20) -> List[Dict]:
        """Get recent changelog entries."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM changelog ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def mark_tested(self, changelog_id: int):
        """Mark a changelog entry as tested."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE changelog SET tested=1 WHERE id=?", (changelog_id,))
        conn.commit()
        conn.close()

    # ─── Context ────────────────────────────────────────────────

    def set_context(self, key: str, value: Any):
        """Set a context key-value pair."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO context (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, json.dumps(value), datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_context(self, key: str) -> Any:
        """Get context value by key."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT value FROM context WHERE key=?", (key,))
        row = cur.fetchone()
        conn.close()
        return json.loads(row['value']) if row else None

    # ─── Experiments ─────────────────────────────────────────────

    def create_experiment(self, name: str, description: str = "") -> int:
        """Create new experiment, return its ID."""
        conn = sqlite3.connect(self.db_path)
        now = datetime.now().isoformat()
        cur = conn.execute("""
            INSERT INTO experiments (name, description, status, created_at, updated_at)
            VALUES (?, ?, 'pending', ?, ?)
        """, (name, description, now, now))
        exp_id = cur.lastrowid
        conn.commit()
        conn.close()
        return exp_id

    def update_experiment(self, name: str, status: str = None, results: Dict = None):
        """Update experiment status and/or results."""
        conn = sqlite3.connect(self.db_path)
        now = datetime.now().isoformat()
        if status:
            conn.execute("UPDATE experiments SET status=?, updated_at=? WHERE name=?", (status, now, name))
        if results:
            conn.execute("UPDATE experiments SET results=? WHERE name=?", (json.dumps(results), name))
        conn.commit()
        conn.close()

    def get_active_experiments(self) -> List[Dict]:
        """Get experiments that are still running."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM experiments WHERE status='running' OR status='pending'")
        return [dict(r) for r in cur.fetchall()]

    # ─── Improvements ───────────────────────────────────────────

    def queue_improvement(self, description: str, source: str = "web_search", priority: int = 5):
        """Add improvement to pending queue."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO improvements (description, source, priority, status, created_at)
            VALUES (?, ?, ?, 'pending', ?)
        """, (description, source, priority, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_pending_improvements(self, limit: int = 10) -> List[Dict]:
        """Get pending improvements sorted by priority."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("""
            SELECT * FROM improvements 
            WHERE status='pending' 
            ORDER BY priority DESC, id DESC 
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def mark_improvement(self, improvement_id: int, status: str):
        """Mark improvement as approved/rejected/implemented."""
        conn = sqlite3.connect(self.db_path)
        now = datetime.now().isoformat() if status == 'implemented' else datetime.now().isoformat()
        conn.execute("""
            UPDATE improvements 
            SET status=?, implemented_at=CASE WHEN ?='implemented' THEN ? ELSE implemented_at END
            WHERE id=?
        """, (status, status, now, improvement_id))
        conn.commit()
        conn.close()

    # ─── Test Results ────────────────────────────────────────────

    def record_test(self, test_name: str, status: str, output: str = "", duration_ms: int = 0):
        """Record a test result."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO test_results (test_name, status, output, duration_ms, ran_at)
            VALUES (?, ?, ?, ?, ?)
        """, (test_name, status, output, duration_ms, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def get_test_history(self, limit: int = 20) -> List[Dict]:
        """Get recent test results."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM test_results ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_last_test_status(self, test_name: str) -> Optional[str]:
        """Get last status for a specific test."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute("""
            SELECT status FROM test_results 
            WHERE test_name=? 
            ORDER BY id DESC LIMIT 1
        """, (test_name,))
        row = cur.fetchone()
        conn.close()
        return row['status'] if row else None

    # ─── Report ──────────────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        """Get full system summary for reporting."""
        current = self.get_current_versions()
        recent_cl = self.get_recent_changelog(5)
        pending_imp = self.get_pending_improvements(5)
        active_exp = self.get_active_experiments()
        test_hist = self.get_test_history(5)
        
        return {
            "agent_version": current['agent_version'],
            "script_count": len(current.get('script_versions', [])),
            "recent_changes": recent_cl,
            "pending_improvements": len(pending_imp),
            "active_experiments": len(active_exp),
            "recent_tests": test_hist,
            "last_updated": current['last_updated'],
        }


if __name__ == "__main__":
    vm = VersionManager()
    print("=== Version Manager Test ===")
    
    # Detect changes
    changed = vm.detect_changed_scripts("/home/workspace/Skills/ravana-interface/scripts")
    print(f"\nScripts scanned: {len(changed)}")
    for s in changed:
        marker = " ← CHANGED" if s.get('changed') else ""
        print(f"  {s['name']}: v{s['version']}{marker}")
    
    # Pending improvements
    pending = vm.get_pending_improvements()
    print(f"\nPending improvements: {len(pending)}")
    for p in pending:
        print(f"  [{p['priority']}] {p['description'][:60]}...")
    
    # Test history
    tests = vm.get_test_history()
    print(f"\nRecent tests: {len(tests)}")
    for t in tests:
        icon = "✅" if t['status'] == 'pass' else "❌" if t['status'] == 'fail' else "⚠️"
        print(f"  {icon} {t['test_name']}: {t['status']}")
    
    # Summary
    summary = vm.get_summary()
    print(f"\n=== System Summary ===")
    print(f"Agent version: {summary['agent_version']}")
    print(f"Scripts: {summary['script_count']}")
    print(f"Pending improvements: {summary['pending_improvements']}")
    print(f"Active experiments: {summary['active_experiments']}")
