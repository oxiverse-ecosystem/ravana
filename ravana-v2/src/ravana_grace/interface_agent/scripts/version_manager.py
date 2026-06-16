"""
RAVANA v2 — Version & Context Manager
Maintains SQLite database tracking versions, context, and changelog.
"""

import sqlite3
import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR.parent / "context.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the context database schema."""
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_version TEXT NOT NULL,
            script_versions TEXT NOT NULL,  -- JSON array
            last_updated TEXT NOT NULL,
            changelog TEXT NOT NULL  -- JSON array
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            current_state TEXT NOT NULL,  -- JSON
            active_experiments TEXT NOT NULL,  -- JSON array
            pending_improvements TEXT NOT NULL,  -- JSON array
            last_updated TEXT NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS changelog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            component TEXT NOT NULL,
            change_type TEXT NOT NULL,  -- 'added', 'improved', 'fixed', 'removed'
            description TEXT NOT NULL,
            tested INTEGER DEFAULT 0,
            notes TEXT
        )
    """)

    conn.commit()
    conn.close()


def compute_checksum(file_path: str) -> str:
    """Compute MD5 checksum of a file."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    return ""


def get_script_versions() -> Dict[str, Dict[str, str]]:
    """Get version info for all scripts."""
    scripts = [
        "llm_interpreter.py",
        "memory_learner.py",
        "ravana_agent.py",
        "ravana_wrapper.py",
        "reality_grounding.py",
        "telegram_reporter.py",
        "version_manager.py",
        "interview_mode.py",
    ]
    versions = {}
    for script in scripts:
        path = SCRIPT_DIR / script
        if path.exists():
            versions[script] = {
                "version": "1.0.0",
                "checksum": compute_checksum(str(path)),
                "size": os.path.getsize(str(path)),
            }
    return versions


def record_version(agent_version: str, changelog_entry: Dict = None) -> int:
    """Record a new version snapshot. Returns rowid."""
    init_db()
    conn = get_db()
    c = conn.cursor()

    script_versions = get_script_versions()

    if changelog_entry:
        # Append to existing changelog
        c.execute("SELECT changelog FROM versions ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        existing = json.loads(row["changelog"]) if row else []
        existing.append(changelog_entry)
        changelog_json = json.dumps(existing)
    else:
        changelog_json = "[]"

    c.execute(
        "INSERT INTO versions (agent_version, script_versions, last_updated, changelog) VALUES (?, ?, ?, ?)",
        (agent_version, json.dumps(script_versions), datetime.now().isoformat(), changelog_json),
    )
    conn.commit()
    rowid = c.lastrowid
    conn.close()
    return rowid


def get_latest_version() -> Optional[Dict]:
    """Get the most recent version record."""
    init_db()
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM versions ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_context() -> Optional[Dict]:
    """Get current context state."""
    init_db()
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM context ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "current_state": json.loads(row["current_state"]),
            "active_experiments": json.loads(row["active_experiments"]),
            "pending_improvements": json.loads(row["pending_improvements"]),
            "last_updated": row["last_updated"],
        }
    return None


def save_context(current_state: Dict, active_experiments: List, pending_improvements: List) -> None:
    """Save context state."""
    init_db()
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO context (current_state, active_experiments, pending_improvements, last_updated) VALUES (?, ?, ?, ?)",
        (json.dumps(current_state), json.dumps(active_experiments), json.dumps(pending_improvements), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def add_changelog(component: str, change_type: str, description: str, tested: bool = False, notes: str = "") -> int:
    """Add a changelog entry. Returns rowid."""
    init_db()
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO changelog (timestamp, component, change_type, description, tested, notes) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), component, change_type, description, 1 if tested else 0, notes),
    )
    conn.commit()
    rowid = c.lastrowid
    conn.close()
    return rowid


def get_changelog(limit: int = 20) -> List[Dict]:
    """Get recent changelog entries."""
    init_db()
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM changelog ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_pending_improvements() -> List[str]:
    """Get list of pending improvements."""
    ctx = get_context()
    if ctx:
        return ctx.get("pending_improvements", [])
    return []


def mark_tested(changelog_id: int) -> None:
    """Mark a changelog entry as tested."""
    init_db()
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE changelog SET tested = 1 WHERE id = ?", (changelog_id,))
    conn.commit()
    conn.close()


def detect_changes() -> List[Dict]:
    """Detect which scripts have changed since last version record."""
    latest = get_latest_version()
    if not latest:
        return []

    previous_raw = json.loads(latest["script_versions"])
    current = get_script_versions()

    # Normalize previous to dict format for uniform handling
    if isinstance(previous_raw, dict):
        previous = previous_raw
    else:
        # Legacy list format - convert to dict keyed by name
        previous = {}
        for p in previous_raw:
            name = p if isinstance(p, str) else p.get("name", "")
            previous[name] = p if isinstance(p, dict) else {}

    changes = []
    for script, info in current.items():
        prev_info = previous.get(script, {})
        prev_checksum = prev_info.get("checksum", "") if isinstance(prev_info, dict) else ""
        if info.get("checksum") != prev_checksum:
            changes.append({
                "script": script,
                "old_checksum": prev_checksum,
                "new_checksum": info.get("checksum", ""),
                "version": info.get("version", "unknown"),
            })

    return changes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAVANA Version Manager")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("init", help="Initialize the database")
    sub.add_parser("status", help="Show current version status")
    sub.add_parser("changes", help="Detect script changes since last version")
    sub.add_parser("changelog", help="Show recent changelog")

    args = parser.parse_args()

    if args.cmd == "init":
        init_db()
        print("Database initialized.")

    elif args.cmd == "status":
        latest = get_latest_version()
        if latest:
            print(f"Agent version: {latest['agent_version']}")
            print(f"Last updated: {latest['last_updated']}")
            scripts_raw = json.loads(latest["script_versions"])
            # Handle both dict (v1.5.0+) and list (older) formats
            if isinstance(scripts_raw, dict):
                script_list = [{"name": k, "version": v.get("version", "1.0.0"), "checksum": v.get("checksum", "")} for k, v in scripts_raw.items()]
            else:
                script_list = scripts_raw
            print(f"Scripts tracked: {len(script_list)}")
            for s in script_list:
                name = s["name"] if isinstance(s, dict) else s
                version = s.get("version", "1.0.0") if isinstance(s, dict) else "1.0.0"
                checksum = s.get("checksum", "") if isinstance(s, dict) else ""
                print(f"  {name}: {version} ({checksum[:8] if checksum else 'no-checksum'})")
        else:
            print("No version recorded yet.")

    elif args.cmd == "changes":
        changes = detect_changes()
        if changes:
            print("Changes detected:")
            for c in changes:
                print(f"  {c['script']}: {c['old_checksum'][:8]} → {c['new_checksum'][:8]}")
        else:
            print("No changes detected.")

    elif args.cmd == "changelog":
        entries = get_changelog(limit=10)
        for e in entries:
            tested = "✅" if e["tested"] else "❌"
            print(f"{tested} [{e['timestamp'][:10]}] {e['component']} ({e['change_type']}): {e['description']}")

    else:
        parser.print_help()
