"""Tests for ravana_grace.agent.version_manager."""

import pytest
import tempfile
import os
import contextlib
from ravana_grace.agent.version_manager import VersionManager, ScriptVersion


class TestVersionManager:
    def test_init_with_temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vm = VersionManager(db_path=db_path)
            assert vm.db_path == db_path
            # Verify tables were created by querying
            import sqlite3
            conn = sqlite3.connect(db_path)
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t[0] for t in tables]
            assert "versions" in table_names
            assert "context" in table_names
            assert "changelog" in table_names
            assert "experiments" in table_names
            assert "improvements" in table_names
            conn.close()
        finally:
            os.unlink(db_path)

    def test_get_current_versions_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vm = VersionManager(db_path=db_path)
            versions = vm.get_current_versions()
            assert versions["agent_version"] == "0.0.0"
            assert versions["script_versions"] == []
        finally:
            os.unlink(db_path)

    def test_save_and_retrieve_version(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vm = VersionManager(db_path=db_path)
            scripts = [{"name": "test.py", "version": "1.0.0", "checksum": "abc123"}]
            vm.save_version("1.0.0", scripts, [])
            versions = vm.get_current_versions()
            assert versions["agent_version"] == "1.0.0"
            assert len(versions["script_versions"]) == 1
        finally:
            os.unlink(db_path)

    def test_add_changelog(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vm = VersionManager(db_path=db_path)
            vm.add_changelog("test_component", "added", "Test change")
            entries = vm.get_recent_changelog(limit=10)
            assert len(entries) == 1
            assert entries[0]["component"] == "test_component"
        finally:
            os.unlink(db_path)

    def test_mark_tested(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vm = VersionManager(db_path=db_path)
            vm.add_changelog("comp", "improved", "desc", tested=False)
            entries = vm.get_recent_changelog(limit=10)
            changelog_id = entries[0]["id"]
            vm.mark_tested(changelog_id)
            entries = vm.get_recent_changelog(limit=10)
            assert entries[0]["tested"] == 1
        finally:
            os.unlink(db_path)

    def test_set_and_get_context(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vm = VersionManager(db_path=db_path)
            vm.set_context("test_key", {"value": 42})
            value = vm.get_context("test_key")
            assert value["value"] == 42
        finally:
            os.unlink(db_path)

    def test_create_experiment(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vm = VersionManager(db_path=db_path)
            exp_id = vm.create_experiment("test_exp", "Test experiment")
            assert exp_id > 0
        finally:
            os.unlink(db_path)

    def test_queue_improvement(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vm = VersionManager(db_path=db_path)
            vm.queue_improvement("Test improvement", source="test", priority=8)
            pending = vm.get_pending_improvements()
            assert len(pending) == 1
            assert pending[0]["priority"] == 8
        finally:
            os.unlink(db_path)

    def test_record_test(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vm = VersionManager(db_path=db_path)
            vm.record_test("test_name", "pass", "output OK", 100)
            history = vm.get_test_history()
            assert len(history) == 1
            assert history[0]["status"] == "pass"
        finally:
            os.unlink(db_path)

    def test_get_last_test_status(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vm = VersionManager(db_path=db_path)
            vm.record_test("test_name", "fail")
            status = vm.get_last_test_status("test_name")
            assert status == "fail"
        finally:
            os.unlink(db_path)

    def test_get_summary(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vm = VersionManager(db_path=db_path)
            summary = vm.get_summary()
            assert "agent_version" in summary
            assert "recent_changes" in summary
            assert "pending_improvements" in summary
        finally:
            # Force GC to release SQLite file locks on Windows
            import gc
            gc.collect()
            if os.path.exists(db_path):
                try:
                    os.unlink(db_path)
                except PermissionError:
                    pass

    def test_detect_changed_scripts(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            vm = VersionManager(db_path=db_path)
            with tempfile.TemporaryDirectory() as tmpdir:
                py_file = os.path.join(tmpdir, "test_script.py")
                with open(py_file, "w") as f2:
                    f2.write("print('hello')")
                changed = vm.detect_changed_scripts(tmpdir)
                assert len(changed) == 1
                assert changed[0]["name"] == "test_script.py"
                assert changed[0]["changed"] is True
        finally:
            os.unlink(db_path)
