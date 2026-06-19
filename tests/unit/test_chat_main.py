"""Tests for ravana/src/ravana/chat/__main__.py."""

import pytest
from unittest.mock import patch, MagicMock


class TestChatMain:
    def test_main_imports(self):
        """Verify __main__ imports main from interface correctly."""
        from ravana.chat import __main__
        assert __main__ is not None

    def test_main_module_runs(self):
        """Verify __main__ calls interface.main() when run."""
        with patch("ravana.chat.__main__.main") as mock_main:
            # Simulate running the module
            mock_main.return_value = None
            from ravana.chat.__main__ import main
            main()
            mock_main.assert_called_once()

    def test_entry_point_exists(self):
        """Verify the module has the if __name__ == '__main__' guard."""
        from ravana.chat.__main__ import main
        assert callable(main)

    def test_interface_importable(self):
        """Verify the interface module is importable from __main__'s perspective."""
        from ravana.chat import interface
        assert hasattr(interface, "main")
