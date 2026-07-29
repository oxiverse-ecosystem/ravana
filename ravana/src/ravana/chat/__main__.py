"""RAVANA Modular Chat - Main entry point."""
import os
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from .interface import main

if __name__ == "__main__":
    main()