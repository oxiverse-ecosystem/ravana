import os
import sys

# Configure python paths globally for all tests in tests/
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [
    os.path.join(_proj_root, "ravana_ml", "src"),
    os.path.join(_proj_root, "ravana", "src"),
    os.path.join(_proj_root, "ravana-v2", "src"),
    _proj_root,
]:
    if p not in sys.path:
        sys.path.insert(0, p)
