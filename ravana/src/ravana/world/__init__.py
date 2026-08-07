"""Ravana world simulations (optional — requires ravana-ml)."""
try:
    from ravana_ml.world import *
except ImportError:
    from ravana_ml._import_guard_shim import report_missing
    report_missing("ravana_ml.world", "world-simulation modules", kind="internal")
