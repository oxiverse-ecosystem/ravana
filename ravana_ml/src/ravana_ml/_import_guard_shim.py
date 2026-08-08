"""Shim so ravana_ml can use the same non-silent import-guard logging as ravana.

ravana_ml is its own installable package and may be imported before `ravana`,
so we re-export `report_missing` from `ravana._import_guard` when available and
fall back to a local minimal logger otherwise (so a missing import is still
never silently swallowed).
"""
try:
    from ravana._import_guard import report_missing  # type: ignore
except Exception:  # pragma: no cover - only when ravana is not installed
    import logging

    _log = logging.getLogger("ravana_ml.imports")

    def report_missing(module: str, capability: str, kind: str = "optional") -> None:
        level = logging.ERROR if kind == "internal" else logging.WARNING
        _log.log(level, "MISSING %s (%s) [%s]", module, capability, kind)

__all__ = ["report_missing"]
