"""Dynamic runtime modes for RAVANA GRACE.

Env flags:
  RAVANA_RUNTIME_MODE=NORMAL|DEV|DEV_STRICT
    NORMAL        -> default installed package
    DEV           -> dev testing imports/behavior
    DEV_STRICT    -> dev testing with stricter checks

Optional:
  RAVANA_LOAD_DEFAULT_WEIGHTS=1|0 (default 1)
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Optional


class RuntimeMode(str, Enum):
    NORMAL = "NORMAL"
    DEV = "DEV"
    DEV_STRICT = "DEV_STRICT"


_LOAD_DEFAULT_WEIGHTS = os.environ.get("RAVANA_LOAD_DEFAULT_WEIGHTS", "1") == "1"


def get_runtime_mode() -> RuntimeMode:
    value = os.environ.get("RAVANA_RUNTIME_MODE", RuntimeMode.NORMAL)
    try:
        return RuntimeMode(value.upper())
    except ValueError:
        return RuntimeMode.NORMAL


def configure_runtime(mode: Optional[RuntimeMode] = None) -> RuntimeMode:
    active = mode or get_runtime_mode()
    if active == RuntimeMode.DEV_STRICT:
        os.environ.setdefault("RAVANA_LOAD_DEFAULT_WEIGHTS", "0")
    return active
