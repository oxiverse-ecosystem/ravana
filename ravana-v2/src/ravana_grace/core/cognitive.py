"
RAVANA v2 — cognitive shim for stable imports and dev-mode behavior.

Signals:
  RAVANA_RUNTIME_MODE=NORMAL|DEV|DEV_STRICT
  RAVANA_LOAD_DEFAULT_WEIGHTS=1|0
" "
from __future__ import annotations

import os
from typing import Optional

from .governor import Governor, GovernorConfig, RegulationMode
from .intent import IntentEngine, IntentConfig, SystemObjective
from .strategy import StrategyLayer, StrategyConfig


def get_runtime_mode() -> str:
    return os.environ.get("RAVANA_RUNTIME_MODE", "NORMAL").upper()


def configure_runtime(
    mode: Optional[str] = None,
    *,
    load_default_weights: Optional[bool] = None,
) -> str:
    active = (mode or get_runtime_mode()).upper()

    if load_default_weights is not None:
        os.environ["RAVANA_LOAD_DEFAULT_WEIGHTS"] = "1" if load_default_weights else "0"

    return active
