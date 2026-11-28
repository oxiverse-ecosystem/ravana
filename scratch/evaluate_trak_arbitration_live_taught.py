#!/usr/bin/env python3
"""Path-only wrapper: run the byte-identical L1-L9 evaluator on the taught run.

No threshold, gate, check, scoring or verdict logic is redefined here; the
original evaluate_trak_arbitration_live module is imported and only its OUT
constant is redirected to the taught-focus artifacts directory.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import evaluate_trak_arbitration_live as ev  # noqa: E402

ev.OUT = os.path.join(HERE, "m123_ab_out_taught")
raise SystemExit(ev.main())
