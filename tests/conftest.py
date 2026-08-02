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

# ── CI reliability: never let a test process trigger a bulk asset download ──
# Engine boot falls back to auto-downloading glove.6B.zip (~822 MB) whenever the
# projected-vector cache is missing. In CI that turned a cache miss into either a
# multi-minute stall or a hard job timeout, and the Stanford mirror regularly
# answers 503. Default every test process to offline so a cache miss degrades
# instantly to "no GloVe vectors" instead. Opt back in per-run with
# RAVANA_ALLOW_GLOVE_DOWNLOAD=1 (or by pre-setting RAVANA_OFFLINE=0).
os.environ.setdefault("RAVANA_OFFLINE", "1")

# Numeric libraries must not oversubscribe cores when pytest-xdist already runs
# N worker processes; unpinned BLAS threads are what made shard runtimes swing
# wildly (and starve out the 12-minute job budget) on 4-vCPU runners.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")
