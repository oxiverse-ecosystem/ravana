import os
import sys

# Configure python paths globally for all tests in tests/
#
# Order matters. Inserting in a loop with insert(0, ...) REVERSES the list, so
# a naive loop over [ml_src, ravana_src, v2_src, proj_root] leaves proj_root at
# sys.path[0] — ahead of every real package source dir. That let the project
# root shadow the installed packages: `ravana_ml` resolved to the bare
# ./ravana_ml directory (no __init__.py, so an implicit *namespace* package)
# instead of ./ravana_ml/src/ravana_ml, and `import ravana_ml.nn.rlm_v2` then
# died with "No module named 'ravana_ml.nn.module'".
#
# Build the final order explicitly and prepend it as a block, so the real
# package roots always precede the project root.
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_path_prefix = [
    os.path.join(_proj_root, "ravana_ml", "src"),
    os.path.join(_proj_root, "ravana", "src"),
    os.path.join(_proj_root, "ravana-v2", "src"),
    _proj_root,
]
for p in reversed(_path_prefix):
    if p in sys.path:
        sys.path.remove(p)
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
