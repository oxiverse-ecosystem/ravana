import os

# Prevent pytest from collecting non-test files inside experiments/.
# Covers: experiment_*, eval_*, profile_*, diag_*, debug_*, run_*, topology_*
_EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")
_EXPERIMENT_PREFIXES = ("experiment_", "eval_", "profile_", "diag_", "debug_", "run_", "topology_")

collect_ignore = [
    os.path.join("experiments", f)
    for f in os.listdir(_EXPERIMENTS_DIR)
    if f.endswith(".py") and any(f.startswith(p) for p in _EXPERIMENT_PREFIXES)
]
