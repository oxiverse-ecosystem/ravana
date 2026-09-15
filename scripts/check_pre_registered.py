#!/usr/bin/env python3
"""
Pre-registration enforcement script.

Verifies that any experiment file in experiments/ has a corresponding
pre-registration manifest in pre_registered/.

Usage:
    python scripts/check_pre_registered.py          # check all
    python scripts/check_pre_registered.py <name>   # check specific
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
PRE_REGISTERED = REPO / "pre_registered"
RESULTS_DIR = PRE_REGISTERED / "results"

from pre_registered import validate_manifest

def check_experiment(name: str) -> bool:
    """Check if an experiment is pre-registered. Returns True if OK."""
    manifest_path = PRE_REGISTERED / f"{name}.json"
    
    if not manifest_path.exists():
        print(f"FAIL: {name} is not pre-registered ({manifest_path} missing)")
        return False
    
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        print(f"FAIL: {name} manifest invalid: {e}")
        return False
    
    # Validate manifest structure
    errors = validate_manifest(manifest)
    if errors:
        print(f"FAIL: {name} manifest errors: {'; '.join(errors)}")
        return False
    
    # Check results exist (if experiment was run)
    results = list(RESULTS_DIR.glob(f"{name}_*.json")) if RESULTS_DIR.exists() else []
    
    print(f"PASS: {name} is pre-registered "
          f"({len(results)} result file{'s' if len(results) != 1 else ''})")
    return True


def check_all() -> bool:
    """Check all experiment files against pre-registered manifests."""
    experiments_dir = REPO / "experiments"
    if not experiments_dir.exists():
        print("No experiments/ directory found")
        return True
    
    experiment_files = [p.stem for p in experiments_dir.glob("experiment_*.py")]
    all_ok = True
    for name in sorted(experiment_files):
        if not check_experiment(name):
            all_ok = False
    
    # Also list pre-registered experiments without corresponding .py files
    if PRE_REGISTERED.exists():
        registered = {p.stem for p in PRE_REGISTERED.glob("*.json")}
        orphans = registered - set(experiment_files)
        if orphans:
            print(f"\nPre-registered but no experiment_*.py file: {sorted(orphans)}")
    
    return all_ok


def main():
    if len(sys.argv) > 1:
        name = sys.argv[1]
        ok = check_experiment(name)
    else:
        ok = check_all()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
