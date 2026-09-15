"""
Pre-registered experiment manifest format.

Each experiment is a JSON file committed to pre_registered/ BEFORE running.
The manifest specifies hypothesis, primary metric, success threshold,
negative controls, and analysis plan. Results are written to
pre_registered/results/ (gitignored — only the manifest is tracked).

This prevents HARKing (Hypothesizing After Results are Known) and
post-hoc threshold tuning.

Format:
{
  "name": "experiment_name",
  "hypothesis": "If X, then Y because Z",
  "primary_metric": "metric_name",
  "threshold": {"operator": ">=", "value": 0.5},
  "negative_controls": ["shuffled_inputs", "disabled_module"],
  "min_seeds": 3,
  "analysis_plan": "compare experimental vs controls via paired t-test"
}
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PRE_REGISTERED_DIR = Path(__file__).parent
RESULTS_DIR = PRE_REGISTERED_DIR / "results"


def load_manifest(name: str) -> Optional[Dict]:
    """Load a pre-registered experiment manifest."""
    path = PRE_REGISTERED_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def list_manifests() -> List[str]:
    """List all registered experiment names."""
    if not PRE_REGISTERED_DIR.exists():
        return []
    return [p.stem for p in PRE_REGISTERED_DIR.glob("*.json")]


def save_result(name: str, result: Dict) -> Path:
    """Save experiment results to the (gitignored) results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = RESULTS_DIR / f"{name}_{timestamp}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def check_pre_registered(name: str) -> bool:
    """Check if an experiment is pre-registered."""
    return (PRE_REGISTERED_DIR / f"{name}.json").exists()


def validate_manifest(manifest: Dict) -> List[str]:
    """Validate a pre-registration manifest. Returns list of errors."""
    errors = []
    required = ["name", "hypothesis", "primary_metric", "threshold"]
    for key in required:
        if key not in manifest:
            errors.append(f"Missing required field: {key}")
    if "threshold" in manifest:
        t = manifest["threshold"]
        if "operator" not in t or "value" not in t:
            errors.append("Threshold must have 'operator' and 'value'")
        elif t["operator"] not in (">=", "<=", ">", "<", "==", "!="):
            errors.append(f"Invalid threshold operator: {t['operator']}")
    if "min_seeds" in manifest:
        # Statistical comparisons need N>=2; structural/behavioral tests allow N=1
        is_structural = manifest.get("test_type", "") in ("structural", "behavioral")
        min_required = 1 if is_structural else 2
        if manifest["min_seeds"] < min_required:
            errors.append(
                f"min_seeds must be >= {min_required} "
                f"({'structural' if is_structural else 'statistical'} test)"
            )
    return errors
