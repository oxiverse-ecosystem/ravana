"""Test pre-registration framework."""
import json
import sys
import os
os.environ["RAVANA_OFFLINE"] = "1"
sys.path.insert(0, r"C:\Users\Likhith\Documents\Projects\ravana")

from pre_registered import (
    load_manifest, list_manifests, validate_manifest,
    check_pre_registered
)


def test_load_manifest():
    manifest = load_manifest("sleep_episodic_replay")
    assert manifest is not None, "Manifest not found"
    assert manifest["name"] == "sleep_episodic_replay"
    print("PASS: test_load_manifest")


def test_validate_manifest_valid():
    manifest = {
        "name": "test_exp",
        "hypothesis": "If X then Y",
        "primary_metric": "metric",
        "threshold": {"operator": ">=", "value": 0.5},
        "min_seeds": 3,
    }
    errors = validate_manifest(manifest)
    assert len(errors) == 0, f"Expected no errors, got: {errors}"
    print("PASS: test_validate_manifest_valid")


def test_validate_manifest_missing_fields():
    manifest = {"name": "incomplete"}
    errors = validate_manifest(manifest)
    assert len(errors) >= 3, f"Expected >=3 errors, got {len(errors)}: {errors}"
    print("PASS: test_validate_manifest_missing_fields")


def test_validate_manifest_invalid_threshold():
    manifest = {
        "name": "bad_threshold",
        "hypothesis": "test",
        "primary_metric": "m",
        "threshold": {"operator": "~~", "value": 0},
    }
    errors = validate_manifest(manifest)
    assert any("Invalid threshold" in e for e in errors), f"Expected threshold error, got: {errors}"
    print("PASS: test_validate_manifest_invalid_threshold")


def test_validate_structural_vs_statistical():
    # Structural test allows N=1
    structural = {
        "name": "structural_test",
        "hypothesis": "test",
        "primary_metric": "m",
        "threshold": {"operator": "==", "value": True},
        "test_type": "behavioral",
        "min_seeds": 1,
    }
    errors = validate_manifest(structural)
    assert len(errors) == 0, f"Structural with N=1 should pass: {errors}"

    # Statistical test requires N>=2
    statistical = {
        "name": "statistical_test",
        "hypothesis": "test",
        "primary_metric": "m",
        "threshold": {"operator": ">=", "value": 0.5},
        "min_seeds": 1,
    }
    errors = validate_manifest(statistical)
    assert any("min_seeds" in e for e in errors), f"Statistical with N=1 should fail: {errors}"
    print("PASS: test_validate_structural_vs_statistical")


def test_list_manifests():
    manifests = list_manifests()
    assert "sleep_episodic_replay" in manifests, f"Expected sleep_episodic_replay, got {manifests}"
    print("PASS: test_list_manifests")


def test_check_pre_registered():
    assert check_pre_registered("sleep_episodic_replay"), "sleep_episodic_replay should be registered"
    assert not check_pre_registered("nonexistent_experiment"), "nonexistent should not be registered"
    print("PASS: test_check_pre_registered")


if __name__ == "__main__":
    test_load_manifest()
    test_validate_manifest_valid()
    test_validate_manifest_missing_fields()
    test_validate_manifest_invalid_threshold()
    test_validate_structural_vs_statistical()
    test_list_manifests()
    test_check_pre_registered()
    print("\nAll tests PASSED")
