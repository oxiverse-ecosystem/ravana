#!/usr/bin/env python
"""Publish all RAVANA packages to PyPI (or TestPyPI)."""
import sys
import os
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PACKAGES = [
    ("ravana-ml", "ravana_ml"),
    ("ravana-grace", "ravana-v2"),
    ("ravana-chat", "ravana"),
    ("ravana-cognitive", "."),
]

def build_all():
    """Build all packages."""
    print("Building all packages...")
    for name, subdir in PACKAGES:
        path = os.path.join(PROJECT_ROOT, subdir)
        print(f"\n=== Building {name} ===")
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--sdist"],
            cwd=path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"FAILED: {name}")
            print(result.stdout)
            print(result.stderr)
            return False
        print(f"OK: {name}")
    return True

def publish_all(repository="pypi", username="__token__", password=None):
    """Publish all built packages to PyPI."""
    if not password:
        password = os.environ.get("TWINE_PASSWORD")
    if not password and repository == "pypi":
        print("ERROR: TWINE_PASSWORD not set for PyPI upload")
        return False
    
    print(f"\n=== Publishing to {repository} ===")
    
    dist_dirs = [
        os.path.join(PROJECT_ROOT, subdir, "dist")
        for _, subdir in PACKAGES
    ]
    
    for dist_dir in dist_dirs:
        if not os.path.exists(dist_dir):
            continue
        files = [os.path.join(dist_dir, f) for f in os.listdir(dist_dir) 
                 if f.endswith(".whl") or f.endswith(".tar.gz")]
        if not files:
            continue
        
        cmd = [
            sys.executable, "-m", "twine", "upload",
            "--repository", repository,
            "-u", username,
        ]
        if password:
            cmd.extend(["-p", password])
        cmd.extend(files)
        
        print(f"Uploading {len(files)} files from {dist_dir}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"FAILED: {dist_dir}")
            print(result.stdout)
            print(result.stderr)
            return False
        print(f"OK: {dist_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Publish RAVANA packages")
    parser.add_argument("--test", action="store_true", help="Upload to TestPyPI instead of PyPI")
    parser.add_argument("--no-build", action="store_true", help="Skip build step")
    parser.add_argument("--password", help="PyPI password/token (or set TWINE_PASSWORD)")
    args = parser.parse_args()

    repository = "testpypi" if args.test else "pypi"
    
    if not args.no_build:
        if not build_all():
            print("Build failed!")
            sys.exit(1)
    
    if not publish_all(repository=repository, password=args.password):
        print("Publish failed!")
        sys.exit(1)
    
    print(f"\n✅ All packages published to {repository}!")

if __name__ == "__main__":
    main()