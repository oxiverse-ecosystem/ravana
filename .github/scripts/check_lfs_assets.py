#!/usr/bin/env python3
"""Fail fast when a git-lfs tracked asset did not actually resolve.

Why this exists
---------------
`actions/checkout` with `lfs: true` can leave a *pointer file* on disk instead
of the real payload (LFS bandwidth quota exhausted, transient LFS backend
error, a runner where the smudge filter was skipped). A pointer file is ~130
bytes of text, so nothing crashes loudly: `os.path.exists()` returns True,
numpy then fails to load it, and the engine silently falls back to
auto-downloading glove.6B.zip (~822 MB). On the Windows soak runner that fetch
is what consumed the entire job budget and produced a bare
"exceeded the maximum execution time" with zero test output.

Checking here turns that whole failure mode into a five-second, clearly-worded
job failure at a known step, instead of a 20-minute timeout that has to be
reverse-engineered from an empty log.

Exit codes: 0 = all assets real, 1 = at least one unresolved pointer.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# git-lfs pointer files always begin with this exact version line.
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

# Non-fatal mode: when CI runs with RAVANA_OFFLINE=1 the engine is told NOT to
# fetch GloVe at runtime, so a missing/unresolved npz is a degraded-but-valid
# run, not a hard failure. Pulling the 144MB LFS asset on every job burns the
# repo's GitHub LFS bandwidth quota (~1 GB per run across the sharded matrix),
# so we no longer force it via `lfs: true` in checkout. Warn instead of fail.
_NONFATAL = os.environ.get("RAVANA_OFFLINE", "") == "1"

# (path relative to repo root, minimum plausible real size in bytes)
_REQUIRED_ASSETS = [
    (os.path.join("data", "ravana_glove_cache.npz"), 1_000_000),
]


def _describe(path: str, min_size: int) -> str | None:
    """Return an error string if the asset is missing/unresolved, else None."""
    if not os.path.exists(path):
        return f"missing entirely (expected at {path})"

    size = os.path.getsize(path)
    try:
        with open(path, "rb") as fh:
            head = fh.read(len(_LFS_POINTER_PREFIX))
    except OSError as exc:
        return f"unreadable: {exc}"

    if head == _LFS_POINTER_PREFIX:
        return (
            f"is still a git-lfs POINTER file ({size} bytes), not the real "
            f"payload — `git lfs pull` did not resolve it"
        )
    if size < min_size:
        return (
            f"is implausibly small ({size} bytes, expected >= {min_size}) — "
            f"likely a truncated or partial checkout"
        )
    return None


def main() -> int:
    failures: list[str] = []

    for rel_path, min_size in _REQUIRED_ASSETS:
        abs_path = os.path.join(_REPO_ROOT, rel_path)
        problem = _describe(abs_path, min_size)
        if problem:
            failures.append(f"  - {rel_path} {problem}")
        else:
            mb = os.path.getsize(abs_path) / 1024 / 1024
            print(f"  [ok] {rel_path} resolved ({mb:.1f} MB)")

    if not failures:
        print("All required git-lfs assets resolved.")
        return 0

    if _NONFATAL:
        # RAVANA_OFFLINE=1 ⇒ engine runs without GloVe; missing npz is expected
        # and acceptable. Do NOT fail the job or consume LFS bandwidth.
        print(
            "\n[warn] git-lfs assets did not resolve, but RAVANA_OFFLINE=1 is "
            "set so tests run without GloVe vectors (degraded coverage, not a "
            "failure):",
            file=sys.stderr,
        )
        print("\n".join(failures), file=sys.stderr)
        print(
            "\nThis is non-fatal: CI no longer pulls the 144MB GloVe LFS asset "
            "to preserve the repository's LFS bandwidth quota.",
            file=sys.stderr,
        )
        return 0

    print("\nERROR: git-lfs assets did not resolve:\n", file=sys.stderr)
    print("\n".join(failures), file=sys.stderr)
    print(
        "\nTests would fall back to downloading glove.6B.zip (~822 MB), "
        "which cannot complete inside the job timeout.\n"
        "Fix: ensure the checkout step sets `lfs: true` and that the "
        "repository has git-lfs bandwidth available, then re-run.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
