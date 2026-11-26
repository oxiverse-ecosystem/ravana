"""Rebuild the pre-fix import shadow used by probe.py --legacy-shadow.

Reads the parent of 254fbc9 (the CAR stale-id fix) from git; writes ONLY under
scratch/car_stale_audit/legacy_shadow/. Never touches production source.

    uv run python scratch/car_stale_audit/make_legacy_shadow.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
SHADOW = HERE / "legacy_shadow"

OVERRIDDEN = (
    "ravana/src/ravana/chat/interface.py",
    "ravana/src/ravana/chat/web_learning.py",
    "ravana/src/ravana/chat/engine_generation.py",
    "ravana/src/ravana/web/web_to_graph.py",
    "ravana_ml/src/ravana_ml/graph.py",
)
PACKAGES = (
    ("ravana/__init__.py", PROJ / "ravana/src/ravana"),
    ("ravana/chat/__init__.py", PROJ / "ravana/src/ravana/chat"),
    ("ravana/web/__init__.py", PROJ / "ravana/src/ravana/web"),
    ("ravana_ml/__init__.py", PROJ / "ravana_ml/src/ravana_ml"),
)

base = subprocess.check_output(
    ["git", "rev-parse", "254fbc9^"], cwd=PROJ, text=True).strip()
print(f"pre-fix base: {base}")

for rel in OVERRIDDEN:
    dest = SHADOW / rel.split("/src/", 1)[1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob = subprocess.check_output(["git", "show", f"{base}:{rel}"], cwd=PROJ)
    dest.write_bytes(blob)
    print(f"  shadowed {rel}")

for rel, current_dir in PACKAGES:
    dest = SHADOW / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    current = current_dir / "__init__.py"
    body = current.read_text(encoding="utf-8").replace("\r\n", "\n")
    header = (
        "# scratch A/B shadow: fall through to current tree for unoverridden submodules\n"
        "__path__ = list(__path__) + "
        + repr([str(current_dir).replace("\\", "/")])
        + "\n"
    )
    dest.write_text(header + body, encoding="utf-8", newline="\n")
    print(f"  package {rel} (path extended)")

print(f"shadow ready at {SHADOW}")
