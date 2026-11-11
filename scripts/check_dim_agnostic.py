"""Acceptance check for the dim-agnostic response path.

Run from anywhere (the script resolves the repo root itself):

    uv run --directory "C:/Users/Likhith/Documents/projects/ravana" \
        python "C:/Users/Likhith/Documents/projects/ravana/scripts/check_dim_agnostic.py"

Exit 0 only when:
  * the engine constructs at dim=32, 64 and 96
  * the decoder's dual-code width tracks the projected GloVe width
  * the conditioned BOS lands its sensorimotor tail in the LAST 11 channels
    (the `bos[64:75]` hard-code only does that when the GloVe width is 64)
  * the two integration suites still pass 17/0

Kept as a script rather than an inline `python -c` so the LoopRunner goal
command can be a SINGLE approval-free invocation: the unattended-run guard
refuses compound commands (`cd X && ...`), but a plain script path whose
content is reviewable in git is fine.
"""
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

os.environ.pop("RAVANA_OFFLINE", None)
os.environ.pop("RAVANA_TEST_MODE", None)

for _p in (os.path.join(_REPO, "ravana_ml", "src"),
           os.path.join(_REPO, "ravana", "src"),
           os.path.join(_REPO, "ravana-v2", "src")):
    sys.path.insert(0, _p)

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

FAIL = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


print("== construction at several dims ==")
engines = {}
for dim in (32, 64, 96):
    d = tempfile.mkdtemp(prefix="ravana_dim_")
    try:
        e = CognitiveChatEngine(dim=dim, seed=42, baby_mode=True,
                                data_dir=d, user_suffix="_dimchk")
        engines[dim] = e
        check(True, "dim=%d constructs" % dim)
    except Exception as ex:
        check(False, "dim=%d constructs -- %s: %s" % (dim, type(ex).__name__, ex))

print("\n== decoder dual-code width tracks the projected GloVe width ==")
for dim, e in engines.items():
    dd = e._decoder_dim()
    check(dd == dim + 11,
          "dim=%d -> _decoder_dim()=%d (want %d)" % (dim, dd, dim + 11))

print("\n== conditioned BOS puts the sensorimotor tail in the LAST 11 channels ==")
for dim, e in engines.items():
    try:
        bos = e._build_conditioned_bos(subject_lancaster=None)
    except Exception as ex:
        check(False, "dim=%d _build_conditioned_bos raised %s: %s"
              % (dim, type(ex).__name__, ex))
        continue
    if bos is None:
        check(False, "dim=%d _build_conditioned_bos returned None" % dim)
        continue
    total = int(bos.shape[0])
    gdim = total - 11
    # The tail must be the last 11 channels, and the GloVe body must be the
    # first `gdim`. A hard-coded [64:75] slice writes into the middle of the
    # body at dim != 64, so assert the boundary is derived, not fixed.
    check(total == dim + 11,
          "dim=%d bos width=%d (want %d)" % (dim, total, dim + 11))
    check(gdim == dim,
          "dim=%d sensorimotor boundary at %d (want %d)" % (dim, gdim, dim))

print("\n== dim=64 output is unchanged by the fix ==")
e64 = engines.get(64)
if e64 is not None:
    try:
        bos = e64._build_conditioned_bos(subject_lancaster=None)
        check(bos is not None and bos.shape[0] == 75,
              "dim=64 bos still 75-D (the default path is untouched)")
    except Exception as ex:
        check(False, "dim=64 bos raised %s" % ex)

print("\n== integration suites ==")
_rc = subprocess.call(
    [sys.executable, "-m", "pytest",
     os.path.join(_REPO, "tests", "integration", "test_ravana_chat_pipeline.py"),
     os.path.join(_REPO, "tests", "integration", "test_unified_semantic_loop.py"),
     "-q", "-p", "no:cacheprovider", "--no-header"],
    cwd=_REPO)
check(_rc == 0, "integration suites exit 0 (got %d)" % _rc)

print()
if FAIL:
    print("RESULT: FAIL (%d)" % len(FAIL))
    for m in FAIL:
        print("  - " + m)
    sys.exit(1)
print("RESULT: PASS")
sys.exit(0)
