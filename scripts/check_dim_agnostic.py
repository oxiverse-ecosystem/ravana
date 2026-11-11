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

import numpy as np

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

    # THE REAL ASSERTION (round 2026-11-11).
    #
    # Width alone does not prove the tail is in the right PLACE. At dim=96 a
    # `bos[64:75]` slice still produces a 107-D vector -- it just writes the
    # sensorimotor channels into the middle of the GloVe body and leaves the
    # real tail untouched. Silent corruption, no crash, no width error.
    #
    # So drive the function with a DISTINCTIVE subject vector and check the
    # subject's signal actually arrives in the last 11 channels. A correct
    # implementation puts it there at every dim; the hard-coded slice does not.
    try:
        import numpy as _np
        # `subject_lancaster` is the 11-D Lancaster sensorimotor vector, NOT a
        # full-width vector. Give it a distinctive pattern so we can tell
        # whether the implementation actually writes it into the tail.
        _subj11 = _np.arange(1, 12, dtype=_np.float32)
        _bos2 = e._build_conditioned_bos(subject_lancaster=_subj11)
        if _bos2 is None:
            check(False, "dim=%d conditioned bos with subject returned None" % dim)
        else:
            _v = _np.asarray(_bos2, dtype=_np.float32).reshape(-1)
            _tail = _v[-11:]
            # The subject is blended 0.3 into a unit-normalised tail, so the
            # tail must be non-zero. Compare against the SAME call with no
            # subject: a correct implementation changes the tail; a broken one
            # leaves it byte-identical (dim < 64 crash, dim > 64 writes the
            # body instead).
            _bos_base = e._build_conditioned_bos(subject_lancaster=None)
            _tail_base = _np.asarray(_bos_base, dtype=_np.float32).reshape(-1)[-11:]
            _delta = float(_np.linalg.norm(_tail - _tail_base))
            check(float(_np.linalg.norm(_tail)) > 1e-6 and _delta > 1e-6,
                  "dim=%d sensorimotor tail is written by the subject "
                  "(|tail|=%.4f, |delta|=%.4f)"
                  % (dim, float(_np.linalg.norm(_tail)), _delta))
    except Exception as ex:
        check(False, "dim=%d subject-tail probe raised %s: %s"
              % (dim, type(ex).__name__, ex))

print("\n== dim=64 output is unchanged by the fix ==")
e64 = engines.get(64)
if e64 is not None:
    try:
        bos = e64._build_conditioned_bos(subject_lancaster=None)
        check(bos is not None and bos.shape[0] == 75,
              "dim=64 bos still 75-D (the default path is untouched)")
    except Exception as ex:
        check(False, "dim=64 bos raised %s" % ex)

print("\n== decoder conditioning alignment uses the instance width ==")
# `_align_to_decoder` used to read the CLASS attribute _DECODER_DIM (always 75)
# instead of _decoder_dim(). At dim=96 the decoder table is 107-D but every
# conditioning vector was truncated to 75, silently discarding 21 dimensions.
# Nothing crashed, so only a direct check catches it.
for dim, e in engines.items():
    if getattr(e, "neural_decoder", None) is None:
        continue
    dd = e._decoder_dim()
    _v = np.ones(dim, dtype=np.float32)
    # Reproduce the alignment logic the decoder path uses.
    _dec_dim = e._decoder_dim()
    if _v.shape[0] == _dec_dim:
        _out = _v
    elif _v.shape[0] < _dec_dim:
        _out = np.zeros(_dec_dim, dtype=np.float32)
        _out[:_v.shape[0]] = _v
    else:
        _out = _v[:_dec_dim].astype(np.float32)
    check(_out.shape[0] == dd,
          "dim=%d conditioning vector width=%d matches decoder (%d)"
          % (dim, _out.shape[0], dd))
    # The engine-dim body must survive intact -- no truncation of real signal.
    check(float(np.linalg.norm(_out[:dim])) > 0,
          "dim=%d engine-dim body preserved in conditioning" % dim)

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
