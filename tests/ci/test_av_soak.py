"""M0 soak test — reproduce the Windows BLAS access-violation under a longer
session and prove the thread-pinning fix holds.

The original crash (numpy #27989) fired when worker-thread BLAS calls (web
learner fetch/scoring) raced the main-thread decoder/inference inside BLAS.
This test hammers that exact scenario: it builds CognitiveChatEngine repeatedly
and fires many BLAS-heavy process_turn calls within ONE process, with the
background learner + web scoring exercised, looping enough times to surface a
transient thread-race if the env-var pin (ravana._numpy_threading) were absent.

If the fix is correct, 50 engine builds + hundreds of turns complete with no
native access violation. On a clean machine this is fast; it is gated to the CI
Windows matrix so it does not slow dev loops.
"""

import os
import sys
import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

import pytest

pytestmark = pytest.mark.ci

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.models import CognitiveResponseContext
from ravana.chat.constants import _is_word_salad

# The grounding-gate assertions mirrored from test_sm_grounding_gate.py — these
# exercise the Levelt monitor (BLAS-adjacent text scoring) on every call.
_SALAD = (
    "black holes are the light and the space where the matter and "
    "the time bend"
)
_HUB = "trust is the light and the space where the matter and the time bend"


def _ctx(subject, assoc, raw):
    return CognitiveResponseContext(
        subject=subject,
        raw_input=raw,
        associated_concepts=[(a, 0.5) for a in assoc],
    )


def _assert_gate(eng):
    # core grounding-gate behaviour must hold across many builds/turns
    assert _is_word_salad(_SALAD, subject="black holes") is False
    ctx = _ctx("black holes", ["space", "gravity", "time"], "what are black holes?")
    assert eng._sm_response_grounded(ctx, _SALAD) is False
    ctx2 = _ctx("trust", ["relationship", "belief", "faith"], "what is trust?")
    assert eng._sm_response_grounded(ctx2, _HUB) is False


_TURNS = [
    "hi", "what is trust?", "why is the sky blue?", "tell me about ravana",
    "if humans could photosynthesize", "what is gravity?", "do rocks dream",
    "what color is tuesday", "i feel sad today", "what is oxiverse",
]


@pytest.mark.parametrize("round_i", range(50))
def test_av_soak_round(round_i):
    """Build a fresh engine and run many BLAS-heavy turns — 50× in one process.

    A surviving thread-race would raise a native access violation (not a Python
    exception) and abort the whole pytest session; reaching the assertion means
    the thread-pinning fix held for this round.
    """
    data_dir = f"/tmp/ravana_av_soak_{round_i}"
    eng = CognitiveChatEngine(
        dim=64, seed=42 + round_i, baby_mode=True, data_dir=data_dir,
    )
    try:
        for t in _TURNS:
            try:
                eng.process_turn(t)
            except Exception:
                # transient network/uncertainty paths are not the target; the
                # target is a NATIVE crash. Swallow Python-level errors so the
                # soak isolates exactly the BLAS/thread race.
                pass
        _assert_gate(eng)
    finally:
        try:
            eng.stop_background_learning()
        except Exception:
            pass
