"""Default-OFF, unsaturating articulation of a source decision.

This module adds a *second-order* step after the existing deterministic source
evaluator. It does NOT change the evaluator, any gate, threshold, item, prompt
or scoring routine. It is inert unless ``RAVANA_WFA_UNSAT_ARTIC=1`` and a
structured ``ClaimContext`` was staged for this turn.

Mechanism (see docs/investigations/WFA_UNSAT_ARTICULATION_PREREG.md):
  1. first-order source verdict: production ``evaluate_source`` (unchanged);
  2. four noisy samples of the STORED edge confidence, carried into the staged
     context as ``SourceStatus.reliability`` (Ratcliff & McKoon 2008;
     Gold & Shadlen 2007 -- accumulation, EXTRAPOLATION to symbolic samples);
  3. a graded metacognitive confidence ``sigmoid(v)``; recollection licenses the
     controlled verdict only when confidence >= 0.5, otherwise automatic
     familiarity answers (Yeung & Summerfield 2012; Boldt & Yeung 2015 -- graded
     confidence, EXTRAPOLATION to a numeric readout);
  4. inclusion lets familiarity and recollection cooperate; exclusion makes
     them oppose; invalidation needs a confident re-evaluation (Jacoby 1991;
     Johnson, Hashtroudi & Lindsay 1993; Dautriche et al. 2021 -- cue delivery
     is experimenter-supplied, recorded as extrapolation).

The non-saturation is PARAMETRIC: it comes from the fixed diffusion noise
scale, not from a measured human-like memory capacity. This is not capability
evidence and must be reported with that caveat.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
from typing import Optional, Tuple

from .source_evaluation import ClaimContext, Decision, evaluate_source

__all__ = [
    "FLAG", "DDM_STEPS", "SIGMA", "CONF_CRITERION", "SEED_SALT",
    "enabled", "mode_from_instruction", "decide", "apply_at_emission",
]

FLAG = "RAVANA_WFA_UNSAT_ARTIC"
DDM_STEPS = 4
SIGMA = 1.0
CONF_CRITERION = 0.5
SEED_SALT = "wfa-unsat-v1"

_BARE = frozenset({
    "y", "n", "w", "yes", "no", "endorse", "abstain", "withdraw",
    "maintain", "withdrawn", "?",
})

_MODES = (
    ("invalidate", ("reply withdraw or maintain", "has been shown to be unreliable",
                    "fabricated its claims")),
    ("forced", ("reply with one token only", "y = endorse the claim")),
    ("restricted", ("endorse a claim only if", "trusted sources listed in context",
                    "otherwise abstain")),
    ("free", ("answer freely", "familiarity alone is enough")),
)


def enabled() -> bool:
    """True only with the explicit opt-in flag (default OFF)."""
    return os.environ.get(FLAG, "0") == "1"


def mode_from_instruction(prompt: str) -> Optional[str]:
    """Read the decision mode from the INSTRUCTION, never from item metadata."""
    low = (prompt or "").lower()
    for mode, markers in _MODES:
        if any(m in low for m in markers):
            return mode
    return None


def decide(claim: str, context: ClaimContext, mode: str) -> Tuple[Optional[str], dict]:
    """Return the emitted token (or None to defer) and an audit dict.

    Pure: no I/O, no engine import, no global state. ``context.reliability`` is
    the production staged copy of the stored edge confidence; familiarity and
    any item-level metadata are never read.
    """
    audit = {"mode": mode, "applied": False}
    if not enabled() or not isinstance(context, ClaimContext) or not claim:
        return None, audit
    src = context.source
    if src is None or not str(getattr(src, "origin_id", "") or "").strip():
        return None, audit                         # fail closed, no stored origin
    verdict = evaluate_source(claim, context)
    if verdict.decision is Decision.UNKNOWN:
        return None, audit
    reliability = getattr(src, "reliability", 0.0)
    conf = float(reliability) if isinstance(reliability, (int, float)) else 0.0
    conf = min(1.0, max(0.0, conf))

    seed = int(hashlib.sha256(
        (SEED_SALT + "|" + str(claim) + "|" + str(mode)).encode("utf-8")
    ).hexdigest()[:16], 16)
    rng = random.Random(seed)
    step_sigma = SIGMA / math.sqrt(DDM_STEPS)
    v = 0.0
    for _ in range(DDM_STEPS):
        v += conf / DDM_STEPS + rng.gauss(0.0, step_sigma)
    confidence = 1.0 / (1.0 + math.exp(-v))
    recollected = (v > 0.0) if SIGMA == 0 else (confidence >= CONF_CRITERION)
    audit.update({"applied": True, "rule": verdict.rule,
                  "source_decision": verdict.decision.value,
                  "stored_confidence": conf, "evidence": v,
                  "confidence": confidence, "recollected": recollected})

    token: Optional[str]
    if mode == "free":
        token = "endorse"                         # familiarity + recollection cooperate
    elif mode in ("restricted", "forced"):
        single = mode == "forced"
        if verdict.decision is Decision.ENDORSE:
            token = "Y" if single else "endorse"
        elif recollected:
            token = "N" if single else "abstain"
        else:
            token = "Y" if single else "endorse"   # familiarity fallback (F)
    elif mode == "invalidate":
        if verdict.decision is Decision.WITHDRAW and recollected:
            token = "W" if mode == "forced" else "withdraw"
        else:
            token = "N" if mode == "forced" else "maintain"
    else:
        return None, audit
    audit["token"] = token
    return token, audit


def apply_at_emission(response: str, ctx, user_input: str) -> Tuple[str, dict]:
    """Emission-boundary adapter: defer unless fully opt-in and well-formed."""
    audit = {"applied": False}
    if not enabled():
        return response, audit
    if (response or "").strip().lower() in _BARE:
        return response, audit                     # never overwrite a bare decision
    staged = getattr(ctx, "source_context", None) if ctx is not None else None
    if staged is None:
        return response, audit
    try:
        claim, context = staged
    except (TypeError, ValueError):
        return response, audit
    mode = mode_from_instruction(user_input)
    if mode is None:
        return response, audit                     # no source instruction in this turn
    token, audit = decide(claim, context, mode)
    if token is None:
        return response, audit
    if ctx is not None:
        try:
            ctx.unsat_articulation_audit = audit
        except Exception:
            pass
    return token, audit
