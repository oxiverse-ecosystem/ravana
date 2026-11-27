"""Self-contained, deterministic source-evaluation mechanism for RAVANA.

Problem this module exists to solve
-----------------------------------
The echo-immune WFA run (``docs/investigations/WFA_ECHO_IMMUNE_RESULT.md``)
measured ``forced_choice_explicit_rate = 0.00`` with a single-token Y/N/W
surface: RAVANA has no routine that turns a claim + provenance into an
*explicit* source decision.  It cannot say "I endorse this because its source
is credible", "I abstain — my source is unknown", or "I withdraw that — the
source was discredited".  This module supplies that routine.  It is pure
logic: no engine import, no global state, no I/O, no side effects.  The engine
is wired by the caller AFTER review, and stays inert until the caller opts in
via :func:`integration_hook` (default-off).

Brain grounding (each citation independently verified against the publisher
or PubMed; full ledger in ``docs/investigations/SOURCE_EVALUATION_DESIGN.md``)
---------------------------------------------------------------------------
* Johnson, Hashtroudi & Lindsay (1993), *Psychological Bulletin* 114(1), 3-28,
  doi:10.1037/0033-2909.114.1.3 — source judgments are a *separate, explicit*
  evaluation process and dissociate from item familiarity; source amnesia is
  a known fact with an unattributed source.  -> ``R_NO_EXPLICIT_CREDIBILITY``
  and ``R_UNKNOWN_PROVENANCE``: familiarity may never stand in for a source.
* Tulving (1985), *Canadian Psychology* 26(1), 1-12, doi:10.1037/h0080017 —
  noetic "know" vs autonoetic "remember"; a bare answer lacks the recollected
  source.  -> an item-level familiarity score is carried in the record but is
  deliberately NEVER read by the evaluator.
* Jacoby (1991), *Journal of Memory and Language* 30(5), 513-541,
  doi:10.1016/0749-596X(91)90025-F — controlled recollection and automatic
  familiarity are separable processes; inclusion lets them cooperate,
  exclusion makes them oppose.  -> the explicit credibility token and trusted
  allow-list are the controlled process; ``familiarity`` is the automatic one.
* Dautriche, Goupil, Smith & Rabagliati (2021), *Open Mind* 5, 1-19,
  doi:10.1162/opmi_a_00038, PMID 34485794 — toddlers reevaluate a word mapping
  after the speaker is shown unreliable.  -> ``R_SOURCE_INVALIDATED`` and
  withdrawal after a prior endorsement.  (Extrapolation: verbal withdrawal is
  a proxy for the child's behavioural reevaluation; see ledger.)
* Sperber et al. (2010), *Mind & Language* 25(4), 359-393,
  doi:10.1111/j.1468-0017.2010.01394.x — epistemic vigilance: trust is
  calibrated on the communicator, not the content's fluency.  -> the trusted
  allow-list is a property of ``origin_id``, never of the claim's wording.
* Ranganath, Johnson & D'Esposito (2000), *J. Neuroscience* 20(RC108), 1-5 —
  left anterior PFC supports monitoring/evaluation of specific memory
  characteristics at retrieval.  -> the evaluator is a distinct step run at
  retrieval time, not a property of the stored fact.

Support vs extrapolation is explicit: the first four are direct analogues of
the implemented rules; epistemic-vigilance and PFC evidence are mechanistic
motivation (extrapolation to a symbolic module), not direct evidence for this
specific decision table.

Public API
---------
``SourceStatus``      typed provenance record (immutable)
``ClaimContext``      claim-level inputs incl. prior decision and cue
``SourceDecision``    decision + rule + reason (the audit trail)
``evaluate_source``   the pure evaluator -- the integration hook
``integration_hook``  default-off opt-in wrapper (returns None until enabled)
``context_from_wfa_item`` / ``decision_token`` -- WFA-shaped adapters

No function here mutates any argument, reads a module-level mutable, or
imports any part of the engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

__all__ = [
    "TrustClass", "Decision", "SourceStatus", "ClaimContext", "SourceDecision",
    "EXPLICIT_CREDIBILITY_TOKENS", "DECISION_TOKENS",
    "evaluate_source", "integration_hook", "context_from_wfa_item",
    "decision_token", "claim_act_key",
]


class TrustClass(str, Enum):
    """The three source-trust classes used by the WFA instrument."""

    CREDIBLE = "credible"
    UNKNOWN = "unknown"
    DISCREDITED = "discredited"


class Decision(str, Enum):
    """The four explicit decision states."""

    ENDORSE = "ENDORSE"
    ABSTAIN = "ABSTAIN"
    WITHDRAW = "WITHDRAW"
    UNKNOWN = "UNKNOWN"


# An endorsement requires an explicit attestation that the source is
# credible.  A high numeric reliability, a familiar claim, or a stored edge
# weight are ALL insufficient: they are item-strength signals, not source
# attributions (Johnson et al. 1993; Tulving 1985).
EXPLICIT_CREDIBILITY_TOKENS: FrozenSet[str] = frozenset(
    {"credible", "trusted", "verified", "reliable"}
)

# Single-token surface consumed by scripts/measure_learn5_wfa.py
# (_FORCED_CHOICE_LABELS = {"y": endorse, "n": abstain, "w": withdraw}).
# UNKNOWN deliberately has no WFA token: it is an indeterminate *input*, not
# a claim-level decision, and must never be read as an endorsement.
DECISION_TOKENS: Dict[Decision, str] = {
    Decision.ENDORSE: "Y",
    Decision.ABSTAIN: "N",
    Decision.WITHDRAW: "W",
    Decision.UNKNOWN: "?",
}


def claim_act_key(claim: str) -> str:
    """Stable key for recollecting *this engine's own act* on a claim.

    Pure and text-only: no item id, expected answer, source label or world
    model is consulted.  It is the same short sha1 claim anchor used by the
    WFA provenance write, so an act recorded under one turn can be found at a
    later turn of the same claim.  Identical claim texts share a key -- which
    is correct for a claim-level "did I endorse this before?" judgment.
    """
    return "wfa_" + hashlib.sha1((claim or "").encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class SourceStatus:
    """Typed provenance record for one origin.

    ``familiarity``-like item strength is NOT a field here on purpose: it is
    claim-level, not source-level.  ``reliability`` is a source-level estimate
    but is still NOT sufficient for endorsement without ``credibility_token``.
    """

    origin_id: str
    source_type: str = "unknown"
    reliability: float = 0.0
    trust_class: TrustClass = TrustClass.UNKNOWN
    active: bool = True
    retrieval_context: str = ""
    credibility_token: Optional[str] = None

    def has_explicit_credibility(self) -> bool:
        """True only with BOTH a credible trust class and an explicit token.

        A credible-looking class with no token (or a stale/unrecognised token)
        fails closed.  This is the operationalisation of "source judgments
        require an explicit evaluation process" (Johnson et al. 1993).
        """
        token = (self.credibility_token or "").strip().lower()
        return (self.trust_class is TrustClass.CREDIBLE
                and token in EXPLICIT_CREDIBILITY_TOKENS)


@dataclass(frozen=True)
class ClaimContext:
    """Everything the evaluator is allowed to consult for one claim.

    ``familiarity`` is included so a caller can pass the automatic signal and
    so tests can prove it is ignored; the evaluator never reads it.
    ``prior_decision`` is the decision previously emitted for this same claim;
    it is what makes a *withdrawal* distinguishable from a first-time abstain.
    """

    source: Optional[SourceStatus] = None
    prior_decision: Optional[Decision] = None
    cue_present: bool = False
    invalidated: bool = False
    trusted_source_ids: Optional[FrozenSet[str]] = None
    familiarity: float = 0.0


@dataclass(frozen=True)
class SourceDecision:
    """Audit record: the decision and the exact rule/reason that fired."""

    claim: str
    decision: Decision
    rule: str
    reason: str
    origin_id: Optional[str] = None

    @property
    def token(self) -> str:
        """WFA single-token surface (Y/N/W; '?' for indeterminate input)."""
        return DECISION_TOKENS[self.decision]

    def as_dict(self) -> Dict[str, object]:
        """Flat dict for the WFA parser / JSON result rows."""
        return {
            "claim": self.claim,
            "decision": self.decision.value,
            "rule": self.rule,
            "reason": self.reason,
            "origin_id": self.origin_id,
            "token": self.token,
        }


def _decision(claim: str, decision: Decision, rule: str, reason: str,
              origin_id: Optional[str]) -> SourceDecision:
    return SourceDecision(claim=claim, decision=decision, rule=rule,
                          reason=reason, origin_id=origin_id)


def evaluate_source(claim: str, context: ClaimContext) -> SourceDecision:
    """Evaluate one claim's provenance into an explicit, auditable decision.

    Pure and deterministic: no randomness, clock, I/O, global state or engine
    mutation.  Rules are checked in this fixed priority order, so the first
    matching rule is always the recorded audit reason:

    1. indeterminate input                          -> UNKNOWN
    2. prior ENDORSE + invalidated / discredited    -> WITHDRAW
    3. prior ENDORSE + dormant, no cue              -> WITHDRAW
    4. invalidation with no prior endorsement       -> ABSTAIN
    5. source discredited (fresh claim)             -> ABSTAIN
    6. source absent / trust class unknown          -> ABSTAIN
    7. dormant sleeper with no retrieval cue        -> ABSTAIN
    8. no explicit credibility token                -> ABSTAIN
    9. origin not in the trusted allow-list         -> ABSTAIN
    10. otherwise                                   -> ENDORSE

    ``context.familiarity`` and ``source.reliability`` are never consulted:
    familiarity is item memory, not source memory.
    """
    # 1. Indeterminate input -- no claim-level decision is licensed.
    if not isinstance(claim, str) or not claim.strip():
        return _decision(claim if isinstance(claim, str) else "", Decision.UNKNOWN,
                         "R_INDETERMINATE_INPUT",
                         "No non-empty claim was supplied; source evaluation "
                         "cannot run.", None)
    if not isinstance(context, ClaimContext):
        return _decision(claim, Decision.UNKNOWN, "R_INDETERMINATE_INPUT",
                         "No ClaimContext was supplied; source evaluation "
                         "cannot run.", None)

    src = context.source
    origin = src.origin_id if src is not None else None
    prior_endorse = context.prior_decision is Decision.ENDORSE

    # 2. Retroactive source invalidation after a prior endorsement
    # (Dautriche et al. 2021).  Explicit invalidation and a discredited class
    # both count; a discredited class dominates any stale credibility token.
    if prior_endorse and (context.invalidated or
                          (src is not None and
                           src.trust_class is TrustClass.DISCREDITED)):
        return _decision(claim, Decision.WITHDRAW, "R_SOURCE_INVALIDATED",
                         "A previously endorsed claim's source is now "
                         "discredited or was explicitly invalidated.", origin)

    # 3. A previously endorsed fact that has gone dormant must be withdrawn
    # until a retrieval cue reactivates it.
    if (prior_endorse and src is not None and not src.active
            and not context.cue_present):
        return _decision(claim, Decision.WITHDRAW, "R_FACT_DORMANT",
                         "A previously endorsed fact is dormant and no "
                         "retrieval cue is present; endorsement is retracted.",
                         origin)

    # 4. Invalidation without a prior endorsement cannot create a withdrawal.
    if context.invalidated:
        return _decision(claim, Decision.ABSTAIN, "R_INVALIDATION_NO_PRIOR",
                         "Source was invalidated but no prior endorsement "
                         "exists to withdraw; fail closed to abstention.",
                         origin)

    # 5. Fresh claim from a discredited source -> abstain, never endorse.
    if src is not None and src.trust_class is TrustClass.DISCREDITED:
        return _decision(claim, Decision.ABSTAIN, "R_SOURCE_DISCREDITED",
                         "Source is discredited; the claim is not admitted "
                         "regardless of familiarity.", origin)

    # 6. Unknown, absent, or unattributed provenance -> abstain (source amnesia).
    # A claimed source with no origin id is not an attribution at all.
    if (src is None or src.trust_class is TrustClass.UNKNOWN
            or not src.origin_id.strip()):
        return _decision(claim, Decision.ABSTAIN, "R_UNKNOWN_PROVENANCE",
                         "Provenance is unknown, absent, or has no origin id; "
                         "a familiar claim without an attributed source is "
                         "not endorsed.",
                         origin)

    # 7. Dormant sleeper: no endorsement until a cue is present.
    if not src.active and not context.cue_present:
        return _decision(claim, Decision.ABSTAIN, "R_DORMANT_NO_CUE",
                         "Source/fact is dormant and no retrieval cue is "
                         "present; the sleeper is not endorsed.", origin)

    # 8. Explicit credibility token required.  Reliability and familiarity
    # are deliberately not consulted here.
    if not src.has_explicit_credibility():
        return _decision(claim, Decision.ABSTAIN, "R_NO_EXPLICIT_CREDIBILITY",
                         "No explicit credibility token for this source; "
                         "item strength or familiarity cannot substitute.",
                         origin)

    # 9. Optional self-consistent allow-list (epistemic vigilance on origin).
    if (context.trusted_source_ids is not None
            and origin not in context.trusted_source_ids):
        return _decision(claim, Decision.ABSTAIN, "R_SOURCE_NOT_TRUSTED",
                         "Source origin is not in the trusted source set "
                         "supplied by the caller.", origin)

    # 10. Endorsement: credible, active, explicitly attributed, trusted.
    return _decision(claim, Decision.ENDORSE, "R_EXPLICIT_CREDIBLE_SOURCE",
                     "Source is credible, active, explicitly attributed, and "
                     "in the trusted set; claim is endorsed.", origin)


def integration_hook(claim: str, context: ClaimContext, *,
                     enabled: bool = False) -> Optional[SourceDecision]:
    """Default-off integration hook.  Returns ``None`` unless ``enabled``.

    This is the only opt-in surface a caller (or, later, the parent agent)
    uses.  It has no global flag and performs no engine mutation, so importing
    or calling it with the default cannot change system behaviour.  Pass
    ``enabled=True`` from an explicit, reviewed wiring site to get a decision.
    """
    if not enabled:
        return None
    return evaluate_source(claim, context)


def decision_token(decision: Decision) -> str:
    """Map a decision to the WFA single-token surface (Y/N/W/?)."""
    return DECISION_TOKENS[decision]


def context_from_wfa_item(item: Mapping[str, object], *,
                          prior_decision: Optional[Decision] = None,
                          cue_present: bool = False,
                          invalidated: bool = False,
                          trusted_source_ids: Optional[FrozenSet[str]] = None,
                          familiarity: float = 0.0,
                          active: bool = True) -> ClaimContext:
    """Build a :class:`ClaimContext` from a WFA-shaped item dict.

    Reads a plain mapping only -- it does not import or touch the harness.
    ``source_status == 'credible'`` is the harness's explicit credibility
    token; every other status fails closed with no token.  ``source_id`` is
    the origin.  ``familiarity`` is accepted but never consulted by the
    evaluator.
    """
    status = str(item.get("source_status") or "unknown").strip().lower()
    raw_origin = item.get("source_id")
    origin = str(raw_origin) if raw_origin else ""
    if status in {"credible", "trusted", "verified", "reliable"} and origin:
        trust = TrustClass.CREDIBLE
        token: Optional[str] = status
    elif status in {"discredited", "unreliable", "invalidated"}:
        trust = TrustClass.DISCREDITED
        token = None
    else:
        # 'unknown', 'internal', missing, or a credible claim with no id.
        trust = TrustClass.UNKNOWN
        token = None

    reliability = item.get("reliability", 0.0)
    rel = (float(reliability)
           if isinstance(reliability, (int, float, str)) else 0.0)

    source = SourceStatus(
        origin_id=origin,
        source_type=str(item.get("source_type") or "unknown"),
        reliability=rel,
        trust_class=trust,
        active=bool(item.get("active", active)),
        retrieval_context=str(item.get("retrieval_context") or ""),
        credibility_token=token,
    )
    return ClaimContext(
        source=source,
        prior_decision=prior_decision,
        cue_present=cue_present,
        invalidated=invalidated,
        trusted_source_ids=trusted_source_ids,
        familiarity=familiarity,
    )
