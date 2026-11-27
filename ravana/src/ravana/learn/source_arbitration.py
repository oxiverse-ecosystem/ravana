"""Default-OFF, pure source-arbitration for typed facts (TRA-K far repair).

WHAT THIS IS
------------
A deterministic decision module for ONE typed slot conflict. It decides
whether a newly asserted (subject, relation, target) may demote an admitted
incumbent, and it handles source-native identity assertions (``X is a X``)
that a slot writer would otherwise skip. It is NOT wired into the engine:
no function here imports a graph, engine, store, benchmark, answer key or
scorer, and no live path calls it. ``enabled()`` is a caller-side opt-in only.

WHY (brain grounding; full ledger in
docs/investigations/TRA_K_SOURCE_ARBITRATION_PREREG.md)
------------------------------------------------------
* Johnson, Hashtroudi & Lindsay (1993), Psychological Bulletin 114(1):3-28,
  doi:10.1037/0033-2909.114.1.3 -- source memory dissociates from item memory;
  a familiar item is not thereby a well-sourced one. -> credibility is read
  only from provenance, never from stored confidence, recency or label text.
* Sperber et al. (2010), Mind & Language 25(4):359-393,
  doi:10.1111/j.1468-0017.2010.01394.x -- epistemic vigilance is calibrated on
  the communicator. -> a challenger needs a STRICT credibility advantage.
* Dautriche et al. (2021), Open Mind 5:1-19, doi:10.1162/opmi_a_00038
  (PMID 34485794) -- a source proven unreliable is reevaluated. -> an
  unqualified challenger is retained as a separate, weak trace.
* Nader, Schafe & Le Doux (2000), Nature 406:722-726,
  doi:10.1038/35021052 -- reactivation permits a genuine update. -> a
  better-sourced, corroborated challenger MAY supersede; this is not a
  no-update rule.
* Hupbach et al. (2007), Learning & Memory 14(1-2):47-53,
  doi:10.1101/lm.365707 -- an old trace can incorporate new items after a
  reminder. -> the rule is applied to every assertion, not only a far turn.
* Leutgeb et al. (2007), Science 315:961-966, doi:10.1126/science.1135801;
  Yassa & Stark (2011), Trends Neurosci 34(10):515-525,
  doi:10.1016/j.tins.2011.06.006 -- pattern separation stores similar inputs
  separately to avoid catastrophic interference. -> SEPARATE keeps the
  incumbent slot and writes the challenger as a low-strength alternative.

These are computational analogies to symbolic typed edges, not a claim that
the brain implements is_a slots, an env flag or a numeric margin.

LIMITS
------
Boot-free scratch evidence only (n=2/pool, 4 far rows): see
scratch/trak_source_arbitration_result.json. No live engine A/B has been run
here; no capability is claimed. The source identity of an assertion is
supplied by the CALLER from its own provenance; this module cannot verify that
the caller did so honestly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Engine-free import: the licence credibility table is not duplicated.
from ravana.learn.subject_type_evidence import LICENSE_CREDIBILITY

__all__ = [
    "ENV_FLAG", "MARGIN", "MIN_CREDIBILITY", "MIN_CORROBORATION",
    "SENSE_WEIGHT", "EVIDENCE_TIER_WEIGHT", "Action", "SourceAssertion",
    "ArbitrationResult", "enabled", "source_credibility", "arbitrate",
    "self_loop_decision", "integration_hook",
]

ENV_FLAG = "RAVANA_SOURCE_ARBITRATION"

# Constants of the NEW module. No production floor, margin or gate is read or
# changed: production CONGRUENCE_FLOOR=0.20 / STRICT_MARGIN=0.10 stay as they
# are. MARGIN is the source-credibility gap required to overturn an incumbent.
MARGIN = 0.05
MIN_CREDIBILITY = 0.50       # absolute floor on any admitted challenger
MIN_CORROBORATION = 1        # source-reliability guard, not a vote count
SENSE_WEIGHT = {"primary": 1.0, "alternate": 0.5}
EVIDENCE_TIER_WEIGHT = {
    "item_description": 1.0, "type_description": 0.75, "label_only": 0.5,
}
SLOT_CONF_BASE = 0.5
SLOT_CONF_SCALE = 0.5
ALT_CONF_SCALE = 0.2         # keeps a separated trace below the 0.20 floor


class Action(str, Enum):
    REJECT = "REJECT"                   # unadmitted or negated source
    REINFORCE = "REINFORCE"             # same label, no conflict
    ADMIT = "ADMIT"                     # empty slot, first admitted assertion
    SUPERSEDE = "SUPERSEDE"             # credible, corroborated challenger wins
    SEPARATE = "SEPARATE"               # incumbent kept; weak alternative kept
    WRITE_IDENTITY_SLOT = "WRITE_IDENTITY_SLOT"
    SKIP = "SKIP"
    NORMAL = "NORMAL"


@dataclass(frozen=True)
class SourceAssertion:
    """One caller-supplied assertion plus its SOURCE credentials only.

    ``evidence_role``, item/pool identifiers, gold labels and stored edge
    confidences are deliberately absent. ``corroboration`` counts independent
    origin ids from the same licensed source class; it never counts repeats of
    one origin.
    """

    subject: str
    relation: str
    label: str
    license: str = ""
    sense: str = ""
    evidence_kind: str = ""
    origin_id: str = ""
    negated: bool = False
    corroboration: int = 0


@dataclass(frozen=True)
class ArbitrationResult:
    """Auditable decision and the two write strengths it licenses."""

    action: Action
    reason: str
    incumbent_label: str
    challenger_label: str
    incumbent_credibility: float
    challenger_credibility: float
    slot_confidence: float
    alternative_confidence: float


def enabled() -> bool:
    """Caller-side opt-in; inert unless the literal value is ``1``."""
    return os.environ.get(ENV_FLAG, "0") == "1"


def source_credibility(assertion: SourceAssertion) -> float:
    """Credibility from source licence x sense x evidence tier; 0 if unadmitted.

    No field of the asserted item value, no stored confidence and no answer
    key is consulted. A positive declaration cannot raise the licence ceiling.
    """
    ceiling = LICENSE_CREDIBILITY.get((assertion.license or "").strip(), 0.0)
    sense = SENSE_WEIGHT.get((assertion.sense or "").strip().lower(), 0.0)
    tier = EVIDENCE_TIER_WEIGHT.get(
        (assertion.evidence_kind or "").strip().lower(), 0.0)
    if not ceiling or not sense or not tier:
        return 0.0
    return round(ceiling * sense * tier, 12)


def _result(action, reason, incumbent, challenger, cred_i, cred_c,
            slot_conf, alt_conf) -> ArbitrationResult:
    return ArbitrationResult(
        action=action, reason=reason,
        incumbent_label=str(getattr(incumbent, "label", "") or ""),
        challenger_label=str(getattr(challenger, "label", "") or ""),
        incumbent_credibility=float(cred_i), challenger_credibility=float(cred_c),
        slot_confidence=float(slot_conf), alternative_confidence=float(alt_conf))


def arbitrate(incumbent: Optional[SourceAssertion],
              challenger: SourceAssertion) -> ArbitrationResult:
    """Decide whether the challenger may take the slot from the incumbent.

    Priority order is fixed and auditable: negation/unadmitted -> same-label
    reinforcement -> empty slot (absolute floor) -> strict credibility
    advantage with source corroboration -> separation. A below-floor challenger
    against an incumbent is separated, not rejected: keeping it as a weak
    distinct trace is the pattern-separation rule. Ties favour the incumbent,
    so recency alone cannot demote a fact.
    """
    cred_c = source_credibility(challenger)
    cred_i = source_credibility(incumbent) if incumbent is not None else 0.0
    label_i = str(getattr(incumbent, "label", "") or "").strip().casefold()
    label_c = str(challenger.label or "").strip().casefold()
    slot = SLOT_CONF_BASE + SLOT_CONF_SCALE * cred_c
    alt = ALT_CONF_SCALE * cred_c

    if challenger.negated:
        return _result(Action.REJECT, "explicit_negation", incumbent, challenger,
                       cred_i, cred_c, 0.0, 0.0)
    if not label_c or cred_c <= 0.0:
        return _result(Action.REJECT, "unadmitted_source", incumbent, challenger,
                       cred_i, cred_c, 0.0, 0.0)
    if incumbent is None or not label_i:
        if cred_c < MIN_CREDIBILITY:
            return _result(Action.REJECT, "below_absolute_credibility_floor",
                           incumbent, challenger, cred_i, cred_c, 0.0, 0.0)
        return _result(Action.ADMIT, "empty_slot", incumbent, challenger,
                       cred_i, cred_c, slot, 0.0)
    if label_i == label_c:
        return _result(Action.REINFORCE, "same_label", incumbent, challenger,
                       cred_i, cred_c, slot, 0.0)
    if (cred_c >= MIN_CREDIBILITY
            and int(challenger.corroboration) >= MIN_CORROBORATION
            and cred_c > cred_i + MARGIN):
        return _result(Action.SUPERSEDE, "higher_credibility_corroborated",
                       incumbent, challenger, cred_i, cred_c, slot,
                       ALT_CONF_SCALE * cred_i)
    return _result(Action.SEPARATE, "incumbent_source_not_inferior",
                   incumbent, challenger, cred_i, cred_c, slot, alt)


def self_loop_decision(subject: str, obj: str,
                       credibility: float) -> Action:
    """Identity assertions are valid typed traces when source-admitted.

    ``X is a X`` is neither a duplicate nor a contradiction: with a licensed
    source it is the subject's own type cue. Only the two endpoints and the
    caller-supplied credibility are read.
    """
    if str(subject or "").strip().casefold() != str(obj or "").strip().casefold():
        return Action.NORMAL
    if float(credibility or 0.0) <= 0.0:
        return Action.SKIP
    return Action.WRITE_IDENTITY_SLOT


def integration_hook(incumbent: Optional[SourceAssertion],
                     challenger: SourceAssertion) -> Optional[ArbitrationResult]:
    """Default-off opt-in wrapper. Nothing happens until ``enabled()`` is true."""
    if not enabled():
        return None
    return arbitrate(incumbent, challenger)
