# Source-evaluation mechanism — design, evidence and wired-endpoint predictions

**Status:** module + unit tests implemented, **not wired into the engine**, **no live
WFA run performed in this change**. Parent reviews, wires, then runs. Branch:
`feat/source-evaluation-module` (worktree `C:/Users/Likhith/Documents/projects/ravana_source_eval`).

## 1. Why a mechanism is needed

The echo-immune WFA run (`WFA_ECHO_IMMUNE_RESULT.md`) produced:

| signal | value | reading |
|---|---|---|
| `pd_unparseable_rate` | 1.00 | no own decisions survived the echo-immune parser |
| `forced_choice_explicit_rate` | 0.00 | with a clean single-token Y/N/W surface, the engine emitted **zero** explicit decisions |
| exclusion echo marker | 0.026 | the trusted-list defect is fixed; echo is not the explanation |
| forced-choice echo marker | 0.000 | echo cannot explain the zero |

That is the pre-registered falsifier in `WFA_FAILURE_ANALYSIS.md` §4 fix 4: once
articulation is made unambiguous and still yields ~0 explicit decisions, the
gap is **upstream of articulation — there is no decision routine**. RAVANA can
hold a fact without holding a source attribution; it has no step that turns
`(claim, provenance)` into `ENDORSE / ABSTAIN / WITHDRAW / UNKNOWN`. This module
is that step, as a self-contained pure function.

## 2. Mechanism map

| Brain mechanism | Verified citation (below) | Operationalisation in `source_evaluation.py` | Rule / field |
|---|---|---|---|
| Source judgments are a distinct, explicit evaluation process; item familiarity alone does not yield a source attribution; source amnesia = fact retained, origin lost | Johnson, Hashtroudi & Lindsay 1993 [1] | Endorsement requires an explicit credibility token **and** an origin id; a familiar claim with unknown/absent provenance abstains | `R_NO_EXPLICIT_CREDIBILITY`, `R_UNKNOWN_PROVENANCE` |
| Noetic "know" vs autonoetic "remember": a bare answer lacks recollected source | Tulving 1985 [2] | `ClaimContext.familiarity` is accepted but **never read**; it is not even present in the audit payload | `familiarity`, `SourceDecision.as_dict()` |
| Controlled recollection and automatic familiarity are separable; inclusion/exclusion make them cooperate vs oppose | Jacoby 1991 [3] | Controlled process = explicit token + trusted allow-list; automatic process = familiarity, deliberately excluded; yields `R = I − E > 0` when wired | `trusted_source_ids`, `R_SOURCE_NOT_TRUSTED` |
| Repetition raises truth ratings without provenance (illusory truth) | Hasher, Goldstein & Toppino 1977 [5] | Predicts that a familiarity-driven system fails exclusion; the evaluator must abstain on discredited-but-familiar near-misses | `R_SOURCE_DISCREDITED` |
| Toddlers reevaluate a recently learned mapping when the speaker is later shown unreliable | Dautriche et al. 2021 [4] | Prior `ENDORSE` + invalidation or discredit → `WITHDRAW`, not merely abstain | `R_SOURCE_INVALIDATED` |
| Source monitoring improves with retrieval cues and full attention; training to distinguish "remember" from "know" improves accuracy | Johnson et al. 1993 [1] (text); Önal & Kumkale 2022 [6] | Dormant sleeper is not endorsed until `cue_present`; with a cue it is re-evaluated normally | `R_DORMANT_NO_CUE`, `R_FACT_DORMANT` |
| Epistemic vigilance is calibrated on the communicator, not the content's fluency | Sperber et al. 2010 [7] | Trust is a property of `origin_id` membership in the caller-supplied allow-list, never of claim wording | `R_SOURCE_NOT_TRUSTED` |
| Left anterior PFC is engaged in monitoring/evaluating specific memory characteristics **at retrieval** | Ranganath, Johnson & D'Esposito 2000 [8] | Evaluation is a distinct retrieval-time function the caller invokes per claim — not a stored property of the fact and not an engine-wide global flag | `evaluate_source()` |

The map is **one-directional**: the literature selects the contrasts and the
rule order; measured WFA numbers may not redefine the mechanism after the fact.

## 3. Verified citations

Each was resolved against the publisher, DOI record, or PubMed in this change.
Volume/issue/pages and DOI are as published (not as remembered).

1. **Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993).** Source monitoring.
   *Psychological Bulletin*, **114**(1), 3–28. doi:**10.1037/0033-2909.114.1.3**.
   Verified: APA PsycNET DOI landing + OpenAlex `W4234477066`; full text at the
   Yale Memory & Cognition Lab archive (states source monitoring improves with
   judgment cues and full attention, and that source judgments use flexible
   criteria and are subject to error/disruption).
2. **Tulving, E. (1985).** Memory and consciousness. *Canadian Psychology /
   Psychologie canadienne*, **26**(1), 1–12. doi:**10.1037/h0080017**.
   Verified: APA PsycNET record 1985-27196-001 + Crossref metadata (volume 26,
   issue 1, pages 1-12, publisher APA).
3. **Jacoby, L. L. (1991).** A process dissociation framework: separating
   automatic from intentional uses of memory. *Journal of Memory and Language*,
   **30**(5), 513–541. doi:**10.1016/0749-596X(91)90025-F**.
   Verified: APA PsycNET record 1992-07943-001 + OpenAlex `W2027380211`.
4. **Dautriche, I., Goupil, L., Smith, K., & Rabagliati, H. (2021).** Knowing how
   you know: toddlers reevaluate words learned from an unreliable speaker.
   *Open Mind*, **5**, 1–19. doi:**10.1162/opmi_a_00038**. PMID **34485794**.
   Verified: MIT Press `direct.mit.edu/opmi` + Edinburgh Research Explorer
   publisher PDF + PubMed. **Correction to existing docs:** `WFA_PROCESS_
   DISSOCIATION.md` calls this "Koenig et al. (2026)"; the publisher record is
   Dautriche et al. 2021. Use the corrected citation.
5. **Hasher, L., Goldstein, D., & Toppino, T. (1977).** Frequency and the
   conference of referential validity. *Journal of Verbal Learning and Verbal
   Behavior*, **16**(1), 107–112. doi:**10.1016/S0022-5371(77)80012-1**.
   Verified: APA PsycNET record 1978-02525-001 + publisher PDF (the title is
   "referential **validity**", not "authenticity" — corrected here).
6. **Önal, İ., & Kumkale, G. T. (2022).** Effectiveness of source-monitoring
   training in reducing halo error and negativity bias in a performance
   appraisal setting. *Applied Psychology*. doi:**10.1111/apps.12350**.
   Verified: Koç University research repository + Crossref DOI. Training on
   remember-vs-know produced significantly more accurate judgments via
   increased reliance on specific memory traces; controls relied on general
   impressions.
7. **Sperber, D., Clément, F., Heintz, C., Mascaro, O., Mercier, H., Origgi, G.,
   & Wilson, D. (2010).** Epistemic vigilance. *Mind & Language*, **25**(4),
   359–393. doi:**10.1111/j.1468-0017.2010.01394.x**.
   Verified: Wiley Online Library + APA PsycNET record 2010-17633-001.
8. **Ranganath, C., Johnson, M. K., & D'Esposito, M. (2000).** Left anterior
   prefrontal activation increases with demands to recall specific perceptual
   information. *Journal of Neuroscience*, **20**(22), RC108, 1–5.
   doi:**10.1523/JNEUROSCI.20-22-j0005.2000**. PMID **11069977**.
   Verified: JNEUROSCI publisher page + PubMed + Yale lab PDF.

## 4. Support vs extrapolation ledger

| Claim made here | Status | Basis / boundary |
|---|---|---|
| Item familiarity is not a source attribution; an explicit evaluation step is required before endorsement | **SUPPORTED** (direct analogue) | [1], [2], [3]. The module's `R_NO_EXPLICIT_CREDIBILITY` and ignored `familiarity` field are direct operationalisations. |
| A discredited source should cause withdrawal of a previously endorsed claim, not silence | **SUPPORTED with extrapolation** | [4] shows behavioural reevaluation in 30-month-olds, not verbal `WITHDRAW`. The token is a **proxy** for reevaluation; recorded as extrapolation. |
| Repetition/familiarity will drive false endorsement of discredited near-misses if no explicit check exists | **SUPPORTED** (predictive) | [5] + [3]. Predicts `E > 0` in a familiarity-driven system; the module removes the path. |
| Retrieval cues allow a dormant fact to be re-evaluated | **SUPPORTED with extrapolation** | [1] reports cue effects on source-monitoring accuracy; mapping "dormant sleeper" → RAVANA's dormant edge is an extrapolation. |
| Trust should attach to the origin, not the claim's phrasing | **SUPPORTED** (mechanistic motivation) | [7]. Apes the communicator-calibration principle; the allow-list is an engineering instantiation, not a brain claim. |
| A distinct evaluation step should run at retrieval time | **EXTRAPOLATION** | [8] is fMRI evidence for left-anterior-PFC monitoring at retrieval. A symbolic function is an analogy, not a neural homologue. |
| This module gives RAVANA human-like source memory | **NOT CLAIMED** | It applies an explicit rule table to provenance **supplied by the caller**. It does not discover, recollect, or internally attribute sources. See §7. |

## 5. Exact API

File: `ravana/src/ravana/chat/source_evaluation.py`. All records are frozen
dataclasses; every function is pure and deterministic (no clock, randomness,
I/O, global mutable state, or engine import).

```python
class TrustClass(str, Enum):   # CREDIBLE | UNKNOWN | DISCREDITED
class Decision(str, Enum):     # ENDORSE | ABSTAIN | WITHDRAW | UNKNOWN

@dataclass(frozen=True)
class SourceStatus:
    origin_id: str                    # required: an attribution, not a feeling
    source_type: str = "unknown"      # web_fact | curated | co_occurrence | ...
    reliability: float = 0.0          # NOT sufficient for endorsement
    trust_class: TrustClass = TrustClass.UNKNOWN
    active: bool = True               # active vs dormant
    retrieval_context: str = ""       # where/when retrieved
    credibility_token: Optional[str] = None   # explicit attestation, required

@dataclass(frozen=True)
class ClaimContext:
    source: Optional[SourceStatus] = None
    prior_decision: Optional[Decision] = None   # makes WITHDRAW possible
    cue_present: bool = False
    invalidated: bool = False
    trusted_source_ids: Optional[FrozenSet[str]] = None
    familiarity: float = 0.0          # accepted, NEVER read by the evaluator

@dataclass(frozen=True)
class SourceDecision:
    claim: str; decision: Decision; rule: str; reason: str
    origin_id: Optional[str] = None
    token: str                        # Y | N | W | ?
    def as_dict(self) -> dict: ...    # flat row for the WFA parser

def evaluate_source(claim: str, context: ClaimContext) -> SourceDecision
def integration_hook(claim: str, context: ClaimContext, *,
                     enabled: bool = False) -> Optional[SourceDecision]
def context_from_wfa_item(item: Mapping[str, object], *,
                          prior_decision=None, cue_present=False,
                          invalidated=False, trusted_source_ids=None,
                          familiarity=0.0, active=True) -> ClaimContext
def decision_token(decision: Decision) -> str
```

**Default-off hook.** `integration_hook` returns `None` unless the caller passes
`enabled=True`. There is no module-level enable flag and no engine mutation, so
importing the module cannot change RAVANA's behaviour. Parent wires after review.

**Rule order (first match becomes the audit reason):**

| # | Condition | Decision | Rule id |
|---|---|---|---|
| 1 | empty/missing claim or context | UNKNOWN | `R_INDETERMINATE_INPUT` |
| 2 | prior `ENDORSE` + explicit invalidation or discredited source | WITHDRAW | `R_SOURCE_INVALIDATED` |
| 3 | prior `ENDORSE` + dormant, no cue | WITHDRAW | `R_FACT_DORMANT` |
| 4 | invalidation with no prior endorsement | ABSTAIN | `R_INVALIDATION_NO_PRIOR` |
| 5 | discredited source, fresh claim | ABSTAIN | `R_SOURCE_DISCREDITED` |
| 6 | source absent / trust unknown / no origin id | ABSTAIN | `R_UNKNOWN_PROVENANCE` |
| 7 | dormant, no cue | ABSTAIN | `R_DORMANT_NO_CUE` |
| 8 | no explicit credibility token | ABSTAIN | `R_NO_EXPLICIT_CREDIBILITY` |
| 9 | origin not in trusted allow-list | ABSTAIN | `R_SOURCE_NOT_TRUSTED` |
| 10 | credible + explicit + active + trusted | ENDORSE | `R_EXPLICIT_CREDIBLE_SOURCE` |

Discredit **dominates a stale credibility token** (rule 2 before rule 10).
`familiarity` and `reliability` are never consulted at any rule.

**Token surface** matches `scripts/measure_learn5_wfa.py`
`_FORCED_CHOICE_LABELS = {"y": endorse, "n": abstain, "w": withdraw}`:
`ENDORSE→Y`, `ABSTAIN→N`, `WITHDRAW→W`, `UNKNOWN→?` (no WFA token, so an
indeterminate input can never be scored as an endorsement).

**Wiring sketch for the parent (not applied here):**

```python
from ravana.chat.source_evaluation import (
    context_from_wfa_item, evaluate_source, integration_hook)
ctx = context_from_wfa_item(item, prior_decision=prior, invalidated=was_invalidated,
                            trusted_source_ids=trusted, cue_present=cue)
decision = integration_hook(claim, ctx, enabled=WFA_SOURCE_EVAL_ENABLED)  # flag default False
token = decision.token if decision is not None else None  # feed the forced-choice responder
```

## 6. Predicted WFA endpoints if wired

These are **predictions, not measurements**. No engine was run in this change.
Assumed wiring: the forced-choice responder emits `decision.token` when the
flag is on; all other harness code and thresholds unchanged.

| Endpoint | Pre-registered / target | Predicted with module wired | Why |
|---|---|---|---|
| `forced_choice_explicit_rate` | **> 0.80** | > 0.80 (≈1.00 for well-formed items) | Every valid `(claim, context)` yields Y/N/W; only indeterminate inputs produce `?` and land in `no_answer`. |
| `R = I − E` (both pools) | **≥ 0.25** | ≥ 0.25 (likely near 1.0) | Credible gap claims → ENDORSE (I high); familiar discredited decoys → ABSTAIN (E ≈ 0). |
| `invalidation_withdrawal_primary/sealed` | **≥ 0.50** | ≥ 0.50 (≈1.00 when `invalidated` is supplied) | Prior `ENDORSE` + invalidation → `WITHDRAW`/W. |
| `pd_exclusion_credible_*` (capability parity) | ≥ 0.70 | ≥ 0.70 | Credible items still endorse; the rule is source-selective, not blanket abstention. |
| `pd_unparseable_rate` | hard-fail ceiling | ≈ 0 | The decision is a typed token, not free text; echoed text never enters the surface. |

**Saturation caveat.** A deterministic rule table can pin R and withdrawal near
1.00. Per `ravana-debugging`, a saturated front is a **G6 FLAG**, not capability
evidence: report `0 < R < 0.999` informativeness and do not tune thresholds to
manufacture a mid-range number. Near-1.00 here would show the decision routine
exists; it would **not** show the engine internally recollects sources (§7).

## 7. What remains unverified / parent checklist

1. **No live WFA run.** All endpoint values in §6 are predictions.
2. **Provenance is supplied, not discovered.** The harness provides
   `source_status`/`source_id`; the module does not infer them from the engine's
   graph. A pass demonstrates an explicit decision surface, not autonomous
   source recollection. To claim the latter, a later instrument must withhold
   the label and require the engine to produce it.
3. **Prior-decision bookkeeping is the caller's responsibility.** `WITHDRAW`
   fires only when `prior_decision=ENDORSE` or `invalidated=True` is passed; if
   the harness does not track prior endorsements, withdrawal will read as
   ABSTAIN and the ≥0.50 prediction will not hold. This is a wiring dependency,
   not a module defect.
4. **RMM/false-memory scope.** Illusory-truth and schema-congruence effects
   [5] are predicted, not measured here; the module has no false-memory
   injection condition.
5. **Cross-lingual / paraphrase robustness** of `credibility_token` values is
   untested; the accepted set is explicit and closed by design.
6. **Unit tests were written but pytest was not invoked in this change** (parent
   runs the suite). Two direct, non-pytest Python harnesses were run: (a) 18
   decision scenarios + four WFA adapter cases + hook default-off, 0 failures;
   (b) all 43 parametrized test bodies from the new test file executed through
   a minimal `pytest` shim, 0 failures. These verify logic, not pytest
   collection/integration.

## 8. Falsifiers (pre-registered, measurable)

| # | Falsifier | If observed |
|---|---|---|
| F1 | Wired forced-choice `explicit_rate ≤ 0.80` with echo markers ≈ 0 | The decision surface is not actually being emitted by the hook; check wiring before blaming the parser. |
| F2 | `R < 0.25` on either pool while `explicit_rate > 0.80` | The rules are not source-selective — likely `source_status` is not reaching `context_from_wfa_item`. |
| F3 | Withdrawal < 0.50 despite `invalidated=True` being supplied | Rule 2 is unreachable in the wiring; verify `prior_decision` is actually tracked. |
| F4 | Credible items endorse at < 0.70 (parity failure) | The allow-list is excluding the credible item's own `source_id` — the old trusted-list defect, reintroduced. |
| F5 | A test shows familiarity=1.0 + unknown provenance → ENDORSE | The no-shortcut property is broken; revert the wiring that injects reliability as a token. |
| F6 | A discredited source with a stale credibility token → ENDORSE | Rule precedence inverted; the invalidation gate was moved below the endorsement gate. |
| F7 | `R ≈ 1.00` and withdrawal ≈ 1.00 with no G6 flag reported | Saturation is being hidden; the result is the rule table echoing harness labels, not a capability finding. |
| F8 | Importing the module changes any existing test's output | Default-off guarantee violated; the module must have no import-time side effects. |
