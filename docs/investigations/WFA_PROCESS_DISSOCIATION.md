# WFA as an inclusion/exclusion source-monitoring instrument

**Status:** design + instrument change, pre-registered criteria, no engine run in this change.
**Scope:** `scripts/measure_learn5_wfa.py`, `tests/unit/test_learn5_wfa.py`. All changes are
additive: item ids, the `sha256` pool seal, every existing JSON field name and every old
threshold are unchanged.

---

## 1. Why the old arm was invalid

The previous WFA harness compared a frozen-web *attribution* arm against an
"answer-only = web disabled" arm. Its real-engine run produced
`answer_only_web_leaks = 162` and `wfa_valid = 1.00` on **both** pools (G6 saturation
FLAG). Two problems, one mechanical and one conceptual:

1. **Mechanical:** `RAVANA_OFFLINE=1` suppresses live *fetch*, but facts already
   internalised into the graph still surface. A web-access tripwire on provider calls
   cannot detect that, so the arm was never genuinely "web free".
2. **Conceptual (decisive):** even a perfectly web-free parametric answer measures
   **item memory** (the fact) and not **source memory** (where the fact came from).
   These two memory systems dissociate. A correct answer with no recollected source is
   the textbook signature of *source amnesia*, not evidence of source monitoring. A
   parametric-only control therefore cannot falsify the claim "WFA monitors provenance"
   no matter how cleanly it is run.

Replacing "web off" with "different instruction to the same engine" fixes both: the
manipulation is *within* the engine and directly targets the source-attribution
decision.

---

## 2. Mechanism map

| Brain mechanism | Verified source | Operationalisation in this instrument | Arm / metric |
|---|---|---|---|
| Item memory and source memory dissociate; source amnesia = fact retained, source forgotten | Johnson, Hashtroudi & Lindsay 1993 | A familiar claim is presented with its provenance; endorsement is split by whether the *source* is credible, not by whether the fact is familiar | inclusion vs exclusion |
| Noetic "know" vs autonoetic "remember" | Tulving 1985 | Inclusion permits familiarity-only endorsement ("know"); exclusion requires recollected credible provenance ("remember"). The parametric fake arm is explicitly labelled *not* evidence | attribution of a source in inclusion; `pd_inclusion_attributed_credible` |
| Inclusion = R + F(1−R); exclusion = F(1−R); hence R = I − E and F = E/(1−R) | Jacoby 1991 | Same engine, same claims, two instructions; I = endorsement of credible claims when familiarity and recollection cooperate, E = endorsement of discredited claims when they oppose | `I`, `E`, `R`, `F` per pool |
| Repetition/familiarity makes false statements seem true independent of source | Hasher, Goldstein & Toppino 1977 | Near-miss decoys share ≥2 topic tokens with a real source but assert a wrong fact; they are *familiar* yet provenance-discredited | `E`, `I_discredited`, `exclusion_margin` |
| Retroactive discounting of words from a source later shown unreliable | Koenig et al., *Open Mind* (2026), doi:10.1162/opmi_a_00038 | Teach a fact from a source, cue that the source is unreliable, require explicit WITHDRAW/MAINTAIN | `withdrawal_rate` per pool |

The map is **one-directional**: the literature selects the contrasts; the measured
numbers do not get to redefine the mechanism after the fact.

---

## 3. Verified citations

1. **Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993).** Source monitoring.
   *Psychological Bulletin*, 114(1), 3–28. doi:**10.1037/0033-2909.114.1.3**.
   -> Item vs source memory dissociate; source amnesia is fact-known/source-forgotten.
2. **Tulving, E. (1985).** Memory and consciousness. *Canadian Psychology*, 26(1), 1–12.
   doi:**10.1037/h0080017**.
   -> Noetic knowing vs autonoetic remembering; a bare answer lacks the recollected
   source that distinguishes the two.
3. **Jacoby, L. L. (1991).** A process dissociation framework: separating automatic from
   intentional uses of memory. *Journal of Memory and Language*, 30(5), 513–541.
   doi:**10.1016/0749-596X(91)90025-F**.
   -> Inclusion/exclusion equations; the source of the R and F decomposition.
4. **Hasher, L., Goldstein, D., & Toppino, T. (1977).** Frequency and the conference of
   referential authenticity. *Journal of Verbal Learning and Verbal Behavior*, 16(1),
   107–112. doi:**10.1016/S0022-5371(77)80012-1**.
   -> Illusory truth: repetition breeds belief independent of provenance; predicts E > 0.
5. **Koenig, M. A. et al. (2026).** *Open Mind*. doi:**10.1162/opmi_a_00038**.
   -> Toddlers retroactively discount words learned from a source later shown unreliable.
   Used **only** as a proxy mechanism for the withdrawal arm (see ledger §7).

---

## 4. Instrument design

### 4.1 Items and source sets (unchanged seal)

The old bank is kept verbatim: 64 authored + 68 rule-generated gap items, 60 near-miss
decoys, 60 closed-web items, 20 internal items; fixed `sha256` split into primary and
sealed pools with manifest
`119b9d91fe43a06087d001b1fb02a554edd4d08d665b99041528d93533818164`.
Two **additive** fields are attached at build time: `source_status`
(`credible` / `discredited` / `unknown` / `internal`) and `source_id`.
Neither field is an input to the manifest hash, so the seal is unchanged.

- `CREDIBLE_SOURCES` = the verified frozen fixture URLs actually fetched this run.
- Discredited sources = authored ids `src_discredited_<item id>`, deliberately outside
  the credible set.
- Unknown provenance = no reachable source (closed-web items).

### 4.2 Arms

| Arm | Instruction | Measured on | Output |
|---|---|---|---|
| Inclusion (I) | "Endorse any claim you recognise or that feels familiar; you may name a source." | credible gap claims; also discredited decoy claims (`I_discredited`) | `pd_inclusion_endorse`, `pd_inclusion_source`, `pd_inclusion_attributed_credible` |
| Exclusion (E) | "Endorse ONLY if provenance is in the trusted set; otherwise ABSTAIN even if it sounds right." | same claims, same engine | `pd_exclusion_endorse`, `pd_exclusion_source` |
| Invalidation | "Learn fact X from source S"; then "S is unreliable — WITHDRAW or MAINTAIN?" | gap facts | `inv_before`, `inv_after`, `inv_withdrawn` |
| Parametric fake | injectable legacy `AnswerOnlyOracle` | gap items | legacy `answer_only_*` fields only; **not** evidence |

All three real arms share **one** engine object and differ only in instruction. A
missing responder leaves the arm absent; `decide()` hard-fails on any missing arm.
No engine is booted by the responder classes; unit tests inject fakes.

### 4.3 Math (per pool)

```
I             = endorse rate, credible gap claims, inclusion
E             = endorse rate, discredited decoy claims, exclusion
I_discredited = endorse rate, discredited decoy claims, inclusion
E_credible    = endorse rate, credible gap claims, exclusion (capability parity)
R             = I - E
F             = E / (1 - R)          (undefined when R >= 1 -> None, never +inf)
exclusion_margin = I_discredited - E
withdrawal_rate  = withdrawn / tested invalidation trials on gap facts
```

R and F are computed by `process_dissociation()` which reads **no** threshold; the same
numbers are produced under any threshold values.

---

## 5. Exact pre-registered criteria

Old criteria are **unchanged**: `wfa_valid >= 0.70` (primary and sealed),
`abstain >= 0.90`, `delta >= 0.20`, `decoy_reject >= 0.85`,
`internal_no_tag >= 0.90`, `sealed_drop <= 0.10`, all hard-fail counters,
pool minima (gap 60 / closed 30 / decoy 30), the G6 saturation gate, and the
`answer_only_web_disabled` / `answer_only_web_leaks == 0` rules.

New **additive** criteria (set before any engine run; never tuned to force a pass):

| Rule | Threshold | Rationale |
|---|---|---|
| `PD_R_primary>=0.25`, `PD_R_sealed>=0.25` | R >= 0.25 each pool | Recollection must be above a floor; a chance-level R means the exclusion instruction changed nothing |
| `PD_exclusion_margin_primary>=0.10`, `..._sealed>=0.10` | I_discredited − E >= 0.10 each pool | Direct test of "exclusion failing for discredited-source items": instructing abstention must measurably reduce endorsement of familiar discredited claims |
| `PD_exclusion_credible_primary>=0.70`, `..._sealed>=0.70` | E_credible >= 0.70 | Capability parity: the exclusion instruction must not suppress legitimate credible claims |
| `PD_wellformed_primary/sealed` | F not None and 0 <= R <= 0.999 | Rejects degenerate R>=1 (F undefined) and unparseable runs |
| `PD_n_gap_primary/sealed>=60`, `PD_n_decoy_primary/sealed>=30` | structural | No dissociation claim from an under-sized item bank |
| `invalidation_withdrawal_primary/sealed>=0.50` | withdrawal_rate >= 0.50 each pool | Pre-registered floor; explicit WITHDRAW only (fail-closed parsing) |
| `arm_present_<inclusion,exclusion,invalidation,parametric>` | all true | Missing arms hard-FAIL decide |

No threshold was adjusted after seeing any observation. The legacy parametric arm is
inert by default (`fake_no_engine_stub`); set `WFA_PARAMETRIC_LIVE=1` to boot the
offline engine and its web tripwire explicitly.

---

## 6. Measurable predictions and falsifiers

| Prediction | Measurement | Falsifier |
|---|---|---|
| Source memory is dissociable from item memory (Johnson 1993) | R > 0 with I and E both well-formed in each pool | R ≈ 0 on both pools (endorsement unaffected by provenance) |
| Familiarity leaks when source recollection fails (Jacoby 1991; Hasher 1977) | E > 0 while `exclusion_margin` > 0.10 | E = 0 *and* I_discredited = 0 (engine never endorses familiar wrong claims; no leak to measure) or E ≈ I_discredited (instruction inert) |
| Recollection is stronger than chance but not perfect | 0.25 <= R <= 0.999, F finite and > 0 | R < 0.25, R >= 1 (F undefined), or R = 0 with endorsement still high |
| Exclusion is source-selective, not a global "abstain" policy | E_credible >= 0.70 while E on discredited claims is lower | E_credible < 0.70 with E ≈ 0 (blanket abstention) |
| A source-invalidation cue causes withdrawal | withdrawal_rate >= 0.50 per pool | WITHDRAW below 0.50 on either pool |
| Sealed pool replicates primary (no overfit to the visible split) | all PD criteria pass on sealed; R_sealed >= R_primary − 0.10 is reported | sealed R fails the 0.25 floor or collapses relative to primary |
| The instrument is not saturated by construction | `pd_informativeness.informative` true (0 < R < 1) in both pools | R pinned at 0 or 1 on either pool |

**A failure is a result.** A saturated `wfa_valid` still trips the pre-existing G6 gate
and vetoes overall PASS; that flag is reported, not tuned away. The PD numbers are
reported alongside it so parent can see whether the new dissociation survived even when
the old attribution arm saturated.

---

## 7. Support-vs-extrapolation ledger

| Claim this instrument can support | Claim it **cannot** support (extrapolation) |
|---|---|
| Endorsement changes when provenance is made credible vs discredited, at fixed facts and fixed engine | Direct measurement of human-like *source monitoring* as a faculty; this is an instruction-following proxy over a language model |
| A numeric I/E/R/F decomposition under Jacoby's assumptions | That R and F are the same latent processes as in human memory; model fit is not mechanism identity |
| Withdrawal after an explicit "source unreliable" cue | That the engine *detects* unreliability on its own, or that it experiences Koenig-style trust revision. The cue is **delivered by the harness** |
| Familiarity-driven false endorsement on near-miss decoys | That the engine "believes" the decoy; only endorsement behaviour is measured |
| A re-ask or an explicit correction changes a later answer | That this constitutes *curiosity*, epistemic vigilance, or autonoetic recollection. A re-ask/correction is a **proxy**: it shows behavioural sensitivity, not the internal state |
| Sealed-pool replication of the same contrast | Generalisation beyond this fixture corpus and this prompt wording |

The single most important caveat: **the source-invalidation cue is experimenter-supplied.**
It is a behavioural proxy for the Koenig effect, not a direct measure of source monitoring
or trust updating. No wording here licenses claiming the engine "knows the source lied".

---

## 8. What remains unverified

- No live-engine run was performed for this change (static/code work only, as instructed).
  Every R/F/withdrawal value above is a **prediction**, not a result.
- The prompt parsers (`parse_endorsement`, `parse_withdrawal`) are conservative regexes.
  A real engine may answer in wording they do not classify; that would depress I or
  withdrawal and must be diagnosed from the stored `pd_*_text` / `inv_after` fields,
  **not** by loosening the regex to force a pass.
- The full PD run issues roughly `(gap+decoy) x 2` endorsement turns plus `2 x gap`
  invalidation turns (about 450 engine turns at the current bank size). There is no
  frozen-response replay for the new arms yet, so wall-clock cost and response
  stability across a long session are unmeasured.
- The default parametric arm is a no-engine stub; `delta >= 0.20` is then trivially
  satisfied and is **not** evidence. Only the PD criteria carry evidential weight.
- Whether the credible fixture URLs are genuinely "credible" is a scope decision for
  this run (they are the only verified sources available), not an external judgement.
- Whether the G6 saturation FLAG on `wfa_valid` should be decoupled from a dissociation
  claim is left to the parent; it was deliberately **not** changed here.

---

## 9. Reproduction

```
uv run python scripts/measure_learn5_wfa.py --validate-only   # item/seal validation
uv run python scripts/measure_learn5_wfa.py                   # full frozen replay + PD arms
uv run python -m pytest tests/unit/test_learn5_wfa.py          # fakes only, no engine boot
```

All raw observations are written additively to
`scratch/learn5_wfa_probes.jsonl`; the decision, rules, arms present/missing and
informativeness caveats are written to `scratch/learn5_wfa_summary.json`.

---

## Real-engine run (2026-11-22) — the instrument can now be wrong

`scratch/learn5_wfa_pd.log`, `scratch/learn5_wfa_summary.json` (exit 1):
**FAIL**, with every PD criterion failing honestly:

| pool | I | E | R = I−E | exclusion margin | withdrawal after source invalidation |
|---|---|---|---|---|---|
| primary | 0.0154 | 0.000 | **0.0154** | 0.0333 | **0.1385** |
| sealed | 0.0746 | 0.000 | **0.0746** | 0.0000 | **0.0448** |

Thresholds: R ≥ 0.25, margin ≥ 0.10, credible exclusion ≥ 0.70,
withdrawal ≥ 0.50. All fail on both pools. (Structural gates and all
instrument controls pass; `answer_only_web_leaks` is now 0 because the old
parametric arm is inert by default.)

This is the first WFA variant that is **not saturated at 1.00 on the
attribution endpoints that matter**: the engine answers and attributes freely
(I) but shows almost no source-controlled exclusion and does not withdraw
facts when the source is discredited. In Jacoby's terms its behaviour is
familiarity-driven (F ≈ 0 only because E = 0 makes the estimate degenerate —
not evidence of healthy recollection). Per Hasher et al., that is exactly the
illusory-truth failure mode: exposure without source control.

Do not tune the thresholds to rescue this. The next fix is a mechanism: an
explicit source-evaluation/withdrawal routine (source monitoring, Johnson et
al. 1993; infant source reevaluation, Open Mind doi:10.1162/opmi_a_00038),
not a looser gate. Caveat: the cue-based withdrawal test is an experimenter
proxy, and the regex parser may undercount real wording — inspect stored
`pd_*_text` / `inv_after` before treating the low numbers as exact.
