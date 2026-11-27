# WFA prior-act trace — brain grounding, scratch A/B, and decision

## 1. The measured gap and the exact missing quantity

WFA Phase 0 (`scratch/learn5_wfa_summary.json`, branch `docs/wfa-phase0-complete`
@ `4ff4beb`) reports `pass = false`. The internal-source arm is 1.00 on all
primary conditions and on sealed credible/unknown, but **sealed discredited =
0.80** (`withdraw_rate 0.80`, `abstain_rate 0.20`). The four recorded failures
`gg039, gg032, g64, g62` fired `R_INVALIDATION_NO_PRIOR`: stored provenance was
found, the invalidation cue tagged 2–3 edges, but there was no record that the
engine had **previously endorsed this claim**. `evaluate_source` rule 4
(`source_evaluation.py:249-254`) therefore correctly fails closed to ABSTAIN.

Subsequent live diagnostics on the current `main` store located the loss
precisely:

- `scripts/measure_learn5_wfa.py:2110-2113` captures the first-turn decision
  from `engine._last_source_evaluation`, a **single global slot**
  (`engine.py:1952`, written at `engine.py:8149`). Any later turn overwrites it.
- `graph_provenance_for_claim` (`:1801-1842`) keys provenance on `source_id`,
  which is `sha256("WFA-IS|<condition>|<claim>")[:12]` (`:1751-1757`).
  `gg032` and `g62` have **byte-identical claim text** —
  `"daytime and nighttime happen because Earth rotates on its axle."` — hence
  identical source ids. When `gg032`'s discredited condition invalidates, it
  tags `g62`'s authoritative claim-id edge too; `g62`'s *first* turn already
  sees `source_status=discredited` and ABSTAINs, so the prior act is never
  formed (`scratch/wfa_g62_isolation.json`).

This is a **source-memory-of-ones-own-act** deficit, not a trust-gate, threshold
or familiarity problem: supplying only `prior_decision=ENDORSE`, with stored
provenance byte-identical, flips ABSTAIN → WITHDRAW in the boot-free probe
(`WFA_PRIOR_ENDORSEMENT_NEXT_EXPERIMENT.md` §3).

## 2. Brain mechanism and citations (each independently verified)

The relevant human capacity is *internal source monitoring*: remembering
whether **I** thought/said/endorsed something, as opposed to knowing the item.
Source attributions are evaluated at retrieval from qualities of experience,
not read off an abstract source tag; they are dissociable from item familiarity
and depend on frontal monitoring systems.

1. **Johnson, Hashtroudi & Lindsay (1993), *Psychological Bulletin* 114(1):3–28,
   doi:10.1037/0033-2909.114.1.3** — the source-monitoring framework. Directly
   names *internal source monitoring* (discriminating what one thought from
   what one said) and states that people do not retrieve an abstract source
   label; memory records are evaluated by decision processes at remembering.
   Verified against APA PsycNet record 1993-40167-001 and the full PDF
   (`onlineacademiccommunity.uvic.ca/.../JohnsonHashtroudiLindsay1993_0.pdf`).
2. **Tulving (1985), *Canadian Psychology* 26(1):1–12,
   doi:10.1037/h0080017** — autonoetic "remember" vs noetic "know". A bare
   answer carries item knowledge without the recollected act of having
   committed to it; this is exactly the ABSTAIN/WITHDRAW dissociation.
3. **Mitchell & Johnson (2009), *Psychological Bulletin* 135(4):638–677,
   doi:10.1037/a0015849** and **Ranganath, Johnson & D'Esposito (2000),
   *J. Neurosci.* 20(22):RC108** — left PFC supports systematic, source-specific
   monitoring at retrieval, right PFC heuristic familiarity; source judgments
   track specific retrieved characteristics, not item strength. -> the trace
   must key on the claim and an emitted decision, never on confidence.
4. **Nader, Schafe & Le Doux (2000), *Nature* 406:722–726,
   doi:10.1038/35021052** — retrieval returns a consolidated trace to a labile
   state in which new information is written into the old trace. -> the prior
   act must be **written at the act** and read later; inferring it from a
   current tag at retrieval is precisely the failure mode observed here.
5. **Dautriche, Goupil, Smith & Rabagliati (2021), *Open Mind* 5:1–19,
   doi:10.1162/opmi_a_00038, PMID 34485794** — 30-month-olds reevaluate a word
   mapping after the speaker is shown unreliable. -> withdrawal requires a
   prior commitment plus a discredited source. (The verbal `W` token and the
   experimenter-delivered invalidation tag remain explicit extrapolations.)
6. **Jacoby (1991), *J. Memory & Language* 30(5):513–541,
   doi:10.1016/0749-596X(91)90025-F**; **Sperber et al. (2010), *Mind &
   Language* 25(4):359–393, doi:10.1111/j.1468-0017.2010.01394.x** —
   controlled recollection vs automatic familiarity, and trust calibrated on
   the communicator rather than content fluency. -> the trace stores a
   decision enum keyed by claim text; no familiarity/reliability shortcut.

**Support vs extrapolation.** 1–3 are direct analogues of the implemented
read/write; 4 is the reason a durable act record is needed; 5 is a
behavioural analogue (verbal withdrawal is a proxy); 6 frames the instrument.
The symbolic `dict[claim_id] -> Decision` is an engineering instantiation, not
a neural claim.

## 3. Scratch A/B design (all production code imported unmodified)

`scratch/wfa_prior_trace_ab.py` monkeypatches **one** engine boundary method,
`CognitiveChatEngine._apply_source_monitoring_gate`, behind `RAVANA_PRIOR_ACT_TRACE`
(default off):

- on the emission boundary it records the engine's **own** emitted
  `SourceDecision.ENDORSE` under `claim_act_key(claim)` (the same sha1 claim
  anchor used by the g05 provenance fix);
- when a staged context is invalidated and has no prior ENDORSE, it substitutes
  the stored ENDORSE via `dataclasses.replace` **before** calling the original
  gate. Nothing else changes: no evaluator rule, gate, threshold, item, prompt
  or grading field.

Anti-leakage properties, asserted in `--self-check` and in the live runs:
the only value stored is the engine's decision enum; item `expected_answer`,
`source_status`, `source_id` and `reliability` are never read (staged
fingerprint invariant under item mutation); distinct claim texts get distinct
keys; with the flag off the wrapper is byte-identical to the unmodified gate.

## 4. Results (all executed, not predicted)

**Scratch self-check** (`scratch/wfa_prior_trace_ab.py --self-check`, boot-free):
`flag_off_byte_identical`, `item_metadata_invariant`, `key_isolation`,
`act_is_engine_decision` all true → `SELF_CHECK_OK`.

**Decisive controlled live probe** (`scratch/wfa_prior_trace_four_controlled.py`,
three independent fresh copies of the same store, four recorded failures,
bank order preserved):

| arm | endpoint | substitutions | control |
|---|---|---|---|
| baseline 1 (unmodified production) | 3/4 WITHDRAW | 0 | — |
| baseline 2 (unmodified production) | 3/4 WITHDRAW | 0 | `baseline_replicates = true` |
| scratch trace ON | **4/4 WITHDRAW** | 1 | `substitutions_only_on = true` |

g62 — the item whose first turn ABSTAINed because a duplicate-claim item had
already invalidated the shared source id — now emits WITHDRAW with stored
provenance unchanged. With the trace off it emits `ABSTAIN / R_INVALIDATION_NO_PRIOR`.

**Broad state-controlled A/B** (`scratch/wfa_prior_trace_broad_ab_v2.log`,
132 gap items, discredited condition only, one shared engine per arm, a fresh
store copy per arm):

| arm | primary correct | sealed correct | sealed failures |
|---|---|---|---|
| unmodified production | 0.9231 (60/65) | 0.6269 (42/67) | 25 items |
| scratch trace ON | **1.0000 (65/65)** | **0.9851 (66/67)** | `g26` only |

On the first-20-per-pool subset used by the recorded Phase 0 summary:
sealed discredited **14/20 → 19/20**, primary 20/20 unchanged. `g26` remains
`ABSTAIN / R_INVALIDATION_NO_PRIOR` with `invalidated=2` — reported honestly as
a residual, not tuned away. Controls: `flag_off_is_unmodified_production = true`,
`same_store_and_bank_order = true`, `substitutions_only_on = true`.

An earlier unrestricted run is **retracted**: its arm-2 improvement was a
store-state confound (both arms shared one `data_dir` and the wrapper had a
tuple-attribute bug, so zero substitutions fired). The retracted numbers are
kept in `scratch/wfa_prior_trace_broad_ab.log` for audit (its JSON was
overwritten by the v2 run); the v2 run above uses distinct store copies and a
fixed wrapper.

**Production implementation verification**
(`scratch/wfa_prior_trace_production_verify.py`, 3 fresh store copies, flag
unset / `0` / `1`): unset == `0` == 3/4, `1` == 4/4, `unset_equals_zero=true`,
`no_trace_when_off=true`, `trace_written_when_on=true`.

**Regression suite:** `test_source_prior_act_trace.py` +
`test_source_gate_early_return.py` + `test_source_evaluation_wiring.py` +
`test_learn5_wfa.py` in one pytest process → **89 passed**.

## 5. Decision

**IMPLEMENT, default-off, no gate/threshold/item/scoring change.** The
claim-ID-keyed prior-act trace fixes the measured endpoint on both the decisive
four-item probe and the 132-item broad A/B, and the gain survives the
flag-equivalence, item-metadata-invariance, key-isolation, baseline-replication
and store-isolation controls. The only production changes are one pure key
helper in `source_evaluation.py`, one new state field and a default-off branch
in `_apply_source_monitoring_gate` (engine.py), and a new unit-test module. No
existing decision rule, threshold, gate, prompt, item or grader was edited.

**Caveats.** The internal-source arm still uses a deterministic rule table;
a saturated endpoint is G6 mechanism evidence, not autonomous human-like
source recollection. The verbal `W` token and the experimenter-delivered
invalidation tag remain extrapolations. One sealed item (`g26`) still fails.
The full 120-condition harness verdict was not re-run here; this change is
measured with the harness's own runner/summarizer on the discredited arm and
must be confirmed by a full harness run before any capability claim.
