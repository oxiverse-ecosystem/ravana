# WFA unsaturating articulation — PRE-REGISTRATION (written before any run)

**Branch:** `invest/wfa-unsat-articulation` (isolated worktree from `origin/main` @ `6174f0a`).
**Engine:** NOT booted. No weight store opened. No `ravana/src` edit at pre-registration time.
No change to any gate, threshold, item, prompt, expected decision or scoring routine.
Production modules are imported UNMODIFIED and called with their real signatures.

## 1. Located cause of the prior saturation

`invest/wfa-articulation-decision-scratch` (PR #547) measured T1 (production
`evaluate_source` on staged provenance) at R = 1.00 / F undefined on both pools —
a G6-saturated deterministic decision table. A boot-free signal census
(`scratch/wfa_unsat_signal_probe.py`) on this branch shows why: every staged
fixture has exactly **one** matching provenance edge with `edge_kind=curated`,
`confidence=0.9`, and a trust class fixed by the experimental condition. A rule
that reads this signal is a perfect classifier of the condition by construction.
The recorded articulation contains no usable decision token (0/768 explicit) and
its lexical familiarity signal is near-zero (137/192 rows have zero claim-word
overlap), so neither channel can supply graded variation.

## 2. Brain-grounded mechanism under test (EXTRAPOLATION-labelled)

**Fixed-horizon evidence accumulation + metacognitive confidence gate + source-memory verdict.**

1. **First-order source verdict** — production `context_from_graph_provenance` +
   production `evaluate_source`; trust comes from stored `edge_kind` /
   `source_status`, never from item metadata, familiarity, or numeric strength.
   (Johnson, Hashtroudi & Lindsay 1993, *Psychol Bull* 114:3–28,
   doi:10.1037/0033-2909.114.1.3 — source judgments are a distinct explicit
   evaluation and dissociate from item familiarity.)
2. **Accumulation** — 4 noisy samples of the *stored edge confidence*, signed as
   confidence **in the evaluator's own verdict**; drift per sample `c/4`, noise
   `N(0, (σ/2)^2)`, σ = 1.0 (canonical diffusion unit). Sum `v` is the decision
   variable. This is the accumulation phase of the DDM without RT or a first-passage
   boundary: **SUPPORTED** for two-choice perceptual decisions (Ratcliff & McKoon
   2008, *Neural Computation* 20:873–922, doi:10.1162/neco.2008.12-06-420;
   Gold & Shadlen 2007, *Annu Rev Neurosci* 30:535–574,
   doi:10.1146/annurev.neuro.29.051605.113038), **EXTRAPOLATION** to 4 symbolic
   samples with no RT.
3. **Graded metacognitive report** — `confidence = sigmoid(v) ∈ (0,1)`; a distinct
   second-order readout, not the source verdict. Confidence at or above 0.5
   licenses the controlled verdict; below it the automatic familiarity fallback
   answers instead. (Yeung & Summerfield 2012, *Phil Trans R Soc B* 367:1310–1321,
   doi:10.1098/rstb.2011.0416 — confidence is graded and shares post-decisional
   machinery with error monitoring; Boldt & Yeung 2015, *J Neurosci* 35:3478–3486,
   doi:10.1523/JNEUROSCI.4114-14.2015 — graded Pe modulation by confidence.)
4. **Process dissociation at the response surface** — inclusion lets recollection
   and familiarity cooperate (endorse from stored familiarity); exclusion makes
   them oppose (a discredited claim is endorsed only when recollection fails);
   invalidation requires a confident re-evaluation after a prior endorsement.
   (Jacoby 1991, *J Mem Lang* 30:513–541, doi:10.1016/0749-596X(91)90025-F;
   Dautriche et al. 2021, *Open Mind* 5:1–19, doi:10.1162/opmi_a_00038 —
   the invalidation cue is experimenter-delivered, recorded as extrapolation.)

Analytic prediction (pre-registered): with `c = 0.9`, `σ = 1.0`,
`P(recollection succeeds) = Φ(0.9) = 0.81594`. Hence
`I = 1.00`, `E ≈ 0.184`, `R ≈ 0.816`, `F ≈ 1.00`,
`exclusion_margin ≈ 0.816`, `withdrawal ≈ 0.816`, `E_credible = 1.00`.
If the measurement disagrees beyond sampling noise, the hook is not reading the
stored confidence or the noise is not as specified.

Noise is pseudo-random but **deterministic per claim+mode** (SHA-256-derived seed,
salt `wfa-unsat-v1`), so runs are reproducible and no item id/metadata/key is read.
Non-saturation is therefore **parametric**: it comes from the pre-registered
diffusion noise, not from a measured human-like memory capacity. This is stated
as a caveat, never as capability evidence.

## 3. Pre-registered constants (fixed before first run)

| constant | value | status |
|---|---|---|
| `DDM_STEPS` | 4 | new mechanism constant, not a harness threshold |
| `SIGMA` | 1.0 | canonical diffusion noise unit |
| `CONF_CRITERION` | 0.5 | chance criterion for a graded confidence report |
| `SEED_SALT` | `wfa-unsat-v1` | deterministic per-claim noise seed |

No existing gate, threshold, item, prompt, expected decision or scoring function
is changed. `w.TH`, `w.decide`, `w.process_dissociation`, `w.pd_quality_counters`,
`w.parse_decision`, `w.parse_forced_choice` are called unchanged.

## 4. Pre-registered metrics (all computed on both pools unless stated)

- **M1 articulation.** `forced_choice_explicit_rate >= 0.90` and
  production `pd_unparseable_rate <= 0.50` after replacing recorded text with the
  candidate's emitted token and re-running the production counters/parsers.
- **M2 process dissociation** via unmodified `w.process_dissociation`:
  `0.25 <= R <= 0.95` (upper bound is a *scratch selection criterion*, not a
  production gate), `F is not None`, `exclusion_margin >= 0.10`,
  `E_credible >= 0.70`, `withdrawal_rate >= 0.50`. Production `PD_wellformed`
  (`R <= 0.999`) is also read from unmodified `w.decide`.
- **M3 design accuracy** `>= 0.75` in each arm (stochastic errors are expected;
  a perfect score would indicate saturation).
- **M4 anti-tautology.** Mutating item metadata (`expected_answer`,
  `source_status`, `source_id`, `reliability`) leaves decision and provenance
  fingerprint unchanged.
- **M5 no shortcut.** Empty graph with familiarity 1.0 fails closed to
  ABSTAIN / N / maintain in every mode.
- **M6 stored-provenance mutation.** Aggregate over primary gap items: changing
  stored `edge_kind` from `curated` to `co_occurrence` drops exclusion endorsement
  by at least 0.50.
- **M7 origin-string mutation.** Rewriting stored `source`/`url` to a
  discredited-looking string (keeping kind/status) leaves every decision unchanged.
- **M8 echo hygiene.** Production parsers find 0 explicit decisions in the 768
  recorded raw texts.
- **M9 determinism.** 3 repeats byte-identical; whole result file identical at
  `PYTHONHASHSEED=0` and `=1`.
- **M10 analytic check.** `|withdrawal_rate − Φ(0.9)| <= 0.08` on both pools.
- **M11 graded-strength monotonicity.** Over stored confidences
  `{0.1,0.3,0.5,0.7,0.9}`, withdrawal rate is strictly increasing and the
  0.9−0.1 endpoint delta is `>= 0.20`.
- **M12 zero-noise saturation control (diagnostic).** σ=0 must give R=1.00 and
  F undefined. If it does not, the analytic account is wrong; the candidate is
  rejected.
- **M13 text-only falsifier.** A keyword accumulator over the recorded text with
  no graph access must have explicit rate `<= 0.10` or design accuracy `<= 0.30`.
- **G6 rule.** Any candidate with `R > 0.95` or `F is None` is marked G6-SATURATED
  and is ineligible, even if the production thresholds pass.

**Selection:** eligible = M1–M11, M13 pass, M12 as predicted, not G6-saturated.
Prefer the candidate with fewest novel primitives. If nothing is eligible the
deliverable is a negative result and NO production code is touched.
If a candidate is eligible, implementation — if any — is behind a default-OFF
flag, with no gate/threshold/item/scoring change, and is reported as unverified
live because this task owns no engine boot.

## 5. Falsifiers (pre-registered)

| id | observation that kills the mechanism |
|---|---|
| F1 | Recorded text supplies explicit decisions after echo stripping (T0 explicit > 0): articulation defect, no new mechanism needed. |
| F2 | Mutating item metadata moves the decision or fingerprint: answer-key leak. |
| F3 | Damaging stored `edge_kind` leaves endorsement unchanged: endpoint not reading the engine's store. |
| F4 | Empty graph + familiarity 1.0 endorses: no-shortcut property broken. |
| F5 | Text-only accumulator recovers the decision (M13 fails): stored provenance unnecessary. |
| F6 | Observed withdrawal deviates from Φ(0.9) beyond M10: stored confidence is not the drift source. |
| F7 | Withdrawal rate does not rise monotonically with stored confidence (M11 fails): the mechanism is not graded. |
| F8 | σ=0 does **not** saturate: an undocumented stochastic source exists; the analytic model is wrong. |
| F9 | R > 0.95 or F undefined with σ>0: G6-SATURATED, do not implement. |
| F10 | Live flag-off/flag-on A/B (parent, later) leaves `pd_unparseable_rate` at 1.00: hook sits below an early return; do not cite scratch numbers as capability. |
