# WFA free-text / forced-choice articulation — boot-free candidate decision routines

**Branch:** `invest/wfa-articulation-decision-scratch` (new worktree from `main` @ `ac4ec88`).
**Status:** Measured result, scratch only. No `CognitiveChatEngine` boot, no weight store
opened, no edit under `ravana/src`, no change to any gate, threshold, item, prompt,
expected decision or grading rule. The only committed artifacts are this document and
the boot-free probe (force-added under `scratch/`, which is gitignored by convention).
Every production function used here is imported UNMODIFIED from this worktree.

**Input:** `scratch/learn5_wfa_summary.json` from the recorded failing run
(manifest `119b9d91fe43a06087d001b1fb02a554edd4d08d665b99041528d93533818164`,
`pd_audit` = 192 rows carrying the engine's raw inclusion / exclusion / invalidation /
forced-choice text). Baseline: `pd_unparseable_rate = 1.00`,
`forced_choice_explicit_rate = 0.00`, `R = 0.0` on both pools.

---

## 1. Research question and mechanism map

The g05 claim-ID anchor fixes **provenance retention** and `source_evaluation.py`
supplies a rule table for the **staged internal-source arm**. Neither puts a decision
at the **articulation surface** used by the free-text PD and forced-choice arms: with
echoes stripped and a single-token response surface, the engine emitted **zero**
explicit Y/N/W tokens (`WFA_ECHO_IMMUNE_RESULT.md`). The question tested here is what
kind of boot-free decision routine could occupy that surface, and whether any
recovers a decision from the recorded rows at all.

| Brain mechanism | Verified source | Engine primitive used here | Grade |
|---|---|---|---|
| Evidence accumulates to a boundary; commitment occurs at the bound; drift rate and boundary separation explain accuracy and RT | Ratcliff & McKoon 2008; Ratcliff, Van Zandt & McKoon 1999 | T2: discrete provenance increments, `DDM_BOUND = 1.0`, first-boundary commit | **SUPPORTED** for perceptual two-choice; **EXTRAPOLATION** for symbolic increments with no noise or RT |
| Deliberation and commitment are common elements of decisions; accumulation-to-bound has a neural implementation | Gold & Shadlen 2007 | boundary-crossing rule before token emission | **SUPPORTED** at the neural level for simple perceptual decisions; **EXTRAPOLATION** to a language engine |
| Confidence is a second-order evaluation that tracks objective performance and prevents overcommitment on unreliable evidence | Yeung & Summerfield 2012 | T2 `DDM_CONF_FLOOR = 0.5` metacognitive veto | **SUPPORTED** as a behavioural dissociation; **EXTRAPOLATION** as a numeric margin in a symbolic gate |
| Right rostrolateral PFC activity tracks metacognitive confidence in perceptual decisions | Fleming, Dolan & Frith 2012 (J Neurosci) | a distinct post-accumulation confidence step | **SUPPORTED** as fMRI correlation; **EXTRAPOLATION** to a symbolic module (no homology claimed) |
| Source judgments are explicit and dissociate from item familiarity (source amnesia) | Johnson, Hashtroudi & Lindsay 1993 | production `evaluate_source` ignores `familiarity`/`reliability`; trust read from `edge_kind`/`source_status` | **SUPPORTED** as the direct analogue |
| Left anterior PFC is engaged in monitoring/evaluating specific source features at retrieval | Ranganath, Johnson & D'Esposito 2000 | staging at articulation, after generation | **SUPPORTED** as activation evidence; **EXTRAPOLATION** to a rule table |
| Toddlers reevaluate a just-learned mapping when the speaker is shown unreliable | Dautriche et al. 2021 | `apply_source_invalidation` + prior ENDORSE -> WITHDRAW | **SUPPORTED** behaviourally in 30-month-olds; **EXTRAPOLATION**: verbal `W` token and an experimenter-delivered cue |
| Inclusion lets recollection and familiarity cooperate; exclusion makes them oppose | Jacoby 1991 | `process_dissociation` on candidate-owned decisions (production scorer) | **SUPPORTED** as the process model; I/E/R/F estimated here are **not** human latent processes |

### Verified citations (publisher / PubMed checked this session)

1. Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: theory and data
   for two-choice decision tasks. *Neural Computation*, 20(4), 873–922.
   doi:10.1162/neco.2008.12-06-420; **PMID 18085991** (PubMed record verified).
2. Ratcliff, R., Van Zandt, T., & McKoon, G. (1999). Connectionist and diffusion
   models of reaction time. *Psychological Review*, 106(2), 261–300.
   doi:10.1037/0033-295X.106.2.261; **PMID 10378014** (verified).
3. Gold, J. I., & Shadlen, M. N. (2007). The neural basis of decision making.
   *Annual Review of Neuroscience*, 30, 535–574.
   doi:10.1146/annurev.neuro.29.051605.113038; **PMID 17600525** (verified).
4. Yeung, N., & Summerfield, C. (2012). Metacognition in human decision-making:
   confidence and error monitoring. *Philosophical Transactions of the Royal
   Society B*, 367(1594), 1310–1321. doi:10.1098/rstb.2011.0416;
   **PMID 22492749** (verified).
5. Fleming, S. M., Dolan, R. J., & Frith, C. D. (2012). Prefrontal contributions
   to metacognition in perceptual decision making. *Journal of Neuroscience*,
   32(18), 6117–6125. doi:10.1523/JNEUROSCI.6489-11.2012; **PMID 22553018** (verified).
6. Fleming, S. M., Dolan, R. J., & Frith, C. D. (2012). Metacognition:
   computation, biology and function. *Phil Trans R Soc B*, 367(1594), 1280–1286.
   doi:10.1098/rstb.2012.0021; **PMID 22492746** (verified).
7. Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring.
   *Psychological Bulletin*, 114(1), 3–28. doi:10.1037/0033-2909.114.1.3
   (verified in `WFA_INTERNAL_SOURCE_DESIGN.md` §3).
8. Ranganath, C., Johnson, M. K., & D'Esposito, M. (2000). Left anterior
   prefrontal activation increases with demands to recall specific perceptual
   information. *J Neurosci*, 20(22), RC108. doi:10.1523/JNEUROSCI.20-22-j0005.2000;
   **PMID 11069977** (verified).
9. Dautriche, I., Goupil, L., Smith, K., & Rabagliati, H. (2021). Knowing how you
   know: toddlers reevaluate words learned from an unreliable speaker.
   *Open Mind*, 5, 1–19. doi:10.1162/opmi_a_00038; **PMID 34485794** (verified).
10. Jacoby, L. L. (1991). A process dissociation framework. *Journal of Memory
    and Language*, 30(5), 513–541. doi:10.1016/0749-596X(91)90025-F;
    Tulving (1985) doi:10.1037/h0080017; Hasher, Goldstein & Toppino (1977)
    doi:10.1016/S0022-5371(77)80012-1 — verified in existing repo docs.

**Support vs extrapolation boundary.** The literature supports *that* an explicit
evaluation, an accumulation-to-criterion, and a confidence readout are separable
processes; it does **not** support identifying a Python rule table with human
source recollection. No claim of autonomous source monitoring is made here.

---

## 2. Candidates (all boot-free)

Every candidate receives only `(claim, arm, raw_engine_text, graph, meta)`. It never
receives `kind`, `source_status`, `expected_answer`, `source_id` or an expected token.
`meta` is nevertheless passed so the anti-tautology control can mutate it. The
experimental condition is staged into a real in-memory `ConceptGraph` by the
fixture builder using production `_write_asserted_wfa_fact` /
`apply_source_invalidation`; the candidate reads only stored provenance or text.

| id | routine | production functions used | novel machinery |
|---|---|---|---|
| **T0** | shipped scoring path replayed on recorded text | `parse_decision`, `parse_forced_choice`, `strip_echoes` | none |
| **T1** | staged context + evaluator + token | `context_from_graph_provenance`, `evaluate_source`, `SourceDecision.token` | none |
| **T2** | DDM-style accumulation to bound + confidence veto | `graph_provenance_for_claim`, `context_from_graph_provenance`, `evaluate_source` (final commit) | 2 constants + integer increments + veto |
| **T3** | hybrid verify: genuine text token first, else T1 | T0 + T1 | one fallback branch |
| **T4** | text-only accumulator (**falsifier**, not deployable) | `strip_echoes` | keyword increments |

Pre-registered constants (fixed before the first run): `DDM_BOUND = 1.0`,
`DDM_CONF_FLOOR = 0.5`. No existing threshold was changed.

---

## 3. Pre-registered metrics and measured results

Metrics were written into the probe docstring **before** running any candidate.
M2 uses the **unmodified** `w.process_dissociation()` on records carrying each
candidate's decisions; the harness's own bounds are reported, never reset.

| candidate | forced-choice explicit | design accuracy (all 4 arms) | R primary / sealed | exclusion margin | withdrawal | controls M4–M9 |
|---|---|---|---|---|---|---|
| T0 production text parser | **0.00** | 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | M6 fails (no decision to move) |
| **T1 production evaluator** | **1.00** | **1.00 / 1.00 / 1.00 / 1.00** | **1.00 / 1.00** | **1.00 / 1.00** | **1.00 / 1.00** | **all pass** |
| T2 accumulator + confidence | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | all pass |
| T3 hybrid verify | 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | all pass |
| T4 text-only falsifier | 0.00 | 0.16 / 0.07 / 0.00 / 0.00 | 0.17 / 0.16 | 0.17 / 0.03 | 0.00 / 0.00 | M6 fails |

Rows per arm: inclusion 192, exclusion 192, invalidation 132, forced-choice 192.
`T0` exactly reproduces the recorded failure (`pd_unparseable_rate = 1.00`,
`forced_choice_explicit_rate = 0.00`, `R = 0`), which validates the replay instrument.
`T4` shows that no accumulator over the engine's **text** can recover a decision:
the decision variable is not in the articulation, it is in stored provenance.

**Selection (pre-registered rule):** eligible = M1 ≥ 0.90, M2 R ≥ 0.25 / margin ≥ 0.10 /
withdrawal ≥ 0.50 on both pools, M3 ≥ 0.90, M4–M9 pass; then maximise
(forced-choice explicit, min R, min design accuracy), tie-break by fewer novel
primitives. **Winner: T1_production_evaluator** (score 1.00/1.00/1.00; eligible T1, T2,
T3). T1 uses production functions end-to-end and adds no new primitive.

### Controls (all on the recorded rows; pass = required)

- **M4 anti-tautology:** mutating `expected_answer`, `source_status`, `source_id`,
  `reliability` leaves decision **and** staged fingerprint unchanged — true for T1/T2/T3.
- **M5 no shortcut:** empty graph + `familiarity = 1.0` -> `ABSTAIN` / `R_UNKNOWN_PROVENANCE`.
- **M6 stored-provenance mutation:** `curated -> co_occurrence` flips `Y -> N` for
  T1/T2/T3. The graded endpoint reads the engine's store, not item metadata.
- **M7 origin-string mutation:** renaming the origin to a discredited-looking string
  (keeping the census query key consistent so only the string's semantics change)
  leaves the decision at `Y`. Trust comes from stored `edge_kind`/`source_status`, not
  from a URL containing "discredited".
- **M8 echo hygiene:** production parsers find **0** explicit decisions in 768 raw texts.
- **M9 determinism:** 3 repeats byte-identical; full result file identical under
  `PYTHONHASHSEED=0` and `=1`.

Two retracted probe defects (fixed before the reported run, recorded for audit):
(i) an early M7 left the census `source_id` stale, so the mutation removed the edge
and looked like a decision change — it measured the filter, not trust; (ii) T2's
no-boundary branch returned `None` instead of a fail-closed `N`, which made M6 look
like a failure. Neither was a finding about the system.

---

## 4. G6 saturation — read this before citing any number above

**T1/T2/T3 produce R = 1.00, exclusion margin = 1.00, withdrawal = 1.00 on both
pools.** That is the "deterministic decision table wired into a harness measures the
table, not the system" failure mode from `ravana-debugging` §1. Clean stored provenance
fed into a production rule table yields perfect agreement **by construction**: the
harness taught the claim, chose the origin, and staged the invalidation. Consequently:

- `F = E/(1-R)` is undefined at `R = 1`; the production `PD_wellformed` gate
  (`R <= 0.999`) still **FAILS**. No threshold was relaxed to manufacture a mid-range R.
- The reported 1.00 is **G6 FLAG: mechanism implementation**, not evidence that RAVANA
  autonomously recollects sources or performs human-like evidence accumulation.
- T2's DDM-style accumulator adds no information over T1 on this fixture: with one
  clean edge the bound is crossed on the first increment in both arms. The confidence
  veto never fires. This is a **negative result for the extra machinery** — do not
  ship T2 on the strength of these rows.
- The brain literature grades T2's accumulation and veto as extrapolations; the only
  directly supported component in the selected candidate is the explicit
  source-evaluation step (T1).

---

## 5. Falsifiers (pre-registered, measurable)

| # | Falsifier | Prediction if the candidate is real | Observed / live action |
|---|---|---|---|
| F1 | A real-engine source question with **no staged context** still yields ~0 explicit decisions | absent articulation routine; scratch 1.00 is rule-table saturation | not run here (would boot the engine); parent must run E5 before citing |
| F2 | Mutating item metadata moves the decision or fingerprint | context leaked from the answer key | scratch: **did not occur** (M4 true) |
| F3 | Damaging stored `edge_kind` leaves ENDORSE unchanged | endpoint is not reading the engine's store | scratch: **did not occur** (M6 true) |
| F4 | `familiarity = 1.0` with no provenance endorses | no-shortcut property broken | scratch: **did not occur** (M5 true) |
| F5 | T4 (text-only accumulator) yields high explicit rate **and** high accuracy | decision is recoverable from articulation; stored provenance unnecessary | scratch: 0.00 forced-choice / 0.07 exclusion accuracy — **not falsified** |
| F6 | Live A/B (flag off vs on) leaves `pd_unparseable_rate` at 1.00 | routine never reached articulation (placement below an early return) | parent: compare `wfa_artic_ab_off/on.log` |
| F7 | Live `R` becomes 1.00 | G6 FLAG; keep `PD_wellformed` failing; **do not tune** | scratch predicts this; expected |
| F8 | Sealed discredited items gg039/gg032/g64/g62 stay at withdrawal 0 with `prior_decision` from the engine audit | H_PEB (prior-endorsement binding) false; re-open provenance/trust | parent's E2/E3 run; not tested here |

---

## 6. Exact live test for the parent (NOT run here)

The scratch result is a mechanism candidate only. To test it live, wire T1 at the
**emission boundary** so the engine's own emitted text must carry the token produced
by `evaluate_source` on the staged `ClaimContext`, behind a **default-off flag
`RAVANA_WFA_ARTIC`** — that wiring is a production edit and belongs to the parent, not
to this branch. The flag does not exist on this branch; running the command below
before wiring will reproduce the current 1.00/0.00 failure, which is itself the
off-arm baseline.

```bash
cd C:/Users/Likhith/Documents/projects/ravana
# 0. seal check — boot-free
RAVANA_OFFLINE=1 PYTHONHASHSEED=0 uv run python scripts/measure_learn5_wfa.py --validate-only

# A/B on identical fixtures and seed; do NOT run while the E2 live sweep owns the store
RAVANA_OFFLINE=1 PYTHONHASHSEED=0 RAVANA_SOURCE_EVAL=1 \
WFA_FRESH_ENGINE=1 RAVANA_WFA_ARTIC=0 \
uv run python scripts/measure_learn5_wfa.py > scratch/wfa_artic_ab_off.log 2>&1

RAVANA_OFFLINE=1 PYTHONHASHSEED=0 RAVANA_SOURCE_EVAL=1 \
WFA_FRESH_ENGINE=1 RAVANA_WFA_ARTIC=1 \
uv run python scripts/measure_learn5_wfa.py > scratch/wfa_artic_ab_on.log 2>&1

# compare the two summaries: pd_unparseable_rate, forced_choice_explicit_rate,
# R/margin/withdrawal per pool, and the four prior-endorsement item ids.
```

Expected on-arm endpoints from the scratch mechanism: `pd_unparseable_rate -> 0`,
`forced_choice_explicit_rate -> 1`, `R = 1.00` (**G6 FLAG**, `PD_wellformed` still
fails). Off-arm must remain at the recorded baseline. If both arms move, the wiring
is leaking the answer; if neither moves, the hook is below an early return.

Equivalent no-staged-context falsifier (F1/E5): run the on-arm command with the
staging call withheld; ~0 explicit decisions is expected and confirms G6.

---

## 7. What this does not establish

- No engine, store, mirror or cold reload was exercised; no new headline score.
- The scratch fixtures stage the claim from a condition-derived source id; the live
  engine must already hold equivalent provenance for T1 to fire. If it does not,
  the live arm fails honestly (F6/F8).
- I/E/R/F from the production scorer are algebraically correct decompositions of
  candidate-owned tokens, not measurements of human recollection/familiarity.
- The invalidation cue is experimenter-supplied; the `W` token is a behavioural proxy.
- A near-1.00 scratch result is **not** capability evidence (§4).

## 8. Reproduction

```bash
cd C:/Users/Likhith/Documents/projects/ravana_wfa_artic
C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/wfa_articulation_decision.py \
  --summary ../ravana/scratch/learn5_wfa_summary.json \
  --out scratch/wfa_articulation_results.json
```
Outputs: `scratch/wfa_articulation_results.json` (full per-candidate rows and
controls), `scratch/wfa_articulation_run.log`. The results file is byte-identical
under `PYTHONHASHSEED=0` and `1`. Probe SHA-256 is recorded in the commit message;
the measurement payload hash over `{candidates, recorded_baseline, preregistered}` is
`944ec6eddd24372d2ad2a759237bbe6785eb1a658a47a5b30c065aa9201056bf`.
