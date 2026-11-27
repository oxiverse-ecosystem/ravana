# WFA unsaturating articulation — measured scratch result and default-off implementation

**Branch:** `invest/wfa-unsat-articulation`; worktree
`C:/Users/Likhith/Documents/projects/ravana_wfa_unsat`, based on
`origin/main` @ `6174f0a`.
**Engine:** never booted; no weight store, mirror or cold reload opened.
The lexical A/B agent owns the engine in this session.
**Production changes:** one new default-off module and one defensive
emission-boundary call. No gate, threshold, item, prompt, expected decision,
parser or scoring function was changed.

Pre-registration (written and committed before any candidate was run):
`docs/investigations/WFA_UNSAT_ARTICULATION_PREREG.md`.

## 1. Why the prior mechanism saturated

A boot-free census (`scratch/wfa_unsat_signal_probe.py`) over staged WFA
fixtures found, for every item and arm: **exactly one** matching provenance
edge, `edge_kind=curated`, `confidence=0.9`, trust class determined by the
experimental condition. The prior T1 evaluator (PR #547) therefore classifies
the condition perfectly by construction: R = 1.00, F undefined, G6-saturated.
The recorded articulation has 0/768 explicit decisions and almost no graded
familiarity signal (137/192 rows have zero claim-word overlap;
`scratch/wfa_unsat_text_signal_probe.py`). Any rule reading only these two
channels is either perfect or blind — hence an explicit noise/accumulation
stage is required for non-saturated R/F.

## 2. Mechanism selected

U1 — **four-sample evidence accumulation + graded metacognitive confidence +
source-memory verdict**:

1. first-order verdict from the production `evaluate_source` on the staged
   `ClaimContext` (source trust from stored `edge_kind`/`source_status`);
2. four samples of the **stored edge confidence** (carried into production as
   `SourceStatus.reliability`), drift `c/4`, noise `N(0,(σ/2)^2)`, `σ = 1.0`;
3. graded confidence `sigmoid(v)`; recollection licenses the controlled verdict
   only at `confidence >= 0.5`, otherwise the automatic familiarity fallback
   answers (this is the F term of Jacoby's process dissociation);
4. mode comes from the instruction text, never from item metadata: free
   inclusion cooperates, restricted exclusion opposes, invalidation requires a
   confident re-evaluation after a prior endorsement.

**Support vs extrapolation.** Two-choice evidence accumulation to a criterion
is supported (Ratcliff & McKoon 2008, *Neural Comput* 20:873–922,
doi:10.1162/neco.2008.12-06-420; Gold & Shadlen 2007, *Annu Rev Neurosci*
30:535–574, doi:10.1146/annurev.neuro.29.051605.113038). Graded confidence and its overlap with error monitoring are supported
(Yeung & Summerfield 2012,
*Phil Trans R Soc B* 367:1310–1321, doi:10.1098/rstb.2011.0416; Boldt & Yeung
2015, *J Neurosci* 35:3478–3486, doi:10.1523/JNEUROSCI.4114-14.2015).
Source judgments are a distinct, fallible evaluation (Johnson, Hashtroudi &
Lindsay 1993, *Psychol Bull* 114:3–28, doi:10.1037/0033-2909.114.1.3);
inclusion/exclusion cooperation and opposition is Jacoby (1991), *J Mem Lang*
30:513–541, doi:10.1016/0749-596X(91)90025-F; withdrawal after an unreliable
speaker is Dautriche et al. (2021), *Open Mind* 5:1–19,
doi:10.1162/opmi_a_00038. **Extrapolation:** a fixed 4-sample symbolic
accumulator with hash-seeded pseudo-noise and a numeric confidence criterion is
not a neural DDM, and the invalidation cue is experimenter-delivered.

## 3. Measured scratch endpoints (recorded 192-row `pd_audit`, frozen manifest `119b9d91…`)

Production parsers, production `process_dissociation`, production
`pd_quality_counters` and production `decide()` were called unchanged on rows
whose emitted text was replaced by the candidate's token.

| candidate | forced-choice explicit | `pd_unparseable_rate` | R primary / sealed | F | exclusion margin | withdrawal | G6 |
|---|---|---|---|---|---|---|---|
| T0 shipped text parser (baseline) | 0.00 | 1.00 | 0.00 / 0.00 | 0.0 / 0.0 | 0.00 / 0.00 | 0.00 / 0.00 | no decision |
| **U1 selected** | **1.00** | **0.00** | **0.80 / 0.90** | **1.00 / 1.00** | **0.80 / 0.90** | **0.877 / 0.821** | **not saturated** |
| U2 zero noise (diagnostic) | 1.00 | 0.00 | 1.00 / 1.00 | undefined | 1.00 / 1.00 | 1.00 / 1.00 | **SATURATED, as predicted** |
| U4 ungraded-confidence control | 1.00 | 0.00 | 0.40 / 0.47 | 1.00 / 1.00 | 0.40 / 0.47 | 0.492 / 0.418 | no |
| T3 text-only falsifier | 0.00 | 1.00 | null | null | null | null | no decision |

All 16 production PD-related rules (`PD_R_*`, `PD_exclusion_margin_*`,
`PD_exclusion_credible_*`, `PD_wellformed_*`, `invalidation_withdrawal_*`,
`unparseable_ceiling_*`, plus n-gap/n-decoy) are **true** on the reconstructed
summary. **This is not a live WFA PASS**: only the recorded rows' PD endpoints
were replaced, and every non-PD gate in that recorded summary was already
1.00/saturated. No engine was run.

Design accuracy per arm (U1): inclusion 1.000, exclusion 0.953,
invalidation 0.844, forced-choice 0.938. Controls: M4 anti-tautology pass,
M5 empty-graph fail-closed pass, M6 stored-provenance damage drops exclusion
endorsement 1.000 → 0.108, M7 origin-string mutation changes 0/20 decisions,
M8 0/768 echo-free explicit decisions, M9 three repeats identical, M10 analytic
`|withdrawal − Φ(0.9)| <= 0.08` pass, M13 text-only falsifier pass, G6 rule
(`R <= 0.95` and F defined) pass. M11 graded-strength monotonicity: withdrawal
0.5385 / 0.6308 / 0.7692 / 0.8462 / 0.8769 at stored confidence
0.1 / 0.3 / 0.5 / 0.7 / 0.9 (analytic Φ: 0.5398 / 0.6179 / 0.6915 / 0.7580 /
0.8159). The measured rate is systematically above Φ at higher confidences
because the deterministic per-claim hash seeds are a fixed, slightly positively
biased sample; M10's pre-registered 0.08 tolerance absorbs it. This is reported
rather than tuned away.

Determinism: `scratch/wfa_unsat_results.json` is byte-identical under
`PYTHONHASHSEED=0` and `=1`
(SHA-256 `616e544a0bc5e50600800cc4a3c94946ed4be26032963a44f34eefd67a868d9d`).
The probe SHA-256 is recorded in the commit message.

## 4. G6 interpretation — read before citing R = 0.80/0.90

R is **non-saturated but parametric**: it is produced by the pre-registered
`σ = 1.0` noise acting on a perfect stored-provenance signal. The zero-noise
control restores the old R = 1.00/F undefined exactly, so the headroom comes
from the diffusion noise term, not from a demonstrated human-like capacity to
recollect sources. These numbers are **mechanism evidence**, not capability
evidence, and must not be used to relax any gate or to claim that RAVANA
autonomously monitors source provenance.

## 5. Implementation status

- `ravana/src/ravana/chat/wfa_unsat_articulation.py` — pure, default-off
  (`RAVANA_WFA_UNSAT_ARTIC=1`) decision adapter; instruction-mode detection,
  accumulation, graded confidence, audit dict.
- `ravana/src/ravana/chat/engine.py` — one defensive emission-boundary call
  after the existing source monitor; `getattr` + `try/except` so older
  duck-typed fakes and flag-off runs are byte-identical.
- `tests/unit/test_wfa_unsat_articulation.py` — 13 tests: flag equivalence,
  mode detection, fail-closed behaviour, bare-token deferral, graded-strength
  dependency, audit trail, and real `process_turn` wiring on a no-boot fake.

Verification: 67 tests passed in one pytest process (new module plus three
existing source-gate modules); the imported module's `__file__` was confirmed
to point at this worktree. A full `tests/unit` run was not completed (it timed
out with unrelated failures); the first failure
(`test_answer_learning_bridge.py::test_web_direct_answer_learns_from_the_snippet`)
was reproduced on an untouched `origin/main` worktree, so it is pre-existing.

**Live verification is NOT established and was not attempted here.** The WFA
harness stages `ClaimContext` only inside `run_internal_source_condition`
(`scripts/measure_learn5_wfa.py`, `set_source_context_for_turn` at ~2108/2121).
The free-text inclusion/exclusion and forced-choice PD arms do not stage one,
so with the flag on those arms will still emit no decision token. Wiring that
staging is a harness change and was deliberately NOT made here (no
item/prompt/scoring change; the lexical A/B agent owns engine runs). Falsifier
F10 therefore remains open: parent must stage a real context on all four arms
and run a flag-off/flag-on A/B on copied stores before any live number is cited.
If the on-arm `pd_unparseable_rate` stays at 1.00, the hook is below an early
return or unstaged — report that, do not tune the mechanism.

## 6. Reproduction

```bash
cd C:/Users/Likhith/Documents/projects/ravana_wfa_unsat
../ravana/.venv/Scripts/python.exe scratch/wfa_unsat_articulation.py \
  --summary ../ravana/scratch/learn5_wfa_summary.json \
  --out scratch/wfa_unsat_results.json
```

Outputs: `scratch/wfa_unsat_results.json`, `scratch/wfa_unsat_run_h0.log`,
`scratch/wfa_unsat_run_h1.log`, plus the two signal-census probes and the noise
check. `scratch/` is gitignored in this repo; all probe artifacts were added
with `git add -f`.
