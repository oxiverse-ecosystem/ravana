# TRA-K live replay/consolidation A/B: instrument repair, plan, and evidence

Branch `invest/trak-live-retention-ab` · base `origin/main` @ `5f3be87` · isolated worktree
`C:/Users/Likhith/Documents/projects/ravana_trak_live_ab`.
Scratch artifacts: `scratch/trak_live_ab_probe.py`, `scratch/trak_live_ab_audit.json`,
`scratch/trak_replay_consolidation_probe.py`, `scratch/trak_replay_results_worktree.json`.
No production file, gate, threshold, item key or answer key was changed.

## 0. Verdict of this phase

**Default-off replay-journal flag: NOT IMPLEMENTED.** The task's condition was an A/B
that fixes cold-reload retention **without gate changes**. No full live engine was
booted in this phase (parent schedules that), so that condition has not been met.
What is delivered is a validated instrument repair and a runnable, pre-registered
scratch A/B wrapper, plus a reproduced graph-level hypothesis. A flag will be
implemented only if the live A/B below passes; a negative live result is itself the
deliverable and names the next bottleneck.

The engine-free audit is **OK** on the current harness: `scratch/trak_live_ab_audit.json`
(`ok=true`, commit `5f3be87`, 10 fake items / 20 delayed rows). It also proves the
recorded delayed scope is unreadable by the typed reader and that the audit detects
a planted bad option set. A boot-free `--dry-run-live` check passed: per-arm paths,
audit gate and report writing are wired (`scratch/trak_live_ab_dryrun/report_OFF.json`).

## 1. Instrument defect, repaired first

`docs/investigations/TRA_K_REPLAY_CONSOLIDATION_DESIGN.md` (commit `51d7ab4`) marked the
recorded delayed rows `INSTRUMENT INVALID for typed read`. This is now measured, not
asserted:

| check (recorded `scratch/tra_k_harness/tra_k_rows.jsonl`, 240 delayed rows) | result |
|---|---|
| delayed rows whose option set differs from the current harness's channel-derived `typed_queries` | **160 / 240** |
| delayed rows displaying single-word domain vocabulary (`glow, mist, hail, gale`; gold `hail`) | **88 / 240** |
| verdict | **INSTRUMENT INVALID for typed read** |

Example recorded row `tra_k_0020 / near / k=8`: options `glow, mist, hail, gale`, gold
`hail`, engine response `"honestly, that is a bit outside what i know..."`. The stored
typed slots in that item's cold-reload DB carry source-native labels (e.g.
`precedes → earlier-event`), so a 0/16 there is a vocabulary mismatch plus an
articulation failure, not replay evidence.

**Repair in the A/B wrapper (not in production):** `audit_delayed_options()` runs
before any endpoint is read. For each delayed row it requires (a) displayed options
byte-equal to `measure_tra_k.query_for(item, condition)['candidates']`, (b) every option
in the channel-derived global label pool, (c) no option drawn from the oracle-side
domain-word candidates, (d) the correct label inside the displayed set, and (e) the
paraphrase prompt actually displays those options. Failure prints `INSTRUMENT INVALID
— no score` and exits 2. The planted-bad-option control confirms the audit is
discriminating: it flags `options_not_current_typed_queries`, `non_channel_label`,
`domain_word_option`, `gold_not_in_options`, `prompt_does_not_display_options`.

On the current harness bank (400 families, frozen Wikidata channel, 238 channel labels)
the fake sweep's **20/20 delayed rows pass** the audit. Fake-arm endpoints through the
harness's own `summarize()` (10 items, mechanism probe only): near 0.80, far 0.80,
delayed 1.00, delayed_retention 1.00, growth 1.00, foil/negative reject 1.00. The
wrapper never invents a metric name or re-scores the harness's numbers.

Two instrument pitfalls were caught by these controls and fixed in scratch: the
production adapter reads `source_metadata['source']` (not only `edge_source`), so the
overlay writes both; and the delayed-option audit must build its label pool from the
FULL frozen bank because foils are drawn from other families (a focused subset makes
valid rows look invalid).

## 2. Brain mechanism and verified citations

Grade **[S]** strong (causal/meta-analytic), **[M]** moderate (converging/review).
"Support" is what the source shows; "Extrapolation" is what this design assumes.

| mechanism | source (verified link) | grade | support | extrapolation in RAVANA |
|---|---|---|---|---|
| compressed replay of recent experience | Wilson & McNaughton 1994, *Science* 265:676–679, doi:10.1126/science.8036517, PMID 8036517 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/8036517/)) | [S] | same-cell ensemble sequences recur during sleep | symbolic offline pass over typed edges, not a ripple |
| forward/reverse replay at sharp waves | Diba & Buzsáki 2007, *Nat Neurosci* 10:1241–1242, doi:10.1038/nn1961 ([PMC2039924](https://pmc.ncbi.nlm.nih.gov/articles/PMC2039924/)) | [S] | bidirectional place-cell sequences during ripples | one forward pass only; no reverse-order claim |
| replay is causal for consolidation | Girardeau et al. 2009, *Nat Neurosci* 12:1222–1223, doi:10.1038/nn.2384 ([Nature](https://www.nature.com/articles/nn.2384)) | [S] | selective SPW-R suppression impairs spatial memory | wrong-tag arms are computational analogues, not lesions |
| complementary learning systems | McClelland, McNaughton & O'Reilly 1995, *Psychol Rev* 102:419–457, doi:10.1037/0033-295X.102.3.419 ([PsycNet](https://psycnet.apa.org/doiLanding?doi=10.1037%2F0033-295X.102.3.419)); O'Reilly et al. 2014, *Cogn Sci* 38:1229–1248, doi:10.1111/j.1551-6709.2011.01214.x ([Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1551-6709.2011.01214.x)) | [M] | fast hippocampal store reinstates into slow neocortex | `episodic`/write trace → `ConceptGraph` is the analogue; graph is not cortex |
| schemas permit rapid consolidation | Tse et al. 2007, *Science* 316:76–82, doi:10.1126/science.1135935 ([OpenAlex](https://openalex.org/works/W2067765425)) | [S] | prior schema makes one-trial traces rapidly hippocampus-independent | the trusted typed-slot family is the schema; gold keys never define it |
| weak traces replay preferentially | Schapiro et al. 2018, *Nat Commun* 9:3920, doi:10.1038/s41467-018-06213-1 ([Nature](https://www.nature.com/articles/s41467-018-06213-1.pdf)) | [M] | human replay prioritizes weakly learned information | edge confidence is a strength proxy; `eta=0.25` is pre-registered, not fitted |
| distributed practice beats massed | Cepeda et al. 2006, *Psychol Bull* 132:354–380, doi:10.1037/0033-2909.132.3.354 ([PsycNet](https://psycnet.apa.org/doiLanding?doi=10.1037%2F0033-2909.132.3.354)) | [S] | distributed practice improves recall | three save-boundary cycles are not a behavioural retention interval |
| retrieval practice strengthens memory | Roediger & Karpicke 2006, *Psychol Sci* 17:249–255, doi:10.1111/j.1467-9280.2006.01693.x ([SAGE](https://journals.sagepub.com/doi/full/10.1111/j.1467-9280.2006.01693.x)) | [S] | taking a test improves long-term retention | R2 is replay, not a test; no success-gated test arm is shipped |
| sleep supports systems consolidation | Diekelmann & Born 2010, *Nat Rev Neurosci* 11:114–126, doi:10.1038/nrn2762 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/20046194/)) | [M] | SWS supports systems, REM synaptic consolidation | `save()`/cold reload is an event boundary, not a sleep stage |
| reactivation makes traces labile | Nader, Schafe & LeDoux 2000, *Nature* 406:722–726, doi:10.1038/35021052 ([Nature](https://www.nature.com/articles/35021052), [PubMed](https://pubmed.ncbi.nlm.nih.gov/10963596/)) | [S] | retrieval returns a memory to a labile state requiring restabilisation | justifies restoring **earned** provenance only, never forging a tag |
| nonspatial replay in humans | Schuck & Niv 2019, *Science* 364:eaaw5181, doi:10.1126/science.aaw5181 ([Science](https://www.science.org/doi/abs/10.1126/science.aaw5181)) | [M] | sequential replay of nonspatial task states | does not establish replay of `is_a` facts specifically |

**Honest boundary.** No cited source establishes replay of typed semantic `is_a` facts
in a symbolic graph. Evidence supports the *shape* (fast trace → offline reactivation →
durable integration, weak items preferentially); the *content* mapping is extrapolation
to be tested, not a brain fact.

## 3. Mechanism implemented in the scratch wrapper

`ReplayBackend(mtk.RealBackend)` is armed only by `TRAK_REPLAY_ARM`; with it unset the
harness is byte-identical. It:

1. captures a **write-time journal** during teaching: only `(subject, type_label)` pairs
   actually asserted as conversational `is a` turns, with earned provenance
   (`edge_kind=curated`, `source=conversation`, `source_type=user`, `relation_type=is_a`,
   `slot_writer=tra_typed_slot`) and the channel's own CC0 source id as audit data.
   Negated surface subjects are excluded; `type_sets`, `world_model`, domain and gold
   are never read. Anti-tautology control: mutating item gold/type/labels leaves the
   journal fingerprint unchanged (`4f022b92…`, verified in the audit).
2. applies a **metadata-only overlay at the save boundary** to stored traces only.
   Intact trusted edges are skipped; an absent stored edge is counted and never
   fabricated. `md['source']` is written as well as `md['edge_source']` because the
   production adapter reads `source_metadata['source']` — this was caught by the
   engine-free overlay self-test, which initially showed R1 restoring `edge_kind`
   but leaving `source` empty (a non-deployable stamp).
3. arms: `OFF`, `R1` (restamp), `R1_CONF` (confidence-only sham), `R1_WRONG` (wrong
   tag), `R2` (R1 + 3 cycles `c ← c + 0.25(1−c)` on source-credible, non-negated,
   taught-and-stored traces), `R2_WRONG` (wrong tag + the same 3 cycles, the
   cycle-matched sham), and `ORACLE` (gold to the scorer only, NON-DEPLOYABLE).
4. records counters (`edges_scanned`, `restamps_attempted`, `cycles`,
   `cycle_iterations`) so matched work can be asserted rather than guessed.

Engine-free overlay self-test result (all six deployable arms): `OFF` unchanged;
`R1` `curated/conversation/is_a`, confidence unchanged 0.19; `R1_CONF` no provenance,
confidence 1.00; `R1_WRONG` and `R2_WRONG` `unknown/system`; `R2` confidence
0.19 → 0.6583 over 3 cycles; `restamps_attempted=1`, `edges_scanned=1` in every arm;
`cycle_iterations=3` in both R2 arms. `overlay_self_test.ok=true`.

## 4. Reproduced static hypothesis (no engine, read-only SQLite)

Re-running the prior static probe from the sibling static-probe code on the 40 recorded
cold-reload stores reproduced it exactly: 468 trusted typed-slot edges, 5 damaged
traces restorable, 122 taught pairs with no exact stored trace; production scorer
controls pass; answer-key mutation leaves replay state unchanged.

| arm | primary near | primary far | foil reject | negative reject |
|---|---|---|---|---|
| OFF | 0.30 | 0.25 | 0.95 | 0.65 |
| R1 | 0.30 | 0.30 | 0.95 | 0.65 |
| R1_CONF | 0.30 | 0.30 | 0.95 | 0.65 |
| R1_WRONG | 0.30 | 0.25 | 0.95 | 0.65 |
| R2 | **0.45** | **0.45** | 0.95 | 0.65 |
| ORACLE (non-deployable) | 0.95 | 1.00 | 0.95 | 0.65 |

This is a hypothesis about a graph read, **not** evidence that live retention moves:
emission is not exercised, and the recorded delayed option space was invalid.

## 5. Pre-registered live A/B (parent to schedule; not run here)

One process per arm, sequentially, identical bank, seed, channel snapshot and
`TRAK_MAX_PER_POOL`; focused `n=2/pool` first (mechanism probe), `n=20/pool` only after a
clean focused result. Warm up one short cycle before timing; use the harness's own
`summarize()` endpoints and its own C1–C15 / G6 / M-A/M-B controls unchanged.

```bash
cd C:/Users/Likhith/Documents/projects/ravana_trak_live_ab
export PYTHONPATH="$PWD/scripts;$PWD/ravana/src;$PWD/ravana_ml/src;$PWD/scratch"
export TRAK_REPLAY_SOURCE_ROOT=C:/Users/Likhith/Documents/projects/ravana
export PYTHONHASHSEED=0 RAVANA_OFFLINE=1 TRAK_MAX_PER_POOL=2
for ARM in OFF R1 R1_CONF R1_WRONG R2 R2_WRONG; do
  TRAK_REPLAY_ARM=$ARM python scratch/trak_live_ab_probe.py --live
done
```

Each arm writes isolated `scratch/trak_live_ab_data/{rows,summary,report}_<arm>.json`
and its own engine `data_dir`. The wrapper re-audits delayed options after the sweep;
a failure means no score, exit 2. Before the live loop, run
`python scratch/trak_live_ab_probe.py --dry-run-live` (no engine) to verify wiring.

**Endpoints (unchanged):** `summary[pool]['delayed']` / `['delayed_retention']` /
`['growth']` at k=8, near/far, foil/negative reject; continuous mean stored confidence.

**Pass (all required, no gate change):**
- `R2 − OFF ≥ +0.10` on primary delayed near/far **and** `R2 − R2_WRONG ≥ +0.10`;
- `R1 − OFF < 0.05`, `R1_CONF − OFF < 0.05`, `R1_WRONG − OFF < 0.05`;
- foil and negative reject drop ≤ 0.05 in R2;
- instrument controls OK in every arm; sealed pool replicates in the same direction;
- gains survive cold reload (delayed rows are post-reload by construction);
- counters equal: `edges_scanned`, `restamps_attempted` across arms; `cycle_iterations`
  equal for R2 and R2_WRONG; wall-clock ratio R2/R2_WRONG ≤ 1.12 after warmup;
- ORACLE > 0.90 (non-deployable upper bound only).

**Pre-registered falsifiers / stop rules:**
1. `R2 − OFF < +0.05` with restored traces > 0 and controls OK → stop tuning replay.
2. Any sham (R1_CONF / R1_WRONG / R2_WRONG) moves the endpoint → stamp artifact; reject.
3. ORACLE ≤ 0.50 → information deficit in the taught channel; fix teaching, not replay.
4. R2 raises near/far but drops foil/negative reject > 0.05 → rejected as broadened acceptance.
5. Delayed audit fails → `INSTRUMENT INVALID — no score`, no interpretation.
6. Live delayed stays 0 while the static graph read is ≥ 0.40 → articulation gap; next
   work item is the emission path, not storage.
7. Counter or wall-clock parity fails → compute confound; revert, do not relax the bound.
8. Mutating item gold changes the journal fingerprint → answer-key leakage; reject.

## 6. Implementation gate

No production replay-journal flag exists on this branch. If and only if the live A/B
passes every clause in §5, the narrow production change would be a default-OFF
`RAVANA_TRA_REPLAY_JOURNAL=1` wired at the same save boundary with the same
earned-provenance restore and the same fail-closed "no stored trace ⇒ no fabrication"
rule, with no gate, threshold, item key or answer key edited. Until then the correct
status is **not implemented / hypothesis only**.

## 7. Files

- `scratch/trak_live_ab_probe.py` — audit + live A/B wrapper (scratch only; force-added).
- `scratch/trak_live_ab_audit.json` — real engine-free audit output (`ok=true`).
- `scratch/trak_replay_consolidation_probe.py` — static probe already tracked on
  `origin/main` (from `51d7ab4`); re-run here read-only for reproducibility.
- `scratch/trak_replay_results_worktree.json` — reproduced static probe output.
- This document — citations, instrument repair, A/B pre-registration, verdict.
