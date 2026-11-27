# M1+M2+M3 combined experiment — pre-registration and exact run queue

**Status: DESIGN / PRE-REGISTRATION ONLY.** No engine was booted, no `process_turn`
was called, no store was opened for write, no network call was made, and no gate,
threshold, item, gold, prompt, scorer, or production file was edited. None of the
commands in §9 have been executed. This document freezes the design *before*
any endpoint is read; it is the single combined successor to the E1–E5 queue in
`UNIFIED_BRAIN_ROOT_CAUSE_MAP.md` §4.

Isolated worktree: `C:/Users/Likhith/Documents/projects/ravana_m123_combined`,
branch `docs/m123-combined-prereg`, base `16c7738e24a3da65136e9117cba640b353b0ccf3`.
Parent repo `C:/Users/Likhith/Documents/projects/ravana` is untouched by this work.

---

## 1. What the experiment tests, in RAVANA's own primitives

From `UNIFIED_BRAIN_ROOT_CAUSE_MAP.md` §3: M1 (durable identity-keyed
provenance + act record read by an engine-owned path), M2 (pre-existing
structured hypothesis space over kinds/relations), and M3 (wired reader /
pre-articulation proposition commitment) are **separate necessary conditions**.
The prior programme tested each family in isolation — replay R1/R2, read-surface
READ, stage-2 re-encode, ostensive H_a, broad-lexicon H_b, lexical superordinate
ON — and every one failed or moved only a sham/control. The combined hypothesis
is that **no single increment can work because the chain is
write → preserve → structure → read, and all four links must be present in the
same run**. This experiment tests that conjunction, not another variant of one
link.

| Label | Mechanism (computational shape) | Where the existing primitives are (read-only; no edits here) | What the arm adds, at runtime, on the engine instance |
|---|---|---|---|
| **M1** | Bind content to origin + the system's own prior decision + a stable subject/claim key; write at encoding/commit, preserve through maintenance, read by a path the engine owns. | `_source_monitor_decision` `engine.py:8114-8169`; `RAVANA_PRIOR_ACT_TRACE` slot `engine.py:8146-8154`; `source_metadata.edge_kind/source/source_type/relation_type/type_label`; typed-slot writer `chain_walker.py:1171-1248`; 468 trusted typed edges survive cold reload (`TRA_K_REPLAY_CONSOLIDATION_DESIGN.md` §0). | Claim/subject-keyed act-record writer (content, `source_id`, prior decision, stable key) written only from earned provenance; engine-owned read at retrieval; metadata-only preservation at the save boundary. **Absent stored trace ⇒ count and decline, never fabricate.** |
| **M2** | A pre-existing, structured hypothesis space makes sparse, subject-side evidence diagnostic. A broad flat lexicon is *not* M2: it produced 0/80 subject-dependent picks and 12/40 foil leaks (`GOAL_UNSEEN_TYPE_TRANSFER.md` §4). | `slot_keyed_forced_choice`, `typed_slot_views_from_graph`, `CONGRUENCE_FLOOR=0.20`, `STRICT_MARGIN=0.10` in `schema_induction.py:124-273`; `induce_schema_hypotheses` (`RSI_LIVE_RULE_INDUCTION.md` §5); external CC0 Wikidata channel (`TRA_K_SPEC.md` inputs 1–3). | Subject-side, **unassertable** schema hypotheses from the independent channel only (min support 2); admit to the decision only on convergence of ≥2 independent trusted traces; no `type_sets`, `world_model`, gold, domain word or query-side answer lookup. |
| **M3** | Wire the reader and commit to `(subject, relation, target)` before form encoding; fail closed when no admissible proposition exists. | Emission boundary `engine.py:8296-8312`; `_typed_slot_candidate` `response_gen.py:7698-7726` (default-OFF `RAVANA_TRA_SLOTS`); read path `schema_induction.py:171-273`; RSI `RealBackend.present()` never issues the query (`measure_learn5_rsi.py:505-529`). | Install the read-surface hook on one engine instance; issue the RSI query turn through `process_turn`; commit the typed proposition before emission; abstain when the admitted view is empty. Production floor/margin and trust predicate are imported unchanged. |

All three are installed as **scratch, instance-level wrappers**, following
`TRA_K_READ_SURFACE_HOOK.md` §3 and `docs/investigations/lexical_ab_artifacts/
lexical_ab_runner.py`. No production class or file is modified. Every arm uses
the *unmodified* harness `run_sweep()`/`summarize()` and the shipped controls.

### Why this order

1. **TRA-K first.** It is the only front with a clean, leak-free input channel
   (`answer_key_leak=0`, `overlap_world_model=0`, `provenance_missing=0` over
   3,200 facts; `TRAK_INTEGRATION.md` §2) and a pinned `k=8` primary endpoint.
   Its last focused run is a genuine, instrument-valid FAIL with
   `retention=0.000`, `growth=+0.000` (`TRA_K_FOCUSED_LIVE.md`), and the
   replay/read/stage-2 A/Bs have already ruled out each single link
   (`TRA_K_REPLAY_LIVE_AB_NEGATIVE.md`, `TRA_K_READ_SURFACE_NEGATIVE.md`,
   `TRA_K_STAGE2_NEGATIVE.md`). If the conjunction cannot move TRA-K, no
   downstream front is a better first test.
2. **RSI/TRA second.** RSI's deficit is a missing query turn plus no written
   trace (`RSI_LIVE_RULE_INDUCTION.md` §1); TRA's is ~0 mutual information
   between available input and gold (`TYPE_LEARNING_BRAIN.md` §0). Both are the
   same chain with a different front-end, so they are the correct replication
   surfaces — and they are where a wiring-only "gain" was previously caught by
   the negation control (`LEXICAL_SUPERORDINATE_LIVE_AB.md` X6).

---

## 2. Brain plausibility — evidence ledger (support vs extrapolation)

Verified during this design with live publisher/index records; inherited DOIs
were checked again where marked **re-verified here**. Grade **[S]** strong
(causal/meta-analytic or direct behavioural), **[M]** moderate, **[W]** weak.
"Support" is what the source shows; "extrapolation" is what this design assumes
beyond it. Nothing here claims the brain runs one module for all three.

| Component | Source (locator, DOI) | Grade | Support | Extrapolation / limit in this experiment |
|---|---|---|---|---|
| **M1 — source memory is not item memory** | Johnson, Hashtroudi & Lindsay (1993), *Psychol Bull* 114(1):3–28, doi:10.1037/0033-2909.114.1.3 (**re-verified here**: PubMed/APA) | [S] | Source attributions dissociate from item familiarity; a known fact need not carry its origin. | A graph edge carrying `source_metadata` is an analogue of source information, not a biological source tag. Justifies **process-dissociation-style controls** (credible/discredited/unknown, withdrawal) rather than "remove the knowledge" ablations. |
| **M1 — autonoetic/internal source** | Tulving (1985), *Can Psychol* 26:1–12, doi:10.1037/h0080017; Mitchell & Johnson (2009), doi:10.1037/a0015849 (inherited, ledger in `UNIFIED_BRAIN_ROOT_CAUSE_MAP.md` §2.1) | [M] | Knowing *that I previously endorsed this* is a distinct memory quality. | The claim-keyed prior decision is a symbolic stand-in for autonoetic recollection. |
| **M1 — reconsolidation** | Nader, Schafe & LeDoux (2000), *Nature* 406:722–726, doi:10.1038/35021052 (**re-verified here**: Nature/PubMed) | [S] | Retrieval returns a consolidated trace to a labile state needing restabilisation. | The save/cold-reload boundary is an event boundary, not sleep; restoring *earned* provenance only is a design caution derived from this, not a neural claim. |
| **M1 — durable one-trial binding needs a schema** | Tse et al. (2007), *Science* 316:76–82, doi:10.1126/science.1135935 (**re-verified here**: Science abstract) | [S] | One-trial traces become persistent only when a prior associative schema exists; causal lesion evidence. | "External ontology = schema" is an analogy. No paper establishes replay of typed semantic `is_a` facts in a symbolic graph. |
| **M2 — sparse induction needs structure** | Xu & Tenenbaum (2007), *Psychol Rev* 114(2):245–272, doi:10.1037/0033-295X.114.2.245 (**re-verified here**: APA/Ovid); Tenenbaum & Griffiths (2001), doi:10.1017/S0140525X01000061 (inherited) | [S]/[M] | Generalisation from one/few examples is rational only over a structured hypothesis space with a prior; count of exemplars alone is not the driver. | The channel's `is_a` hierarchy is the prior; the unassertable-hypothesis rule is an engineering instantiation, not a licensed neural mechanism. |
| **M2 — fast mapping is subject-specific** | Carey & Bartlett (1978), *Papers & Reports on Child Lang Dev* 15:17–29 (**re-verified here**: full PDF/ERIC; no DOI); Dollaghan (1985), doi:10.1044/jshr.2803.454; Medina et al. (2011), doi:10.1073/pnas.1105040108 (inherited) | [S] | "Chromium" was learned as a colour word from the contrastive phrase about *that* word; ~90% of natural exposures are uninformative and a single hypothesis is retained. | Using the external channel to supply subject-side type evidence is legitimate input; feeding the subject's own answer key is not (hard fail). |
| **M3 — commit before form encoding** | Levelt, Roelofs & Meyer (1999), *BBS* 22(1):1–38, doi:10.1017/S0140525X99001776 (**re-verified here**: Cambridge/APA) | [S] | Conceptual preparation → lemma selection → morpho-phonological encoding; selection completes before form encoding. | A forced-choice pick produced by a graph reader is a weak analogue of lemma commitment. No claim about human forced-choice grading. |
| **M3 — production-side conflict monitor** | Nozari, Dell & Schwartz (2011), *Cogn Psychol* 63(1):1–33, doi:10.1016/j.cogpsych.2011.05.001 (**re-verified here**: ERIC/PubMed/CogPsych) | [S] | Error detection correlates with production-side conflict, not comprehension measures; frontal monitoring. | The engine's abstain path is a behavioural analogue of a declined commitment, not an ACC/ERN model. |
| **M3 — cue overlap at retrieval** | Tulving & Thomson (1973), *Psychol Rev* 80(5):352–373, doi:10.1037/h0020071 (**re-verified here**); Morris, Bransford & Franks (1977), *JVLVB* 16(5):519–533, doi:10.1016/S0022-5371(77)80016-9 (**re-verified here**) | [S]/[M] | Encoding specificity; transfer-appropriate processing — a cue is effective only if it overlaps the encoded representation. | Reading the stored target node's own label (not a redundant write-time stamp) is the operationalisation; this is an analogy, not a claim about human stamps. |

**Explicitly not supported by the literature and therefore excluded:** a single
"source module" doing source monitoring, consolidation and type induction;
a binary retrieve-vs-generate switch; raising a coverage threshold as a
conservatism knob; candidate-side flat lexicons as subject typing; token-level
suppression of inadmissible words as a production constraint. These were each
measured as failures in the cited investigation docs and are not re-run here.

---

## 3. Arms — frozen before any run

Six runs per stage. `--arm` values are literal; tag aliases are for paths.

| # | `--arm` | tag | M1 | M2 | M3 | Operations installed on the instance |
|---|---|---|---|---|---|---|
| 0 | `OFF` | `off1` | – | – | – | None. Unmodified harness, unmodified engine. |
| 0b | `OFF2` | `off2` | – | – | – | Second independent OFF baseline on a **different store copy and different process**. F8 requires byte-identical graded endpoints to OFF. |
| 1 | `M1` | `m1` | ✓ | – | – | Claim/subject-keyed act-record write + engine-owned read + provenance-preserving save-boundary pass. |
| 2 | `M1+M2` | `m12` | ✓ | ✓ | – | + independently sourced, subject-side unassertable schema hypotheses (min support 2). |
| 3 | `M1+M2+M3` | `m123` | ✓ | ✓ | ✓ | + wired reader, RSI query turn, and `(subject, relation, target)` commitment before emission. |
| 4 | `SHAM` | `sham` | compute-matched sham | sham | sham | Same loops, hook calls, scorer calls, cycles and `process_turn` counts as `M1+M2+M3`, but wrong-tag provenance (`edge_kind=unknown`, `source=system`), permuted subject, and negated teaching sources. Must admit nothing. |

Nested contrasts only; **no threshold, gate, item, prompt, score, flag default
or gold changes in any arm.** Every arm uses the same frozen bank, channel
snapshot, seed, focused-subset rule and harness code. The only difference is
the wrapper installed for that arm. The `SHAM` arm gets its **own fresh store
copy** — the earlier `R1_CONF → R1_WRONG` carry-over (`TRA_K_STAGE2_FINAL_RESULTS.md`
"Why R1_WRONG's immediate=0.5 is NOT a capability") is the named hazard this
prevents.

### Isolation invariant

- Canonical snapshot: `weights/ravana_weights.pkl`, `weights/ravana_weights.db`
  and any `junk_*` files, copied fresh into a distinct `data_dir` per arm (and
  per item where the harness already does so). SHA-256 of each copied file is
  recorded; all arms must start from identical hashes.
- One arm per process, run sequentially; no shared mutable store, no shared
  in-memory graph. `TRAK_AB_DATA_DIR` / runner `--base` is distinct per arm.
- After each arm's sweep, re-read the store (cold) and record its hash. A harness
  run re-saves from its in-memory copy and can clobber a repair; a post-run hash
  change is reported, not silently accepted (`ravana-harness` §"Repairing
  persisted state").
- No arm reads `type_sets`, `world_model`, gold, domain tag, or the answer key.

---

## 4. Endpoints and exact deltas

Definitions are the harness's own `summarize()` keys; nothing is re-scored.

For pool `P ∈ {primary, sealed}` and endpoint `m`, define:

```
D_off  = OFF2 − OFF
D_1    = M1 − OFF
D_2    = (M1+M2) − M1
D_3    = (M1+M2+M3) − (M1+M2)
D_tot  = (M1+M2+M3) − OFF
D_sham = SHAM − OFF
```

Rates are computed at the registered `k=8` rows and at the harness's own
denominators. Report each rate with its denominator and the focused subset
size; a figure without `n` is not a measurement.

### 4.1 Stage A — TRA-K (focused `TRAK_MAX_PER_POOL=2`; escalate to 20 only on PASS)

| Endpoint (harness key) | PASS criterion on `M1+M2+M3` |
|---|---|
| `delayed` (k=8, post-cold-reload, typed source-native options), near and far | `D_tot ≥ +0.10` on **both** pools |
| `immediate` (k=8) | `D_tot ≥ +0.10` on **both** pools |
| `delayed_retention` | `≥ 0.80` on both pools; non-null |
| `growth` | `≥ +0.15`; coverage loss `≤ 0.05` |
| `foil` / `negative` reject | drop `≤ 0.05` vs OFF on both pools |
| step attribution | each of `D_1`, `D_2`, `D_3` `≥ +0.03` on at least one pool, and none `≤ −0.05` on either; otherwise report **combined-only**, no per-mechanism claim |
| controls | C1–C15 OK; `delayed_option_audit.ok=true`; `subject_independence_ok`; `channel_independence_ok`; `c11_anti_fake_ok`; mutations move when saturated |
| baselines / sham | `D_off = 0` on every graded endpoint (F8); `|D_sham| ≤ 0.05` on every graded endpoint |

Gains must survive cold reload by construction (delayed rows are post-reload).
`OVERALL: PASS` from the harness is necessary but not sufficient: the deltas
above are the registered judgement.

### 4.2 Stage B — RSI (shipped 2 items/pool, same focused path as `lexical_ab_runner.py`)

| Endpoint | PASS criterion |
|---|---|
| structural `near`, `far` target/match | `D_tot ≥ +0.10` on both pools; near/far **byte-differ** at fixed item+k (the query-turn reachability check) |
| query-turn reachability | `process_turn` count on the query `> 0`; explicit `relation=`/`target=`/`verdict=` fields `> 0` |
| `foil`, `negative_reject` | drop `≤ 0.05`; no acceptance of negated teaching |
| `growth_headroom`, monotone/graded | no regression vs OFF |
| `engine_response_unparsed` | not worse than OFF by `> 5%` |
| baselines / sham | `D_off = 0`; `|D_sham| ≤ 0.05`; `process_turn` counts equal |

A near/far delta measured on demo reactions alone is **not** a result: at fixed
item+k the demos are identical, so byte-identical emissions mean the instrument
still did not see the query.

### 4.3 Stage C — TRA (`TRA_MAX_PER_POOL=20`, same arms)

| Endpoint | PASS criterion |
|---|---|
| `transfer`, `near`, `far` | `D_tot ≥ +0.10` on both pools |
| cold-reload `retention` | `≥ 0.80` when the endpoint is defined (null retention ⇒ no retention claim) |
| `foil_reject`, `negative_reject`, `chance`, `counterexample_reject` | drop `≤ 0.05` |
| subject dependence | same-subject vs `zeppelin` picks differ on `>0` graded near/far rows; 0/80 repeated would be a red result |
| `role_coverage` | not worse than OFF |
| baselines / sham | `D_off = 0`; `|D_sham| ≤ 0.05` |

---

## 5. Controls, each with a pre-registered verdict rule

### 5.1 Negation controls (the X6 hazard)

- **TRA-K:** run `channel_negation_surfaces` (`measure_tra_k.py:217-232`) on
  every arm. On `M1+M2+M3`, negated channel teaching must produce **0 new
  admissible picks** and `negative` reject must not drop `> 0.05`. A direct
  probe must show the reader defers when the staged subject comes only from a
  negated turn. The SHAM arm's negated sources are the compute-matched control.
- **RSI/TRA:** negated demonstration ("not <phrase>"), "what is NOT true of X?",
  unrelated subject `zeppelin`, and OOV options must each yield **0 emits/picks**
  in every mechanism arm. Any emit on negated teaching is an X6-class failure:
  reject the mechanism even if the positive endpoint rises.

### 5.2 Source / subject / channel independence

- TRA-K `channel_overlap()` must report `answer_key_leak=0`,
  `oracle_domain_tag_leak=0`, `provenance_missing=0`, `overlap_world_model=0`.
  The global `overlap_type_sets` count (demo keys) is reported, not failed.
- **Gold-mutation anti-tautology:** mutate each item's gold/type/labels on a
  copied bank and assert the arm's staged context, act-record fingerprint and
  decision are unchanged. A change ⇒ answer-key leakage ⇒ withdraw.
- **Source invalidation:** discredit the taught `source_id` and require
  WITHDRAW; unknown source ⇒ ABSTAIN; credible ⇒ ENDORSE/emit. Decisions must
  follow stored provenance, never the item's credibility label.
- **Subject independence:** same options with an unrelated subject (`zeppelin`)
  must yield a different (usually abstaining) decision; identical picks carry
  zero subject information and are scored as a failure.
- RSI/TRA deployable path must never consult `type_sets`, `world_model`,
  `typed_queries` or gold; the independent ontology comes from the external
  CC0 substrate, not the item bank.

### 5.3 Answer-key leakage AST scan (boot-free, before any arm)

Reusing the proven pattern in
`scratch/ontology_type_bridge_experiment.py::ast_anti_leakage()` and
`scratch/lexical_superordinate_experiment.py`:

1. Pin SHA-256 (newline-normalised) of the unmodified harnesses,
   `scripts/front_gates.py`, item banks, frozen channel/substrate, and the
   imported production primitives; worktree and parent must be identical.
2. For **every** arm-specific function (act-record builder/reader, M2
   hypothesis scorer, M3 reader hook, SHAM wrapper, and each `decide`/`pick`),
   `ast.parse(textwrap.dedent(inspect.getsource(fn)))`, skipping only the
   docstring node.
3. Reject any `ast.Name`, `ast.Attribute`, or `ast.Subscript` constant key in
   the banned set: `gold`, `type_sets`, `world_model`, `answer_key`,
   `answer_key_leak` (as a source), `typed_queries`, `counterexample_target`,
   `forbidden_channel_subjects`, `oracle`, `q['gold']`, `item['gold']`.
4. Write `ast_anti_leakage.ok`, the function list, banned-token list, all SHA
   pins, and the runtime gold-mutation fingerprint into the preflight artifact.
   **Any hit or SHA mismatch ⇒ `INSTRUMENT INVALID — no score`, no engine boot.**

### 5.4 G6 saturation

Run the shipped `front_gates.gate_saturation()` per arm on the harness's own
scores. If a front saturates, `M-A/M-B` mutation controls must move the
decision (flip provenance status; permute gold within the displayed option set
and require accuracy to collapse toward chance, historically `≤ 0.40`). A
saturated PASS is **G6 mechanism evidence, not capability evidence**; record
`status`, `detail` and `flagged_fronts` in every report. If mutations do not
move on saturation, withdraw that arm's positive claim.

### 5.5 Compute parity

Assert equal across arms: `process_turn`, engine boots, forced-choice turns,
`hook_calls`, `scorer_calls`, `edges_scanned`, `restamps_attempted`, `cycles`,
`cycle_iterations`, and equal start-state store hashes. After one warm-up,
wall-clock ratio of `M1+M2+M3` (and SHAM) to OFF `≤ 1.12`. Extra work must be
matched by SHAM; a delta that exists only in the arm with more calls is a
compute confound, not a mechanism result.

---

## 6. Stop rules / falsifiers (evaluated in this order; do not regrade to rescue one)

| ID | Falsifier | Action |
|---|---|---|
| F1 | AST scan fails, SHA pin mismatch, or boot-free self-test `< 15/15` equivalent checks | **INSTRUMENT INVALID — no engine boot, no score.** Fix the runner, re-register if any threshold changes. |
| F2 | `delayed_option_audit.ok=false`, any C1–C15 control fails, backend error, or M-A/M-B stops moving on a saturated front | **No score** for that arm; do not compare deltas. |
| F3 | `D_off ≠ 0` on any graded endpoint | Store isolation failure; invalidate all arm deltas in that stage. |
| F4 | `|D_sham| > 0.05` on any graded endpoint or hook/scorer/turn counters differ between `M1+M2+M3` and SHAM | Stamp/compute artifact; **reject mechanism, STOP**. |
| F5 | Negated teaching, unrelated subject or OOV produces an emit/pick in any mechanism arm | X6-class fail-closed violation; **reject mechanism, STOP** even if primary rises. |
| F6 | Gold mutation changes staged context/fingerprint, or AST/overlap audit finds a leak | Non-deployable; withdraw arm. |
| F7 | `D_tot < +0.05` on the primary endpoint with controls OK | **STOP tuning this chain.** Report negative and name the missing link; no full-bank run. |
| F8 | `D_tot` in `[+0.05, +0.10)` | Partial mechanism signal, **not PASS**; focused result only, no escalation. |
| F9 | Primary pool moves but sealed does not (or vice versa), or a sham-only movement appears | Carry-over/selection; re-run with fresh per-arm stores before any claim. |
| F10 | `D_tot ≥ +0.10` but retention `< 0.80`, or foil/negative drop `> 0.05` | Broadened acceptance / no durable gain; reject as capability. |
| F11 | Voice/parse regression (`engine_response_unparsed` worse by `> 5%`) | Articulation regression; stop adoption. |
| F12 | Wall-clock or maintenance counters differ `> 12%` | Compute confound; revert matching, do not relax the bound. |
| F13 | G6 FLAG with mutations not moving | Withdraw saturated front's positive claim. |
| F14 | Focused PASS but full-bank result contradicts | Trust the larger set; do not ship; report both with denominators. |

**Stage sequencing rule:** a preflight/leak/negation/isolation failure at TRA-K
stops the whole experiment. A clean **capability** failure at TRA-K does *not*
cancel Stage B/C — RSI and TRA are different front-ends of the same chain and
the combined design explicitly requires them — but no stage escalates to the
full bank without its own focused PASS.

---

## 7. Implementation contract for the (future) runner — not written here

The pre-registration fixes a CLI so the queue is executable without further
design decisions. The runner will be a docs-following implementation commit,
modelled on `docs/investigations/lexical_ab_artifacts/lexical_ab_runner.py`
(fresh store copy per arm; unmodified `run_sweep()`/`summarize()`; reachability
counters) and `scratch/trak_live_ab_probe.py` (delayed-option audit; overlay
self-test; counters). It must expose:

```
python scratch/m123_preflight.py
  # AST scan, SHA pins, gold-mutation fingerprint, 15-check self-test,
  # delayed-option audit, store hashes, config print. NO engine boot.

python scratch/m123_combined_runner.py
  --stage {trak,rsi,tra}
  --arm {OFF,OFF2,M1,M1+M2,M1+M2+M3,SHAM}
  --tag <off1|off2|m1|m12|m123|sham>
  --base <isolated data root>  --out <report.json>
  [--max-per-pool N]
```

Each invocation: one process, one store copy, prints `CONFIG`/`RESULT`/`DONE`,
writes one JSON report containing harness summary, counters, control statuses,
G6 status, AST result, start/end store hashes and elapsed seconds.

---

## 8. Artifact contract

- Reports: `scratch/m123_out/{trak,rsi,tra}_{off1,off2,m1,m12,m123,sham}.json`
- Stores: `scratch/m123_data/{trak,rsi,tra}_{tag}/` (git-ignored)
- Logs: `scratch/m123_{stage}_{tag}.log`
- Preflight: `scratch/m123_preflight_result.json`
- Per-stage decision memo: `docs/investigations/M123_COMBINED_RESULT.md`
  (written **after** the run, never before; must contain every endpoint with
  denominators, every falsifier verdict, and raw deltas `D_off..D_sham`).

---

## 9. Exact run queue (for the parent; none of this was executed in this task)

Working assumptions: base commit `16c7738`; run worktree
`C:/Users/Likhith/Documents/projects/ravana_m123_run` with the runner from §7
present; `PY=<repo>/.venv/Scripts/python.exe`; all commands run from the run
worktree, sequentially, with no other engine process active.

```bash
# ── 0. Boot-free preflight (must exit 0 and print PREFLIGHT OK) ──────────────
cd C:/Users/Likhith/Documents/projects/ravana_m123_run
export PYTHONHASHSEED=0 RAVANA_OFFLINE=1
unset RAVANA_TRA_SLOTS RAVANA_TRA_K_SLOTS RAVANA_SOURCE_EVAL \
      RAVANA_PRIOR_ACT_TRACE RAVANA_LEXICAL_SUPERORDINATE
export TRAK_CHANNEL_JSON=C:/Users/Likhith/Documents/projects/ravana/scratch/trak_channel_from_substrate.json
"$PY" scratch/m123_preflight.py | tee scratch/m123_preflight.log
# On any failure: STOP (F1). No command below is run.

# ── A. TRA-K focused (6 runs, mechanism probe, n=2/pool) ────────────────────
STAGE=trak; MAX=2
for SPEC in "OFF:off1" "OFF2:off2" "M1:m1" "M1+M2:m12" "M1+M2+M3:m123" "SHAM:sham"; do
  ARM="${SPEC%%:*}"; TAG="${SPEC##*:}"
  export TRAK_AB_DATA_DIR="$PWD/scratch/m123_data/${STAGE}_${TAG}"
  timeout 1800 "$PY" scratch/m123_combined_runner.py \
    --stage "$STAGE" --arm "$ARM" --tag "$TAG" \
    --base "$PWD/scratch/m123_data" \
    --out "$PWD/scratch/m123_out/${STAGE}_${TAG}.json" \
    --max-per-pool "$MAX" 2>&1 | tee "scratch/m123_${STAGE}_${TAG}.log"
done
# Evaluate §4.1 then F1..F14 in order. Only on a focused PASS:
#   repeat the same six commands with MAX=20 (n=20/pool);
#   only on an n=20 PASS, run the full 400-family arm with OFF/OFF2/SHAM.

# ── B. RSI focused (6 runs, shipped 2 items/pool) ──────────────────────────
STAGE=rsi
for SPEC in "OFF:off1" "OFF2:off2" "M1:m1" "M1+M2:m12" "M1+M2+M3:m123" "SHAM:sham"; do
  ARM="${SPEC%%:*}"; TAG="${SPEC##*:}"
  timeout 1800 "$PY" scratch/m123_combined_runner.py \
    --stage "$STAGE" --arm "$ARM" --tag "$TAG" \
    --base "$PWD/scratch/m123_data" \
    --out "$PWD/scratch/m123_out/${STAGE}_${TAG}.json" \
    2>&1 | tee "scratch/m123_${STAGE}_${TAG}.log"
done
# Evaluate §4.2 then F1..F14; check query-turn counters before reading accuracy.

# ── C. TRA focused (6 runs, n=20/pool) ─────────────────────────────────────
STAGE=tra; MAX=20
for SPEC in "OFF:off1" "OFF2:off2" "M1:m1" "M1+M2:m12" "M1+M2+M3:m123" "SHAM:sham"; do
  ARM="${SPEC%%:*}"; TAG="${SPEC##*:}"
  timeout 3600 "$PY" scratch/m123_combined_runner.py \
    --stage "$STAGE" --arm "$ARM" --tag "$TAG" \
    --base "$PWD/scratch/m123_data" \
    --out "$PWD/scratch/m123_out/${STAGE}_${TAG}.json" \
    --max-per-pool "$MAX" 2>&1 | tee "scratch/m123_${STAGE}_${TAG}.log"
done
# Evaluate §4.3 then F1..F14. Write M123_COMBINED_RESULT.md.

# ── Post-run invariant (all stages) ─────────────────────────────────────────
# Re-read every arm's cold store, print hashes, confirm the preflight SHA pins
# of harness/gates/banks are unchanged. A post-run clobber invalidates that arm.
```

Queue length as registered: **1 boot-free preflight + 6 TRA-K + 6 RSI + 6 TRA
= 18 live arm-runs**, plus at most 6 focused-escalation repeats per stage only
on a clean focused PASS. Each arm is one process with its own store; no arm may
be skipped or reordered, and OFF/OFF2 must be read before any mechanism arm's
delta.

---

## 10. Honest boundaries

- This is a design, not a result. No endpoint in §4 has been measured; every
  number there is a **pre-registered threshold**.
- The three components are a *computational analogy* across source monitoring,
  schema-assisted consolidation and production commitment — not a claim that
  the brain has one combined module. M2 and M3 remain separate necessary
  conditions; the arms exist to test whether they are jointly sufficient.
- The self-test's oracle-shaped graph is a non-deployable reachability check.
  Only live gains with `D_off = 0`, flat SHAM, clean negation controls and
  unmodified gates count as deployable evidence.
- If F7 fires on all three stages with clean instruments, the correct
  conclusion is that the chain is broken upstream of the reader and the next
  work item is a new **subject-side input**, not another threshold, prompt or
  accumulation of replay cycles.
- No capability PASS is claimed here for WFA, RSI, TRA, TRA-K or CAR; each keeps
  its standing verdict until a measurement that satisfies §4 and §5 exists.
