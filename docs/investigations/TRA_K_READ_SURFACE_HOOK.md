# TRA-K — read-admission hook experiment (scratch only, design + boot-free probe)

**Status:** SCRATCH + DESIGN. **No engine was booted by this worktree**; no
`process_turn`, no store write, no network. No production file, harness gate,
threshold, item, prompt, gold or scoring code was edited. All code lives in
`scratch/` and mutates only a single engine *instance* at runtime.
Worktree `C:/Users/Likhith/Documents/projects/ravana_trak_read_surface`,
branch `invest/trak-read-surface-hook`, base `6e32214` (parent's
`docs/trak-stage2-negative` head).

**Where this sits in the failure history:** replay A/B (all arms
`genuine_fail`, shams moved) and stage-2 write-time re-encode (`S2` movement
reproduced by `SHAM`) both failed. This experiment does not repeat either
manipulation: it changes only the **read surface** at emission, with **no
writes and no extra turns**.

---

## 1. The exact read-admission gap

The graded read is a forced choice over **source-native channel type labels**
(e.g. `snow storm`), not domain words. The delayed option audit
(`scratch/trak_live_ab_probe.py:121-173`) confirms 0 bad delayed rows and that
every displayed option is a channel-derived label, so the option space is
**not** the defect. Likewise, the stage-2 boot-free cue probe returned the gold
label on P3 and P4/P6 surfaces for 12/12 pairs given a trusted view, so subject
resolution and parsing in the hook are **not** the defect.

The read path is:

| step | code | property |
|---|---|---|
| emission boundary | `engine.py:8296-8312` | `_typed_slot_candidate` is called on the final response, **only** when `slots_enabled()` (`RAVANA_TRA_SLOTS=1`) |
| subject resolution | `response_gen.py:7702-7726` | parses `Options:`, resolves subject, then delegates |
| census | `schema_induction.py:171-205` | `typed_slot_views_from_graph` reads the **whole** graph; `type_label` is taken from `source_metadata` only |
| admission | `schema_induction.py:208-273` | `slot_keyed_forced_choice` admits an option **only** if a trusted `is_a` edge carries a **non-empty `type_label` stamp equal to the target label**, scores `>= CONGRUENCE_FLOOR` (0.20) and wins by `>= STRICT_MARGIN` (0.10) |

**Gap:** the only retrieval cue the reader recognizes is a **redundant
write-time stamp**. A provenance-trusted `is_a` edge whose *target node's own
label* is the correct source-native type is treated as non-evidence unless a
separate `type_label` copy was written onto that same edge by
`chain_walker.py:1171-1248` (`slot_writer='tra_typed_slot'`). At the channel
arm's first graded read the admitted view is therefore empty (absent stamp, or
stored confidence/margin below the fixed gate), so the hook fails closed, the
engine emits a hedge, `_parse_engine_answer` (`measure_tra_k.py:268-342`)
records `no_answer`, `immediate(k=8)=0`, and `delayed_retention` is `null` **by
construction** (`measure_tra_k.py:803-814`).

This is consistent with, and distinguishes, the prior evidence:

* 468 typed-slot edges survive cold reload → **storage is not the failure**;
* `delayed_option_audit.ok=true`, fake arm reaches 1.00 on the same option
  space → **instrument scope is not the failure**;
* cue probe 12/12 with an admitted view → **the hook/parser is not the failure**;
* replay acts inside `save_reload()`, after `immediate` → **replay cannot move
  the denominator**;
* `S2` movement was reproduced by `SHAM` → the write-time re-assertion was not
  causal; the extra `process_turn` calls were.

**What the A/B decides.** If READ raises `immediate` with SHAM flat, the fact
was present but inadmissible at the first read (the stamp cue was the gap). If
READ and SHAM are both flat, the fact was **absent** from that arm's store at
that moment, and no read hook can help: the next work item is a
subject-specific type-evidence input, not another read mechanism. This is the
pre-registered fork — it is not decided by this design doc.

---

## 2. Brain basis — retrieval cues / encoding specificity

Per the standing design rule, the mechanism is taken from the human-memory
literature, not invented. All six citations were resolved against publisher /
index records in this session.

| source | finding | consequence for this design |
|---|---|---|
| **Tulving & Thomson (1973)**, *Psychological Review* 80(5):352–373, doi:10.1037/h0020071 | encoding specificity: the memory trace, and hence which retrieval cues are effective, are determined by the encoding operations performed on the input | Teaching encodes a proposition with a node labelled by the source-native type. A test cue must address that representation; a redundant write-time stamp is a second, optional encoding that may never have occurred |
| **Morris, Bransford & Franks (1977)**, *Journal of Verbal Learning and Verbal Behavior* 16(5):519–533, doi:10.1016/S0022-5371(77)80016-9 | transfer-appropriate processing: retention is best when operations at test overlap operations at encoding | Read the label that was actually encoded (the graph node label) rather than a parallel stamp the test never sees |
| **Roediger & Karpicke (2006)**, *Psychological Science* 17(3):249–255, doi:10.1111/j.1467-9280.2006.01693.x | testing effect: a retrieval attempt strengthens later retention — but only when the attempt can contact the trace | P4 is a retrieval attempt that currently fails to contact the trace; the fix is to make it contact a stored, provenanced slot, not to amplify a hedge |
| **Yassa & Stark (2011)**, *Trends in Neurosciences* 34(10):515–525, doi:10.1016/j.tins.2011.06.006 | pattern completion: a partial cue reinstates a stored pattern when the cue overlaps the encoded representation | The trusted typed-slot edge plus the displayed source-native label is the partial cue; overlap is the node label itself |
| **Godden & Baddeley (1975)**, *British Journal of Psychology* 66(3):325–331, doi:10.1111/j.2044-8295.1975.tb01468.x | context-dependent memory: recall is better when retrieval context matches encoding context | **Extrapolation flagged**: a physical-context result applied to an answer surface; consistent with, not proof of, the read-surface mechanism |
| **Johnson, Hashtroudi & Lindsay (1993)**, *Psychological Bulletin* 114(1):3–28, doi:10.1037/0033-2909.114.1.3 | source monitoring: item memory and source memory dissociate | Guard, not mechanism: admit only edges with earned provenance held in `source_metadata`; never a blanket stamp, never the answer key |

A machine-readable verification ledger is in
`scratch/trak_read_surface_citations.json`.

---

## 3. Mechanism in RAVANA's own primitives (scratch only)

`scratch/trak_read_surface_experiment.py` imports the **unmodified** harness
(`measure_tra_k`), the **unmodified** live wrapper (`trak_live_ab_probe`) and the
**unmodified** production primitives (`_is_trusted_edge`, `CONGRUENCE_FLOOR`,
`STRICT_MARGIN`, `parse_forced_choice`, `slots_enabled`). It then installs a
hook **on one engine instance** via `eng._typed_slot_candidate = hook`; the
production class and files are untouched.

The hook:

1. Calls the production primitive first. If it already admits a pick, **defer**
   (structural deferral; no option wording is scored by the new path).
2. Parses the forced-choice prompt with the production `parse_forced_choice`
   and resolves the subject with the same head-noun / last-content-word
   fallback as the production hook (works with `subject=None` on early-return
   paths).
3. Calls `read_surface_pick(graph, subject, options, trust=...)` — a **pure,
   read-only** function whose entire input is the live graph, the resolved
   subject and the displayed options. It admits an option only if a
   provenance-trusted `is_a` edge goes from that subject to a **target node
   whose own label is that exact option**; a stored `type_label` stamp, when
   present, must agree (a contradictory stamp is refused). Score is the stored
   edge confidence; the floor and strict margin are the **imported production
   constants**, not new numbers.
4. Returns the bare pick to the emission boundary, where the **unmodified**
   harness grades it with the **unmodified** `_parse_engine_answer`.

Deliberately absent: any write, any `process_turn` call, any item/gold/
`world_model`/`type_sets`/domain-word read, any new threshold, any gate or
scoring change, any production edit.

**Arms and isolation.** `OFF` (no hook installed), `READ` (production trust
predicate), `SHAM` (same hook, same scan and the same number of scorer calls,
but a wrong-tag predicate that admits nothing). Each arm runs in its **own
process** with its own `TRAK_AB_DATA_DIR` and `TRAK_AB_OUT`; one fresh store
copy per arm, no sequential carry-over. Compute parity is structural: no arm
issues an extra turn, and `hook_calls` / `scorer_calls` must be equal between
READ and SHAM.

**Oracle / deployable boundary.** The boot-free self-test's fake graph is an
**oracle-shaped reachability check, non-deployable**. The only deployable
evidence is the live READ arm, whose sole extra information is a provenance-
stored typed fact already in the engine's own graph and a label already in the
displayed option set. No arm reads the item bank's gold target.

---

## 4. Boot-free preflight (already run: PASS)

```
cd C:/Users/Likhith/Documents/projects/ravana_trak_read_surface
TRAK_READ_SOURCE_ROOT=C:/Users/Likhith/Documents/projects/ravana \
PYTHONHASHSEED=0 \
C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/trak_read_surface_experiment.py --self-test
```

Result: `self_test=PASS`, 15/15 checks: no answer-key inputs, node-label cue
admits, production constants reused, wrong-tag SHAM defers, unrelated subject
defers, below-floor defers, contradictory stamp defers, tie defers,
answer-key-invariant, production read flag on, READ hook picks, SHAM defers,
OFF hook absent, matched scorer calls, no extra turns.

---

## 5. Live A/B plan (parent runs; this worktree boots no engine)

Sequential arms, one process each, same seed, same focused subset first:

```bash
cd C:/Users/Likhith/Documents/projects/ravana_trak_read_surface
export PYTHONPATH="$PWD/scripts;$PWD/ravana/src;$PWD/ravana_ml/src;$PWD/scratch"
export TRAK_READ_SOURCE_ROOT=C:/Users/Likhith/Documents/projects/ravana
export TRAK_CHANNEL_JSON="$TRAK_READ_SOURCE_ROOT/scratch/trak_channel_from_substrate.json"
export PYTHONHASHSEED=0 RAVANA_OFFLINE=1 RAVANA_TRA_SLOTS=1 RAVANA_TRA_K_SLOTS=1
PY=C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe

for ARM in OFF OFF2 READ SHAM; do
  export TRAK_READ_ARM="$ARM"
  export TRAK_AB_DATA_DIR="$PWD/scratch/read_store/$ARM"
  export TRAK_AB_OUT="$PWD/scratch/read_data/$ARM"
  mkdir -p "$TRAK_AB_DATA_DIR" "$TRAK_AB_OUT"
  TRAK_MAX_PER_POOL=2 timeout 1800 "$PY" \
    scratch/trak_read_surface_experiment.py --live --arm "$ARM"
done
```

`OFF2` is a second independent OFF baseline on a separate store copy (F8).
Read only `scratch/read_data/*/report.json` and evaluate F1–F11 in order.
**Do not relax a threshold, change a prompt, or regrade a row to rescue a
falsifier.** All instrument controls, the delayed option audit, the item bank
and the scorer stay byte-identical across arms.

Endpoints (unchanged definitions in `measure_tra_k.summarize`):
`immediate` / `delayed` at `k=8`, `delayed_retention = delayed/immediate`,
`near`/`far`/`foil`/`negative`, per pool.

**Pre-registered PASS criterion:** READ `immediate >= 0.50` on **both** primary
and sealed, `retention >= 0.80`, foil/negative rise `<= 0.05`, SHAM `0.000`,
all controls OK, delayed option audit OK, F1/F2/F4/F6/F7/F8/F9/F10 clear.
Anything less is a negative result with the missing input named, never a
threshold retune.

---

## 6. Pre-registered falsifiers (stop rules)

| # | Falsifier | Measurable prediction | If observed |
|---|---|---|---|
| F1 | Boot-free self-test fails | 15/15 checks true | `INSTRUMENT INVALID — no score`; fix probe before any engine |
| F2 | `delayed_option_audit.ok=false`, instrument controls fail, or M-A/M-B stop moving | same as OFF | `INSTRUMENT INVALID — no score`; no capability claim |
| F3 | READ `immediate(k=8)` stays `0.000` on both pools | must be `>= 0.50` both | Fact absent at first read; report negative and name the missing subject-side type-evidence input. **STOP** |
| F4 | SHAM `> 0` on immediate/delayed, or `scorer_calls`/`hook_calls` differ READ vs SHAM | SHAM `0.000`, counters equal | Control leaked through compute; revert |
| F5 | READ raises immediate but `delayed=0` / retention `null` | require retention `>= 0.80` | No retention claim; locate the post-read consolidation gap |
| F6 | Primary moves, sealed does not | both pools move together | Store carry-over/selection, not capability |
| F7 | `api_calls['process_turn']` differs between arms, or wall ratio `> 1.15` | byte-identical turn counts, ratio `<= 1.15` | Compute confound; revert and re-match cycles |
| F8 | OFF and OFF2 reports differ in any graded endpoint | byte-identical | Store isolation failure; invalidate all arm deltas |
| F9 | Any pick label absent from displayed source-native options, or a hook input touched by gold/`world_model`/`type_sets` | 0 such picks/leaks | Oracle leakage; arm is non-deployable, discard |
| F10 | `engine_response_unparsed` worse than OFF by `> 5%` | no regression | Bare pick not being parsed; articulation regression, stop |
| F11 (G6) | READ saturates at `1.00` on both pools | require mutation controls to move | Saturation is mechanism evidence, not capability; flag G6 |

---

## 7. Honest boundaries

* No capability claim is made here. This is a design + boot-free reachability
  probe; the live endpoint is unmeasured.
* The self-test's perfect fake graph is an oracle upper bound. Only a live
  READ gain with a flat SHAM is deployable evidence.
* The encoding-specificity mapping is an analogy between human episodic memory
  and a symbolic graph read path; Godden & Baddeley (1975) is explicitly an
  extrapolation to an answer surface. Nothing here claims the brain has a
  `type_label` stamp or a forced-choice grader.
* If F3 fires, the correct conclusion is **not** "tune the reader harder": it
  is that no admissible typed fact exists at the first read, and the next
  experiment must supply subject-specific type evidence rather than another
  read surface.
