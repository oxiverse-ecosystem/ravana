# TRA-K stage 2 — root cause of zero retention and the next brain-grounded mechanism

Status: **SCRATCH ONLY / DESIGN + BOOT-FREE EVIDENCE**. No production file,
harness gate, threshold or prompt was edited. No engine was booted by this
worktree. Base commit `4c81b01` (the live A/B worktree head). Evidence was
read from `ravana_trak_live_ab/scratch/trak_live_ab_all3.log`,
`scratch/trak_live_ab_data/report_*.json`, `summary_*.json` and `rows_*.jsonl`
without writing to that worktree. Branch: `invest/trak-stage2-cue-reinstatement`.

## 1. What the live A/B actually shows (read-only)

| arm | near/far (primary) | immediate (k=8) | delayed (k=8) | delayed_retention | controls OK | interpretation | elapsed |
|---|---|---|---|---|---|---|---|
| OFF | 0.000 / 0.000 | 0.000 | 0.000 | `null` | true | genuine_fail | 474 s |
| R1 (provenance restamp at save) | 0.333 / 0.333 | 0.000 | 0.000 | `null` | true | genuine_fail | 414 s |
| R1_CONF (confidence=1.0 at save) | 0.333 / 0.333 | 0.000 | **0.500 primary / 0.000 sealed** | `null` | true | genuine_fail | 718 s |

Common to all completed arms: 240 rows, `backend=live`, `live=True`,
`instrument_controls_ok=true`, `invalid_controls=[]`,
`delayed_option_audit.ok=true` (12 delayed rows, 0 bad, 238 channel labels),
`engine_observation_missing=0`, `engine_response_unparsed=83–89/240`.
The paired arm audit in `trak_live_ab_audit.json` also passes: planted
domain-word options are detected, journal fingerprint is gold-independent,
production scorer picks trusted and rejects untrusted/disjoint/other-subject.

### Decisive ordering fact

`summarize()` (measure_tra_k.py:805–814) computes retention as
`delayed / immediate` and gates both on `k == 8`. **Immediate is measured
before `save_reload()`; the replay/consolidation overlay is applied inside
`save_reload()`** (trak_live_ab_probe.py `ReplayBackend.save_reload`). Every
replay arm therefore acts strictly *downstream* of the retention denominator.
With `immediate = 0`, `delayed_retention` is `null` **by construction**, no
matter what the overlay does. R1_CONF confirms the mechanism is live but
downstream: it moved primary delayed from 0.000 to 0.500 while leaving
immediate at 0.000 and retention undefined.

### Row-level read-surface evidence

For the same subject, same channel-derived options and the same engine:

* `tra_k_0020`, blizzard, near, k=8: P3 graded query emitted bare token
  `snow storm` on later arms, but the **channel arm's P3 response was a
  hedge** (`"i'm not totally sure about true yet..."`) and P4 immediate was
  `"i don't really have a solid grasp on best..."`; P6 delayed was another
  hedge. `grade()` requires `exact_structural_match` = verdict `match` +
  role + target, and `_parse_engine_answer` only supplies a role from a bare
  option token or an explicit `role=`/relation phrase. A hedge or a sentence
  like `"guild is linked to an association"` is scored `no_answer`.
* Offline cross-tab: in P3, 17–19/72 near+far rows end in a bare gold token
  and are correct; **all of them are k>=1**. In P4/P6, 0/12 and 0/12 do.
  Gold target appears in free text but is not parseable in 1–2 rows; the rest
  are hedges. So the failure is *not* "forgot after a delay" — it is "never
  admitted at the first read".
* Boot-free production-hook probe (`scratch/trak_stage2_cue_probe.py`, run
  with the venv interpreter, no engine boot): with an identical trusted
  typed-slot view, `_typed_slot_candidate` returns the gold label on **both**
  the P3 and P4/P6 surfaces (12/12 pairs, including with `subject=None`).
  Therefore the parser/subject-resolution in the hook is **not** the defect;
  the live graph does not present an admitted view at that moment.
* Read-only SQLite census of the *final* store for `tra_k_0020` (copy of
  `live_data/tra_k_0020/ravana_weights.db`, unpickled `source_metadata`,
  no engine boot): `blizzard -[is_a]-> snow storm` exists at confidence 1.0
  with `edge_kind=curated`, `source=conversation`, `source_type=user`,
  `type_label=snow storm`, `slot_writer=tra_typed_slot`. The correct trace is
  eventually stored, but only after the later non-channel teaching blocks;
  the channel arm's own first read never saw an admitted version of it.

### Instrument confound that must be fixed before any stage-2 claim

`ReplayBackend.base_dir` is `TRAK_AB_DATA_DIR` or `<out>/live_data`, and the
parent loop sets `TRAK_AB_OUT` but **not** `TRAK_AB_DATA_DIR`. All arms in the
current A/B therefore reuse the same per-item stores sequentially. This
violates the A/B isolation rule ("give every arm its own fresh copy of the
weight store"), and plausibly contributes to the near 0.333 vs sealed 0.333
parity, the R1_CONF primary-only delayed gain, and the 414 s -> 718 s elapsed
drift. Stage-2 commands below set `TRAK_AB_DATA_DIR` per arm. Treat the
current arm deltas as hypothesis-generating, not attributable.

## 2. Root-cause verdict on the three candidate explanations

1. **Instrument delayed-option mismatch — RULED OUT for this run.** The old
   defect (domain-word options; recorded audit: 160/240 mismatches, 88
   domain-word rows, `INSTRUMENT INVALID for typed read`) is repaired:
   live `delayed_option_audit` has 0/12 bad rows, gold is inside the displayed
   channel-label options, and the fake arm reaches `delayed=1.0` on the same
   typed option space. A *different* articulation issue remains (free-text
   answers are not parsed as structural matches), but it is downstream of
   `immediate=0` and cannot explain why no arm ever produces a bare pick in
   P4/P6 while P3 does.
2. **Missing replay — RULED OUT as the primary cause of zero retention.**
   Replay is applied after the immediate probe; it cannot change the
   retention denominator. The needed edge is eventually stored without any
   replay, and R1_CONF shows a 0 -> 0.5 delayed shift from confidence alone.
   Replay is a plausible *delayed-only* amplifier, not the cause of
   `immediate=0`.
3. **Wrong read surface / write-time admission — SUPPORTED, with a precise
   location.** The first graded read of the channel arm's own teaching does
   not admit the stored proposition into the fail-closed forced-choice
   answer; the engine emits a hedge, the parser records `no_answer`, and the
   retention metric has a zero denominator. The hook itself is surface-
   invariant when given a trusted view; the missing input is an *admitted
   typed-slot view at write time*. Stage 2 must act during teaching, before
   P4, not at save.

## 3. Next mechanism: write-time, source-backed slot re-encoding

**Claim.** Re-encode the already-taught, source-backed proposition into the
reader's own typed-slot format at *write time*, and re-assert no more than
twice until the unmodified fail-closed scorer admits it. This is retrieval
practice that repairs the encoding cue rather than a post-hoc confidence
bump or a threshold relaxation.

**Brain grounding (verified citations).**

* Encoding specificity: what is stored is determined by the encoding
  operations, and those operations determine which retrieval cues are
  effective (Tulving & Thomson 1973, *Psychological Review* 80(5):352–373,
  doi:10.1037/h0020071). RAVANA teaches a relation in prose but tests a
  typed forced choice; a trace that is present but not encoded in the
  reader's slot format is inaccessible to that cue.
* Transfer-appropriate processing: memory is best when operations at test
  overlap those at encoding (Morris, Bransford & Franks 1977, *Journal of
  Verbal Learning and Verbal Behavior* 16(5):519–533,
  doi:10.1016/S0022-5371(77)80016-9). Hence re-encode into the slot surface
  the test actually reads, rather than merely raising weight on the prose
  trace.
* Testing effect: retrieval attempts strengthen later retention, but only
  when the attempt can contact the trace (Roediger & Karpicke 2006,
  *Psychological Science* 17(3):249–255,
  doi:10.1111/j.1467-9280.2006.01693.x). Our P4 probe is a retrieval attempt
  that fails; the fix is to make it contact a source-backed slot, not to
  amplify an ungrounded answer.
* Pattern completion: a partial cue reinstates a stored pattern when the
  cue overlaps the encoded representation (Marr 1971, as reviewed in Yassa
  & Stark 2011, *Trends in Neurosciences* 34(10):515–525,
  doi:10.1016/j.tins.2011.06.006). The trusted slot view is the partial cue.
* Context/cue reinstatement is also consistent with context-dependent memory
  (Godden & Baddeley 1975, *British Journal of Psychology* 66(3):325–331,
  doi:10.1111/j.2044-8295.1975.tb01468.x); applying a physical-context result
  to a linguistic answer surface is an **extrapolation**, flagged as such.
* Source monitoring remains the guard: re-encoding must restore provenance
  earned at write time from the item's own accepted channel fact, never a
  blanket stamp (Johnson, Hashtroudi & Lindsay 1993, *Psychological
  Bulletin* 114(1):3–28, doi:10.1037/0033-2909.114.1.3).

**Mechanism in RAVANA's own primitives** (all scratch-only in
`scratch/trak_stage2_live_probe.py`):

1. After each channel teaching block with `k>=1` and a positive condition,
   read `typed_slot_views_from_graph(eng.graph)` and call the unmodified
   `slot_keyed_forced_choice(views, subject, displayed_options)`.
2. If the selected pick is not the label the channel itself taught
   (`rp.taught_pairs`, derived from teaching surfaces only), emit one extra
   `process_turn("subject is a label.")` and restore that edge's earned
   `source_metadata` (`rp.EARNED`, `slot_writer=tra_typed_slot`,
   `type_label`, `channel_source_id`). Never consult `item['gold']`,
   `unseen_test_words` gold, `world_model` or `type_sets`.
3. Cap at 2 re-encodes per (subject, label); SHAM arm makes the same number
   of `process_turn` calls with a neutral surface and no write.
4. All graded prompts, options, gates and thresholds stay byte-identical.

## 4. Pre-registered falsifiers (stop rules)

| # | Falsifier | Measurable prediction | If observed |
|---|---|---|---|
| F1 | Write-time census already admits the slot after ordinary channel teaching (trusted, `is_a`, `type_label` matches, confidence >= 0.20, margin >= 0.10) | If admitted yet P3/P4 still hedge, bottleneck is articulation/downstream guard, **not** write quality | Stop stage 2; instrument articulation at emission boundary |
| F2 | Boot-free hook with the actual post-teach view returns `None` on either surface | `scratch/trak_stage2_cue_probe.py` already shows 12/12 picks with an oracle view; a real-view mismatch would locate the defect in the view, not the cue | Fix view/provenance before re-encoding |
| F3 | S2 `immediate(k=8)` stays 0.000 | Stage 2 must raise primary+sealed immediate to >= 0.50 | Zero effect = write-time re-encoding is not the missing mechanism; report negative and stop |
| F4 | SHAM rises above 0 on immediate/delayed, or wrong-tag control is admitted | SHAM and WRONG must stay at 0.000 | Mechanism leaked through matched compute or trust gate; revert |
| F5 | `delayed_retention` remains `null` because immediate == 0 | With immediate > 0, require retention >= 0.80 on both pools | No measurable retention = no capability claim, regardless of delayed |
| F6 | Instrument controls fail, `delayed_option_audit.ok=false`, or mutation M-A/M-B does not move | All must stay as in OFF | `INSTRUMENT INVALID — no score`; no capability inference |
| F7 | Compute parity broken: extra `process_turn` counts differ by >0 or wall-clock ratio >1.15 | Re-encodes/sham turns counted, arms must match | Confound; revert and redesign with matched cycles |
| F8 | Any stage-2 write uses a label absent from admitted channel facts | 0 such writes allowed | Oracle leakage; result is non-deployable, discard |
| F9 | Two repeated OFF arms on separate `TRAK_AB_DATA_DIR` copies are not byte-identical | Must be identical | Store isolation failure; invalidate all arm deltas |
| F10 | Sealed pool delayed stays 0 while primary rises to 0.5 under a write-time fix | Both pools must move together | Primary-only gain is store-carryover/selection, not consolidation |

**G6 caveat:** a saturated control front (`tra_k_near=0`) is non-informative
but not blocking; a PASS must come from positive endpoints with controls
intact, not from a relaxed gate.

## 5. Exact parent run command

Run from the stage-2 worktree. The first two commands are boot-free and must
pass before any engine is started. Stop immediately on any preflight failure.

```bash
cd C:/Users/Likhith/Documents/projects/ravana_trak_stage2
export PYTHONPATH="$PWD/scripts;$PWD/ravana/src;$PWD/ravana_ml/src;$PWD/scratch"
export TRAK_REPLAY_SOURCE_ROOT=C:/Users/Likhith/Documents/projects/ravana
export TRAK_CHANNEL_JSON="$TRAK_REPLAY_SOURCE_ROOT/scratch/trak_channel_from_substrate.json"
export PYTHONHASHSEED=0 RAVANA_OFFLINE=1 RAVANA_TRA_SLOTS=1 RAVANA_TRA_K_SLOTS=1
PY=C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe

# Boot-free preflight
"$PY" scratch/trak_stage2_cue_probe.py
"$PY" scratch/trak_stage2_live_probe.py --self-test

# Sequential A/B; each arm gets its OWN store copy (no carryover)
for ARM in OFF S2 SHAM; do
  export TRAK_STAGE2_ARM="$ARM"
  export TRAK_AB_DATA_DIR="$PWD/scratch/trak_stage2_store/$ARM"
  export TRAK_AB_OUT="$PWD/scratch/trak_stage2_data/$ARM"
  mkdir -p "$TRAK_AB_DATA_DIR" "$TRAK_AB_OUT"
  echo "=== STAGE2 ARM $ARM START $(date) ===" >> scratch/trak_stage2_run.log
  TRAK_MAX_PER_POOL=2 timeout 1800 "$PY" scratch/trak_stage2_live_probe.py \
    >> scratch/trak_stage2_run.log 2>&1
  echo "ARM_${ARM}_EXIT=$? $(date)" >> scratch/trak_stage2_run.log
done
echo STAGE2_DONE >> scratch/trak_stage2_run.log
```

Then read only `scratch/trak_stage2_data/*/report.json` + `summary.json`, and
evaluate F1–F10 in order. Do not relax a threshold or change a prompt to
rescue a failed falsifier. If F3/F5 fail, the negative result is the
deliverable: name the next missing input (likely a subject-specific
type-evidence channel, not another consolidation tweak) in a docs-only
follow-up.

## 6. Deployable vs oracle boundary

`scratch/trak_stage2_cue_probe.py` injects a perfect trusted view to prove
reachability — that arm is an **oracle upper bound, non-deployable**. The
deployable signal is the live S2 arm, whose only extra write is a re-assertion
of a label already present in the arm's accepted channel teaching surfaces.
No stage-2 arm reads the item bank's gold target or oracle-only fields.
