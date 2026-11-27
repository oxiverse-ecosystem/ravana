# TRA-K far transfer after M123 — root cause and falsifiable A/B design

**Status: boot-free root cause, no engine boot, no production file, gate,
threshold, item, prompt, gold key or scorer changed.** Worktree
`C:/Users/Likhith/Documents/projects/ravana_trak_far_rootcause`, branch
`invest/trak-far-rootcause`, base `e3a52a0`.

Diagnostic: `scratch/trak_far_rootcause_probe.py`; raw output
`scratch/trak_far_rootcause_result.json`; supporting edge dump
`scratch/trak_far_edge_dump.py` / `.log`. It imports the **unmodified**
staging and reader primitives the M123 runner used (`m123_preflight.py`,
`m3_graded_probe.py`) and replays them on the **saved** M123 stores
(`ravana_m123_run/scratch/m123_data/trak_m123/<item>/ravana_weights.pkl`,
`durable_provenance.json`) plus the frozen channel/snapshot. No engine is
constructed and no store is opened for write.

## 1. Verdict

**FAR_TRANSFER_BLOCKED_BY_STORED_TYPE_EVIDENCE, not coverage, not option
overlap, not a reader threshold defect, not missing far subjects.** All four
far query rows in the executed M123 TRA-K sweep reach the M3 reader, are
admitted, and defer with `no_trusted_label_above_floor`: the deployable
confidence of the far gold label is `0.0000` or `0.00093`, below the
*unchanged production* `CONGRUENCE_FLOOR = 0.20`. The two near k=8 rows on
the same stores pick their gold label with confidence `0.6114` and `1.0`.
The near/immediate/delayed/growth gain is therefore a near-only gain carried
by evidence that survived the write path; the far items' evidence did not.

Two write-time pathologies, both visible in the saved edge metadata, produce
the deficit:

1. **Slot supersession by the planted low-credibility conflict.** On
   `tra_k_0020`, far subject `guild`, the positive channel fact
   `guild is an association` is stored as `is_a → association` with
   confidence `0.0009328` and metadata
   `superseded_by: "student association"`; the deliberately planted
   *misleading* fact `guild is a student association` is stored on the same
   subject with confidence `1.0` and `revisions: 8`. `intake._revise`
   (`ravana/src/ravana/learn/intake.py:792-824`) halves the incumbent's
   confidence and stamps `superseded_by` on every revision of a functional
   slot. The correct label is therefore repeatedly demoted by a source the
   benchmark itself labels lower-credibility.
2. **Self-loop positive fact.** On `tra_k_0000` / `tra_k_0006`, far subject
   `market`, the positive label is the subject itself (`market → market`).
   The typed-slot writer skips `sid == oid`
   (`ravana/src/ravana/chat/chain_walker.py:1206-1209`), so no correct edge
   exists at all; only `market → economy` (the misleading conflict) is stored
   at `1.0`.

Both are *write/evidence* failures. M1, M2 and M3 are all working as
designed: they defer exactly when the admitted label has no stored
confidence above floor.

## 2. Evidence (all boot-free)

### Four-way classification

| candidate cause | verdict | decisive observation |
|---|---|---|
| Subject-side evidence coverage | **ruled out** | 400/400 near and 400/400 far rows have admitted snapshot evidence for the actual query subject; 400/400 far subjects have their gold label among those evidence labels |
| Candidate label overlap | **ruled out** | 400/400 far rows have the gold label in the displayed option set; 400/400 have at least one admitted label in the option set; the preflight's `delayed_option_audit` was `ok=true`, 0 row mismatches |
| Far subjects absent / wrong pool | **ruled out** | far subjects are drawn from `OBJECTS[far_domain]` with `len(far_pool) >= 4`; all four replayed far subjects resolve to real snapshot evidence (`market`, `rank`, `guild`) |
| Reader threshold / wiring | **ruled out as the defect** | `hook_calls=160`, `production_calls=160`, `scorer_calls=160`, `records_admitted=1081`; all far rows reach `read_graded_probe`. Non-deployable oracle control: raising only the gold label's confidence to 1.0 in a scratch bundle makes the **unmodified** reader pick the gold option on 4/4 far rows (`admitted_graded_probe_winner`) |
| **Stored confidence for the gold label** | **confirmed** | near k=8 scores `{'cloud cover': 0.6114}`, `{'snow storm': 1.0}`; far k=8 scores `{'market': 0.0}`, `{'association': 0.0009}`; every far row reason is `no_trusted_label_above_floor` |

### Replayed rows (saved stores, unmodified reader)

| item | pool | k | cond | subject | reader reason | scores | pick |
|---|---|---|---|---|---|---|---|
| tra_k_0000 | sealed | 8 | near | overcast | admitted_graded_probe_winner | cloud cover=0.6114 | cloud cover |
| tra_k_0000 | sealed | 8 | far | market | no_trusted_label_above_floor | market=0.0 | None |
| tra_k_0006 | sealed | 0 | far | market | no_trusted_label_above_floor | market=0.0 | None |
| tra_k_0008 | primary | 0 | far | rank | no_trusted_label_above_floor | class=0.0 | None |
| tra_k_0020 | primary | 8 | near | blizzard | admitted_graded_probe_winner | snow storm=1.0 | snow storm |
| tra_k_0020 | primary | 8 | far | guild | no_trusted_label_above_floor | association=0.0009 | None |

Foil/negative/unrelated rows remain fail-closed (`no_admitted_subject_evidence`
or below floor), matching the reported 0/12 picks on each.

### Saved-edge census (the direct cause)

```
guild --is_a--> association        conf=0.0009328  edge_kind=curated
   source_metadata.superseded_by='student association', revisions=0
guild --is_a--> student association conf=1.0        edge_kind=curated, revisions=8
market --is_a--> economy            conf=1.0        edge_kind=curated   (no market->market edge)
blizzard --is_a--> snow storm       conf=1.0        edge_kind=curated, revisions=8
```

No gate, trust predicate, floor or margin was changed to obtain any of these
numbers. The reader's own constants (`CONGRUENCE_FLOOR=0.20`,
`STRICT_MARGIN=0.10`) are imported from production and untouched.

## 3. Why the far domain is hit and near is not

Near query subjects are drawn from the taught domain; their best substrate
type is a specific, unambiguous label (`cloud cover`, `snow storm`, `wind`)
and the planted conflicts are either junk-like (`Wikimedia disambiguation
page`) or a measurement (`wind speed`) that the typed-slot writer does not
admit as a replacement. Far subjects are drawn from a *different* domain and
are typically polysemous (`market`, `guild`, `rank`); their best type is a
general/hub label (`market`, `association`, `class`) and the alternate sense
is itself a plausible type (`economy`, `student association`, `card game`).
With a functional `is_a` slot, the later, lower-credibility assertion wins
the slot by recency and the reader's source-derived confidence for the
correct label collapses. This is the schema-plus-retrieval failure predicted
by the literature below: a schema cue is only useful if the encoding that
created it survives to retrieval; an overridden or self-looped encoding
leaves no cue above the decision floor.

## 4. Literature (verified this task; web citations)

- Barnett, S. M. & Ceci, S. J. (2002). When and where do we apply what we
  learn? A taxonomy for far transfer. *Psychological Bulletin*, 128(4),
  612–637. doi:10.1037/0033-2909.128.4.612 — far transfer varies along
  content and context dimensions; domain distance alone does not predict it.
  https://psycnet.apa.org/record/2002-01514-006
- Gick, M. L. & Holyoak, K. J. (1983). Schema induction and analogical
  transfer. *Cognitive Psychology*, 15, 1–38.
  doi:10.1016/0010-0285(83)90002-6 — analogies between semantically
  disparate domains are retrieved by a *schema*; without an abstracted
  schema there is no retrieval cue. https://gwern.net/doc/psychology/1983-gick.pdf
- Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy.
  *Cognitive Science*, 7(2), 155–170. doi:10.1207/s15516709cog0702_3 —
  transfer maps relations, not attributes; a surface-similar but
  relation-mismatched candidate is not a valid source.
- Vendetti, M. S., Wu, A. & Holyoak, K. J. (2014). Far-Out Thinking:
  Generating solutions to distant analogies promotes relational thinking.
  *Psychological Science*, 25(4), 928–933. doi:10.1177/0956797613518079 —
  semantically *distant* analogies, not near ones, induced relational
  transfer; supports treating far as a distinct mechanism.
- Tse, D. et al. (2007). Schemas and memory consolidation. *Science*, 316,
  76–82. doi:10.1126/science.1135935 — one-trial assimilation requires an
  already-established, *consistent* schema; inconsistent associations do not
  show rapid assimilation. https://www.science.org/doi/10.1126/science.1135935
- Nader, K., Schafe, G. E. & Le Doux, J. E. (2000). Fear memories require
  protein synthesis in the amygdala for reconsolidation after retrieval.
  *Nature*, 406, 722–726. doi:10.1038/35021052 — retrieval reopens a trace;
  an update can replace/demote the incumbent. This is the computational
  analogue of `_revise`'s `superseded_by` demotion.
- Tulving, E. & Thomson, D. M. (1973). Encoding specificity principle.
  *Psychological Review*, 80(5), 352–373. doi:10.1037/h0020071 and Morris,
  Bransford & Franks (1977), *JVLVB* 16, 519–533,
  doi:10.1016/S0022-5371(77)80016-9 — retrieval succeeds only for cues
  overlapping the encoding; the stored type label is the cue here.
- Johnson, M. K., Hashtroudi, S. & Lindsay, D. S. (1993). Source monitoring.
  *Psychological Bulletin*, 114(1), 3–28. doi:10.1037/0033-2909.114.1.3 —
  item familiarity and source credibility dissociate; a plausible misleading
  label must not displace a credible one merely by recency.

These are computational analogies to typed graph edges, not a claim that the
brain has `is_a` slots. The non-deployable oracle arm is kept separate from
all evidence arms for the same reason.

## 5. Falsifiable far-transfer A/B (DESIGN ONLY — not executed; no engine boot)

### Arms and manipulation

All arms use the shipped `scratch/m123_combined_runner.py --stage trak`, one
process per arm, the same bank/channel/seed, `PYTHONHASHSEED=0`, and a
**fresh store copy per arm** via `seed_data_dir()`. No trust predicate,
`CONGRUENCE_FLOOR`, `STRICT_MARGIN`, item, prompt, gold key or scorer changes.
The 13 pinned files are re-hashed and must match; the arm-key extension is a
scratch-only addition to `pf.ARM_FLAGS` / `select_items`, not to production.

| arm | flags | manipulation | predicted far(k=8) |
|---|---|---|---|
| `OFF` | none | unchanged | 0.000 |
| `OFF2` | none | independent repeated baseline on a fresh copy | 0.000, byte-identical to OFF |
| `M123` | DURABLE+SUBJECT_EVIDENCE+M3 | unchanged | 0.000 (replicate) |
| `REPAIR` | same as M123 | replace ONLY the planted `misleading` channel assertion in the **far** P2c teaching with a length-matched neutral turn; positive subject assertion, near/foil/negative teaching, M1/M2/M3 code and all gates unchanged | >0.000 on non-self-loop far items if supersession is causal |
| `REPAIR_SELFLOOP` | same as M123 | `REPAIR` plus one post-conflict re-affirmation of the same admitted subject fact on self-loop items (positive object == subject) | moves only the `market` rows if self-loop is the remaining cause |
| `SHAM` | same as M123 | `wrong_tag_trust` + permuted subject, matched scorer/trust calls | 0.000 |
| `ORACLE` | n/a | boot-free scratch bundle: inject the displayed gold label's confidence only; never fed to a live arm | 4/4 picks, labelled NON-DEPLOYABLE |

Compute matching: the neutral replacement turn has the same token count, so
`process_turn`, `query_turns`, `scorer_calls` and `hook_calls` must match
M123 within the pre-registered equality check; `records_admitted` may differ
only by the removed misleading records and must be reported.

### Pre-registered endpoints and falsifiers

Endpoints: `far`, `far_minus_chance`, `near`, `immediate`, `delayed`,
`delayed_retention`, `growth`, `foil`, `negative`, `unrelated`, per-pool
primary/sealed, plus row-level far-only immediate/delayed from the saved
`rows`. Report primary and sealed side by side; n=2/pool here is a mechanism
probe, not a full-bank verdict.

| # | falsifier | threshold / action |
|---|---|---|
| F1 | Pin hashes, boot-free preflight or AST anti-leakage fail | `INSTRUMENT INVALID — no score`; fix before any run |
| F2 | `OFF != OFF2` on any endpoint | store-carryover; invalidate all deltas |
| F3 | `SHAM` emits a pick, or its hook/scorer/trust counters differ from M123 | control leak; discard contrast |
| F4 | far `REPAIR − M123 < +0.10` on either pool while ORACLE reaches 4/4 | supersession is **not** sufficient; the missing input is the positive subject→type assertion itself, not a reader tweak — stop, do not tune the floor |
| F5 | `near` drops >0.05 vs M123, or foil/negative/unrelated pick >0 | regression or gate leak; revert arm |
| F6 | `REPAIR` moves far but also moves foil/negative | answer-side leak through the manipulation; discard |
| F7 | k=0 far rows move | null-control failure; the arm changed exposure, not evidence |
| F8 | G6 saturation (`far = 1.00`) without M-A/M-B mutation movement | mechanism evidence only, no capability claim |
| F9 | `REPAIR_SELFLOOP` leaves self-loop far rows at 0 while non-self-loop rows rise | self-loop is a separate, bank-construction ceiling; report, do not change the option generator from the gold key |
| F10 | Any production file, gate, threshold, item, prompt, gold or scorer changes (`git diff --stat` over the 13 pins) | abort; the result cannot license a mechanism claim |

Expected reading: F4 is the decisive fork. A positive F4 with F2/F3/F5/F6/F7
clean localises the zero to low-credibility supersession of the far type slot;
a negative F4 with the ORACLE control passing localises it further upstream to
the positive subject-assertion write (including the self-loop case). Neither
outcome licenses lowering the congruence floor.

### Run command (to be executed in a fresh A/B worktree, not here)

```bash
cd <trak-far-ab-worktree>
export PYTHONHASHSEED=0 RAVANA_OFFLINE=1
export TRAK_CHANNEL_JSON=C:/Users/Likhith/Documents/projects/ravana/scratch/trak_channel_from_substrate.json
PY=C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe
for SPEC in "OFF:off1" "OFF2:off2" "M1+M2+M3:m123" "REPAIR:repair" \
            "REPAIR_SELFLOOP:repair_self" "SHAM:sham"; do
  A="${SPEC%%:*}"; T="${SPEC##*:}"
  timeout 1800 "$PY" scratch/m123_combined_runner.py --stage trak --arm "$A" \
    --tag "$T" --base "$PWD/scratch/m123_ab_data" \
    --out "$PWD/scratch/m123_ab_out/trak_${T}.json" --max-per-pool 2
done
# boot-free before the loop:
#   "$PY" scratch/trak_far_rootcause_probe.py   (re-run; must reproduce this doc)
```

## 6. Boundaries

- No engine was booted and no live A/B was run in this worktree; the A/B in
  §5 is a pre-registered design, not a result.
- The executed M123 TRA-K probe was `max-per-pool=2` (4 items); `far_n=2` per
  pool and G6=`FLAG`. The root cause is established on those four saved
  stores plus a 400/400 full-bank static census of the up-stream channels;
  a full-bank far claim still needs a clean focused A/B first.
- The ORACLE arm injects answer-side confidence and is explicitly
  non-deployable. It proves reachability, not capability.
