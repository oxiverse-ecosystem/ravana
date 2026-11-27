# TRA-K far: source-arbitration after F4 — PRE-REGISTERED DESIGN (no result yet)

**Status: DESIGN ONLY.** Base `cd711c8` (PR #570 head, `docs/trak-far-repair-negative`).
Worktree `C:/Users/Likhith/Documents/projects/ravana_trak_arbitration`, branch
`invest/trak-source-arbitration`. No engine is booted, no store is opened for
write, and no production file, gate, threshold, item, prompt, gold key or
scorer is changed by this worktree. Scratch files need `git add -f scratch/...`.

## 1. The defect this addresses

F4 of `docs/investigations/TRA_K_FAR_REPAIR_AB.md` halted:
`far REPAIR − M123 = +0.000` with oracle 4/4. The saved-store replay shows the
cause is *write-time*, not read-time:

```
guild --is_a--> association          conf=0.00093  superseded_by='student association'
guild --is_a--> student association  conf=1.00000  revisions=8
market --is_a--> economy             conf=1.00000  (no market->market edge)
```

`intake._revise` halves an incumbent on every conflict regardless of the
**source** of the challenger. `chain_walker` skips `sid == oid`, so a
source-native identity type (`market is a market`) is never written. Both
facts are visible in the *channel's own licensed provenance*: the incumbent is
`primary` + `item_description`; the challenger is `alternate` +
`type_description`. The engine's own typed writes collapse both to
`conversation/user/curated`, so confidence, recency and `edge_kind` cannot
arbitrate them. This is the classic item-vs-source dissociation: the wrong
item is more familiar because it was written last, not because it is better
sourced.

## 2. Brain mechanisms and the computable mapping

Evidence is graded **[S]trong / [M]oderate / [W]eak**; each mechanism is a
*computational analogy*, not a claim that the brain has typed slots.

| mechanism | evidence | citation | rule in this design |
|---|---|---|---|
| source monitoring is a separate evaluation, dissociable from item familiarity | [S] human behavioural + review | Johnson, Hashtroudi & Lindsay (1993), *Psychological Bulletin* 114(1):3–28, doi:10.1037/0033-2909.114.1.3 (PMID 8346328) | arbitration reads **origin credentials only**, never edge weight, recency or label familiarity |
| epistemic vigilance is calibrated on the communicator, not content fluency | [S] human behavioural/theory | Sperber, Clément, Heintz, Mascaro, Mercier, Origgi & Wilson (2010), *Mind & Language* 25(4):359–393, doi:10.1111/j.1468-0017.2010.01394.x | need a **strict source-reliability advantage** before a challenger may demote an incumbent |
| reevaluation after source discrediting | [S] human developmental | Dautriche, Goupil, Smith & Rabagliati (2021), *Open Mind* 5:1–19, doi:10.1162/opmi_a_00038 (PMID 34485794) | a low-credibility challenger is retained as a separate trace, never as a slot replacement |
| reactivation opens a labile window in which an update can replace an incumbent | [S] rodent neurobiology | Nader, Schafe & Le Doux (2000), *Nature* 406:722–726, doi:10.1038/35021052 (PMID 10963596) | a *legitimate* higher-credibility update may supersede; "no demotion" is not "no updating" |
| reminder-driven integration can import new items into an old trace | [M] human behavioural | Hupbach, Gomez, Hardt & Nadel (2007), *Learning & Memory* 14(1–2):47–53, doi:10.1101/lm.365707 | teaching order matters; arbitration must hold across conditions, not just the far turn |
| pattern separation prevents a similar new input from overwriting an existing representation | [S] rodent electrophysiology + human imaging | Leutgeb, Leutgeb, Treves, Moser & Moser (2007), *Science* 315:961–966, doi:10.1126/science.1135801; Yassa & Stark (2011), *Trends in Neurosciences* 34(10):515–525, doi:10.1016/j.tins.2011.06.006 | an unqualified challenger is written as a **separate alternative trace**, leaving the incumbent slot intact |

Citations were resolved against publisher/PubMed records in this task. The
PFC/localisation claim (frontally mediated source monitoring, Johnson et al.
1993) is a motivation only; no neural localisation is implemented. `.venv`
conflict-of-interest note: "source credibility" here means *licensed source
metadata* (`license`, `sense`, `text_evidence_kind` from the frozen CC0
snapshot), never the benchmark's `evidence_role` or any answer-side field.

## 3. Candidate rule (pure, default-off)

Inputs for one typed slot: incumbent `(label, source_credibility)`, challenger
`(label, source_credibility)`, an explicit negation flag, and `subject == object`.

```
credibility = LICENSE_CREDIBILITY[license] * SENSE_WEIGHT[sense] * TIER_WEIGHT[evidence_kind]
decide(incumbent, challenger):
  if challenger.negated            -> VETO (no write)
  if same label                    -> REINFORCE
  if challenger.cred > incumbent.cred + MARGIN and corroborated
                                   -> SUPERSEDE (demote incumbent, write challenger as slot)
  else                             -> SEPARATE (keep incumbent slot; write challenger
                                                 as low-confidence alternative)
self_loop(subject, object, admitted_source):
  if subject == object and source admitted -> write identity type as a slot trace
  (the production writer's sid==oid skip is not applied to this admitted case)
```

Constants belong to the NEW module; no existing threshold is read or changed.
`MARGIN = 0.05`, `SENSE_WEIGHT = {primary: 1.0, alternate: 0.5}`,
`TIER_WEIGHT = {item_description: 1.0, type_description: 0.75, label_only: 0.5}`.
`LICENSE_CREDIBILITY` is **imported unchanged** from
`ravana.learn.subject_type_evidence` (do not duplicate). The self-loop rule
reads only `subject`, `object` and the admitted source status — no item id,
domain, gold label or answer-side token.

## 4. Boot-free experiment (scratch only)

Runner `scratch/trak_source_arbitration_experiment.py` imports the UNMODIFIED
`m123_preflight`, `m3_graded_probe`, `tra_k_items`, `measure_tra_k`,
`subject_type_evidence` and `source_evaluation` primitives. It uses:

* frozen bank + frozen CC0 snapshot + frozen channel JSON;
* saved M123/REPAIR stores read-only (`pickle.load`), one **independent deep
  copy per arm** — no arm ever writes another arm's or the parent's store;
* the harness's own `channel_teaching_surfaces(item)` / `ordered_channel_facts`
  to build the recorded assertion stream (condition-independent by design,
  which is exactly how the near turn poisons the far slot in F4);
* the UNMODIFIED M3 reader and production `CONGRUENCE_FLOOR=0.20` /
  `STRICT_MARGIN=0.10` to grade every arm.

Arms (all boot-free):

| arm | manipulation | status |
|---|---|---|
| `BASELINE` | production `_revise` semantics: every challenger halves the incumbent; `sid==oid` skipped | deployable control |
| `BASELINE2` | independent repeat on a fresh deep copy | carryover control |
| `ARBITRATION` | Section 3 rule, applied to the same recorded stream | candidate, default-off |
| `SHAM` | same call count, source credentials replaced by a wrong-tag sentinel | control leak gate |
| `ORACLE` | unmodified reader, answer-side confidence injected into a scratch evidence bundle | **NON-DEPLOYABLE reachability only** |

AST anti-leakage extends the preflight banned-token scan to every NEW decision
function. The 13 pinned files are re-hashed and must match the parent tree.
No live `process_turn` call is made; compute parity is measured in **decision
calls**, not wall clock.

## 5. Pre-registered endpoints and falsifiers

Endpoints: per-item, per-condition far/near/foil/negative/unrelated reader
`pick`, `reason`, `scores`; slot confidences before/after; decision counters;
oracle pick. Primary/sealed reported side by side; n=2/pool is a mechanism
probe, never a full-bank verdict.

| # | falsifier | threshold / action |
|---|---|---|
| A1 | AST anti-leakage hit, pin mismatch, or snapshot hash mismatch | `INSTRUMENT INVALID — no score` |
| A2 | `BASELINE != BASELINE2` on any endpoint | shared-state carryover; discard all deltas |
| A3 | `SHAM` emits a pick, or its decision/trust call counts differ from `ARBITRATION` | control leak; discard contrast |
| A4 | `ORACLE` does not reach 4/4 far | reader path not shown reachable; no interpretation of any arm |
| A5 | primary mechanism: `ARBITRATION` does not keep every high-credibility incumbent above `CONGRUENCE_FLOOR` on all 4 far subjects, or `BASELINE` has no demoted incumbent | mechanism did not fire / no substrate: **do not implement** |
| A6 | far `ARBITRATION − BASELINE < +0.50` on either pool (2 items/pool) | write arbitrations do not restore the read endpoint; **do not implement** |
| A7 | `ARBITRATION` far < `ORACLE` far on either pool, or any near/foil/negative/unrelated pick changes in the wrong direction | regression / answer-side leak; **do not implement** |
| A8 | k=0 far rows change, or any gate/threshold/item/scoring pin changes (F10-equivalent) | instrument or production violation; abort |
| A9 | self-loop `market→market` still absent under `ARBITRATION`, or self-loop arm moves only one of the two `market` rows | self-loop rule incomplete; **do not implement** |

**Decision rule:** implement a pure default-off arbitration module with unit
tests **only if A1–A9 all pass**. Any failure is a negative deliverable:
commit the scratch script/results/doc and state the missing mechanism. No
capability claim is made either way; the live A/B remains the parent's call.

## 6. Exact parent A/B command (to run AFTER this worktree is verified)

```bash
cd C:/Users/Likhith/Documents/projects/ravana
export PYTHONHASHSEED=0 RAVANA_OFFLINE=1
export TRAK_CHANNEL_JSON=C:/Users/Likhith/Documents/projects/ravana/scratch/trak_channel_from_substrate.json
PY=C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe
for SPEC in "OFF:off1" "OFF2:off2" "M1+M2+M3:m123" \
            "ARBITRATION:arb" "SHAM:sham"; do
  A="${SPEC%%:*}"; T="${SPEC##*:}"
  timeout 1800 "$PY" scratch/m123_combined_runner.py --stage trak --arm "$A" \
    --tag "$T" --base "$PWD/scratch/m123_ab_data" \
    --out "$PWD/scratch/m123_ab_out/trak_${T}.json" --max-per-pool 2
done
```

`--base` must point at a **fresh copied store per arm** (`seed_data_dir()` in
the runner); `ARBITRATION` is *not* added to `pf.ARM_FLAGS` here — the parent
must decide the live wiring. This command is a proposal, not a result.
