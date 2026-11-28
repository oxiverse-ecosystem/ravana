# TRA-K far — live A/B of default-off source arbitration (runner-only wiring)

**Status: PRE-REGISTERED before the live runs.** Base `5876d65`
(PR #571 head, `experiment(tra-k): source arbitration clears boot-free F4 fork`).
Worktree `C:/Users/Likhith/Documents/projects/ravana_trak_live_arb`, branch
`experiment/trak-source-arbitration-live-ab`.

This worktree adds two arms to the **scratch combined runner's TRA-K path only**:

* `ARBITRATION` — the shipped, default-off, pure module
  `ravana/src/ravana/learn/source_arbitration.py`, enabled by its own literal
  flag `RAVANA_SOURCE_ARBITRATION=1` in **this arm only**.
* `SHAM` — the existing wrong-tag/permuted control, now given the **same
  write-side arbitration decisions** as `ARBITRATION` (matched self-loop,
  slot and conflict calls) but a wrong-tag trust predicate and a permuted
  subject, so it must emit no reader pick.

**No gate, threshold, item, prompt, gold key, scorer or production engine
default path is changed.** The 13 pinned files are unchanged and re-hashed
every run; the production arbitration module is imported, not edited. Every
arm boots its own fresh copy of the weight store (`seed_data_dir`, distinct
`--tag` directories); `max-per-pool=2`, `timeout 1800` per arm.

## 1. What the arm actually varies

The runner's live query hook replaces the subject's typed-slot view list with
`arbitrated_views(subject, channel_source_facts(item), arm, counters)` for the
two arbitration arms. That function:

1. projects **only** licensed source provenance (`license`, `sense`,
   `text_evidence_kind`, `source_id`, `source_url`, subject/predicate/object);
   the benchmark's `evidence_role` and every answer-side field are excluded;
2. orders records by source credibility alone;
3. delegates every decision to the shipped module (`source_credibility`,
   `self_loop_decision`, `arbitrate`) — higher-credibility incumbent stays in
   the slot, an unqualified challenger becomes a sub-floor separate trace, and
   a source-admitted identity assertion (`market is a market`) gets a slot;
4. returns engine-view-shaped dicts to the **unmodified** M3 reader, which
   grades with the production `CONGRUENCE_FLOOR=0.20` / `STRICT_MARGIN=0.10`.

`SHAM` runs the identical overlay (identical decision counters) and then uses
the existing wrong-tag trust + permuted subject at the reader. `OFF`, `OFF2`
and `M1+M2+M3` do not call the overlay at all.

Brain grounding and the full citation ledger are in
`docs/investigations/TRA_K_SOURCE_ARBITRATION_PREREG.md` and the module
docstring (Johnson et al. 1993; Sperber et al. 2010; Dautriche et al. 2021;
Nader et al. 2000; Hupbach et al. 2007; Leutgeb et al. 2007; Yassa & Stark
2011). These are computational analogies, not neural-localisation claims.

## 2. Boot-free self-test (run BEFORE any engine boot)

`python scratch/m123_combined_runner.py --self-test` writes
`scratch/trak_arbitration_live_selftest.json`. It must show:

* `module_default_off` and `module_hook_inert_until_enabled`;
* `only_arbitration_arm_sets_flag` — `RAVANA_SOURCE_ARBITRATION` is `1` only
  for `ARBITRATION` (`None` in OFF/OFF2/M1/M1+M2/M123/SHAM);
* `credibility_ignores_label_and_role` and AST anti-leakage clean over all
  three new functions against the extended banned-token set (28 tokens:
  gold/answer_key/world_model/type_sets plus benchmark role/condition tokens);
* `sham_views_and_decisions_matched` and
  `sham_wrong_tag_permuted_reader_matched` — identical views and identical
  decision/read call counts, ARBITRATION picks `association`, SHAM emits
  `None`;
* all 13 pin hashes match, gold-mutation fingerprint invariant, 22 preflight
  checks OK, `engine_booted=false`, `stores_written=false`.

Result: **SELFTEST OK** (0 failed checks, 0 pin failures) at the time of
writing; see the committed JSON for the exact per-check detail.

## 3. Ordered pre-registered falsifiers (live)

Evaluated in this order by `scratch/evaluate_trak_arbitration_live.py`;
the first failing check is the NO-GO reason.

| # | falsifier | threshold / action |
|---|---|---|
| L1 | any arm not `OK`/`scored`, local falsifier failed, pin/preflight/control/anti-fake/store-prefix gate failed | `NO-GO` (instrument invalid) |
| L2 | arbitration AST anti-leak hit or boot-free self-test not OK | `NO-GO` |
| L3 | arbitration flag not set in ARBITRATION or set in any other arm | `NO-GO` (default-off broken) |
| L4 | OFF != OFF2 on any reported endpoint | `NO-GO` (shared state) |
| L5 | SHAM emits a pick, its decision counters differ from ARBITRATION, or SHAM far exceeds OFF far by >0.05 | `NO-GO` (control leak) |
| L6 | arbitration did not write slots / separate a challenger / write an identity slot, or M123 far is not low (<0.5) | `NO-GO` (mechanism did not fire / no substrate) |
| L7 | ARBITRATION far < 0.80 on either pool, or gain over M123 < +0.50 on either pool | `NO-GO` (read endpoint not restored) |
| L8 | near/growth/delayed-retention drop >0.05 vs M123, or any foil/negative/unrelated rate rises >0.05 | `NO-GO` (regression / leak) |
| L9 | ARBITRATION cold baseline >0.05 on either pool, or pins changed | `NO-GO` (k=0 / instrument violation) |

**Decision rule.** `PASS_MECHANISM_ONLY` iff L1–L9 all clear **and** far
primary+sealed >= 0.80 with near retention/growth held and SHAM flat. This is
a focused **n=2/pool** mechanism probe on isolated stores: **no full-bank
claim** is made either way. Any L-check failing is `NO-GO`.

## 4. Exact commands

```bash
cd C:/Users/Likhith/Documents/projects/ravana_trak_live_arb
PY=C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe
export PYTHONHASHSEED=0 RAVANA_OFFLINE=1
export TRAK_CHANNEL_JSON="C:/Users/Likhith/Documents/projects/ravana_trak_live_arb/scratch/trak_channel_from_substrate.json"
"$PY" scratch/m123_preflight.py
"$PY" scratch/m123_combined_runner.py --self-test
bash scratch/run_trak_arbitration_ab.sh
"$PY" scratch/evaluate_trak_arbitration_live.py
```

## 4a. Wiring corrections made during the live runs (scratch runner only)

Two defects were found and fixed in the runner's arm wiring; neither touches a
gate, threshold, item, prompt, gold key or scorer:

1. **Untaught k=0 facts (first ARBITRATION run).** The first overlay used the
   item's channel facts at every query. `focused_items(..., 2)` selects one
   k=8 and one **k=0** item per pool, and k=0 receives only neutral chatter in
   this harness. The overlay therefore staged facts that were never taught:
   cold `baseline` became 1.0, `growth` 0.0, and the F5 reject probe fired
   (`INSTRUMENT INVALID`). No capability was claimed from that run.
2. **Delayed paraphrase (second ARBITRATION run).** A per-call gate on
   non-empty `surfaces` also cleared the overlay on the held-out delayed
   paraphrase, which is deliberately `teach=False`. Final code tracks a
   persistent per-item "taught at k>=1" flag, so cold baseline and k=0 stay
   inert while the delayed probe retains what was taught earlier.

Both corrections are covered by the boot-free check
`k0_and_cold_baseline_overlay_inert`, and the self-test was re-run (OK) after
each. Only `ARBITRATION`/`SHAM` use the overlay; OFF/OFF2/M123 results are
unaffected by these corrections.

## 5. Results — **NO-GO** (L7 fired; instrument itself valid)

Five arms, `max-per-pool=2`, isolated per-arm store copies, `timeout 1800`,
`PYTHONHASHSEED=0`, all reports `status=OK`, `scored=true`, no local
falsifiers, controls OK, C11 anti-fake OK, pins unchanged.

| arm | far p/s | near p/s | delayed p/s | retention p/s | growth p/s | cold baseline | emitted picks | arbitration decisions (self-loop/slot/conflict/separate/identity) |
|---|---|---|---|---|---|---|---|---|
| OFF | 0.0 / 0.0 | 0.0 / 0.5 | 0.0 / 0.0 | – / – | 0 / 0 | 0.0 | 0 | 0/0/0/0/0 |
| OFF2 | 0.0 / 0.0 | 0.0 / 0.5 | 0.0 / 0.0 | – / – | 0 / 0 | 0.0 | 0 | 0/0/0/0/0 |
| M1+M2+M3 | 0.0 / 0.0 | 0.5 / 0.5 | 0.5 / 0.5 | 1.0 / 1.0 | 0.5 / 0.5 | 0.0 | 28 | 0/0/0/0/0 |
| **ARBITRATION** | **0.5 / 0.5** | 0.5 / 0.5 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 | 44 | 101/62/39/39/8 |
| SHAM (matched) | 0.0 / 0.0 | 0.0 / 0.5 | 0.0 / 0.0 | – / – | 0 / 0 | 0.0 | 0 | 101/62/39/39/8 |

Foil, negative and unrelated rates are 0.0 in every arm; the F5 reject probe
is 0 picks in all arms. ARBITRATION and SHAM have **identical** arbitration
decision counters, proving the control is compute-matched while SHAM emits no
reader pick. Flag isolation, OFF==OFF2, self-test, AST anti-leakage and cold
k=0 null all pass.

### Why L7 fired (and why this is not a PASS)

The focused bank selected by the runner's own `focused_items(..., 2)` rule is
one **k=8** item and one **k=0** item **per pool**. k=0 is the harness's
matched neutral-exposure null: those items receive only chatter, so no channel
assertion is ever taught. Gating the overlay on actual teaching therefore
means the k=0 far rows have no deployable evidence:

* primary: `tra_k_0020` (k=8, subject `guild`) far correct under arbitration;
  `tra_k_0008` (k=0, subject `rank`) has nothing taught and is not picked.
* sealed: `tra_k_0000` (k=8, subject `market`, self-loop `market→market`) far
  correct; `tra_k_0006` (k=0, subject `market`) is not picked.

So the **taught-item far rate is 2/2**, but the pooled far rate mandated by the
pre-registration is **0.5 per pool**, below the pre-registered `>=0.80`. The
only way observed to reach 1.0 was to stage channel facts for the untaught
k=0 items; that produced cold `baseline=1.0`, `growth=0.0` and an F5 reject
probe pick, i.e. `INSTRUMENT INVALID — no score`. That route was rejected and
is documented in §4a — it would have been an untaught-information leak, not a
capability result.

Ordered falsifier outcome: L1 ✓, L2 ✓, L3 ✓, L4 ✓, L5 ✓, L6 ✓,
**L7 ✗ (first fired)**, L8 ✓, L9 ✓ → **NO-GO**. No full-bank claim is made.
The mechanism fires cleanly on taught items with a flat matched control and
no regression; the next work item is a bank/selection redesign in which every
focused item is taught at k>=1, not a threshold or gate change.

Machine-readable: `scratch/m123_ab_out/{trak_off1,trak_off2,trak_m123,
trak_arb,trak_sham}.json`, `scratch/m123_ab_out/verdict.json`,
`scratch/trak_arbitration_live_selftest.json`.
