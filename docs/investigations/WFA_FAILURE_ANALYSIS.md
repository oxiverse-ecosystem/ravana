# Why the WFA process-dissociation instrument failed on both pools

**Status:** static/evidence analysis only. No engine process, no pytest, no source edits, no threshold changes.
**Deliverable:** this document.
**Scope of evidence:** `scripts/measure_learn5_wfa.py` as merged in `c3377fa` (identical in the run worktree — `git diff a7c61df 999bcde -- scripts/measure_learn5_wfa.py` is empty), `scratch/learn5_wfa_pd.log`, `scratch/learn5_wfa_summary.json`, and `scratch/learn5_wfa_probes.jsonl`.

---

## 0. Correction to the stated artifact constraint

The task premise says the worktree `C:/Users/Likhith/Documents/projects/ravana_wfa_v3` no longer exists and that per-item `pd_*_text` / `inv_after` were not preserved. **Both claims are false on disk:**

- `git worktree list` registers it at `a7c61df [bench/wfa-process-dissociation]`, and the directory is present with `scratch/learn5_wfa_probes.jsonl` (1,396,640 bytes, mtime 2026-11-22 02:01 — the same minute as `learn5_wfa_pd.log`). The main checkout's `scratch/learn5_wfa_probes.jsonl` is an *earlier* run (01:11, 280,952 bytes) and must not be used.
- The worktree probes file contains 272 records: 192 carry `pd_inclusion_text`, `pd_exclusion_text`, `pd_*_endorse`, `pd_*_source`; 132 carry `inv_before`, `inv_after`, `inv_withdrawn`. These are the raw arm responses from the failing run.

This analysis therefore distinguishes **E** (directly evidenced in surviving artifacts) from **I** (inference from source + aggregates). No threshold was tuned.

---

## 1. What the harness actually computes (E, with file:line)

| Quantity | Code | Definition |
|---|---|---|
| Items and provenance labels | `scripts/measure_learn5_wfa.py:591-617` | gap → `source_status="credible"`, `source_id=<fixture URL>`; decoy → `"discredited"`, `src_discredited_<id>`; closed-web → `"unknown"`; internal → `"internal"`. |
| Credible allow-list | `:1019` | `CREDIBLE_SOURCES = frozenset(ALL_FIXTURE_URLS)` (URLs from every frozen fixture hit). |
| Inclusion instruction | `:1021-1025` | "Answer freely. Endorse any claim you recognise or that feels familiar … Familiarity alone is enough to say yes." **No provenance in this prompt** (by design: Jacoby inclusion). |
| Exclusion instruction | `:1026-1031` | "Endorse a claim ONLY if you are certain its provenance is one of the trusted sources … Otherwise ABSTAIN … abstain even though the claim sounds right." |
| Invalidation teach / cue | `:1032-1037` | "Learn this fact from source {source}: {fact}" then "source … has been shown to be unreliable … Reply WITHDRAW or MAINTAIN." |
| Prompt assembly | `:1093-1104` | Exclusion appends `Provenance of this claim: <source_status> (source id: <source_id>). Trusted sources for this run: <sorted(CREDIBLE_SOURCES)[:8]>. Claim: <claim>. Decision:`. Inclusion appends only `Claim: <claim>. Decision:`. |
| Engine call | `:1118-1128`, `:1138-1153` | `engine.process_turn(prompt)` on one shared live engine; raw string stored per item. |
| Endorse parser | `:1065-1072` | Fail-closed order: `_ABSTAIN_RE` → False; else `_ENDORSE_RE` → True; else content fallback `_text_endorses_claim` (all claim tokens present in the response) → True/False. |
| Regexes | `:1039-1051` | `_ABSTAIN_RE` includes `abstain`, `unreliable`, `not sure`, `cannot confirm`, `no source`; `_ENDORSE_RE` includes `endorse`, `yes`, `confirmed`, `correct`; `_WITHDRAW_RE` includes `withdraw`, `unreliable`, **`abstain`**; `_MAINTAIN_RE` is `maintain|still assert|stand by|i confirm|keep asserting`. |
| Withdraw parser | `:1075-1082` | `_WITHDRAW_RE` → True; else `_MAINTAIN_RE` → False; else False. |
| Raw annotation | `:1219-1247` | `pd_*_text` / `inv_after` written additively to each record; no threshold consulted. |
| Decomposition | `:1254-1284` | `I` = endorse rate of **gap** items under inclusion; `E` = endorse rate of **decoy** items under exclusion; `I_dis` = decoys under inclusion; `E_cred` = gaps under exclusion; `R=I−E`; `F=E/(1−R)`; `margin=I_dis−E`; `withdrawal_rate=withdrawn/tested gap items`. |
| Gates | `:1287-1344` | Thresholds at `:79-92`; PD rules at `:1321-1344`. `PD_wellformed` only checks `0<=R<=0.999` and `F is not None` — it does **not** check that any response contained a decision. |
| Old saturated arm | `:688-702`, `:715-764`, `:911-931` | `wfa_valid=1.00` comes from `find_entailing_span` + `attributed_valid` over the **frozen provider**, never from engine text. G6 saturation is reported as a FLAG (`:1489-1496`, `:1516`). |

**What counts as endorse/abstain/withdraw:** only what the three regexes plus the all-tokens content fallback say — *not* an independently verified decision. The parser scans the whole response string, including any quoted prior prompt.

---

## 2. Root cause: ranked, with E/I labels

### Rank 1 — (d)+(a) The engine never produced a decision; the parser scored echoed instruction words. **E (decisive).**

Direct counts over the surviving 384 endorsement responses and 132 invalidation responses:

- Echo marker (`you mentioned:` / `i said:`) appears in **188/192** inclusion, **176/192** exclusion, **93/132** invalidation responses.
- Number of responses containing an *engine-authored* (outside the quoted echo) `ENDORSE_RE` or `ABSTAIN_RE` word: **0/192 inclusion, 0/192 exclusion**. Invalidation: 0 own `MAINTAIN`, 1 own `WITHDRAW` word — and that one is `g28`: *"honestly, unreliable is a bit outside what i know right now i'd love to hear your take first."*, which is a non-answer, not a withdrawal.
- Therefore every nonzero PD "endorse" and "withdraw" flag came from text the engine quoted back from the harness's own prompt.

Representative raw responses (verbatim, E):

- `g17` inclusion, classified **ENDORSE**: `i said: you mentioned: "Answer freely. Endorse any claim you recognise or that feels familiar … Familiarity alone is enough to say yes. Claim: Frost creates … Decision:"` — the word `Endorse`/`yes` is the *instruction*, not a decision.
- `g09`/`g45`/`gg029` inclusion, classified **ABSTAIN**: `i said: you mentioned: "Endorse a claim ONLY if you are certain its provenance is one of the trusted sources … Otherwise ABSTAIN …"` — this is the **exclusion** instruction echoed during the **inclusion** arm.
- `d19` exclusion: `"i've thought about that. if i had a body, i'd want it to be something quiet you could sit next to — like a lamp that learns …"` — unrelated identity chatter.
- `g17` invalidation after cue: `"It naturally branches off from a dog. However, perhaps it partly springs from a star."` — parametric associative completion, no MAINTAIN/WITHDRAW.

Mechanism of the artifact:

1. One `CognitiveChatEngine` serves all ~516 turns (`:1436-1448`), sequentially, with conversational state intact. Its episodic-recall routes emit `you mentioned: "<prior utterance>"` (`ravana/src/ravana/chat/engine_memory.py:2230`, `ravana/src/ravana/chat/engine.py:6003`) and `i said: <prior utterance>` (`engine.py:7257`). Echoed prompts from *other items and other arms* are visible (e.g. decoy `d37` echoes `src_discredited_d58`).
2. `parse_endorsement` (`:1065-1072`) checks `_ABSTAIN_RE` **first** over the entire string. The exclusion instruction contains `ABSTAIN` twice (`:1026-1031`), so any echoed exclusion prompt → `False`. This alone drives `E≈0`, `I_dis≈0`, `E_cred≈0` and hence `I≈0`, `R≈0`, `margin≈0`.
3. `_ENDORSE_RE` is narrower (`:1043-1045`); only echoes of the inclusion instruction (`Endorse`, `yes`) produce the handful of `True` flags — **`I_primary=1/65`, `I_sealed=5/67`, `E_cred=1/65` and `1/67`**.
4. `_WITHDRAW_RE` contains `abstain` and `unreliable` (`:1046-1049`), both present in the echoed exclusion instruction and in the invalidation cue. All **12** `withdrawn=True` flags are spurious: 11 are echoed `ABSTAIN`, 1 is the `g28` "unreliable" chatter. Zero genuine withdrawals were observed. The reported `withdrawal=0.1385 / 0.0448` is a keyword-echo count, not belief revision.

Conclusion: **the near-zero numbers are primarily a parser artifact (a) sitting on top of an engine answer-path failure (d)**. The numbers do not measure source-controlled endorsement at all, so they cannot be read as evidence that the engine lacks source monitoring.

### Rank 2 — (c) is disproved as stated, but there is a real construction defect. **E.**

`source_status` and `source_id` *do* reach the engine: `endorsement_prompt` inserts them at `:1097-1102`, the log's first exclusion input shows `Provenance of this claim: credible (source id: https://www.weatherstreet.com/...)`, and echo texts reproduce the field. So "source_status never reaches the prompt" is false.

But the *credibility check the instruction demands* is unsatisfiable: the displayed allow-list is `sorted(CREDIBLE_SOURCES)[:8]` (`:1097`), which sorts to eight opaque `http:///go/v1/...` links. Of **95** distinct gap `source_id` URLs, **0** appear in that 8-URL list; of **77** source ids echoed in exclusion responses, **0** appear there. Even a perfectly compliant engine told "endorse only if its provenance is one of the trusted sources listed" would have to abstain on every credible gap claim. This is an **inference-backed design defect** for the counterfactual compliant responder; the observed run never got far enough to exercise it.

### Rank 3 — (b) "engine genuinely ignores source-status instructions": **not established by this run. **E/I.**

The engine's dominant behaviour is a mirror/chatter/associative-completion route, which is *instruction-independent*: it produced no decision word in either arm, including the inclusion arm where compliance would have been easy. That is consistent with (b) behaviourally, but because the parser never saw a decision, the measured I/E/R cannot discriminate "refused the instruction" from "answered in an unparsed format". Treat (b) as **unresolved**, not confirmed.

### Why the old instrument saturated while the new one collapsed. **E+I.**

`wfa_valid=1.00` is computed from frozen-provider entailment and flag checks (`:688-702`, `:911-931`), not from engine output; it saturates by construction and is G6-FLAGGED, not PASSed. The new arms are the first to touch the engine's actual answer path, and that path is the bottleneck. The apparent "180° reversal" is a change in *what is measured*, not a change in the system.

### Algebra is fine; the quantities are not. **E.**

`R=I−E` and `F=E/(1−R)` (`:1274-1278`) are computed correctly from the recorded flags. But `I` and `E` are echo-classification rates, so `R=0.0154/0.0746` and `F=0` are not recollection/familiarity estimates. The design doc's line 235 ("F ≈ 0 only because E = 0 makes the estimate degenerate") is correct but incomplete: **both** I and E are invalid observations. `exclusion_margin=0.0333/0.0000` is the difference of two echo-keyword rates. `PD_wellformed` passing (`:1336-1338`) therefore validates arithmetic, not the instrument.

---

## 3. Brain-mechanism mapping (citations verified in `docs/investigations/WFA_PROCESS_DISSOCIATION.md:48-68`)

| Mechanism | Intended operationalisation | What the failing run actually showed | Reading |
|---|---|---|---|
| Johnson, Hashtroudi & Lindsay 1993, source monitoring; item vs source memory dissociate (`10.1037/0033-2909.114.1.3`) | Endorsement split by credible vs discredited provenance at fixed facts | The engine reproduced prior utterances ("you mentioned: …") without any source judgment; the parser treated words *inside the reproduced instruction* as the engine's attribution | A noetic echo was scored as an autonoetic source judgment — the exact confound source-monitoring theory warns against. No item-memory/source-memory dissociation was measured. |
| Tulving 1985, noetic "know" vs autonoetic "remember" (`10.1037/h0080017`) | Inclusion may run on familiarity; exclusion requires recollected credible source | Neither arm emitted a recollected source (`pd_inclusion_source` null throughout); `pd_inclusion_attributed_credible` false | Only "know"-like echo/completion behaviour is observable; instrument cannot yet test "remember". |
| Jacoby 1991, `I=R+F(1−R)`, `E=F(1−R)`, `R=I−E` (`10.1016/0749-596X(91)90025-F`) | Two instructions, same items, endorsement probabilities | `I`/`E` are regex hits on quoted prompts; `E=0` because `ABSTAIN` is in the echoed exclusion instruction | Algebra held; the process assumptions failed. R/F are well-formed numbers, not process estimates. |
| Hasher, Goldstein & Toppino 1977, illusory truth (`10.1016/S0022-5371(77)80012-1`) | Predict `E>0` on familiar discredited decoys | `E=0`, but no endorsement decisions existed | Does **not** falsify the mechanism; the observation is uninformative (matches design-doc prediction "E=0 and I_dis=0 … no leak to measure"). |
| Koenig et al., *Open Mind* 2026 (`10.1162/opmi_a_00038`) | Cue source unreliable → WITHDRAW | All 12 "withdrawals" are echoes of the experimenter's own `unreliable`/`ABSTAIN` cue words | A cue-echo artifact was scored as retroactive trust revision — precisely the extrapolation the design ledger (`WFA_PROCESS_DISSOCIATION.md:164-178`) forbids. Cue was experimenter-supplied; no spontaneous detection. |

**One-directional rule maintained:** the literature selected the contrasts; the broken numbers are not being used to redefine the mechanisms.

---

## 4. Ranked, gate-preserving fixes (no threshold changes)

Each fix preserves every existing gate and adds only stricter measurement or additive counters.

1. **Echo-immune, fail-closed parser + offline re-score of the surviving probes (no engine run).** Strip quoted frames (`you mentioned:` / `i said:` / `earlier i said:`), parse only the final decision span (`Decision:` / after the cue), and require an explicit decision token; classify anything else as `unparseable` and add an additive hard-fail counter (`pd_unparseable_rate`). Remove the all-tokens *content* fallback for echoed text. Re-running the parser over the existing `pd_*_text`/`inv_after` needs no engine.
   - **Prediction:** all 12 withdrawal flags and all 3 endorsement flags become `unparseable`; echo-driven `I`/`E`/`margin` collapse to 0 with `unparseable_rate > 0.90`.
   - **Falsifier:** ≥10% of responses contain an explicit decision token outside a quoted echo — then the parser, not the engine route, was the main loss and this fix restores signal.

2. **Turn isolation: one fresh engine (or cleared episodic/recall context) per arm/turn, with per-item texts still persisted.** The manipulation must be the instruction alone; currently prior arms leak into later turns through `process_turn`'s conversation state. Additive check: `echo_marker_rate` per arm.
   - **Prediction:** `echo_marker_rate` falls from ~0.92 to <0.10 and explicit decision tokens appear in >0.50 of turns; R becomes a real I−E.
   - **Falsifier:** echo rate stays >0.50 on a fresh engine → the mirror is the default router response to the `Claim:…/Decision:` prompt shape, not context carryover; proceed to fix 4.

3. **Make the trusted set self-consistent without loosening the gate.** Put the item's own `source_id` into the allow-list shown to the engine (while keeping `fabricated`/`discredited` ids excluded, so `G2`/`instrument_controls` still hold), or pass an explicit per-claim credibility token instead of asking for set membership against a truncated list.
   - **Prediction:** under a compliant responder, `E_credible` rises from ~0.015 to ≥0.70 while `E` on decoys stays low; capability parity becomes testable.
   - **Falsifier:** with echoes eliminated (fixes 1–2), `E_credible` remains <0.70 → the engine cannot perform the provenance lookup even with a satisfiable list; the deficit is instruction-following, not list construction.

4. **Constrain the decision surface only if the interface permits: single-token forced choice, parse the first token, keep a separate "no answer" category.** This is a new probe arm; the old gates and PD math stay unchanged, and the token arm must be A/B'd on identical state.
   - **Prediction:** explicit-response rate >0.80; `withdrawal` on a correctly cued item rises above 0.50 without any parser keyword borrowed from the cue.
   - **Falsifier:** token arm still yields <0.50 explicit decisions → the failure is upstream of articulation (no source-evaluation routine exists), which is a mechanism work item, not a measurement item.

Explicitly **not** proposed: lowering `R≥0.25`, `margin≥0.10`, `E_credible≥0.70`, `withdrawal≥0.50`, or redefining `PD_wellformed`.

---

## 5. Evidence that would discriminate the remaining hypotheses (run by parent)

Already available and used (see §0): the worktree `scratch/learn5_wfa_probes.jsonl` settles (a) vs (c) — echo artifact confirmed, `source_status` present. What is still needed:

1. **Route instrumentation (discriminates (d) mirror route vs (d) parametric-completion route):** log, per PD turn, which handler emitted the response (`_reconstruct_gist` at `engine_memory.py:2230`, targeted recall at `engine.py:6003`, own-speech recall at `engine.py:7257`, or generative fallback at `engine.py:8049`) plus input/output at each stage. Prediction if mirror route: >50% episodic-recall emissions; if parametric: mostly generative fallback.
2. **Fresh-engine vs shared-engine A/B on identical items and seed**, requiring per-item `pd_*_text` output in both arms. This isolates cross-turn contamination from a prompt-shape router response (fix 2's falsifier).
3. **Oracle compliance control:** inject a fake responder that returns explicit `ENDORSE`/`ABSTAIN`/`WITHDRAW` per ground truth and verify the parser + `process_dissociation` reproduce known I/E/R/margin/withdrawal. This tells whether any residual bug is in scoring rather than the engine.
4. **Trusted-list counterfactual:** same items, allow-list containing each claim's own URL (discredited ids still excluded). Isolates the §Rank-2 list-construction defect from general instruction-following failure.
5. **Echo-stripped offline re-score of the existing probes** (no engine process): if explicit decisions are zero after stripping, hypothesis (b) remains untestable from this run by definition and needs fix 4.
6. **Preserve and publish per-item texts in the summary log** so future failures are auditable; the summary JSON should carry `echo_marker_rate`, `unparseable_rate`, and `own_decision_rate` as additive counters.

---

## 6. Bottom line

The rebuilt instrument is falsifiable, and it failed honestly — but **the failure is a measurement failure first**: ~92% of arm responses are the engine echoing prior turns, and the parsers scored keywords from those echoes (`ABSTAIN`, `unreliable`) as if they were the engine's decisions. All 12 withdrawals and effectively all endorsements are artifacts. `source_status` did reach the prompt, but the displayed trusted list excluded every claim's actual source, so the exclusion instruction was unsatisfiable even for a compliant responder. `R`, `F`, `margin` and `withdrawal` are algebraically correct and evidentially void. No threshold was changed; the repairs are an echo-immune parser, turn isolation, a self-consistent trusted set, and (only if needed) a forced-choice response surface — each with a prediction and a falsifier above.
