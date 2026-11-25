# Autonomous source monitoring — can RAVANA retrieve its own provenance at answer time?

**Branch:** `goal/autonomous-source-monitoring` (from `b75a24e`).
**Status:** SCRATCH experiment + doc only. **No production file was edited. No live engine was booted.**
**Verdict (lead):** A *read-side provenance walk* is a legitimate, brain-grounded mechanism that closes the measured "engine never reads its own source memory" gap — **but the credibility judgment it feeds is a rule-table scaffold (G6)**. **DO NOT IMPLEMENT as a capability claim.** If the parent wants the walk as a default-off instrument, the plan in §7 is the narrow path; it must be reported as a saturated mechanism implementation, not autonomous source recollection.

---

## 1. The exact gap being attacked

Known state inherited from the branch:

| Artifact | Measured fact |
|---|---|
| `WFA_ECHO_IMMUNE_RESULT.md` | `forced_choice_explicit_rate = 0.00`, `pd_unparseable_rate = 1.00`, forced-choice echo `0.000` → pre-registered falsifier fired: the gap is **upstream of articulation**, there is no decision routine. |
| `SOURCE_EVALUATION_WIRING.md` | `RAVANA_SOURCE_EVAL=1` + `engine.set_source_context_for_turn(claim, ClaimContext)` makes the emission gate emit a decision, but the context is **staged by the caller**. |
| `WFA_INTERNAL_SOURCE_DESIGN.md` | Same arm with context derived from **graph provenance** is non-tautological in the sense that item answer labels cannot move it; reported as 15/15 focused in the parent's isolated confirmation. Still **harness-staged**: the harness supplies the claim string, the origin filter, and the prior decision. |

Code inspection of the real wiring (`ravana/src/ravana/chat/engine.py:8090-8153`) locates the missing input exactly:

- `_source_monitor_decision(ctx)` returns `None` unless `ctx.source_context` is populated.
- The **only** writer of that field is the harness setter `set_source_context_for_turn()` (copied into `ctx` at `engine.py:11786`).
- Therefore, with no harness call, the gate is inert on **every** turn, and a source question produces 0 explicit decisions regardless of what is stored in the graph.

There is **no engine-owned read path** from a prompt to its own `edge.source_metadata`. That is the gap this experiment closes in scratch.

A second, deeper gap was found by reading the real write paths:

- `web_to_graph.py:308-315` stores `{source, relation, edge_kind:"web_fact", context, contexts}`.
- `learn/intake.py:135-145` (`Provenance.as_metadata`) stores `{source, source_type, url, ingested_at, extractor, operations, corroboration}`.
- `measure_learn5_wfa.py:1982-1986` (`_write_asserted_wfa_fact`) stores `{source, url, edge_kind:"curated", source_type:"seed", source_status:"", explicit_assertion:True}`.

**No real writer stores a credibility token, a trust class, or a per-claim decision.** The evaluator's `SourceStatus.has_explicit_credibility()` requires a token from `{credible, trusted, verified, reliable}`; that token does not exist in the engine's own store. Any read-side probe must *manufacture* it from `edge_kind`. That is the scaffold.

---

## 2. Brain mechanisms and why they pick this design

Literature was verified against publisher/PubMed/DOI records during this change. Support vs extrapolation is explicit in §3.

1. **Source memory is not item memory.** Source judgments are a separate, explicit evaluation based on *qualities of experience* (perceptual + reflective), with deliberativeness varying; item familiarity cannot stand in for an origin. *Johnson, Hashtroudi & Lindsay (1993), Psychological Bulletin 114(1), 3–28, doi:10.1037/0033-2909.114.1.3.* Noetic "know" vs autonoetic "remember": *Tulving (1985), Canadian Psychology 26(1), 1–12, doi:10.1037/h0080017.* This is why V3 (familiarity-only) is a control, not a candidate.
2. **Recollection and familiarity dissociate and have partially distinct substrates.** Recollection is a threshold-like retrieval process relying on hippocampus + PFC; familiarity is a signal-detection process relying on surrounding regions. *Yonelinas (2002), Journal of Memory and Language 46(3), 441–517, doi:10.1006/jmla.2002.2864.* Left PFC supports systematic monitoring of specific source information while right PFC supports heuristic familiarity/recency; *Mitchell & Johnson (2009), Psychological Bulletin 135(4), 638–677, doi:10.1037/a0015849*; specific perceptual details at retrieval: *Ranganath, Johnson & D'Esposito (2000), J. Neuroscience 20(22), RC108, doi:10.1523/JNEUROSCI.20-22-j0005.2000.* → the probe must read a *source-specific field* (`edge_kind`/`source`), never a numeric strength (`confidence`/`reliability`).
3. **Cue overlap enables selection *before* recollection.** ERP evidence: when retrieval cues overlap more with a targeted trace than with competitors, the left-parietal recollection effect is selective; selection can act before recollection given sufficient overlap. *Moccia & Morcom (2022), Cognitive, Affective, & Behavioral Neuroscience 22(3), 492–508, doi:10.3758/s13415-021-00971-0.* → the engine's claim content words are the retrieval cue; the walk selects stored edges by label overlap instead of being handed a `source_id`.
4. **Prospective source-monitoring cues.** In the multiprocess framework, a *focal* cue can trigger spontaneous retrieval of an intention without strategic monitoring; nonfocal cues require top-down monitoring. *McDaniel & Einstein (2000)*; *Einstein, McDaniel, Thomas, Mayfield, Shank, Morrisette & Breneiser (2005), J. Exp. Psychol. Gen. 134(3), 327–342, doi:10.1037/0096-3445.134.3.327*; reviewed in *PMC4500919 / PMC4868976*. → the probe is triggered by the claim embedded in the question (focal cue), not by harness bookkeeping.
5. **Controlled vs automatic contribution.** Inclusion/exclusion decomposition: `R = I − E`, `F = E/(1−R)`. *Jacoby (1991), JML 30(5), 513–541, doi:10.1016/0749-596X(91)90025-F.* Repetition breeds belief without provenance: *Hasher, Goldstein & Toppino (1977), JVLVB 16(1), 107–112, doi:10.1016/S0022-5371(77)80012-1.*
6. **Trust is calibrated on the communicator, not the content's fluency.** *Sperber, Clément, Heintz, Mascaro, Mercier, Origgi & Wilson (2010), Mind & Language 25(4), 359–393, doi:10.1111/j.1468-0017.2010.01394.x.* → the read side must key on stored `source`, not on wording.
7. **Source invalidation causes reevaluation even in toddlers.** *Dautriche, Goupil, Smith & Rabagliati (2021), Open Mind 5, 1–19, doi:10.1162/opmi_a_00038, PMID 34485794.* → withdrawal rule; the invalidation cue is experimenter-delivered and stored in metadata, which is an **extrapolation/proxy**, not autonomous detection.

---

## 3. Support vs extrapolation ledger

| Claim | Status | Boundary |
|---|---|---|
| A claim's content words can cue retrieval of a source trace before a decision | **SUPPORTED** (direct analogue) | [3], [4]; implemented as label-overlap walk. |
| A source-specific stored field, not item strength, must drive trust | **SUPPORTED** | [1], [2], [5]; V3 is the negative control. |
| Reading the engine's *own* provenance is a legitimate instantiation of source recollection | **SUPPORTED as method, NOT a brain fact** | [1], [6] motivate source-specific attribution; the graph-read implementation is engineering. |
| The credibility token derived from `edge_kind` is a recollected judgment | **NOT SUPPORTED — SCAFFOLD** | The token is synthesized by the probe from a write-time tag. This is the G6 hazard. |
| WITHDRAW after a stored invalidation tag | **SUPPORTED WITH EXTRAPOLATION** | [7] shows behavioural reevaluation in 30-month-olds, not a verbal token; cue is delivered/stored by the harness. |
| V2's "prior endorsement from discredited tag" | **NOT SUPPORTED** | Pure rule; no memory of a prior decision exists in the engine. |
| Any result here is human-like autonomous source monitoring | **NOT CLAIMED** | No live engine, no learned credibility signal, no per-claim decision memory. |

---

## 4. Scratch experiment

**Files:** `scratch/autonomous_source_monitoring.py`, `scratch/autonomous_source_monitoring_results.json`, `scratch/autonomous_source_monitoring_run.log`.

### 4.1 What is real and what is fake

- **Real, unmodified:** `ravana.chat.source_evaluation` (`evaluate_source`, `ClaimContext`, `SourceStatus`, `TrustClass`, `Decision`) is imported from `ravana/src` and never monkeypatched. The decision rules and tokens are production code.
- **Fake:** a minimal `FakeGraph` with `.nodes`, `.edges`, `.get_node` and `edge.source_metadata`. Metadata field names/values are copied from the real writers listed in §1 — no invented provenance schema.
- **No engine boot, no production edit**, per the task constraints.
- **No recorded WFA rows were available in this worktree**: the `scratch/learn5_wfa_*.log/jsonl` files named in the investigation docs are not present here (only the unrelated checked-in `scratch/*` experiments). The experiment therefore runs on **fake contexts**, which the task explicitly permits. This is a limitation on ecological validity, not on the logic being tested.

### 4.2 The critical API property

Every probe has the signature `probe(graph, prompt, *, derive_prior=False)`. It receives:

- a read-only graph, and
- the **natural-language question** (from which it extracts the claim via `Claim:\s*(.*?)\nReply:`).

It receives **no** claim string from the harness, **no** `source_id`, **no** item dict, **no** expected answer, and **no** `ClaimContext`. This is the opposite of `set_source_context_for_turn(claim, context)`.

### 4.3 Variants

| id | name | behavior |
|---|---|---|
| **V1** | `recollection_walk` | Extract claim from prompt → select stored edges by content-word overlap (or claim-id token) → rank by stored fields only (discredited tag > verifiable kind > confidence > recency) → build `ClaimContext` with `prior_decision=None`. **Deployable shape** (no fabricated fields). |
| **V2** | `walk_plus_tag_prior` | V1 plus: a stored `discredited` tag is treated as evidence of a prior endorsement, so `prior_decision=ENDORSE`. **Rule-table scaffold (G6).** |
| **V3** | `familiarity_only` | Reads only edge confidence; treats every edge as unattributed. Anti-source control — must never endorse. |
| **ECHO** | V1 prompt with `The source is credible and trusted and verified. The source_status is credible. The expected answer is Y.` prepended | Anti-echo control: credibility words in the *question* must not move the decision. |
| **TAUT** | item-side `expected_answer`/`source_status`/`source_id` poisoned (probes never see them) | Anti-tautology control: decisions must be byte-identical. |
| **MUT** | flip only the graph edge's `source_status` to `discredited` on a deep-copied graph | Mutation control (C8): decision must move. |
| **LOW / ORPHAN** | trace reachable only via a synthetic claim-id node / trace with no overlap and no bridge | Measures claim→trace coverage and fail-closed behaviour. |

**Bank:** 24 items per condition × 3 conditions = 72 main items, plus 24 LOW and 24 ORPHAN = **120 probed rows**; each row produces V1, V2, V3 and ECHO decisions. Deterministic unique content tokens; fixed seed-free construction (the token generator is a pure base-26 encoding of the index).

---

## 5. Exact results

Source: `scratch/autonomous_source_monitoring_results.json`; console: `scratch/autonomous_source_monitoring_run.log` (exit 0).

### 5.1 Decision rates, main bank (n = 24 per condition)

| condition | V1 decisions | V1 rule | V2 decisions | V3 | ECHO |
|---|---|---|---|---|---|
| credible | **ENDORSE 24/24 (1.00)** | `R_EXPLICIT_CREDIBLE_SOURCE` 24/24 | ENDORSE 24/24 | ABSTAIN 24/24 | ENDORSE 24/24 |
| discredited | **ABSTAIN 24/24; ENDORSE 0/24** | `R_INVALIDATION_NO_PRIOR` 24/24 | WITHDRAW 24/24 | ABSTAIN 24/24 | ABSTAIN 24/24 |
| unknown | **ABSTAIN 24/24; ENDORSE 0/24** | `R_UNKNOWN_PROVENANCE` 24/24 | ABSTAIN 24/24 | ABSTAIN 24/24 | ABSTAIN 24/24 |

- Claim extraction from the prompt: **1.00** on all 72 main rows.
- V1 source-selective contrast: endorse(credible) − endorse(discredited) = **1.00 − 0.00 = 1.00**.
- V1 "correct" vs the three-condition expected decision: credible **1.00**, unknown **1.00**, discredited **0.00** (it abstains instead of withdrawing — see §6, missing input #3).
- V2 correct: credible **1.00**, discredited **1.00**, unknown **1.00** — **saturated by construction (G6)**.

### 5.2 Process-dissociation reading (V1)

With exclusion defined as the discredited condition: `I = 1.00`, `E = 0.00`, so `R = I − E = 1.00` and `F = E/(1−R)` is undefined (degenerate). Per the standing rule, this is **not** evidence of human-like recollection: it is a deterministic rule table over clean stored tags. **Do not tune thresholds to manufacture a mid-range R.**

### 5.3 Controls

| control | measured | reading |
|---|---|---|
| ECHO identical to V1 (all 120 rows) | **1.00** | Credibility words in the question never move a decision. |
| Unknown claims endorsed under ECHO | **0/24** | No answer-label leakage from the prompt. |
| V3 familiarity-only endorsements (main bank) | **0/72** | Item strength/stored confidence cannot substitute for source — matches [1], [2], [5]. |
| TAUT identical to V1 | **1.00** | Poisoning item answer labels changes nothing; context is graph-derived. (The first run reported 0.667 because the mutation control mutated graphs in place *before* this check; fixed by deep-copying. The corrected number is 1.00.) |
| MUT: credible → flip stored `source_status` to `discredited` | V1 flips to ABSTAIN **24/24 (1.00)**; V2 flips to WITHDRAW **24/24 (1.00)** | The decision is driven by stored provenance, not by the harness's labels. |
| LOW (claim-id bridge authored in the graph) | retrieval mode `claim_token` **24/24**, ENDORSE **24/24** | A bridge makes low-overlap claims reachable. |
| ORPHAN (no overlap, no bridge) | `no_stored_trace` **24/24**, ABSTAIN **24/24**, ENDORSE **0/24** | Strict fail-closed: no trace → source amnesia → abstain. |

### 5.4 Instrument validation (skill §1: print input and output, hand-check)

Sampled from the saved rows:

- credible: input claim `The toka matches the toksg pattern.`; audit = `{retrieval: lexical_overlap, n_matching_edges: 1, edge_kind: curated, status: "", prior: null}`; output V1 `ENDORSE`, rule `R_EXPLICIT_CREDIBLE_SOURCE`. Pass criterion = "a stored verifiable origin with no invalidation is endorsed" — matches.
- discredited: `The tokcw matches the tokwc pattern.`; `{lexical_overlap, 1, curated, "discredited", prior: null}` → V1 `ABSTAIN`, rule `R_INVALIDATION_NO_PRIOR`; V2 `WITHDRAW`. Pass criterion = "invalidation without a recollected prior endorsement cannot withdraw" — matches.
- unknown: `The tokgs matches the tokzy pattern.`; `{lexical_overlap, 1, co_occurrence, "", null}` → `ABSTAIN`, rule `R_UNKNOWN_PROVENANCE` — matches.
- orphan: `The tokalm matches the tokbes pattern.`; `{no_stored_trace, 0}` → `ABSTAIN` — matches fail-closed.

The instrument measures what it claims: a read-side walk over stored provenance, not a keyword count of the prompt.

---

## 6. Verdict: genuine capability or rule-table scaffold?

**Both, in different layers — and the capability layer is not enough.**

- **Genuine (mechanism):** the claim is recovered from the engine's own input and mapped to a stored trace by cue overlap; the trust-relevant fields are read from the engine's own `source_metadata`; the decision follows a mutation of that stored metadata and is invariant to item answer labels and to credibility words in the question. This closes the specific measured defect "no engine-owned read path to its provenance".
- **Scaffold (G6):** the token the evaluator demands (`credibility_token="credible"`) does not exist in the engine's store — the probe *manufactures* it from `edge_kind ∈ {curated, web_fact}`. V2 likewise *manufactures* a prior endorsement from the invalidation tag. A deterministic rule over clean tags gives 1.00 by construction. **Call it G6 — saturated / mechanism implementation — on any live run that reproduces these rates.**

### Missing inputs/mechanisms, precisely

1. **No engine-owned read path.** `engine.py:8090-8153` gates only on caller-staged `ctx.source_context`. This is the gap the scratch walk closes, and it is implementable.
2. **No stored credibility attestation.** No real writer stores `credibility_token` / `trust_class`. The engine has no record that a source was *judged* credible — only a write-time `edge_kind`. A genuine source-recollection capability needs a stored, revisable source-status record produced by the engine's own epistemic history, not a tag translated into a token by the probe.
3. **No per-claim decision memory.** `_last_source_evaluation` is a single global slot, not keyed by claim. Withdrawal after invalidation therefore either needs harness bookkeeping (staged arm) or the fabricated prior of V2. V1's 0/24 discredited "correct" is the honest signature of this missing input. A real withdrawal capability needs a claim-keyed audit store: "did I previously endorse *this* claim?"
4. **Claim→trace coverage depends on a write-side bridge.** Lexical overlap works on content-rich claims (24/24); a synthetic claim-id bridge works (24/24) but is authored by the harness's `_write_asserted_wfa_fact`; orphan traces fail closed (24/24). The engine's normal ingest does not mint a claim-id node for an arbitrary source question, so deployable coverage cannot be assumed from this fake bank.
5. **Invalidation is experimenter-supplied.** The discredited tag is written into metadata by the harness. Per [7] this is a proxy; it is not the engine detecting that a communicator lied.

---

## 7. Implementation plan (plan only — do not apply)

**DECISION: DO NOT IMPLEMENT into production as a capability.** If the parent wants the walk as an instrument to locate the next gap, implement it **default-off** as follows; no threshold, gate, item bank, or hash seal changes.

1. Add `engine._autonomous_source_context_from_graph(user_input)` near `process_turn` (above all ~66 early returns), mirroring the scratch `probe()` read half: extract claim → `read_provenance`-equivalent over `self.graph` → rank by `source_metadata` only → build `ClaimContext` → stage into `ctx.source_context`. Reuse the existing `_source_monitor_decision` / `_apply_source_monitoring_gate`; **add no new decision rule**.
2. New flag `RAVANA_SOURCE_EVAL_AUTONOMOUS=1`, read at call time. With it off, byte-identical to today. Prove flag equivalence with a differential probe before any A/B.
3. Fail-closed: no matching trace, malformed metadata, or any exception → `ClaimContext(source=None)` → `ABSTAIN`/`N`; never fall through to parametric endorsement.
4. Never read item metadata, expected answers, `source_status` from the harness, or caller-supplied `source_id`. Expose an audit with `n_matching_edges`, `retrieval` mode, `edge_kind`, `status`, `rule`.
5. **Do not port V2's derived prior.** Withdrawal must wait for a claim-keyed decision store (#3). Until then, report discredited items as ABSTAIN, not WITHDRAW.
6. Gate every future live run on: C1–C8 from `WFA_INTERNAL_SOURCE_DESIGN.md`, echo-identical control, familiarity-only control, provenance mutation control, primary + sealed pools, and an explicit **G6 FLAG** if any endpoint saturates near 1.00. Report `0 < R < 0.999` informativeness where defined; never tune thresholds to fabricate it.
7. The real capability work is upstream: give the engine a stored, revisable source-status record it writes itself (#2) and a claim-keyed decision memory (#3). Those are production changes requiring their own worktree, brain-grounded design and A/B; they are **not** authorised by this doc.

---

## 8. Reproduction

```
cd C:/Users/Likhith/Documents/projects/ravana_goal_source
uv run python scratch/autonomous_source_monitoring.py
# -> scratch/autonomous_source_monitoring_results.json
# -> scratch/autonomous_source_monitoring_run.log
```

No engine boot, no network, no store writes. The only production import is the pure `ravana.chat.source_evaluation` module.
