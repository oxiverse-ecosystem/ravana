# CAR root-cause investigation — cold-reload loss, answer reach, monotonicity reversal

**Status:** final (static analysis + per-item audit + verified literature). **Method:** READ-ONLY. No engine, pytest, or measurement
script was run; no source, scratch, or weights file was modified. This document is the
only artifact owned by this task. Evidence sources: `scratch/learn5_car_measured.json`
(commit `eb1af7f`, 2026-11-20T08:56Z, 120 records — **pre-527f401**),
`scratch/car_full_v2.log`, `scratch/car_answer_reach_probe.log`,
`scratch/car_coldreload_probe.log`, `scratch/car_cold_inspect2.py`,
`docs/CAR_ANSWER_REACH_CODEMAP.md`, `docs/LEARN5_CAR_MEASURED.md`, and source at
HEAD `d7a3495` (branch `docs/replay-fix-negative`, includes `527f401`).

**Instrument caveat (skill §1):** the audit JSON predates the answer-reach fix
`527f401`. Its `answer_verdict` fields therefore describe the OLD pipeline; the
post-fix 0.40 reach is known only from the commit message's A/B. Per-item
classification below separates what the JSON proves from what is inferred from code.
Also: the JSON's primary k=2 immediate rate is 0.70 (Δ+0.50), while
`LEARN5_CAR_MEASURED.md`'s table reports 0.80 (Δ+0.60, cited to `43db6cd`). The
per-item JSON is treated as authoritative here; the discrepancy is noted, not smoothed.

---

## 1. Root causes with file:line evidence

### RC-1 — Cold-reload retention loss

#### 1.1 The save/load path, mapped

- `save()` — `ravana/src/ravana/chat/engine.py:12856`. Pickles `self.graph` directly
  (`state['graph']`, :12925) plus `_concept_keywords` (:12926). No pruning or
  renormalization happens inside `save()` itself.
- `ConceptGraph.__getstate__` — `ravana_ml/src/ravana_ml/graph.py:1251-1268`: drops
  only `_lock`, `_fact_encode_hook`, `dual_code`. Nodes, edges, ids, `next_id` all
  persist. `__setstate__` (:1270-1328) rebuilds indices and re-attaches
  `edge.parent_graph`. **No node-id remapping and no node dedup at load.**
- `ConceptEdge.__getstate__` — `graph.py:304-328`: `source_metadata` **is kept**
  (the historical provenance-drop defect is fixed; the docstring records it).
  `__setstate__` (:471-499) only backfills missing attributes. **Provenance fields
  are pickled correctly today.**
- `_load()` — `engine.py:13211-13400+`: restores graph object wholesale (:13282),
  restores `_concept_keywords` from the snapshot (:13325). Off-dim node repair
  (:13261-13281) can drop nodes only when >50% are off-dim (not triggered here:
  no `[Load warn]` lines in `car_full_v2.log`).
- Post-load mutation sequence — `engine.py:2144-2199` (the persisted-weights early
  return, skill §3): `_revector_existing_nodes()` (:2147) → `_sanitize_graph()`
  (:2148) → `_bootstrap_domain_concepts()` (:2149) → decoder restore →
  `_typed_edges_bootstrap()` (:2188) → `_seed_curriculum_causal()` (:2194) →
  `return` (:2199).
  - `_revector_existing_nodes` (`engine_graph.py:551-578`) only replaces node
    vectors with GloVe projections; ids and edges untouched.
  - `_typed_edges_bootstrap` (`engine_graph.py:1001-1043`) is idempotent: returns 0
    when typed edges already exist (:1027-1032). Not a drop point on a loaded graph.

#### 1.2 Drop point A (line-pinned): load-time curriculum re-seed overwrites conversation provenance

`_seed_curriculum_causal` — `engine_graph.py:887-943`, called on the LOAD path at
`engine.py:2194`. For every authored curriculum hop it calls
`_ensure_relation(src, tgt, "causal", 0.8)` (:915), then:

```python
kind = md.get("edge_kind")
if kind not in (None, "curated"):
    continue  # never downgrade a web/conversation edge   ← engine_graph.py:931-932
md.update({"edge_kind": "curriculum",
           "source": "curriculum:authored:" + subject,
           "evidence": ..., "acquired_by": "authored_scaffold"})   # :933-938
```

**The guard is wrong about its own taxonomy.** Conversational teaching stamps
`edge_kind="curated"`: `ravana/src/ravana/learn/intake.py:80-81` maps
`"user" → "curated"`, `"seed" → "curated"`, and every taught hop in the audit JSON
carries `edge_kind='curated'`. So the comment "never downgrade a web/conversation
edge" exempts `web_fact` but **admits conversation edges for overwrite**: on every
cold load, any taught edge whose (src,tgt) pair coincides with a curriculum hop is
relabeled `curriculum` and its `source`/`evidence`/`acquired_by` (the episode
provenance) is replaced with the authored-scaffold stamp.

Measured consequence (audit JSON, `rust`): the taught hop `oxidation→rust` is
`curated, w=0.168, c=0.702` immediately after teaching, but
`curriculum, w=0.79, c=0.297` at cold reload. The kind flip is exactly this code
path. (The weight/confidence change is NOT produced by :933-938, which touches
metadata only — see Drop point C; `_ensure_relation` creates edges at w=0.8/c=0.9
only when absent, :977-982, and backfills kind only when absent, :995-999.)

Both `curated` and `curriculum` are trusted by `_hop_provenance`
(`learn/explanation.py:292-293`), so this overwrite alone does not make a chain
unassertable — but it (a) destroys the record of *which episode* taught the edge,
defeating any future episode-conditional consolidation, and (b) is the mechanism
by which load-time re-seeding silently edits learned state. It is a trust-gate
provenance defect in the skill §6 sense: the maintenance routine changes a stored
item without preserving the fields that identify its origin.

#### 1.3 Drop point B (line-pinned): `_sanitize_graph` deletes phrase nodes and ALL incident edges at load

`chain_walker.py:1314-1363`, called at `engine.py:2148`. Removes self-loops
(:1331-1333) and every node whose label passes `_is_question_phrase` (:1337-1339),
**together with every incident edge in both directions** (:1346-1348), then clears
keyword/label bookkeeping (:1350-1357). Measured: `car_full_v2.log:11352` —
`[Sanitize] Pruned 8 question-phrase nodes and any self-loops from loaded weights.`

Teaching sentences are grounded as whole phrases (`car_answer_reach_probe.log:882`:
`query_phrase='amber happens because resin causes fossil, and fossil causes amber'`),
and the audit's hop census shows literal surface nodes `causes` and `happens`
carrying hundreds of causal edges (`dew→causes→flood`, `dew→cooling→happens`, …).
Any taught hop routed through a phrase/surface node is deleted at cold load with no
trace. This is a concrete, line-pinned candidate for the amber/magnify/stalactite
losses; confirming which of the 8 pruned nodes carried taught hops requires a
read-only inspection of `scratch/learn5_car_data/ravana_weights.pkl` (proposed as a
verification step for the execution-owning agent, §4).

#### 1.4 Drop point C (measured effect, exact line not yet pinned): load-time value drift

`scratch/car_coldreload_probe.log:20,48`: `marble` taught hops
`w=0.5, c=0.423/0.407` in-process → `w=0.5, c=0.475/0.456` after save+reload
(+0.052/+0.049), kind preserved `curated`; `callus` hops unchanged
(:42,:50). Audit JSON `rust`: `w 0.168→0.79, c 0.702→0.297` across reload.
So reload does not merely restore values: some edges gain confidence, others are
re-valued entirely. `save()` and the pickle hooks do not do this; the remaining
candidates are the unread tail of `_load()` (engine.py:13400+), a reconsolidation
boost, or re-seeding resolving a *different* node id for the same label
(`_concept_keywords[src][0]`, `engine_graph.py:916-922` — first-id resolution can
pick the curriculum node instead of the taught surface node when a label has two
ids). **This sub-question is open; it is flagged, not guessed.**

#### 1.5 Drop point D (line-pinned): trust-blind chain scoring makes retention budget-sensitive

`score_chain` — `explanation.py:147` (docstring :22):
`sum(log(w*c)) − 0.35·len − 0.25·unprovenanced`. `enumerate_chains` sorts by this
score and truncates BEFORE grading (`explanation.py:261-265`). A taught edge at
w=0.25, c=0.39 contributes log(0.0975)≈−2.33 per hop; an untrusted co-occurrence
edge at w=0.8, c=0.9 contributes log(0.72)≈−0.33 and pays only −0.25. **Untrusted
high-weight chains systematically outrank trusted low-weight taught chains.**
Measured consequences:

- `honey`: trusted chain absent at limit 64, present at 256 after reload
  (`LEARN5_CAR_MEASURED.md:112-115`; `car_cold_inspect2.py`).
- `lightning`: immediate probe raw=64, graded_assert=0 (taught chain below the
  cut); cold probe finds it (audit JSON).
- Rank drift across turns: `callus` 0→41 after a single `why` turn, `tornado`
  21→3→absent (`car_reach_drift_probe.log`, codemap R2).
- Secular growth: 631/640 enumerated chains trust-rejected at cold for primary k=8
  (`LEARN5_CAR_MEASURED.md:58-62`).

This is why "retention" in CAR is partly a search-budget artifact: the edges may
survive while the chain becomes unreachable at the fixed limit.

#### 1.6 Per-item traces (audit JSON, primary k=2 and controls)

| item | baseline | immediate | cold | diagnosis |
|---|---|---|---|---|
| `amber` | 0 | 1: `mineralization→fossil→amber` (curated 0.5/0.484, 0.421/0.763) | 0, **absent at limit 1024** | The detected immediate chain is a **cross-item composite**, not the taught `resin→fossil→amber`: it borrows the `mineralization→fossil` hop (identical values appear in `fossil`'s own k=4 record). The taught `resin→fossil` hop never surfaced. Cold loss = genuine edge/provenance loss (Drop points A/B/C), not truncation. |
| `magnify` | 0 | 1: `refraction→enlargement→magnify` (curated 0.489/0.536, 0.246/0.892) | 0, absent ≤1024 | Same class: taught chain present in-process, gone after reload. |
| `stalactite` | 0 | 1: `dripping→deposition→stalactite` (curated 0.191/0.719, 0.5/0.512) | 0, absent ≤1024 | Same class. Note very low weight 0.191 — also a rank victim. |
| `honey` | 0 | 1: `heat→evaporation→honey` (curated 0.25/0.39 + **curriculum** hop `evaporation→heat` 0.348/0.761) | 0 at 64, **present at 256** | Composite again (borrowed curriculum hop); cold loss is rank/truncation (Drop point D), possibly plus one-hop value drift. |
| `rust` | 1 (curriculum `oxygen→oxidation→rust`) | 1: taught `moisture→oxidation→rust` | 1, but hop `oxidation→rust` now `curriculum 0.79/0.297` | Retained, but **provenance overwritten at load** (Drop point A, measured). Delta capped by baseline. |
| `earthquake` | 1 (curriculum `plate motion→stress`) | 1 | 1: taught `shift→tremor` (curated 0.85/0.95) | Retained; delta capped by baseline. |
| `lightning` | 0 | 0 (raw 64, assert 0) | 1 | Immediate truncation; cold gain. |
| `hurricane` | 0 | 1 | 1 | Clean retention. |
| `comet` | 0 | 0, **raw_chains=0** | 0, raw=0 | **Acquisition failure**: no causal chain ever existed on the resolved node; the emitted answer `ice leads to vapor, and vapor leads to` (truncated) is an utterance-store echo of the taught sentence, not graph retrieval. |
| `coral` | 0 | 0 (raw 64, assert 0; definition answer) | 0 | Never acquired an assertable chain; definition consult answers instead. |

Additional cold losses outside k=2 (same class): primary k=8 `fruit`, k=4 `aurora`
(immediate chain `emission→glow→aurora`, again a composite borrowing `glow`'s hop),
k=4 `fossil`, sealed k=2 `echo`. **Interference control case:** sealed k=1 `rainbow`
had a baseline curriculum chain `refraction→dispersion→rainbow` (trusted); after
teaching the *same labels*, immediate census shows the curriculum hops gone
(curated:3, assert:0) and cold shows curated:0 — teaching overwrote the curriculum
edges (`add_edge` on an existing pair overwrites weight/confidence,
`graph.py:1699-1714`) and the result no longer asserts. This is genuine
retroactive interference at the edge level, but it is a sealed k=1 item, not part
of the primary k=2 reversal.

### RC-2 — Answer reach 0.40 (post-527f401): which exits still reject

#### 2.1 What 527f401 changed — and did not

Commit `527f401` touches **only** `response_gen.py` (+44/−8) and a new test file.
At HEAD `d7a3495` the changed code is:

- `mechanism_intent()` — `response_gen.py:233-250`: admits
  `why|how|what causes|what makes|explain|mechanism|effects?`; returns `forward`
  only for `effects of` / `what does X cause`; otherwise `backward`.
- `_why_how_causal_chain` — `response_gen.py:8508-8614`: calls
  `mechanism_intent(ctx.raw_input)` (:8527), resolves node seeds (:8560-8578),
  enumerates at **limit=64** (:8579-8582), grades with unchanged
  `grade_chain`/`assertable_chain` (:8583-8585), returns the narration verbatim
  via `_generate_response` (:8661-8664).
- `longest_label_seed()` — `response_gen.py:253-261`: **defined but has ZERO
  callers** (grep-verified). The seed experiment was measured and REVERTED
  (commit message); the helper is dead code.

Measured A/B (commit message): reach ≤0.10 → 0.30–0.50; CAR primary k=8 reach
0.40, sealed 0.40; grounded −2, thinking +10; monotonicity FAIL not waived.

**Not changed:** `engine_self_query.py`. The pre-generation guard
`_mechanism_chain_available` (`engine_self_query.py:2446-2468`) and its admission
at `:2606-2608` still carry the OLD direction regex and `limit=3` (codemap R4).
The skill §4 rule — "one shared intent predicate at every router; a helper-level
guard is useless after an early return" — is therefore only half-applied.

#### 2.2 Early exits before `_why_how_causal_chain` (enumerated)

Pre-generation, in `process_turn`/`_process_turn_impl` (engine.py:8041-8103,
:8131+): self-reference (:8903-8911), structured recall (:8915-8929), fact
reasoning (:8933+), combined-fact (:9026-9034), evaluative (:9045-9053),
`_consult_internal_knowledge` (:9112-9120 → engine_self_query.py:2436), memory
(:9128-9144), abstract/uncertainty (:9338-9379). Inside `_generate_response`
(response_gen.py:8588 pre-fix numbering), `_why_how_causal_chain` runs at the TOP
(:8633-8638) and returns verbatim on success (:8634-8636).

Live-probe evidence (codemap §2, 36 probes): `_generate_response` and
`_why_how_causal_chain` were reached on EVERY paraphrase; no earlier router was
the observed drop point on that state. The action router, creative pre-router,
assertion mirror, autobiographical and self-cued episodic recall did not intercept
the paraphrase forms. The assertion mirror DOES intercept the identical-surface
form (`you mentioned: "…"`, no GEN event) — that is the 0.50 echo signal, not reach.

#### 2.3 Classification of the failing 60% (code + pre-fix per-item evidence)

No post-527f401 per-item JSON exists in scratch (verified by listing). The
classification below is therefore: (i) items the pre-fix JSON proves never had a
reachable trusted chain, plus (ii) code-level exits that 527f401 left in place.

1. **No trusted chain to reach (acquisition ceiling).** Items with 0 trusted chains
   at cold in the JSON cannot reach regardless of routing: primary `comet` (raw=0),
   `coral`, `sleet`, `whisper`, `shale`, `reverb`, `crater`, `graphite`, `geyser`;
   sealed `snowfall`, `bruise`, `shiver`, `nectar`, `cocoon`, `volcano`. At k=8
   primary, `geyser` (0 chains) and `fruit` (cold loss) cap reach at ≤0.8 even with
   perfect routing. This is the largest single bucket and is an acquisition/write
   problem, not a routing problem.
2. **Wh-cue grounding still substitutes a phrase (R3, unfixed — confirmed at HEAD).**
   The assembler seeds ONLY `ctx.subject`: `_ck.get(ctx.subject)` +
   `_singular_candidates(ctx.subject)` + `_graph_node_id_for_label(ctx.subject)`
   (`response_gen.py:8567-8576`). For `what is the mechanism behind X?` grounding
   yields `mechanism behind X` (`car_answer_reach_probe.log`, codemap R3); none of
   the three lookups resolves that phrase, `_nids` is empty, and the assembler
   returns `None` at `:8577-8578` — **before any enumeration**. `mechanism_intent`
   now ADMITS the form, converting a regex rejection into a resolution miss. The
   cold confirmatory probe is exactly PARAPHRASES[2] (this form), so cold reach
   stays near zero regardless of the limit raise; the 0.40 is the IMMEDIATE probe
   (PARAPHRASES[1], `how is X produced?`, which grounds to X correctly). The
   reverted `longest_label_seed` (:253) was the candidate fix; it added seeds
   unconditionally and regressed unrelated fronts, so it was reverted — the narrow
   fallback version (F5 below) is untested.
3. **Trust-blind ranking at limit 64 (Drop point D).** `honey`-class chains rank
   below 64 after secular growth; rank drift (callus 0→41) means even a chain found
   on one turn can be missed on the next. The fix raised the budget but did not
   change the scoring that pushes trusted chains down.
4. **Definition-consult bypass with the stale guard (R4, unfixed — confirmed at HEAD).**
   `_mechanism_chain_available` (`engine_self_query.py:2436-2469`) still uses the
   OLD direction regex (`why|what causes|what makes` → backward, else FORWARD,
   :2460-2461), **limit=3** (:2465), no graph-label fallback, and checks only
   `assertable_chain` (no `grade_chain`). Its admission at `:2606-2608` requires
   `why|how` in the text — so `what is the mechanism behind X?` never even reaches
   the skip check. For `how is X produced?` the helper walks FORWARD at limit 3,
   cannot see the incoming taught chain, returns False, and the stored definition
   is allowed to short-circuit the turn (`coral`, `fossil`, `rainbow` emitted
   definitions in the audit). 527f401 fixed the assembler but not this guard —
   the skill §4 "one predicate at every router" rule is half-applied.
5. **Direction residue.** Genuine effect questions walk forward (correct), but any
   paraphrase the predicate still misclassifies walks forward over incoming taught
   edges and finds nothing (codemap R1 measured this on P1/P3 pre-fix).

**Post-generation mutators — ruled out with evidence (codemap §4):**
`_detect_and_handle_correction` (no explicit correction in probes),
`_last_response_was_salad` (no stale flag observed), `_forward_model_check` →
`_human_like_uncertainty` (the uncertainty replies are the generator's OWN strategy
`metacognitive_uncertainty`, not replacements of a produced chain),
`_final_emit_guard` (returned input unchanged on all 36 probes; explicitly exempts
`strategy == "gated_explanation_chain"`, engine.py:12831-12854). When the assembler
returns a chain, `_generate_response` emits it verbatim — so reach ≈ "assembler
fired with a trusted chain", and the remaining gap is upstream of generation.

### RC-3 — Monotonicity reversal k1→k2 = −0.50: per-item attribution

Primary k=2, all 10 items (audit JSON):

| item | base→cold | contribution to Δ_cold(k=2)=+0.20 | cause class |
|---|---|---|---|
| rust | 1→1 | 0 (capped) | (a) composition — curriculum-known |
| earthquake | 1→1 | 0 (capped) | (a) composition — curriculum-known |
| amber | 0→0 (imm 1) | 0 instead of +1 | (c) save/reload loss, absent ≤1024 |
| magnify | 0→0 (imm 1) | 0 instead of +1 | (c) save/reload loss, absent ≤1024 |
| stalactite | 0→0 (imm 1) | 0 instead of +1 | (c) save/reload loss, absent ≤1024 |
| honey | 0→0 (imm 1) | 0 instead of +1 | (c/D) rank loss at 64, present at 256 |
| lightning | 0→1 | +1 | retained (immediate truncation only) |
| hurricane | 0→1 | +1 | retained |
| comet | 0→0 | 0 | acquisition failure (raw=0 throughout) |
| coral | 0→0 | 0 | acquisition failure (never assertable) |

**Attribution:** the reversal is **(c) save/reload loss (4 items) + (a) item
composition (2 capped items)**, amplified by **(d) n=10** granularity (each item
is 0.10 of the rate). Counterfactual arithmetic: repairing the 3 hard losses plus
honey gives cold rate 0.8, Δ+0.60, adjacent reversal −0.10 — exactly at the bound
(criterion 7 fails only on >0.10). Repairing only the 3 hard losses gives Δ+0.50,
reversal −0.20 — still FAIL. Removing the 2 curriculum-known items from the pool
(baseline 0) would add +0.20 to Δ. **(b) genuine retroactive interference is NOT
supported for these 10 items** — no primary k=2 item lost a baseline chain; the
interference signature exists elsewhere (sealed `rainbow`, §1.6) and in CLI
(A-only 0.900 vs sequential 0.700, `LEARN5_RESULTS.md:27-30`), but it does not
explain this reversal. The failure is deterministic item-level loss, not sampling
noise — though at n=10 the criterion's −0.10 bound is within one item's worth of
granularity, so replication at n≥40 remains required (design §3.5).

---

## 2. Brain mechanisms — how the brain protects new associations and makes them retrievable on paraphrase

All citations below were resolved to publisher / PubMed / PsycNET records during
this investigation (author list, year, journal, volume/pages, DOI taken from those
records). Where a source is a theory or review rather than a direct experiment, it
is flagged. Each mechanism is stated in RAVANA's own primitives:
`edge.weight` / `edge.confidence` / `edge.source_metadata` (`edge_kind`, `source`,
`evidence`, `acquired_by`) / `edge.relation_type`, `grade_chain`, `assertable_chain`,
`enumerate_chains` traversal.

### 2.1 Retrieval practice / testing effect
**Roediger, H. L., III, & Karpicke, J. D. (2006).** Test-enhanced learning: Taking
memory tests improves long-term retention. *Psychological Science, 17*(3), 249–255.
doi:10.1111/j.1467-9280.2006.01693.x — verified (SAGE publisher page + PsycNET
2006-05194-012). Restudy beats testing at 5 min; testing beats restudy at 2 days
(68% vs 54%) and 1 week (56% vs 42%). **Karpicke, J. D., & Blunt, J. R. (2011).**
Retrieval practice produces more learning than elaborative studying with concept
mapping. *Science, 331*(6018), 772–775. doi:10.1126/science.1199327 — verified in
`BRAIN_LEARN5_DESIGN.md` §11[4] (publisher record); retrieval advantage extends to
inference questions.
**RAVANA mapping:** the delayed cold-reload paraphrase IS the retrieval test — the
design is right. What is missing is retrieval *as a learning event*: today a probe
turn reads the graph but never reinforces the retrieved hops. A successful
retrieval (`grade_chain verdict=='assert' AND assertable_chain`) should increment
exactly the traversed hops' `weight`/`confidence` and refresh their
`source_metadata` episode stamp — reinforcement conditional on an external
criterion (the gate), not on the generated text (skill §7). This is F6.

### 2.2 Retrieval-induced forgetting (inhibitory account)
**Anderson, M. C., & Spellman, B. A. (1995).** On the status of inhibitory
mechanisms in cognition: Memory retrieval as a model case. *Psychological Review,
102*(1), 68–100. doi:10.1037/0033-295X.102.1.68 — verified (PsycNET + multiple
publisher reference lists). Retrieving a target can suppress competing associates
that share cues; the account is a **theory with contested boundary conditions**
(cue-independence debate), not an unconditional law.
**RAVANA mapping:** the analogue of competitor suppression is candidate selection
in `enumerate_chains`. Today the competition is resolved by `score_chain`
(`explanation.py:147`), which lets high-weight UNTRUSTED co-occurrence chains
out-rank the taught trace — the wrong item wins the retrieval competition. The
brain-faithful move is to inhibit the *untrusted competitors* at selection
(trust-first partition, F3), never to weaken the taught edge. Note the direction
of the analogy: RAVANA's measured "interference" (rainbow, CLI sequential) is
overwrite-at-write, not retrieval-induced suppression; do not conflate them.

### 2.3 Reconsolidation
**Nader, K., Schafe, G. E., & LeDoux, J. E. (2000).** Fear memories require protein
synthesis in the amygdala for reconsolidation after retrieval. *Nature, 406*(6797),
722–726. doi:10.1038/35021052 — verified (Nature publisher page; PMID 10963596).
Reactivated consolidated memories return to a labile state requiring de novo
protein synthesis to restabilize. **Kumaran, D., Hassabis, D., & McClelland, J. L.
(2016).** What learning systems do intelligent agents need? Complementary learning
systems theory updated. *Trends in Cognitive Sciences, 20*(7), 512–534.
doi:10.1016/j.tics.2016.05.004 — verified (PubMed 27315762 + Cell Press); replay is
goal-dependent weighting of experience statistics, and schema-consistent
information can be learned rapidly by the slow system.
**RAVANA mapping:** every probe turn reactivates the taught chain, and save/load is
a reconsolidation window. A restabilizing load must restore the trace AS IT WAS
WRITTEN (episode `source_metadata` intact); RAVANA's load path instead RE-STAMPS
reactivated curriculum-coincident edges with schema defaults
(`engine_graph.py:933-938`) — the computational analogue of blocking
reconsolidation and letting the schema overwrite the episode. F1 makes the load
path restabilize rather than rewrite; F6 makes retrieval-triggered reinforcement
the "protein synthesis" step (no retrieval, no strengthening).

### 2.4 Complementary learning systems / interleaved consolidation
**McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995).** Why there are
complementary learning systems in the hippocampus and neocortex: Insights from the
successes and failures of connectionist models of learning and memory.
*Psychological Review, 102*(3), 419–457. doi:10.1037/0033-295X.102.3.419 — verified
(PsycNET doiLanding + PubMed 7624455). Fast hippocampal one-shot episode learning
vs slow neocortical statistical structure; interleaved replay avoids catastrophic
interference. **McCloskey, M., & Cohen, N. J. (1989).** Catastrophic interference
in connectionist networks: The sequential learning problem. *Psychology of Learning
and Motivation, 24*, 109–165. doi:10.1016/S0079-7421(08)60536-8 — verified in design
doc §11[14] (edited-volume chapter; a connectionist model, not human data).
**RAVANA mapping:** taught `curated` edges with episode provenance are the fast
store; the PMI/co-occurrence graph is the slow store. CLI already measured the
CLS prediction: sequential A→B retention 0.700/0.833 vs concurrent 0.875/1.000
(`LEARN5_RESULTS.md:27-30`). The sleep routine must therefore REPLAY INTERLEAVED
taught episodes at matched compute (F6), and the fast-store trace must remain
distinct (its own `edge_kind`/`source`) instead of being merged into slow-store
defaults at load (F1). The trust gate is the system's source-monitoring boundary
between the two stores.

### 2.5 Synaptic tagging & capture
**Frey, U., & Morris, R. G. M. (1997).** Synaptic tagging and long-term
potentiation. *Nature, 385*(6616), 533–536. doi:10.1038/385533a0 — verified (Nature
publisher page; PMID 9020359). A weak stimulus sets a local synaptic tag that
captures plasticity-related proteins supplied by a strong stimulus, converting
early LTP into late LTP; the tag is local and time-limited.
**RAVANA mapping:** `source_metadata` IS the tag — it marks which episode wrote
the edge and makes it eligible for consolidation. Two consequences: (a) the tag
must survive the capture window: the load-time `curated→curriculum` overwrite
destroys the episode tag (F1); (b) weak taught edges (measured w=0.19–0.50,
c=0.39–0.54) are early-LTP analogues that decay below the traversal/grade horizon
unless captured — retrieval-practice reinforcement (F6) is the capture event, and
`grade_chain`'s confidence term (`explanation.py:345`) is where the decay shows up
as lost assertability. A maintenance routine that bumps weight without restoring
the tag cannot repair a provenance-damaged edge (skill §6, measured in d7a3495's
reverted replay fix).

### 2.6 Systems consolidation & schema
**Tse, D., Langston, R. F., Kakeyama, M., Bethus, I., Spooner, P. A., Wood, E. R.,
Witter, M. P., & Morris, R. G. M. (2007).** Schemas and memory consolidation.
*Science, 316*(5821), 76–82. doi:10.1126/science.1135935 — verified (Science
publisher page + Edinburgh/Dundee records). Once a neocortical schema exists,
schema-consistent paired associates are assimilated in one trial and become
rapidly hippocampal-independent; schemas play a causal role in lasting associative
memory.
**RAVANA mapping:** the curriculum scaffold is RAVANA's schema. The measured
pattern fits: curriculum-known subjects (`rust`, `earthquake`) retained best and
their taught chains assimilated onto existing structure; `amber`'s immediate
"trusted chain" was itself a schema-like composite borrowing `fossil`'s hop.
Schema congruence should ACCELERATE assimilation (keep the scaffold, let taught
edges attach to it) — but the schema must PROPOSE, not overwrite: Tse's result is
about new traces being assimilated, not about old traces being relabeled, and
schema-driven reconstruction is also a principal source of false memory (skill
§7a; Johnson & Raye 1981 / Johnson et al. 1993, verified in design doc §11[6,7]).
F1 preserves the episode record while keeping the schema available.

### 2.7 Spacing
**Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006).**
Distributed practice in verbal recall tasks: A review and quantitative synthesis.
*Psychological Bulletin, 132*(3), 354–380. doi:10.1037/0033-2909.132.3.354 —
verified (doi.org landing record + PubMed 16719566). Meta-analysis of 839
assessments / 317 experiments: the optimal interstudy interval grows with the
retention interval.
**RAVANA mapping:** k as spaced exposures is the correct manipulation and the
monotonicity gate operationalizes the spacing law. CAR's spacing is within-session
(a reported limitation), so the curve tests short-lag spacing only. The measured
k1→k2 reversal is item-level retention loss (§RC-3), NOT evidence against the
spacing law — repair the load path and acquisition first, then re-test the curve
at n≥40 with the same fixed spacing parameter (never tuned after results).

### 2.8 Encoding specificity / transfer-appropriate processing (paraphrase reach)
**Tulving, E., & Thomson, D. M. (1973).** Encoding specificity and retrieval
processes in episodic memory. *Psychological Review, 80*(5), 352–373.
doi:10.1037/h0020071 — verified (PsycNET 2005-09647-002). **Morris, C. D.,
Bransford, J. D., & Franks, J. J. (1977).** Levels of processing versus transfer
appropriate processing. *Journal of Verbal Learning and Verbal Behavior, 16*(5),
519–533. doi:10.1016/S0022-5371(77)80016-9 — verified (PsycNET 1978-26673-001 +
Crossref). Retrieval succeeds when the cue overlaps the encoding context; deep
encoding alone does not guarantee retrieval on a mismatched test.
**RAVANA mapping:** this is the exact principle `mechanism_intent`'s docstring
cites (`response_gen.py:236-237`). A paraphrase cue must address the SAME node the
teaching episode wrote. Today the wh-cue phrase (`mechanism behind X`) shares no
resolution path with the taught label, so encoding specificity is violated at the
node-seeding step (`response_gen.py:8567-8578`) — the trace exists but the cue
cannot address it. F5's narrow fallback restores cue→trace addressability without
adding unrelated seeds (the reverted experiment's failure mode).

## 3. Ranked fix proposals

Ranking criterion: strength of measured evidence × narrowness × gate preservation
(no proposal changes `grade_chain`, `assertable_chain`, thresholds, or the trust
set). Each names exact targets, the primitive change, a measurable prediction, and
a falsifier. All A/Bs: identical copied state, one process, `PYTHONHASHSEED=0`,
`RAVANA_OFFLINE=1`, per skill §1; execution belongs to the execution-owning agent.

### F1 — Provenance-preserving load-time re-seed (rank 1: line-pinned, measured overwrite)
- **Targets:** `_seed_curriculum_causal` `engine_graph.py:930-938`; `_ensure_relation`
  backfill `engine_graph.py:995-999`; called from `engine.py:2194` (load) and `:2272`
  (cold start).
- **Change in primitives:** never overwrite an existing trusted `edge_kind`. Replace
  `if kind not in (None, "curated"): continue` with a rule that relabels ONLY
  `kind is None` or `kind == "curated" AND source in {"common_facts", "seed"}`;
  conversation-stamped `curated` edges (intake.py:80-81) keep `edge_kind`, `source`,
  `evidence`, `acquired_by`. Curriculum may still CREATE absent hops (schema
  proposes); it may not rewrite episode records (sources admit).
- **Prediction:** on the identical saved scratch state, cold census shows taught
  hops retain `edge_kind='curated'` + original episode source (rust `oxidation→rust`
  stays curated); trusted-chain cold rate ≥ current; accept/reject counts on
  known-bad probes byte-identical.
- **Falsifier:** if amber/magnify/stalactite remain absent at limit 1024 with
  provenance preserved, the overwrite was trust-neutral for them (both kinds are
  trusted) and their loss lies in F2/F4 — record that and demote F1 to a
  provenance-integrity fix, not a retention fix.
- **Brain basis:** synaptic tagging (Frey & Morris 1997, §2.5) — the tag must
  survive capture; reconsolidation (Nader et al. 2000, §2.3) — restabilize the
  reactivated trace, don't replace it; schema (Tse et al. 2007, §2.6) — assimilate,
  don't overwrite.

### F2 — Provenance-aware sanitization: census first, re-anchor instead of delete (rank 2)
- **Targets:** `_sanitize_graph` `chain_walker.py:1335-1359`; `_is_question_phrase`.
- **Change in primitives:** before deleting a phrase node's incident edges, census
  them by `edge_kind`/`relation_type` under the existing trace flag. If a trusted
  causal edge is incident, RE-ANCHOR it to the canonical concept node (resolve the
  endpoint label via `_concept_keywords` / graph labels) instead of deleting; log
  every re-anchor. Delete only untrusted phrase debris.
- **Prediction:** the 8 pruned nodes (car_full_v2.log:11352) take zero trusted hops
  with them; if any taught hop was phrase-routed, its item's cold chain recovers;
  `[Sanitize]` log enumerates exactly what was removed/re-anchored.
- **Falsifier:** if the census shows NO trusted incident edge on any pruned node,
  this mechanism is innocent for CAR — record the negative and revert (skill §7a:
  measure the substrate before keeping the machinery).
- **Brain basis:** pattern separation must not discard the episode trace
  (CLS, §2.4); source monitoring (Johnson et al. 1993) — cleanup decisions must
  read provenance, not surface form alone.

### F3 — Trust-first candidate selection (rank 3: strongest single-item evidence)
- **Targets:** `score_chain` `explanation.py:147-154`; `enumerate_chains` sort/truncate
  `explanation.py:261-265`; callers `response_gen.py:8579-8582`,
  `engine_self_query.py:2462-2465`.
- **Change in primitives:** partition candidates by `assertable_chain` BEFORE
  ranking; rank trusted candidates among themselves by the existing score; only
  fall back to untrusted candidates if no trusted chain exists (and the gate still
  declines them). Equivalently, add a trusted-hop bonus that dominates the
  log-weight term. Traversal budget stays 64; `grade_chain`/thresholds unchanged.
- **Prediction:** honey-class chains found at limit 64 (cold rate at 64 ≈ rate at
  1024 — budget-insensitivity); rank drift eliminated (callus stays top-ranked
  across turns); primary k=2 Δ_cold rises by the honey item (+0.10); answer reach
  on P1 rises for rank-limited subjects (blush/tornado/limestone/velvet class).
- **Falsifier:** if trusted chains still rank below 64 after re-scoring, the loss
  is edge ABSENCE, not ranking (confirm with the 1024 census) → F1/F2/F4 own it.
  If unrelated-front scores regress, the partition leaked untrusted chains into
  emission — revert.
- **Brain basis:** inhibitory selection of the target over prepotent competitors
  (Anderson & Spellman 1995, §2.2); fast-store traces must not be drowned by
  slow-store statistics (McClelland et al. 1995, §2.4).

### F4 — Fix the acquisition write path; stop relation-word cross-product (rank 4)
- **Targets:** the assertion-intake parse behind `teach_conversational`
  (`scripts/measure_learning_curve.py:247` → `process_turn`); `learn/intake.py`
  (`edge_type_for_relation` ~:260, source map :80-81); `web/web_to_graph.py:_ensure_node`
  (:126) for the surface-node class; grounding phrase path that mints `causes`/
  `happens` nodes.
- **Change in primitives:** parse `X happens because a causes b, and b causes x`
  into EXACTLY two causal edges `a→b`, `b→x` between canonical concept nodes,
  stamped `edge_kind='curated'`, `source='conversation:<turn>'`,
  `acquired_by='user_assertion'`. Never mint relation-word nodes (`causes`,
  `happens`) or wire them causally. Instrument the WRITE count per episode
  (comet wrote zero chains — the answer echo hid it).
- **Prediction:** comet/coral-class items show raw_chains>0 immediately; the
  per-subject untrusted `unknown` causal census (currently ~124–140 of 128 hops)
  drops by the cross-product amount; trusted-chain rate rises at k=1/2; the
  truncated echo `ice leads to vapor, and vapor leads to` disappears (it is an
  utterance-store artifact, not retrieval).
- **Falsifier:** if comet's two edges are verified written but raw_chains stays 0,
  the defect is node RESOLUTION (label→ids), not writing — instrument
  `node_ids_for` vs the write-time ids before touching the parser further.
- **Brain basis:** one-shot episode encoding must write a distinct, addressable
  trace (CLS §2.4; encoding specificity §2.8).

### F5 — Complete the shared-predicate wiring: guard + narrow paraphrase resolution (rank 5)
- **Targets:** `_mechanism_chain_available` `engine_self_query.py:2436-2469` and its
  admission `:2606-2608`; assembler resolution `response_gen.py:8567-8578`; dead
  helper `longest_label_seed` `response_gen.py:253-261`.
- **Change in primitives:** (a) guard imports `mechanism_intent`, replaces the old
  direction regex, raises limit 3→64, adds the graph-label fallback and the
  `grade_chain` check — one predicate at every router (skill §4). (b) Resolution
  fallback ONLY when `ctx.subject` resolves to no node: match the longest taught
  label present in the raw query (the existing helper), as a REPLACEMENT seed, not
  an additional one. The reverted experiment added seeds unconditionally and
  regressed unrelated fronts; the narrow fallback is a different, untested change.
- **Prediction:** P2/P3 paraphrase reach rises from ~0 toward the P1 level (0.40);
  cold-probe reach becomes nonzero; definition bypass no longer intercepts
  mechanism turns for definition-known subjects; k=0 and def-only controls stay
  0.00; grounded/thinking fronts do not regress beyond noise.
- **Falsifier:** if unrelated-chain emissions rise on non-mechanism fronts or any
  control moves, revert the resolution half (as the first experiment was) and keep
  only the guard wiring, which is gate-preserving by construction.
- **Brain basis:** encoding specificity / transfer-appropriate processing
  (Tulving & Thomson 1973; Morris et al. 1977, §2.8); retrieval practice on altered
  cues is the criterion (Roediger & Karpicke 2006; Karpicke & Blunt 2011, §2.1).

### F6 — Retrieval-practice replay (testing effect), interleaved, compute-matched (rank 6)
- **Targets:** sleep/replay routine `engine_generation.py:~1200-1550` (prune at
  :1257, junk-node prune at :1516); CLI harness `scripts/measure_learn5_cli.py`.
- **Change in primitives:** replay RE-RETRIEVES taught episodes (walk
  `enumerate_chains` from the taught subject, grade with the unchanged gate); on
  `assert`, reinforce exactly the traversed hops' `weight`/`confidence` and restore
  episode `source_metadata` FROM THE ITEM'S OWN RECORD (never a blanket stamp —
  skill §6). Interleave items (concurrent-training protection); match wall-clock
  and step count across on/off arms; spacing schedule per Cepeda et al. 2006.
- **Prediction:** CLI sleep-on − sleep-off A-retention ≥0.15 at matched compute
  (the PASS bound the current routine misses at 0.000); CAR k=2 cold retention
  rises for weak-weight items; NEGATIVE metric: untrusted-edge count does not rise
  and known-bad probe rejections are unchanged.
- **Falsifier:** if on/off arms stay identical at matched compute (as the reverted
  provenance-restore replay did, d7a3495), the retrieval signal carries no
  information the graph lacks — demote replay to a fluency gate and record the
  negative (skill §7). If the gain appears only with a compute overrun, it is
  "more replay", not consolidation — revert (skill §7b).
- **Brain basis:** testing effect (Roediger & Karpicke 2006; Karpicke & Blunt 2011,
  §2.1); reconsolidation (Nader et al. 2000; Kumaran et al. 2016, §2.3); synaptic
  capture (Frey & Morris 1997, §2.5); interleaved CLS replay (McClelland et al.
  1995; McCloskey & Cohen 1989, §2.4).

### F7 — Prerequisite instrumentation and replication (applies to all of the above)
- Pin Drop point C: read the `_load()` tail (`engine.py:13400+`) for the
  confidence-renormalization line; read-only unpickle census of
  `scratch/learn5_car_data/ravana_weights.pkl` (ids/kind/weight/confidence for every
  causal edge incident to amber/magnify/stalactite/honey labels) before vs after
  load-path mutations; trace-flag census of the 8 sanitized nodes.
- Replicate CAR at n≥40 per pool, same config twice, fixed hash split and spacing
  parameter (design §3.5, §10). Until then the −0.50 reversal is a binding FAIL
  with a located cause, and every fix above is a hypothesis with a falsifier, not
  a result.

## 4. What the evidence does NOT support

- **Not a pickle provenance drop.** `ConceptEdge.__getstate__` keeps
  `source_metadata` (graph.py:304-328); the historical defect is fixed. The audit's
  cold hops still carry `edge_kind`, so "provenance not pickled" is ruled out.
- **Not node-id remapping or dedup at load.** `ConceptGraph.__setstate__` restores
  ids unchanged; `_revector_existing_nodes` touches vectors only.
- **Not load-time pruning of low-confidence edges.** `prune_low_quality_edges`
  (graph.py:1846) runs only in the sleep routine (engine_generation.py:1257);
  `car_full_v2.log` contains no sleep/prune lines. The only load-time deletion is
  `_sanitize_graph`'s phrase-node class (§1.3).
- **Not a post-generation mutator discarding chains.** Codemap §4: emit guard
  exempts the chain strategy; uncertainty replies are generator-native.
- **Not "the trust gate rejected facts it should accept."** `grade_chain` on a
  fully-trusted causal chain computes adequacy = min(1, 1 + min_conf − 0.5·gaps)
  with gaps=0 (explanation.py:343-356) → always ≥ THETA_ASSERT=0.70. Cold losses
  are absence of a fully-trusted chain in the enumeration, not threshold rejection.
- **Not retroactive interference as the cause of the primary k=2 reversal** (§RC-3).
- **Not a traversal-budget fix alone.** Raising the limit cannot repair amber /
  magnify / stalactite (absent at 1024) or comet / coral (never acquired); the
  harness correctly refused to raise it again after seeing results.
- **Not answer-level learning from the +0.80 headline.** The headline is a
  graph-object endpoint; emitted-answer reach is 0.40 post-fix and the
  identical-surface 0.50 echo is assertion-mirror parroting, not mechanism reach
  (codemap R5). The two real pre-fix reach hits (`ivory`, `tusk`) emitted FORWARD
  chains, not the taught backward mechanism.

### Open sub-questions (for the execution-owning agent; read-only probes proposed)

1. Which exact line re-values edges across reload (Drop point C): read
   `engine.py:13400+` tail of `_load()`; then a read-only unpickle of
   `scratch/learn5_car_data/ravana_weights.pkl` listing, for amber/magnify/
   stalactite/honey, every causal edge incident to the taught labels with
   id/kind/weight/confidence — before vs after the load-path mutations.
2. Which 8 phrase nodes `_sanitize_graph` pruned, and whether any carried a taught
   hop (instrument `_sanitize_graph` under the existing trace flag; no gate change).
3. Whether `_concept_keywords` first-id resolution (engine_graph.py:916-922) picks
   a different node for a taught label at cold than in-process (label→ids census).