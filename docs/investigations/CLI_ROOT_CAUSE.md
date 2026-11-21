# CLI root-cause investigation — compute parity, criterion_rate, node loss, sealed 0.000

**Status:** investigation doc (read-only analysis; no engine, pytest or measurement
script was run; no source file touched). Branch `docs/replay-fix-negative`, HEAD
`d7a3495`. Exclusive ownership: this file only.
**Evidence base:** `scratch/learn5_cli_measured.json` (the FIX run, commit
`527f401`, full run, `instrument_valid: true`, elapsed 1409.7 s — verified by
reading the file, not from memory), `scratch/learn5_cli_fix_full.log`,
`docs/LEARN5_CLI_MEASURED.md` (baseline `c47ac09`), `docs/LEARN5_REPLAY_FIX.md`,
`docs/CLI_REPLAY_CODEMAP.md`, and direct source reads cited file:line below.
Per-item numbers in §3–§6 were extracted programmatically from the JSON's
`arm_records`, not eyeballed.

---

## 1. Measured state (fix run 527f401, from scratch JSON)

| pool | criterion | ret_on | ret_off | on−off | sleep wall ON/OFF (s) | ratio |
|---|---|---|---|---|---|---|
| PRIMARY | 10/20 = 0.500 | 0.800 (8/10) | 0.700 (7/10) | +0.100 | 8.3984 / 7.4871 | **1.122** |
| SEALED | 6/20 = 0.300 | 0.833 (5/6) | 0.833 (5/6) | 0.000 | 7.5234 / 7.6011 | 0.990 |

Pre-registered release rule (task + `docs/BRAIN_LEARN5_DESIGN.md` §6.6):
on−off ≥ 0.15 **both pools**, ret_on ≥ 0.85 both pools, wall-clock ratio in
[0.90, 1.10], criterion_rate ≥ 0.60. All four currently fail or tie.

**Harness band discrepancy (must be fixed before the next run, not relaxed):**
the reverted harness on this branch carries `COMPUTE_RATIO_LO, COMPUTE_RATIO_HI
= 0.70, 1.43` (`scripts/measure_learn5_cli.py:104`, ±30%) and the criterion text
at :614 still says "+/-30%". The fix had tightened this to 0.90/1.10
(`docs/LEARN5_REPLAY_FIX.md` §3.2); the revert restored the loose band. The
pre-registered requirement is ±10%; the next implementation must restore
0.90/1.10 in the harness. Nothing in this doc proposes relaxing it.

---

## 2. Q1 — Compute parity: exactly where restored edges create extra work

The fix's ON arm restored a handful of causal edges (earned provenance) that
OFF's wrong-tag arm did not keep. Steps/cycles matched (10/3), yet PRIMARY wall
clock ran 1.122×. The overrun is **structural**: five loops in the sleep path
have cost proportional to live graph size, and restoration changes live graph
size mid-phase. Located, in execution order:

1. **STAGE 2 homeostatic downscaling** — `ravana/src/ravana/core/sleep.py:224-234`
   (`_normalize_outgoing_weights`): iterates every node's outgoing list and sums
   weights, O(E). ON carries restored edges through this scan in cycles 2–3.
2. **STAGE 3 weight prune** — `sleep.py:236-273` (`_prune_weak_edges`):
   `for (src,tgt),edge in list(graph.edges.items())` — a full O(E) scan per
   cycle. Restored edges with ≥2 contexts are *scanned then skipped*
   (:264-268); OFF's wrong-tag low-weight edges are scanned then removed
   (:271-272, `remove_edge` itself mutates adjacency lists). The two arms'
   edge populations diverge here and the divergence compounds each cycle.
3. **STAGE 3b noise prune** — `sleep.py:140-144` calling
   `graph.prune_low_quality_edges`, plus a SECOND call in the engine path at
   `ravana/src/ravana/chat/engine_generation.py:1256-1258`. Another full-edge
   kind-based scan; ON's restored `curated` edges survive, OFF's
   `inferred`/`semantic`-tagged recreations are culled → further size divergence.
4. **REM relation-type flip budget scales with E** — `sleep.py:352`:
   `n_flip = max(1, int(len(graph.edges) * cfg.rem_flip_rate))`. This is the
   sharpest coupling: the number of flip candidates sampled and processed is a
   direct function of live edge count, so ON literally executes more flip work
   than OFF in every cycle after the first restoration. Counterfactual
   generation (`sleep.py:307-349`) and creative recombination (:368-405) are
   node-count-scaled (`n_nodes * rate`, degree sort at :370) and do `np.dot` +
   two `np.linalg.norm` per sampled pair (:330-331, :389-390); any restored
   NODE changes these budgets too.
5. **Engine Phase 3b drain** — `engine_generation.py:1305-1399`: iterates
   `hippocampal_buffer.get_consolidation_candidates()` and calls
   `_intake.intake(...)` per candidate (:1373). Each intake evaluates
   consistency/elaboration/entrenchment by scanning the subject's in/out edges
   (`ravana/src/ravana/learn/intake.py:383-505`), i.e. O(degree) per fact —
   restored edges change both the degree scanned and the verdicts, so work and
   outcomes diverge. The fix harness drove `eng._sleep_consolidate()` in both
   arms, which also runs belief reconcile/prune (:1267-1278), personal-fact
   decay (:1284-1295) and `hippocampal_replay.sleep_cycle(replay_count=100,
   interleave_count=50)` (:1296-1304) — all O(state-size). This is why the fix
   run's sleep phase costs ~8 s vs the baseline harness's ~0.31 s detached
   `run_cycle`: the engine path is ~25× heavier, and the +0.91 s ON−OFF delta
   is the O(E) divergence above, not the 10 replay steps themselves.

**Conclusion:** matching *step counts* (10 triples × 3 cycles) cannot match
*compute* while any maintenance loop reads live `len(graph.edges)` /
`graph.nodes` / adjacency sizes that restoration mutates. The confound is
unfixable by padding replay lists; it must be removed structurally.

### 2.1 Structural compute-matching design (recommended): snapshot maintenance + metadata-overlay commit

Make every O(state) loop a pure function of an **identical frozen snapshot** in
both arms, and make restoration a **metadata-only overlay applied at commit**,
after all maintenance:

1. At sleep-phase start (post-B, pre-sleep — identical state in both arms by
   construction), deep-copy the graph once per arm: `G_snap = deepcopy(G_postB)`.
   Identical copy cost in both arms.
2. Run ALL cycles' maintenance (downscale, both prunes, drift defense, full REM
   sabotage, Phase 3b drain) **against `G_snap`**, with the REM RNG pinned as
   today (`measure_learn5_cli.py:398` pattern). Because the input graph, the
   budgets (`len(G_snap.edges)` etc.) and the RNG are identical, every loop
   executes the **same iterations, same predicate evaluations, same removals**
   in both arms — bit-identical work and bit-identical post-maintenance
   topology. REM sabotage stays fully intact (it runs on the snapshot exactly
   as it runs today on the live graph).
3. Replay restoration is computed as a **delta list** of ≤10 (src,tgt) records
   per cycle from the write-time episodic capture (the fix's
   `capture_replay_memory`, provenance deep-copied at A-teaching time). ON's
   records carry earned `source_metadata`/`relation_type`/confidence; OFF's
   carry the same keys with wrong (`inferred`/`semantic`) tags — the fix's
   existing wrong-tag control, unchanged.
4. **Commit:** `G_live = G_snap` then apply the accumulated delta list as
   metadata-only writes (restore fields on surviving edges; recreate deleted
   edges with the recorded metadata — ON earned, OFF wrong-tag). This is O(30)
   dict/attribute writes total, arm-independent in count. No maintenance loop
   ever sees a restored edge, so no downstream budget can diverge.

Predicted wall-clock ratio: 1.000 ± noise (the only arm-dependent work is ≤30
metadata writes and the tag values themselves). The ±10% band is satisfied by
construction, not by tuning. Iteration-count counters (edges scanned, flips
attempted, prunes evaluated, intake calls) should be asserted EQUAL across arms
as a new instrument control — stronger than wall clock alone.

**Brain-faithfulness of the ordering:** this is synaptic tagging & capture
(Frey & Morris 1997, *Nature* 385:533–536 — verification pending §7): the
reactivation "tag" is set on the specific synapses, and the capture of
plasticity-related protection is applied to those tagged synapses while the
untagged population undergoes downscaling (Tononi & Cirelli 2014). Implementing
protection as a post-downscaling restore of the recorded weight/confidence is
operationally the same final state as skipping the tagged edges during the
pass, but with identical compute. It is NOT a blanket exemption: only the ≤10
reactivated keys per cycle are touched, and only with their own write-time
record. The alternative "skip replayed keys inside the live loops" (the fix's
design) is what created the parity violation, because skipping changes which
edges survive and therefore the next loop's input size.

**Rejected alternatives:**
- *Fixed iteration budgets over live lists* (cap n_flip at pre-sleep E₀, scan
  only the first E₀ keys): weaker — predicate outcomes still differ inside the
  budget, removals still diverge, cycles 2–3 still compound. Use its counters
  as instrumentation, not as the matching mechanism.
- *Pad OFF with extra dummy edges to equalize E*: forges graph state OFF never
  earned, changes OFF's prune/flip outcomes, and is exactly the "extra compute
  to one arm" confound in reverse.
- *Relax the band to ±15%*: forbidden by pre-registration; the 1.122 overrun is
  a real confound (skill §7b: a 12% overrun with a primary-only gain is not
  consolidation).

---

## 3. Q2 — criterion_rate 0.50/0.30: where A-teaching drops items (before sleep)

Extracted from `arm_records[PRIMARY_on/SEALED_on].items[*].a1` (last of 4 reps;
all 4 reps checked — the failures are stable, not census fluctuation). The 24
criterion failures split into **two distinct classes**:

### Class 1 — raw_chains = 0 on every rep (18 of 24 failures)

PRIMARY: A_001/dew, A_008/bruise, A_018/reverb, A_012/yawn, A_011/shiver,
A_032/geyser, A_026/tornado (7).
SEALED: A_004/hiccup, A_019/prism, A_007/callus, A_010/sneeze, A_005/blush,
A_013/tear, A_017/tarnish, A_006/sunburn, A_038/aurora, A_039/meteor,
A_027/hurricane (11).

`gate_census` returns `raw_chains=0` meaning `enumerate_chains(nids, g,
direction="backward", min_steps=2, max_steps=4, limit=64)`
(`scripts/measure_learn5_car.py:261-271`, `ravana/src/ravana/learn/explanation.py:156-265`)
found **no causal-typed backward path of ≥2 hops** from the subject's node ids.
Teaching (`car.do_exposure` = `mlc.teach_conversational` + `mlc.teach_web`,
`measure_learn5_car.py:300-303`; the conversational statement is
`"X happens because A causes B, and B causes X."`,
`scripts/measure_learning_curve.py:247-254`) never produced an enumerable
trusted chain for these items across 4 full rounds. Candidate drop points, in
path order (verification status marked):

- **(a) Write-time confidence cap.** The intake gate's first-sighting merge
  targets `confidence = clamp01(0.15 + 0.35 * retention_prior)`
  (`ravana/src/ravana/learn/intake.py:751`) — **at most 0.50**, and subsequent
  merges approach the target only by `effective_write` fraction
  (`intake.py:762`). `grade_chain` uses `prov = min(hop.confidence)`
  (`explanation.py:345`) and `adequacy = typed + prov − 0.5·gaps + 0.5·margin`
  (:352). A 0.50-confidence hop is not by itself fatal (1.0+0.50 ≥ 0.70), but
  combined with a gap penalty (hop with empty `.provenance` string AND
  conf < 0.3 counts as a gap, :349-350 — note `provenanced` reads the string
  attribute, NOT `source_metadata`, while `assertable_chain` reads metadata via
  `_hop_provenance` :268-299) it can push adequacy below THETA_ASSERT=0.70
  (:47). This is the diagnosed mechanism for A_015/frost's post-sleep
  `graded_assert=0` (§5) and plausibly contributes here. **Verified code path;
  per-item confidence values at A1 not in the JSON (chain=None rows carry no
  hop detail) — needs a graph-dump probe by the execution owner.**
- **(b) Gate routing to buffer, not graph.** `KnowledgeIntake.intake`
  (`intake.py:677`) can route a fact to `episodic_buffer` (`_buffer`, :805)
  when consistency is low or a contradiction with an incumbent is detected
  (`detect_contradiction`, :507). Items whose a/b tokens collide with existing
  curriculum nodes (dew: cooling/moisture; yawn: fatigue; shiver: chill —
  shared with fog/frost) are exactly the ones where an incumbent edge could
  block the write. Buffered facts never become graph edges until a Phase 3b
  drain graduates them — and the CLI harness's sleep phase is the FIRST drain
  these items ever see, i.e. after criterion was already measured. **Plausible,
  consistent with the raw=0 pattern; needs intake-route counters per teaching
  turn to confirm (execution owner).**
- **(c) Relation typing at write.** `edge_type_for_relation`
  (`intake.py:262`) maps the extracted relation to a graph edge type;
  `web_to_graph.py:57-59` records the historical defect that "causes" was not
  in `ConceptEdge`'s relation table (only "causal"), so web-extracted causal
  facts were written non-causal. If any teaching channel still writes a
  relation string outside `CAUSAL_RELATIONS = {causal, physical_cause}`
  (`explanation.py:38`), `enumerate_chains` skips the hop at :209-210 and
  raw=0 results even though the edge exists. **Verified gate behavior; which
  channel mis-types these 18 items is unconfirmed.**
- **(d) Node-id resolution.** `mlc.node_ids_for` + `label_map` resolve labels
  to ids; the skill's surface-shadow trap (`rains` vs `rain`) and the
  `_concept_keywords` staleness trap apply. raw=0 with a present node would
  also occur if the taught edges attach to a DIFFERENT node id than the one
  the detector seeds. **Unconfirmed; cheap to check with a graph dump.**

### Class 2 — raw_chains = 64, trusted_all_hops = 0, rejected_by_trust = 64 (6 of 24)

PRIMARY: A_036/tide, A_023/snowfall, A_029/flood.
SEALED: A_016/mist, A_022/sleet, A_028/drought.

The traversal budget is **saturated with junk causal chains through literal
verb nodes**. Verified hop_gates examples from the JSON:

- tide: `tide→causes (unknown)`, `causes→flood (unknown)` — the graph contains
  a node labeled **"causes"** wired causally to subjects.
- flood: `flood→causes (unknown)`, `causes→happens (unknown)` — a node
  **"happens"** too.
- drought: `drought→causes (unknown)`, `causes→shift (unknown)`.
- snowfall: `snowfall→crystals (curated)` then `crystals→cooling (unknown)` —
  the taught b→x hop IS trusted, but the a→b hop (cooling→crystals) carries
  `edge_kind=unknown`, so `assertable_chain` fails closed on that one hop
  (`explanation.py:319-328`).
- mist: `mist→humidity (curated)`, `humidity→causes (unknown)`.
- sleet: `sleet→freezing (unknown)`, `freezing→cooling (curriculum)`,
  `sleet→slush (curated)` — again one untrusted hop kills the chain.

Two compounding defects:
1. **Junk-node generation from the teaching sentence itself.** The words
   "causes"/"happens" from `"X happens because A causes B…"` become concept
   nodes with causal-typed, unprovenanced edges. These enumerate as valid
   causal paths (they pass `_edge_is_causal`) and, being numerous, fill the
   limit-64 budget before any real chain is reached — the exact
   "untrusted cross-product edges crowding real ones out of a limited-size
   traversal" trap in the ravana-debugging skill. The generator of these
   specific nodes is not yet pinned to file:line (candidates: the
   conversational causal parse in `chat/chain_walker.py` ~:567-601 which
   treats connective words as causal splitters, or OpenIE argument extraction
   in the `teach_web` channel); **the execution owner should dump the
   `causes`/`happens` nodes' `source_metadata` and creating call site.**
2. **One-hop provenance asymmetry.** Where the real chain IS enumerated
   (snowfall, sleet, mist), exactly one hop lacks an earned `edge_kind` while
   the other is `curated`/`curriculum`. This is a WRITE-time metadata gap on
   one of the two taught relations, not a sleep defect — consistent with the
   skill's rule "the real drop point is often a missing metadata field at
   write time."

**Why this gates the whole component:** criterion_rate is measured at A1,
before B and before sleep; no replay fix can move it. Until ≥12/20 PRIMARY and
≥12/20 SEALED items reach criterion, the retention denominators stay 10 and 6,
and SEALED's on−off arithmetic is brutal (§6). The fix must therefore include
a write-time repair (Class 1 route/typing/confidence + Class 2 junk-node
suppression and one-hop provenance), not only a replay repair.

---

## P2 resolution (2026-11-21, parent execution owner)

Execution-owner probes (`scratch/p2_write_probe.py`, `p2_write_trace.py`,
`p2_openie.py`, `p2_gate_after_conv.py`) pinned Class 1 to two WRITE-time
defects, both confirmed on the failing item `dew` (a=cooling, b=moisture):

1. **Causal effects treated as contradictions.** `intake.detect_contradiction`
   treated ANY same-(subject, relation_type) pair with a different object as a
   competing value. Pre-existing `cooling→condensation` (curriculum) made the
   taught `cooling→moisture` route to `episodic_buffer` ("margin 0.08 < 0.10")
   — buffered facts never reach the graph before the A1 criterion. Causal
   relations are many-to-many (one cause has many effects); slot exclusivity
   belongs to functional relations. Fixed by excluding causal/physical_cause
   from contradiction detection (reconsolidation ADDS an effect; Nader, Schafe
   & LeDoux 2000, Nature 406:722–726). Functional relations still contradict.
2. **Existing semantic edge never upgraded.** `moisture→dew` existed as a weak
   semantic edge; teaching "moisture causes dew" only bumped its confidence and
   left `relation_type="semantic"`, so `enumerate_chains` skipped the hop.
   Fixed: `_write_edge` upgrades a non-causal incumbent to causal when a trusted
   causal fact reconsolidates it (never downgrades).

Supporting fixes:
- Causal CONNECTIVE words ("causes"/"happens") were fed back into the GloVe
  cross-product as arguments, minting unprovenanced junk causal edges to every
  mentioned entity; they now filter out of both argument lists (connectives are
  relational operators, not concepts — the do-calculus function-word skeleton).
- `WebToGraph.learn_text` gained an explicit `source_type` param (default "web");
  the authored teaching channel (`teach.local`) declares `source_type="seed"`
  (reliability 0.8) so it is source-monitored as teaching material, not an
  arbitrary page (Johnson, Hashtroudi & Lindsay 1993, Psychol Bull 114:3–34).

Verified: after the fixes, `dew` has both taught hops as causal+curated and
`raw_chains=13` with `[cooling, moisture, dew]` FIRST; `present=True` on all
4 teaching reps (was False/raw=0). 68 intake/contradiction/chain-walker tests
and 4 new `test_cli_write_path` tests pass. Full CLI harness re-measuring.

---

## 4. Q3 — raw_chains=0 after sleep: erosion and lightning (node vs edge loss)

Post-sleep ON-arm census (from JSON):
- PRIMARY A_030/erosion: `raw=0, trusted=0, assert=0`.
- SEALED A_025/lightning: `raw=0, trusted=0, assert=0` (lost in BOTH arms —
  this single item is the entire SEALED on−off=0.000 story, §6).

**Important correction to the fix doc's "node gone" claim:** for erosion the
node is demonstrably ALIVE post-sleep — A_031/canyon retains the structural
chain `['erosion','cutting','canyon']` in the same arm/snapshot, which requires
the `erosion` node and its outgoing `erosion→cutting` causal edge. What is gone
for A_030 is its INCOMING taught chain `weathering→wear→erosion`: either the
`wear→erosion` or `weathering→wear` hop was deleted or relation-flipped to a
non-causal type. So erosion is **edge/hop loss, not node loss**. For lightning
(`discharge→flash→lightning`) no other retained item references its nodes, so
node deletion is possible but NOT established by the JSON; the same
edge-loss explanation fits.

**Deleters that can remove a hop during B + sleep (verified code):**
- Weight prune `sleep.py:236-273`: any hop driven below weight 0.1 by
  downscaling (:224-234) without ≥2 corroborating contexts is removed.
  B-teaching grows the graph; per-node outgoing budget 5.0 (:25) means A's
  hops get scaled down as B's edges share nodes' budgets — the retroactive
  interference mechanism itself.
- Kind-based noise prune `graph.prune_low_quality_edges` (called at
  `sleep.py:141` and `engine_generation.py:1257`): culls edges whose
  `edge_kind` is noise-class. A hop whose metadata was damaged (or never
  stamped, §3 Class 2) is culled here even at healthy weight.
- REM flip `sleep.py:351-366`: `edge.relation_type = new_type` at :364 —
  a flipped causal hop becomes invisible to `enumerate_chains`
  (`explanation.py:209-210`) without any deletion. raw=0 is fully explained
  by both hops flipped.
- Node-level deletion: NOT yet located in source (no `remove_node` call site
  verified in the sleep path; `sleep.py` only removes edges). **Open item for
  the execution owner: grep `remove_node`/node-prune in `ravana_ml/graph.py`
  and the engine's turn-time maintenance before claiming node deletion.**
  The current evidence supports hop loss; "node gone" should be downgraded to
  "hop chain destroyed (delete or flip), node status unverified" until a graph
  dump shows the node absent.

**Can write-time episodic capture restore nodes too — brain-faithful or forgery?**
Yes, with a strict condition: the capture must record the NODE's own
write-time identity (label, vector, source_metadata) alongside the edge
record, and restoration may only recreate a node that the episode itself
created. This is hippocampal index reactivation: the hippocampus stores a
sparse index pointing to cortical patterns, and replay reactivates the index
which reinstates the cortical pattern (McClelland, McNaughton & O'Reilly 1995,
*Psychological Review* 102:429–457; Diekelmann & Born 2010, *Nat Rev Neurosci*
11:114–126 — both already cited in `docs/BRAIN_LEARN5_DESIGN.md` §11 and
`sleep.py:4`). Recreating a node from its OWN episodic record is reactivation
of a stored index, not forgery; recreating a node that was never encoded, or
stamping it with provenance it never earned, would be a false memory
(Nader, Schafe & LeDoux 2000, *Nature* 406:722–726; Kumaran, Hassabis &
McClelland 2016, *Neuron* 92:36–49 — cited in the fix doc §2 and
`engine_generation.py:1328-1336`). The fix's `EARNED_EDGE_KINDS` admission
(web_fact/definition/curriculum/curated) is the right gate and must extend
unchanged to node records. Under the §2.1 snapshot design, node restoration is
part of the same O(k) commit overlay, so it costs no extra maintenance work.

---

## 5. Q4 — SEALED 0.000: which items fail and why

SEALED criterion items (6): A_009/sweat, A_035/tsunami, A_033/volcano,
A_034/earthquake, A_021/hail, A_025/lightning.
Post-sleep ON: sweat retained (non-structural: `['irritant','secretion','sweat']`
— a curated prior chain outranks the taught `heating→secretion→sweat`; presence
metric counts it), tsunami/volcano/earthquake/hail retained, **lightning lost
(raw=0)**. OFF: identical — lightning lost, same 5 retained. Hence 0.833/0.833,
on−off = 0.000.

The fix recovered fog in PRIMARY (REM-flip/deleted-hop repair works when the
hop's record exists and the node lives) but could not recover lightning
because its loss is the raw=0 class (§4): the replay list is built from
captured hops, and restoration recreates edges — but if the failure is a
non-causal flip of BOTH hops plus a kind-prune of the recreated wrong-tag
edges in OFF, ON should have recovered it too. It did not, which means either
(a) lightning's hops were never captured (capture happens post-A-teaching;
lightning reached criterion at A1, so its chain existed then — capture should
have it), or (b) the restored chain still fails the grade/enumeration
post-commit (e.g. restored confidence below the adequacy bar, or the restored
edge is re-pruned by the NEXT cycle's maintenance because the fix applied
restoration BEFORE later cycles' prunes — the live-graph ordering defect of
§2). The fix doc's own diagnosis ("raw_chains=0 … node gone") plus the §2
ordering analysis makes (b) the leading hypothesis: **restoration applied
mid-phase is destroyed by subsequent cycles' prune/flip passes in ON too.**
The §2.1 commit-at-end ordering removes this failure mode by construction.
This is a concrete, falsifiable prediction: under snapshot+overlay, lightning
should recover in ON and not OFF, giving SEALED on−off = 1/6 = 0.167 ≥ 0.15.

**SEALED arithmetic (power warning):** with only 6 criterion items, the
smallest nonzero on−off is 0.167 — barely above the 0.15 bound — and ret_on
≥ 0.85 requires 6/6 (5/6 = 0.833 fails). So SEALED passes ONLY if every
criterion item is retained in ON and at least one is lost in OFF. This is
achievable (lightning is exactly such an item) but fragile; raising
criterion_rate (§3) to ≥0.60 gives ≥12 SEALED items and makes the bound
statistically meaningful. The design's n≥40 (§6.5) remains the real fix for
power; nothing here changes the gate.

PRIMARY post-sleep detail (ON): frost `raw=2, trusted=1, graded_assert=0` —
the trusted chain exists but fails `grade_chain`. From the codemap probe 2,
frost's hops pre-sleep were conf 0.728/0.901 (adequacy 1.0); post-B+sleep the
min-confidence hop fell and/or the margin term collapsed (junk runner-up
chains from REM counterfactuals raise the runner-up score, shrinking
`chain.margin`, `explanation.py:261-264, 351`). With `adequacy = 1.0·typed +
min_conf − 0.5·gaps + 0.5·margin` (:352), a min_conf ≈ 0.4 hop plus margin ≈ 0
plus one gap (empty `.provenance` string, conf < 0.3 sibling hop) lands below
0.70. The fix restored provenance but only restored confidence "when it drifted
below the earned value" — if the earned value itself was ≤0.50 (intake
first-sighting cap, `intake.py:751`), restoration cannot lift adequacy over
the bar. **frost is a grade-confidence problem, not a provenance problem; it
needs the write-time confidence repair (§3a) or replay confidence accumulation
that is earned (rehearsal-count-gated, cf. the Phase 3b corroboration policy
`engine_generation.py:1358-1375`), never a THETA_ASSERT change.**

---

## 6. Q5 — Brain literature (verification in progress)

Citations already verified to publisher records in `docs/BRAIN_LEARN5_DESIGN.md`
§11 and reused here: Wixted 2004 (*Annu Rev Psychol* 55:235–269); McCloskey &
Cohen 1989 (*Psychol Learn Motiv* 24:109–165, edited-volume chapter,
connectionist model not human data); Walker, Brakefield, Hobson & Stickgold
2003 (*Nature* 425:616–620); Diekelmann & Born 2010 (*Nat Rev Neurosci*
11:114–126, doi:10.1038/nrn2762); McClelland, McNaughton & O'Reilly 1995
(*Psychol Rev* 102:429–457); Nader, Schafe & LeDoux 2000 (*Nature*
406:722–726); Kumaran, Hassabis & McClelland 2016 (*Neuron* 92:36–49).

Task-requested references — **verified against publisher/PubMed records
2026-11-20 (parent)**:

- **Rasch, Büchel, Gais & Born (2007)**, *Science* 315(5817):1426–1429,
  doi:10.1126/science.1138581 — re-exposure to the learning-context odor during
  SWS improved retention of hippocampus-dependent declarative memories (not
  procedural). Supports: replay must carry the experience's OWN encoding
  context (the write-time provenance record), and targeted reactivation of
  specific traces — not blanket strengthening — is the active ingredient.
- **Oudiette & Paller (2013)**, *Trends in Cognitive Sciences* 17(3):142–149,
  doi:10.1016/j.tics.2013.01.006 — TMR review: cuing specific memories during
  sleep biases their consolidation; cues must match encoding context.
  Supports the ≤10-trace targeted replay window over whole-graph replay.
- **Frey & Morris (1997)**, *Nature* 385:533–536, doi:10.1038/385533a0 —
  synaptic tagging: weak stimulation sets a local, protein-synthesis-
  independent tag (decays <3 h) that captures plasticity-related proteins
  induced by strong stimulation at OTHER inputs to the same population.
  Supports §2.1: the reactivated (tagged) keys receive protection/restoration
  while the untagged population is downscaled — and the tag is input-specific,
  never a blanket stamp.
- **Tononi & Cirelli (2014)**, *Neuron* 81(1):12–34,
  doi:10.1016/j.neuron.2013.12.025 — synaptic homeostasis hypothesis: sleep
  renormalizes synaptic strength by down-selection; the review explicitly
  describes computational implementations with a "protection from depression"
  rule for the most-activated synapses (Hashmi et al. 2013; Nere et al. 2013).
  Supports: downscaling stays intact for the whole population; protection is
  restricted to the reactivated subset. Applying protection as a
  post-downscaling restore of recorded values yields the same final state as
  skip-during-pass, with identical compute (§2.1).
- **McNamara, Tejero-Cantero, Trouche, Campo-Urriza & Dupret (2014)**,
  *Nature Neuroscience* 17(12):1658–1660, doi:10.1038/nn.3843 — burst
  stimulation of hippocampal dopaminergic fibers during novelty/goal learning
  enhanced subsequent sleep SWR reactivation of the learning assemblies and
  stabilized memory. Supports: prioritizing replay of behaviorally significant
  (here: criterion-failing, fading) traces — the fix's
  `order_replay_experiences` fading-first ordering.
- **Ambrose, Pfeiffer & Foster (2016)**, *Neuron* 91(5):1124–1136,
  doi:10.1016/j.neuron.2016.07.047 — reverse replay rate (not forward) scales
  with reward increase/decrease. Supports reward/PE-modulated replay
  prioritization; honest flag: this is rodent place-cell evidence, the
  analogue in RAVANA is a scheduling prior, not a claimed homolog.

Interleaving already works in this system (concurrent arm 0.875/1.000 baseline;
0.778/0.857 in the fix run — note the fix run's concurrent numbers DIFFER from
baseline, consistent with the fix's engine-path sleep changing shared graph
state; the concurrent arm is a positive control, not a target). The
brain-faithful reading (McClelland et al. 1995; Kumaran et al. 2016): interleaved
replay DURING B teaching prevents the sequential-overwrite catastrophe at
encoding time, whereas post-hoc sleep replay must repair damage already done.
Harness implication: an `interleaved_replay` arm (A micro-replays inserted
between B items, matched total teaching turns) is the right NEXT experiment,
but it changes the manipulation from "sleep on/off" to "replay scheduling" —
it cannot replace the pre-registered sleep contrast without re-registration.
The pre-registered component can still pass via §2.1 + §3 + §4 repairs; the
interleaved arm should be added as a pre-registered SECONDARY comparison, not
a metric change. (Detailed evaluation patched after literature verification.)

---

## 7. Ranked fix proposals (RAVANA primitives, predictions, falsifiers)

**P1 — Snapshot-maintenance + metadata-overlay commit (§2.1).**
Primitives: `deepcopy(graph)`, existing `SleepConsolidation.run_cycle` on the
snapshot, existing `capture_replay_memory`/`order_replay_experiences`/
`replay_window` (fix harness), `EARNED_EDGE_KINDS` admission, commit-time
metadata writes. No gate change, no REM change, no threshold change.
- Prediction: wall ratio → ~1.00 (band satisfied by construction); iteration
  counters equal; PRIMARY on−off ≥ +0.10 preserved (fog stays recovered);
  SEALED lightning recovers in ON only → on−off 0.167 (§5 prediction).
- Falsifier: ratio still >1.10 with equal counters → an unlocated O(state)
  loop outside the snapshot (instrument the counters to find it); lightning
  still raw=0 in ON after commit → its loss is node-level or capture-missing,
  escalate to P3 node records.

**P2 — Write-time criterion repair (§3), the gate for everything else.**
Primitives: `KnowledgeIntake` route counters per teaching turn; stamp earned
`source_metadata.edge_kind` on BOTH taught hops at the conversational write
site (the snowfall/sleet/mist one-hop asymmetry); suppress literal
connective nodes ("causes"/"happens") at the parse/write site that creates
them (locate via node provenance dump); investigate first-sighting confidence
cap (`intake.py:751`) — if rehearsal across 4 rounds cannot lift a trusted
hop's confidence toward the earned target, wire the existing
rehearsal-corroboration accumulation (`engine_generation.py:1371-1375` policy)
into the conversational merge, NOT a threshold change.
- Prediction: criterion_rate ≥ 0.60 both pools (≥12/20 each); Class 2 items
  (tide/snowfall/flood/mist/sleet/drought) reach criterion once junk nodes stop
  crowding the 64-budget and the second hop is stamped; Class 1 items need the
  route/typing fix confirmed by counters first.
- Falsifier: counters show writes ARE happening with causal type and earned
  kind yet raw=0 → the drop is node-id resolution (§3d) or traversal direction,
  not intake; junk-node suppression changes unrelated front scores → revert
  (skill §7a: ship behind default-OFF flag, A/B, revert on no gain).

**P3 — Episodic node-index capture & restore (§4), only if P1's falsifier fires.**
Primitives: extend `capture_replay_memory` to record node identity
(label/vector/source_metadata) for episode-created nodes; commit overlay
recreates node+edges from the record, same `EARNED_EDGE_KINDS` admission.
- Prediction: lightning/erosion-class raw=0 items recover in ON, not OFF.
- Falsifier: restored node's chain still fails grade → confidence problem
  (P2), not index problem; any restored node lacks an episodic record →
  REFUSE (forgery guard, unit-test it).

**P4 — Interleaved-replay secondary arm (§6), pre-registered as secondary.**
Primitives: existing concurrent-arm scheduling + the fix's replay window
inserted between B items; matched total turns.
- Prediction: retention ≈ concurrent control (0.875+), confirming replay
  scheduling is the active ingredient in the interleaving protection.
- Falsifier: no advantage over sequential+sleep → the protection is
  encoding-time novelty/context, not replay; report honestly.

**Ordering:** P2 first (it gates the denominator and is unchanged by sleep
design), P1 in parallel (independent code paths), then re-measure; P3 only on
P1's falsifier; P4 as a separately registered follow-up. Every proposal keeps
THETA_ASSERT=0.70, `grade_chain`/`assertable_chain` byte-identical, REM sabotage
intact, the wrong-tag OFF control, and the ±10% band.

---

## 8. Open items for the execution owner (no engine runs were done here)

1. Graph dump post-B for dew/tide/snowfall/lightning/erosion: node presence,
   hop relation_type, edge_kind, confidence, and the creating call site of
   "causes"/"happens" nodes (§3 Class 1/2 drop-point confirmation).
2. Intake route counters (graph write vs episodic_buffer vs revise) per
   teaching turn for the 18 raw=0 items.
3. `remove_node` call-site grep in `ravana_ml/graph.py` + engine turn-time
   maintenance (§4 node-vs-edge question).
4. Restore harness band to 0.90/1.10 (`measure_learn5_cli.py:104,614`) before
   the next reported run (§1).
5. Literature verification patch to §6 (in progress in this investigation).