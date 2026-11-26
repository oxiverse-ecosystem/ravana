# CAR sealed adjacent-k monotonicity failure — exact finding (scratch-only)

**Verdict:** the sealed k=1→k=2 reversal of **−0.40** is an **in-process write-time
node-id resolution failure**, not a trust-gate, ranking, or cold-reload defect.
Seven of ten sealed k=2 items had their two curated causal hops attached to
**node ids that no longer existed in `graph.nodes`** yet remained in the engine
label index. `enumerate_chains` walks live nodes only, so the chain existed in
the raw edge table with correct `curated` provenance but yielded
`raw_chains=0` immediately after teaching. The three items whose endpoints
stayed live are exactly the 3/10 that acquired — which is the measured
sealed k=2 rate of **0.30**.

No production file, gate, threshold, rubric or weight store was changed. No
engine was booted. All work is under this worktree's `scratch/car_adjk/`.

---

## 1. The observation being explained

Source: `scratch/learn5_car_measured_PRE_LIVENESS.json` (commit `20aca05`,
`2026-11-21T12:54:54Z`, 120 records, 1333 s), the run logged in
`car_full_p2.log`. It is the only artifact with the given curves.

| pool | k=0 | k=1 | k=2 | k=4 | k=8 | worst adjacent step |
|---|---|---|---|---|---|---|
| primary Δ_cold | 0.0 | 0.8 | 0.8 | 0.7 | 0.8 | −0.10 (PASS) |
| sealed Δ_cold  | 0.0 | 0.7 | **0.3** | 0.5 | 0.9 | **−0.40 (FAIL, bound −0.10)** |

`measure_learn5_car.py:571-580` computes `drops = curve[i+1]-curve[i]` and fails
when `min(drops) < -0.10`; the failing step is `1->2` (`0.3 - 0.7 = -0.4`).

Per-item sealed k=2 audit from that JSON (immediate = in-process, before save):

| subject | baseline chain | immediate `raw_chains` | immediate trusted | cold `raw_chains` |
|---|---|---|---|---|
| boil | 0 | **0** | 0 | 0 |
| bruise | 0 | **0** | 0 | 0 |
| cocoon | 0 | **0** | 0 | 0 |
| echo | 0 | **0** | 0 | 0 |
| snore | 0 | **0** | 0 | 0 |
| tsunami | 0 | **0** | 0 | 0 |
| webbing | 0 | **0** | 0 | 0 |
| hiccup | 0 | 1 | 1 | 1 |
| pearl | 0 | 61 | 12 | 61 |
| voice | 0 | 64 | 15 | 64 |

Immediate and cold `raw_chains` are **identical** for all ten items. The loss is
therefore acquisition under load, before any save/reload — not consolidation
loss. This contradicts the earlier conjecture in `LEARN5_CAR_POSTFIX.md:111-114`
that it was "not a write-path defect"; the later in-process census in
`docs/investigations/CAR_K2_ROOT_CAUSE.md` (same commit, same JSON) located it
precisely.

## 2. Mechanism, with code locations

1. `measure_learn5_car.py:214-218` assigns one k per item over the hash-sorted
   bank; `:471-489` teaches in rounds 1..8 with `elif it["k"] < r: continue`.
   A k=2 item is taught in rounds 1 **and** 2 — the second exposure lands after
   round 1's 100 teaching turns, which trigger ~11 sleep cycles
   (`engine.py:11426-11459`, ~1 sleep per 9 turns; 106 cycles / 988 turns in the
   full run). A k=1 item is taught once in round 1, before that churn.
2. Every sleep cycle calls `prune_low_degree_junk_nodes(tau_low=1, tau_high=8)`
   (`engine_generation.py:1513-1521`; `graph.py:1504-1579`). Weak first-sighting
   edges lose weight; low-degree tokens whose remaining edges have no trusted
   provenance are deleted by `ConceptGraph.remove_node`
   (`ravana_ml/src/ravana_ml/graph.py`).
3. `remove_node` has no access to the engine's `_concept_keywords` / `_all_labels`
   index, so a deleted id stays in both. Before the fix, `WebToGraph._ensure_node`
   returned `self.ge._all_labels.get(label)` unconditionally. The second
   exposure therefore resolved a **dead** id and wrote the second hop to it.
4. The edge is accepted silently: `ConceptGraph.add_edge` does not require live
   endpoints (verified in `run.log` / `results.json`: a dangling `(0,1)` edge is
   written with `edge_kind=curated`).
5. The probe seeds from `label_map(g)` (`measure_learning_curve.py:131-141`),
   which enumerates **live nodes only**. `enumerate_chains`
   (`learn/explanation.py:204-253`) requires both endpoints of each hop to be
   live (`graph.get_node(other_id)` → `None` → `continue`). Result: `raw_chains=0`,
   `graded_assert=0`, `trusted_all_hops=0`, `hop_gates=[]`. `grade_chain` and
   `assertable_chain` never reject anything — there is no candidate to grade.

The seven failures are precisely the shared-cue cluster (`spinning` ×3 for
webbing/cocoon/silk, `impact` for bruise, `heating` for boil, `vibration` for
snore, `displacement` for tsunami, `reflection`/`repetition` for echo). Cue
overload makes those tokens low-degree and prune-eligible; the recorder in
`junk_labels.jsonl` logs pruned labels for exactly this family. `hiccup` uses
unique tokens; `voice` and `pearl` had live endpoints in the audit despite a
shared cue.

**Why this reads as a monotonicity failure:** the second repetition is supposed
to be the consolidation event. Here it is the event that re-writes the trace
onto a dead endpoint, so k=2 ends up *worse* than k=1. The brain analogue is
synaptic tagging and capture: a weak first rep sets a labile tag that must be
captured by a second event; if the tag (provenance + node identity) is erased
first, the second rep cannot stabilize it (Frey & Morris 1997). Sleep is
selective down-selection, not uniform decay: tagged/reactivated synapses must be
protected (Tononi & Cirelli 2014). `assertable_chain` is the production-side
source-monitoring check; it cannot admit a fact whose source node no longer
exists.

## 3. Scratch test (engine-free, exact code in `reproduce.py`)

Imports UNMODIFIED production functions: `ConceptGraph.add_edge/remove_node`,
`WebToGraph._ensure_node`, `enumerate_chains`, `grade_chain`, `assertable_chain`.
Replays the recorded post-sleep state (node absent from `graph.nodes`, id still
in the index) for the ten sealed k=2 items. Arms differ **only** in resolution /
prune guard; the gate and all thresholds are byte-identical across arms.

| arm | what changes | failures acquired (n=7) | successes (n=3) | all ten |
|---|---|---|---|---|
| A0 | pre-fix resolution: stale id trusted unconditionally | **0/7** | 3/3 | **0.30** |
| F2 | production `_ensure_node` live-id check at write choke point | 7/7 | 3/3 | 1.00 |
| F4 | prune-side guard: never prune a node with earned trusted edge | 7/7 | 3/3 | 1.00 |
| F2F4 | both (current production) | 7/7 | 3/3 | 1.00 |

The A0 arm reproduces the measured sealed k=2 rate **0.30** exactly from the
recorded stale-id pattern. Hand-checked example (`bruise`): A0 probe seeds `[2]`,
one dangling curated edge `(0,1)`, `raw_chains=0`; F2 mints fresh id `3` for the
dead label `impact`, writes `3→1→2`, `raw_chains=1`, `graded_assert=1`,
`trusted_all_hops=1`, zero dangling edges.

Controls (all pass):
- **definition-only**: an `is_a` definition edge yields no trusted 2-hop causal
  chain — the control arm stays flat.
- **untrusted competitor**: a `co_occurrence` chain at weight/confidence 0.95
  is still rejected by `assertable_chain`; adding it never raises
  `trusted_all_hops` above the curated chain's count. The fix reorders nothing
  and admits nothing.
- **oracle upper bound (NON-DEPLOYABLE)**: stamping the gold answer key into the
  graph makes even A0 acquire; this shows the information would help, not that
  the mechanism works. The deployable counterpart is F2, which uses only live
  node ids derived from the engine's own store and reaches the same endpoint
  without the answer key.
- **instrument validation**: three explicit input/output checks recorded in
  `results.json`; one initially failed because I expected ≥2 dangling edges —
  one dead hop is sufficient because `enumerate_chains` requires both endpoints
  live. The expectation was corrected; the system was not.

## 4. Status of the fix in this checkout

The current `main` tip (`0829813`) already contains all three guards, so this
finding explains the **historical P2 failure** and why the later stratified run
(`car_full_stratified.log`) is all-PASS with curves [0.0,0.9,0.9,1.0,1.0] /
[0.0,0.9,0.9,0.9,0.9]:

- `ravana/src/ravana/web/web_to_graph.py:152-163` — `_ensure_node` verifies
  `get_node(existing)` and strips the stale id from `_concept_keywords` (F2).
- `ravana_ml/src/ravana_ml/graph.py:1546-1566` — junk prune never removes a node
  carrying an earned `curated`/`web_fact`/`definition`/`curriculum` edge (F4).
- `ravana/src/ravana/chat/engine_generation.py` — label index synced after the
  prune sweep (`test_car_node_liveness.py:92-94` asserts the fix marker).
- `learn/explanation.py:261-289` — F3 trust-first partition prevents high-weight
  untrusted chains from crowding a taught chain out of the traversal budget.

No change is proposed here. If the sealed dip reappears on a fresh 100-item run,
re-check `_concept_keywords` liveness first with a read-only graph census; do
not lower the monotonicity bound or the trust gate.

## 5. Limitations

- This is a **replay** of the recorded stale-id condition, not a blind
  prediction of which items fail; the item-to-dead-label mapping is taken from
  `CAR_K2_ROOT_CAUSE.md`'s census of the P2 saved graph. It establishes that the
  mechanism is *sufficient* to produce the exact observed 7/10 pattern and that
  the live-id guard eliminates it — it is not an independent n=10 sweep.
- No engine was booted, so the A/B does not re-measure answer reach, cold reload
  or the full 100-item interference dynamics. Those are already recorded in the
  P2 artifacts.
- n=10 per k remains low; the sealed −0.40 magnitude varies across seeds
  (−0.20 to −0.30 in the replications cited in `LEARN5_CAR_POSTFIX.md:116-125`),
  but the adverse k1→k2 step persisted in every recorded run.
- The F2/F4 arms are measured on identical synthetic post-prune state; F4's
  effect is "prevent the stale state", F2's is "survive it". Both are real guards
  in the current source; the scratch isolates their contributions rather than
  claiming either alone explains the whole 100-item run.

## 6. Citations (verified)

- Frey, U. & Morris, R.G.M. (1997). Synaptic tagging and long-term potentiation.
  *Nature* 385:533–536. doi:10.1038/385533a0. PMID 9020359. — a weak trace sets a
  transient tag that captures plasticity products from a later strong event.
- Tononi, G. & Cirelli, C. (2014). Sleep and the price of plasticity: from
  synaptic and cellular homeostasis to memory consolidation and integration.
  *Neuron* 81:12–34. PMID 24411729; PMCID PMC3921176. — sleep down-selects
  synapses; repeatedly reactivated / well-integrated synapses are protected.
- Anderson, M.C. & Spellman, B.A. (1995). On the status of inhibitory mechanisms
  in cognition: memory retrieval as a model case. *Psychol Rev* 102:68–100. —
  retrieval of one trace suppresses competitors; basis for trust-first selection
  and for keeping untrusted competitors rejected.
- McClelland, J.L., McNaughton, B.L. & O'Reilly, R.C. (1995). Why there are
  complementary learning systems in the hippocampus and neocortex.
  *Psychol Rev* 102:419–457. — fast labile store vs slow statistical store;
  interleaved replay must not dissolve new traces.

## 7. Reproduce

```
cd C:/Users/Likhith/Documents/projects/ravana_car_adjk
uv run python scratch/car_adjk/reproduce.py
```

Artifacts: `reproduce.py`, `results.json`, `run.log`, this document.
