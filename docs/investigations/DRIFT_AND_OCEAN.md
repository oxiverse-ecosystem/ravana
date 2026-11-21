# DRIFT_AND_OCEAN — investigation

Branch `docs/replay-fix-negative`, HEAD `d7a3495`. Range under investigation:
`466fcd1..HEAD` (17 commits). Execution owner: single sequential engine/pytest
process; no `eng.save()` to `weights/`; no source edits; no commits. Scratch
artifacts: `scratch/ocean_trace.py`, `scratch/ocean_trace_results.json`,
`scratch/drift_*`.

---

## TASK 1 — `if the ocean vanished` counterfactual test failure

### Reproduction (confirmed)

```
RAVANA_OFFLINE=1 PYTHONHASHSEED=0 uv run python -m pytest \
    tests/unit/test_chat_query_fixes.py -q -k ocean
```

Result: `1 failed, 1 passed`. The failing param is
`test_counterfactual_premise_returns[if the ocean vanished-ocean]`:

```
out = engine._simulate_counterfactual(ctx)   # ctx.subject='ocean', associated_concepts=[]
assert out is not None
E   AssertionError: counterfactual bailed for 'if the ocean vanished'
E   assert None is not None
```

The sibling param `what if the sun disappeared` (and all 8 other params) PASS.
The fixture is a **fresh baby engine**, not production weights:
`CognitiveChatEngine(dim=64, seed=42, baby_mode=True, data_dir=<tempdir>)`
(`tests/unit/test_chat_query_fixes.py:48-53`). So this is a cold-start substrate
question, independent of `weights/` state.

### Stage-by-stage trace (one process, `scratch/ocean_trace.py`)

Substrate check on the identical baby engine (`_concept_keywords` + causal
outgoing edges):

| concept | in `_concept_keywords` | causal outgoing edges |
|---|---|---|
| **ocean** | **yes (nid 29)** | **[] (NONE)** |
| sun | yes (nid 235) | light, heat, earth, gravity |
| earth | yes (nid 51) | life, sun |
| moon | yes (nid 267) | gravity |
| water | yes (nid 13) | life, oxidation |
| sea | yes (nid 50) | [] |

Per-case INPUT → OUTPUT (wrapped instance methods, no source edits):

| query | gated chain | `_removal_causal_lines` | `_abductive_counterfactual` | `_causal_forward_simulate` | `_counterfactual_web_escape` | FINAL |
|---|---|---|---|---|---|---|
| **if the ocean vanished** | 0 candidates | **`[]`** | **None** | `[]` | None (offline) | **None** |
| what if the sun disappeared | `sun→light→photosynthesis` (adeq 1.00) | — | — | — | — | gated_explanation_chain |
| what if the earth was destroyed | `earth→sun→light` (1.00) | — | — | — | — | gated_explanation_chain |
| what if the moon was made of cheese | `moon→gravity→earth` (1.00) | — | — | — | — | gated_explanation_chain |
| if humans could photosynthesize | — | — | premise template, coherent | — | — | counterfactual_simulation |
| cats took over the world | — | — | control template, coherent | — | — | counterfactual_simulation |
| dogs seized power | — | — | control template | — | — | counterfactual_simulation |
| if mountains were made of gold | — | — | compositional template | — | — | counterfactual_simulation |
| ai took over the government | — | — | control template | — | — | counterfactual_simulation |
| suppose trees could photosynthesize at night | `tree→fall→vibrate` (1.00) | — | — | — | — | gated_explanation_chain |

### Exact bail point (file:line)

The premise cue **does** match and the scope gate **does not** block. The bail is
the **missing ocean causal skeleton**:

1. Subject resolution resolves `start='ocean'` (opener-strip fallback,
   `response_gen.py:6419-6420`). `vanished` is correctly in `_VERB_BLOCK`
   (`:6253`) so it is never treated as the intervened node.
2. Gated explanation chain runs first (`response_gen.py:6485-6528`):
   `_nids=_ck.get('ocean')[:1]=[29]`, `enumerate_chains(..., forward, min_steps=2)`
   → **0 candidates** (ocean has no causal edges). Trace:
   `gated chain: no assertable chain (0 candidates)`.
3. Premise path `_abductive_counterfactual(ctx,'ocean')` (`:6539` → `:6807`):
   the **removal/absence** pattern matches cue `vanished`
   (`_PREMISE_PATTERNS`, `response_gen.py:6885-6890`, `scope=None` → no scope
   gate). Its consequences lambda calls `_removal_causal_lines('Ocean')`.
4. **`_removal_causal_lines` returns `[]` at `response_gen.py:6776`**: ocean IS in
   the graph (`start_ids=[29]`, `:6721-6725`) but the provenance-gated causal BFS
   (`:6737-6766`) finds **no path** (ocean has zero causal outgoing edges), so it
   hits `if not paths: return []` (`:6767-6776`) — the *designed* honest-decline
   (the empty "everything that depends on X would shift" template was deliberately
   removed; see comment `:6768-6775`).
5. Back in `_abductive_counterfactual`, `lines=[]` is falsy → the premise return
   branch (`:6919`) is skipped; the generic definition-deps path finds no clean
   grounded causal deps → `return None` (`:6989`).
6. `_causal_forward_simulate('ocean')` → `[]` (`:6544-6547`);
   `_counterfactual_web_escape` → None offline (`:6557-6559`); **`return None` at
   `response_gen.py:6560`**.

So: **not** a premise-pattern miss (`vanished` is a listed cue), **not** a
control-scope gate (`scope=None`), **not** web-disabled-only (web escape is the
last resort). The binding cause is that **ocean has no causal edges in the seeded
world-skeleton**, while sun/earth/moon/water/tree do.

### Why sun passes and ocean does not

The authored physics skeleton `_seed_physics_causal`
(`ravana/src/ravana/chat/engine_reasoning.py:3185-3251`, edges list `:3197-3241`)
seeds 23 causal edges: `sun→light/heat/earth`, `light→photosynthesis→plants→animals`,
`earth→life`, `gravity→orbit`, `water→life`, `tree→fall→vibrate→air→sound`, etc.
**`ocean` is absent from this list.** sun/earth/moon therefore have ≥2-hop
assertable chains and the gated explanation chain fires; ocean has none, so every
downstream path fail-closes to None.

### Probe vs system verdict (skill §1 / §3d)

- The **gates are behaving correctly**: the empty generic template is suppressed
  and the system honestly declines (None) rather than emitting fluent nonsense.
  This is the designed fail-closed behavior, not a logic bug.
- The defect is a **substrate gap**: `ocean` is a world-scale entity exactly like
  `sun`/`earth`/`moon`/`water`, but was omitted from the authored causal skeleton.
  The probe's pass criterion (a real, non-filler counterfactual) is valid; it
  assumes a precondition (ocean causal material) the cold-start substrate does not
  supply. Per skill §3d this is "a probe that does not create the state its
  capability needs" — but here the right resolution is to supply the substrate,
  not weaken the probe.
- **Pre-existing, unrelated to the drift**: `527f401` (the only production change
  in range) touched `_why_how_causal_chain` only, not the counterfactual path or
  the skeleton. The ocean failure reproduces identically on baseline `466fcd1`.

### Narrowest proposed fix (NOT implemented)

**Primary (substrate, brain-faithful):** add genuine ocean causal priors to the
world-skeleton edge list at `engine_reasoning.py:3197-3241`, e.g. a water-cycle
chain `ocean→evaporation→clouds→rain` (weights ~0.85), so the gated chain / removal
BFS has a ≥2-hop assertable path. Constraints from the skills:
- Author the edges as real causal priors (avoid a false co-occurrence prior like
  the removed `gravity→earth`, `:3209-3222`); they must pass `assertable_chain`
  provenance the same way the existing skeleton edges do.
- Skill §3: a cold-start-only seed writes zero edges on a normal persisted boot —
  if the seed must also hold in production, add it to BOTH boot paths and verify
  the count on a cold load. For THIS test (baby_mode cold start) the
  `_seed_physics_causal` path is the one exercised.
- Skill §7a: measure the seed (count assertable chains from `ocean` AND check the
  emitted answer) before keeping; revert if it adds no assertable chain.

**Alternative (narrower, template):** give the removal pattern a concrete
subject-scoped consequence for water-world subjects (`ocean`/`sea`/`water`) at
`response_gen.py:6885-6890` instead of delegating solely to
`_removal_causal_lines` (which returns `[]` off-graph) — e.g. "the water cycle
would stall; rain and rivers would fade; sea life would die". This must still pass
`_causal_narrative_degenerate` (`:6934`). Less brain-faithful than building the
substrate; prefer the skeleton seed.

**Do NOT** loosen the degenerate gate or re-add the empty "everything that depends
on X would shift" template — that is the exact defect the gate exists to catch.

---

## TASK 2 — grounded front drift 57 → 35 (pending A/B; section updated below)

### Established facts (read-only)

- Baseline at `466fcd1` (perf: linear `detect_gaps`, PR #438) + curriculum
  scaffold: conversation 95, **grounded 57**, opinions 80, thinking 13, speaking 7,
  code 100, LL 80, **mean 62** (`docs/SEVEN_FRONTS_STATUS.md:5-13`).
- The `detect_gaps` perf fix itself moved grounded **17 → 58** (perf-fix-only) /
  **57** (perf+scaffold) — commit message of `466fcd1`. The old O(nodes×edges) scan
  read absent `edge.source/target`, gave every node degree 0, flagged every node as
  a gap. **That measurement-validity change is already baked into the 57 baseline**;
  it is NOT part of the 57→35 drift.
- Current on-disk baseline (`scratch/front_probes_n30_summary.json`, mtime Nov 20
  14:59): conversation 93, **grounded 35**, opinions 90, thinking 10, speaking 7,
  code 100, LL 80, **mean 59** — matches the task's "current baseline".
- **Only ONE production source file changed in `466fcd1..HEAD`**:
  `ravana/src/ravana/chat/response_gen.py`, via commit **`527f401`** ("fix(car):
  make mechanism questions address learned chains", PR #454). Everything else in
  the 17 commits is docs, new bench scripts, web fixtures, and tests.
  `scripts/measure_fronts.py` and `scripts/front_gates.py` are **UNCHANGED** in the
  range (verified `git diff --stat` empty) → the instrument/probes/judges are
  identical, so the drift is not a probe-set change.
- `527f401` diff: added `mechanism_intent()` + `longest_label_seed()`; rewrote
  `_why_how_causal_chain` admission/direction to use `mechanism_intent`, and raised
  `enumerate_chains` **limit 3 → 64**. `longest_label_seed` is defined but NOT
  called anywhere in `ravana/src` (grep: only the def at `:253`).
- **`weights/` is NOT committed** (`git ls-files weights/` empty) → it is a live,
  drifting store. The historical 57 was measured on the weights state of Nov 19;
  many harness runs since (web capture/frozen, learn5, car_reach) may have changed
  it.

### Current grounded per-probe evidence (n=60, score 35 = 21/60 pass)

Failure-reason histogram: `deflecting` 8, `no_predicate;deflecting` 11, and 20
`subject_missing(...);...;deflecting` (all hard "role of / drives / controls /
determines" probes). The dominant failure register is **deflection** ("honestly, X
is a bit outside what i know", "i don't really have a solid grasp on X", "hmm, i'm
not totally sure about X"). Several PASSING answers are **wrong-sense definitions**
(structural pass, not truth): `atom`→"a desktop application built with HTML,
JavaScript, CSS, and Node" (actually FAILS: subject missing), `electron`→"the former
Atom Shell…" (Electron framework), `dreams`→"a fun platform game… bouncy cheese
wheel". This is the wrong-sense/disambiguation store poisoning the skill warns about
(§2n) — a **weights-state** signature, not a code signature.

(Full per-probe table and old-vs-new A/B appended below once the runs complete.)

---

## RESOLUTION (2026-11-20, parent execution owner)

### Decisive A/B — code exonerated
`scratch/drift_grounded.py` runs the identical instrument (`measure_fronts.GROUNDED`
+ `judge_grounded`, n=60) on the production weights, offline, one process.
- **new** arm (HEAD `response_gen.py`): grounded **33** (20/60).
- **old** arm (`response_gen.py` reverted to `466fcd1`, same weights/seed):
  grounded **33** (20/60). **0 of 60 rows differ** (byte-identical answers).
→ The only production code change in range (`527f401`) has **zero** effect on the
grounded front. The drift is entirely **weights-state**, not code.

### Root cause — truncated authoritative pickle + provenance-less mirror
1. `weights/ravana_weights.pkl` (5,118,689 B, mtime **2026-11-19 22:42**) is
   **truncated**: `pickle.load` raises `EOFError: Ran out of input`.
   `pickletools.genops` parses 655,675 ops over the whole file and stops only at
   the missing final `STOP` + outer-dict `SETITEMS` — a save killed mid-write.
2. **Cause of truncation:** `_safe_pickle_dump` wrote directly to the target
   (`open(fpath,'wb'); pickle.dump`). The Nov 19 22:03 web-arm run was killed
   (SIGTERM/timeout) mid-dump, leaving a half-written file. The `.sha` sidecar
   (`10530d4335d9e37e`) no longer matches the file.
3. **Silent cold start:** `_load()` raised on the truncated pkl → returned
   `False` → `__init__` cold-started (209 PMI concepts + 15 common facts + 23
   physics edges), discarding the learned graph. The existing SQLite fallback
   only fired for an *empty graph object*, not an *unparseable pkl*. Boot log:
   `[Load error] Ran out of input` then `[PMI] Seeded 209 concepts`.
4. **Provenance-less mirror:** `CognitiveDB.save_graph` persisted topology only
   (no `source_metadata` column), so even a db rehydration restored structure
   but the trust gate fail-closed on every learned edge (`edge_kind` unknown).
5. The web-arm run also grew the graph 732→2167 words and overwrote the
   definitions mirror with wrong-sense entries (`atom`→Electron app,
   `electron`→Atom Shell, `dreams`→platform game, `iron`→Rust framework,
   `cloud`→managed storage, `pressure`→game mechanic) and truncated snippets
   (`dew`→"used as a symbol", `osmosis`→"…as illustrated in Fig").

### Repair performed (`scratch/repair_weights.py`)
- Salvaged the truncated pkl with the pure-Python unpickler: at EOF the memo
  holds **all 2176 ConceptNode objects** and **3554 ConceptEdge objects WITH
  provenance** (61 curated, 55 web_fact, 3 curriculum, 122 co_occurrence) even
  though the outer dict's `SETITEMS` never ran.
- Rehydrated the **complete structure** (2175 nodes / 18933 edges) from the ACID
  SQLite mirror, then overlaid the salvaged node objects and edge provenance.
- Rebuilt `concept_keywords`, took definitions from the db metadata mirror,
  defaulted non-graph state from a throwaway baby engine, and wrote a clean
  15.5 MB pkl **atomically**. Corrupt file preserved as
  `weights/ravana_weights.pkl.corrupt-20261119`.
- Verified: clean unpickle, cold boot loads 2167 words, heart/frost/rainbow/
  blood/ocean/climate nodes present, 54 curriculum edges re-seeded.

### Durability fixes implemented + verified (branch `fix/weight-store-durability`)
- **A — atomic save** (`engine_persistence.py:_atomic_pickle_dump`): dump to a
  sibling tmp, `fsync`, `os.replace`. A killed save can never truncate the
  authoritative snapshot again. Both write sites (direct + sanitizer) use it.
- **B — load resilience** (`engine.py:_load`): on pkl parse failure, rehydrate
  the graph from the SQLite mirror with a loud `[Load recover]` warning instead
  of a silent cold start; non-graph state resets to defaults. Brain basis:
  complementary learning systems — the slow neocortical mirror retains structure
  when the fast hippocampal snapshot is damaged (McClelland, McNaughton &
  O'Reilly 1995, *Psychol Rev* 102:419–457).
- **C — provenance mirror** (`storage/db.py`): added `source_metadata` BLOB to
  `edges`/`concepts` (idempotent migration), persisted on `save_graph`, restored
  on `load_graph`. The mirror is now a TRUE backup of the trusted graph.
- **Verified** (`scratch/verify_durability.py`): normal boot + atomic save OK;
  mirror holds `source_metadata` for all 18,984 edges; a deliberately truncated
  pkl + mirror boots via `[Load recover]` and rehydrates 2167 nodes / 18984 edges
  **with provenance** (curriculum 54, web_fact 55, curated 61). Targeted tests:
  `test_db_persistence` + `test_provenance_persistence` + `test_sleep_episodic_replay`
  = **9 passed**.

### Honest grounded status after repair
Grounded re-measured on the repaired state: **33** (20/60), +2 flips vs the
cold-start baseline (lightning gained, oxygen lost). The repair restores
structure and prevents recurrence, but the **Nov-19 web_fact provenance for the
medium causal probes was destroyed by the truncation and never mirrored** — it
is permanently lost (those edges rehydrate as `unknown`, trust gate fail-closes).
The 14 easy definition probes (canyon/comet/forest/galaxy/…) lack definitions in
the overwritten mirror. The 20 hard probes (`what is the role of X` / `what
drives X`) resolve subject→`"that"` (a pre-existing parser gap, code-identical
A/B; bail at `response_gen.py:_reflective_response:8056` → `_human_like_uncertainty`).
**Restoring grounded to 57 requires re-learning the lost web facts** (the loop
must re-acquire them with provenance); the durability fixes guarantee that
future learning survives a killed save. The 20-hard-probe subject resolution is
the next capability lever (separate from the drift).

**Post-CAR-fix re-measure (F1/F3/F5 applied):** `scratch/drift_grounded_postfix.json`
= **33 (20/60), 0 flips** vs the repaired baseline — the CAR trust-first ranking
and guard wiring do NOT regress grounded (F3/F5 falsifiers satisfied).

### Ocean test — FIXED
`_seed_physics_causal` (`engine_reasoning.py`) had no `ocean` causal edges, so
"if the ocean vanished" fail-closed to None on every path while sun/earth/moon
passed. Added genuine water-cycle priors `ocean→evaporation→clouds→rain` and
`ocean→life`. `tests/unit/test_chat_query_fixes.py -k "counterfactual_premise or
conditional_detection"` = **17 passed** (ocean + all siblings).