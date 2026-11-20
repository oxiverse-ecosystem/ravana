# Why learned trusted chains almost never reach emitted answers (CAR answer-reach codemap)

Scope: investigation only. No code, weights, or other docs were changed. The only
repo artifact owned by this task is this file. Probes live in `scratch/` and boot a
**copy** of the measured cold state (`scratch/learn5_car_data` →
`scratch/car_reach_probe_data`); `weights/` was never a boot or save target.

Probe artifacts (one process each, `PYTHONHASHSEED=0`, `RAVANA_OFFLINE=1`):
- `scratch/car_answer_reach_probe.py` / `.log` — 6 trusted-chain subjects × 6
  question forms = 36 turns. Wraps, in one process, every candidate early exit
  (`_consult_internal_knowledge`, `_mechanism_chain_available`,
  `consult_internal`, `_structured_recall`, `_try_combined_fact_query`,
  `_try_memory_query`, `_detect_and_handle_correction`, `_generate_response`,
  `_why_how_causal_chain`, `_graph_node_id_for_label`, `_final_emit_guard`) and
  all three explanation gate functions. Prints grounded subject, node ids,
  direction, limit, verdicts, raw emitted answer.
- `scratch/car_reach_rank_probe.py` — rank of the first assertable chain under
  `limit=3` vs `limit=64` on a fresh boot.
- `scratch/car_reach_drift_probe.py` / `.log` — same ranks after each live turn,
  to separate "chain absent" from "chain dropped by truncation".

## 1. Why the two numbers diverge (instrument reading, per debugging skill §1)

The +0.80 headline and the ~0.10 answer reach are **not two measurements of the
same decision**:

| Endpoint | Call site | What it actually reads |
|---|---|---|
| graph chain rate (+0.80) | `scripts/measure_learn5_car.py:11-17,239-258` (`trusted_chain_detected`), called at `:320` | Calls `enumerate_chains(nids, graph, "backward", min_steps=2, max_steps=4, limit=64)` on the **node resolved directly from the item's taught label** (`node_ids_for`, `measure_learning_curve.py:143-147`), then grades the object. No user question is involved in the decision. |
| answer reach (≤0.10) | `measure_learn5_car.py:491-494`, via `mlc.answer_verdict` (`measure_learning_curve.py:233-244`) | `generated = bool(why_hits texts)` where the texts are the **return value of `_why_how_causal_chain`**; `reached = text in emitted answer`. Seeking the chain goes through the full question→answer pipeline. |

`probe()` (`measure_learn5_car.py:315-327`) calls `mlc.ask` **first** and the
structured detector **after**, on the graph the turn just mutated. That ordering
matters (see cause 2 and caveat C3).

The two numbers are logically reconciled by one fact: when `_why_how_causal_chain`
*does* return a chain, `_generate_response` returns it verbatim
(`response_gen.py:8634-8636`), so `answer_verdict.reached` is necessarily true —
the two measured records with `has_chain=true` (`ivory`, `tusk` in
`scratch/learn5_car_measured.json`) show exactly that: emitted answers are the
chain narration `"ivory leads to growth, and growth leads to bone"`. Therefore
**answer reach ≈ "did `_why_how_causal_chain` fire with a trusted chain"**, and
low reach means the assembler returned `None`, not that a later stage discarded it.

## 2. Question → emitted answer, with the measured exit at each step

`process_turn` (`engine.py:8041-8103`) → `_process_turn_impl`
(`:8131`): self-reference `:8903-8911`, structured recall `:8915-8929`,
fact reasoning `:8933+`, combined-fact `:9026-9034`, evaluative `:9045-9053`,
`_consult_internal_knowledge` `:9112-9120`, memory `:9128-9144`,
abstract/uncertainty `:9338-9379` → `_generate_response` called at `:11661` →
`_final_emit_guard` at `:12024`.

Inside `_generate_response` (`response_gen.py:8588`): `_why_how_causal_chain`
is called **at the top** (`:8633-8638`), before question-type detection and every
other generator branch, and returns immediately on success (`:8634-8636`).

Measured on all 36 live probes: `_generate_response` was reached (GEN event on
every paraphrase) and `_why_how_causal_chain` was called on every paraphrase.
Zero emitted answers contained a chain narration. So **no earlier router was
the observed drop point** on this state; the loss is inside the chain assembler.

### Per-form evidence (6 subjects: callus, blush, tornado, limestone, velvet, amber)

| Form | Question shape | Grounded subject | Direction chosen | `why_how` result | Emitted answer |
|---|---|---|---|---|---|
| TAUGHT | `X happens because a causes b, and b causes x.` | `a b` (e.g. `friction thickening`) | regex rejects | `None` | uncertainty |
| IDENTICAL | `why does X happen? is it because a causes b, and b causes x?` | `a b` | backward | `None` (5/6) | assertion mirror: `you mentioned: "…taught sentence…"` |
| P0 | `why does X occur?` | `X` (correct, `ck`/`label_map` agree) | **backward** | `None` (6/6) | uncertainty / stored definition |
| P1 | `how is X produced?` | `X` (correct) | **forward** (wrong) | `None` (6/6) | uncertainty |
| P2 | `what is the mechanism behind X?` | `mechanism behind X` | **regex rejects** (no `enumerate_chains` call at all) | `None` (6/6) | `…solid grasp on mechanism…` |
| P3 | `explain how X comes about.` | `X comes` | **forward** (wrong) | `None` (6/6) | uncertainty |

Concrete call evidence (`scratch/car_answer_reach_probe.log`):
- `callus`, P0: `LABELRES('callus')→325`, same id as the harness detector
  (`label_map→[325]`, `ck→[325]`, `backward/64` asserts
  `friction→thickening→callus`). Then
  `EXPL_enumerate_chains ids=[325] dir=backward lim=3 → 3`, all three
  `grade_chain verdict=decline / assertable_chain=False`. The assembler never
  saw the trusted chain, though it exists in the same graph.
- Same subject, P1: `ids=[325] dir=forward lim=3` — the taught edges are
  **incoming** to `callus`, so no forward chain can be the learned mechanism.
- Same subject, P2: `helper_calls=[]`; the regex at `response_gen.py:8496-8498`
  returned before node resolution.

## 3. Ranked root causes (file:line evidence)

### R1 — Direction and admission regex mismatch for mechanism questions (largest measured cause)
`response_gen.py:8496-8498` admits only `why|how|what causes|what makes|explain`;
`what is the mechanism behind X?` matches none → returns `None` before any
lookup. `response_gen.py:8548-8554` sets backward only for
`why|what causes|what makes`; `how is X produced?` and `explain how X comes
about.` therefore walk **forward**, while the taught relation `a→b→x` is
incoming to X (`explanation.py:204-207` confirms `backward = get_incoming`).
The harness measures answer reach on the **immediate** probe
(`measure_learn5_car.py:449` → `PARAPHRASES[1]` = `how is X produced?`,
`:307`), and its cold probe uses `PARAPHRASES[2]` (`:308`, regex-rejected) —
so the reported answer-reach endpoint is biased toward the two forms the
assembler cannot handle. The earlier 0.583 answer-chain rate used
`why does X happen?` and `can you explain why X occurs?`
(`measure_learning_curve.py:361,387`), both `why` → backward, which is why the
path demonstrably worked under that probe style. The same wrong predicate is
duplicated in the pre-generation guard `engine_self_query.py:2460-2465`
and its admission at `:2606-2608`.

### R2 — Candidate truncation at `limit=3` (co-primary; interacts with R1)
`response_gen.py:8551-8554` and `engine_self_query.py:2462-2465` enumerate at
`limit=3`, then grade only those. `explanation.py:255-265` sorts by chain score
and truncates **before** the caller grades. The harness detector uses
`TRAVERSAL_LIMIT=64` (`measure_learn5_car.py:79-87,239-258`) precisely because
the trusted chain ranks below 8 after secular graph growth. Measured fresh-boot
ranks (0-based, `backward/64`): `callus` 0, `blush` 3, `tornado` 21,
`limestone` 21, `velvet` 8, `amber` absent. `blush`/`tornado`/`limestone`/
`velvet` are unreachable at `lim=3` regardless of direction. Worse, ranks are
not stable across turns (`car_reach_drift_probe.log`): `callus` 0 → 41 after a
single `why` turn, `blush` 3 → 2 → 3, `tornado` 21 → 3 → absent. This is a
search-budget defect, identical in class to the already-documented harness fix
(`docs/LEARN5_CAR_MEASURED.md:150-155`); widening the candidate pool does **not**
change `grade_chain` or `assertable_chain`.

### R3 — Wh-cue grounding substitutes a phrase for the taught subject
`what is the mechanism behind X?` grounds to `mechanism behind X` and
`explain how X comes about.` to `X comes` (probe log), as does the taught
surface to the *cause pair* `a b`. Feeding those strings to node resolution
gives no trusted node even if R1/R2 are fixed. This is the known wh-cue trap
(`ravana-debugging` skill §2; `pipeline-stages.md` "coarse upstream parser")
and it is the dominant failure on the cold paraphrase. The harness detector
never sees it because it resolves the item's taught label directly
(`measure_learning_curve.py:143-147`).

### R4 — The definition-consult bypass uses the same weak chain test (latent here)
`engine.py:9112-9120` returns a stored/internal answer before `_generate_response`.
`engine_self_query.py:2606-2608` is supposed to let a mechanism turn skip that
short-circuit, but `_mechanism_chain_available`
(`engine_self_query.py:2446-2468`) repeats R1's direction regex and R2's
`limit=3`, and checks only `assertable_chain` against `_concept_keywords`
(no graph-label fallback). In the live probes `consult_internal`
(`brain_regions.py:542-620`) was called on 16/36 turns and **missed** in this
state, so it was not the observed cause; `tornado` nevertheless emitted a
definition-shaped answer from inside generation while `why_how` still ran.
Fix R1/R2 at both sites or a subject with a stored definition can still be
answered by the definition before the generative chain path.

### R5 — Surface echo is not mechanism reach (interpretation trap, not a defect to fix)
The taught and identical-surface forms ground to `a b`; the identical probe is
answered by the assertion mirror (`you mentioned: "…"`, no GEN event), and the
measured 5/10 "mechanism words in answer" signal
(`measure_learn5_car.py:495-498`) is that echo. Even the two real answer-reach
hits (`ivory`, `tusk`) emitted **forward** chains, not the taught backward
mechanism. This is why the earlier 0.583 style (fresh `why` question, no taught
surface in the question) succeeded where the identical-surface style does not.

## 4. Explicitly ruled out (with evidence)

- **Definition consult winning before the explanation path:** not observed.
  `_generate_response` and `_why_how_causal_chain` were called on all 36 probes;
  consults returned `None`.
- **Quality gate / `_human_like_uncertainty` discarding a produced chain:** no
  chain was produced. All emitted replies carry strategy
  `metacognitive_uncertainty` as the generator's own output, not as a
  post-hoc replacement of a chain.
- **Final emit guard discarding the chain:** `_final_emit_guard` was called
  after each generation and returned the input text unchanged in all 36
  probes. It also explicitly exempts this strategy at
  `engine.py:12831-12854` (`strategy == "gated_explanation_chain"` → return
  text). No weakening needed.
- **Node resolution / provenance at write time:** for P0/P1
  `label_map`, `_concept_keywords` and `_graph_node_id_for_label` all agree on
  the correct node; the grader and `assertable_chain` are byte-identical to the
  harness's and were never loosened. Failures are candidate ranking and
  question interpretation, not missing trust.

## 5. Narrowest correct change

Introduce **one shared mechanism-intent helper** (module level in
`response_gen.py`, imported by `engine_self_query.py`) and use it at the two
assembler/guard sites. Concretely:

1. **One predicate for "asking for the mechanism/cause of X"**: admit
   `why`, `how ... produced|formed|arisen|comes about`, `what is the mechanism
   behind`, `explain how`, `what causes`, `what makes`. Return `backward` for
   all of these; return `forward` only for genuine effect questions
   (`what does X cause`, `what are the effects of X`). Replace the two regexes
   at `response_gen.py:8496-8498,8548-8550` and `engine_self_query.py:2460-2461,2606`.
2. **One candidate budget for candidate enumeration before grading**: raise
   `limit=3` → 64 at `response_gen.py:8551-8554` and
   `engine_self_query.py:2462-2465`. Keep `grade_chain`, `assertable_chain`,
   `_hop_provenance`, thresholds, and the `verdict=="assert" AND
   assertable_chain` filter **unchanged** (the harness's documented
   `TRAVERSAL_LIMIT` fix). Grade across the whole candidate pool, not a
   pre-ranked top-3.
3. **One subject-resolution fallback**: when grounding yields a wh-cue phrase
   (`mechanism behind X`, `X comes`, the `a b` cause pair from a taught
   assertion), match the longest term actually present in `_concept_keywords`
   / graph labels against the raw query before resolving nodes; only then
   fall back to `ctx.subject`. This is the already-proven resolution pattern in
   `_why_how_causal_chain` (`response_gen.py:8529-8545`) extended to the guard.

Why this is narrow and safe: it changes **which questions count as mechanism
questions, how many candidates are examined, and which node is seeded** — it
never changes what counts as a trusted hop. The existing fail-closed gates
still decide; a subject with no trusted chain still gets the current
uncertainty/definition behaviour. Because `_generate_response` returns the
assembled narration verbatim (`:8634-8636`) and the emit guard already exempts
that strategy (`engine.py:12852-12853`), no downstream change is required.

Verification recipe (A/B, identical state, one process, per debugging skill
§1): A/B the unmodified and patched engine on the same copied
`scratch/learn5_car_data`, asking the four `PARAPHRASES` plus the taught and
identical forms for the same ≥20 trusted-chain subjects. Require (a) emitted
answer reach rises, (b) emitted chain labels equal the taught `a→b→x` order
(not a forward chain), (c) `grade_chain`/`assertable_chain` accept/reject
counts on known-bad probes are unchanged, (d) definition-only and k=0 controls
stay flat. If (b) fails, the direction change is incomplete; if (c) or (d)
fails, the change touched a gate and must be reverted.

## 6. Caveats / instrument discipline

- Six subjects are enough to locate a path defect, not to estimate a rate;
  the direction and truncation failures were reproduced on every one of the
  six and on all four paraphrase forms, so the mechanism is not in doubt, but
  the size of the eventual gain must be re-measured at n≥20 with fixed hash
  split.
- C1: `answer_verdict` requires the narration *string* to appear in the
  emitted answer; this only works because successful assembler output is
  returned verbatim. A future change that wraps or hedges that string would
  break the metric without breaking the system — check this before reusing it.
- C2: `why_hits` is not stored in `measure_learn5_car.py`'s per-item records;
  only `answer_verdict.chain_texts` is. The 0/120 hit counts in the old JSON
  were a field-name artifact; the live probe supplies the real call record.
- C3: `probe()` detects the chain *after* asking, on a mutated graph. For one
  checked subject (`amber`) a chain was absent at fresh boot but present at
  `limit=64` after the cold question. The +0.80 itself is therefore measured
  on post-turn state; this does not explain the answer gap (the live path
  still ran first inside that same turn) but it belongs in any re-measurement.
- No threshold, rubric, gate, or weight file was modified. See
  `docs/LEARN5_CAR_MEASURED.md:160-166` for the previously recorded limitation
  that this document maps to code.
