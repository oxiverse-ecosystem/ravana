# RAVANA — Brain-Science Positioning & Related Work

## How RAVANA's graph-override instantiates Spens & Burgess (2026)

RAVANA implements the hippocampal–neocortical loop as a **dual-process,
generate-then-verify** architecture (Yonelinas 1994 recollection-vs-familiarity):

- **HRR vector composition = familiarity.** The `HRRReasoner` proposes a
  top-k of candidate objects per hop from the bundled (subject, verb)
  structure. Decode cosine is the familiarity strength — fast, but near-tied
  on confusable siblings (lion/tiger/bear ~0.3 correlated), and *non-calibrated*
  (~0.58 decode conf), so it can propose but cannot disambiguate on its own.
- **ConceptGraph exact edge = recollection.** When HRR is uncertain on a hop
  (no top-k candidate is a real graph edge of the current node), the query is
  *deferred* to `graph.infer_chain(verb=verb)` — an exact on-verb edge
  traversal that returns the authoritative object. This is **not** a wholesale
  replacement: HRR still proposes; the graph corrects only where HRR failed.

This is exactly the **"retriever-conditions-generator" loop** of
Spens & Burgess (2026, *RAG-as-hippocampus↔neocortex*, Nature Communications):
the fast parametric/generative system (here HRR familiarity) proposes, and the
hippocampal-style memory (here the graph) conditions / corrects the proposal
when it is uncertain. The canonical CLS deferral (McClelland 1995; O'Reilly
1995) — fast hippocampal storage supports neocortical inference until slow
cortex catches up.

## Contrast with HippoRAG (NeurIPS 2024)

HippoRAG uses the knowledge graph as a **retrieval index**: the graph's
Personalized PageRank over nodes *selects which passages to retrieve*, then an
LLM generates from those passages. The graph is the *router to text*.

RAVANA uses the graph as an **exact-edge corrector for vector composition**:
the graph is not a router to a generator — it is the *authoritative relation
store* that resolves the specific (subject, verb) → object binding HRR got
wrong. HRR does the compositional traversal (A→B, B→C); the graph supplies the
ground-truth edge at the uncertain hop.

**Complementary, not competing.** HippoRAG's graph-augmented retrieval improves
*what context reaches the generator*; RAVANA's graph-override improves *the
correctness of the compositional step itself* by deferring to exact edges when
vector familiarity is unreliable. One could embed HippoRAG-style graph retrieval
as the front-end that populates RAVANA's ConceptGraph, after which RAVANA's
override corrects the compositional read-out — the two address different stages
of the same hippocampal→neocortical loop.

## Calibration honesty

The override never relabels its own success: graph-corrected hops keep the
**HRR cosine** as their confidence (we do NOT replicate the old `confs=1.0`
bug). Per-hop `sources` (`hrr` vs `graph_corrected`) let the benchmark report
HRR-attributed vs graph-corrected fractions separately, so graph correctness is
never counted as HRR transitive reasoning. ECE is computed on HRR top-1 vs
HRR-correctness, avoiding the ECE-worsening trap.

## The VSA resonator (M3) — deliberately held OFF (IV-D)

The iterative HRR resonator (Frady et al. 2020 *Resonator Networks*; Hiratani &
Sompolinsky 2022 *optimal quadratic binding*) is **kept OFF** (`_resonator_max_iter=1`)
until a bounded-convergence proof holds. A guard `resonator_allowed()` runs a
probe of the rebind-subtract operator and **forbids `max_iter>1` unless the
residual contracts monotonically and the decoded label stabilizes** (the
contraction condition). On the current correlated GloVe-bound structures the
probe fails, so the resonator stays OFF *by proof, not by assertion* — and any
attempt to flip it on is auto-disabled. See `experiments/reasoning_bench.py`
lesion arm V.5, which asserts this auto-disable on the confusable set.

## Cite in writeup

- HippoRAG (NeurIPS 2024) — graph as retrieval index (contrast anchor).
- Spens & Burgess (2026, Nature Communications) — RAG-as-HC↔neocortex; the
  retriever-conditions-generator loop RAVANA instantiates with HRR+graph.
- Word2HyperVec (Ayar et al. 2024) — the random-linear-lift method used in
  `dual_code_space.py` (cite as lift-method reference).
- Yonelinas (1994, 2002) — recollection vs familiarity (the dual-process frame).
- McClelland (1995) / O'Reilly (1995) — CLS hippocampal→neocortical deferral.
- Frady et al. (2020); Hiratani & Sompolinsky (2022) — resonator bounds (why M3
  is gated OFF).

## CLS sleep gate (SHY down-selection)

RAVANA's CLS consolidation has **two gates**, and until this work only one was
context-aware:

- **Ingest-time gate (IV-C, done)** — `web_to_graph.learn_text` tags every
  `web_fact` edge with `source_metadata['contexts']` and XdG-protects it from
  being overwritten by a *different* context at write time.
- **Sleep-time gate (this work)** — `sleep.py._prune_weak_edges` was
  weight-only blind: a `web_fact` edge corroborated in 2+ contexts but with low
  weight (e.g. 0.08) was pruned exactly like noise. So XdG protected the synapse
  at write time but sleep deleted it later.

**Fix:** `_prune_weak_edges` now reads `edge.source_metadata['contexts']`; an
edge with `weight < threshold` is exempt from removal when it carries
`>= min_contexts` (default 2) distinct contexts. A second, separate pass runs
`graph.prune_low_quality_edges` to cull `co_occurrence`/`auto_expand` orphan
noise — keeping the two predicates (weight-downscale vs kind-based
noise-removal) separable, never conflated.

**Brain basis.** Synaptic homeostasis hypothesis / SHY (Tononi & Cirelli 2014;
Nere et al. 2013): sleep is competitive down-selection — synapses "reactivated"
/ well-integrated are protected from depression; isolated ones depress. The SHY
paper is explicit: *"a neuron detects suspicious coincidences and protects the
associated synapses from depression … synapses activated in isolation are not
protected and thus depress."* Cross-context corroboration is the computational
analog of "reactivated across multiple offline bouts" → it fits prior structure
→ protect. This is exactly the signal XdG already records in `contexts`.

**Complementarity (van de Ven et al. 2020, Nature Communications):** brains use
both a write-time gate (metaplasticity / XdG) and a sleep-time gate
(replay / pruning) side by side. So the ingest gate + sleep gate are the two
halves of one consolidation mechanism — not redundant.

**Honesty bar (same as IV-C / benchmark):** protection is grounded in
*independent-context corroboration* (a real signal), **not a blind weight
floor**. An edge is pruned unless it carries ≥2 distinct contexts. Auto-expand /
co-occurrence edges carry no `contexts` and stay prunable. No EWC/SI-style weight
importance is added here — XdG + sleep-gate are the two cheap, brain-faithful
first layers the plan named.

**Measurement:** `experiments/measure_sleep_crosscontext.py` (mirrors
`measure_ivc_xdg.py`). Golden: E1 `water—is_a→chemical_compound` weight 0.08,
`contexts=[ctxA,ctxB]`; E2 `noise—related→junk` weight 0.08, `contexts=[ctxA]`.
Run `_prune_weak_edges(threshold=0.1)`:
- gate ON → E1 SURVIVES, E2 PRUNED → **CONFIRMED**
- gate OFF (control) → both PRUNED → proves the gate (not a weight artifact)
  saves E1 → **CONFIRMED**

`experiments/stress_continual.py` additionally runs an end-to-end check through
the real `WebToGraph.learn_text` ingest path (which sets `contexts`) and asserts
the low-weight cross-context fact survives sleep while single-context noise is
pruned.
