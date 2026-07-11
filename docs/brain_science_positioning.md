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

## Situation-Model monitor as a Levelt / Wernicke comprehension loop

The Situation-Model production path (neural decoder + narrative + syntactic
assembly) does **free** generation, then only a permissive degeneracy check
(`_is_word_salad`, whose `>=3`-novel-word safety valve lets fluent-but-false
text through). Free decoding with no reality constraint is exactly the
architecture that produces **word salad** — and its signature in humans is
**Wernicke's (receptive) aphasia**: *"fluent speech that doesn't make sense …
patients are unaware of their errors (anosognosia)"* (Cleveland Clinic; NCBI
StatPearls). The grounding gate is, in effect, a Wernicke-area comprehension
check bolted onto a fluent syntactic production system.

**The residual failure class.** A reply can be grammatically fluent,
subject-anchored, and semantically *empty* — e.g. the live output
`"Gravity semantic pet … gravity semantic going"`,
`"black holes bend spacetime is black holes bend"`,
`"Life semantic people … life causal great"`. The original monitor judged the
*whole paragraph*, so a good sentence donated its novelty/anchoring to a
degenerate tail and the tail rode along (it literally admitted this in its own
docstring). This is the computational analog of a **partial comprehension
lesion**: gist passes but clause-level meaning fails.

**The fix — clause-grained monitoring (Levelt's perceptual-loop monitor).**
Levelt, Roelofs & Meyer (1999); Levelt (1989): the comprehension-based monitor
re-parses the formulated utterance and compares it to the intended message
*before* articulation, via an internal loop (inner speech) and an external loop
(overt). Critically, comprehension is **incremental** — listeners (and the
monitor) assign provisional interpretations clause-by-clause and revise as input
arrives (Novick et al. 2005; Altmann & Kamide 2007). So the monitor must operate
**per sentence, not on the final paragraph**. `_sm_response_grounded` now fails
closed: if ANY clause fails, the whole reply is withheld (the monitor
intercepts one bad clause), and it also runs on every sub-answer of the
decomposition path (clause-level filtering — drop the degenerate clause, keep
the good one, like a human self-repair).

Per clause it checks four things, reusing the SAME notion of "grounded" the
decomposition path already trusts:

1. **Reference** — the clause must touch a verified neighbour or the stored
   definition's content (NOT merely "≥3 novel words"; that safety valve belongs
   to `_is_word_salad` and must not substitute for a verified anchor, or
   fluent-but-false text passes).
2. **Topical coherence** — the fraction of the clause's content words anchored
   to the subject (GloVe cosine ≥ 0.30) and the clause centroid aligned.
   Per-clause, so a degenerate tail cannot hide behind a good clause's
   anchoring (the M4-grade factual gate, now at clause grain).
3. **Conflict / self-reference (Nozari, Dell & Schwartz 2011; Botvinick
   2001)** — a truncated repetition (`"black holes bend … black holes bend"`)
   is a low-conflict / low-novelty production: the same head noun + verb recurs
   with no new argument. Detected via the subject's head chunk recurring
   (GloVe-independent), so it fires even without embeddings.
4. **Per-clause substance / originality (Murphy & Castel 2022)** — a clause
   whose content is only subject words + glue verbs, or only vague
   concept-metawords (`semantic`, `contrastive`, `causal`, `even` … the exact
   filler vocabulary the free decoder emits), asserts nothing specific. Humans
   are *"skilled and unaware"* of the originality of their own fluent output;
   RAVANA supplies the missing per-clause substance check.

**Honesty bar.** The monitor now operates at the **same grain as production
(clause-by-clause)**, closing the partial-comprehension lesion. It errs toward
*withholding*: a degenerate clause → honest uncertainty / reflective response
("I don't know that yet") — which is strictly better than confident garbage,
and correctly routes the query into the learning loop instead of scoring it
0.55 and never learning. The decomposition path skips the monitor's step (1)
graph-presence requirement (its subject was already vetted by
`_decomp_grounded`'s web/association check), so novel-concept answers (black
holes, quantum effects) are not over-suppressed — only their degenerate clauses
are dropped.

**Universal articulation-boundary guard.** The clause monitor was first wired
into the Situation-Model decoder/narrative/syntax gates (H1/H2) and the
decomposition per-sub-answer filter. Tracing the live `ravana_chat.py` run
revealed a residual: when the gated SM and decomposition paths returned None,
the query fell through to the **ventral/dorsal association binders**, which emit
fluent-but-empty text ("Life semantic people, which semantic cannot") with *no*
clause gate of their own (only the old whole-text `_is_word_salad` salve, which
the `>=3`-novel-word valve let through). The fix is applied at the single
terminal point all paths share: `_forward_model_check` — the pre-articulation
inner-speech loop in `engine.py` — now runs the *per-sentence* salad detector and
the clause SM monitor on the **final composed reply**, so a degenerate clause
from ANY production system is intercepted before overt articulation. This is
exactly Levelt's external loop: the formulated utterance is silently re-parsed
clause-by-clause and compared to the gist before it is spoken.

**Measurement.** `experiments/measure_sm_grounding.py` forces the exact
degenerate strings observed in the live `ravana_chat.py` run (Q3 gravity, Q5
black holes, Q8 life) through both `_sm_response_grounded` and the universal
`_forward_model_check`, and asserts they are withheld, while genuinely grounded
answers (Q9 oxiverse; a real web sentence) still pass. Control arm: the old
whole-text `_is_word_salad` *passed* these strings (the `>=3`-novel-word valve),
proving the per-clause monitor — not a GloVe/luck artifact — is what now
withholds them. `tests/unit/test_sm_grounding_gate.py` (19 tests) covers the SM
dispatch, the per-clause regression (good sentence + degenerate tail), the
control arm, the decomposition per-sub-answer drop/keep behavior, and the
universal forward-model guard.
