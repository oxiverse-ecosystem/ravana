# Gap -> Web -> Graph -> Answer: brain-grounded design

This document records WHY every design decision in the mechanism-gap learning
loop is what it is. Each decision is the computational analogue of a cited
human-brain / cognitive-science mechanism; none is ad-hoc plumbing. The
implementation lives in `ravana/src/ravana/chat/web_learning.py`
(`close_mechanism_gap`), is wired from the gap detector in
`ravana/src/ravana/chat/engine.py` (`_process_turn_impl`), and writes through
`ravana/src/ravana/web/web_to_graph.py` (`WebToGraph.learn_text`).

## The behavioural gap being closed

Before this change, `why does X happen` with no trusted causal chain recorded a
`FailedQuery` in `_impossible_queries`, boosted curiosity and queued a topic for
background learning — then **stopped**. The gap was logged, never acted on.
This is exactly the failure information-gap theory predicts a curious system
should NOT make: a perceived gap should drive information-seeking, not merely
be archived.

## Citations (peer-reviewed sources, verbatim references)

1. **Loewenstein, G. (1994).** The psychology of curiosity: A review and
   reinterpretation. *Psychological Bulletin, 116*(1), 75–98.
   DOI 10.1037/0033-2909.116.1.75.
   Curiosity is "a form of cognitively induced deprivation that arises from the
   perception of a gap in knowledge or understanding"; curiosity is partly
   driven by the estimated probability that a given piece of information will
   resolve the gap.

2. **Schacter, D. L., & Addis, D. R. (2007).** The cognitive neuroscience of
   constructive memory: Remembering the past and imagining the future.
   *Philosophical Transactions of the Royal Society B: Biological Sciences,
   362*(1481), 773–786. DOI 10.1098/rstb.2007.2087.
   The *constructive episodic simulation* hypothesis: a system "draw[s] on the
   past in a manner that flexibly extracts and recombines elements of previous
   experiences" to simulate events never experienced in that exact form.
   Remembering and imagining share a constructive substrate.

3. **Friston, K. (2010).** The free-energy principle: a unified brain theory?
   *Nature Reviews Neuroscience, 11*(2), 127–138. DOI 10.1038/nrn2787.
   Agents minimise long-term surprise by changing predictions (perception) or
   changing sensory input (action); perception optimises efficacy (learning)
   and gain (attention/salience). Exploration is an action selected for its
   expected (epistemic) value, and that value is bounded.

4. **Gerstenberg, T. (2024).** Counterfactual simulation in causal cognition.
   *Trends in Cognitive Sciences, 28*(10), 924–936.
   DOI 10.1016/j.tics.2024.04.012.
   Causal judgment and responsibility rest on *counterfactual simulation*,
   which needs three ingredients: "a generative mental model of the world, the
   ability to perform interventions on that model, and the capacity to simulate
   the consequences of these interventions."

5. **Johnson, M. K., & Raye, C. L. (1981).** Reality monitoring.
   *Psychological Review, 88*(1), 67–85. DOI 10.1037/0033-295X.88.1.67.
   Memory distinguishes information derived from external sources from
   information generated internally (reasoning/imagination). This is the
   source-monitoring rationale the inherited emit guard already applies to a
   quoted web snippet; we apply the same principle to a graph-enumerated chain.

## Decisions derived from the evidence

| Decision | Mechanism | Citation |
|---|---|---|
| A detected explanatory gap **triggers a targeted read** rather than being only logged. The query names the specific missing cause (`"{subject} causes mechanism why process explained"`), not a generic topic sweep. | A gap is a cognitively induced deprivation; the information sought is selected by its estimated power to close *that* gap. | Loewenstein 1994 |
| The read is **bounded**: at most 2 sources per gap, at most 4 gap-driven reads per session, once per subject, `RAVANA_OFFLINE=1` short-circuits, every error swallowed at the call site. | Epistemic value is bounded; an action that cannot sample the environment has no value. Bounds also prevent an infinite gap->read->new-gap loop. | Friston 2010 |
| A gap counts as **closed only when >=2 sourced causal hops** form a connected chain. One hop is recall; >=2 recombined hops are a simulated mechanism. | Constructive episodic simulation recombines multiple retrieved elements into a novel scenario; a single retrieved association is not a simulation. | Schacter & Addis 2007 |
| Every newly learned causal edge keeps `edge_kind="web_fact"` and additionally records `source_url`, `method="mechanism_gap_read"`, `retrieval_conf`, `converging_sources`, `acquired_by`, `gap_subject`, `gap_query`. | Counterfactual simulation needs a generative model with traceable provenance (was-derived-from the source that stated the relation), so an intervention can be audited. | Gerstenberg 2024 |
| Edge **strength scales with convergent evidence**: `confidence = min(1, extractor_conf + 0.15*(converging_sources - 1))`. One source is provisional (0.70); two independent sources raise it to 0.85. | A generative model may commit to an intervention only as strongly as the independent evidence supporting it; strength is not a flat constant. | Gerstenberg 2024 |
| The learned edges are written as **typed `causal` edges into the graph**, which `eng.save()` persists to the pkl (authoritative on boot) and the SQLite db mirror. The next turn's `why/how` resolver walks them through the existing adequacy + all-hops-provenance gate. | The simulated mechanism must be reconstructible from stored structure, not from a one-off chat string. | Schacter & Addis 2007; Gerstenberg 2024 |
| On a later `why/how` turn, a stored **definition must not short-circuit** the question when a trusted causal chain now exists (`_mechanism_chain_available` in `engine_self_query.py`). | A mechanism question asks for a simulation built on retrieved facts; answering with a bare recollection substitutes recall for constructive simulation. | Schacter & Addis 2007 |
| The source-monitoring final-emit guard **passes through** a `gated_explanation_chain` (it is assembled only from provenance-trusted, adequate graph edges), instead of re-censoring it as self-generated salad. | Source monitoring distinguishes retrieved structure from internally generated free association; a vetted external/trusted source is by construction not word salad. | Johnson & Raye (1981), "Reality monitoring", *Psychological Review* 88(1):67–85 — the same citation the inherited guard already uses for `web_direct_answer`; Schacter & Addis 2007; Gerstenberg 2024 |

### Concrete numbers

- **How many sources?** Two (`_GAP_MAX_SOURCES = 2`). Friston's bounded
  epistemic value licenses a small, finite sample; two sources are enough to
  corroborate a relation (`converging_sources = 2`) while keeping the user turn
  cheap. Not "as many as possible": an unbounded read is not a bounded
  epistemic action and risks an exploration loop.
- **What strength / provenance?** Provenance is a full PROV-style record on
  `edge.source_metadata` (kind, source URL, method, retrieval confidence,
  converging-source count, acquiring mechanism, gap subject/query). Strength
  starts at the OpenIE extractor confidence (0.70 for the process verbs in
  PR #412) and gains 0.15 per additional independently stating source, capped
  at 1.0. `edge_kind` stays `web_fact` so the existing trusted-provenance
  gates (`learn/explanation.py:_hop_provenance`,
  `chat/response_gen.py:_edge_provenanced`) continue to admit the chain.

## Validation instrument

`tests/unit/test_gap_driven_web_learning.py` prints the RAW before/after:

- before: 0 causal edges, gap unresolved;
- after: `water vapour -causal-> clouds -causal-> rain`, each stamped
  `edge_kind=web_fact`, `acquired_by=mechanism_gap`, a real `source_url`;
- next-turn: `enumerate_chains(..., direction="backward")` on the learned
  subject node returns `(['water vapour','clouds','rain'], 'assert', True)`
  and the narrative `water vapour leads to clouds, and clouds leads to rain`;

The pass criterion is the STRUCTURED chain cross-validated against the graph
and its provenance gate — not a regex on a fluent answer. Offline, budget,
per-subject dedup, single-hop, and convergence-strength cases are covered as
negative/positive controls.

### End-to-end confirmation (isolated data_dir copy, no `save()` in probe)

A real `CognitiveChatEngine` was booted from a pristine copy of the shipped
weights, with a stub search that answers ONLY the loop's targeted causal query
(subject `sonoluminescence`, absent from the graph). Raw output:

- BEFORE: `BEFORE causal chain edges: []`
- TURN 1 (`why does sonoluminescence happen`): definition answer, weak-response
  gap recorded, gap registry `[('sonoluminescence', True)]`, targeted query
  `sonoluminescence causes mechanism why process explained` issued.
- AFTER: `('ultrasound','bubble collapse', acquired_by=mechanism_gap,
  edge_kind=web_fact, source_url=https://example.org/ultrasound, k=1)` and
  `('bubble collapse','sonoluminescence', ... https://example.org/sono, k=1)`.
- TURN 2 (same question): the chain was found at adequacy 1.00 and emitted —
  `ultrasound leads to bubble collapse, and bubble collapse leads to
  sonoluminescence`.

Instrumenting the post-generation stages isolated two gates that had to be
corrected for the learned chain to survive: `_consult_internal_knowledge`
returned a stored definition before generation ran, and `_final_emit_guard`
vetoed the chain as `fluent_tautology`. Both are now gated on the SAME
provenance/adequacy property the chain resolver already enforces (see the two
rows above), so the fix is to the routing of a trusted chain, not a probe
change. The probe script lives at
`<scratch>/gap_e2e_probe2.py`; it never calls `eng.save()`.
