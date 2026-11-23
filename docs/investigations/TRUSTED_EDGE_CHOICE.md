# H3 — trusted-edge choice for forced-choice answers (`RAVANA_EDGE_CHOICE`)

Status: **static code + pure-helper verification only; no engine process, no
pytest** (parent owns both). This note implements the pre-registered falsifier
**F2** from `RSI_DECLARATION_STORAGE.md`: storage and subject resolution are
already exonerated, so the only remaining change is answer construction. No
grounding gate, threshold, or write path is edited here.

## 1. Evidence this change responds to

`RSI_DECLARATION_PROBE.md` and the parent's live census establish, on identical
state:

- after two teaching turns the graph holds `gale->rainbow` and `mist->thaw`,
  both `causal`, `edge_kind='curated'`, `source='conversation'`;
- after PR #483 the query subject resolves to `thaw`, not an option token;
- the emitted answer is still chatter (`"true thaw — that's a fun image…"`)
  naming no candidate.

Trusted edge present + subject correct + no candidate => F2 is met. The gap is
the answer builder, not storage (`F1`), grounding (`H2`), or subject resolution
(`H1`). Per F2, **do not tune grounding or thresholds further**; this note adds
one answer-construction path and then stops.

## 2. Exact hook

Default **OFF**: `RAVANA_EDGE_CHOICE=1`. Unset or any other value => every path
below returns its input unchanged, byte-for-byte.

Pure helper (module-level in `ravana/src/ravana/chat/response_gen.py`, no engine
import, no state, no env read inside the selector):

- `parse_forced_choice(text) -> (question, candidates)` — structural split on
  the `Options:` frame and list punctuation (`,` / `;` / newline / A) B)
  markers). No item wording, no option URLs, no benchmark token. No frame =>
  `candidates == []`.
- `select_trusted_candidate(subject, candidates, edges) -> Optional[str]` —
  pure selector. A candidate is admitted **only** when an edge:
  1. has `edge_kind` in the already-trusted set
     `{curated, web_fact, definition, curriculum}`,
  2. has provenance `source == 'conversation'`,
  3. has `relation_type` in `{causal, physical_cause}`,
  4. is **incoming** at the subject (`edge.target_label == subject`), and
  5. has `edge.source_label` equal to one of the candidates.
  Distinct supported candidates must **converge to exactly one**; zero or two+
  fails closed (`None` = no answer). No weight, confidence, Wilson gate,
  `THETA_ASSERT`, `THETA_HEDGE`, `fast_merge_threshold`, or GloVe bar is read,
  strengthened, regraded, or lowered.
- `trusted_candidate_edges_from_graph(graph, subject, keyword_index)` — read-only
  adapter resolving endpoint ids to labels via `graph.get_node` (skill: `nodes`
  is a dict, never iterate it as nodes). Stale label-id entries simply drop out.
- `edge_choice_enabled()` — the only env read, `os.environ.get(..., "0")=="1"`.

Wiring (three flag-gated sites, all stateless):

1. `engine.py` `_try_fact_reasoning`, in the forced-choice / triplet-MC block
   (~7513 after this change), after every evidence-based handler abstains and
   before the learned-triplet and fluency fallbacks: returns the selected
   candidate. It can therefore only preempt a guess, never an evidence answer.
2. `response_gen.py` `_human_like_uncertainty`: an early branch names the
   candidate instead of hedging, for any fallback of this query shape that
   reaches the uncertainty emitter. Flag off / no candidate falls straight
   through to the existing hedge text.
3. `engine.py` `process_turn`, at the emission boundary immediately after
   `_process_turn_impl` returns and before the source-monitor gate:
   `_apply_trusted_edge_choice(response, user_input)`. This is the backstop for
   the observed F2 case where a generative path produced fluent chatter: with
   the flag on the final emitted answer is the single trusted candidate, with
   no downstream gate able to rewrite it. It writes **no** instance state, so
   nothing can leak into the next turn.

`ctx.subject` is used only when it actually occurs in the question text;
otherwise the generic question text is used and the selector fails closed.
Non-forced-choice queries never have candidates and are never diverted.

## 3. Brain basis (DOIs verified in `RSI_DECLARATION_STORAGE.md` §3)

1. **Tse, D., et al. (2007). Schemas and memory consolidation. *Science,
   316*(5821), 76–82.** doi:10.1126/science.1135935 — one-trial assimilation
   requires a pre-existing schema. The trusted causal relation vocabulary is
   the analogue of the schema; the single taught edge is the trace that a
   candidate may be assimilated into.
2. **McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there
   are complementary learning systems in the hippocampus and neocortex.
   *Psychological Review, 102*(3), 419–457.**
   doi:10.1037/0033-295X.102.3.419 — rapid storage, gradual extraction of
   structure (mapping to buffer/graph is an ANALOGY, not a homology claim).
3. **Yassa, M. A., & Stark, C. E. L. (2011). Pattern separation in the
   hippocampus. *Trends in Neurosciences, 34*(10), 515–525.**
   doi:10.1016/j.tins.2011.06.006 — convergent traces must keep distinct
   identities; two different candidates do not merge into an answer.
4. **Gentner, D. (1983). Structure-mapping: A theoretical framework for
   analogy. *Cognitive Science, 7*(2), 155–170.**
   doi:10.1207/s15516709cog0702_3 (publisher alias
   10.1016/S0364-0213(83)80009-3) — relations, not object attributes, are
   transferred; the selector keys on directed relation + provenance, never on
   item wording.

Ledger: Tse/McClelland/Yassa/Gentner are SUPPORT for the design (conceptual /
framework where noted), not a claim that RAVANA implements human neural
machinery. Webb et al. 2016 (doi:10.1016/j.neuropsychologia.2016.09.023)
supports the "schemas propose, sources admit" caution; it is why a diverged
set of candidates fails closed.

## 4. Predicted focused-live endpoints (parent to measure, flag ON vs OFF)

Same state, one process, deterministic seed, `RAVANA_EDGE_CHOICE=1` compared
with the flag unset:

- forced-choice query answer names the trusted candidate (`mist` for the
  `mist->thaw` teaching), and the query-turn subject remains `thaw`;
- structural near/far target accuracy **> 0 at k=8**;
- `growth_headroom > 0` (k=8 minus k=1 positive — not a saturated curve);
- foil rejection and negative rejection stay at their current live rates
  (1.00 foil; 1.0 negative at k=8) — negative metrics must not move down;
- flag OFF remains byte-identical to current output, checked by a differential
  probe before the A/B; fake backend arm stays 1.00 with no `engine_answer`
  fields (no live store).

## 5. Falsifier and stop rule

**F2 stop rule:** if a trusted edge is present, `ctx.subject` is correct
(`thaw`), and the emitted answer still names no candidate with this hook
enabled, then the articulation/decision path was **not reached** — the answer
is being built or rewritten somewhere other than the three sites above. In
that case **stop editing this helper** and instrument the actual emitting path
in one pass (all early exits, correction guard, salad flag, forward-model
check, final emit guard, source monitor) before making any further change.
Do not compensate by lowering a gate or weakening convergence: a gain that
only appears after the trust or threshold surface is relaxed is worse than a
flat curve.

**G6 caveat:** this is a deterministic selector wired at the answer-emission
gate. If focused-live near/far reaches 1.00 (or any score saturates), label the
result **G6 saturation — mechanism scaffold, not autonomous recollection**.
A table that hands the grader the right token by construction measures the
table, not evidence monitoring; require the negative/foil metrics to hold and
report `growth_headroom` before claiming a learning gain.

## 6. Files and verification performed here

- Edited: `ravana/src/ravana/chat/response_gen.py` (pure helpers + mixin helper
  + `_human_like_uncertainty` hook), `ravana/src/ravana/chat/engine.py`
  (import, MC-path hook, emission hook, `process_turn` call),
  `tests/unit/test_trusted_edge_choice.py` (new fake-based tests), this doc.
- Ran: `python -m py_compile` on both modules (clean); a plain-Python fake probe
  exercising every selector branch, the graph adapter, both flag states, and
  the unbound engine emission hook (flag ON -> `mist`, flag OFF -> original
  chatter unchanged). **No engine process and no pytest were run**, per task
  constraint; parent must run the test file with the project env.
- Unverified: live focused-live endpoints, real `process_turn` integration,
  and the pytest suite result — all parent-owned.
