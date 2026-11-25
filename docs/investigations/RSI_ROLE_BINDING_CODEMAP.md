# RSI analogical role binding — code map and smallest-change plan

Status: **STATIC code reading only.** No engine boot, no `process_turn`, no
pytest, no source edits beyond this document. All line numbers are at main
`dbb3e7847f3de75809383d31bf068fb3ed72448e`.

Scope: map the existing "trusted demonstration → unassertable schema hypothesis
→ forced-choice ranking" path, locate where relation type / edge direction
live, and specify the smallest set of functions that must change so the query
`what is true of thaw?` is answered by **binding relational roles
(subject → r1, object → r2)** from demonstrations rather than by
`ConceptNode`/GloVe **node-vector similarity**.

Legend:

* **[EVIDENCE]** — directly read from the cited file:line in this worktree.
* **[INFERENCE]** — design conclusion drawn from that evidence; not yet code.

Reference documents (also static): `RSI_SCHEMA_WIRING.md`,
`RSI_ALL_RELATIONS.md`, `RSI_DECLARATION_STORAGE.md`, `TRUSTED_EDGE_CHOICE.md`,
`PDR_RSI_UNSATURATED.md`. The live failure being addressed is recorded in
`RSI_ALL_RELATIONS.md:6-18` (near 0.50, far 0.00; CAUSE near works, PART_OF /
PRECEDES wrong; foil and negative correctly 0) and in the module docstring at
`schema_induction.py:3-19`.

---

## 1. How a trusted demonstration becomes a schema hypothesis today

### 1.1 Write side (demonstrations become provenance-trusted edges)

**[EVIDENCE]**

* `ravana/src/ravana/chat/chain_walker.py:878-930` —
  `_consolidate_asserted_causal_facts(text)`. For each `(cause, effect)` from
  `_asserted_causal_propositions`, it builds
  `Fact(subject=cause, relation="causes", obj=effect, confidence=0.9,
  source_sentence=text)` (`:910-911`) and
  `Provenance(source="conversation", source_type="user",
  extractor="conversational-assertion",
  operations=("taught-in-conversation",))` (`:912-915`), then calls
  `intake.intake(fact, prov, salience=1.0)` at `:916`. A question asserts
  nothing (`:887-888`).
* `chain_walker.py:1052-1070` `_consolidate_asserted_relations` keeps the
  CAUSE path and adds `PART_OF / PRECEDES / RESEMBLES` via
  `_consolidate_asserted_non_causal_relations` (`:1072-1134`), which uses the
  same `Provenance(source="conversation", source_type="user")` write and
  defensively restores `meta["edge_kind"] = "curated"` on unpickled edges
  (`:1113-1121`). Inverse phrases swap roles at canonicalisation (`:943-945`,
  `:989-1051`).
* `ravana/src/ravana/chat/engine.py:10850-10867` — the only declarative
  consolidation call inside a turn: duck-typed `getattr(self,
  "_consolidate_asserted_relations", None)` with fallback to
  `_consolidate_asserted_causal_facts`, invoked as `_relation_consolidator(
  user_input)` at `:10865`.
* `ravana/src/ravana/learn/intake.py:74-81` — `_EDGE_KIND_FOR_SOURCE` maps
  `source_type="user"` → `edge_kind="curated"` (`:80`). `_write_edge`
  (`intake.py:705-762`) persists `Provenance.as_metadata()` into
  `edge.source_metadata` (`:750-751`) and stamps `edge_kind` (`:760-761`).
  `Fact.relations → graph relation_type` mapping is at `intake.py:262-288`
  (per `RSI_ALL_RELATIONS.md:42-44`).

### 1.2 Read side (trusted edges → traces → hypothesis → rank)

**[EVIDENCE]**

* `ravana/src/ravana/chat/response_gen.py:22-25` imports the pure helpers from
  `.schema_induction` (no engine import, no module state).
* `schema_induction.py:62-72` — flags and trust constants: `ENV_FLAG =
  "RAVANA_RSI_DECLARATION_STORAGE"` (`:62`), `SCHEMA_ENV_FLAG =
  "RAVANA_SCHEMA"` (`:66`), `TRUSTED_EDGE_KINDS = {web_fact, definition,
  curriculum, curated}` (`:70-71`), `TRUSTED_SOURCE = "conversation"` (`:72`).
  `schema_enabled()` (`:83-93`) is the default-OFF gate.
* `schema_induction.py:96-115` — `_is_trusted_edge`, the shared provenance
  predicate. It reads `edge.source_metadata` **as a dict** (`:108-110`) and
  falls back to attribute reads for duck-typed fakes (`:111-114`); it widens
  nothing (`:96-101`).
* `response_gen.py:7689-7734` — `ResponseGenMixin._schema_candidate(raw_input,
  subject)`, the hook. Exact order:

  1. `if not schema_enabled(): return None` (`:7697-7698`).
  2. `parse_forced_choice(raw_input)` (`:7700`, implementation `:74-101`);
     `<2` options → `None` (`:7701-7702`); subject resolved with the
     head-noun fallback (`:7703-7708`).
  3. **Defer to a directly taught edge** (`:7710-7716`): builds incident
     `CandidateEdge`s with `trusted_candidate_edges_from_graph` (`:7712-7714`)
     and returns `None` if `select_trusted_candidate` decides. (Selector at
     `:104-150`; direction is incoming-only `target_label == subject` at
     `:143`, source must be a candidate at `:145-147`.)
  4. `edges = schema_trusted_edges_from_graph(self.graph)` (`:7720`; adapter
     at `:196-218`, which resolves `(sid, tid)` keys to node labels and reads
     `edge.relation_type` + `source_metadata.edge_kind/source` at `:207-217`).
  5. `traces = traces_from_edges(edges)` (`:7721`; implementation
     `schema_induction.py:316-351`). One distinct `(subject, relation,
     object)` triple = one demonstration; duplicates deduped by `Declaration.key`
     (`:341-344`); optional `relation_map` normalisation (`:339-340`); trace id
     `"%s:%s->%s"` (`:349`). **This call passes no `relation_map`**, so the
     graph's own relation strings are used verbatim [INFERENCE: cross-vocabulary
     convergence between graph `causal/part_of/temporal/analogical` and harness
     `CAUSE/PART_OF/PRECEDES/RESEMBLES` is therefore not normalised].
  6. `hypotheses = induce_schema_hypotheses(traces)` (`:7722`; implementation
     `schema_induction.py:275-296`). Groups by `declaration.relation` (`:284-288`),
     requires `>= min_support` **distinct trace ids** (`:291`), records sorted
     `instances` pairs (`:293-295`), and constructs `assertable=False`
     (`:295`), forced off again by `__post_init__` (`:240-242`).
  7. `pairs = [pair for h in hypotheses for pair in h.instances]` (`:7726`)
     then `cscores, iscores = schema_association_scores_from_graph(graph,
     index, subj, opts, pairs, self._glove_vector)` (`:7727-7730`).
  8. `rank_candidates_for_unseen_subject(subj, opts, hypotheses, traces,
     cscores, iscores)` (`:7731-7732`; implementation
     `schema_induction.py:354-423`). Fail-closed conditions: admitted
     hypothesis must be non-assertable, `support >= 2` and source-admitted by
     `admit_for_query` (`:383-387`; `admit_for_query` at `:299-311`); congruence
     floor `floor = min(instance_scores.values())` (`:391-395`); candidates
     with `score <= 0` or `< floor` dropped (`:403-415`); **strict** winner
     over runner-up by `min_margin` (`:418-423`).

* **The vector-similarity step, precisely** — this is the part role binding
  must replace, and it is *not* in `schema_induction.py`; it lives in the graph
  adapter `response_gen.py:221-285` (`schema_association_scores_from_graph`):

  | Step | File:line | Operation |
  |---|---|---|
  | label → node | `response_gen.py:235-246` (`_node`) | keyword index, else label scan over `graph.nodes` |
  | node → vector | `response_gen.py:248-257` (`_vec`) | `nd.vector`, lazy `glove_vector(label)` fallback |
  | **cosine** | `response_gen.py:259-268` (`_cos`) | `float(np.dot(a,b)) / (norm(a)*norm(b))`; `0.0` if either vector is `None` or norm is 0 |
  | **subject↔candidate score** | `response_gen.py:270-276` | `svec = _vec(_node(subject)); scores[key] = _cos(svec, _vec(_node(c)))` |
  | **demonstrated-pair score** | `response_gen.py:277-285` | `inst[(s,o)] = _cos(_vec(_node(s)), _vec(_node(o)))` for `v > 0` |

  These two dicts are passed as `candidate_scores` / `instance_scores` into
  `schema_induction.py:354-423`, where the instance cosines set the floor
  (`:391-395`) and the subject-candidate cosines decide the winner
  (`:403-423`). The stored node vector is `ConceptNode.vector`
  (`ravana_ml/src/ravana_ml/graph.py:63`). This is **node-vector similarity**,
  not relational role mapping: for `thaw` the winner is whichever option is
  nearest `thaw` in embedding space, regardless of which role it played in the
  demonstrations.

* Emission surfaces [EVIDENCE]: the hook is called immediately after H3 in the
  hedge path `response_gen.py:7743-7768` (H3 at `:7747-7754`, H4 defensive
  `getattr(self, "_schema_candidate", None)` at `:7759-7768`), and again at the
  `process_turn` boundary `engine.py:8257-8275` so early-return turns are
  covered. H3's boundary twin is `_apply_trusted_edge_choice`
  (`engine.py:8155-8185`, called `:8249-8256`).

**[INFERENCE]** The minimal conceptual fix is to keep the provenance/hypothesis
half byte-identical and change only **what is scored**: bind the query subject
to the demonstrations' `r1` slot and score each candidate in the `r2` slot
(i.e. compare candidate vectors to demonstrated *objects*, weighted by how
well the subject fits the demonstrated *subjects*), instead of comparing the
candidate to the query subject directly.

---

## 2. Where relation type and edge direction are stored / retrieved

**[EVIDENCE]**

* `ravana_ml/src/ravana_ml/graph.py:166-257` — `ConceptEdge`:
  * endpoints `self.source` / `self.target` are integer node IDs (`:198-199`);
  * `self.relation_type` (default `"semantic"`) at `:210`, allowed vocabulary
    and default weights at `:168-181` (`causal`, `temporal`, `analogical`,
    `physical_cause`, `part_of`-style custom strings are just values);
  * `self.predicate_token_id` (`:211-215`) optionally carries the creating verb
    token — a natural extra carrier for a relation word, currently unused by
    the RSI path;
  * `self.source_metadata` dict (`:238-257`) holds `edge_kind`, `source`,
    `source_url`, `source_type`, `operations`, etc.
* `graph.py:279-302` — `__getattr__` lazy defaults (`:279-290`, includes
  `source_metadata: None` for unpickled edges) and `_ensure_source_metadata()`
  (`:292-302`) for safe writes; `__getstate__` (`:304+`, note `:308`) keeps
  `source_metadata` on save; `__setstate__` at `:471`.
* `graph.py:1135-1150` — `ConceptGraph.nodes: Dict[int, ConceptNode]` (`:1149`)
  and `ConceptGraph.edges: Dict[Tuple[int,int], ConceptEdge]` (`:1150`): the
  dict key **is** the direction `(source, target)`.
* `graph.py:1694-1782` — `add_edge(source, target, ... relation_type=...)`;
  key at `:1719`, adjacency indices `_outgoing[source]` / `_incoming[target]`
  at `:1770-1771`, relation-type bucket at `:1773`. `get_edge` at `:1784-1785`;
  `get_outgoing` / `get_incoming` at `:1583-1589`; `find_similar` at `:2109`.
* `graph.py:58-66` — `ConceptNode.vector` (`:63`), plus `core_vector` /
  `genesis_vector` (`:64-65`) and optional `sensorimotor_vector` (`:66`).
* Read adapters used by the RSI path:
  * `response_gen.py:196-218` `schema_trusted_edges_from_graph` — all edges,
    both directions, labels resolved from the `(sid, tid)` key, relation from
    `edge.relation_type`, provenance from `source_metadata` (`:207-217`).
  * `response_gen.py:153-188` `trusted_candidate_edges_from_graph` — incoming
    only (`_tid in nids`, `:175-177`), produces `CandidateEdge(source_label,
    target_label, relation_type, edge_kind, edge_source)` (`:64-71`, `:181-187`).
  * `schema_induction.py:149-159` `EdgeView(source, relation_type, target,
    edge_kind, edge_source, confidence)` — the read shape used by the pure code.
  * `schema_induction.py:212-221` `Declaration(subject, relation, obj,
    polarity)` with `key = (subject.lower(), relation, obj.lower())`;
    `:224-228` `TrustedTrace(declaration, stamp, trace_id)`; `:231-242`
    `SchemaHypothesis(relation, support, instances, assertable)`.
  * `schema_induction.py:250-272` `parse_declaration` already implements
    **role swap for inverse phrases** (`:269-270`: `subject, obj = obj,
    subject` when `polarity == -1`) and negation rejection (`:255`). Tests at
    `tests/unit/test_schema_induction.py:106-112`.

**[INFERENCE]** The two slots already exist in `Declaration.subject` / `.obj`
and in `SchemaHypothesis.instances`. Role binding therefore needs **no new
graph field**: `r1` is the demo edge's `source`, `r2` is its `target`, and the
query subject is bound to `r1` while each option is scored in `r2`. Note that
the live `_schema_candidate` never calls `parse_declaration`, so inverse-phrase
role orientation is currently inherited from whatever `add_edge` stored; any
new role-alignment helper must keep that orientation (do not re-derive roles
from raw text in the hook).

---

## 3. Smallest set of functions to add / modify to carry bound roles

All items below are **[INFERENCE]** (design); the cited lines are **[EVIDENCE]**
for *where* the change lands.

1. **ADD (pure, no engine import) `schema_induction.role_aligned_candidate_scores`**
   — or keep the math in the graph adapter if `numpy` must stay out of
   `schema_induction`. Contract: given `subject`, `candidates`, `instance_pairs`
   and an injected cosine function, return `Dict[str, float]` where each
   candidate's score is an aggregate over demonstrations of
   `fit(subject → demo_r1) × fit(candidate → demo_r2)` (e.g. mean or min of
   `min(cos(subj, s_i), cos(cand, o_i))`, with `0.0` for missing vectors).
   This is the only new *behaviour*. Input `instance_pairs` already exists at
   `response_gen.py:7726`.

2. **MODIFY `response_gen.schema_association_scores_from_graph`**
   (`response_gen.py:221-285`) minimally: factor the three nested helpers
   `_node` (`:235-246`), `_vec` (`:248-257`), `_cos` (`:259-268`) into
   module-level `_graph_node/_graph_vec/_graph_cos` (behaviour identical),
   and add a sibling `role_aligned_scores_from_graph(graph, keyword_index,
   subject, candidates, instance_pairs, glove_vector=None)` that reuses them.
   Leave the existing subject↔candidate calculation (`:270-276`) intact as the
   byte-identical fallback, so flag-off and H3 deferral cannot change.

3. **MODIFY `schema_induction.rank_candidates_for_unseen_subject`**
   (`schema_induction.py:354-423`) by one **optional** keyword-only parameter,
   e.g. `role_scores: Optional[Mapping[str, float]] = None`. At `:404-407`
   select `scores = role_scores if role_scores is not None else candidate_scores`.
   Every existing gate — admission (`:383-387`), floor (`:391-395`),
   positive/margin/sort (`:403-423`) — is untouched. Default `None` keeps all
   current callers and tests byte-identical.

4. **MODIFY `response_gen.ResponseGenMixin._schema_candidate`**
   (`response_gen.py:7689-7734`) at `:7726-7732` only: compute
   `rscores = role_aligned_scores_from_graph(graph, index, subj, opts, pairs,
   glove)` from the already-computed `pairs`, and pass `role_scores=rscores`
   to the ranker. Keep steps 1-3 (`:7697-7716`) exactly, including the defer to
   the directly taught edge. No new instance attribute is set.

5. **Do NOT change** `traces_from_edges`, `induce_schema_hypotheses`,
   `admit_for_query`, the trust predicate `_is_trusted_edge`, the emission
   hooks, `ConceptEdge`, or `add_edge`. If cross-family vocabulary must be
   normalised, pass the already-supported `relation_map` at
   `response_gen.py:7721` (`schema_induction.py:318,339-340`); keep it optional.

**Why this is the smallest set:** the demonstrable defect is entirely in the
score source. The provenance gate, dedupe, support threshold, unassertable
hypothesis, source admission, strict-margin fail-closed and both emission
surfaces are already correct and protected (§4). Reusing `Declaration.subject`
and `.obj` as `r1/r2` avoids any schema/migration, and an optional parameter
avoids changing ranker callers.

**Explicitly rejected heavier alternatives [INFERENCE]:** `ConceptBinding`
(`graph.py:507-557`, token↔concept with `source`/decay) and
`core/vsa.py:89` `construct_schema_vector(fillers)` are vector-binding
machinery, not relational-role slots over trusted edges; they would add new
state and a new pickled surface without addressing the score source.

---

## 4. Existing tests that protect the gate, foils, flag equivalence, emission

**[EVIDENCE]**

* **Trust gate / provenance**
  * `tests/unit/test_schema_induction.py:57-63` trusted admitted, untrusted
    rejected; `:51-54` flag-off disables lookup.
  * `tests/unit/test_schema_wiring.py:145-148` an untrusted edge cannot seed a
    hypothesis; `:261-268` real `ConceptEdge.source_metadata` is read (and an
    `unknown` / `None` stamp fails closed).
  * `tests/unit/test_rsi_all_relations.py:112-148` every relation family writes
    a trusted stamped edge (`source`, `source_type`, `edge_kind`, `extractor`,
    `operations`, `asserted_relation`); `:151-159` inverse phrases swap roles;
    `:187-201` unknown / hedged / negated / question assertions write nothing;
    `:236-239` untrusted edge cannot seed.
  * `schema_induction.py:104-115` (fallback for duck-typed fakes and reloaded
    mirrors) is exercised by `test_schema_wiring.py:261-268`.
* **Foil / negative rejection**
  * `test_schema_wiring.py:162-167` tie → `None`; `:170-175` hypothesis alone
    never admits (no observed trace → `None`); `:192-200` demonstrated anchor
    below congruence floor rejected; `:203-210` no trusted demos → no
    hypothesis; `:252-258` non-forced-choice query never diverted.
  * `test_rsi_all_relations.py:268-286` foil below floor rejected for
    PART_OF / PRECEDES / RESEMBLES; `:289-305` negated demos induce no
    hypothesis and the hook fails closed; `:244-249` a non-causal trusted edge
    is not direct evidence; `:252-265` direct taught edge wins and the schema
    hook defers.
  * Harness-level controls (not run here): `scripts/measure_learn5_rsi.py:82-97`
    thresholds, `:860-912` C1-C8 (C2 foil power at `:863-867`, C5 negative at
    `:881-886`), `:915-942` `decide()`.
* **Default-off flag equivalence**
  * `test_schema_wiring.py:30-42` autouse fixture clears all three flags;
    `:215-219` flag off is inert; `:222-228` the older `ENV_FLAG` does **not**
    enable the schema path; `:233-237` end-to-end and `:240-249` no cross-turn
    state.
  * `test_rsi_all_relations.py:35-46` same fixture; `:308-316` flag off inert
    for all families.
  * `tests/unit/test_schema_boundary.py:36-41` flag off, or missing hook, leaves
    `process_turn` output unchanged.
* **Emission boundary**
  * `test_schema_boundary.py:7-28` binds the **real**
    `CognitiveChatEngine.process_turn` method onto a duck-typed `_SchemaEngine`
    exposing only `turn_count`, `_last_subject`, `_source_gate_ctx`,
    `_pending_source_context`, `_apply_trusted_edge_choice` and an optional
    `_schema_candidate`; `:30-33` proves a converged schema can decide on an
    early-return path, `:36-41` proves off/absent is inert.
  * `test_schema_wiring.py:180-187` and `test_rsi_all_relations.py:252-265`
    protect H3 > H4 precedence; `engine.py:8249-8287` is the boundary code those
    tests exercise.

**[INFERENCE]** A role-binding change is safe only if all four groups still
pass unchanged; the new tests in §6 must be added *alongside* these, never in
place of them. In particular `test_schema_boundary.py:7-28` defines the
duck-typing contract that any new `self` attribute would violate.

---

## 5. Risks and how the design avoids each

**[INFERENCE]** (each mitigation cites **[EVIDENCE]** anchors.)

1. **Cross-turn state.** Risk: caching a role map / bound subject on `self` so a
   later turn answers from stale bindings. Avoidance: the score mapping is a
   local value returned by a pure helper and passed as an argument; no
   `self._role_*` attribute is introduced. This is exactly the invariant
   `test_schema_wiring.py:240-249` asserts (`vars(fake)` unchanged; removing
   demos flips the answer to `None`), and the hook itself is documented
   stateless at `response_gen.py:7685` and `schema_induction.py:378-379`.
2. **Duck-typed fakes in the shared `process_turn`.** Risk: a new mandatory
   attribute or a direct `self.role_scores` read crashes `_SchemaEngine` and
   every older fake. Avoidance: `process_turn` uses defensive `getattr` +
   `callable` for both hooks (`engine.py:8254-8256`, `:8263-8275`), the hedge
   path does the same (`response_gen.py:7759-7768`), and the read-side adapters
   are **module-level functions** precisely so fakes exposing only `.graph` /
   `._concept_keywords` work (`response_gen.py:191-195`). All new helper inputs
   are explicit parameters with defaults; `test_schema_boundary.py:13-24` stays
   valid.
3. **Provenance overwrite.** Risk: writing role information into
   `source_metadata` clobbers a conversation episode record. Existing guards:
   `engine_graph.py:933-953` (F1) refuses to relabel a `curated` edge whose
   source/acquired_by marks it as a user episode; `chain_walker.py:1113-1121`
   restores `edge_kind` defensively; `intake.py:750-761` owns the original
   stamp. Avoidance: role binding is **read-only** — carry `r1/r2` in local
   `Declaration`/`instances`, never write edge metadata. If metadata is ever
   required, use `edge._ensure_source_metadata()` (`graph.py:292-302`) and
   mirror the episode guard at `engine_graph.py:947-953`; never `md.update`
   wholesale.
4. **Cold-reload sanitizer / unpickled edges.** Risk: after a save/load the new
   signal disappears because unpickling skips `__init__`, or the sanitizer
   flattens a container type. Existing behaviour: `ConceptEdge.__getstate__`
   keeps `source_metadata` (`graph.py:304-320`), while `__getattr__` supplies
   `source_metadata: None` on reloaded edges (`graph.py:270-290`) and
   `_ensure_source_metadata()` rebuilds it (`graph.py:292-302`); the save
   sanitizer shallow-clones non-container objects and preserves mapping
   subclasses (`engine_persistence.py:286-398`, especially `:314-358` and
   `:377-392`); `chain_walker._sanitize_graph` (`:1532-1571`) prunes self-loops
   and question-phrase nodes. Avoidance: depend only on already-persisted
   fields (`node.label`, `node.vector`, `edge.relation_type`,
   `source_metadata.edge_kind/source`) and on `_is_trusted_edge`'s dict/attr
   fallback (`schema_induction.py:104-115`). Add a fake test with
   `source_metadata = None` to prove fail-closed.
5. **Role alignment degenerating into the same node-similarity shortcut.**
   Risk: implementing "binding" as plain `cos(subject, candidate)` reproduces
   the defect. Avoidance: require both slot fits (`subject↔demo_r1` **and**
   `candidate↔demo_r2`) across `>= 2` demos, and keep the floor + strict-margin
   gates (`schema_induction.py:391-423`) untouched. A candidate that is
   embedding-near `thaw` but near no demonstrated object must score 0 and be
   rejected.
6. **Relation-vocabulary mismatch [EVIDENCE for risk]:** `_schema_candidate`
   calls `traces_from_edges(edges)` with no `relation_map` (`response_gen.py:7721`)
   while the harness uses `CAUSE/PART_OF/...` (`measure_learn5_rsi.py:101-126`).
   Avoidance: keep `relation_map` optional and opt into it explicitly per
   family; do not hardcode a family table into the pure helper.

---

## 6. Concrete implementation sequence + fake-based test plan

**[INFERENCE]** Ordered so each step is independently revertible; no engine is
booted until the parent's live A/B.

1. `tests/unit/test_rsi_role_binding.py` (new) — write the fakes first:
   reuse the `FakeNode` / `FakeGraphEdge` / `FakeGraph` / `FakeSelf` shapes from
   `test_schema_wiring.py:47-114`, plus a vector layout where `cos(thaw, mist)`
   is *low* but `cos(thaw, gale)` and `cos(mist, rainbow)` are *high* (two demos
   `gale→rainbow`, `shower→climate`). Assert the **new** role-scorer would pick
   `mist` while the legacy subject-cosine scorer would pick a different option;
   red first.
2. `ravana/src/ravana/chat/response_gen.py` — factor `_node`/`_vec`/`_cos`
   (`:235-268`) to module level with identical behaviour; add
   `role_aligned_scores_from_graph(...)`. Keep
   `schema_association_scores_from_graph` (`:221-285`) call-compatible.
3. `ravana/src/ravana/chat/schema_induction.py` — add the optional
   `role_scores=None` parameter to `rank_candidates_for_unseen_subject`
   (`:354-423`, select the score dict at `:404-407`); no other gate changes.
4. `ravana/src/ravana/chat/response_gen.py` — wire `pairs` (`:7726`) through the
   new scorer and pass `role_scores` at `:7731-7732`; leave `:7697-7716`
   untouched.
5. Extend `tests/unit/test_rsi_role_binding.py` (all fakes, no `process_turn`,
   flags restored by the same autouse fixture pattern as
   `test_schema_wiring.py:30-42`):
   * role alignment picks `mist` where node-only similarity does not;
   * foil: candidate near `thaw` but far from every demo object → `None`;
   * tie → `None` (`test_schema_wiring.py:162-167` pattern);
   * below-floor → `None`; one demo / mismatched relations → `None`;
   * untrusted edge and `source_metadata = None` → `None`;
   * direct taught edge (`mist→thaw`) → hook returns `None` (defer);
   * flag off, legacy `ENV_FLAG` only, and non-forced-choice → `None`;
   * no cross-turn state: `vars(fake)` unchanged after two calls;
   * inverse phrase orientation: role map is built from stored
     `(source, target)`, not re-parsed text.
6. Extend `tests/unit/test_schema_boundary.py:7-28` with a second `_SchemaEngine`
   that omits every new optional attribute, proving the shared `process_turn`
   path stays duck-typed; keep `:36-41` off/absent assertions.
7. Only after all fake tests pass: parent runs the live A/B
   `scripts/measure_learn5_rsi.py --live` with `RAVANA_SCHEMA=1` against the
   same store, checking C2 foil / C5 negative / C6 monotonic / C7 graded
   (`measure_learn5_rsi.py:860-912`) and the F1-F4 falsifiers in
   `RSI_SCHEMA_WIRING.md:134-153`. This subagent does not run it.

**Single highest-risk integration point [INFERENCE]:**
`response_gen.ResponseGenMixin._schema_candidate`
(`ravana/src/ravana/chat/response_gen.py:7689-7734`), specifically the new
`role_scores` wiring at `:7726-7732` as reached from the **second** call site at
`engine.py:8257-8275`. It is the only place where a new optional score source
meets (a) two independent emission surfaces (`response_gen.py:7759-7768` and
`engine.py:8263-8275`), (b) duck-typed fakes that lack the hook entirely, and
(c) the H3-deferral contract (`:7712-7716`). A defaulted parameter or a leaked
`self` attribute there silently breaks flag-off byte-identity and foil/negative
rejection at once — which is why steps 1 and 6 (fake tests, including the
duck-typed boundary) must land before the scorer is wired.
