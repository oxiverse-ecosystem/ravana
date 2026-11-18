# Conversational teaching now consolidates trusted causal edges

**Branch:** `fix/conversational-causal-trust`
**Instrument (unchanged):** `scripts/measure_learning_curve.py`
**Audit:** `scratch/learning_curve_audit.json`, raw run log `scratch/learning_curve_after2.log`
**Measurement:** commit `47bfe04` code + this branch's source edits, `RAVANA_OFFLINE=1`,
one cold boot, one process, scratch `data_dir` copy, **no `eng.save()`**, weights/ untouched.

---

## 1. The exact drop point

A taught assertion goes through:

```
engine.py:10506  _process_turn_impl
  -> chain_walker.py:_extract_and_store_causal_relations
     -> chain_walker.py:_create_causal_edge
```

`_create_causal_edge` created/strengthened edges with `relation_type="causal"`,
`weight=0.75`, `confidence=0.7`, **but never wrote `edge.source_metadata`**.
`ConceptEdge` therefore kept its default `edge_kind="unknown"`.
`learn/explanation.py::_hop_provenance` maps `edge_kind` to a trusted tag only
for `web_fact`, `definition`, `curriculum`, `curated`; `unknown` becomes `""`,
so `assertable_chain()` declined every chain of taught edges.

Measured before the fix, raw probe output for one teaching turn:

```
TURN 'fog happens because chill causes condensation, and condensation causes fog.'
  NEW CAUSAL EDGES: 14
    chill -> condensation  meta={'edge_kind': 'unknown', 'source': None, ...}
    condensation -> fog    meta={'edge_kind': 'unknown', 'source': None, ...}
  TRUSTED CHAIN for 'fog':
    labels=['chill','condensation','fog'] ... verdict=decline assertable=False
```

So the delegate's "the schema gate rejected 24/24 causal facts" was **not** the
drop point for the conversational channel. `SchemaConsistencyGate.route()`
returns `fast_merge` or `novel_write` for consistent/novel facts and both write;
the conversational channel never called it at all. The web channel's 0/24 was
a downstream consequence: the untrusted cross-product edges created by the
conversational turn occupy the same `(subject, causal)` slots, so the gate's
contradiction detector routes the later web facts to `episodic_buffer`.

A second, independent blocker: `_create_causal_edge` takes a cross product of
*every* content word on each side of a causal connective ("causes →
condensation", "fog → condensation", …). Those high-weight untrusted edges
outscored the two genuine propositions in `enumerate_chains(..., limit=8)` and
pushed them out of the returned set even after they became trusted.

## 2. The fix (mechanism, not a bypass)

1. `chain_walker.py::_consolidate_asserted_causal_facts` extracts the
   **stated propositions** (explicit causal predicates, OpenIE process-verb
   fallback) from a declarative turn, then routes each through the existing
   `KnowledgeIntake` with `Provenance(source_type="user",
   source="conversation", extractor="conversational-assertion",
   operations=["taught-in-conversation"])`. `intake._write_edge` stamps
   `edge_kind="curated"` (its existing user-source map) plus the full source
   record. The provenance gate is unchanged and still fail-closed.
2. It runs **before** the associative extraction in `engine.py`, so the gate
   sees an empty slot instead of the cross-product junk.
3. Questions assert nothing and are never consolidated (a retrieval cue is
   not a source). On turns that did consolidate, the leftover cross-product
   edges are written weak (`w=0.25, c=0.30`) and never rewrite an existing
   trusted edge. On all other turns the legacy weights are unchanged.

Raw output after the fix, identical probe input:

```
TURN 'fog happens because chill causes condensation, and condensation causes fog.'
    chill -> condensation  edge_kind='curated' source='conversation' ops=['taught-in-conversation']
    condensation -> fog    edge_kind='curated' source='conversation' ops=['taught-in-conversation']
  TRUSTED CHAIN for 'fog':
    labels=['chill','condensation','fog'] verdict=assert assertable=True
```

## 3. Brain rule: why this design

- **McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there
  are complementary learning systems in the hippocampus and neocortex:
  Insights from the successes and failures of connectionist models of
  learning and memory. *Psychological Review, 102*(3), 419–457.**
  Fast hippocampal binding captures the episode; slow neocortical
  consolidation extracts reusable structure. Implemented as: the assertion is
  first stored episodically (`_ingest_episodic`), then only its explicit
  propositions are consolidated into the typed graph; the incidental
  cross-product co-activations are left weak and labile.
- **Squire, L. R., & Zola, S. M. (1996). Structure and function of declarative
  and nondeclarative memory systems. *Proceedings of the National Academy of
  Sciences, 93*(24), 13515–13522.**
  Declarative (fact/events) knowledge is initially dependent on the medial
  temporal lobe and becomes gradually independent through consolidation.
  Implemented as: the trusted causal edge is a durable typed graph fact with
  a source record, not an ephemeral conversational echo.
- **Johnson, M. K., & Raye, C. L. (1981). Reality monitoring. *Psychological
  Review, 88*(1), 67–85.**
  Source is not a stored label but an attribution computed from record
  features; internally generated thoughts and externally presented
  information must be distinguished. Implemented as: provenance is carried
  on the edge (`source_type=user`, `operations=[taught-in-conversation]`,
  timestamp), the schema gate still decides the write, and a question or a
  free association cannot acquire a trusted source tag.

The provenance gate was deliberately **not** relaxed: `assertable_chain()`
still requires every hop to be trusted, and the definition-only control still
produces zero causal edges.

## 4. Before / after curve (unmodified instrument)

| Arm | n | before Δ | after Δ | after new trusted edges | after answer-chain rate |
|---|---|---:|---:|---:|---:|
| **TEACH_BOTH** | 12 | 0.000 | **+0.833** | 22 | 0.750 |
| NO_TEACH_CONTROL | 6 | 0.000 | 0.000 | 0 | 0.167 |
| DEF_ONLY_CONTROL | 6 | 0.000 | 0.000 | 0 | 0.000 |
| WEB_UNGATED_POSITIVE | 6 | +0.167 | +0.500 | 4 | 0.333 |

Instrument self-check, run before the curve (unchanged):
`positive_trusted_2hop_detected=true`, `negative_definition_rejected=true`,
`direction_forward_ok=true`; positive chain
`['zza','zzb','zzx']`, verdict `assert`.

Raw hand-verifiable pair (from `scratch/learning_curve_audit.json`):

```
subject fog
  BASE  Q 'why does fog happen?'
  BASE  A 'honestly, fog is a bit outside what i know right now.'
  TEACH 'fog happens because chill causes condensation, and condensation causes fog.'
  POST  Q 'can you explain why fog occurs?'
  POST  A 'chill leads to condensation, and condensation leads to fog'
  hops  fog->condensation [curated], condensation->chill [curated]
```

Definition control, raw: `web facts=[1,1]`, `new_trusted_causal_edges=[]`,
post answer `i don't really have a solid grasp on marble so far ...` — no chain.

### Honest caveats

- The instrument runs all arms in one engine. `hiccup` appears in both TEACH
  and NO_TEACH, so by the time the control runs its **baseline** already has
  the taught chain (baseline 0.167); its *delta* is still 0.000. The positive
  control shares intermediates (`condensation`, `oxidation`, …) with the
  teach arm, so its rise from +0.167 to +0.500 includes some cross-arm
  transfer; it is not a clean independent replication of the gate bypass.
- n=12 taught subjects is a hypothesis test, not a population estimate.
- `intake._fast_merge` still writes novel facts weakly; 2/12 subjects
  (`bruise`, `sweat`) had one hop buffered or low-confidence and did not
  clear the chain gate. No gate threshold was weakened to force them through.
- No `eng.save()` was called. Serialisation durability was checked by a
  pickle round-trip of the graph: the trusted chain and its provenance
  survive; SHA-256 hashes of `weights/` were identical before and after.

## 5. Reproduction

```bash
cd C:/Users/Likhith/Documents/projects/ravana
uv run python -m pytest tests/unit/test_conversational_causal_consolidation.py -q
uv run python -m pytest tests/unit/test_intake.py tests/unit/test_explanation.py \
    tests/unit/test_answer_learning_bridge.py tests/unit/test_gap_driven_web_learning.py -q
uv run python scripts/measure_learning_curve.py     # writes scratch/learning_curve_audit.json
```
