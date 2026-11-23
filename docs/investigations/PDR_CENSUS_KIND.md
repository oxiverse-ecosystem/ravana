# PDR live census — why the graded `kind` was `unknown` after a real write + cold reload

Scope: STATIC / evidence + code. No engine process was booted and no pytest was
run in this worktree (the parent runs the suite). Owned files:
`scripts/measure_learn5_pdr.py`, `tests/unit/test_learn5_pdr.py`, this document.
No threshold, gate rule, item, sealing split, control definition, decide() rule
or saturation flag was changed.

Input artifact: `scratch/pdr_one_item_probe.log` (parent's real one-item probe,
`C:/Users/Likhith/Documents/projects/ravana/scratch`). Source line numbers below
are from the pre-edit `main` unless marked "new".

---

## 1. The four candidate causes, adjudicated

| Candidate | Verdict | Evidence |
|---|---|---|
| Stamp applied to a different edge than census reads | **CONFIRMED — primary cause** | See §2. `_find_engine_edge` resolves the object by the *exact full-sentence* label; `_census` resolved by *substring* of the target node's label. After the stamped edge is deleted on reload, the substring path grades an unrelated, unstamped edge. |
| `source_metadata` lost on save | **RULED OUT as primary** | `ConceptEdge.__getstate__` keeps `source_metadata` (`ravana_ml/src/ravana_ml/graph.py:304-328`, "source_metadata is NOW KEPT"); the SQLite mirror stores it as a BLOB (`ravana/src/ravana/storage/db.py:85,139,272`). The stamped edge is *deleted*, not unstamped. |
| Census reading a different key | **PARTIALLY — a contributing amplifier** | The census reads the right *metadata key* (`meta.get("edge_kind")`, `_census` `:844-845`) but used the wrong *target-match key* (substring, `:834-838`), and `edge_kind='unknown'` is the **truthy** ConceptEdge default (`graph.py:250`), so the `or relation_type or "unknown"` fallback never fires. |
| Intake route buffering | **RULED OUT for the original** | Probe: `intake=1`, `api_error=None`, graph has 18,985 edges at write. The authored edge existed before save; it disappeared only after reload (18,984). Buffering remains a real route for contradictions, but it is not this failure. |

## 2. Exact causal chain

EVIDENCE (each step is directly observable in code or in the probe log):

1. The authored object is a **full sentence** (`_roster`, `measure_learn5_pdr.py:763-776`;
   claim text in the probe sample: *"Diffusion Diffusion is the movement of molecules
   from an area of high concentration …"*). Measured: 23 lowercase `[a-z']+` tokens.
2. At write, `_ensure_engine_node` (`:533-574`) minted a node whose **label was that
   full sentence**, registered under the full label in `eng._concept_keywords`.
   `RealBackend._real_write` (`:629-665`) then called the real intake and
   `_stamp_engine_edge` (`:611-627`) stamped the resulting edge with
   `edge_kind='curated'`, `source_url=<real url>`, `source_type='seed'`, `active=True`.
   Probe: 18,985 edges, so this edge existed in-process.
3. `eng.save()` (engine.py `:13103`) persists the graph and the whole
   `_concept_keywords` dict in the pickle state (`:13148, :13173`), restored verbatim
   on load (`:13625`).
4. On cold reload the engine calls `_sanitize_graph()` (engine.py `:2169`;
   `chain_walker.py:1328-1377`). It removes every node whose label
   `_is_question_phrase` accepts, **including its incident edges and its keyword
   entries** (`chain_walker.py:1349-1373`). `_is_question_phrase`
   (`chat/constants.py:530-556`) returns `True` for any multi-word phrase with
   **>=5 word tokens** (`:544-547`). The full-sentence object qualifies.
5. Probe confirms step 4 directly: *"[Sanitize] Pruned 1 question-phrase nodes and
   any self-loops from loaded weights"*, and the edge count drops 18,985 → 18,984
   across `save_and_reload`. The single pruned node is the single authored object
   node minted by this one-item probe.
6. The census then cannot find the stamped edge. Its fallback (old `_census`
   `:834-838`) accepted any target label with `key in tnorm or tnorm in key` — a
   **substring** match — and recorded `found[tnorm]` keyed by that label. With the
   authored edge gone, an unstamped cold-start edge from the subject (the probe
   shows several such edges, e.g. `200→203 analogical`, `200→204..206 semantic`)
   matched a fragment of the claim.
7. That unstamped edge carries the ConceptEdge default metadata, whose
   `edge_kind` is the **string** `'unknown'` (`graph.py:238-257`, specifically `:250`).
   Because `'unknown'` is truthy, `meta.get("edge_kind") or getattr(edge,
   "relation_type") or "unknown"` (`:844-845`) returns `'unknown'`; `url` is `None`,
   so `source_url` is `None`. Probe: `query()` returns ONE fact with
   `kind: 'unknown'`, `source_url: None`, `trusted: False`. Every observed field matches.

INFERENCE (tightly constrained by the evidence, not directly printed by the probe):

* The exact target label of the surviving noise edge the substring path captured
  (a fragment of the claim such as `diffusion` / `concentration`) was not logged.
  It is necessarily a substring relation, since the final fact is keyed by
  `norm(claim)` (`:867-868`) and exact label equality no longer held.
* The 6/8 control failure pattern follows from the same field: the original reads
  `kind='unknown', trusted=False`, so `believed()` rejects it; C1/C2/C4/C5/C6/C7
  depend on graded facts, while C3 (attribution) and C8 (warning/URL/span) read only
  authored item fields.

**One-sentence root cause:** the real stamped edge is deleted on cold reload because
the authored object is a >=5-word "question-phrase" node, and the census then grades
a different, unstamped cold-start edge whose truthy default `edge_kind='unknown'`
becomes the graded fact — not a save-provenance loss, not a buffer route, and not a
wrong metadata key.

## 3. Changed functions (all in the two owned source files)

1. **`RealBackend._safe_label` (new).** Deterministic short token
   `"pdr_" + sha1(norm)[:16]`. It has **no space**, so `_is_question_phrase`
   short-circuits `False` at `constants.py:536` and the node survives the reload
   sanitizer. The authored full label is *not* used as the node label any more.
2. **`RealBackend._ensure_engine_node`.** Mints the node with `_safe_label(norm)`
   while still registering it under the authoritative `norm` key in
   `eng._concept_keywords`. Label resolution, gate `_node_id` and intake writes are
   unchanged; only the stored node label changed.
3. **`RealBackend._find_engine_edge`.** Full-scan fallback now accepts a target
   node labelled either with the full object label or with its `_safe_label`
   token, so write-time stamping and post-cue flips keep resolving the same edge.
4. **`RealBackend._census`.** Resolves each authored fact to the real engine edge by
   **(subject, object) node identity** from `_concept_keywords` (+ a live label map
   built from `graph.nodes.items()`, never the bare int-keyed dict). Exact stored
   label and provenance (`source_url`) are secondary fallbacks. **The substring
   match is removed.** The graded `kind`/`source_url`/`trusted`/`active` are still
   read from that edge's `source_metadata`, so the observation remains engine-sourced.
5. **Module docstring** records the cold-reload seam.
6. **`tests/unit/test_learn5_pdr.py`** adds fake-store tests: sanitize-safe node;
   stamp survival across a simulated cold reload (pickle round-trip of edge state);
   census kind/source correctness for all three poison kinds after reload; the
   no-substring regression; and a one-edge mutation control (change one real edge's
   `edge_kind` → the graded `kind` changes). No engine boot.

`FakeBackend`, `PDRGrader`, `TH`, `instrument_controls()` (all 8), `decide()`,
`gate_saturation()`, `run_durable_sweep()`, the item bank and the sealed hash split
are byte-identical.

## 4. Predicted control outcomes (no threshold moved)

With the graded original recovered as `kind='curated', source_url=<real fixture
url>, trusted=True` and each poison as `co_occurrence` / `web_fact` / `definition`
with a real URL, at `p=5`:

| Control | Depends on `obs["facts"]` | Predicted |
|---|---|---|
| C1 p=0 baseline | yes | FAIL → **ok** |
| C2 naive non-resistant | yes | FAIL → **ok** |
| C3 attribution | no | ok (unchanged) |
| C4 coverage | yes | FAIL → **ok** |
| C5 forged provenance + reject | yes | FAIL → **ok** |
| C6 sleeper dormant + reject | yes | FAIL → **ok** |
| C7 durable | yes | FAIL → **ok** |
| C8 warning/URL/span | no | ok (unchanged) |

All 8 expected `ok`; `engine_observation_missing` should fall to ~0 per query;
`backend=live` with zero `api_error` and no threshold change.

## 5. Falsifier (what would disprove this diagnosis — do NOT tune thresholds)

If the corrected original already reads `kind='curated'` and `source_url=<url>` in
the one-item probe **but the 8 controls still fail**, the remaining drop is in
persistence, not in resolution. Do **not** adjust any gate, threshold, item,
sealing split or control. Instead:

1. Census the **on-disk** store directly, bypassing `RealBackend.query()`:
   `pickle.load` the saved weights pkl and walk `graph.edges[*].source_metadata`,
   and separately query the SQLite mirror (`ravana/src/ravana/storage/db.py` edge
   `source_metadata` BLOB at `:139`, read at `:272`). Compare `edge_kind`,
   `source_url`, `source_type`, `active` before save and after reload.
2. If the pkl has the stamp but the reloaded engine reads `None`, the loss is in
   the load path (pkl truncation → db rehydration, or index rehydration); inspect
   `[Load error]` / `[Load recover]` lines. If both pkl and db lack the stamp, the
   write-time stamp was applied to a different edge object than the one persisted.
3. Keep the mutation controls: plant one wrong real edge kind and require the
   graded kind to move (`test_mutation_of_one_real_edge_kind_changes_graded_kind`),
   and flip one poison edge to trusted and require `contaminated` to move
   (`test_mutation_of_engine_provenance_moves_graded_observation`). If either score
   cannot move, the observation is still an oracle, not the engine — report
   `INSTRUMENT INVALID — no score`.

## 6. Verified / unverified

Verified here (plain Python direct-call smoke check, no pytest, no engine boot): all
10 exercised fake paths pass — the 5 new tests plus the 5 existing materialization /
census / threshold-neutrality tests.

Unverified (parent to run): the real `--live` one-item probe after this change; that
`eng.save()` persists the short-label node/edge and `_concept_keywords` key on the
production store; the full 82-item wired-live control outcome; and the pytest suite.
