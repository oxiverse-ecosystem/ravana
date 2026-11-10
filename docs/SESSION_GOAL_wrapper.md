# Session goal: land the ChatInterface wrapper

## Decide first, then build

The blocker is a **contract decision**, not a bug. Read this before touching code.

`ChatInterface` (ravana/src/ravana/chat/interface.py) is a parallel 2,163-line
generation stack. `CognitiveChatEngine` is canonical (294 test files vs 4;
`scripts/ravana_chat.py:44`; `docs/ARCHITECTURE.md:13,179`). The plan is to
reduce the interface to a thin CLI wrapper over the engine.

A wrapper has been written and is structurally correct, but **6 integration
assertions fail** because the two stacks emit different responses:

| | old `ChatInterface` | `CognitiveChatEngine` |
|---|---|---|
| casing | capitalised | lowercase |
| strategies | `fast_ventral`, `unknown_subject` | `honest_uncertainty`, `reflective_uncertainty`, `web_direct_answer` |
| style | topic-referencing | honest-uncertainty |

**Proven not to be a width bug**: a bare engine produces byte-identical output
at dim=32 and dim=64. **Proven not pre-existing**: the original interface
passes 14/14 in the same environment.

### The decision

Decide which contract is correct, and record the reasoning in
`docs/GENERATION_STACK_DECISION.md`. Then:

- **If the engine's contract is correct** — update the 6 assertions with a
  comment explaining why the engine's behaviour is intended. Do not weaken
  the assertions; restate them against the new contract.
- **If the old contract is correct** — reconcile the engine's response path
  to it, and say what that costs.

Either way, justify it from evidence (which path is more brain-faithful, which
is documented, which the tests were written to protect). Do not flip a test to
make it pass without saying why the new behaviour is right.

**Do not land the wrapper before this decision is written down.**

## Already done — do not redo

- #307 collapsed the duplicate `_generate_with_decoder_and_syntax`
- #309 fixed four hard-coded GloVe-64 assumptions (add_node returning None,
  the cache's 64x100 projection, relation prototypes, decoder dual-code width)
- #310 fixed the intent router's dim-specific fitted centroids

`dim=32` and `dim=64` both construct and run.

## State on disk

`$TMPDIR/wrapper_interface.py` holds an EARLIER wrapper (298 lines). It is
**stale** — it lacks the `_WebLearnerShim.get_web_to_graph` /
`curiosity_e_step` / `knowledge_gap` delegation, the `_GraphEngineShim`
`_all_labels` / `dim` / `_promote_min_sources`, and the restored `main()`.
Use it only as a starting sketch; re-derive the shims from what the tests and
CLI actually call.

## Acceptance

`goal_cmd` must exit 0:

```
uv run pytest tests/integration/test_ravana_chat_pipeline.py tests/integration/test_unified_semantic_loop.py -q
```

Target: 17 passed, 0 failed — the same count the original interface achieves.

## Deferred (decide or record, do not silently drop)

- `decoder_engine` / `bootstrap_manager` / `relation_predictor` /
  `propagation` — the engine has none of these by design. Either they are
  public contract (build them) or the test is wrong (change it). Adding four
  dead attributes to satisfy an assertion is not acceptable.
- 3 latent `vec[:64]` hard-codes in `response_gen.py` (`_vsa_event_narrative`,
  `_build_conditioned_bos`, `_align_to_decoder`) — same class as #309,
  currently harmless at dim=64.
- `DualCodeSpace` always projects at 64-D regardless of engine dim.
- `data/intent_router.json` is a committed dim-specific artefact; #310 made
  the router tolerant, but the artefact itself is still fitted at 64.
