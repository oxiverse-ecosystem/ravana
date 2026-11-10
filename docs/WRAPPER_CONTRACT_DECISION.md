# Wrapper landing: response-contract decision (2026-11-11)

Companion to `GENERATION_STACK_DECISION.md` (#306), which decided the engine is
canonical. That decision was recorded but the wrapper was not landed. This
document decides the *contract* question the wrapper exposed: when the two
stacks emit different responses, which contract is correct?

## Decision: the engine's contract is correct

`ChatInterface.process_turn` must return exactly what
`CognitiveChatEngine.process_turn` returns, byte for byte. The integration
assertions are restated against that contract, not weakened.

## Evidence

### 1. The old assertions encode the duplicate stack, not the product

Measured under the pytest environment (`RAVANA_OFFLINE=1` +
`RAVANA_TEST_MODE=1`, both set by `tests/conftest.py`), same seed, same dim:

| query | old `ChatInterface` | `CognitiveChatEngine` |
|---|---|---|
| "Tell me about trust" | `unknown_subject` — "Perhaps trust links to trust." | `reflective_uncertainty` — "the closest i get to trust is people and maybe life … what's your take on it?" |
| "What is respect" | `unknown_subject` — "Perhaps respect results in respect." | `honest_uncertainty` — "i don't really know that yet — i'd have to look it up." |
| "How does trust work?" | `graph_fallback` — "It appears that work runs counter to hard. …" | `honest_uncertainty` — "i don't really know that yet — i'd have to look it up." |
| "Tell me more" | `fast_ventral` — "It differs from hard. Also, it challenges things. …" | `chitchat` — "i'm ravana, a cognitive architecture that's been learning. …" |
| "This is really bad and sad" | `fast_ventral` — "Sad: it challenges enjoy. It directly results in class." | `assertion` — "ok, noted: sad. anything else on your mind?" |

The old stack's output is **degenerate**: "Perhaps trust links to trust",
"respect results in respect", "it challenges things". These are the
`_graph_fallback_response` templates firing because the old stack has no
modern response paths. The engine's output is a coherent conversational turn.

`_is_valid_response` (capital + terminal punctuation) was written for the old
stack's `.capitalize()`-first templates. It is a *shape* check, not a contract:
the engine's lowercase replies are intended (see #304) and end in `.`/`?`.

### 2. The engine is the documented, tested, running stack

From `GENERATION_STACK_DECISION.md` (#306): 294 test files import the engine
vs 4 for the interface; `scripts/ravana_chat.py:44` instantiates the engine;
`docs/ARCHITECTURE.md:13,179` calls it "the orchestrator". The interface is
not mentioned. The docs already prescribe this direction
(`LANGUAGE_PRODUCTION_PIVOT.md` G13, `:599`).

### 3. Case is a deliberate register signal, not a defect

`5d6f954` (#304) removed `return response.lower()` from two engine early-return
paths precisely because flattening case destroys the register signal Phase 2
style conditioning reads. The engine's lowercase is the *intended* contract.
Restating the assertion as "first character is a letter" keeps the real
invariant (a complete sentence) without re-imposing the old stack's casing.

### 4. The strategy names are the engine's taxonomy

`fast_ventral`, `unknown_subject`, `dorsal_reasoned`, `associative`,
`graph_fallback`, `syntactic_pipeline` are the old stack's internal labels.
The engine's 74 `_last_strategy` values (`honest_uncertainty`,
`reflective_uncertainty`, `web_direct_answer`, `chitchat`, `assertion`, …)
are what 294 test files assert against. The assertion is restated as a
"non-empty string" check plus a comment, because pinning one exact value would
make the test brittle against the engine's routing — the old pinned set was
brittle against the old stack's routing.

### 5. The four deferred attributes are not the engine's contract

`decoder_engine`, `bootstrap_manager`, `relation_predictor`, `propagation`
are `None` on the engine by design:

* `decoder_engine` — the engine owns a `NeuralDecoder` directly
  (`engine.py:2032`, built in `response_gen.py:_build_decoder_vocab`). The
  `DecoderEngine` wrapper class is part of the old stack.
* `bootstrap_manager` — zero references anywhere in the engine stack.
* `relation_predictor` / `propagation` — `ravana.nn.rlm` is not imported by
  the engine at all; the engine uses `ConceptGraph` + `Plasticity`.

The test's attribute list is a **monolith inventory**, not a contract. It is
restated against the engine's actual surface. Adding four dead attributes
would be the exact anti-pattern the goal doc forbids.

## What was restated

* `_is_valid_response` — first character is a letter (not necessarily upper).
* `test_basic_response` strategy set — non-empty string, with the engine's
  taxonomy documented.
* `test_all_components_initialized` — the engine's real component surface
  (`graph`, `pfc_workspace`, `emotion`, `identity`, `meaning`, `dual_process`,
  `gw`, `meta_cog`, `sleep_engine`, `belief_store`, `basal_ganglia`,
  `cerebellar_ngram`, `syntactic_assembly`, `surface_realizer`, `plasticity`,
  `neural_decoder`).

## One engine defect fixed, not a test loosened

`test_response_references_topic` was failing for a *content* reason, not a
contract reason: the engine's test-mode honest-uncertainty gate
(`engine.py` `if _offline and _test_mode:`) returned a bare
`"i don't really know that yet — i'd have to look it up."` that names no
topic. That is a worse failure mode than the confabulation the gate exists to
prevent — an honest abstention should still be about what was asked.

The engine already has the right path: `_human_like_uncertainty` grounds the
abstention in the subject ("hmm, i'm not totally sure about X yet…"). The gate
was bypassing it. Fixed by routing the gate through
`_ground_query` + `_human_like_uncertainty`, with the bare sentence kept only
as a fail-open fallback. The assertion was left exactly as written.

## Cost of the alternative

Reconciling the engine to the old contract would mean re-implementing
`DecoderEngine`, `BootstrapManager`, `RelationPredictor` and `PropagationEngine`
on the engine, and restoring the `_graph_fallback_response` templates — i.e.
recreating the G13 divergence the docs already name as a defect, and
re-breaking the register signal #304 fixed. Rejected.
