# Solving the "Cyc Trap": ConceptGraph Knowledge Representation at Scale

## The Cyc Lesson

Douglas Lenat's Cyc project (1984–present) is the most famous cautionary tale in knowledge representation. The core problem: compressing messy, ambiguous human experience into typed (subject, relation, object) triples works well for clean, structured facts, but real-world reasoning relies on fuzzy context, sensory ambiguity, idioms, and multi-layered metaphors.

Cyc's failure mode was twofold:
1. **Rigidity**: The ontology became too rigid to handle open-world nuances. Every new domain required manual ontology engineering.
2. **Noise**: Automated graph generation introduced relational noise that corrupted multi-hop spreading activation — the "semantic fog" problem where too many weak edges dilute meaningful inference.

The core insight: **knowledge is context-dependent**. The word "bank" activates "river" in a nature context and "money" in a finance context. "Spill the beans" literally means spilling legumes but idiomatically means revealing secrets. "Time is money" maps time-related concepts to money-related concepts (save, waste, spend) only in a metaphor context.

## How RAVANA Avoids the Cyc Trap

RAVANA's ConceptGraph avoids the Cyc trap through three mechanisms:

### 1. Context-Dependent Edge Gating

The primary mechanism for representing fuzzy/ambiguous knowledge. Edges can be tagged with context labels, and only edges whose context labels match the current cognitive context are active during spreading activation and inference.

```python
# The same "bank" node can activate different neighbors in different contexts
graph.add_context_edge(bank_id, river_id, context_labels=["nature"], weight=0.8)
graph.add_context_edge(bank_id, money_id, context_labels=["finance"], weight=0.9)

graph.set_context(["nature"])  # Only bank->river is active
graph.set_context(["finance"]) # Only bank->money is active
```

This is **not hardcoding** because:
- The context labels are **data** stored on edges, not if/elif branches in code
- RAVANA can **learn new context labels at runtime** from conversation or web learning
- Removing one context label degrades gracefully (the edge just becomes context-independent)
- The same graph structure represents multiple meanings without duplicating the graph

### 2. Confidence-Weighted Edges with Bayesian Posteriors

Every edge carries a confidence score and a Beta(alpha, beta) posterior over its weight. This provides:
- **Uncertainty tracking**: edges with high uncertainty transmit less signal during spreading activation
- **Online learning**: confidence updates incrementally as edges are used in successful/failed predictions
- **Noise resistance**: weak or uncertain edges are naturally suppressed without manual pruning

### 3. Multi-Hop Inference with Confidence Decay

The `infer_chain` method uses sparse multi-hop inference with:
- **Frontier budget**: max nodes per hop (winner-take-most) prevents semantic fog
- **Confidence threshold**: stops propagating low-confidence chains
- **Coherence gate**: each hop must maintain semantic alignment with the start node
- **Entropy penalty**: penalizes paths that branch too broadly

This prevents the Cyc problem of "automated graph generation introduces relational noise that corrupts multi-hop spreading activation."

## Three Types of Ambiguity Handled

### 1. Idiom
"Spill the beans" literally means spilling legumes, but idiomatically means revealing secrets. The graph represents both meanings:
- A context-independent edge `spill -> beans` (weak, always available)
- A context-dependent edge `spill -> secrets` (strong, only in "idiom" context)

### 2. Metaphor
"Time is money" maps time-related concepts to money-related concepts. The graph represents the metaphor:
- A context-dependent edge `time -> money` (only in "metaphor" context)
- Context-dependent edges `time -> save`, `time -> waste` (only in "metaphor" context)

### 3. Sensory Ambiguity (Polysemy)
"Bank" can mean a financial institution or a river bank. The graph represents both:
- Context-dependent edges `bank -> river`, `bank -> water` (only in "nature" context)
- Context-dependent edges `bank -> money`, `bank -> interest` (only in "finance" context)

## Integration with Existing Architecture

The context-gating mechanism integrates with the existing spreading activation and multi-hop recall paths:

- `spread_activation()`: context-inactive edges are skipped during propagation
- `infer_chain()`: context-inactive edges are not traversed during multi-hop inference
- `PropagationEngine.propagate()`: automatically respects context gating via the graph

No new components were added — the mechanism extends the existing ConceptGraph + GRACE governor architecture.

## Acceptance Criteria Status

- [x] A concrete mechanism for representing fuzzy/ambiguous context in the ConceptGraph (context-dependent edge gating)
- [x] Unit tests demonstrating the mechanism handles at least 3 types of ambiguity (idiom, metaphor, sensory ambiguity) — 14 tests in `tests/unit/test_context_gating.py`
- [x] A research document summarizing the Cyc lesson and how RAVANA's approach avoids it (this document)
- [x] Integration with the existing spreading activation / multi-hop recall path

## Files Changed

- `ravana_ml/src/ravana_ml/graph.py`: Added `context_labels` to `ConceptEdge`, `set_context()`, `get_context()`, `is_edge_active()`, `add_context_edge()`, `get_context_edges()`, `clear_context()`, and integrated context gating into `spread_activation()` and `infer_chain()`
- `tests/unit/test_context_gating.py`: 14 unit tests covering all 3 ambiguity types, context gating, spreading activation, inference, and pickle compatibility
