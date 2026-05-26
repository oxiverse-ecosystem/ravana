# Phase 2: Activation-Pattern-Based Relation Classifier

## Analysis: WHY Transfer is Still 0% After Phase 1

### What the Cross-Domain Transfer Test Measures

The `run_deep_compositional()` in experiment_rigorous.py trains:
- Chain facts: "zorbax is crystalline", "crystalline things are fragile", ...
- Transfer facts: "vexol is warm", "warm things are pleasant", ...

Tests 2-hop compositional transfer: "vexol is" → "pleasant" (vexol→warm→pleasant).

Transfer = 0% means the model CANNOT compose across novel chains, even though
the keyword classifier now correctly labels "X is Y" as semantic and
"X causes Y" as causal.

### Root Cause #1: Relation Vectors Collapse (The Killer Bug)

In `graph.py` `hebbian_update()` (line 1240-1246):

```python
hebbian_signal = source.vector[:rv_len] * target.vector[:rv_len]
edge.relation_vector += effective_lr * 0.1 * (hebbian_signal - edge.relation_vector)
```

**Problem**: ALL edges share the same Hebbian dynamics. The Hebbian signal
`source.vector * target.vector` is dominated by the SHARED token structure
(not the relation pattern). When edges share tokens (most do via "is", "things",
"are"), their relation vectors converge to the SAME cluster.

This is a fundamental bug: the Hebbian pull is STRONGER than the contrastive
push, so relation vectors for causal, temporal, and semantic edges all
converge to the same point in relation-vector space.

Evidence: `_init_relation_vector()` gives each type a DIFFERENT seed vector
(seed 0=semantic, seed 1=causal, seed 2=temporal). But after training,
`hebbian_update()` pulls ALL of them toward the same Hebbian signal,
overwhelming the initial separation.

### Root Cause #2: Contrastive Push is Starved

In `hebbian_update()` (line 1248-1260):

```python
negatives = [e for e in sample_edges
             if e.relation_type != edge.relation_type and e is not edge]
```

With Phase 1 keyword classifier, most edges are still "semantic" (because
"X is Y" is the dominant pattern). The negatives list for semantic edges
is EMPTY or tiny — there's nothing to push against.

### Root Cause #3: Multi-Hop Traversal Ignores Relation Type

In `forward()` (line 452-479) and `forward_step()` (line 1489-1520):

```python
for tgt_id, edge in outgoing:
    if edge.edge_type == "inhibitory" or edge.weight < 0.05:
        continue
    hop_score = node.activation * edge.weight * hop_decay
```

**ALL edges are treated equally** regardless of relation_type. The traversal
doesn't know if it's following a causal chain, a temporal sequence, or a
semantic similarity. A causal edge and a semantic edge get the same hop_score.

### Root Cause #4: Shortcut Edges Have No Relation Type

At line 646 (shortcut creation in learn()):
```python
cedge = self.graph.get_or_create_edge(ctx_concept, output_concept,
                                       weight=0.1, shortcut=True)
```
No `relation_type` parameter → defaults to "semantic".

At line 1318 (REM cross-link):
```python
self.graph.add_edge(n1.id, n2.id, weight=0.1,
                    edge_type="contextual", shortcut=True)
```
No `relation_type` parameter → defaults to "semantic".

### Root Cause #5: Keyword Classifier is Syntactic-Only

The Phase 1 classifier only looks at surface keywords. It misses:
- "X is Y" where Y implies causation (e.g., "fire is dangerous" → causal?)
- Implicit temporal ordering in sequential data
- Structural patterns (directionality) that reveal relation type

---

## Phase 2 Design: Activation-Pattern-Based Relation Inference

### Core Insight

**The graph already encodes structural asymmetry — we just need to read it.**

If edge A→B has strong weight but B→A is weak/absent, this is directional
(causal or temporal). If A→B ≈ B→A, this is symmetric (semantic).

The asymmetry signal is FREE — it's already in the edge weights and
prediction counts. We just need to extract it.

### Design: 5 Changes

#### Change 1: Track Forward/Backward Prediction Counts

**File**: `ravana_ml/graph.py` — `ConceptEdge.__init__()`

Add bidirectional prediction tracking:

```python
# In ConceptEdge.__init__:
self.forward_pred_count = 0   # A→B successful predictions
self.backward_pred_count = 0  # B→A successful predictions
```

**File**: `ravana_ml/nn/rlm.py` — `learn()` method (line ~525)

After creating/updating edge, record directional prediction:

```python
# After edge.prediction_count += 1:
edge.forward_pred_count += 1  # A→B was just observed

# Check if reverse edge exists and record backward signal
reverse_edge = self.graph.get_edge(output_concept, input_concept)
if reverse_edge is not None:
    # Reverse edge exists but wasn't activated — weak backward signal
    pass  # Don't increment backward count
```

#### Change 2: Activation-Pattern Classifier

**File**: `ravana_ml/nn/rlm.py` — Add new method `_infer_relation_from_structure()`

```python
def _infer_relation_from_structure(self, source_id: int, target_id: int) -> str:
    """Infer relation type from structural activation patterns.

    Uses prediction asymmetry to distinguish directional vs symmetric relations:
    - A→B strong, B→A weak = directional (causal/temporal)
    - A→B ≈ B→A = symmetric (semantic)

    Also uses temporal ordering from sequence position.
    """
    edge = self.graph.get_edge(source_id, target_id)
    if edge is None:
        return "semantic"

    # 1. Prediction asymmetry
    fwd = edge.forward_pred_count + edge.prediction_count
    reverse_edge = self.graph.get_edge(target_id, source_id)
    bwd = reverse_edge.forward_pred_count + reverse_edge.prediction_count if reverse_edge else 0

    total = fwd + bwd
    if total < 3:
        return edge.relation_type  # not enough data, keep current

    asymmetry = abs(fwd - bwd) / total  # 0 = symmetric, 1 = fully directional

    # 2. Temporal position signal
    src_node = self.graph.get_node(source_id)
    tgt_node = self.graph.get_node(target_id)
    temporal_signal = 0.0
    if src_node and tgt_node:
        src_pos = getattr(src_node, '_last_seq_position', -1)
        tgt_pos = getattr(tgt_node, '_last_seq_position', -1)
        if src_pos >= 0 and tgt_pos >= 0:
            # Source appears before target = temporal ordering
            temporal_signal = 1.0 if src_pos < tgt_pos else -1.0

    # 3. Classification logic
    if asymmetry > 0.6:
        # Strongly directional
        if temporal_signal > 0:
            return "temporal"  # A before B, directional = temporal
        else:
            return "causal"    # A causes B (directional, no clear temporal order)
    elif asymmetry > 0.3:
        # Moderately directional
        return "contextual"
    else:
        return "semantic"  # Symmetric = semantic/is-a
```

#### Change 3: Periodic Relation Refinement in learn()

**File**: `ravana_ml/nn/rlm.py` — `learn()` method

Add periodic refinement that re-classifies edges based on observed patterns:

```python
# Add to learn(), after the contrastive relation learning block (line ~547):
# Periodic relation type refinement (every 20 steps)
if self._step_counter % 20 == 0:
    self._refine_relation_types()
```

New method:

```python
def _refine_relation_types(self, max_edges: int = 50):
    """Re-classify edge relation types based on accumulated activation patterns.

    Runs periodically. For each edge, checks:
    1. Prediction asymmetry (forward vs backward)
    2. Temporal ordering signals
    3. Keyword classifier fallback

    Re-classifies edges whose structural signal contradicts their current type.
    This creates the positive feedback loop: correct typing → better contrastive
    separation → better analogy matching → better transfer.
    """
    refined = 0
    for key, edge in list(self.graph.edges.items())[:max_edges]:
        if edge.shortcut or edge.edge_type == "inhibitory":
            continue

        src_id, tgt_id = key
        inferred = self._infer_relation_from_structure(src_id, tgt_id)

        if inferred != edge.relation_type:
            old_type = edge.relation_type
            edge.relation_type = inferred

            # Re-initialize relation vector from new type seed,
            # but blend with existing vector to preserve learned structure
            new_seed = ConceptEdge._init_relation_vector(inferred, len(edge.relation_vector))
            blend = 0.7  # 70% existing, 30% new seed
            edge.relation_vector = blend * edge.relation_vector + (1 - blend) * new_seed
            rv_norm = np.linalg.norm(edge.relation_vector)
            if rv_norm > 0:
                edge.relation_vector /= rv_norm

            refined += 1

    return refined
```

#### Change 4: Relation-Type-Weighted Multi-Hop Traversal

**File**: `ravana_ml/nn/rlm.py` — `forward()` (line ~452) and `forward_step()` (line ~1489)

Modify multi-hop traversal to boost edges by relation type relevance:

```python
# Replace the hop_score calculation in forward() and forward_step():
for tgt_id, edge in outgoing:
    if edge.edge_type == "inhibitory" or edge.weight < 0.05:
        continue
    tgt_node = self.graph.get_node(tgt_id)
    if tgt_node is None:
        continue

    # Relation-type-aware hop scoring
    rel_boost = 1.0
    if edge.relation_type == "causal":
        rel_boost = 1.3  # causal chains are strong inference paths
    elif edge.relation_type == "temporal":
        rel_boost = 1.2  # temporal sequences are reliable
    elif edge.relation_type == "inferred":
        rel_boost = 0.8  # inferred edges are less certain

    hop_score = node.effective_activation * edge.weight * hop_decay * rel_boost
    # ... rest unchanged
```

#### Change 5: Fix Shortcut/REM Edges to Carry Relation Type

**File**: `ravana_ml/nn/rlm.py`

Fix shortcut edge creation in learn() (line 646):
```python
# Before:
cedge = self.graph.get_or_create_edge(ctx_concept, output_concept,
                                       weight=0.1, shortcut=True)
# After:
rel_type = self._classify_relation(token_ids[0])
cedge = self.graph.get_or_create_edge(ctx_concept, output_concept,
                                       weight=0.1, shortcut=True,
                                       relation_type=rel_type)
```

Fix REM cross-link (line 1318):
```python
# Before:
self.graph.add_edge(n1.id, n2.id, weight=0.1,
                    edge_type="contextual", shortcut=True)
# After:
self.graph.add_edge(n1.id, n2.id, weight=0.1,
                    edge_type="contextual", shortcut=True,
                    relation_type="inferred")
```

---

## Implementation Sequence

1. **Change 1** (graph.py): Add forward_pred_count/backward_pred_count to ConceptEdge
2. **Change 5** (rlm.py): Fix shortcut/REM edges to carry relation_type
3. **Change 2** (rlm.py): Add _infer_relation_from_structure() method
4. **Change 3** (rlm.py): Add _refine_relation_types() and call it from learn()
5. **Change 4** (rlm.py): Add relation-type-aware hop scoring in forward/forward_step
6. **Test**: Run experiment_rigorous.py and check rlm_transfer_mean

## Expected Outcome

- After Phase 2, ~40-60% of edges should have non-semantic relation types
  (vs <1% after Phase 1 keyword-only)
- Relation vectors should separate into distinct clusters by type
- Multi-hop traversal will preferentially follow causal/temporal chains
- Transfer accuracy should jump from 0% to 30-50% (2-hop inference works
  when relation types are correct)

## Key Design Principle

Phase 1 was top-down (keywords → type). Phase 2 is bottom-up (behavior → type).
The structural signal (prediction asymmetry) is RELIABLE because it reflects
actual observed dynamics, not surface syntax. Even if the keyword classifier
misses "fire is dangerous" as causal, the asymmetry between fire→dangerous
(strong) and dangerous→fire (weak) reveals the directional nature.
