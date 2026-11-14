# Web → graph → plan: verified end to end

**Status:** measured. `scripts/verify_web_learning_chain.py`.

## Why

PR #378 fixed `WebToGraph.RELATION_TO_EDGE["causes"]` → `"causal"`. A mapping
change is only worth something if the whole chain runs, so this script drives
the real classes and asserts each hop.

## The chain

```
web text ──OpenIE──> WebToGraph.learn_text ──> GraphEngine edge
                                                        │
                                          relation_type == "causal"
                                                        │
                                    message_from_best_edge + message_to_plan
                                                        │
                                              committed SurfacePlan
                                                        │
                                          "smoking leads to lung cancer"
```

## Measured

```
facts learned: 2
edge count: 2
relation types: {'causal': 1, 'is_a': 1}
  causal         in weight table: True
  is_a           in weight table: False

edge rel=causal   weight=0.50
  plan: ['smoking', 'leads', 'to', 'lung cancer']
  committed: True
  vacuous? False
```

Before the fix the same input produced `relation_type='causes'`, which is
absent from `ConceptEdge.RELATION_DEFAULT_WEIGHTS`, from
`graph_typing.TYPED_RELATION_TYPES`, and from `VACUOUS_RELATIONS` — so the edge
was invisible to the plan route and to ranking.

## The `is_a` gap, still open

`is_a` is also absent from the weight table. It does not break the plan route
because `learn/linearization.py` normalises `is_a` → `isa`, and `isa` IS in
`TYPED_RELATION_TYPES`. But the graph's own weight table gives it no default
(weight 0.50 here came from the writer, not the table). Recorded, not fixed.

## Reproduce

```
uv run python scripts/verify_web_learning_chain.py
```

## Related

- `docs/BRAIN_EXPLANATION_AND_REASONING.md` §5.1 — the defect report
- `tests/unit/test_web_relation_types.py` — pins the mapping invariant
