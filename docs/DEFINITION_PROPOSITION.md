# Relation from the definition, not from a typed edge

**Status:** prototype, measured. A route that needs no typed edge and no
retraining.

Instrument: `scripts/prototype_definition_plan.py`, persisted engine.

## The problem this answers

The plan route needs a typed graph edge. Typed edges are sparse (~40% of
eval subjects), and online typing is measured as not working
(`docs/TYPING_ACCURACY_IS_CIRCULAR.md`). The obvious next step was "acquire
more typed edges" -- a data project.

But RAVANA already has **324 stored definitions** -- real retrieved text. And
the brain's route for a novel concept is exactly that: it does not need a
stored relation label, it reads the relation off the concept's description.

"Gravity is the force by which..." asserts a categorizing relation. "A species
is the basic unit of classification..." asserts one. The relation is IN the
text.

## Mechanism

```
definition -> copula pattern (is_a | has_property) -> predicate phrase
           -> head noun (POS map, English right-headedness) -> graph concept
           -> SurfacePlan -> symbolic formulator
```

Two details that measurement forced:

1. **The target must resolve to a graph concept.** Copying the phrase verbatim
   gave "gravity is a force by which" and "ocean is a body of salt" -- targets
   the graph has never heard of, which is a claim the graph cannot support.
   Resolving to the subject's graph neighbours fixed it.
2. **The head noun, not the first content word.** "basic unit of
   classification" heads on `unit`; taking the first word gave `basic`, an
   adjective, which is a claim about nothing. English noun phrases are
   right-headed, so the last POS-tagged noun wins.

## Result on the corpus that has definitions

```
definitions: 324
with a recovered proposition: 98/324
  on_topic    = 0.9439
  unsupported = 0.0000
```

```
ancient    is a period       animal     is an organism
aristotle  is a philosopher  blue       is a colour
book       is a work         area       is a measure
```

**On-topic 0.944 with zero unsupported assertions** -- the target is a graph
neighbour of the subject by construction, so the claim is graph-supported.

## Honest limits

- **98/324 = 30% recall.** 226 definitions yield no proposition. Most are
  definitions whose predicate is not a simple copula ("Music is sound
  organized in time...", "Brain ..."). This is a precision route, not a
  replacement.
- **POS errors survive**: "ocean is a salt" (the POS map missed `body`),
  "brain is a serves". The head-noun rule is only as good as `_concept_pos`.
- **Grammar is not the realizer's**: the prototype's `bind_to_sentence` call
  ignores the plan's construction and picks its own verb -- "gravity
  contributes to force", "dog builds on descendant". The PLAN is right; the
  formulator overrides it. That is the next thing to fix.
- **on_topic 0.944 is measured against the graph**, not against truth. A
  wrong head noun still scores well if it is a graph neighbour.

## Why this is brain-faithful

The brain grounds a novel concept in its DESCRIPTION and in its relations to
known concepts -- it does not wait for a curated label. This is the same move
as `docs/BRAIN_RELATION_TYPING_SPEC.md`'s "learned tier", but the source of
supervision is the retrieved text rather than a vector prototype.

## Relationship to the deprecated semantic proxy

This route does NOT use `_infer_relation_type` or the vector prototypes at
all. It reads the relation from text. That is the redirect the research
recommended: away from a semantic-similarity proxy and toward structural
evidence.

## Reproduce

```
uv run python scripts/prototype_definition_plan.py gravity ocean species time
```

## Related

- `docs/TYPING_ACCURACY_IS_CIRCULAR.md` -- why the vector proxy fails
- `docs/PLAN_NARRATIVE_POSITIVE.md` -- the plan route this extends
- `docs/BRAIN_RELATION_TYPING_SPEC.md` -- the research spec
