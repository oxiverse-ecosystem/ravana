# Prototype: linearization of a role-bound message

**Status:** prototype, first cut. Not wired into production. The research spec
this tests is `docs/BRAIN_LINEARIZATION_SPEC.md`.

## The hypothesis

RAVANA's decoder can be made to emit the right content words (on-topic 1.00,
unsupported 0) but not in grammatical order
(`docs/ANCHOR_BOOST_DESTROYS_SYNTAX.md`). Selection works; linearization does
not. So condition the decoder on an ORDERED SURFACE PLAN derived from the
graph message, rather than on a logit bias over words.

Chain: `macroplanning.build_message` (spreading activation → Luce-ratio
selection → role assignment) → `linearization.message_to_plan` (relation
semantics → construction → slot filling) → the decoder elaborates from the
plan.

## What the module does

`ravana/src/ravana/learn/linearization.py` — pure function, no decoder, no
authored reply.

A **construction** is a form-meaning pairing (Goldberg 1995; Tomasello 2003):
a surface frame with slots, and the roles that fill them. Five constructions
cover the graph's typed relation vocabulary:

| relation_type | construction | plan |
|---|---|---|
| `is_a`, `part_of`, `member_of` | categorize | `<theme> is a <theme>` |
| `causes`, `leads_to`, `affects` | cause | `<agent> leads to <ground>` |
| `has_property`, `used_for` | attribute | `<theme> has <relation>` |
| `located_in`, `part`, `time` | locate | `<theme> is located in <ground>` |
| anything else | relate | `<theme> is related to <ground>` |

The subject always fills slot 0 (it is the utterance's topic). A construction
with no binding for a slot is dropped rather than rendered with a hole, so the
caller falls back.

## Unit-level result

```
species  is_a        -> categorize   species is a organism
sky      causes      -> cause        sky leads to fire
sky      located_in  -> locate       sky is located in orbit
sky      has_property-> attribute    sky has blue
gravity  related_to  -> relate       gravity is related to force
species  part_of     -> categorize   species is a taxonomy

empty message -> None
no bindings   -> None
```

Six of six relations linearize; both empty cases fail closed.

## On the real graph -- the blocking finding

```
gravity    gravity is related to two
ocean      ocean is located in sea
species    species is related to birds
time       time is related to making
sky        sky is related to blue
music      music is a sound
```

**The plans are grammatical and semantically wrong.** `gravity is related to
two` and `time is related to making` are well-formed clauses about nothing.

The cause is upstream: the graph's highest-activation edges for these subjects
are noise. `macroplanning.build_message` selects by spreading activation over
edge weight and confidence, and the top-scoring edges for `gravity` and `time`
point at `two` and `making`. The linearization is doing its job -- it
faithfully realizes a bad message.

This is the same finding as `docs/DECODER_GROUNDING_VETO.md` from the other
direction: the bottleneck is the MESSAGE, not the formulator. A perfect
formulator over a bad message produces fluent, grammatical, false sentences --
which is worse than word salad, because it is harder to detect.

## What this means for the next cut

1. **The plan must be scored before it is used.** A plan whose fillers are
   function-ish or whose relation is the vacuous `related_to` fallback should
   be rejected, not realized. `response_gen.py`'s `_DISTRIBUTIONAL_RELATIONS`
   denylist already encodes this distinction.
2. **Prefer typed edges.** `music is a sound` came from a typed `is_a` edge
   and is correct. `gravity is related to two` came from a `related_to`
   fallback. The construction choice should be gated on the edge being TYPED,
   not on its activation rank.
3. **The decoder test is still pending.** Whether elaborating FROM a plan
   beats a word-list boost can only be measured once the plan is correct.

## Reproduce

```
uv run python -c "..."
```
See `scripts/prototype_linearization.py` for the three-arm comparison
(baseline / word-boost / plan).

## Related

- `docs/BRAIN_LINEARIZATION_SPEC.md` -- the research spec
- `docs/ANCHOR_BOOST_DESTROYS_SYNTAX.md` -- the word-boost failure this
  responds to
- `ravana/src/ravana/learn/macroplanning.py` -- the message layer
- `ravana/src/ravana/learn/committed_proposition.py` -- the commitment step
