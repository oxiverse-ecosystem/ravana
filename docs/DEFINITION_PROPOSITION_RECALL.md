# Definition-derived propositions: recall 30% -> 80%

**Status:** measured. `ravana/src/ravana/learn/definition_proposition.py`.

Instrument: 324 stored definitions, persisted engine.

## The gap

The definition route recovered 98/324 propositions (30%). The 226 "failures"
were assumed to be non-copular definitions. They were not:

```
acid       a molecule or ion capable of either donating a proton...
adult      an animal that has reached full growth
australia  a country comprising the mainland of the Australian continent...
author     the creator of an original work that has been published...
biology    the scientific study of life and living organisms
```

These are copular -- the subject is simply omitted and the definition opens on
the predicate nominal. The prototype's regex demanded `is a`, so it missed
every one.

## Fix 1: article-first pattern

Added `^\s*(?:a|an|the)\s+...` alongside the copula patterns. Recall
**30% -> 81%**.

## Fix 2: word-bounded capture, not character-bounded

The first attempt captured `{2,40}` CHARACTERS and truncated mid-word:
"a molecule or ion capable of either donatin" -> target `donatin`. The
patterns now capture a bounded number of WORDS.

## Fix 3: head noun FIRST for article-first, LAST for copula-first

English noun phrases are right-headed ("basic unit of classification" heads on
`unit`), so the copula patterns take the last POS-tagged noun. But
article-first definitions already mark the phrase start, so the head is FIRST
there -- "a molecule or ion capable of..." heads on `molecule`, not `either`.

MEASURED before / after:

```
acid        either        -> molecule
australia   mainland      -> country
author      work          -> creator
biology     organ         -> study
attention   awareness     -> concentration
anger       strong        -> intense (still wrong -- POS map)
book        length        -> length   (still weak)
```

## Result

```
recovered   260/324 = 80%
committed   222/324 = 69%
on_topic    0.6734
unsupported 0.0045
```

Against the previous 98/324 (30%) at on_topic 0.944 / unsupported 0.000. The
on-topic figure is lower because coverage now includes harder subjects; the
unsupported rate stays effectively zero, which is the property that matters --
the target is a graph concept by construction.

## Remaining rejects (38)

Two kinds:

1. **Unresolved target** -- the head noun is not a graph concept (`aristotle ->
   polymath`, `dog -> wolf`, `brain -> center`, `evolution -> change`). The
   graph does not contain the word the definition actually uses. This is a
   genuine graph-coverage gap, not a parser bug.
2. **Not committed** -- `authority -> authority` (the subject is its own
   target; correctly rejected as a tautology).

## Honest limits

- `anger -> intense`, `book -> length` are wrong: the POS map tags the
  adjective as a noun. The head-noun rule is only as good as `_concept_pos`.
- `carbon -> c`, `career -> s` -- single-letter captures from chemical
  notation; the word-bounded pattern still picks up stray tokens.
- on_topic 0.673 is measured against the graph, not truth.
- This route needs a stored definition. It does not help subjects with no
  definition and no typed edge -- for those the honest hedge is correct.

## Reproduce

```
uv run python -c "..."   # see the measurement in the commit message
```

## Related

- `docs/DEFINITION_PROPOSITION.md` -- the original 30% route
- `ravana/src/ravana/learn/definition_proposition.py` -- the module
- `ravana/src/ravana/learn/linearization.py` -- the plan the proposition feeds
