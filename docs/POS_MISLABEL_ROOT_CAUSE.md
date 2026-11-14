# Root cause: `"an only"` — a POS map that defaults to noun

**Status:** diagnosed, not yet fixed. Instrument:
`scripts/measure_pos_mislabel.py`.

## The failure

With the plan route enabled (#369), live output included:

```
It appears that time is different from an only.
It appears that climate sets the stage for an environment.
It appears that gravity brings about an earth.
```

`"an only"` is grammatically well-formed after the a/an fix (#370) and still
nonsense: `only` is an adverb, and the frame is trying to put it in a noun slot.

## Root cause: `_build_concept_pos` defaults everything to `noun`

`ravana/src/ravana/chat/chain_walker.py:174-205` tags every graph node label
with four tests, in order:

1. `ll in KNOWN_VERBS` (46 entries), or a 3rd-person `-s` form of one
2. `ll in KNOWN_ADJS` (26 entries)
3. verb suffixes `ing ed ize ify ate en ish`
4. adjective suffixes `able ible ful less ous al ic ive`
5. `ll in FUNCTION_WORDS` -> `FUNCTION_POS.get(ll, 'adv')`
6. **else -> `'noun'`**

The word lists are tiny (26 adjectives, 46 verbs) and the suffix rules are
English-specific and incomplete. Everything unmatched falls through to
`'noun'`, and `_determine_article` then emits `a`/`an` for it.

Measured on the persisted store:

```
concept labels tagged        259
tagged 'noun'                199  (77%)
noun-tagged adverbs           5   there, only, however, rather, now
noun-tagged adjectives        9   different, ancient, primary, ...
```

## The second half: the selector cannot see the error

`select_target` (`learn/macroplanning.py:251`) ranks candidates purely by
`0.7*weight + 0.3*confidence`. It has **no POS term**, so an adverb target with
a strong edge beats a noun target with a weaker one.

For `time`, the graph offers:

```
rel=semantic     target=years        pos=noun
rel=analogical   target=only         pos=noun   <- chosen
rel=analogical   target=well         pos=noun
rel=semantic     target=good         pos=adj
rel=analogical   target=way          pos=noun
rel=semantic     target=two          pos=noun
```

`only` wins on edge strength. Even if the POS map were correct, the selector
would still pick it, because `analogical -> only` is a stronger edge than
`semantic -> years`.

## So there are two independent defects

| # | defect | effect |
|---|---|---|
| 1 | POS map defaults 77% of labels to `noun` | adverbs/adjectives become noun-slot candidates |
| 2 | `select_target` has no POS term | a mislabelled target can win on edge strength alone |

Fixing only #1 removes the current examples but leaves the selector blind.
Fixing only #2 is impossible without #1 — there is nothing to filter on.

## Constraints on a fix

- No trained POS tagger: `nltk` is installed but **no tagger data is
  downloaded** (only a `corpora` dir), and there is no spaCy.
- `KNOWN_ADJS` (26) and `KNOWN_VERBS` (46) are hand lists.
- GloVe vectors ARE available (400k, projected to 64D) and there is a corpus
  (`data/corpora/grounded_descriptions.txt`, 512 lines) — so distributional
  evidence exists but is currently unused for category.
- The graph has 259 nodes; the vocab is 15k.

## Related

- `docs/BRAIN_POS_SELECTION.md` — how the brain selects a word's category
- `docs/BRAIN_LEXICAL_CATEGORY.md` — how category is learned/represented
- `docs/PLAN_FLAG_ON_MEASURED.md` — the flag measurement
- `docs/TYPING_ACCURACY_IS_CIRCULAR.md` — the same lesson for relation typing:
  a proxy that is not measured independently will look better than it is
