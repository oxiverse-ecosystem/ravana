# The formulator now realizes the plan's construction

**Status:** fixed and measured. The plan was right; the formulator overrode it.

Instrument: `scripts/prototype_definition_plan.py`, persisted engine.

## Before

```
gravity   plan: gravity is a force       -> "It appears that gravity is force."
species   plan: species is a unit        -> "It appears that species is unit."
time      plan: time is a progression    -> "It appears that time is progression."
```

Three separate defects, all in the formulator rather than the plan.

## Defect 1: the typed vocabulary was absent from the verb lexicon

`VerbLexicon.COMPOSITION_RULES` had `semantic / causal / contrastive /
analogical / temporal / episodic / inverse` but NOT the graph's typed
vocabulary (`is_a`, `has_property`, `capable_of`, `used_for`, `part_of`).
Every typed edge fell through to `COMPOSITION_RULES["semantic"]`, so a typed
`is_a` plan was realized with a vague associative verb: "gravity contributes
to force".

Added the five typed relations plus a `copula` morpheme category (the copula is
a frame, not a verb).

## Defect 2: the copula was suffix-inflected

`_conjugate_root` appended "s" to everything, so `is -> iss`, `was -> wass`,
`am -> ams`, `were -> weres`. Added an irregular-verb table and restricted the
copula pool to present tense (a definitional claim is present-tense; the
frame's tense is chosen by `_determine_tense`).

## Defect 3: the article was suppressed for common nouns

Two independent causes, both measured:

1. `_determine_article` returned "" for EVERY object -- the last line was
   `return ""` with a comment saying concept labels are generic. So no object
   ever got an article.
2. Even once that was fixed, `_build_noun_phrase` dropped it: `force`, `unit`,
   `salt`, `sound`, `body` are all in `proper_nouns` (7,889 entries, seeded
   from every capitalized word mid-sentence in `teen_seeds.txt`, which catches
   sentence-initial common nouns) and in the realizer's broad
   abstract/uncountable sets.

Fixed by letting a learned common-noun POS tag override both the proper-noun
set and the abstract/uncountable heuristics -- for the ARTICLE only.
Capitalization and hedging still use the original sets.

**The bug that hid this**: `pos_map or {}` is wrong for `ConceptPosDict`, a
lazy dict whose `__len__` is 0 until a key is touched, so `or` discarded the
map and every override silently no-op'd. Explicit `is not None` checks now.

## After

```
gravity   plan: gravity is a force       -> "It appears that gravity is a force."
species   plan: species is a unit        -> "It appears that species is a unit."
time      plan: time is a progression    -> "It appears that time is a progression."
dog       plan: dog is a descendant      -> "It appears that dog is a descendant."
ocean     plan: ocean is a salt          -> "It appears that ocean is a salt."
```

on_topic 0.67, unsupported 0.000, verb=True, func_share 0.57 (was 0.50).

## Honest limits

- **`ocean is a salt`** -- the head-noun rule picked `salt` over `body`
  because `_concept_pos` tags `salt` as a noun and `body` was not reached.
  The POS map is the authority and it is imperfect.
- **`dog is a descendant`** is true but weak; the definition's head noun is
  "descendant of the wolf", and the graph has `descendant`.
- The `It appears that` hedge is the realizer's epistemic tag for a subject
  with no verified source. It is honest, not a defect.
- Still 30% recall on definitions (98/324) -- most definitions are not simple
  copulas.

## Reproduce

```
uv run python scripts/prototype_definition_plan.py gravity ocean species time dog
```

## Related

- `docs/DEFINITION_PROPOSITION.md` -- the definition-derived proposition route
- `docs/TYPING_ACCURACY_IS_CIRCULAR.md` -- why the vector proxy was deprecated
- `ravana/src/ravana/language/verb_lexicon.py` -- COMPOSITION_RULES, copula
- `ravana/src/ravana/language/surface_realizer.py` -- `_build_noun_phrase`
