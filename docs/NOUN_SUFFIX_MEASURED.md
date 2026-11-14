# Noun-suffix tables: measured, and mostly not worth it

**Status:** evaluated. **Recommendation: do not add a noun-suffix table.**
`scripts/evaluate_noun_suffix.py`.

## Why this exists

`docs/POS_MISLABEL_ROOT_CAUSE.md` identified the POS map's fall-through to
`noun` as defect 1, and the research doc
(`docs/BRAIN_LEXICAL_CATEGORY.md`) recommended a noun-suffix table as one step
of its five-step rule. This note measures whether such a table actually helps
on the graph's own labels.

## Method

An earlier probe reported **0.803** for a table with `al/ory/ary`. A second
ad-hoc run of the *same table* reported **0.475**. Same table, same code path,
different number -- the two runs used different hand-labelled gold sets.

That instability is the finding, so this evaluation fixes the methodology:

- one gold set, labelled from dictionary knowledge only, **never** read from
  `_concept_pos` (reading the map under test is the
  `docs/TYPING_ACCURACY_IS_CIRCULAR.md` error);
- **5-fold cross-validation**, reporting the mean and the per-fold spread so a
  table that wins by noise is visible;
- the decision evaluated is the one that matters: *may this word head a noun
  phrase?*

Gold: 63 graph labels (40 noun, 23 non-noun).

## Result

```
table               mean      sd                      folds
baseline           0.732   0.055   0.69 0.69 0.69 0.75 0.83
A al+ory+ary        0.608   0.120   0.54 0.46 0.54 0.75 0.75
D no-al            0.732   0.055   0.69 0.69 0.69 0.75 0.83
E D+ic             0.747   0.053   0.77 0.69 0.69 0.75 0.83
F minimal          0.732   0.055   0.69 0.69 0.69 0.75 0.83
```

**Table A is WORSE than the baseline** (0.608 vs 0.732) and has double the
variance. It rescues 6 real nouns and breaks 11 real adjectives:

```
animal      baseline=non  A=noun  gold=noun   <- fixed
journal     baseline=non  A=noun  gold=noun   <- fixed
mammal      baseline=non  A=noun  gold=noun   <- fixed
natural     baseline=non  A=noun  gold=noun   <- fixed
environmental  baseline=non  A=noun  gold=non <- BROKEN
ethical        baseline=non  A=noun  gold=non <- BROKEN
global         baseline=non  A=noun  gold=non <- BROKEN
mathematical   baseline=non  A=noun  gold=non <- BROKEN
psychological  baseline=non  A=noun  gold=non <- BROKEN
... 11 broken in total
```

`-al` is genuinely ambiguous: `animal`, `global`, `moral`, `natural` are nouns;
`environmental`, `mathematical`, `psychological`, `physical` are adjectives. A
suffix cannot separate them.

**Table E (`-ic` added, no `-al`) is 0.747 vs 0.732** -- a +0.015 difference
with overlapping fold ranges (E: 0.69-0.83, baseline: 0.69-0.83). That is
noise, not a result.

## What this means

1. **Do not add a noun-suffix table.** The measured effect is zero to negative.
   The research doc's step 3 is not supported on this graph.
2. **The suffix-precedence defect is real but small.** The words it actually
   gets wrong are `climate`→verb (`ate`), `learning`→verb (`ing`),
   `often`→verb (`en`), `even`→verb (`en`), `oxygen`→verb (`en`), and
   `music`→adj (`ic`). That is ~6 of 259 labels, and fixing it is a
   **targeted correction**, not a table.
3. **Defect 2 is the bigger lever.** `select_target` ranks by edge strength
   with no POS term, so a mislabelled target can still win. A filter on the
   candidate pool addresses that directly, and it does not depend on getting
   the POS map perfect.

## Honest limits

- 63 gold labels is still small; the fold spread (±0.055) is the honest
  uncertainty. A table would need to beat 0.732 by more than that to be worth
  wiring in, and none does.
- Gold is one annotator's dictionary knowledge. `global`/`moral`/`natural`
  are borderline (they have noun uses), which is exactly why `-al` fails.
- The evaluation is over graph labels only, not the 15k decoder vocabulary.

## Reproduce

```
uv run python scripts/evaluate_noun_suffix.py
uv run python scripts/design_noun_suffix.py
```

## Related

- `docs/POS_MISLABEL_ROOT_CAUSE.md` -- the two-defect diagnosis
- `docs/BRAIN_LEXICAL_CATEGORY.md` -- the research spec (its §3.3 table is the
  one this note does not replicate)
- `docs/DETERMINER_FRAME_REPLICATION.md` -- the same inflation, for the
  determiner frame
- `docs/TYPING_ACCURACY_IS_CIRCULAR.md` -- the circular-gold-set trap
