# Does enabling the plan route change production output?

**Status:** measured. `scripts/measure_plan_flag_on.py` (corpus) and
`scripts/measure_plan_flag_on_chunked.py` (reachable set, resumable).

## First answer was wrong, and the correction matters

An early run over 40 corpus subjects reported **0/40 changed**, which looked
like "the flag is neutral". That was measuring the wrong population: every one
of those subjects has a stored definition, so `consult_internal` answers from
it (`HIT definition`) and the situation-model path is never consulted.

Restricting to subjects **without** a stored definition -- the only population
the plan route can reach -- the flag changes **114 of 138 (83%)**.

## Population

```
corpus subjects                     512
with a stored definition           333  (65.0%)
without a stored definition        179  (35.0%)   <- reachable set
measured (time-budgeted)           138
```

## Result on the reachable set (n=138)

```
changed                            114  (83%)
  cosmetic (whitespace / [source:] tag)  30
  real text change                        84
    hedge -> real answer                  10
    answer -> hedge                        0
    other                                 74
      ON longer (elaboration)            55
      ON shorter                         16
```

## The 10 hedge -> answer conversions are the point

```
possibility   OFF: honestly, Possibility is a bit outside what i know right now.
              ON : Possibility is the quality of being possible and may refer to:...
adaptation    OFF: i don't really have a solid grasp on adaptation so far.
              ON : In biology, adaptation has three related meanings. Firstly, ...
mouth         OFF: i don't really have a solid grasp on Mouth so far ...
              ON : A mouth, also referred to as the oral, is the body orifice ...
cause         OFF: i don't really have a solid grasp on cause so far.
              ON : Causality is an influence by which one event, process, ...
curiosity     OFF: hmm, i'm not totally sure about Curiosity yet, and that's okay
              ON : Curiosity can be a quality related to inquisitive thinking, ...
sleep         OFF: honestly, Sleep is a bit outside what i know right now ...
              ON : Sleep is a state of reduced mental and physical activity ...
fairness      OFF: honestly, Fairness is a bit outside what i know right now ...
              ON : Fairness or being fair can refer to:Justice in particular: ...
accuracy      OFF: honestly, accuracy is a bit outside what i know right now.
              ON : Accuracy and precision are measures of observational error ...
```

**Zero** went the other way (answer -> hedge). That asymmetry is the argument
for enabling: the flag converts honest uncertainty into a real answer, and
never replaces a real answer with uncertainty.

## The 74 "other" changes are mostly benign

55 are elaborations (ON adds a sentence: `week` gains "It is the standard
calendrical period between a day and a month"). 16 are shorter, and most of
those are the plan route dropping the `I believe X are` hedge prefix rather
than losing content -- e.g. `lakes`: OFF `I believe lakes are A lake is often
...` -> ON `A lake is often ...`. That is a fluency improvement, not a
regression.

## Honest limits

- `ones` -> `Ones may refer to:Ones (album), by Selena, 2002 ...` and
  `significance` -> `... It can also refer to:Significance (magazine), ...`
  are **disambiguation lists**, not definitions. The plan route is not the
  cause (the definition text is), but enabling the flag makes production
  surface them more often.
- 138 of 179 reachable subjects were measured; the remaining 41 were cut by
  the process time budget, not by a filter.
- `on_topic` / `unsupported` are graph metrics, not truth.
- The plan route's own measured quality (n=22): `unsupported 0.0909` vs
  `1.7368` for the fallback.

## Reproduce

```
uv run python scripts/measure_plan_flag_on.py --limit 40
uv run python scripts/measure_plan_flag_on_chunked.py \
    --limit 512 --chunk-size 30 --out <abs path>/reach.jsonl
```

## Related

- `docs/PROPOSITION_UNION.md` -- the two proposition sources (94.4% union)
- `docs/PLAN_NARRATIVE_POSITIVE.md` -- the plan route
- `docs/TYPING_ACCURACY_IS_CIRCULAR.md` -- why the vector proxy is not used
