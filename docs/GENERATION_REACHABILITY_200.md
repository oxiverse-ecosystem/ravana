# Generation reachability: what actually runs

**Status:** measured, 200/200 queries, 0 errors.
**Method:** `scripts/audit_generation_reachability.py` — `sys.settrace` on
`_process_turn_impl`, recording the last `engine.py` line executed before each
turn returns. That line is the return site that produced the reply.

## The result

```
line     hits   pct    what
11671     138   69.0%  generation tail (successful turn)
8711       27   13.5%  idle path — notify_user_idle() then return
10845      11    5.5%  greetings
8934        6    3.0%  idle path
8735        4    2.0%
10835       3    1.5%
8761        3    1.5%
8491        3    1.5%
8513        1    0.5%
8448        1    0.5%
10744       1    0.5%
10813       1    0.5%
11203       1    0.5%
```

13 distinct return sites. The ranking was stable across three snapshots
(10, 60 and 200 queries) — 11671 and 8711 held the top two throughout.

## What this corrects

An earlier audit over 46 queries reported **0/46 reach generation**. That was
wrong, and it drove several wrong conclusions.

**69% of turns reach the generation tail.** The fixes landed since then
(#298's analogy gate and neighbour-quality check, the `RAVANA_TEST_MODE`
split, the `RAVANA_OFFLINE` separation) genuinely moved the number.

The early returns are **not** the wall. Only ~16% are preempted by the idle
path, and the rest are a long tail of one-offs.

## Which generation path runs

Instrumenting every generation entry point over 16 queries:

| path | calls | share |
|---|---|---|
| `_consult_internal_knowledge` | 16 | **100%** |
| `_web_direct_answer` | 11 | 69% |
| `_definition_response` | 7 | 44% |
| `_decomposition_generation_path` | 4 | 25% |
| `_reflective_response` | 1 | 6% |
| `_generate_with_situation_model` | **0** | — |
| `_generate_with_decoder_and_syntax` | **0** | — |

So the decoder path is not blocked by early returns — it is **never
selected**. `_consult_internal_knowledge` fires on every query and hands off
to the web/definition/decomposition paths, which produce the reply.

## The actual design question

Not "remove the early returns" but:

> **Should the decoder get a turn when the KB has an answer?**

The current hierarchy is deliberate — KB and web answers are more reliable than
free generation, so they are tried first. The decoder only speaks when
everything else fails, and on ordinary queries something else always succeeds.

This is a design decision, not a bug. See the companion research on how the
brain switches between retrieval and generation.

## Reproducing

```
uv run python scripts/audit_generation_reachability.py --n 200 --out reach.tsv
```

Resumable: re-running reads back `reach.tsv`, rebuilds the counters and skips
completed queries. `--fresh` starts over.

Expect ~10-12 min for 200 queries plus engine build. The trace is *faster*
than the untraced path (3.1 s/query vs 27.2 s/query) because it skips RAVANA's
per-line debug printing.

**Note:** a `sys.settrace` filter must match the code object
(`frame.f_code is impl_code`), not the filename — `process_turn` and
`_process_turn_impl` live in the same file, so a filename filter traces both
and the wrapper's return always wins.
