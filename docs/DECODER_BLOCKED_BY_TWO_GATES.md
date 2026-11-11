# The `use_linggen` flag is ON in production, and the decoder still never runs

**Status:** traced against the real persisted state, not a fresh engine.
**Corrections:** this file supersedes the "never set to True" claim in
`WHY_DECODER_NEVER_RUNS.md`. That claim was measured on a fresh `baby_mode`
engine with an empty `data_dir`, which never loads the persisted weights.

## The flag is on

```
weights/ravana_weights.pkl   52,585,666 bytes   Nov 9 15:51
use_linggen in saved state:  True
turn_count:                  18
```

Loading that state as production does:

```
use_linggen:        True
neural_decoder:     NeuralDecoder
_decoder_vocab_built: True
```

So `engine.py:1278` (`self.use_linggen = False`) is only the *default*. The
promoted value is persisted, and it is `True`. **The decoder is enabled in
production.**

## And it still never fires

Instrumenting every generation entry point over 8 queries with the real state:

```
_consult_internal_knowledge      8     <- every query
_web_direct_answer               5
_definition_response             3
_decomposition_generation_path   2
_reflective_response             1
_generate_with_situation_model   0
_generate_with_decoder_and_syntax 0
```

## Why: two independent blockers

**Blocker 1 — `_generate_response` is not reached on most turns.**

```
why do we sleep    _generate_response=1  situation_model=0
what is love       _generate_response=0  situation_model=0
what is gravity    _generate_response=0  situation_model=0
```

Two of three turns return before the generation funnel. That is the outer
early-return chain in `_process_turn_impl` — the 13 sites measured in
`GENERATION_REACHABILITY_200.md`.

**Blocker 2 — even when `_generate_response` runs, `situation_model` does not.**

`use_linggen` is `True`, the decoder exists, the vocab is built — and the SM
path still fires 0 times. Something between the funnel entry and the
`use_linggen` gate returns first.

From the codebase audit, the inner order in `_generate_response`
(`response_gen.py:7189`) is:

```
1. style conditioning
2. PFC classifier -> qtype
3. ctx.subject in self._definitions  -> _definition_response      (returns)
4. try_situation_path -> _reflective_response                     (returns)
5. use_linggen gate -> _generate_with_situation_model
6. ventral / dorsal paths
7. graph fallback
```

Step 3 and step 4 both return before step 5. So the decoder is behind two
more gates *inside* the funnel, on top of the outer chain.

## Corrected conclusion

The decoder is not "one flag away". It is behind:

```
outer early-return chain (13 sites, 31% of turns exit here)
  -> _generate_response
       -> _definition_response gate
       -> _reflective_response gate
       -> use_linggen gate
            -> _generate_with_situation_model
```

The flag being `True` was necessary but not sufficient. Both gates above it
must also fail to return.

## What this changes

The question is still **"when should the decoder compete"**, but the answer
has to be per-query rather than a global flag. A global `use_linggen` cannot
express "use the decoder for this question but not that one" — and the brain
does not have a global switch either.

## Related

- `docs/GENERATION_REACHABILITY_200.md` — outer dispatch, 200 queries
- `docs/WHY_DECODER_NEVER_RUNS.md` — earlier, superseded on the flag question
- `docs/DECODER_REACHABILITY.md` — the original narrow finding
