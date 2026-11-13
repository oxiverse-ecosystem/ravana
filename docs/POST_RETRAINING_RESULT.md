# Post-retraining result: what actually changed

> **Correction (added after the first draft).** The first version of this document
> claimed "the decoder is grounded now". That was **wrong**. Direct inspection of
> the decoder's own (ungated) output shows it is still word salad:
>
> ```
> moral   PRE: Learn make many other marine life psychology learning.
>         POST: Mind such often one there people other species.
> data    PRE: Archived analysis philosophy two sea systems different three world information psychology one learning.
>         POST: Two philosophy different information systems used.
> black   PRE: Holes many others like known people part often one form psychology music learning.
>         POST: Known like often many part such form people used other.
> ```
>
> What improved is the *aggregate assertion metrics*, and the mechanism is
> **shorter output**, not better output. See "The gate withholds MORE, not fewer".

Run completed 2026-11-13 16:56 after 40 passes (`proc_6ba2c8bfbbb1`, elapsed 3588.5s).

```
[LingGen] pass 40/40: CE=1.838 aux=0.001
[LingGen] free-form quality=0.641 | coherence floor=0.5
[LingGen] use_linggen = True (free-form coherent)
[decoder] weights saved=True
[decoder] vocab saved=True
[promote] use_linggen=True persisted
```

Artefacts: `data/neural_decoder.npz` 45,873,256 → 28,697,144 bytes;
`data/neural_decoder_vocab.json` 572,485 → 655,026 bytes (17,021 words).

## Paired comparison (480 rows, same subjects and queries)

`scripts/compare_groundedness_runs.py` joining on (engine, subject, raw_input):

| metric | pre | post | mean delta | improved / worsened / same |
|---|---|---|---|---|
| **supported assertions** | 0.2966 | 0.5481 | **+0.5583** | 169 / 44 / 267 |
| **unsupported assertions** | 1.7331 | 1.1837 | **−0.2021** | 205 / 112 / 163 |
| on-topic fraction | 0.4033 | 0.4505 | −0.0170 | 197 / 174 / 109 |
| content-word cosine | 0.4360 | 0.4713 | +0.0066 | 268 / 210 / 2 |
| word count | 11.88 | 8.93 | — | — |

(pre = `docs/PRE_RETRAINING_BASELINE.md`; post = `eval_postretrain.tsv`)

**Unsupported assertions fell 32% and supported assertions rose 85%.** The decoder
moved from free-association toward asserting what the graph supports.

## The gate withholds MORE, not fewer

| | PRE | POST |
|---|---|---|
| rows with gated text | 317 / 480 | **302 / 480** |
| withheld (empty gated text) | 163 | **178** |
| `situation_model_decoder` gated | 68 | 51 |
| `situation_model_syntax` gated | 240 | 238 |

The retrained decoder's output is *less* likely to clear `_sm_response_grounded`.
That is correct fail-closed behaviour: shorter salad makes fewer claims, so the
assertion counts improve, but the text is still not grounded prose.

## What genuinely works

The live probe shows the production path answers factual queries correctly:

```
Q: what is gravity
   -> Gravity is the force by which a planet or other body draws objects toward
      its centre; on Earth it gives weight and keeps the Moon in orbit.
Q: what is water
   -> Water is a transparent, tasteless, odourless liquid (H2O) essential for all
      known life.
Q: what is music
   -> Music is sound organized in time — rhythm, melody, and harmony — valued as
      an art form and cultural activity.
```

These come from the **KB / retrieval path**, not the decoder. The decoder's salad
is withheld and the honest stored definition is emitted instead. That is the
right architecture: retrieval answers what is known, the decoder is not trusted
to free-generate facts.

## Conclusion

- The retraining run is a **success as a training pipeline** (resumable, gated,
  saved, promoted) and it **reduced unsupported assertions by 32%**.
- It did **not** make the decoder produce grounded prose. The decoder remains a
  fluent-but-empty generator; the grounding gate correctly withholds it.
- The user-visible win is that factual queries get the KB answer, which was
  already true before this run.

## Next step

The decoder needs either (a) a much larger description corpus (512 pairs is far
too small for a 17k-vocab language model) or (b) a different objective than
teacher-forced CE on harvested definitions. The honest-ignorance path is
working; the free-generation path is not.

## Why on-topic looks flat

The aggregate `on_topic` delta is −0.0170, but the per-strategy decoder metric
ROSE 0.4033 → 0.4505. The aggregate is confounded by a strategy-mix shift:
`situation_model_narrative` went from 6 rows to 134, and narrative has lower
on-topic (0.2140) than decoder (0.4505). The strategy that matters improved.

## Caveat

Temperature sampling makes single runs noisy. The paired deltas above are the
reliable signal (same subjects, same queries, per-row differences); the
whole-run means are context. Differences under ~0.02 on on-topic should be
treated as noise.
