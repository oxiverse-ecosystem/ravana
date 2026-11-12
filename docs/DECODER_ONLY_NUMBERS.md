# The decoder-only numbers are far worse than the aggregate

**Status:** measured independently, confirming the instrument's per-strategy
split.
**Consequence:** the headline baseline (on-topic 0.133) **averages the decoder
with its fallbacks**. The decoder itself is much worse, and the gate is doing
more work than the aggregate suggests.

## The split

`scripts/eval_decoder_groundedness.py`, persisted engine, ungated, graph-known:

```
strategy                    n     on-topic   cosine   unsupported
situation_model_decoder   207      0.069     0.313      1.179
situation_model_narrative 232      0.242     0.371      2.112
situation_model_syntax      1      0.364     0.564      2.000

all ungated, graph-known   440      0.151     0.344      1.698
gated, graph-known         255      0.384        —        0.673
```

**The neural decoder scores on-topic 0.069** — under 7% of its content words
are graph-neighbours of the subject. The narrative path, which is not the
neural network at all, scores 3.5× better.

The per-row trace shows the pattern starkly: **every `decoder` row has
`topic=0.00`**, while narrative rows land at 0.50-0.88.

## What this corrects

`DECODER_GROUNDEDNESS_BASELINE.md` reported the aggregate (on-topic 0.133) as
"the decoder's" score. That was wrong — it mixes the decoder with two fallback
paths that are substantially better.

The decoder-specific number is **0.069**. The problem is worse than the
headline suggested, and it is specific to the neural path.

## The gate is doing more than it appears

```
ungated, all strategies   on-topic 0.151   unsupported 1.698
gated, all strategies     on-topic 0.384   unsupported 0.673
```

Gating raises on-topic from 0.151 to 0.384 and cuts unsupported assertions by
60%. That is a large effect — much larger than the earlier "rejects 43% of
outputs" framing implied.

The gate is a genuinely effective filter. It is not the problem. The decoder
feeding it is.

## What this means for the fix

Two things are now clear:

1. **The decoder is the worst component**, at 0.069 on-topic. Improving the
   conditioning input (the R² = 0.22 finding) targets exactly this.
2. **The gate works and should be kept.** Any change must not weaken it. The
   earlier framing — "the gate makes the decoder rarer, not better" — was
   misleading: the gate is the only thing keeping output quality at 0.384
   rather than 0.151.

## Also worth noting

`corr(on-topic, unsupported) ≈ 0.03` — essentially zero. Being on-topic does
not protect against ungrounded assertions. A decoder can name the right
concepts and still attach them to relations the graph does not support.

That is exactly the failure CPG addresses: commit to a *proposition*, not just
a topic.

## Related

- `docs/DECODER_GROUNDEDNESS_BASELINE.md` — the aggregate baseline (superseded
  on the decoder-specific number)
- `docs/CONDITIONING_INPUT_LACKS_SIGNAL.md` — why the decoder is off-topic
- `docs/GROUNDING_MECHANISM.md` — the proposed fix
