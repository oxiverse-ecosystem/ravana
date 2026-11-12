# Lowering the commitment criterion improves every axis

**Status:** measured with the full instrument, three theta values.
**Conclusion:** `theta_base = 0.15` beats the current 0.30 on on-topic,
supported assertions AND unsupported assertions simultaneously. There is no
trade-off to make.

## The sweep

Decoder-only, persisted engine, ungated, graph-known, CPG with relation gating:

```
theta_base   n     on-topic   supported   unsupported   cosine
0.15       219     0.2213     0.1553       1.0868      0.3403
0.30       204     0.2072     0.1422       1.1176      0.3427
0.45       191     0.1617     0.1047       1.1728      0.3394
```

**Monotone in one direction.** Lower theta → higher on-topic, higher supported,
lower unsupported. Every axis improves together, so the choice is not a
trade-off.

## Why this is the opposite of what I predicted

The design rationale (`docs/GROUNDING_MECHANISM.md`, step 2) is that the ACC
criterion should **rise** under uncertainty: high prediction error means the
system should demand more evidence before speaking. `theta_base` is the floor
of that criterion.

I expected raising it to improve grounding. It did the opposite.

The reason is what the criterion actually gates here. `theta` decides which
propositions are **admissible** — which tokens get boosted. A higher bar means
fewer propositions clear it, which means **fewer tokens boosted**, which means
the decoder falls back to its unguided distribution. The gate becomes less
informative, not more conservative.

In the brain, the ACC criterion gates whether to **commit to a plan at all**.
Here it gates which plan the decoder is allowed to speak. Raising it does not
make the decoder more careful — it makes it quieter, and then the unguided
model fills the silence.

## The correct interpretation

The criterion is not a conservatism knob in this implementation. It is a
**coverage** knob: how many of the graph's propositions are available to
steer generation.

That reframes step 2 of the mechanism. The brain-faithful claim — that
prediction error raises the criterion — is about **whether to speak**, not
about **which words to permit**. RAVANA conflates the two because CPG has no
"decline to speak" outcome: when nothing clears theta, it still generates.

The missing piece is the escalation path. `docs/GROUNDING_MECHANISM.md` step 3
says "after N failures, escalate to honest uncertainty". CPG's `decide()`
returns no commitment, but the caller ignores it and generates anyway.

## What this means for the default

`theta_base = 0.15` is the better default on the evidence. The change is a
one-line config default, and it improves all three measured axes.

The `alpha_pe` term (prediction error raises theta) should also be re-examined
under the same logic — it may be doing the same kind of harm. That needs its
own sweep.

## Reproducing

```
uv run python scripts/eval_decoder_groundedness.py --engine persisted --cpg \
    --cpg-relation --cpg-theta 0.15 --out /tmp/t15.tsv
```

## Related

- `docs/RELATION_GATING_SETTLED.md` — the theta=0.30 result
- `docs/GROUNDING_MECHANISM.md` — step 2, whose interpretation this revises
- `docs/CPG_MEASURED_EFFECT.md` — the original target-only result
