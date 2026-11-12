# Decoder groundedness: the baseline

**Instrument:** `scripts/eval_decoder_groundedness.py` (1161 lines, read-only on
production code).
**Method:** builds a `CognitiveResponseContext` exactly as `process_turn` does
(activate → spread → situation model → ctx), then calls
`engine._generate_with_situation_model(ctx)` twice per subject — once with the
grounding gate disabled (what the network says) and once in production config
(what actually gets emitted).

## Headline numbers

```
engine state        n     mean_on_topic   mean_unsupported   mean_cosine
baby (tempdir)    210        0.5021            2.0143           0.4076
persisted (real)  440        0.1501            1.6955           0.3422
```

**The real production state is worse than a fresh engine** on every axis:
on-topic 0.150 vs 0.502, cosine 0.342 vs 0.408.

That is the opposite of what you would expect if the decoder had learned
anything useful from the graph. It suggests the persisted decoder has drifted
toward fluent generality.

## What the numbers mean

**`mean_on_topic = 0.1501`** — only ~15% of the decoder's content words are
graph-neighbours of the subject it was asked about. It is off-topic 85% of the
time.

**`mean_unsupported = 1.6955`** — nearly two assertions per output that attach a
relation word to a content word the graph does not support. That is the
measured form of *"It is opposite to data."*

**`mean_cosine = 0.3422`** — the output's content words sit at cosine 0.34 from
the subject vector. For reference, the grounding gate's own threshold is around
0.45, so the typical output is well below the bar the gate uses.

## Breakdown by strategy

```
strategy=narrative   n=252
strategy=decoder     n=187
strategy=syntax       n=1
```

The decoder path and the narrative path are both heavily used, so the problem
is not confined to one code path.

## Gated vs ungated

```
UNGATED, graph-known   n=440
GATED,   graph-known   n=249
```

The gate rejects **191 of 440** outputs — 43%. So `_sm_response_grounded` is
doing substantial work, and the 249 that pass still carry ~1.7 unsupported
assertions each.

**The gate is necessary but nowhere near sufficient.** It filters the worst
half; it does not make the survivors true.

## What this tells us about the fix

Two independent problems, and they need different fixes:

1. **Off-topic (0.150).** The decoder is not being steered by the subject.
   Consistent with the conditioner bug found in PR #318 — the fitted `W_sm`
   projection was never loaded at boot, so the grounding signal the training
   pass fit was discarded at inference.
2. **Unsupported assertions (1.70).** Even on-topic words get attached to
   relations the graph does not support. That is the problem
   `docs/GROUNDING_MECHANISM.md` addresses with Committed-Proposition Gating:
   commit to a proposition before decoding, gate tokens at selection time, and
   retry rather than fall through.

## Reproducing

```
uv run python scripts/eval_decoder_groundedness.py --out /tmp/eval.tsv
```

Resumable: re-running reads back the TSV and skips completed subjects.

## Caveats

The instrument measures **surface overlap with the graph**, not truth. A
sentence can be on-topic and still false, and a true sentence can score low if
the graph is missing the relation. It is a proxy — but it is a proxy that
discriminates, and it is the number a fix has to move.

## Related

- `docs/GROUNDING_MECHANISM.md` — the proposed mechanism (CPG)
- `docs/GROUNDING_PASS_TOO_WEAK.md` — why the grounding pass was outweighed
- `docs/SESSION_GOAL_grounded_decoder.md` — the session goal
- PR #318 — the conditioner-at-boot fix, which should move `mean_on_topic`
