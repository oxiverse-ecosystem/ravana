# CPG moves the decoder: on-topic 4.5×, supported assertions 10×

**Status:** measured with `scripts/eval_decoder_groundedness.py --cpg`.
**Conclusion:** Committed-Proposition Gating works. It is the first
intervention that moves the decoder's groundedness numbers.

## The numbers

Decoder-only, persisted engine, ungated, graph-known:

```
                        CPG OFF    CPG ON    change
on-topic fraction        0.0508    0.2270    4.5x
supported assertions     0.0107    0.1101    10.3x
unsupported assertions   1.1337    1.1927    ~flat
content-word cosine      0.2937    0.3444    +17%
word count               8.83      9.27      +5%
```

**On-topic more than quadrupled.** Supported assertions — the graph-backed
claims the decoder actually makes — went from essentially zero to 0.11.

Per-subject, the change is visible:

```
subject   CPG OFF                      CPG ON
gravity   topic=0.00  unsup=1        topic=0.50  unsup=1
ocean     topic=0.00  unsup=0        topic=0.50  unsup=0
time      topic=0.80  unsup=1        topic=0.89  unsup=0
```

## What did not change

**Unsupported assertions stayed flat** (1.13 → 1.19). CPG boosts tokens that
realize committed propositions; it does not *penalize* tokens that do not.
The decoder still emits unsupported relations at the same rate.

That is the expected result of a boost-only mechanism, and it is the honest
limit of this step. The next increment is the other half of step 3: make
non-admissible tokens actively suppressed, not merely unboosted.

## How much room is left

On-topic 0.227 is still far below the narrative path's 0.242 and far below
what the graph supports. The mechanism is working but the boost
(`token_boost = 2.5 * strength`) is gentle relative to the decoder's learned
distribution.

The retry schedule also matters: `theta_base = 0.30` admits most propositions
on the first attempt, so the retry path is rarely exercised. A higher base
criterion would force more selective commitment.

## Reproducing

```
uv run python scripts/eval_decoder_groundedness.py --engine persisted --cpg \
    --out /tmp/cpg_on.tsv
```

Compare against the same command without `--cpg`.

## Related

- `docs/GROUNDING_MECHANISM.md` — the mechanism
- `docs/DECODER_ONLY_NUMBERS.md` — the pre-CPG baseline
- `docs/WHAT_THE_DECODER_ROWS_CONTAIN.md` — what the scored string contains
- PR #323 — the CPG module
