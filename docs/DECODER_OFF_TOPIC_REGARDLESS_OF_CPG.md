# CPG does not steer the decoder -- the decoder is off-topic regardless

**Status:** measured. This corrects a conclusion I drew in
`docs/ESCALATION_INERT.md`.

## The wrong conclusion

I compared CPG-off vs CPG-on and saw:

```
CPG off     decoder 226 / narrative 217 / syntax   1
CPG on      decoder   0 / narrative 178 / syntax 302
```

and concluded "CPG redirects every decoder output to the narrative path".
That is not what is happening.

## What is actually happening

Instrumented the decoder call inside `_generate_with_situation_model`:

```
CPG OFF  text='archived one such psychology like ethics learning'  grounded=False
CPG ON   text='archived ethics learning theories such'             grounded=False
```

**Both are ungrounded.** The subject is `gravity`; neither output mentions
gravity, force, or anything related. `_sm_response_grounded` returns False for
both, so the grounding gate withholds the text and falls through to the
narrative path.

CPG changes the wording slightly but does not make the output on-topic. The
decoder was never going to pass the gate for this subject either way.

## Why the counts differ

The decoder arm count (226 vs 0) is not "CPG removed the decoder". It is that
with CPG on, the boosted tokens push the output further from the subject, so
it fails the gate more often. With CPG off, some outputs happen to pass.

The decoder is not being skipped -- it runs, produces 5-6 words, and is
withheld. The trace confirms `generate` is called every time.

## What this means for the mechanism

CPG is supposed to commit to a lexical-semantic plan and steer the decoder
toward it. Measured, it does not:

- The decoder's output is off-topic with and without the boost.
- The boost changes a few tokens but not the topic.
- The gate withholds the result either way.

The token boost is too weak to overcome the decoder's learned distribution.
This is consistent with `docs/WSM_IS_NOISE.md` and
`docs/BOOST_STRENGTH_NOT_THE_LEVER.md` -- the decoder's conditioning pathway
is not carrying the signal.

## The honest summary of the session's measurements

| finding | status |
|---|---|
| escalation is inert at scale | confirmed -- identical to 4 decimals |
| CPG removes the decoder | **wrong** -- the gate withholds ungrounded output |
| the decoder is off-topic regardless of CPG | confirmed |
| CPG improves unsupported assertions (PR #327) | still stands -- but the mechanism is not "steering the decoder" |

## What to do next

The decoder's conditioning is not carrying the graph signal. Either:

1. The conditioning embeddings are not reaching the decoder (check
   `_align_to_decoder` and the `initial_emb` / `persistent_emb` path).
2. The decoder's learned distribution dominates the conditioning (the
   `W_sm` conditioner is fitted but may not be applied -- see
   `docs/CONDITIONER_FITTING.md`).
3. The decoder needs retraining with conditioning, not just a fitted read-out.

Option 2 is the most likely and the cheapest to check: PR #318 fixed the
conditioner not being loaded at boot. Verify it is loaded **and applied** on
the live path.

## Related

- `docs/ESCALATION_INERT.md` -- the inert-escalation finding (still valid)
- `docs/ESCALATION_MEASUREMENT_CONFOUND.md` -- the confound
- `docs/WSM_IS_NOISE.md`, `docs/BOOST_STRENGTH_NOT_THE_LEVER.md`
- PR #318 -- the conditioner load fix
