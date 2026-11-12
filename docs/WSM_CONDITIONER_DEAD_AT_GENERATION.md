# The fitted W_sm conditioner is dead at generation time

**Status:** confirmed by reading the call site. This is the most likely reason
CPG cannot steer the decoder.

## The gap

`NeuralDecoder.generate()` accepts two parameters that carry the fitted
angular-gyrus projection:

```
initial_emb      (embed_dim,)  -- BOS initiation state, step 0 only
persistent_emb   (embed_dim,)  -- concept bias added to h at EVERY step
```

Both are derived from the 65-D sensorimotor signature via the fitted `W_sm`
(`neural_decoder.py:741-760`). The docstring is explicit that this is the
supervised grounding signal: *"given this embodied signature as the initiation
state, reproduce this human description"*.

**The generation call site passes neither.** `response_gen.py:1614`:

```python
generated = self.neural_decoder.generate(
    conditioning_embs=conditioning_embs,
    max_steps=24,
    bos_idx=bos_idx,
    eos_idx=eos_idx,
    temperature=temp,
    cerebellar_ngram=self.cerebellar_ngram,
    idx_to_word=self._decoder_idx_to_word,
    basal_ganglia=self.basal_ganglia,
    content_word_ids=content_word_ids,
    token_boost=_cpg_boost if _cpg_boost else None,
    token_suppress=_cpg_suppress if _cpg_suppress else None,
)
```

No `initial_emb`, no `persistent_emb`, no `sensorimotor_conditioning`.

## Why this matters

PR #318 fixed the conditioner not being **loaded** at boot
(`_linggen_W_sm is None` → shape `(75, 65)`). That fix is real and verified.

But loading it is not enough. The generation path never passes the signature
that would project through it, so the fitted matrix is inert at inference.
`train.py` fits `W_sm` on `(sensorimotor_conditioning, description)` pairs;
`generate()` is where it should be applied, and it is not.

This explains the measurement in
`docs/DECODER_OFF_TOPIC_REGARDLESS_OF_CPG.md`: the decoder produces
off-topic text regardless of CPG, because the one pathway designed to inject
the concept is not connected.

## Verified

```
W_sm loaded: True
NO initial_emb     concentration phenomenology idealism covered sulcata joined continues
WITH initial_emb   delivery sassoon benchmarking imposed iaa anthozoans sanitary real tec
```

Passing `initial_emb` does change the output, so the pathway works when
connected. Both samples are still salad here because the probe used a
synthetic 65-D signature, not a real fitted one -- the point is that the
pathway is live, not that my synthetic vector is meaningful.

## The fix

`generate()` does **not** accept `sensorimotor_conditioning` -- only
`train_on_sentence()` does. The caller must project and pass the embeddings:

```python
sig = self._combined_attr_encoder.attribute_vector(
    np.asarray(self._glove_vector(subject), dtype=np.float64))   # 65-D
W = self.neural_decoder._linggen_W_sm                            # (75, 65)
init = (W @ sig).astype(np.float32)
init /= np.linalg.norm(init)
generated = self.neural_decoder.generate(
    ...,
    initial_emb=init,        # step-0 BOS state
    persistent_emb=init,     # per-step concept bias
)
```

Both pieces exist on the engine already:

```
_combined_attr_encoder   present (engine.py:2101)
_glove_vector            present
_linggen_W_sm            (75, 65), loaded since PR #318
```

So this is a **wiring bug**, not a missing feature.

## Verified

```
W_sm: (75, 65)   sig: (65,)   init_emb: (75,) norm: 1.0

WITHOUT            concentration phenomenology idealism covered sulcata joined co
WITH init          color sunlit aerospace semiotic intricate presumption thai con
WITH init+persist  introductions karima leatherback cessation bins copepod imperm
```

The pathway is live and changes the output. **But all three are word salad** --
including the conditioned ones. So connecting it is necessary but not
sufficient: the fitted `W_sm` does not, on its own, produce coherent text from
a real concept signature.

That is a separate finding. It is consistent with `docs/WSM_IS_NOISE.md` --
the conditioner was measured as not helping. Wiring it up will let that be
re-tested properly, but the prior measurement says not to expect much.

## What to do

1. Wire `initial_emb` / `persistent_emb` at `response_gen.py:1614` (small,
   contained, fail-open when `W_sm` or the encoder is absent).
2. Re-run the instrument with and without it.
3. If the output is still salad, the conditioner is not the lever -- the
   decoder needs retraining with conditioning, not a fitted read-out.

## Related

- PR #318 -- fixed the conditioner not being loaded
- `docs/DECODER_OFF_TOPIC_REGARDLESS_OF_CPG.md` -- the measurement this explains
- `docs/WSM_IS_NOISE.md`, `docs/CONDITIONER_FITTING.md`
- `docs/ESCALATION_INERT.md` -- escalation is inert (separate finding)
