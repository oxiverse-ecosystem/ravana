# The decoder was trained on aphorisms, not descriptions

**Status:** measured. This supersedes `PENALTY_SCALE_ROOT_CAUSE.md` -- the
penalty scale is a real defect but NOT the cause of the salad.

## The correction

I measured the penalty scale and found it 56× the logit range, then concluded
it was the root cause. **It is not.** With the penalties completely removed
(patched to `-= 0.0`), the output is unchanged:

```
gravity  penalties  : sharma therapies axiomatization mohammedan pending dishonor ganges dwi
gravity  NO penalties: sharma antioxidative array scientifically dismiss massage explor
ocean    NO penalties: romantic siegel briefly davies lorraine bioassay ecotourism exer
music    NO penalties: page nutrient appearance eugene steinfatt composite breeding pha
```

Nearly identical. The penalties are a real defect (they are scaled wrong) but
they are not why the decoder produces salad.

## What actually causes it

The training corpus. `scripts/train.py:98`:

```python
primary = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
```

14,037 lines of **aphorisms and book blurbs**:

```
A bird does not sing because it has an answer. It sings because it has a song.
A clever person turns great troubles into little ones, and little ones into none at all.
A fascinating, innovative, and optimistic look at how humanity and technology...
```

The intended grounding corpus, `grounded_descriptions.txt`, is **250 lines** --
two orders of magnitude smaller, and it is not the primary corpus.

So the decoder learned the *style* of aphorisms and blurbs: fluent,
well-formed, and about anything. That is exactly the observed failure mode --
fluent, grammatical, empty.

## Why the metrics looked fine

```
_avg_cross_entropy : 1.59   (perplexity 4.9)
CE on real sentences: 0.7 - 2.8
```

The model is genuinely well-fit -- to the wrong task. Teacher-forced CE on
real sentences is low because aphorisms and descriptions share vocabulary and
syntax. But the *conditional* distribution it learned is "produce an
aphorism", not "describe this concept".

The decisive test:

```
CE on the decoder's OWN output : 3.65 - 6.98
CE on 'gravity is a concept'   : 1.04 - 1.32
```

It does not recognize its own generations as language. Greedy argmax over a
flat, aphorism-trained distribution walks off the manifold immediately.

## Temperature does not help

```
temp=0.05  inference isran gold chrysophytes mitochondrial styles attrition entra
temp=0.2   eggplant dickinsonia routing delhi forkful exteriors exit construct pr
temp=0.5   deprecated esa particuli radula effortlessly life' licensed encouragin
temp=1.0   nazwa morphologist's ideological edged engberg preventing lid observat
temp=2.0   recognize doctrine payback manned puts stood prototypical connectionis
```

Every temperature produces salad. At temp=2.0 the output is more English-like
(*"recognize doctrine payback manned puts stood"*) because near-uniform
sampling favours common words -- but it is still not about the subject.

## What this means for the whole session's findings

Every intervention measured this session was trying to steer a decoder that
was never trained to be steered:

- **CPG** (PR #327) improves unsupported assertions -- but the mechanism is
  not "steering the decoder"; the decoder's output is withheld either way.
- **Escalation** is inert -- because the decoder is not the binding constraint.
- **The W_sm conditioner** (#330) is dead at generation -- necessary to fix,
  but the conditioner was fitted on 250 lines of descriptions against a
  decoder trained on 14,037 lines of aphorisms.
- **The penalty scale** is 56× too large -- a real bug, but not the cause.

## The actual fix

Retrain the decoder on descriptions, not aphorisms. `grounded_descriptions.txt`
is 250 lines -- far too small. The corpus needs to be:

1. Concept-anchored: `<concept>\t<description>` pairs.
2. Large: tens of thousands of pairs, not 250.
3. Consistent in form, so the conditional distribution is "describe this
   concept" rather than "produce an aphorism".

The harvest pipeline exists (`train.py linggen` -- "harvest grounded corpus").
It produced 250 lines. That is the bottleneck, not the model.

## Related

- `docs/PENALTY_SCALE_ROOT_CAUSE.md` -- the penalty defect (real, not causal)
- `docs/DECODER_OFF_TOPIC_REGARDLESS_OF_CPG.md`
- `docs/WSM_CONDITIONER_DEAD_AT_GENERATION.md`
- `docs/ESCALATION_INERT.md`
- `scripts/train.py:98` -- the corpus choice
