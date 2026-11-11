# Validating the retrieval-vs-generation rule against live signals

**Status:** the rule runs on real engine signals and discriminates. Not yet
wired into dispatch.
**Module:** `ravana/src/ravana/learn/retrieval_generation.py`
**Tests:** `tests/unit/test_retrieval_generation.py` — 26 passed.

## What was run

The rule was fed signals from a real `CognitiveChatEngine` loaded with the
production state (`weights/ravana_weights.pkl`: 80 definitions, 99 concept
sources), across 14 subjects.

```
subject          R      T      D   decision
gravity        0.660  0.355  0.119  RETRIEVE
ocean          0.620  0.496  0.112  RETRIEVE
time           0.610  0.588  0.403  RETRIEVE
light          0.450  0.355  0.045  RETRIEVE
water          0.410  0.357  0.067  RETRIEVE
quantum        0.310  0.363  0.119  COMPOSE
photosynthesis 0.150  0.353  0.030  COMPOSE
consciousness  0.138  0.347  0.164  COMPOSE
love           0.130  0.469  0.522  COMPOSE
trust          0.000  0.350  0.000  COMPOSE
respect        0.000  0.350  0.000  COMPOSE
memory         0.000  0.350  0.000  COMPOSE
dream          0.000  0.350  0.000  COMPOSE
zzz            0.000  0.350  0.000  COMPOSE
```

5 retrieve, 9 compose. The separation is meaningful: every subject with a
stored definition or web source and a settled node retrieves; every subject
with neither composes.

## The rule behaves sensibly at the margins

**`time`** is the instructive case: R=0.610, T=0.588 — it retrieves by only
0.022. Its density is 0.403, which raised the bar substantially. A more
connected subject is harder to answer from a single stored form, which is the
neighbourhood-density effect the rule is meant to capture.

**`love`** composes despite having a web source (R=0.130) because its density
is 0.522 — the highest in the set — pushing T to 0.469. A heavily-connected
abstract concept does not have one stored answer, so composing is the right
call.

**`consciousness`** is marginal in the other direction: R=0.138, T=0.347.

## Two honest caveats

**1. `_concept_sources` is empty in a fresh engine.**

```
fresh engine:   _definitions 258, _concept_sources 0
production:     _definitions  80, _concept_sources 99
```

`_concept_sources` is populated by live web learning, so offline/fresh
engines always score `has_web = False`. That is an environment artifact of the
probe, not a rule defect — but it means the rule's `w_web` term is untested in
CI. Any wiring must tolerate an empty `_concept_sources`.

**2. The weights are judgment, not fitted.**

`w1..w6`, `t_base`, `k_pe`, `k_d`, `k_f`, `k_p` are the starting values
proposed in `docs/RETRIEVAL_VS_GENERATION.md` §10. Nothing has been fit to a
target. The rule is falsifiable (§11) but not yet validated against a
behavioural target — there is no ground truth for "should this have been
retrieved".

## Not done

**The rule is not wired into dispatch.** It is a pure decision function. Per
§10.8 it should be inserted at the top of `_generate_response`
(`response_gen.py:7189`) as a *gate* — not a reordering — so the existing
safety hierarchy (the ACC/ERN `_topic_grounded` monitor) still vetoes.

Wiring it is a separate change and needs its own verification.

## Related

- `docs/RETRIEVAL_VS_GENERATION.md` — the research and the rule
- `docs/DECODER_BLOCKED_BY_TWO_GATES.md` — why the decoder never runs today
- `docs/GENERATION_REACHABILITY_200.md` — the 200-query dispatch audit
