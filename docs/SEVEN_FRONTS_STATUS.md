# Seven fronts: where the measurement stands

> **Superseded (2026-11-19).** The "now" scores below are from the small
> (3–4-probe per front) tightened-probe run documented here; they are **not**
> the current measurement. The current honest **n=60** offline result, measured
> after the linear-`detect_gaps` instrument fix ([PR #438], `466fcd1`), is:
> conversation 95, grounded 57, opinions 80, thinking 13, speaking 7,
> code 100, language learning 80, **mean 62**. This supersedes the earlier
> n=60 result of mean 56 (conversation 93, grounded 17, opinions 87, thinking 8,
> speaking 7): that run used a 19 s/turn O(nodes×edges) gap scan that read
> absent edge attributes and flagged every node as a knowledge gap. The large
> grounded gain is attributable to that fix, not to the curriculum scaffold;
> the scaffold's own A/B contribution is thinking 5 → 13. The web arm at n=60
> timed out (2400 s, 207 of 434 probes, no valid summary), so **no n=60 web
> score is recorded**. The probe-tightening narrative below is retained as
> history, not as current status.

[PR #438]: https://github.com/oxiverse-ecosystem/ravana/pull/438

**Status (historical):** measured by `scripts/measure_fronts.py`. Scores below
are from the tightened small-probe run described in this note; see the
supersession note above for current values.

## The scores

| front | first | now | note |
|---|---|---|---|
| language understanding / conversation | 100 | **100** | |
| grounded factual answers | 100 | **100** | was 75; the honest hedge scored better than the false claim it replaced |
| opinions / stance | 100 → 67 | **100** | |
| thinking / reasoning | 33 | **0** | was 67, then 33 — see below |
| speaking / surface realization | 100 → 0 | **100** | |
| code generation | 0 | **100** | |
| language learning (online) | 40 | **100** | |
| **mean** | 68 → 49 | **86** | |

The 100→67→100 and 100→0→100 swings are the honest story: the first probes
passed defective output, the tightened ones caught it, and fixes then moved
real scores.

### Why thinking reads 0, and why that is the honest number

The score moved **67 → 33 → 0** as the probe was tightened. Each drop was the
probe getting honest, not the system regressing:

- **67** counted the empty counterfactual template *"everything that depends on
  X would shift; other systems would scramble to fill the gap"* as a causal
  answer, because it contains **"depends on"** — a causal keyword. An honest
  *"i'm not sure"* scored 0. The probe rewarded the nonsense it was written to
  detect.
- **33** after that template was removed.
- **0** after the probe also rejected the *"the closest i get to X"* association
  frame and required ≥4 distinct content words.

Measured per probe at 0:

| probe | causal keyword? | actual answer |
|---|---|---|
| why do you think it rains | no | "the closest i get to rains is rain and maybe fall…" |
| explain how a battery works | no | "honestly, battery is a bit outside what i know right now." |
| what would happen if the moon disappeared | no | "Moon is the only natural satellite of Earth." |

**None produces a causal explanation.** The system retrieves definitions and
associations; it does not yet assemble a mechanism. That is the real remaining
capability gap, and it is now measured rather than flattered.

Do not "restore" the 67. It was never real.

### Why grounded reads 75, and why that is not a regression

The one failing probe is `what causes rain`. Before #390 it returned
**"Rain leads to heat, and that in turn leads to climate"** — a false causal
claim that PASSED the probe (it has a subject and a predicate). After #390 it
returns an honest hedge, which the probe correctly marks BAD.

So the score fell because the probe got *better at catching* a wrong answer,
and the wrong answer was replaced by an honest one. **Do not read 75 as a
loss.**

**Update:** grounded has since returned to **100**. The gap was real and was
closed at the source — `rain` now has a stored definition (*"drops of liquid
water falling from the sky"*), acquired from a live web read and repaired
through four defects in the definition pipeline (a wrong-sense store, a quality
tie that made a bad value permanent, a skipped `X is described as Y` pattern,
and a participle captured into the value). See `docs/RAIN_DEFINITION_GAP.md`.

## The caveats that matter more than the number

**1. Six 100s measure my probes, not RAVANA.** Five scores moved on *probe
corrections* alone. Each correction was the same class of error:

| probe | moved | cause |
|---|---|---|
| speaking | 100 → 0 | passed a degenerate association template |
| opinions | 100 → 67 | passed deflection |
| opinions | 67 → 100 | failed a legitimate stance |
| language learning | 40 → 100 | import-checked modules instead of behaviour |
| code generation | 0 → 100 | never taught the function it asked for |

The recurring lesson: **a probe that does not set up the precondition it tests
will score correct behaviour as failure.** Every score prints its evidence for
audit.

**2. Grounded's 100 is STRUCTURAL, not truthful.** It checks subject +
predicate + not-deflecting. No regex judges truth. A wrong causal claim
(`rain → heat`) passed this probe for the whole session until it was traced at
the source (#390).

**3. Thinking's 33 understates the route.** Its probe asks about `battery` and
`moon`, where no assertable chain exists — so it measures fail-closed behaviour.
The route itself was verified directly: `sun leads to earth, and earth leads to
life` (#388). The probe was deliberately NOT tuned to look better.

## The defect pattern across 26 PRs (as of that round; 68 PRs merged on main as of 2026-11-19)

Almost nothing was a missing capability. It was plumbing:

| kind | examples |
|---|---|
| wrong string | `causes` vs `causal` (#378) |
| dropped field | `source_metadata` popped on pickle (#385) |
| absent after reload | unpickling skips `__init__` (#386) |
| stale flag | salad flag set before the good answer (#388) |
| missing fallback | no phrase vector in the stance branch (#383) |
| over-broad regex | bare `tell me` shadowing `tell me about` (#376) |
| proxy outliving its measurement | 0/32 typer written as `causal` (#390) |
| ordering | `if not known: return None` before the honest branch (#375) |

And one meta-lesson: **a proxy that is not measured independently will look
better than it is.** Relation typing 75.5%→0/32; determiner frame
0.787→0.730; noun-suffix table 0.803→0.608; explanation adequacy 1.000 on a
false chain.

## Genuinely open

- **Front 2 truth.** #390 removed one false claim at its source, but no
  output-side check judges truth. The next bad edge surfaces the same way.
- **The authored causal skeleton** (`engine_reasoning.py:3185`) contains
  hand-written priors; `gravity → earth` was removed (#386) but the rest are
  unreviewed. See `docs/AUTHORED_CAUSAL_SKELETON.md`.
- **Language learning from ordinary chat** is unmeasured — the probe exercises
  the code learner, which is a teach→recall loop but not language learning.

## Reproduce

```
uv run python scripts/measure_fronts.py
```
