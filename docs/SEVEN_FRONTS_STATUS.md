# Seven fronts: where the measurement stands

**Status:** measured by `scripts/measure_fronts.py`. Last full run on a
current `main`: see the table below.

## The scores

| front | first | now |
|---|---|---|
| language understanding / conversation | 100 | 100 |
| grounded factual answers | 100 | 100 **(structural — see caveat)** |
| opinions / stance | 100 → 67 | 100 |
| thinking / reasoning | 33 | 33 **(route works; probe measures fail-closed)** |
| speaking / surface realization | 100 → 0 | 100 |
| code generation | 0 | 100 |
| language learning (online) | 40 | 100 |
| **mean** | 68 → 49 | **90** |

The 100→67→100 and 100→0→100 swings are the honest story: the first probes
passed defective output, the tightened ones caught it, and fixes then moved
real scores.

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

## The defect pattern across 26 PRs

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
