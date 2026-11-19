# Why `what causes rain` cannot be answered

**Status:** root cause found. Not a probe problem, not a regression, and **not**
an IntentForge outage — an earlier version of this note claimed IntentForge was
down. It was not. Corrected below.

## The symptom

`what causes rain` returns an honest hedge:

> the closest i get to rain is water and maybe fall — and from what i've seen,
> water is a transparent, tasteless, odourless liquid (H2O)...

The seven-fronts probe (small, historical probe set) marks this BAD, so
grounded reads 75 instead of 100. At the current honest n=60 measurement
grounded is **17/100** — see `BENCHMARK_60_ITEMS.md` §4.
**That is correct scoring of a real gap.** Before #390 the same query returned
"Rain leads to heat, and that in turn leads to climate" — a false causal claim
that PASSED the probe. The score fell because the probe got better at catching
a wrong answer and the wrong answer was replaced by an honest one.

## The gap: `rain` has no definition

```
definitions: 514
'rain' in definitions: False
related keys: water, waters, brain, training
```

So the honest hedge is the right behaviour — the system does not know what rain
is. Raising the score means acquiring the definition.

## Why web learning did not acquire it

`learn_from_web('rain')` runs and reports success:

```
learned 1 new things about rain
  -> "New York, NY Weather Radar | AccuWeather. Rain? Ice? Snow? Track storms..."
rain now in definitions: False
```

It found a page and stored a **snippet**, not a definition. The definition
extractor is regex-based and needs an explicit `X is Y.` / `X refers to Y.`
form (`web_learning.py:1740-1748`). The AccuWeather result is a page title with
no such form, so nothing is stored in `_definitions`.

## CORRECTION: IntentForge is UP

An earlier draft of this note said IntentForge (localhost:4000) was "down, so
the query falls through to DuckDuckGo". **That was wrong** — asserted from a
single failed call without checking. Measured properly:

```
TCP 0.0.0.0:4000 LISTENING
GET /health        -> HTTP 200 in 0.026s
GET /search?q=rain -> HTTP 200, 8 results
```

IntentForge is the FIRST provider in the list (`web/learner.py:179`) and it
served the query. The earlier `RAVANA_OFFLINE=1` run returned
`('offline - web learning disabled', 'offline')` — that is the harness's own
guard, not an outage.

## The actual root cause: the snippet is not a definition

`learn_from_web('rain')` runs and reports success, but stores a **snippet**:

```
read about rain but already knew the words
  -> "National Aquarium - Upland Tropical Rain Forest. Explore // Award-Winning
      Exhibits // Upland Tropical..."
```

The definition extractor is regex-based and needs an explicit `X is Y.` /
`X refers to Y.` form (`web_learning.py:1740-1748`). IntentForge's top hits for
`rain` are an aquarium exhibit and weather-radar pages — none has that form, so
nothing is written to `_definitions`.

The results DO contain definition-shaped text further down (Dictionary.com:
*"water that is condensed from the aqueous vapor in the atmosphere and falls to
earth in drops..."*), but the extractor takes the top result and does not scan
for a definition-bearing candidate.

## What would fix it

Two independent options, in order of leverage:

1. **Scan candidates for a definition-shaped sentence** rather than taking the
   top result. The regexes already exist; they are applied to one candidate.
   IntentForge returns 8 results and at least one carries definitional text.
2. **Add a Wikipedia summary source.** `page/summary/Rain` returns exactly the
   right sentence and is reachable from this host with the browser User-Agent
   the engine already sends (`web_learning.py:644`:

   > Rain or rainfall is a form of atmospheric precipitation composed of
   > droplets of water aerosols that have condensed from atmospheric water
   > vapor and fall by gravity.

Option 1 is better: it uses results already fetched and generalises to every
topic, not just those Wikipedia covers.

## Honest limits

- A summary/definition is not a causal explanation. It fixes `what is rain`;
  `what causes rain` additionally needs a causal edge, which is the separate
  provenance problem.
- Either change alters behaviour for **every** query, so it wants a
  before/after measurement of definition-acquisition rate, not one example.
- The `RAVANA_OFFLINE=1` guard in the harness disables web learning entirely;
  any web measurement must run with the network up.
