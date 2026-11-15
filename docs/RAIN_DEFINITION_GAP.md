# Why `what causes rain` cannot be answered

**Status:** root cause found. Not a probe problem, and not a regression.

## The symptom

`what causes rain` returns an honest hedge:

> the closest i get to rain is water and maybe fall — and from what i've seen,
> water is a transparent, tasteless, odourless liquid (H2O)...

The seven-fronts probe marks this BAD, so grounded reads 75 instead of 100.
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

## Why AccuWeather and not Wikipedia

Wikipedia's summary API returns exactly the right text:

```
Rain or rainfall is a form of atmospheric precipitation composed of droplets
of water aerosols that have condensed from atmospheric water vapor and fall by
gravity.
```

But the search provider list (`web/learner.py:178-183`) has **no Wikipedia
source**:

```python
self.apis = [
    ("intentforge", "http://localhost:4000/search?q={}", ...),   # down
    ("local_api",   self._local_url, ...),                       # down
    ("duckduckgo",  "https://html.duckduckgo.com/html/?q={}", ...),
    ("oxiverse",    "https://api.oxiverse.org/search?q={}", ...),
]
```

IntentForge (localhost:4000) is tried first and is not running, so the query
falls through to DuckDuckGo, whose top result for `rain` is a weather-radar
page. The engine's own fetcher DOES send a browser User-Agent
(`web_learning.py:644`), so Wikipedia is reachable from this host — verified
directly against `page/summary/Rain`.

## The fix that follows

Add a Wikipedia summary source to the provider list, ahead of the generic
scrapes. It is the highest-yield definition source available:

- it returns a single declarative sentence in exactly the form the extractor
  wants;
- it is reachable from this host with the User-Agent the engine already sends;
- it needs no API key.

Expected effect: `learn_from_web('rain')` stores
`"Rain or rainfall is a form of atmospheric precipitation..."` in
`_definitions`, and `what causes rain` / `what is rain` answer from it.

## Honest limits

- Wikipedia's summary endpoint 403s without a browser User-Agent; the engine
  sends one, but any new caller must too.
- A summary sentence is a definition, not a causal explanation. It would fix
  `what is rain`; `what causes rain` additionally needs a causal edge, which is
  the separate provenance problem.
- Adding a provider changes search behaviour for every query, so it wants a
  measurement of definition-acquisition rate before and after, not just this
  one case.
