# Harvest: 500+ grounded pairs, from a pipeline that returned 0

**Status:** measured. The corpus bottleneck from #331 is cleared.

## Before

The LingGen harvest produced **nothing usable**. Two independent defects:

1. **#332** — `data/case_dist.json` was 73 NUL bytes. `CaseDistributionStore.load`
   raised `JSONDecodeError`, and `kb_describe`'s `except Exception: continue`
   swallowed it *after* a good Wikipedia description had been fetched. Every
   concept returned `None`.
2. **#333** — the concept list came from frequency alone. The raw top-N from
   `teen_seeds.txt` is function words: *which, there, when, been, very, more*.
   Harvesting those teaches the decoder to describe "which".

## After

```
pairs=500  prose=500 (100%)
duplicates: 0
```

Sample concepts:

```
water, education, programming, years, philosophy, environment, science, work,
development, mind, information, university, data, medicine, system, behavior
```

Measured Wikipedia yield on 25 sampled strong nouns: **25/25 (100%)**.

## The two fixes

**#332 — corrupt store must not discard good results**

- `CaseDistributionStore.load()` catches `JSONDecodeError` / `UnicodeDecodeError`
  / `OSError` and degrades to the empty store.
- `kb_describe` isolates the `case_infer` call in its own try/except. Casing is
  cosmetic; it must never discard a description.

**#333 — concepts must be nouns**

WordNet strong-noun test: the word's **first** (most frequent) synset must be a
noun **and** it must have ≥ 2 noun senses. WordNet alone is weak evidence —
`have`, `there`, `then` all have a marginal noun sense.

```
gravity first=n noun_count=3 strong=True
ocean   first=n noun_count=2 strong=True
have    first=n noun_count=1 strong=False
there   first=n noun_count=1 strong=False
then    first=n noun_count=1 strong=False
```

Same corpus, same frequency ranking, with the filter:

```
before: which, there, when, been, very, more, their, will, then, your
after:  marine, life, ethics, learning, species, natural, water, people,
        education, programming, research, climate, nature, time, history
```

## Making it run at all

Two operational problems, both fixed:

- **WordNet costs ~1.03s per lookup.** The corpus scan asks about 2000 words, so
  an uncached scan is ~34 min. Results now persist to
  `data/corpora/.noun_cache.json` (15,837 entries: 47s first call, 0.24s after).
- **The gateway kills long jobs.** 1000 Wikipedia lookups at ~0.8s is ~14 min.
  The harvest is now resumable: each pair is appended as it is fetched, and the
  existing file is loaded into the seen-set on startup. A killed run keeps
  everything it got.

Measured: 267 → 498 → 500+ pairs across repeated kills, with 0 duplicates.

## What this unblocks

The decoder can now be retrained on descriptions instead of aphorisms — the
fix #331 identified. `scripts/train.py --mode linggen` reads this file.

## Related

- #331 — the root cause (trained on aphorisms)
- #332 — the corrupt case store
- #333 — the noun filter, cache, and resume support
- `docs/DECODER_TRAINED_ON_APHORISMS.md`
