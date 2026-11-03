# Capability: honest uncertainty for factual/world-knowledge queries

## What it does

When RAVANA holds no stored fact for a specific factual attribute of a named
entity (e.g. "what is the capital of france?", "who invented the telephone?"),
it now emits honest uncertainty instead of confabulating via analogical
reasoning. The analogical fallback ("france is like second — they sit close
in the same part of the map") is skipped for queries that contain a
factual-attribute word (capital, population, inventor, located, founded, …).

Conceptual/definitional queries ("what is entropy") still reach analogical
reasoning, since those are exactly the cases where analogy is appropriate.

## How it grew from the conversation

The round report (ravana-2026-11-03T1533Z) identified the confabulation:

> When no stored fact exists for "capital of france", `_consult_internal_knowledge`
> fell through to `_try_analogical_reasoning`, which returned a GloVe-neighbour
> confabulation ("france is like second"). This is a confabulation dressed as
> reasoning: the answer's content did not come from RAVANA's state.

The fix adds a seed vocabulary of factual-attribute words at the point where
`consult_internal(subj, self)` returns `None` (no stored fact), in
`ravana/src/ravana/chat/engine_self_query.py:2354` (`_consult_internal_knowledge`).

## Design

**Location:** `ravana/src/ravana/chat/engine_self_query.py:2499-2512`

When `consult_internal(subj, self)` returns `None`:

1. Read the seed vocabulary from `self._factual_attribute_words`, seeding it
   lazily on first call with a set of factual-attribute words (capital,
   population, inventor, invented, located, location, founded, founder,
   discovered, discovery, president, prime, minister, currency, language,
   area, height, weight, born, died, year, distance, speed, temperature,
   density, gdp).
2. Tokenise the lowercased query (`re.findall(r"[a-z']+", t)`).
3. If any token is in the factual-attribute set, return `None` so the caller
   emits honest uncertainty.
4. Otherwise fall through to `_try_analogical_reasoning(subj, user_input)`.

## Why this is SEED (not hardcoding)

- No authored reply prose — the method only decides *whether* to skip analogy;
  the actual uncertainty wording comes from the existing honest-uncertainty
  path.
- No per-topic tables — the seed is a vocabulary of attribute *words*, not
  question→answer mappings.
- Runtime-expandable — stored on `self._factual_attribute_words`, so the same
  store pattern as the learned lexicon can grow it at runtime. Removing one
  entry degrades gracefully (that attribute just falls back to analogy).
- The deciding test holds: RAVANA can change this by itself, through
  experience, via the same store pattern.

## Verified behaviour

- `ravana/src/ravana/chat/engine_self_query.py:2499` — guard is applied only
  after `consult_internal` returns `None`, so stored facts still answer
  directly.
- Conceptual queries ("what is entropy") contain no factual-attribute token,
  so they still reach analogical reasoning.
- Factual queries ("what is the capital of france") contain "capital", so
  they return `None` and the caller emits honest uncertainty.

## Related

- `ravana/src/ravana/chat/engine_self_query.py` — `_consult_internal_knowledge`
- `ravana/src/ravana/chat/brain_regions.py` — `consult_internal`
