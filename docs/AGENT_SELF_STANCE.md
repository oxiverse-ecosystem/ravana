# Agent Self-Stance Formation & Recall — RAVANA's own learned views

RAVANA is not a passive mirror of your opinions; it *forms, records, and recalls its
own* stance on a topic it has discussed with you. When you have expressed a real
view, the agent derives a grounded lean from that evidence, stores it as its own
stance (durable across turns and sessions), and answers a self-opinion question
from that store. When no evidence exists anywhere, it stays honestly silent.

This capability was added in round **2026-08-11T1328Z** (feature `t_e7371e2c`,
commit `ba6c912`). All claims below are backed by a live in-process probe
(`dim=64`, offline) and the unit suite `tests/test_dehardcode_plan.py`
(`test_F1`/`test_F2`/`test_F3`/`test_F4`, 4 passed). No LLM, no retraining, no authored reply
pool — the words come from computed polarity/confidence, wrapped in a short
connective.

## The gap it fills

A self-opinion question — *"what do you think about X?"* / *"what's your read on
X?"* — previously fell through to the hollow *"i'm still figuring that out"* frame
even when **you had spent turns stating strong views on X** (the round's residual
limitation T2/T5/T28/T48/T60/T65/T74). Worse, the *primary* answer path
(`engine.py` ~2624) rendered `opinions.query_stance` — **your** stance store — as
`"i'm {_w} {topic}"`, i.e. it presented **your** view as **the agent's own** and
never recorded an agent-owned stance. The agent had no durable, attributable view
of itself to recall next time: a self/other boundary leak.

Agent Self-Stance Formation & Recall is the remedy:

1. **RECALL** — answers from RAVANA's own durable `_agent_stances` store (personality
   continuity across turns/sessions; persisted in save/load).
2. **FORM** — grounds a *new* stance on your real learned stance (`UserStanceStore`),
   attenuated (the agent *leans*, never *copies*), records it as its own so it
   persists + recalls stably.
3. **Honest silence only with zero evidence** — no constitutive value, no recalled
   stance, no user stance.

## What it does (verified)

Probe: user states *"i really love chanterelles — they're the best thing i find."*
twice (so the user-stance confidence clears the 0.35 floor), then asks the agent
for its read. Observed output:

```text
user: i really love chanterelles — they're the best thing i find.
user: i really love chanterelles more than any other mushroom.
user: what do you think about chanterelles?
RAVANA: i'm strongly for chanterelles.          # FORM: derived from user's +polarity, recorded
user: what's your take on chanterelles?
RAVANA: i'm strongly for chanterelles.          # RECALL: same stored stance, stable
```

The engine's own store after the first ask:

```text
_own = engine._agent_stances
_own['chanterelles'] -> Stance(topic='chanterelles', polarity=0.665,
                               confidence=0.52, valence=1.0, ...)
# polarity 0.665 = user's polarity * 0.7 (agent leans, not copies)
# confidence 0.52 = user confidence * 0.8, clamped to [0.35, 0.85]
```

And on a topic with **no evidence at all**:

```text
user: what do you think about blefuscu?
RAVANA: i'm still figuring that out. i don't have a settled view on that yet — what do you think?
# no entry written to _agent_stances['blefuscu'] — honest silence, no fabrication
```

This is exactly what `test_F1` (forms from user stance + records it), `test_F2`
(recalled stably on a second ask), and `test_F3` (honest on an unseen topic)
assert.

## How it grew from the conversation (source citations)

### The store — `ravana/src/ravana/chat/engine.py`

A new durable store holds RAVANA's **own** derived stances. It is declared next to
the other self-model stores and seeded empty — a *store*, not an `if/elif` table,
so every derived stance is written here and RAVANA can expand it at runtime
(online/incremental):

```python
# ravana/src/ravana/chat/engine.py  (L931)
# Agent Self-Stance Formation & Recall (feature, round 2026-08-11T1328Z).
# RAVANA's OWN learned stances — distinct from the constitutive seed in
# `_agent_values` (which is READ-ONLY by design). ... SEED (a store, not an
# if/elif), RAVANA-EXPANDABLE at runtime (every derived stance is written
# here), and learning stays ONLINE/incremental. No hardcoding, no LLM.
# topic.lower() -> Stance (same Stance type as the user's opinion store, so the
# agent's and user's value judgments live in one shape).
self._agent_stances: Dict[str, Any] = {}
```

### Primary path — `engine.py`, the self-stance resolver

The primary *"what do you think about X"* resolver now consults RAVANA's own store
**first** (`engine.py` L2663 comment header; canonical-key derivation L2692-2702;
authoritative store binding L2705-2708):

```python
# ravana/src/ravana/chat/engine.py  (L2709-2722 — RECALL branch)
_own_stance = _own_store.get(_own_key) if _own_key else None
if _own_stance is not None and getattr(_own_stance, "confidence", 0.0) >= 0.35:
    # RECALL: answer from the agent's own durable stance.
    _pol = _own_stance.polarity
    if   _pol >= 0.6:  _w = "strongly for"
    elif _pol > 0.1:   _w = "for"
    elif _pol <= -0.6: _w = "strongly against"
    elif _pol < -0.1:  _w = "against"
    else:              _w = "uncertain about"
    return f"i'm {_w} {_own_key}."
```

When no own stance is recalled yet, it **forms** one from your real learned stance,
attenuated, and records it so the next ask hits the recall branch (L2724-2750):

```python
# ravana/src/ravana/chat/engine.py  (L2724-2750 — FORM branch, abridged)
if _s is not None and getattr(_s, "confidence", 0.0) >= 0.35 and _own_key:
    _conf = max(0.35, min(0.85, float(_s.confidence) * 0.8))
    _pol  = float(_s.polarity) * 0.7          # agent leans, not copies
    from ravana.chat.personal_fact_store import Stance
    _own_store[_own_key] = Stance(            # <-- recorded as the agent's OWN stance
        topic=_own_key, polarity=_pol, confidence=_conf,
        valence=getattr(_s, "valence", 0.0),
        arousal=getattr(_s, "arousal", 0.0),
        turn_number=getattr(self, "turn_count", 0) or 0,
        rehearsal_count=1)
    ...
    return f"i'm {_w} {_own_key}."
```

The legacy self/other-leaking fallback (answering from your stance as if it were
the agent's) is demoted to a last resort that fires only when the agent has *no*
own stance to recall/form — so the self/other boundary is preserved (L2752+).

### Secondary path — `engine_self_query.py`, `_route_self_query`

The deepest self-opinion path (the one that previously returned the hollow frame)
now does the same recall→form→honest-silence ladder. Recall (L413-424) and
formation (L425-469) both read live store state; the honest fallback survives only
for genuine zero-evidence cases (L481-482):

```python
# ravana/src/ravana/chat/engine_self_query.py  (L413-424 — RECALL)
_own = getattr(self, "_agent_stances", None) or {}
_own_key = self._agent_stance_key(target)
_recalled = _own.get(_own_key)
if _recalled is not None and getattr(_recalled, "confidence", 0.0) >= 0.35:
    _word = self._agent_stance_word(_recalled.polarity, _recalled.confidence)
    stance = f"i {_word} {target}"
    result = (stance, "you've shared how you feel about {target}, and that "
                      "shaped where i land")
    return result
```

### Two helpers keep replies honest and grounded

```python
# ravana/src/ravana/chat/engine_self_query.py  (L484-505)
def _agent_stance_word(self, pol, conf) -> str:
    # single short LEXICON tokens, never a sentence — the caller wraps
    # REAL cognitive state in a thin connective.
    if   pol >= 0.6:  return "strongly value"
    if   pol >= 0.3:  return "lean toward"
    if   pol > 0.05:  return "am drawn to"
    if   pol <= -0.6: return "am against"
    if   pol <= -0.3: return "am wary of"
    if   pol < -0.05: return "am cool on"
    return "feel neutral about"

# ravana/src/ravana/chat/engine_self_query.py  (L507+)
def _agent_stance_key(self, target: str) -> str:
    # canonical key + JUNK guard so a non-topic ("right"/"it"/"that") can
    # never become a stored stance — the confabulation class the resolver rejects.
    _t = (target or "").strip().lower()
    _JUNK = {"all","really","it","that","things","right","way","matter",...}
    if not _t or _t in _JUNK: return ""
    return _t
```

### Persistence — save/load

The store is serialized alongside the other self-model stores, with guarded
rehydration (a bad shape must not wipe the store or break boot; junk keys rejected
so a corrupted save cannot replay a hollow stance):

```python
# ravana/src/ravana/chat/engine.py  (L6425-6444 — _serialize_agent_stances)
# topic -> (topic, polarity, confidence, valence, arousal, turn_number,
#           rehearsal_count) — same tuple shape UserStanceStore.get_state uses.

# ravana/src/ravana/chat/engine.py  (L6688 — in save())
'agent_stances': self._serialize_agent_stances(),

# ravana/src/ravana/chat/engine.py  (L6974-7000 — in load())
_as = state.get('agent_stances', {})
...  # each entry rehydrated via _agent_stance_from_tuple; _JUNK keys skipped
self._agent_stances = _restored
```

## Why this is not hardcoding

- **Stances are stored, not match→reply.** A new topic you discuss is captured with
  zero code change — the agent derives its lean from your polarity on the fly.
- **Every reply reads real state.** The only free text is the connective
  `i'm {_w} {topic}` / `i {_word} {target}`; `_w`/`_word` are short *lexicon tokens*
  selected by computed polarity — not authored sentences. A different topic still
  yields an answer whose *content* comes from the polarity/confidence the engine
  computed, only the token varies (passes the seed-vs-hardcoding "if the topic
  changed, does the content still come from state?" test).
- **The store is seed structure RAVANA grows at runtime.** It starts empty; every
  derived stance is written into it through normal conversation. Removing nothing
  breaks; adding a topic needs no code.
- **Grounding is read live, every call** from `UserStanceStore` — no retraining, no
  authored answers, no GloVe-transitivity confabulation (that path was the source
  of the old junk stances and was deliberately removed).

## Tests

`tests/test_dehardcode_plan.py` (feature `t_e7371e2c`, commit `ba6c912`), 4 tests,
all passing (`4 passed in 4.87s`, live-run verified 2026-08-12):

- `test_F1_agent_self_stance_forms_from_user_stance` — after the user states a
  strong, repeated view on `chanterelles`, *"what do you think about chanterelles?"*
  renders a grounded lean (not the hollow frame) **and** writes
  `_agent_stances['chanterelles']` with positive polarity. Fails on the
  pre-capability code.
- `test_F2_agent_self_stance_recalled_stably` — a second ask (*"what's your take on
  open hardware?"*) hits the recall branch and still renders a grounded stance with
  confidence preserved. Fails on the pre-capability code.
- `test_F3_agent_honest_when_no_evidence` — on an unseen topic (`blefuscu`) the
  agent stays silent and writes **no** stance. Fails if the agent fabricates.
- `test_F4_agent_stance_key_rejects_junk_topics` — `_agent_stance_key` returns the
  empty key for deictic/junk tokens (`right`/`it`/`that`/`really`/`all`/`matter`/
  `topic`/`question`/`ok`/`okay`/…) so a hollow cue can never become a stored
  agent stance, and canonicalizes a real topic (`Chanterelles`→`chanterelles`,
  `open Hardware`→`open hardware`). Added by the docs pass (this card) to cover the
  documented junk-key limit — previously untested.

## Limits

- A derived stance is **attenuated** (`polarity × 0.7`, `confidence × 0.8`, clamped
  to `[0.35, 0.85]`) — the agent leans with you, it does not copy your conviction.
  A genuinely neutral user stance (`|polarity| < 0.05`) leaves the agent honestly
  undecided.
- The confidence floor `0.35` gates both recall and formation, so a weakly-held or
  single-stated user view does not produce a brittle agent stance.
- Junk-topic keys (`"right"`/`"it"`/`"that"`/`"really"`/`"all"`/`"matter"`/… — the
  full `_JUNK` set in `engine_self_query.py:_agent_stance_key`, which notably does
  **not** include open-class words like `"source"`) are rejected at the key-derivation
  and load-rehydration boundaries — the old confabulation class cannot become a
  stored stance.
- Distinct from the constitutive `_agent_values` seed (read-only by design): this
  store is what RAVANA *learns* about topics through conversation, and it is what
  makes the agent's view attributable to itself rather than borrowed from you.
