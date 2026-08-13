# Self-Opinion on a Relative-Clause Topic — resolving the CONTENT HEAD

RAVANA can now answer a *single-topic* self-opinion question whose topic is a
**relative clause** — *"your honest read on people who talk in theatres?"* /
*"what's your take on friends who keep their promises?"* — by resolving the
**content head** (`people who talk`, `friends who keep`) instead of collapsing
to the trailing last token (`theatres`, `promises`). The head it computes now
**matches the stance key it mined** from the user, so the agent engages the
real lean it learned instead of the hollow *"i'm still figuring that out"*.

This capability was added in round **2026-08-13T0634Z** (feature `t_4297f732`,
commit `cd12e2b`). All claims below are backed by the live unit suite
`tests/test_self_opinion_query_head.py` (4 tests, `4 passed in 13.72s`, live-run
verified 2026-08-13). No LLM, no retraining, no authored reply pool — the
extractor is a **structural generalizer** over the tail that reuses the miner's
relative-clause resolver.

This is a refinement of the base **Agent Self-Stance** capability (see
[AGENT_SELF_STANCE.md](AGENT_SELF_STANCE.md)) and closes the residual **D-B**
limitation left by the **Contrastive Self-Opinion** work (see
[CONTRASTIVE_SELF_OPINION.md](CONTRASTIVE_SELF_OPINION.md)) — the *query* side
of the relative-clause extractor was never hardened the way the *mining* side
was.

## The gap it fills

A single-topic self-opinion question names one topic. When that topic is a
relative clause — *"people who talk in theatres"* — the topic's real concept is
the **noun + its relative descriptor** (`people who talk`), because that is the
attitude object the user actually evaluated. The prior **query-side** extractor
in `engine_self_query.py` reduced the topic to the **last content token** of the
tail:

```python
# ravana/src/ravana/chat/engine_self_query.py  (PRE-FIX, ~L1076)
_target = _toks[-1] if _toks else ""
```

So:

- *"your honest read on people who talk in theatres"* -> tail
  `people who talk in theatres` -> last token `theatres`.
- `_agent_stance_on("theatres")` does **not** match the mined stance key
  `people who talk` -> the agent answers the hollow
  `"i'm still figuring that out"` even though it had learned a strong view from
  the user (Defect B).

This is the **same defect** the D-C fix (commit `1e23670`) hardened on the
**mining** side (`user_model._opinion_topic`'s relative-CLAUSE BRIDGE at
`user_model.py` L2533) — but the query side never got the same treatment, so the
two paths were asymmetric and the mined key never resolved at query time.

## What it does (verified)

Live probe (`RAVANA_OFFLINE=1`, `dim=64`): seed a real user stance on the
relative-clause head (`people who talk` and `friends who keep`), then ask. The
suite `tests/test_self_opinion_query_head.py` drives these through the **full
`CognitiveChatEngine.process_turn`** pipeline and asserts the extractor hands
`_agent_stance_on` the resolved head, not the last token. Observed replies (from
the round end-to-end driver):

```text
user: your view on people who talk in theatres?
RAVANA: i'm against people who talk.
        # head resolved to "people who talk" (matches mined key) -> real lean

user: what's your take on friends who keep their promises?
RAVANA: i'm strongly for friends who keep.
        # head resolved to "friends who keep" -> real lean engaged
```

A flat (non-relative) topic is unchanged — *"your honest read on privacy"* still
resolves to `privacy`. And an **ungrounded** relative clause stays honest, never
fabricated: *"your honest read on people who jog at midnight?"* returns a
non-empty, grammatical reply without inventing a conviction.

## How it grew from the conversation (source citations)

The fix makes the **query-side** tail extractor symmetric with the **mining-side**
`_opinion_topic` resolver: a relative pronoun (`who`/`whom`/`that`/`which`) is a
BRIDGE that keeps its clause content, the head is cut at the first internal
closed-class word, and a trailing closed-class/modifier is trimmed. No new
knowledge path — it reuses the miner's existing `_OPINION_STOP` seed.

### Query-side fix — `engine_self_query.py`, `_route_self_query` `_agent_opinion` block

The broken `_target = _toks[-1]` was replaced with the same head-resolution
discipline the miner uses:

```python
# ravana/src/ravana/chat/engine_self_query.py  (L1076 block header; resolver L1099-1134)
# D-B FIX (round 2026-08-13T0634Z): resolve the tail's CONTENT HEAD,
# not the last token. Mirror _opinion_topic's relative-CLAUSE BRIDGE.
if _toks:
    _REL_BRIDGE = {"who", "whom", "that", "which"}          # L1099
    _OPINION_STOP = getattr(getattr(self, "user_model", None),
                            "_OPINION_STOP", None)           # reuse miner seed
    if _OPINION_STOP is None:
        _OPINION_STOP = { ... }                             # inline fallback copy
    _head = []
    for _t in _toks:
        if _t in _REL_BRIDGE:
            _head.append(_t); continue                      # bridge: keep clause
        if _t in _OPINION_STOP:
            break
        _head.append(_t)
    while len(_head) > 1 and _head[-1] in _OPINION_STOP:
        _head.pop()                                         # trim trailing modifier
    _target = " ".join(_head) if _head else ""              # L1134
else:
    _target = ""
if not _target:
    _target = _toks[-1] if _toks else ""                    # fail-safe
_stance, _reason = self._agent_stance_on(_target)           # L1139 — real grounding
```

### Symmetric miner-side — `user_model.py`, `_opinion_topic`

The query extractor now mirrors the **exact** logic the D-C miner uses to
produce the stance key, so the two paths agree by construction:

```python
# ravana/src/ravana/chat/user_model.py  (L2499 def; _OPINION_STOP seed L2460;
#                                         BRIDGE L2533)
def _opinion_topic(self, phrase):           # L2499
    ...
    _REL_BRIDGE = {"who", "whom", "that", "which"}   # L2533
    head = []
    for t in toks:
        if t in _REL_BRIDGE:
            head.append(t); continue          # bridge: keep the clause
        if t in self._OPINION_STOP:           # L2460 seed set
            break
        head.append(t)
    while len(head) > 1 and head[-1] in self._OPINION_STOP:
        head.pop()                            # trim trailing modifier
    return " ".join(head)
```

Because the query path now reuses the **same** `_OPINION_STOP` seed (`user_model.py`
L2460) and the **same** relative-clause BRIDGE convention, *"people who talk in
theatres"* resolves to `people who talk` on **both** sides — the mined key and
the queried head are identical, so `_agent_stance_on` engages the real lean.

## Why this is not hardcoding

- **The extractor is a structural generalizer.** It does not key on the words
  `theatres`, `promises`, or any specific topic. It resolves the content head of
  *any* tail using closed-class/relative-clause rules. Change the relative clause
  and the resolved head changes accordingly — only the computed topic changes,
  never the logic.
- **No added reply strings.** `git diff cd12e2b -- ravana/src/ | grep -oE '"[a-z][^"]{45,}"'`
  returns **empty** — zero authored reply sentences. The hardcoding-audit gate
  (see `tests/test_dehardcode_plan.py`) stays green.
- **`_agent_stance_on` still does the real grounding.** The only behavioral change
  is *which string* reaches the existing resolver; the resolver itself (which
  reads real state and answers honestly when ungrounded) is untouched.
- **Fail-open, not fabricate.** When the relative-clause topic was never taught,
  the agent answers honestly (a non-empty, grammatical reply) — it never invents a
  polarity. The deciding test (`test_ungrounded_relative_clause_query_is_honest`)
  asserts a non-empty reply and never asserts a conviction it doesn't have.
- **Seed-based, runtime-expandable.** The only "knowledge" reused is the miner's
  `_OPINION_STOP` closed-class set (a legitimate seed — already runtime-expandable
  and shared with the miner). No per-topic table, no LLM, no retraining.

## Tests

`tests/test_self_opinion_query_head.py` (feature `t_4297f732`, commit `cd12e2b`),
4 tests, all passing (`4 passed in 13.72s`, live-run verified 2026-08-13). Driven
through the **full `CognitiveChatEngine.process_turn`** path so a routing
regression is caught (not just the extractor in isolation):

- `test_relative_clause_query_resolves_mined_head` — seeding a real user stance on
  `people who talk in theatres`, *"your honest read on people who talk in
  theatres?"* must hand `_agent_stance_on` the head `people who talk` (NOT the last
  token `theatres`) and engage a real lean, not the hollow fallback. FAILS on the
  pre-fix code (`_agent_stance_on(target='theatres')`).
- `test_your_read_relative_clause_head` — *"your read on friends who keep their
  promises?"* must resolve the head `friends who keep` (NOT `promises`). Fails on
  pre-fix code.
- `test_flat_topic_query_still_resolves` — the fix must not break the simple
  single-noun case (`"your honest read on privacy"` -> `privacy`). Guards against
  over-fitting the relative-clause path.
- `test_ungrounded_relative_clause_query_is_honest` — a relative clause the user
  never taught (`people who jog at midnight`) returns a non-empty, grammatical
  reply with no fabricated polarity.

**No new test was required by this docs pass** — the capability ships with the 4
above. Scoped regression at land: 47 passed (contrastive self-opinion +
self-opinion-query-head + round-08f + dehardcode_plan hardcoding guard) + 2 unit
(self-experience routing), all green, 0 collateral damage.

## Limits

- The fix applies to the `_agent_opinion` query path — the *"your read/take/view/
  stance on X"* / *"what's your read on X"* / *"your honest read on X"*
  scaffoldings. Phrasings like *"give me your read on X"* route through a separate
  "you mentioned" echo branch (not the D-B site) and are out of scope — they did
  not exhibit the last-token collapse.
- The head resolver reuses the miner's **5-word** capture cap for a long
  location-style topic; an exceptionally long relative clause may be trimmed at
  the same bound as the miner (a pre-existing, separate limitation).
- Both resolver paths share the `_OPINION_STOP` seed, so a topic whose only
  content words are closed-class collapses to the fail-safe last token and answers
  honestly — by design, never fabricated.
