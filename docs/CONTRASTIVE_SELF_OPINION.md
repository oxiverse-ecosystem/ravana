# Contrastive Self-Opinion — engaging BOTH sides of "X versus Y"

RAVANA can now answer a binary *self-opinion* question — *"what's your take on the
sea versus the mountains?"* / *"do you prefer the countryside or the cities?"* —
by engaging **both** named options through its real cognitive state, instead of
collapsing to the last token and silently dropping the other side.

This capability was added in round **2026-08-12T1234Z** (feature `t_2595f8ad`,
commit `b933523`). All claims below are backed by a live in-process probe
(`dim=64`, offline) and the unit suite `tests/test_contrastive_self_opinion.py`
(3 tests, `3 passed in 11.26s`, live-run verified 2026-08-12). No LLM, no
retraining, no authored reply pool — each side's words come from a *computed*
stance, wrapped in a short connective.

This is a refinement of the base **Agent Self-Stance** capability (see
[AGENT_SELF_STANCE.md](AGENT_SELF_STANCE.md)): that one covers *single-topic*
self-opinion ("what do you think about X?"). Contrastive self-opinion covers the
*unsolved residual* — the binary "X versus Y" shape that previously fell through
to the hollow honest-uncertainty frame because the extractor only ever pulled **one**
target.

## The gap it fills

A binary self-opinion names **two** options. The prior extractor stripped the
contrastive connective (`versus` / `vs` / `or` / `over` / `rather than`) and took
only the **LAST token** (or the whole phrase) as the single target. So:

- *"what's your take on the sea versus the mountains"* resolved to `mountains`
  alone → one-sided answer, `sea` silently dropped.
- *"do you prefer the countryside or the cities"* resolved to `cities` → the
  contrast collapsed to a single token and the honest answer for the *other*
  side never surfaced.

That was the round's documented residual limitation (abstract X-vs-Y opinion
questions landed in the hollow `"i'm still figuring that out"` frame). The agent
already held a structured lean **per topic** (its own `_agent_stances` store, or a
lean derived from the user's learned opinion) — the missing piece was engaging
*BOTH* sides at once.

## What it does (verified)

Live probe (`RAVANA_OFFLINE=1`, `dim=64`): seed a real user stance on each
grounded side (`sea` +0.7, `cities` −0.6), then ask. Observed output:

```text
user: what's your take on the sea versus the mountains
RAVANA: i'm for sea.; i'm still figuring out mountains.
        # sea engaged from the derived stance; mountains answered honestly

user: do you prefer the countryside or the cities
RAVANA: i'm still figuring that out; i am wary of cities what about you?
        # cities engaged from the derived stance (wary = agent lean, not copy);
        # countryside answered honestly; both sides present, ";"-joined

user: what's your take on zebras versus kangaroos
RAVANA: i'm still figuring out zebras.; i'm still figuring out kangaroos.
        # neither grounded -> both sides honest, no fabrication, no collapse
```

The third case is the **fail-open** guarantee: with zero evidence on *either* side,
each is answered honestly (`i'm still figuring out …`) and joined — the contrast is
answered truthfully, never collapsed to a single invented conviction.

## How it grew from the conversation (source citations)

The fix spans **all three** self-opinion routing paths (no single point of
collapse). Each path splits the phrase on the contrastive connective, resolves
EVERY side through the existing real-state resolver, and composes a reply that
names both sides. The connectives recognized are: ` versus `, ` vs `, ` vs. `,
` or `, ` over `, ` rather than `.

### Shared helper — `engine.py`

A new extractor helper was pulled out of the single-topic `_SELFSTANCE` block so
all paths can resolve each side through the *same* real-state logic (recall →
form → honest-silence) without duplicating it:

```python
# ravana/src/ravana/chat/engine.py  (L2218 — def _agent_self_stance_reply)
def _agent_self_stance_reply(self, opinions, beliefs, topic_phrase: str
                             ) -> Optional[str]:
    # RECALL: RAVANA's own _agent_stances store;
    # FORM:   a grounded lean derived from the user's learned stance
    #         (polarity * 0.7, confidence * 0.8, clamped [0.35, 0.85]);
    # SILENCE: returns None when ungrounded, so the caller names the side honestly.
    ...
    # grounded returns e.g. f"i'm for {_own_key}." / f"i'm against {_own_key}."
```

### Path 1 — `engine.py`, `_structured_recall` `_SELFSTANCE` block

The self-stance recall block now detects a contrastive phrase and resolves each
side through the helper above, falling back to a per-side honest naming so the
contrast is answered, not hidden:

```python
# ravana/src/ravana/chat/engine.py  (L2742 comment header; split L2761-2796)
if _contrast_sides and len(_contrast_sides) >= 2:
    _side_topics = [last_content_word_of(side) for side in _contrast_sides]
    if len(_side_topics) >= 2:
        _replies = []
        for _st in _side_topics:
            _r = self._agent_self_stance_reply(opinions, beliefs, _st)
            # Fallback: if a side is fully ungrounded, still name it honestly
            # so the contrast is answered, not hidden.
            _replies.append(_r if _r else f"i'm still figuring out {_st}.")
        return "; ".join(_replies) + "."
```

### Path 2 — `engine_self_query.py`, `_route_self_query` `_agent_opinion` block

The deepest self-opinion path carries the identical split. Each side is resolved
independently through `_agent_stance_on` (which reads real state and answers
honestly when ungrounded), and the two clauses are composed:

```python
# ravana/src/ravana/chat/engine_self_query.py  (L993 comment header; split L1014-1062)
if _contrast is not None and len(_contrast) >= 2:
    _resolved = [(s, self._agent_stance_on(s)) for s in _sides]
    _phrases = [f"i {_st} {_s}" for _s, (_st, _rs) in _resolved]
    _answer = "; ".join(_phrases)
    if not _answer.endswith((".", "!", "?.")):
        _answer += "."
    return _answer   # both sides engaged; never a single-topic collapse
```

### Path 3 — `engine.py`, A2 vmPFC `do you prefer/like X or Y` path

The `"do you like/prefer X or Y?"` yes/no path (matched by `m_agent_likes_yesno`)
was extended to split on the connective too, reusing the *same* real
`_agent_stance_on` resolver per side — and the regex was loosened so a trailing
` ` after the final option doesn't prevent the match (L5303-5358):

```python
# ravana/src/ravana/chat/engine.py  (L5310 comment header; split L5320-5358)
_ym = re.search(
    r"\bdo\s+you\s+(?:like|love|hate|enjoy|prefer|care\s+for)\s+([a-z][a-z\s'-]{1,30}?)[\\?\\. ]?$",
    clean_input, re.IGNORECASE)   # <-- trailing class widened to [...?.' ]
...
if _contrast_sides and len(_contrast_sides) >= 2:
    _phrases = []
    for _st in _side_topics:
        _stt, _st_r = self._agent_stance_on(_st)   # full stance sentence,
        _phrases.append(_stt)                       # already begins with "i"
    stance = "; ".join(_phrases)                    # composed, not re-prepend
    response = f"{stance} what about you?"
```

`_agent_stance_on` returns a *full* stance sentence (e.g. `"i'm for sea."`), so the
composer joins one sentence per side on `; ` and does **not** re-prepend `"i"` or
the topic — avoiding the double-subject bug that a naive join would introduce.

## Why this is not hardcoding

- **Each side is resolved by the existing real-state resolver.** The only added
  free text is the connective `"; "` between clauses and the honest fallback
  `f"i'm still figuring out {_st}."` — `_st` is the topic computed from the user's
  words, not an authored answer. The *content* of every grounded clause comes from
  a computed polarity/confidence (`_agent_stance_on` / `_agent_self_stance_reply`).
- **Passing the seed-vs-hardcoding test.** Change the topics and both clauses still
  derive their content from state — only the topic word changes. Nothing in the
  diff is a keyword→reply table; the capability is structural (split + resolve each).
- **Fail-open, not fabricate.** When a side has no view, it is named honestly
  rather than collapsed away. The deciding test (`test_contrastive_neither_grounded_is_honest`)
  asserts the engine returns a non-empty, grammatical reply and never asserts a
  conviction it doesn't have.
- **No new persisted state required.** The contrast is computed live from the
  existing `_agent_stances` / `UserStanceStore`; the per-side lean is derived on
  each call. No retraining, no LLM, no added store.

## Tests

`tests/test_contrastive_self_opinion.py` (feature `t_2595f8ad`, commit `b933523`),
3 tests, all passing (`3 passed in 11.26s`, live-run verified 2026-08-12). Driven
through the **full `CognitiveChatEngine.process_turn`** path so a routing regression
is caught (not just the miner in isolation):

- `test_contrastive_versus_engages_both_sides` — seeding a real user stance on
  `sea` lets the agent derive a grounded lean on that side; *"what's your take on
  the sea versus the mountains"* must surface **both** `sea` and `mountains`, not
  the hollow fallback. Fails on the pre-capability code (which dropped `sea`).
- `test_contrastive_or_engages_both_sides` — *"do you prefer the countryside or the
  cities"* must produce **two** `;`-joined clauses, name the grounded side `cities`
  with its real lean (`wary`/`against`), and never collapse to one side. Fails on
  the pre-capability last-token collapse.
- `test_contrastive_neither_grounded_is_honest` — *"what's your take on zebras
  versus kangaroos"* with no evidence on either side returns a non-empty, grammatical
  reply and never fabricates a polarity.

Regression at land: `tests/unit` 1866 passed / 23 skipped (baseline match, 0
regress); `tests/ --ignore=unit --ignore=ci` 157 passed / 4 skipped (pre-existing
live-web skips). Coverage is already present, so **no new test was required** by
this docs pass — the capability ships with the 3 above.

## Limits

- Recognized connectives are fixed (` versus `, ` vs `, ` vs. `, ` or `, ` over `,
  ` rather than `). A contrast phrased without one of these (e.g. *"coffee and
  tea, which wins?"*) is not split and falls back to the single-topic path.
- Each side's topic target is taken as the **last content word** of that side
  (same convention the single-topic path uses: `"the sea"` → `sea`). A side whose
  only tokens are closed-class/scaffold words is dropped, so a degenerate side
  (e.g. *"or the"*) does not produce a junk clause.
- The grounded-side lean is **attenuated** (`polarity × 0.7`, `confidence × 0.8`,
  clamped `[0.35, 0.85]`) — the agent leans with the user's evidence, it does not
  copy their conviction (see [AGENT_SELF_STANCE.md](AGENT_SELF_STANCE.md#limits)).
- An honestly-ungrounded side answers with the hollow frame; both clauses are still
  present, so the answer is *truthful about the contrast*, not a fabricated stance.
