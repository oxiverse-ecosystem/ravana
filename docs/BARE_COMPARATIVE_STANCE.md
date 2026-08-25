# Bare Comparative Stance Mining — without a leading opinion frame

RAVANA now mines a durable **stance on the winner (X)** of a bare value-comparative
— *"teaching kids to cook is more important than coding"* — even when the user does
**not** front it with an opinion verb ("i think/believe/feel/..."). The comparative
slot itself ("more important than") is the attitude signal, so the leading frame is
made **optional** in the existing copula opinion shapes. A bare *downward* comparative
("X is less worthwhile than Y") is captured as the negative mirror. A universal
collision guard prevents the optional frame from seeding a second garbled topic.

This capability was added in round **2026-08-13T1136Z** (feature `t_2be50e09`,
commit `e5db1c8`). All claims below are backed by the live unit suite
`tests/unit/test_round_2026_08_13T1136_fact_mining.py` (9 tests, `9 passed in
71.53s`, live-run verified 2026-08-13). No LLM, no retraining, no authored reply
pool — the miner is a **structural generalizer** over comparative value-adjectives.

This builds on the base **Agent Self-Stance** capability (see
[AGENT_SELF_STANCE.md](AGENT_SELF_STANCE.md)) and on the **value-verb comparative**
mining added earlier in the same round (the `test_C_*` cases in the round file): a
comparative value judgment now lands whether it is framed ("i think X ..."), led by a
value-verb ("X means more than Y"), or a **bare copula** ("X is more <ADJ> than Y").

## The gap it fills

Round 2026-08-13T1136Z left the *bare* comparative un-mined. The copula opinion
shapes `(h)` only matched when the user opened with an opinion verb:

```python
# ravana/src/ravana/chat/user_model.py  (PRE-FIX, (h) shape)
(r"\bi\s+(?:think|believe|feel|find|reckon)\s+(.+?)\s+(?:is|are)\s+"
 r"(?:much\s+|far\s+|way\s+|more\s+|so\s+)?(?:important|valuable|...)\b", 0.75, 0.5),
```

So:

- *"i believe teaching kids to cook is more important than coding"* -> matched,
  stance on `teaching kids` (+0.75).
- *"teaching kids to cook is more important than coding"* (no leading frame) ->
  **no match** -> no stance -> a later *"what do you think about teaching kids?"*
  had nothing to cite and fell to the hollow *"i'm still figuring that out"*.

The reason the bare form was originally left un-mined was a **mid-string collision**:
making the `i (think|believe|...)` frame merely *optional* as a separate bare pattern
let one framed sentence ("i believe silence is more important than noise") match
**twice** — once framed (topic `silence`) and once mid-string from the bare `i`
(topic `believe silence`) — seeding a garbled key that merged with the correct one.

## What it does (verified)

Live suite (`RAVANA_OFFLINE=1`, `dim=64`) drives the full
`CognitiveChatEngine.process_turn` pipeline and asserts the stance store. Observed
behaviour (test assertions, live-run verified 2026-08-13):

- *"teaching kids to cook is more important than coding"* -> a durable stance on
  `teaching kids` with `polarity > 0`. The bare (no-frame) sentence now produces a
  stance at all (was dropped before).
- *"tidal energy is less worthwhile than protecting the reef"* -> a durable stance
  on `tidal energy` with `polarity < 0` (the downward `less` mirror).
- *"i believe silence is more important than noise"* -> exactly **one** stance on
  `silence`; the collision guard drops the garbled `believe silence` key (was the
  bug that forced bare comparatives to be un-mined).
- The framed form still works (no regression on the earlier fix): *"i believe
  solitude is more honest than company"* -> stance on `solitude`.

A flat (non-comparative) value statement is unchanged, and an ungrounded topic stays
honestly silent — the capability only adds the comparative path; the resolver that
answers self-opinion queries is untouched.

## How it grew from the conversation (source citations)

The fix makes the **leading frame optional INSIDE the single pattern** (not as a
separate bare pattern) so it cannot re-match mid-string, and adds a universal guard
that drops any topic whose first token is a frame verb.

### Optional-frame copula shapes — `user_model.py`

```python
# ravana/src/ravana/chat/user_model.py  (L1966-1971, pattern (h))
# the leading "i (think|believe|feel|find|reckon)" frame is OPTIONAL:
(r"(?:\bi\s+(?:think|believe|feel|find|reckon)\s+)?\b(.+?)\s+(?:is|are)\s+"
 r"(?:much\s+|far\s+|way\s+|more\s+|so\s+)?(?:"
 r"important|valuable|meaningful|useful|helpful|worthwhile|"
 r"significant|relevant|fair|just|honest|beautiful|wonderful|"
 r"vital|crucial|essential|wise|healthy|kind|free|true|real|"
 r"alive|human|warm|clean|brave|strong|necessary|right)\b", 0.75, 0.5),

# ravana/src/ravana/chat/user_model.py  (L1978-1983, pattern (h-neg-q))
# the SAME copula shape but led by "less" — the DOWNWARD mirror (-0.75):
(r"(?:\bi\s+(?:think|believe|feel|find|reckon)\s+)?\b(.+?)\s+(?:is|are)\s+"
 r"(?:much\s+|far\s+|way\s+)?less\s+(?:important|valuable|meaningful|"
 r"useful|helpful|worthwhile|significant|relevant|fair|just|honest|"
 r"beautiful|wonderful|vital|crucial|essential|wise|healthy|kind|"
 r"free|true|real|alive|human|warm|clean|brave|strong|necessary|right)\b",
 -0.75, 0.5),
```

The frame `(?:\bi\s+(?:think|believe|feel|find|reckon)\s+)?` is wrapped in an
optional group, so both *"i believe X is more important than Y"* and *"X is more
important than Y"* resolve to `X` as the content head via the existing
`_opinion_topic` resolver (which cuts at the preposition `to`). Polarity comes from
the lexical value-adjective set — a **seed** vocabulary that RAVANA revises by
talking (no per-topic answers).

### Collision guard — `user_model.py`, the stance-mining loop

```python
# ravana/src/ravana/chat/user_model.py  (L2042-2050, inside the opinion loop)
# Limitation #3 fix (round 2026-08-13T1136Z): the frame is now OPTIONAL, which
# lets a single framed sentence ("i believe silence is more important than
# noise") match twice ... Drop any resolved topic whose FIRST token is a frame
# verb so only the correct topic survives; no real attitude object begins with
# one.
if _topic.split()[0] in self._FRAME_VERBS:
    continue
```

The guard is the universal closure for the optional-frame change. It rejects the
garbled `believe silence` collision key (whose first token `believe` is a frame verb)
while letting the correct `silence` topic (from the framed match) survive.

```python
# ravana/src/ravana/chat/user_model.py  (L2707-2708, the seed set)
_FRAME_VERBS = {"think", "believes", "believe", "felt", "feel",
                "find", "finds", "reckon", "reckons"}
```

This is a **general grammatical guard**, not a per-topic deny-list: no real attitude
object begins with a frame verb, so removing a word degrades gracefully (the topic
simply isn't dropped). It mirrors pattern `(a)` above, which already mines bare
comparatives ("the sea is a better teacher than any classroom") with an optional
frame and no collision.

## Why this is not hardcoding

- **The miner is a structural generalizer.** It keys on the *comparative copula
  shape* + a seed value-adjective lexicon, not on the words `teaching`, `coding`,
  `silence`, or any specific topic. Change the comparative and the resolved winner
  changes accordingly — only the computed topic changes, never the logic.
- **No added reply strings.** `git diff e5db1c8 -- ravana/src/ | grep -oE
  '"[a-z][^"]{45,}"'` returns only the **seed valence lexicon** regex alternations
  (legitimate vocabulary, not replies) and doc-comment example phrases — **zero
  authored reply sentences**. The hardcoding-audit gate (see
  `tests/test_dehardcode_plan.py`) stays green. (Per the skill's rule, a long
  *vocabulary* literal is a seed; a long *reply* literal is the violation — here
  only the former appears.)
- **Polarity is lexical and RAVANA-revisable.** The +0.75 / -0.75 come from the
  pattern's seed values; RAVANA updates any stance by talking — no per-topic answer
  table, no LLM, no retraining.
- **Fail-open, not fabricate.** When the comparative topic was never taught, the
  agent answers honestly (the existing self-opinion resolver returns a non-empty,
  grammatical reply with no invented polarity). The capability only adds the
  *mining* path; the *query* resolver that answers self-opinion questions is
  untouched.
- **Seed-based, runtime-expandable.** The only "knowledge" is the value-adjective
  lexicon (a legitimate seed — already runtime-expandable via the opinion store).
  No Q→A dict, no authored prose.

## Tests

`tests/unit/test_round_2026_08_13T1136_fact_mining.py` (feature `t_2be50e09`,
commit `e5db1c8`), 9 tests total, all passing (`9 passed in 71.53s`, live-run
verified 2026-08-13). The limitation #3 capability is covered by **3** `test_D_*`
regression tests, driven through the **full `CognitiveChatEngine.process_turn`**
path so a routing/collision regression is caught (not just the miner in isolation):

- `test_D_bare_comparative_stance_without_leading_frame` — *"teaching kids to cook
  is more important than coding"* must produce a durable stance on `teaching kids`
  with `polarity > 0` (the bare form was dropped before the fix). Also asserts the
  framed form *"i believe solitude is more honest than company"* still lands on
  `solitude` (no regression).
- `test_D_bare_comparative_negative_without_leading_frame` — *"tidal energy is less
  worthwhile than protecting the reef"* must produce a durable stance on `tidal
  energy` with `polarity < 0` (the downward `less` mirror).
- `test_D_bare_comparative_no_midstring_collision` — *"i believe silence is more
  important than noise"* must land exactly one stance on `silence` and must NOT seed
  a second garbled key like `believe silence` (the collision the optional-frame
  change would otherwise introduce).

The broader **opinion/stance** unit scope was re-run live as a regression check and
came back **21 passed** (no regression). No new test was required by this docs pass
— the documented behaviour (bare positive comparative, bare negative `less`
comparative, and the frame-verb collision guard) is already covered by the three
`test_D_*` cases above.

## Limits

- The capability applies to the **copula comparative** shapes (`X is [more/less]
  <VAL-ADJ> than Y`). The sibling **value-verb** comparative ("X means more than
  Y" / "X matters less than Y") is a separate, earlier fix in the same round (the
  `test_C_*` cases) and still requires the leading frame verb — it was never subject
  to the collision and was left framed by design.
- The value-adjective lexicon is a **seed**; a comparative on an adjective outside
  the set (e.g. *"X is more <novel-word> than Y"*) is not captured until RAVANA
  expands the store from conversation or web-learning (online, no retraining).
- The collision guard is keyed on the first token of the resolved topic being a
  frame verb. A genuine attitude object that begins with one of those verbs is
  astronomically unlikely by grammar; if it ever arose, the topic would be dropped
  rather than garbled — fail-closed, never fabricated.
- The resolved winner `X` is the content head per `_opinion_topic` (it cuts at the
  preposition `to`), so *"teaching kids to cook"* is the key, not the full sentence.
