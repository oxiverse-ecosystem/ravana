# Capability: multi-word entity extraction for bereavement disclosures

**Status:** shipped (commit `c3276936`, branch `auto/round-20260922T0209Z`). NOT pushed.
**Feature card:** `t_0866216c` (round `2026-09-22T0209Z`).
**Verified:** `tests/unit/test_loss_multiword_entity.py` passes (6/6). A live
in-process probe on this branch reproduced every example below (real engine
output, `dim=64, seed=42, baby_mode=True`, offline). Hardcoding self-audit
clean — the only change is seed vocabulary (the filler set); no authored reply
strings.

## What it does

When a user discloses a bereavement with a **multi-word self-possessive entity**
(*"i think i might be losing my sense of self"*), RAVANA now extracts the
**real head noun** (`"self"`) and names it in its empathy reply:

```
Q:  "i think i might be losing my sense of self"
A:  "i'm so sorry about your self. that's a real loss, and it hurts."
```

Before the fix, the entity capture regex matched at most 2 words, so
`"sense of self"` was captured as `"sense of"` and the filler stripper picked
`"of"` as the head → the reply read *"i'm so sorry about your of."*

The fix handles both word orders and trailing modifiers:

```
"i lost my grandmother last spring"      -> "grandmother" (trailing temporal strips)
"i lost my best friend in the world"     -> "friend" (4-word entity, trailing preposition strips)
"my dear old dog died"                   -> "dog" (leading fillers strip)
"my father died yesterday"               -> "father" (trailing temporal strips)
"the wind dies down at dusk"              -> NOT empathy (third-entity guard holds)
```

## How the extraction works (no synonym table, no LLM, no retrain)

The bereavement detector lives in `_appraised_affective_reply`
(`ravana/src/ravana/chat/response_gen.py`). Two cooperating changes:

1. **Wider entity capture.** The self-possessive loss regex
   (`response_gen.py:4431-4435`) was `\w+(?:\s+\w+)?` (1–2 words). Expanded to
   `\w+(?:\s+\w+){0,3}` (1–4 words) so multi-word entities like
   `"grandmother last spring"` and `"best friend in the world"` are captured
   whole. The regex covers both word orders — noun-first (`my <noun> died`)
   and verb-first (`losing my <noun>`) — via one alternation.

2. **Extended filler stripper.** The `_FILLER` set
   (`response_gen.py:4477-4496`) previously held only possessive/temporal
   adjectives (`dear`, `old`, `late`, `last`, …). Extended with:
   - **Prepositions** (`of`, `in`, `on`, `at`, `from`, `with`, `by`, `for`,
     `to`, `about`, `like`, `through`, `over`, `into`, `after`, `before`,
     `between`) — strips trailing prepositional modifiers.
   - **Determiners** (`the`, `a`, `an`) — strips trailing articles.
   - **Temporal words** (`yesterday`, `today`, `tomorrow`, `ago`, `later`,
     `soon`, `now`, `spring`, `summer`, `fall`, `autumn`, `winter`,
     `morning`, `afternoon`, `evening`, `night`, `year`, `month`, `week`,
     `day`) — strips trailing temporal modifiers.

   The stripper runs two passes — leading then trailing — each dropping one
   token while the list has >1 element, so `"grandmother last spring"` →
   `"grandmother"` and `"sense of self"` → `"self"`.

The entity is then encoded as `"loss:<entity>"` and routed to the empathic
responder, which names it: *"i'm so sorry about your <entity>."*

## Fail-closed (no false positives)

The third-entity guard (`response_gen.py:4436-4448`) still holds: a bare
loss-term inside a non-self-possessive utterance (`"the wind dies down at
dusk"`) does NOT fire bereavement empathy. The loss word must belong to a
self-possessive structure (`"my <noun> <loss-term>"` or `"<loss-term> my
<noun>"`). The narrative-frame guard (`response_gen.py:4399-4404`) also
still abstains on creative/request frames (`"tell me a story about someone
who died"`).

## How it grew from the conversation

The chat round of this cycle (round `2026-09-22T0209Z`) surfaced the bug at
turn 27: *"i think I might be losing my sense of self"* produced
*"i'm so sorry about your of."* Investigation showed the entity capture
regex only matched 2 words, and the filler set lacked prepositions — so
`"of"` survived stripping and became the named entity.

### Root cause — 2-word capture limit + narrow filler set (commit `c3276936`)

In `_appraised_affective_reply` (`response_gen.py`):

```python
# BEFORE (2 words max):
r"(my|our)\s+(\w+(?:\s+\w+)?)\s+" + _LOSS_VERB + r"|"
r"\b" + _LOSS_VERB + r"\s+(my|our)\s+(\w+(?:\s+\w+)?)"
```

For `"losing my sense of self"` the verb-first branch matched
`"sense of"` (2 words), the filler stripper removed `"sense"` (not in
`_FILLER`), and `"of"` (also not in `_FILLER`) survived as the head.

**Fix** (`response_gen.py:4433-4434`):

```python
# AFTER (4 words max):
r"(my|our)\s+(\w+(?:\s+\w+){0,3})\s+" + _LOSS_VERB + r"|"
r"\b" + _LOSS_VERB + r"\s+(my|our)\s+(\w+(?:\s+\w+){0,3})"
```

Now `"sense of self"` captures whole, and the extended `_FILLER` set
strips `"of"` (preposition) → `"self"`.

The same `_FILLER` set is a small seed vocabulary (not a per-entity table);
removing one entry only loses that one shape. The prepositions/determiners/
temporal words added are structural vocabulary, not reply text.

## Hardcoding audit (summary)

Every change this round is seed vocabulary or a structural regex — **no
authored reply prose, no `random.choice` reply pools, no keyword→response
tables, no Q→A dict**:

- `\w+(?:\s+\w+){0,3}` (`response_gen.py:4433-4434`) — structural regex
  quantifier change, not reply text.
- `_FILLER` extension (`response_gen.py:4477-4496`) — seed vocabulary
  (prepositions, determiners, temporal words); the stripper is a general
  algorithm, not a per-entity table.

**Seed-vs-hardcoding:** the filler set is a small, flat token list the
stripper consumes algorithmically. Removing one entry only loses that one
strip shape; the stripper itself is a general head-noun extractor. The
prepositions/determiners/temporal words are closed-class grammar, not
per-entity content. PASS.

**No retraining:** all changes are online/incremental — the regex and
filler set are consulted at runtime per turn.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| Self-possessive loss regex (both word orders) | `ravana/src/ravana/chat/response_gen.py:4431-4435` |
| Loss-term alternation `_LOSS_VERB` | `ravana/src/ravana/chat/response_gen.py:4427-4428` |
| Filler set `_FILLER` (extended) | `ravana/src/ravana/chat/response_gen.py:4477-4496` |
| Leading/trailing stripper | `ravana/src/ravana/chat/response_gen.py:4497-4501` |
| Entity encoding `"loss:<entity>"` | `ravana/src/ravana/chat/response_gen.py:4503` |
| Third-entity guard | `ravana/src/ravana/chat/response_gen.py:4436-4448` |
| Narrative-frame guard | `ravana/src/ravana/chat/response_gen.py:4399-4404` |

## Test coverage

`tests/unit/test_loss_multiword_entity.py` (6 tests, all pass):

- `test_multi_word_entity_loss` — covers 5 bereavement disclosures (verb-first
  3-word, verb-first with trailing temporal, verb-first 4-word, noun-first
  with leading fillers, noun-first with trailing temporal) plus the
  third-entity guard (`"the wind dies down at dusk"` correctly NOT empathy).

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_loss_multiword_entity.py -v
```

The broader loss/empathy suite stayed green at the round (the parent feature
card reports `test_loss_multiword_entity` 6 passed, `test_loss_verb_first`
5 passed, `test_empathy` 12 passed, `test_fact_empathy_collision` all
passed).
