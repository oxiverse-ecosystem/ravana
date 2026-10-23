# CAPABILITY: Autobiographical recall bridges an entity head to a stored composite attribute (colour recall fix)

**Status:** shipped (commit `f6d4ee3b`, branch `auto/round-2026-10-23T0254Z`). NOT pushed.
**Feature card:** `t_39b424ee` (round `2026-10-23T0254Z`).
**Docs card:** `t_a32083d3`.
**Verified:** `tests/test_round_2026_08_18_autobio_recall.py::test_colour_recall_when_topic_precedes_tell_verb`
passes (`1 passed in 48.32s`, offline, `.venv-real` / Python 3.11). A real in-process probe on this
branch reproduced every example below (real engine output, `dim=64, seed=42, baby_mode=True`,
offline). Hardcoding self-audit clean (no authored reply prose, no per-topic answer table — the
candidate loop reads real store values via `_match_fact`; the only new literal is a seed stop-word
vocabulary of question/tell scaffolds).

## What it does

RAVANA can now recall a **composite attribute** (e.g. `('i','favourite colour','indigo')`) when the
user asks about it with the **entity head in front of the tell-verb** — a phrasing where the
attribute head precedes the verb instead of following it:

> *\"what colour did i say i like\"*  →  *\"yes — i remember: your favourite colour is indigo.\"*

Before this fix the same query returned `None` / an honest miss: the post-tell-clause residue
collapsed to a generic word (`\"like\"`), `_match_fact` missed the stored `favourite colour`
attribute, and the confirmation branch could not reach it. The entity head (`colour`) lives in the
**full query**, not in the post-tell-clause residue.

Real engine output (fresh engine, seeded with *\"my favourite colour is indigo\"*, then queried):

```
Q: what colour did i say i like
A: "yes — i remember: your favourite colour is indigo."

Q: did i say i like a colour
A: "yes — i remember: your favourite colour is indigo."

Q: my favourite food is ramen  →  (stored: ('i','favourite food','ramen'))
Q: what food did i say i like
A: "yes — i remember: your favourite food is ramen."
```

And an honest miss when nothing maps:

```
Q: did i tell you i liked blue
A: "not that i recall — you haven't told me about that yet. what did you want me to know?"
```

**This is a real capability, not a hardcoded reply.** The answer slot is read from the live
`PersonalFactStore` via the existing `_render_fact_line`. No new reply strings were added; the
bridge is a candidate-expansion loop over content words of the query. The deciding test
(\"can RAVANA change this by itself, through experience?\") passes: the content comes entirely
from the learned store.

## How it bridges (the mechanism)

The confirmation branch of `_autobiographical_recall` (sub-intent (B)) first recovers the topic
from the post-tell-clause residue and runs `_match_fact(_topic)`. When that returns `None` (the
residue collapsed to a generic word like `\"like\"`), the bridge kicks in:

1. Collect every content word of the **full query** `q` (regex `[a-z']+`, length ≥ 3) that is not
   in the seed stop set `_CW_STOP` (question/tell scaffolds: `what, did, say, tell, like, love,
   about, you, …` — structural, not per-topic).
2. Sort candidates by length (longest first) so a multi-token head is tried before a fragment.
3. For each candidate `_cw`, call `_match_fact(_cw)`. The first non-`None` result wins — i.e. the
   stored composite attribute that **contains that head word** (e.g. `favourite colour` contains
   `colour`) is reached.

So `\"what colour did i say i like\"` → candidates `{colour}` (after stop-set filtering) →
`_match_fact(\"colour\")` → matches the stored `('i','favourite colour','indigo')` → rendered as
*\"your favourite colour is indigo.\"*

**Boundary (honest):** the bridge only works when the head word survives into the stored
attribute/value. A fact stored as `('pet','name','wren')` is NOT reached by `\"what pet did i say
i have\"`, because the stored attribute is `name`, which does not contain `pet`. The bridge reads
real store values; it does not invent a link. (Probe: `match_fact('pet')` → `None`.) This is the
correct fail-closed behaviour, not a bug.

## Fail-closed

- The candidate loop only runs when `_match_fact(_topic)` already returned `None` — it never
  overrides a real match from the residue.
- `_CW_STOP` is a seed vocabulary of question/tell scaffolds; it does not list topics, so it
  cannot be tuned to a probe question.
- `_match_fact` itself is fail-closed (returns `None` when no attr/value contains or overlaps
  the candidate), so an unknown head yields an honest *\"not that i recall\"* rather than a
  fabricated answer.
- A genuine agent-self question (no first-person **user** framing) never reaches this branch:
  `_autobiographical_recall` returns `None` earlier and the query falls through to the self-model
  path. The self/other boundary is preserved by construction.

## How it grew from the conversation

The chat round of this cycle (`t_39b424ee`, commit `f6d4ee3b`) surfaced a residual in the
autobiographical recall confirmation branch (sub-intent (B)):

- **Residual** — a fact stored as `('i','favourite colour','indigo')` was **not** recalled by
  *\"what colour did i say i like\"*. The attribute head (`colour`) precedes the tell-verb in
  that phrasing, so the post-tell-clause residue extraction (which assumes the topic follows the
  verb) collapsed to a generic word (`\"like\"`), `_match_fact` missed, and the confirmation
  branch fell through to the honest miss.

Root cause — the resolver assumed the disclosure topic **follows** the tell-verb. When the user
instead leads with the entity head (*\"what colour did i say i like\"*), the residue is not the
topic at all, so the matcher had nothing to grip.

Fix — generalize the confirmation branch to try **each content word of the full query** as a
candidate topic, so a stored composite attribute can be reached from any of its constituent
head words. No new reply strings; the answer is rendered from the live `PersonalFactStore` via
the existing `_render_fact_line`. The stop set is a seed vocabulary of question/tell scaffolds
(structural, not per-topic).

## Hardcoding audit (summary)

Every change this round is store-driven logic or a structural guard — **no authored reply
prose, no `random.choice` reply pools, no keyword→response tables, no Q→A dict**:

- The candidate loop (`engine.py:5485-5491`) calls `_match_fact` (a real matcher over the live
  store) for each candidate — it does not author an answer.
- `_CW_STOP` (`engine.py:5472-5483`) is a seed vocabulary of question/tell scaffolds; it
  contains no topics and cannot be tuned to a probe question. It is structural (a closed-class
  functional-word set), the same shape as the existing `_OBJ_NONCONTENT` / `_PRONOUN_STOP`
  vocabularies.
- The connective string (`\"yes — i remember: \" + _render_fact_line(...) + \".\"`) wraps a real
  store value; the CONTENT is always a stored fact.
- No new reply strings were added (the commit message is explicit: \"No new reply strings\").

**Seed-vs-hardcoding:** the stop set is seed vocabulary (question/tell scaffolds). The deciding
test (\"can RAVANA change this by itself, through experience?\") → the *content* lives in the
stores, not the code; the stop set is a closed-class functional-word list, the same kind of
seed the architecture already ships (and grows) for opinion-topic resolution. PASS.
**No retraining:** all changes are online/incremental; no corpus rebuild, no batch re-fit.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| `_autobiographical_recall` (the capability; sub-intent (B) confirmation branch) | `ravana/src/ravana/chat/engine.py:5328-5591` |
| `_match_fact` (the matcher the bridge calls) | `ravana/src/ravana/chat/engine.py:5272-5314` |
| `_extract_disclosure_topic` (residue extraction that collapses on head-first phrasing) | `ravana/src/ravana/chat/engine.py:5315-5327` |
| `_match_fact(_topic)` first attempt (residue-based) | `ravana/src/ravana/chat/engine.py:5462` |
| Bridge: `_CW_STOP` seed stop set | `ravana/src/ravana/chat/engine.py:5472-5483` |
| Bridge: candidate collection + sorted loop | `ravana/src/ravana/chat/engine.py:5485-5491` |
| Render via `_render_fact_line` (existing) | `ravana/src/ravana/chat/engine.py:5494-5495` |
| Parent dispatcher `_structured_recall` (gate order; (0a) autobio gate) | `ravana/src/ravana/chat/engine.py:2394` / `3106` |
| Agent-own-recall gate it precedes | `ravana/src/ravana/chat/engine.py:4405` |
| Regression test (colour) | `tests/test_round_2026_08_18_autobio_recall.py::test_colour_recall_when_topic_precedes_tell_verb` |

## Test coverage

The feature commit added one regression test, which is the source of truth for this capability:

- `test_colour_recall_when_topic_precedes_tell_verb` — seeds *\"my favourite colour is indigo\"*,
  calls `e._autobiographical_recall(\"what colour did i say i like\")`, and asserts the answer is
  not `None`, contains *\"indigo\"*, and does not contain *\"not that i recall\"*. Passes
  (`1 passed in 48.32s`, offline).

A second test (`test_entity_head_bridge_generalizes_to_other_head`) is added by this docs round
to back the documented generality claim beyond `colour` — it seeds *\"my favourite food is
ramen\"* and asserts *\"what food did i say i like\"* recalls *\"ramen\"*. Same mechanism, same
assertions; it guards against the bridge silently regressing to a colour-only special case.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/test_round_2026_08_18_autobio_recall.py -v
```

## Verification (real executed output)

In-process probe (`dim=64, seed=42, baby_mode=True`, offline), fresh suffix, no stale pickle:

```
STORED: [('i', 'favourite colour', 'indigo')] -> indigo
Q: "what colour did i say i like"  →  "yes — i remember: your favourite colour is indigo."
Q: "did i say i like a colour"     →  "yes — i remember: your favourite colour is indigo."
Q: "did i tell you i liked blue"   →  "not that i recall — you haven't told me about that yet. ..."
Q: "what is my favourite colour"   →  None   (not a confirmation phrasing; honest fall-through)
```

Boundary probe (pet case, to confirm fail-closed):

```
STORED: [('pet', 'name', 'wren')] -> wren
match_fact('pet') → None
autobio("what pet did i say i have") → "not that i recall — you haven't told me about that yet. ..."
```

The bridge reads real store values and fails closed when the head word is absent from the stored
attribute — no fabrication.
PYEOF
echo "written"