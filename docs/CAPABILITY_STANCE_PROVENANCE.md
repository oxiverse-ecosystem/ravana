# Capability: stance provenance + resolver bridge (broader-concept co-mention links to a held stance)

**Status:** shipped (commits `11648f7`, `bd10101`, branch
`auto/round-2026-08-20T0701Z`). **Verified:** the six provenance tests below
pass (4 in `tests/unit/test_personal_fact_store.py`, 2 in
`tests/unit/test_round_2026_08_09T1953_stance_reversal.py`); plus a live
end-to-end in-process probe (real engine output, `dim=64, seed=42,
baby_mode=True`, offline), reproduced below. Hardcoding self-audit clean — no
added reply prose in either commit (grep for quoted `>=45`-char literals on added
lines = 0 hits); the bridge is store-driven, no per-topic answer table.

## What it does

When the user expresses a view whose **subject is a subordinate head** but whose
**sentence also names a salient broader concept** — *"i love the silence of deep
winter"* — RAVANA now records, alongside the held stance, the **salient content
nouns of the whole utterance** (the *provenance*) and can later **bridge a
co-mention of that broader concept back to the held stance**.

Concretely, the utterance *"i love the silence of deep winter"* is mined as a
stance **keyed on the subordinate head `silence`** (that is what the opinion
miner keys on), but the provenance set records
`['silence', 'deep', 'winter']` — including the broader concept **`winter`** the
user actually named. A later *"am i for or against winter?"* would previously
fall through the exact / substring / Jaccard resolver passes (because `winter` is
neither the key `silence` nor a token of it) and hit the honest-fallback *"i
don't have a read on that"* path despite a clear expressed view. With the
provenance bridge, `resolve_topic("winter")` returns `"silence"` and the held
stance is rendered.

The same bridge fixes the **street-art reversal class**: a disclosure and a
reversal both name a salient concept (*"street art"*) that differs from the
keyed head (*"murals"*); the reversal miner routes through `resolve_topic`, so
proving the bridge resolves *"street art"* → *"murals"* proves the reversal links
correctly instead of leaving the honest-fallback path firing.

Real engine output (fresh persona, offline probe):

```text
turn 1: "i love the silence of deep winter, it's the only quiet i get"
        → stance keyed 'silence', polarity +1.0
        → provenance = ['deep', 'silence', 'winter']   # broader concept captured
        ack: "good to know — you love the silence of deep winter. i'll keep that in mind."

turn 2: "am i for or against winter?"
        → resolve_topic('winter') == 'silence'   # provenance bridge, not fallback
        reply: "from what you've told me, you're strongly for silence."
                # a real stance read — NOT the honest-fallback hedge
```

(The reply names the **keyed** stance `silence` because the bridge links the
*co-mention of `winter`* to the held stance on `silence`; the user's view about
winter is answered via the stance they actually expressed.)

**Seed is empty; grown online.** Each stance starts with an empty provenance
set. Provenance is grown from the live utterance's real content words and
**merged across encounters** (`express_stance` unions new nouns into the held
provenance), so RAVANA can revise it by further talk. There is **no per-topic
table and no retraining** — the link is derived at query time from the real
provenance recorded at mining time.

No LLM, no retraining, store-driven. Satisfies the seed + online-learning
constraints: a new salient noun appears in provenance the first time it is
co-named with a stance, without any rebuild.

## Known rough edges (honest — logged for a future round)

- The bridge prefers the stance with the **strongest provenance overlap**; on a
  tie it slightly prefers the more confident stance (score = overlap + 0.001 ×
  confidence). If two held stances share a salient noun, the broader-concept
  query links to the best-overlapping one rather than asking which the user
  meant. This is the same content-word resolver the miner already uses, so it
  inherits the miner's topic-key quality.
- Provenance is mined from content nouns (closed-class stop set stripped). A
  genuinely ambiguous broader noun that co-occurs with many stances may bridge to
  an unintended one. This is bounded by the overlap-count score; a phrase sharing
  no provenance noun still returns `None` (honest abstention).

These are within the resolver's intended behavior; the bridge itself is
store-driven and fail-closed.

## How it grew from the conversation

The chat round of this cycle (round `2026-08-20T0701Z`) surfaced, among its
residual limitations, that a view keyed on a **subordinate head** left a later
question about the **broader concept** it named falling to the honest-fallback
path. The feature card (`t_66dfc855`, residual limitation #1) picked it as a
concrete resolver gap.

**Root cause / prior behavior.** `UserModel.mine_stance` keyed the stance on the
single content **head** of the opinion object (`_opinion_topic`, e.g. `silence`)
and discarded the other salient nouns of the phrase (e.g. `winter`). The stance
resolver `resolve_topic` then had only exact / substring / Jaccard passes over the
**key**, so a co-mention of `winter` — neither the key nor a token of it — found
no held stance and the query hit the honest-fallback hedge. The engine in fact
**held the user's view** (on `silence`), but had **no path to link the broader
co-mention back to it**.

**Fix (commits `11648f7`, `bd10101`).**

1. *Record provenance at mining time* (`bd10101`). A new helper
   `_opinion_provenance(phrase)` (`user_model.py:3161`) returns **all salient
   content nouns** of the FULL opinion-object phrase (the head plus the modifiers
   that survive the closed-class strip), reusing the same stop set the miner
   already routes through. `mine_stance` captures this before `_opinion_topic`
   collapses the phrase to the head and passes it to `express_stance`
   (`user_model.py:2466`).
2. *Store + merge provenance* (`11648f7`). `Stance` gains a `provenance:
   List[str]` field seeded to an empty list (`personal_fact_store.py:250`).
   `express_stance` accepts an optional `provenance` and **unions** it into the
   held stance online (`personal_fact_store.py:339-342`), so a second encounter's
   nouns add to the first's. `get_state`/`set_state` serialize the provenance as
   a backward-compatible tuple tail (`personal_fact_store.py:500-501, 509`).
3. *Bridge at resolve time* (`11648f7`). `resolve_topic`
   (`personal_fact_store.py:350`) gains a final pass after the exact / substring
   / Jaccard checks miss: when the phrase shares a content noun with a held
   stance's **provenance** (and not merely with a different stance's key), it
   bridges to that stance, preferring the strongest provenance overlap
   (`personal_fact_store.py:394-411`). Returns `None` when nothing connects.

**Hardcoding audit.** Neither commit adds authored reply prose or a per-topic
answer table — only a new `List[str]` field, an `express_stance` parameter +
union, a `resolve_topic` bridge pass, serialization of the tuple tail, and the
`_opinion_provenance` content-noun extractor (which reuses the miner's existing
closed-class stop set, no new hardwired topic list). Grep for quoted `>=45`-char
literals on added lines returns 0 hits. The bridge is generic and derived from
live state.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| `Stance.provenance` field (seed empty list) | `ravana/src/ravana/chat/personal_fact_store.py:250` |
| `express_stance(topic, …, provenance=None)` signature | `ravana/src/ravana/chat/personal_fact_store.py:306` |
| Online union of new provenance into held stance | `ravana/src/ravana/chat/personal_fact_store.py:339-342` |
| `resolve_topic` def (exact → substring → Jaccard) | `ravana/src/ravana/chat/personal_fact_store.py:350` |
| Provenance-bridge pass (after the above miss) | `ravana/src/ravana/chat/personal_fact_store.py:394-411` |
| `get_state` serializes provenance (tuple tail) | `ravana/src/ravana/chat/personal_fact_store.py:500-501` |
| `set_state` restores provenance (backward-compatible) | `ravana/src/ravana/chat/personal_fact_store.py:505, 509` |
| `mine_stance` captures full-phrase provenance | `ravana/src/ravana/chat/user_model.py:2466` (`_prov = self._opinion_provenance(_raw)`) |
| `_opinion_provenance(phrase)` — all salient content nouns | `ravana/src/ravana/chat/user_model.py:3161` |
| Provenance tests (store-level) | `tests/unit/test_personal_fact_store.py:298-340` |
| Provenance tests (real-miner + reversal-class) | `tests/unit/test_round_2026_08_09T1953_stance_reversal.py:78-117` |

## Test coverage

Six tests cover the behavior; all pass (run below, `30 passed` includes the
pre-existing stance/fact suite alongside the 6 new ones).

`tests/unit/test_personal_fact_store.py` (4 tests):

- `test_stance_provenance_bridges_broader_concept_query` — key `silence` with
  provenance `['silence','deep','winter']`; asserts `resolve_topic("winter") ==
  "silence"` and `resolve_topic("am i for or against winter") == "silence"`
  (exact/substring/Jaccard miss; only the bridge resolves).
- `test_stance_provenance_empty_seed_does_not_fabricate` — a stance with **no**
  provenance; asserts `resolve_topic("winter") is None` (honest abstention, no
  fabricated link).
- `test_stance_provenance_persists_across_serialization` — `get_state` →
  `set_state` round-trips the provenance and the bridge still resolves.
- `test_stance_provenance_merges_across_encounters` — a second `express_stance`
  on the same key with a distinct noun (`snow`) unions into the held provenance
  (online growth).

`tests/unit/test_round_2026_08_09T1953_stance_reversal.py` (2 tests):

- `test_provenance_captured_from_subordinate_keyed_stance` — real `UserModel`
  mines *"i love the silence of deep winter…"*; asserts the stance is keyed
  `silence` and `winter` is in its provenance, then `resolve_topic("winter") ==
  "silence"` end-to-end.
- `test_provenance_bridges_street_art_reversal_class` — seeds a held stance keyed
  `murals` with provenance `['murals','street','art']`; asserts
  `resolve_topic("street art") == "murals"` (the reversal class links instead of
  falling through).

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_personal_fact_store.py \
    tests/unit/test_round_2026_08_09T1953_stance_reversal.py -v
```

The surrounding stance/opinion/fact suite stays green (the 6 new tests **fail** if
the capability is stashed out — proving a real RED→GREEN — and pass with it).
