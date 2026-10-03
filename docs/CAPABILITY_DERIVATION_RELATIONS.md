# CAPABILITY — DERIVATION relations (`<thing> is named after <referent>`)

Status: **partially wired, documented honestly.** Mining and store-level recall
are verified working; the chat reply path does **not** yet surface a derivation
fact. The gap is stated below with the exact reason. Nothing on this page is
inferred — every claim is backed by executed output or a `path:line` citation.

Source of the capability: `ravana/src/ravana/chat/derivation_attrs.py` (315
lines, one shared vocabulary) plus the `_mine_derivation` miner in
`ravana/src/ravana/chat/user_model.py:2266`.

---

## 1. What it does

A **derivation** disclosure relates TWO things: an owned thing and the referent
it is named after / called after / inspired by / based on.

    my sourdough starter is named after my uncle bartholomew

RAVANA stores it **entity-keyed**, with the relation as the attribute:

    ('sourdough starter', 'named after', 'uncle bartholomew')

Three properties matter, and each is enforced by a structural rule rather than
a phrase list:

| Property | Why it matters | Enforced at |
|---|---|---|
| Subject is the **owned thing**, not the user | A derivation is about the starter, not about the speaker | `user_model.py:2364-2376` — subject = NP before the naming verb, leading determiner/possessive dropped; `return False` if the subject is `i` |
| **Relation survives** as the attribute | Without it the fact degenerates into a restatement of its own key | `derivation_attrs.derivation_of` (`derivation_attrs.py:240`) resolves a naming verb + attribution preposition to one canonical relation |
| Referent **keeps its relationship word** | `uncle bartholomew` stays recallable by the relationship | `user_model.py:2347-2360` — bounded noun-phrase scan that lets a kin word legitimately start the span |

## 2. What it grows from, and how

The vocabulary is **seed structure, not an answer table**, and it is extended at
runtime. Nothing here is a source of reply text.

- **Seed derivation predicates** — `derivation_attrs.py:43-66`, `_DERIVATION_SEED`.
  23 surface phrases → 4 canonical relations (`named after`, `inspired by`,
  `modeled on`, `based on`).
- **Seed naming verbs** — `derivation_attrs.py:76-79`, `_NAMING_VERB_SEED`.
  A closed class of 9 high-frequency English verbs (name, call, christen,
  title, nickname, dub, label, rename, dedicate).
- **Attribution prepositions** — `derivation_attrs.py:87-103`. `_ATTRIBUTION_PREPS`
  is the morphological recognizer; `_NAMING_PREPS` is the narrower set that may
  admit a verb RAVANA has **never seen** (`derivation_attrs.py:274-275`).
- **Online growth** — `learn_derivation` (`derivation_attrs.py:218`) and
  `learn_naming_verb` (`derivation_attrs.py:137`). Both mutate module-level
  stores during a normal turn. **No retraining, no rebuild, no code change**:
  `learn_naming_verb("tagged")` returns `"tag"`, after which
  `derivation_of("tagged after")` resolves and `naming_verb_of("tagging")` /
  `naming_verb_of("tags")` fold onto the same stem.

The miner calls `learn_naming_verb` on **every** token as it scans
(`user_model.py:2314`), so a naming verb RAVANA has never heard of is registered
during the very disclosure that uses it, and is addressable on the next one.

### Morphological, not phrase-lookup

Resolution is `naming verb + attribution preposition`, tried longest-preposition
first (`derivation_attrs.py:259`). So an unseen combination reduces correctly
with no table entry:

    derivation_of("dubbed after")  -> "dub after"   (unseen verb keeps its stem)
    derivation_of("named after")   -> "named after" (seed verb keeps the SEED key)

The seed-verb rule is load-bearing. Canonicalizing `"named after"` to the stem
`"name after"` produces a key **no later surface question can resolve** — the
fact would be written and never recalled. Hence `_SEED_BY_STEM_PREP`
(`derivation_attrs.py:119-134`) is built *from the seed table itself*, so the
canonical form of an inflected seed verb folds back onto the seed key.
Guarded by `test_canonical_form_keeps_the_english_verb_not_a_stem`.

### Why the class does not leak

An unseen head verb is admitted only under a naming-specific preposition
(`after`, `in honor of`, `in memory of`, `tribute to`) — never under a generic
one (`from`/`by`/`on`/`to`), which every transitive verb takes with an ordinary
object. Verified live:

    "my grandmother indira weaves baskets from river reeds"
      -> NO derivation fact stored; the plain activity fact IS stored
         ('i', 'grandmother indira', 'weaves baskets')

Without that gate the generic-preposition path resolves an ordinary activity as
an attribution and **steals the disclosure from the relationship miner**
(`derivation_attrs.py:268-273`).

`learn_naming_verb` additionally refuses a bare noun carrying no verb
inflection (`derivation_attrs.py:156-164`), so merely being scanned does not
register it: `learn_naming_verb("starter")` returns `None`.

## 3. Verified output

Live in-process probe, `RAVANA_OFFLINE=1`, `.venv-real`, `dim=64 seed=42`,
on a clean `user_suffix`:

    "my sourdough starter is named after my uncle bartholomew"
      ('sourdough starter', 'named after', 'uncle bartholomew') -> uncle bartholomew
    "my espresso machine is called after my grandfather salvatore"
      ('espresso machine', 'named after', 'grandfather salvatore') -> grandfather salvatore
    "my api is called after my old notebook"
      ('api', 'named after', 'old notebook') -> old notebook

Note `"called after"` canonicalizes to the **same** attribute `"named after"` —
one relation, two surfaces, one key.

Store-level recall, `PersonalFactStore.query_fact` (`personal_fact_store.py`):

    query_fact("sourdough starter", "named after")
      -> [PersonalFact(subject='sourdough starter', attribute='named after',
                       value='uncle bartholomew', confidence=0.75,
                       turn_number=1, rehearsal_count=2, source='seed_regex',
                       superseded=False)]
    query_fact("i", "named after")
      -> []          # nothing is filed under the user
    query_fact("sourdough starter")
      -> [the same fact]

The fact is retrievable, non-superseded, and carries the store's normal
rehearsal accounting.

Renderer, `derivation_attrs.render_derivation` (`derivation_attrs.py:298`):

    render_derivation("sourdough starter", "named after", "uncle bartholomew")
      -> "your sourdough starter is named after uncle bartholomew"

Every content word comes from the store; only the connectives (`your`, `is`) are
fixed grammar. It is a renderer over stored state, not an authored reply.

## 4. Honest limitations

1. **The chat reply path does not surface a derivation.** Verified live:

       "what is my sourdough starter named after?"
         -> "your sourdough is after."

   The stored fact is correct and `query_fact` returns it, but the query does
   not resolve the entity key (`sourdough` truncates `sourdough starter`) and no
   reply path calls `render_derivation` — `grep -rn render_derivation ravana/src`
   returns **only its definition**. So a derivation is currently
   **write-only from the user's point of view**. The renderer is exercised by
   tests but not reachable from a turn. This is a real gap, not a phrasing
   nit; closing it needs an entity-keyed recall path for derivation attributes,
   which does not exist yet.
2. The confirmation turn is ungrammatical
   (`"your sourdough starter's named after is uncle bartholomew"`) — the
   acknowledgement renderer has no derivation frame.
3. `_DERIVATION_SUBJ_AUX` (`user_model.py:438`) is the auxiliary set stripped
   from the right of the subject key (`user_model.py:2386-2387`). It is a
   closed class, so an unlisted auxiliary would still leak into the key; the
   degradation is a stored key no later question matches, never a wrong answer.

## 5. Coverage

`tests/test_derivation_capability.py` — 16 tests, all passing
(`pytest tests/test_derivation_capability.py` -> `16 passed in 45.69s`):

- vocabulary: seed canonicalization, non-derivation rejection, seed-key vs stem,
  runtime growth of an unseen verb, refusal of non-verbs, no memorisation of a
  failed lookup, renderer output;
- miner: entity-keyed storage, no degenerate fact, unseen verb mined + vocabulary
  grows, non-attribution disclosure produces no derivation fact, the user's own
  naming is left to the name miner;
- subject key: bare noun phrase, arbitrary-length perfect-auxiliary strip,
  over-strip guard, whole-suite auxiliary invariant.

## 6. Why a shared module

`derivation_attrs` follows `relation_attrs` (kin) and `pet_slots` (species):
**one** shared notion so the miner and every future recall site agree by
construction, instead of N copies of a hand-kept synonym table that drift apart.
This is the same remedy the round applied to the pet-species slot collapse.

## 7. Not this

No LLM, no retraining, no per-entity or per-person table, and no reply string
authored for any probe sentence. The vocabularies are class structure; the
renderer's output is composed from stored state at call time.