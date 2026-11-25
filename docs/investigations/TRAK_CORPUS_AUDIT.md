# TRA-K frozen distributional corpus channel — static/data audit

**Verdict: NO-GO for a TRA-K deployable corpus channel (channel C).**

Static/data audit only: no engine boot, no live network fetch, no pytest, no
production edit. `type_sets`, `world_model` and `unseen_test_words.gold` were
read **only** in the overlap section, where they are the object of the leakage
measurement; none of them is an input to a type fact or a scorer.

Context: the external-ontology channel (B) returned NO-GO
(`docs/investigations/TRAK_ONTOLOGY_AUDIT.md` on `goal/trak-ontology`). The
TRA-K spec (`docs/investigations/TRA_K_SPEC.md` on `origin/docs/tra-k-spec`)
lists the frozen distributional corpus as channel C, "deferred unless B fails".
This audit evaluates that channel against the actual frozen fixtures in this
checkout. Spec for the harness/JSON contract: `scripts/measure_tra_k.py` and
`scripts/tra_k_items.py` on `feat/tra-k-harness` / commit `c962f75`.

Reproduce (from the worktree root):

```
python3 scratch/trak_corpus_audit.py
# machine output: scratch/trak_corpus_coverage.json
```

## 1. What frozen corpus/fixtures exist, and their provenance / licence

| Source | Status | Provenance / licence |
|---|---|---|
| `tests/fixtures/web_snapshots/web_fixtures.json` (3,075,009 B, sha256 `328184927d0679f8538374ebc988c80e1c4b466c8e257a0734f72f25d1b4d3bb`) | **The only substantive frozen text corpus.** 1,087 query→hits entries, 61 Wikipedia `kb_describes` extracts, 207 `tool_web_search` responses. | Schema `ravana-web-fixtures-v1`; `captured_at=2026-11-19T18:40:01Z`; `captured_by=scripts/capture_web_fixtures.py`; `provider=intentforge(localhost:4000) -> local_api -> duckduckgo -> oxiverse`. Every hit carries `title`, `url`, `sources`, `authority`, `score`, `is_local` — page-level provenance is intact. |
| `tests/fixtures/web_snapshots/README.md` | Capture protocol and replay guarantees. | Two live capture passes (985+1049 queries, two top-up rounds) produced 1,087 keys; replay `scripts/measure_fronts_web_frozen.py` runs 434 probe rows at zero network attempts, zero saves, zero fixture misses. This is an independent snapshot, not generated from the item bank. |
| `data/common_facts.json` | 15 authored `is_a` facts (`blue`, `cat`, …); **0/25** subject words. | Internal authored seed, no independent provenance; already rejected in the ontology audit. |
| `data/curriculum/causal_chains.json` | Internal causal curriculum; **2/25** subject words incidentally (`cloud`, `heart`). | Authored in-repo, not an independent snapshot. |
| `data/snippet_pe.json`, `data/functional_lexicon.json`, `data/eval_results.json` | 0/25, 0/25, 1/25 (`shower`, inside eval prose). | Internal data / eval logs, not a corpus. |
| ConceptNet `data/conceptnet/ont.pkl` | Absent, gitignored (see ontology audit). | CC BY-SA 4.0 but no data in checkout. |

**Licence.** The repo is under the Oxiverse Community License (`LICENSE`). The
fixture file is generated from third-party web pages (Wikipedia, Met Office,
National Geographic, Medical News Today, …); the snapshot records titles and
URLs but **no per-page reuse licence or text licence grant**. Short attributed
snippets with source URLs are the whole record; redistribution terms are not
established by the repo. Any shipped corpus fact would have to carry the
fixture path + query + URL as provenance (as the proposed format below does),
and per-page licence clearance would still be an open question. This is a
secondary concern: the channel fails on coverage first.

**No other local corpus covers the vocabulary.** The fixture bundle's dedicated
"one sentence" descriptions cover animals/plants/rocks (`dolphin`, `geyser`,
`mirage`, `granite`, …) — none of the 25 TRA query subjects.

## 2. Coverage of the 25 subjects and 6 type labels

Subject set = the 40 graded `near` subjects of the recorded TRA replay
(`scratch/type_acquisition_results.json`, rows with `condition=near`), 25 unique
words, 24 `weather` + 16 `body` instances; labels from `tra_items.ROLE_TO_TYPE`:
`effect` 7, `structured-whole` 7, `earlier-event` 7, `similar-form` 7,
`supported-outcome` 6, `contrast-case` 6.

"Any doc" = any of the 1,355 blobs contains the word as a whole word.
"Topic query" = a query *directed at* the word (bare word, `what is X`,
`about X`, `describe X …`, `role of X`, …). "Topic URLs" = distinct page URLs
from those topic queries.

| Subject | any doc | topic query | topic URLs | topic context? |
|---|---:|---:|---:|:---:|
| `artery` | 0 | 0 | 0 | no |
| `blizzard` | 0 | 0 | 0 | no |
| `cartilage` | 8 | 0 | 0 | no |
| `chill` | 1 | 0 | 0 | no |
| `climate` | 57 | 2 | 6 | **yes** |
| `cloud` | 28 | 0 | 0 | no |
| `dew` | 18 | 8 | 39 | **yes** |
| `diaphragm` | 4 | 0 | 0 | no |
| `drizzle` | 1 | 0 | 0 | no |
| `gale` | 0 | 0 | 0 | no |
| `heart` | 33 | 2 | 7 | **yes** |
| `joint` | 1 | 0 | 0 | no |
| `ligament` | 0 | 0 | 0 | no |
| `liver` | 17 | 9 | 10 | **yes** |
| `marrow` | 0 | 0 | 0 | no |
| `mist` | 6 | 0 | 0 | no |
| `monsoon` | 0 | 0 | 0 | no |
| `reflex` | 0 | 0 | 0 | no |
| `shower` | 3 | 0 | 0 | no |
| `squall` | 1 | 0 | 0 | no |
| `storm` | 28 | 0 | 0 | no |
| `tempest` | 0 | 0 | 0 | no |
| `tendon` | 0 | 0 | 0 | no |
| `thaw` | 1 | 0 | 0 | no |
| `typhoon` | 10 | 0 | 0 | no |

Headline: **16/25** subjects appear anywhere, but only **4/25** have any
word-directed context (`dew`, `liver`, `heart`, `climate`); **9/25** have zero
occurrences. Most "any doc" hits are incidental mentions inside a page about
something else (e.g. `storm` inside a tornado/hurricane page, `cloud` inside a
fog page, `thaw` in one unrelated sentence). The 4 topic contexts are genuine
but shallow:

| Subject | Query | Example page (URL in JSON) |
|---|---|---|
| `dew` | `dew` | "Dew is the small droplets of water that appear on objects such as leaves or grass in the morning or evening…" — Met Office, `https://weather.metoffice.gov.uk/learn-about/weather/types-of-weather/temperature/dew` |
| `liver` | `how does role of the liver work` | "Inside the Liver and Its Constant Role in Detoxification, Metabolism, Bile, Blood Filtration" — Medical Daily |
| `heart` | `role heart` | "The heart: Anatomy, how it works, and more" — Medical News Today |
| `climate` | `drives climate` | "What Drives Climate Change? Key Natural and Human Influences" — Environmental Studies Institute |

These are subject descriptions, not role/type assertions.

**Label coverage: 1/6.** Literal occurrences of the six label strings across the
whole corpus: `effect` 38 docs, all other five **0 docs**. There are **0**
occurrences of the pattern `is/are a/an <label>` for any label. Using the
ontology audit's pre-registered lexical targets (e.g. `effect → consequence,
result, outcome, …`) under a *strict subject-topic* filter gives **1/40** items
with any co-occurrence — and that single hit is spurious (`climate` / supported-
outcome licensed only by the phrase "the **result** of one long exchange" in a
disclaimer on an AI-essay page). Effective genuine grounding is **0/40**.

## 3. Are contexts subject-specific, independent of candidates and of gold?

* **Subject-specific, weakly.** For the 4 words with topic contexts, the pages
  really are about that word (`dew` → 39 distinct URLs, `liver` → 10, `heart`
  → 7, `climate` → 6) and no candidate shares those topic-URL sets
  (`candidates_sharing_context = []` for all 40 items). But 36/40 items have
  *no* topic context for either the subject or its candidates, so this is an
  absence of candidate-invariance by emptiness, not evidence of a usable
  channel. Polysemy also leaks in: 2 of the 6 hits on the bare `dew` query are
  Mountain Dew and a state employment portal (SC DEW) — the same string, a
  different referent.
* **Not role-specific.** Even the clean contexts say "dew is condensed water",
  "the liver filters blood", "the heart pumps blood". They do not say which of
  the six *item-relative* TRA roles the word instantiates. As the ontology audit
  established, each of the 25 subjects carries 4–6 different role labels across
  items, so no context that names only the word can select the item's role.
  Naive whole-blob co-occurrence *appears* to ground 26/40 items, but the hits
  are inside queries about other subjects (`storm` + `result` in a tornado
  counterfactual page); the strict subject-topic filter removes 25 of them.
* **Independent of gold in construction.** The fixture file predates the item
  bank, has no item ids or gold fields, and the audit never selects a fact by a
  gold token. Measured only: 97/142 gold words across the 400-family bank occur
  somewhere in the corpus text — incidental overlap, not a channel input.
* **C15 constraint measured.** All **25/25** subject words appear as answer-key
  tokens in *some* bank item, so a channel that names the query subject as a
  fact subject would trip the TRA-K `answer_key_leak` HARD FAIL. By contrast,
  the taught exemplar (`r1`/`r2`) words collide with *their own item's* answer
  key **0/6,400** times; a gold-free channel must use only those exemplar words
  as fact subjects (the same restriction the deterministic fallback honours).
  Baseline overlap with the fallback is clean: 3,200 facts, 0 leaks, 0 domain-
  tag leaks, 0 missing provenance.

## 4. Concrete channel JSON format (validated against the real loader)

`TRAK_CHANNEL_JSON` is read by `tra_k_items.load_channel_evidence()`; the
accepted shapes and the required fact keys are specified in that module on
`feat/tra-k-harness`. A corpus-backed file would look like this:

```json
{
  "schema": "ravana-trak-corpus-channel-v1",
  "corpus": {
    "fixture": "tests/fixtures/web_snapshots/web_fixtures.json",
    "sha256": "328184927d0679f8538374ebc988c80e1c4b466c8e257a0734f72f25d1b4d3bb",
    "captured_at": "2026-11-19T18:40:01Z",
    "snapshot": "ravana-web-fixtures-v1"
  },
  "items": {
    "tra_k_0000": [
      {
        "subject": "storm",
        "predicate": "is_a",
        "object": "effect",
        "channel": "C",
        "source": "tests/fixtures/web_snapshots/web_fixtures.json",
        "source_id": "web_fixtures@2026-11-19:328184927d0679f8",
        "snapshot": "ravana-web-fixtures-v1",
        "context": {
          "query": "if under conditions tornado earth effects",
          "title": "…",
          "url": "https://…"
        }
      }
    ]
  }
}
```

Contract (checked statically, no engine boot): `subject`, `predicate`, `object`
and `source` are mandatory; predicate must be one of `is_a`, `type_of`,
`instance_of`; `channel_assertion()` renders the example as
`storm is an effect.` and silently ignores non-`is_a` predicates. Extra fields
(`context`, `channel`, `source_id`, `snapshot`, `item_id`) survive normalisation.
Loading the example with the actual `tra_k_items.load_channel_evidence()` gave
`external_items=2, invalid=0, error=None`; `build_items(channel_override=…)` used
both facts and `channel_overlap()` reported zero provenance/domain-tag leaks.
The *only* leak in that test was a deliberate demonstration: I used `frost` for
`tra_k_0001` where `frost` is that item's answer key → `answer_key_leak=1`. A
real generator must restrict subjects to that item's taught exemplar words.

Gold-free by construction: the generator would read only (a) the frozen
fixture, (b) the item's `item_id`, `taught_exemplars` and `type_label`
(all deployable fields in `DEPLOYABLE_INPUT_FIELDS`), and (c) a fixed corpus→
role evidence rule. It would never read `type_sets`, `world_model`, gold,
`domain`/`far_domain` as a type, or the query subject.

**But the corpus cannot supply the evidence such facts would claim.** There is
no `is/are a/an <role-label>` sentence for any word, 5/6 labels never occur at
all, and the remaining co-occurrences are off-topic. Emitting
`storm is an effect.` from a tornado page would fabricate a type assertion the
fixture does not contain — exactly the provenance laundering the skill forbids
("fix the write, stamp provenance; never forge a blanket stamp"). A channel that
supplies the label from `item["type_label"]` and merely cites an unrelated page
as "evidence" is not an independent knowledge channel; it is the fallback with
a misleading citation.

## 5. GO / NO-GO and numbers

**NO-GO.** `scratch/trak_channel_corpus.json` was therefore **not** generated.

| Check | Result | Threshold implied by the task |
|---|---|---|
| Independent frozen corpus with provenance | PASS — 1,087 queries, 61 extracts, 207 tool responses, per-hit URL/source/title | exists |
| Unique subject words with any mention | **16/25** (64%) | high coverage |
| Unique subject words with word-directed context | **4/25** (16%) | high coverage |
| Unique subject words with zero mentions | **9/25** (36%) | ~0 |
| Type-label strings present in corpus | **1/6** (`effect` only, 38 docs, 0 as `is_a`) | 6/6 or a defensible map |
| Items with a corpus `X is_a <required label>` assertion | **0/40** | >0 |
| Items with strict topic-grounded label evidence | **1/40 raw, 0/40 genuine** (the one hit is an unrelated "result") | most items |
| Topic contexts shared with a candidate | **0/40** (but 36/40 have no topic context at all) | 0 |
| Subject words that are answer-key tokens somewhere | **25/25** → naming the query subject trips C15 | 0 for a deployable fact |
| Own-item exemplar/answer-key collisions | **0/6,400** → safe subjects exist | — |
| Gold used as a channel input | **No** (97/142 gold words appear incidentally; measured only) | 0 |

The finding is an **information deficit, not a tuning problem** — the same
class of result as the ontology audit: the TRA label is item-relative, so no
word-level snapshot, lexical or distributional, can derive it. The corpus adds
subject-specific descriptions for 4 of 25 words but none of the role evidence.

Next step (benchmark-design change, not another threshold): either define the
role as query input and test candidate-side typing with an independently
authored single-type-per-word lexicon, or keep channel A (ostensive teaching)
as the declared non-deployable positive control and drop channel C. Do not open
a TRA-K harness PR on this channel. Supporting data: `scratch/trak_corpus_coverage.json`.
