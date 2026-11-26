# Independent semantic type substrate for TRA-K — source choice, build, audit

**Verdict: GO for the substrate artifact.** It is a static, provenance-stamped
Wikidata type lexicon, independent of the TRA-K answer key. This is a
docs-and-data deliverable only: no engine boot, no TRA/RSI change, no gate
change, and no benchmark score is claimed.

## 1. Why a new source was needed

The two prior channel audits were NO-GO because they tried to recover TRA's
six *authored* role labels (`effect`, `structured-whole`, `earlier-event`,
`similar-form`, `supported-outcome`, `contrast-case`) from a word-level source.
WordNet covered the subjects but only 1/6 of those label strings, and each of
the 25 focused subjects carried 4–6 conflicting role labels across items
(`docs/investigations/TRAK_ONTOLOGY_AUDIT.md`, `goal/trak-ontology`). The frozen
web fixtures gave word-directed text for only 4/25 subjects and contained no
`is_a <role-label>` assertion (`docs/investigations/TRAK_CORPUS_AUDIT.md`,
`goal/trak-corpus`). Those results are untouched by this work.

The changed requirement is decisive: the substrate's type labels must be
**source-native**, not TRA-authored. A source that cannot say `dew is an
effect` can still say `dew is condensation water` and `dew is a hydrometeor`;
the latter are genuine category facts a child could already possess.

## 2. Source selection (Phase 1)

| Candidate | Authority / licence | Decision |
|---|---|---|
| **Wikidata structured data** | All structured data is released into the public domain under CC0 [1][2]; `P31` is "instance of" and `P279` is "subclass of" with distinct meanings [5][6]; read through the documented Action API [7][8] and query service [4]. Cite Vrandečić & Krötzsch, *CACM* 57(10):78–85, doi:10.1145/2629489 [9]. | **Chosen.** CC0 imposes no attribution or share-alike obligation; explicit instance/subclass edges; real coverage of ordinary English nouns. |
| ConceptNet 5 | CC BY-SA 4.0 with explicit attribution and share-alike obligations [10]. | Not chosen: `api.conceptnet.io/c/en/dew?rel=/r/IsA` returned HTTP 502 during the capture window, and share-alike is unnecessary given CC0 alternative. |
| DBpedia | Shallow cross-domain ontology with ~768 classes, SPARQL endpoint and dumps [11]. | Not chosen: no local dump in this checkout and no verified ordinary-word instance coverage in the window; CC0 source preferred. |
| WordNet 3.0 | Princeton licence permits use but requires the notice/disclaimer on all copies [12]. | Previously audited NO-GO for TRA labels; retained only as a documented alternative, not a source of this substrate. |
| Frozen web fixtures | Provenance-rich but no per-page reuse licence; see corpus audit. | Previously audited NO-GO; not reused. |

Independence: the only item-bank data read is the **public vocabulary** in the
`OBJECTS` pools of `scripts/tra_k_items.py` at `c962f75` (144 ordinary words),
used as query strings. `type_sets`, `world_model`, gold targets,
`unseen_test_words` and every per-item answer key are excluded from the
substrate and were read only by the audit step that *measures* overlap.

**Static-snapshot limitation.** WDQS was in an outage/rate-limit window
(`429: 1 req / min` during outage 797a132, then `504`) and the MediaWiki Action
API throttled sustained searches. The durable input is therefore a static
subset: six SPARQL batches of 24 words fetched with 66 s spacing, 994 raw
bindings saved verbatim in `scratch/wikidata_sparql_raw.json` (sha256
`52f74c9d5fa2f3a1d5376cd7ef43508ea98490b2eee248ff5824b6a33eb18812`). This is
a snapshot, not a live API dependency, and it is fully reproducible offline
from the checked-in raw file.

## 3. What a fact contains

Each fact in `scratch/trak_substrate.json` carries `subject`, `predicate=is_a`,
`type_label`, `source_url`, `license` (`CC0-1.0`), `license_url`, `span` and
`text_evidence`, plus `wikidata_property` (`P31`/`P279`), `subject_qid`,
`type_qid`, `subject_description`, `type_description`, `enwiki_title`, `sense`
and `sense_selection`. `span` is an explicit **rendering of a structured
statement** (e.g. `dew — subclass of (P279) → hydrometeor (Q1972098)`), not a
quotation from a web page; `text_evidence` is a verbatim English description
taken from the same Wikidata items. Evidence tiers: item description, type
description, or (5 facts) label-only where neither description exists. All
980 published facts were machine-checked against the raw snapshot: **0
facts are absent from the raw bindings**.

## 4. Exact type labels — source-native, not TRA labels

Type labels are the English labels of Wikidata's type *items*, never a
mapping onto TRA's authored labels. Examples: `dew → condensation water,
hydrometeor`; `heart → animal organ, organ type`; `cloud → weather, type of
meteorological phenomenon`; `shower → plumbing fixture, industrial plant`
(primary/Wikipedia topic) with alternate sense `Shower (precipitation) → fall
of precipitation`. There are **638 distinct type labels** and **0/638** equal
any of TRA's six authored labels.

Composition is preserved rather than flattened: P31 means the subject is an
instance of a type while P279 means every instance of the subject is an
instance of the type [5][6]. A consumer can therefore interpret an
`is_a`-phrase compositionally (instance-of vs kind-of, and chains of kind-of)
using the retained `wikidata_property` field. This is the pre-existing typed
schema a child could bring into a type-phrase task; no TRA or RSI code is
modified here.

## 5. Vocabulary coverage

| Measure | Result |
|---|---|
| Vocabulary words queried | 144 |
| Words with ≥1 published type fact | **140 / 144 (97.2%)** |
| Words with a primary-sense fact | 139 / 144 |
| Facts published (all senses) | 980 |
| Facts in primary sense | 429 |
| Facts in alternate senses | 551 |
| Distinct type labels | 638 |
| Words with no typed item in snapshot | `chill`, `marrow`, `sleet`, `thorn` (4) |
| Facts with only label evidence | 5 |

The four gaps have **zero** P31/P279 bindings in the snapshot; no fact was
invented for them.

## 6. Overlap with the answer key (measured, never used as input)

The substrate is a global lexicon, so its subjects necessarily cover words
that are gold or query subjects in *some* item. Measured only:

| Measure | Result |
|---|---|
| Answer-key vocabulary tokens across the 400 families | 144 |
| Substrate subjects that are answer-key tokens somewhere | **140 / 140** — expected for a global lexicon, **not** an item-level leak |
| Substrate type labels equal to TRA's six authored labels | **0 / 638** |
| Items with ≥1 deployable fact on a taught exemplar word (subject excluded from *that item's* answer key) | **400 / 400** |
| Deployable facts per item (all senses) | min 40, mean 112.7, max 220 |
| Items with ≥1 deployable primary-sense fact | 400 / 400 (mean 51.0) |

No substrate record contains a per-item field, an `item_id`, a gold target, a
`type_sets` value, a `world_model` value or an `unseen_test_words` value. Any
future item-level projection must repeat the harness's own filter: exclude
facts whose subject is in that item's answer-key token set.

## 7. Conflicts and limitations

Wikidata is multi-typed by design: 108/140 covered words have more than one
distinct type label even after primary-sense selection (127/140 across all
senses). This is honest source behaviour, not benchmark evidence; a
type-phrase mechanism must choose a label from the phrase rather than expect
one type per word. Primary-sense selection used: exact English-Wikipedia title
match for 94 words, description-contains for 11, and most-typed-item/QID
tie-break for 35. Domain senses can differ from the primary Wikipedia sense
(e.g. `shower`), which is why alternate senses are retained with explicit
`sense` and `enwiki_title`. No claim is made that these facts improve TRA, RSI
or any gate; that requires a future, separately preregistered harness run.

## 8. Manifest hashes

| Artifact | SHA-256 |
|---|---|
| `scratch/wikidata_sparql_raw.json` | `52f74c9d5fa2f3a1d5376cd7ef43508ea98490b2eee248ff5824b6a33eb18812` |
| `scratch/trak_substrate.json` (file, LF-normalised) | `a24e2714b3ab077c05f03e894600bace822a8c9b1f2b2ad8d1d7226e1bd02de6` |
| Substrate canonical facts digest | `ed4759ea57184fd74f2d6d932d12eeecec2e84de6bc075eb3cdc27469c106a90` |

The file digest is taken over LF-normalised bytes: the committed blob is LF,
while `core.autocrlf=true` checkouts materialise CRLF. Hashing raw worktree
bytes made the frozen TRA-K provenance `snapshot` (and therefore its manifest
pin) platform-dependent; `tra_k_items.substrate_sha256()` normalises first.

Full machine-readable audit: `scratch/trak_substrate_coverage.json`.
Build scripts: `scratch/trak_fetch_sparql_batches.py` and
`scratch/trak_build_substrate.py`.

## 9. GO / NO-GO

**GO** for the deliverable: an independent, CC0-licensed, provenance-stamped
type substrate with 97.2% vocabulary coverage, 980 source-traceable facts,
zero overlap with TRA's authored label vocabulary, and deployable exemplar
facts for all 400 families. **NO-GO** for any claim about TRA/RSI performance —
this benchmark change is out of scope here, and the WFA run was not touched.

## Sources

[1] https://www.wikidata.org/wiki/Wikidata:Licensing — Wikidata:Licensing
[2] https://creativecommons.org/publicdomain/zero/1.0 — CC0 1.0 Universal
[4] https://query.wikidata.org — Wikidata Query Service endpoint
[5] https://www.wikidata.org/wiki/Property:P31 — Wikidata P31 instance of
[6] https://www.wikidata.org/wiki/Property:P279 — Wikidata P279 subclass of
[7] https://www.mediawiki.org/wiki/API:Wikidata — Wikibase API documentation
[8] https://www.wikidata.org/w/api.php — Wikidata Action API endpoint
[9] https://doi.org/10.1145/2629489 — Vrandecic & Krotzsch 2014, Wikidata, CACM 57(10)
[10] https://github.com/commonsense/conceptnet5/wiki/Copying-and-sharing-ConceptNet — ConceptNet licence
[11] https://www.dbpedia.org/resources/ontology — DBpedia ontology
[12] https://wordnet.princeton.edu/license-and-commercial-use — WordNet licence
