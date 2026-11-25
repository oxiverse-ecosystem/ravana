# TRA-K external ontology channel — static/data audit

**Verdict: NO-GO for building TRA-K on channel B (external ontology).**

Static/data audit only: no engine boot, no pytest, no production edit, no network
call in the measurement. `type_sets`, `world_model` and `unseen_test_words.gold`
were read **only** in the overlap section, where they are the target of the
measurement; none of them is an input to any type assertion or scorer. Spec
audited: `origin/docs/tra-k-spec:docs/investigations/TRA_K_SPEC.md`.

Reproduce: `C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe scratch/trak_ontology_audit.py`
(nltk 3.10.0 lives in the main checkout's venv; the worktree has none). Machine
output: `scratch/trak_ontology_coverage.json`.

## 1. What is actually available offline

| Source | Offline status | Licence / citation |
|---|---|---|
| **NLTK WordNet 3.0** (English) | **Available.** `nltk 3.10.0`; `wordnet.zip` + `omw-1.4.zip` at `C:/Users/Likhith/AppData/Roaming/nltk_data/corpora/`. Local `wordnet/LICENSE` says *WordNet Release 3.0*. | WordNet 3.0 licence, Princeton University (2006): use/copy/modify/distribute free, **but the copyright notice + disclaimer must appear on all copies**, no Princeton advertising use, citation required. Authoritative: https://wordnet.princeton.edu/license-and-commercial-use . Cite Miller (1995), *CACM* 38(11):39–41; Fellbaum (ed.) (1998), MIT Press. |
| ConceptNet 5 (`data/conceptnet/ont.pkl`) | **Absent** — directory does not exist in this checkout; gitignored (`data/conceptnet` in `.gitignore`). `ravana/src/ravana/ontology/conceptnet.py` is a loader, not data. | CC BY-SA 4.0 (attribution + share-alike, no extra restrictions): https://conceptnet.io/ , https://zenodo.org/records/3739540 . Licence is fine; data is not here. |
| Bundled ontology/type files in repo | **None.** No `.owl`, `.rdf`, `.ttl`, `.nt`, `ont.pkl` or `wordnet.zip` anywhere in the worktree. `data/common_facts.json` has 15 authored is_a facts (`blue`, `cat`, `dog`, …) — internal seed, not an independent snapshot; **0/25** of the queried subjects appear in it. | Internal authored data; no independent provenance. |

WordNet is therefore the only candidate channel B source, and it is usable under
its NOTICE obligation (a shipped snapshot would have to carry `wordnet/LICENSE`).

## 2. Coverage of the 40 subjects and 6 type labels

**Subject set.** The 40 graded near-condition query subjects of the recorded TRA
mechanism replay (`scratch/type_acquisition_results.json`: 20 primary + 20
sealed), all verified equal to `build_items()[id].unseen_test_words.near.subject`.
They are **25 unique words**; this focused slice contains only the `weather`
(24 instances) and `body` (16) domains.

**Six labels** (`tra_items.ROLE_TO_TYPE`): `effect`, `structured-whole`,
`earlier-event`, `similar-form`, `supported-outcome`, `contrast-case`.

| Measurement | Result |
|---|---|
| Subjects with ≥1 WordNet noun sense | **25/25 unique words** (40/40 instances) — lexical coverage is *not* the problem |
| Label strings that exist as WordNet noun synsets | **1/6** (`effect`, 6 synsets). The other five are role labels invented by TRA, not lexical classes. |
| Subjects whose WordNet hypernym closure matches a pre-registered lexical target for that label | `effect` **0/40**, `earlier-event` **0/40**, `similar-form` **0/40**, `supported-outcome` **0/40**, `contrast-case` **0/40**, `structured-whole` **6/40** |
| Domain-level hypernym coverage (weather/body) | **32/40** (weather 19/24, body 13/16); 0 instances of the other four domains in this slice |

The 6 `structured-whole` hits are lexical accidents, not role evidence: `artery`,
`diaphragm`, `liver`, `ligament` have *structure*/*whole* as a hypernym in some
sense, and the same words are used under other roles in other items.

Mapping examples (pre-registered label target → what WordNet actually returns):

| Subject | Required label | WordNet top-sense hypernyms | Label target match |
|---|---|---|---|
| `tempest` | effect | abstract entity, abstraction, attribute, commotion | none |
| `drizzle` | earlier-event | atmospheric condition, atmospheric phenomenon, downfall | none |
| `dew` | similar-form | atmospheric phenomenon, condensate, condensation | none |
| `climate` | supported-outcome | abstract entity, abstraction, condition | none |
| `storm` | contrast-case | atmospheric phenomenon, natural phenomenon, physical entity | none |
| `artery` | structured-whole | structure, whole (one sense only) | spurious hit; `artery` also appears under 5 other roles |

Full per-subject results, glosses and hypernym samples are in the JSON.

## 3. Overlap with `type_sets` / `world_model`

**Low where it matters, high where it does not — and the high part is explained.**

- **Role-label overlap ≈ 0.** With the gold word in hand (oracle-only diagnostic,
  never fed to a deployable scorer), WordNet assigns the item's role label to the
  gold in **1/40** cases — the spurious `esophagus is_a structure` hit for a
  `structured-whole` item.
- **The key's "type" is not a lexical property.** Of the 144 vocabulary words
  carrying a role label in `type_sets`, **144 (100%)** carry *conflicting* labels
  across items: `angle`, `artery`, `axle`, `bark`, `belt`, … each receive all six
  labels. Cause: `build_items()` writes `type_sets[w] = [type_label, domain_event]`
  per item, with `type_label` determined by the item's hash-assigned role. All
  **25/25** queried subjects appear under 4–6 different roles (distribution:
  9×6 roles, 13×5, 3×4). No word-level ontology can reproduce an item-relative
  label; any apparent agreement would be contamination, not coverage.
- **Domain-tag overlap 32/40** (WordNet recovers `weather`/`body` for the subject).
  This does not solve the benchmark: `near` candidates are drawn from the *same*
  domain pool, so a domain-level type assigns the same class to the gold and to
  every distractor. The key's `domain + "-other"` distractor tag is authored and
  has no ontology counterpart.
- **`world_model` is structurally out of reach.** It maps `(role, subject)` to a
  sha256-selected domain word; that mapping is not a lexical relation and has no
  standalone ontology entry. 0/40 reproducible from WordNet by construction.

## 4. Can the channel produce a per-subject type assertion without the gold?

**No.** For each subject, WordNet's top-sense hypernym gives a domain-level
assertion (`tempest is_a abstract entity`, `shower is_a artefact`, `drizzle is_a
atmospheric condition`) — **0/40** equal the required TRA-K label. **19/40**
subject-instances have multiple plausible sense types, so even the domain-level
assertion is not stable. To emit `X is an effect` the channel would need the
item's role, which is supplied by the item assignment (and, at query time, by the
question's relation phrase), not by any word's taxonomy entry. There is no
gold-free subject→label map to snapshot.

## 5. GO / NO-GO and the next work item

**NO-GO.** Numbers: 25/25 lexical subject coverage but **1/6** label vocabulary,
**0–6/40** role-type coverage (the 6 are spurious), **1/40** oracle-only gold hit,
**144/144** key words with conflicting role labels, **25/25** subjects under
multiple roles, **0/40** gold-free correct assertions. WordNet is available and
properly licensable; it simply does not contain the information TRA-K asks an
external ontology to supply.

This is an information deficit, not a tuning problem. The narrow next step is a
benchmark-design change, not more thresholds: either (a) define the role type as
part of the query input (the relation phrase is already present) and test
candidate-side typing with an independently authored single-type-per-word
lexicon, or (b) drop channel B and use channel A (ostensive teaching) only as the
declared non-deployable positive control. Until the key's type assignment stops
being item-relative, no independent snapshot can satisfy the spec's
channel-independence audit. Do not open a production PR or a TRA-K harness PR on
this channel.
