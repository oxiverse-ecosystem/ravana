# Compositional type/relation phrase parsing for TRA-K — scratch experiment

**Verdict: DO-NOT-IMPLEMENT in production now.** Existing paths already cover
every single-clause declarative that the merged `RAVANA_TRA_SLOTS` write/read
path needs. The measured scratch gain is confined to multi-clause and wider-NP
surfaces; the read side already consumes those facts once stamped. The
unmeasured prerequisite is a write-side provenance/cold-reload check, not a
new production parser and not a gate change.

## Scope and authority

- Worktree `C:/Users/Likhith/Documents/projects/ravana_parser`, branch
  `goal/type-phrase-parser`, head `b75a24e` at run time.
- STATIC + FAKE only: no engine boot, no `process_turn`, no `KnowledgeIntake`
  call, no store/network access, no edit under `ravana/src`, `scripts` or
  `tests`. Two production modules were imported read-only and hash-checked
  before and after (`chain_walker.py`, `schema_induction.py`).
- Production functions called **unchanged**: `ChainWalkerMixin._typed_slot_assertions`,
  `ChainWalkerMixin._asserted_relation_propositions`,
  `schema_induction.typed_slot_views_from_graph`,
  `schema_induction.slot_keyed_forced_choice`.
- No benchmark answer key: no item bank, `domain`, `far_domain`, `world_model`
  or gold target was imported, opened or supplied to any scorer. The 68 labels
  are authored in the script from the surface grammar defined below.
- The scratch parser is gated by `RAVANA_TRA_PHRASE=1` (default OFF). With the
  flag off it returns `[]`; production outputs are byte-identical with it on
  and off. No production flag was enabled.

## Brain-grounded design decision

Compositional phrase interpretation is implemented as **clause-wise binding of
relational roles into a typed slot**, with source provenance deciding whether
the slot is admissible at answer time.

- Pylkkänen, L. (2019). The neural basis of combinatory syntax and semantics.
  *Science*, 366(6461), 62–66. doi:10.1126/science.aax0050 — composition
  peaks in left anterior temporal lobe with later medial-prefrontal engagement;
  meaning combination is a fast, local operation, not whole-utterance lookup.
  → Bind within a clause; do not require the entire turn to be one template.
- Fedorenko, E., Scott, T. L., Brunner, P., Coon, W. G., Pritchett, B.,
  Schalk, G., & Kanwisher, N. (2016). Neural correlate of the construction of
  sentence meaning. *PNAS*, 113(41), E6256–E6262.
  doi:10.1073/pnas.1612132113 — γ-power builds monotonically over words for
  sentences but not word lists or Jabberwocky; the signal tracks constructed
  meaning, not syntax alone or word meaning alone. → Incremental clause-wise
  accumulation, with both arguments present before a fact is admitted.
- Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy.
  *Cognitive Science*, 7(2), 155–170. doi:10.1207/s15516709cog0702_3 —
  interpretation rules depend on syntactic roles, not domain content; relations
  map before attributes. → Source-independent vocabulary and canonical
  `(subject, relation, object)` roles; polarity inversion is role-preserving.
- Tse, D., Langston, R. F., Kakeyama, M., Bethus, I., Spooner, P. A., Wood,
  E. R., Witter, M. P., & Morris, R. G. M. (2007). Schemas and memory
  consolidation. *Science*, 316(5821), 76–82. doi:10.1126/science.1135935 —
  an existing schema assimilates a one-trial associate rapidly. → An `is_a`
  typed slot is a schema an unseen subject can be assimilated into.
- Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring.
  *Psychological Bulletin*, 114(1), 3–28. doi:10.1037/0033-2909.114.1.3 —
  memory for content and memory for its origin dissociate. → The reader admits
  a slot only through provenance, never through label similarity.

These citations motivate the design; **this experiment does not validate a
neural analogue**. They are recorded here and in the commit body as required
by the RAVANA design rule, not as evidence for the regex grammar.

## Label specification (hand-authored, not an answer key)

A sentence is expected to yield a fact iff every clause is declarative, has no
question/hedge/negation marker, and matches one of:

| Surface | Canonical fact |
|---|---|
| `X is a/an [kind\|type\|sort\|form of] Y` | `IS_A(X, Y)` |
| `X is part of Y` / `Y contains X` / `Y includes X` / `X is a component of Y` | `PART_OF(X, Y)` |
| `X comes/happens before Y` / `Y comes/happens after X` / `Y follows X` | `PRECEDES(X, Y)` |
| `X resembles Y` / `X is similar/akin to Y` | `RESEMBLES(X, Y)` |

Clauses split on `. ; , newline` and ` and `. NP span is 1–3 subject words and
1–4 object words (determiners stripped). Confidence is `0.9` for a
single-clause, short-NP fact and `0.8` for a wider-NP or multi-clause fact.
Vocabulary is neutral and source-independent (`lantern`, `mosaic`, `petal`,
`beacon`, `marsh`, …); no benchmark domain or world fact is used.

**Pools.** 68 probes: 52 primary (authored alongside the grammar) and 16
sealed (`H01`–`H16`, added *after* the parser was fixed, using fresh vocabulary
and surface forms absent from the primary set: `is a sort of`, `is a form of`,
`is akin to`, `includes`, `is a component of`, `comes after`, `happens before`).
A 1.00 on the primary pool is **spec agreement**, not independent
generalisation; the sealed pool is the only surface-generalisation probe.

## Measured results

Facts are counted as multisets of `(relation, subject, object)`.

| Arm | TP | FP | FN | Precision | Recall | F1 | Item exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Production parsers, all 68 | 36 | 0 | 19 | 1.0000 | 0.6545 | 0.7912 | 53/68 (0.7794) |
| Production, primary 52 | 24 | 0 | 16 | 1.0000 | 0.6000 | 0.7500 | 40/52 (0.7692) |
| Production, sealed 16 | 12 | 0 | 3 | 1.0000 | 0.8000 | 0.8889 | 13/16 (0.8125) |
| Scratch extended, all 68 | 55 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 68/68 |
| Scratch extended, primary 52 | 40 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 52/52 |
| Scratch extended, sealed 16 | 15 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 | 16/16 |

The extended 1.00 is a correctly implemented deterministic grammar over
authored surfaces; it must not be read as semantic understanding or as a
deployable capability gain.

### Where the existing path actually stops

- All 8 short `is a`/`is a kind of` sentences are accepted. All 6 wider-NP
  `is_a` sentences (`civic event is a structured whole.` → `IS_A(civic event,
  structured whole)`, `tremor is a sudden ground motion.`, …) are refused by
  the anchored one-token-subject / ≤2-token-object regex.
- The relation parser splits clauses: 6/8 single-clause relation facts and
  6/14 multi-clause facts are recovered. It has no `IS_A` family, so an
  `is_a` + relation multi-clause turn always loses the type facts.
- **Over-broad negation regex:** `_ASSERTION_NEGATION_RE` uses
  `[a-z]n['’]?t\b`, which matches the `ent` in `event` and `component`.
  `event is part of ceremony.` and `engine is a component of turbine.` are
  therefore refused (false negatives), while the typed parser's
  word-boundary `_TYPED_REFUSE_RE` correctly accepts `event is a structured
  whole.`. Two different negation predicates exist in the same production
  path; only one is word-boundary-safe.
- Permutation: direction is preserved on the 3 forms the production path
  accepts (`tapestry resembles mosaic.`, `blossom is part of petal.`,
  `dawn happens after night.`), but `light source is a lantern.` is refused
  by the anchored typed regex. Production permutation pass 3/4; scratch 4/4.
- `PRECEDES` canonicalisation across `comes before` / `follows` /
  `happens after` passes on both arms.
- Controls: 22/22 negation, hedge, question and malformed items are refused
  by **both** arms. No item is rejected by extended but accepted by
  production; 15 items are fully correct only under the extended parser.

### READ side — fake typed views, unmodified production reader

15/15 cases pass with the production `slot_keyed_forced_choice` over views
from `typed_slot_views_from_graph`:

- `D00` (fact taken **from the unmodified production parser**, not scratch):
  teaching `lantern is a light source.` → `pick='light source'` with four
  disjoint typed options; option-order permutation gives the same pick.
- `D08`/`D09`/`D10`/`D14`: facts available only from the scratch parser
  (multiword subject, multi-clause, 0.8 confidence) are also read correctly
  once stamped as an `is_a` edge with `type_label`.
- Fail-closed `None` with reason `no_trusted_typed_slot_above_floor` for
  an untaught subject, disjoint options, unrelated subject, untrusted edge,
  wrong `relation_type`, missing `type_label`, and negated/hedged teaching
  surfaces.
- `D13`: a relation fact written production-style (`part_of`, no
  `type_label`) yields `None`. The slot reader is IS_A-specific: phrase
  coverage alone does not make relation facts queryable through it.

## Instrument validation

All pre-registered gates pass: flag-off production byte-identical, flag-off
extended inert, no engine instance, no production file hash change, no weight
store write, no answer-key import, extended controls/permutation/canonicalisation
pass, read-side controls pass. Every probe's input and both parsers' outputs
are logged in `scratch/type_phrase_parser_run.log`. The measured production
false negatives (R07, H04, P01) show the fact-level metric can move; it is not
a saturated oracle count.

## EVIDENCE

1. Existing parse primitives accept every short declarative `is a` and the
   three non-causal relation families, and they refuse all 22 controls.
2. Existing parse primitives produce **0** facts for 15/68 probes admitted by
   the scratch grammar; the loss is multi-clause `IS_A`, wider NPs, two
   over-broad-negation false negatives, and one multiword-subject permutation.
3. A deterministic scratch parser recovers all 55 expected facts with 0 FP on
   both pools, including the sealed surface forms, at 52 authored primary +
   16 sealed probes.
4. The unmodified typed-slot read path is sufficient for taught `is_a` facts
   from **either** parser and fail-closes on all five adversarial controls.
5. That same read path cannot consume relation-family facts: `type_label`
   stamping is IS_A-only.

## INFERENCE (not measured here)

1. That regex spec agreement implies human-like compositional understanding.
2. That multi-clause facts written through `KnowledgeIntake` would keep every
   stamp after a cold reload — no intake call or reload was performed.
3. That any of this improves an answer front, TRA scorecard or gate — no
   engine, harness or benchmark was run.
4. That the cited brain mechanisms map one-to-one onto this grammar.
5. That the sealed pool transfers to spontaneous prose or real-world words.

## Recommendation

**Do not implement in production now.** The existing merged path already
suffices for the single-clause teaching surfaces it documents. The next work
item is a *write-side* validation: parse a multi-clause turn, push each fact
through the same `Fact` + user `Provenance` intake path, save, cold-reload and
census the stamped edges. Until that is measured, adding a production parser
would create facts whose retention is unverified.

If revisited, the only acceptable shape is: scratch-first behind
`RAVANA_TRA_PHRASE=1` (default OFF), identical outputs with the flag off, same
intake and provenance, no new trust predicate, no threshold or C10 gate
change, and an A/B with positive and negative endpoints on identical state.
The two negation-regex false negatives are a separate production-file fix
requiring their own tests; they are not a reason to loosen a gate.

## Reproduce

```bash
cd C:/Users/Likhith/Documents/projects/ravana_parser
PYTHONHASHSEED=0 PYTHONPATH="C:/Users/Likhith/Documents/projects/ravana_parser/ravana/src;C:/Users/Likhith/Documents/projects/ravana_parser/ravana_ml/src;C:/Users/Likhith/Documents/projects/ravana_parser" \
  C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/type_phrase_parser_experiment.py > scratch/type_phrase_parser_run.log 2>&1
```

Artifacts: `scratch/type_phrase_parser_experiment.py`,
`scratch/type_phrase_parser_results.json`, `scratch/type_phrase_parser_run.log`.
