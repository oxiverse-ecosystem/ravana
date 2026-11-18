# Productive use of a newly learned word: brain evidence and a mechanism for RAVANA's symbolic realiser

**Date:** 2026-11-19
**Scope:** Evidence-first answer to one question: RAVANA learns a novel word from one
turn and recalls its definition correctly (3/3 taught, 1/1 untaught declined), but
`use the word zorblat in a sentence` returns the stored definition
(*"It seems to me that zorblat is a small blue device that measures air pressure"*).
What does the human brain do that turns a stored meaning into a **new sentence**, and
what is the corresponding mechanism in `ravana/src/ravana/language/`?

**Method note:** every empirical claim below carries an inline citation `[n]` into
the ledger-backed Sources list. Claims from model knowledge are marked `[unverified]`.
No code is changed by this document.

**Companion docs (do not re-read):** `docs/BRAIN_NOVEL_TERM_LEARNING.md` (fast mapping,
extended mapping, ATL hub, consolidation), `docs/BRAIN_NOVEL_SUBJECT_GROUNDING.md`
(commitment before form), `docs/DEFINITION_PROPOSITION.md` (reading a relation off a
definition string), `docs/CAPABILITY_LEARNED_CONSTRUCTIONS_PERSIST_REUSE.md` (M16
construction inventory), `docs/LANGUAGE_LEARNING_CONFIRMED.md` (the measured failure).

---

## 0. Executive summary

The brain does not "look up a word and put it in a sentence" by replaying the lookup.
Three separable findings say so:

1. **Definition recall and productive use are different knowledge states.** Remembering
   a text means reproducing it; learning from it means using the information "in other
   ways, not just for reproduction" [18]. Lexical competence has a distinct
   *receptive → productive use* dimension [15], and controlled productive ability is
   measurably a different test from recognition [16]. At the neural level, memorized
   lexical items depend on temporal-lobe declarative memory while composition into
   larger phrases and sentences depends on frontal/basal-ganglia procedural circuitry
   [17]. A definition echoed back is the declarative route; a fresh sentence is the
   compositional one.
2. **A one-turn word is a placeholder, not a finished entry.** Carey and Bartlett's
   "chromium" study showed that the correct grammatical category is fast-mapped while
   referential content remains partial; Carey's own retrospective says only 2 of 34
   children fully mapped the word by the end of testing [1][2]. McMurray, Horst and
   Samuelson's model and data place referent selection and slow associative learning in
   the same integrated system: a single mapping event is not a usable lexical entry
   [3]. Fast mapping of a *phrase* is possible too — children map a novel phrasal form
   to a novel meaning after a short exposure [4] — which is why the mechanism below
   binds a **construction**, not a definition string.
3. **The unit of deployment is the construction, not the word.** Constructions are
   form–meaning pairings that carry meaning independently of the verb [7][8], and
   sentence meaning is predicted by the argument-structure construction above and
   beyond the verb [9]. Children learn these pairings from skewed input, with one
   high-frequency exemplar facilitating the generalization [5], and children use the
   syntactic frame itself to infer a novel verb's meaning [19][20]. The construction
   inventory is built from use [10]. Hippocampal recombination supplies the other half:
   episodic memory is *constructive*, flexibly extracting and recombining elements of
   past experience into scenarios that never occurred in that form [11][12]. Bilateral
   anterior temporal lobe is the amodal hub that binds features into a coherent concept
   supporting generalization, and its atrophy yields fluent but empty speech [13][14].

**One-sentence mechanism:** keep the definition as one proposition among several,
decompose it into a feature bundle, bind the new word into a **non-definitional**
construction with a **fresh graph-supported target**, realize it with the existing
symbolic `SurfaceRealizer`, and reject any output that still echoes the definition.

**RAVANA's current failure is exactly this gap.** `chain_walker.py:2542-2571` handles any
subject in `_definitions` by calling
`bind_to_sentence(subject=subject, relation="semantic", target=defn, ...)` and realizing
the result in `discourse_type="explain"` — the stored definition string *is* the object
slot, so the output cannot be anything but the definition. Subject grounding was fixed
separately in `engine_generation.py:2102-2104`; this document does not revisit it.

---

## 1. Why the current output is a recall, not a use

| observation | brain reading | code reading |
|---|---|---|
| Output contains the definition nearly verbatim | textbase reproduction, not situation-model use [18] | `target=defn` at `chain_walker.py:2553` |
| Predicate is the definitional copula `is` | definition's `is_a` frame reused unchanged | `discourse_type="explain"`, `relation="semantic"` at `chain_walker.py:2552/2558` |
| No new participant appears | no recombination of stored elements into a novel scenario [11] | no second frame bound; association sentences are optional and use `target=label` of the *subject*, not of the word's features |
| Recall of the same word still succeeds | declarative store is intact [17] | the definition store must **not** be deleted |

The important negative result: this is **not** a missing-word problem. A word absent
from the store returns an honest calibrated gap (`docs/LANGUAGE_LEARNING_CONFIRMED.md`),
so the store is doing its job. The missing component is a *composer* downstream of it.

---

## 2. The evidence in detail

### 2.1 Fast mapping gives a category and a feature bundle, not a sentence

Carey and Bartlett (1978) introduced "chromium" contrastively ("the chromium tray, not
the blue one") and found durable but incomplete learning; Carey (2010) reports that
all children who were affected learned that chromium was *a colour word*, while only
2 of 34 fully identified olive green [1][2]. The lesson for a novel noun taught as
"a zorblat is a small blue device that measures air pressure" is that the system has
acquired (i) the **category** (`device`), (ii) **attributes** (`small`, `blue`), and
(iii) a **function/relation** (`measures air pressure`) — not a sentence to replay.

McMurray, Horst and Samuelson (2012, *Psychological Review*) model word learning as an
interaction between online referent selection and **slow associative learning**, i.e.
the provisional mapping and its consolidation are one continuing process, not two
switches [3]. Casenhiser and Goldberg (2005, *Developmental Science*) show the same for
phrasal form–meaning pairs: 5–7-year-olds mapped a novel verb in a novel construction
to new scenes after only a short exposure [4]. So the "fragile entry" is not useless —
it is usable **if a construction is available to bind it into**.

### 2.2 A coherent concept, not a stored string

The hub-and-spoke account holds that modality-specific features are distilled by the
bilateral anterior temporal lobe into amodal representations that support generalisation
and inference [13]; Lambon Ralph, Jefferies, Patterson and Rogers (2017, *Nature
Reviews Neuroscience*) make the same case computationally [14]. Semantic dementia —
fluent, grammatical, empty speech with preserved syntax — is what remains when the
hub degenerates while the formulator is intact [13]. RAVANA's failure is the inverse
shape: the concept *is* stored, but the formulator is handed the verbatim string rather
than a proposition to combine. The correct analogue is not "make the formulator
fancier" but "give it a **different proposition**".

### 2.3 Constructions carry the meaning the word alone does not

Goldberg (1995) argues that argument-structure patterns are form–meaning pairings whose
meaning is not read off the verb [7]; Goldberg (2006) extends this to a usage-based
network in which full generalizations and item-specific patterns coexist [8]. Bencini
and Goldberg (2000, *Journal of Memory and Language*) provide the behavioural proof:
participants sorting sentences by meaning sorted by **construction**, not only by verb
[9]. Goldberg, Casenhiser and Sethuraman (2004, *Cognitive Linguistics*) show that the
constructional semantics can be learned from skewed input, with a single frequent
exemplar facilitating the generalization [5]; Goldberg (2002, *Cognitive Linguistics*)
argues each surface pattern should be treated on its own terms [6]. Tomasello's
usage-based account makes the same point developmentally: grammar emerges from
repeated symbol sequences and item-based constructions [10].

Syntactic framing also supplies meaning: Gleitman (1990, *Language Acquisition*)
proposes that children infer verb meaning from the syntactic environments the verb
appears in [19], and Naigles (1990, *Journal of Child Language*) demonstrates that
2-year-olds map a novel verb to causative versus non-causative events according to
whether it appeared in a transitive or intransitive frame [20].

**Design rule:** the deployment step must choose a *frame* whose meaning is not the
definitional one. Replacing the copula frame with a `used_for`, `has_property`,
`part_of`, `capable_of`, locative or causal frame changes the proposition without
changing a single stored fact — exactly what Bencini and Goldberg predict.

### 2.4 Recombination is constructive, not reproductive

Schacter and Addis (2007, *Philosophical Transactions of the Royal Society B*) argue
that episodic memory is constructive because it must "flexibly extract, recombine and
reassemble" elements of past experience to simulate events that never occurred in that
form [11]; Schacter, Addis and Buckner (2007, *Nature Reviews Neuroscience*) extend
this to the "prospective brain" [12]. This is the neural counterpart of *using* a newly
learned word: the stored elements (category, attributes, function, graph neighbours)
are recombined under a new frame, not replayed.

### 2.5 Consolidation bounds the claim

Sleep-associated lexical integration (lexical competition) emerges only after a
consolidation interval, not immediately [21]. This document does **not** claim that a
single-turn word is fully integrated; it claims that a *limited, construction-bound*
deployment is possible from the fast-mapped bundle, which is what children's novel-phrase
mapping [4] supports. A system should therefore track whether a deployment was built
from a fragile bundle or from a consolidated entry, and hedge accordingly.

---

## 3. Observable criteria: novel composition vs. regurgitated definition

Five criteria. Each is machine-checkable on the emitted string plus the plan that
produced it. **All five must pass** before an output counts as productive use.

| # | Criterion | Operational test | Evidence |
|---|---|---|---|
| **C1** | **Construction change.** The matrix predicate is not the definitional copula plus the definition's head noun, and the realized construction differs from the definition's `is_a`/SVO frame. | Parse the emitted clause. Assert `relation_type ∉ {is_a, isa}` for the top-level frame **or** `verb ∉ {is, are}` when the complement is the definition head noun; assert selected merge variant ≠ the definitional `svo` frame. | [7][8][9][6] |
| **C2** | **Fresh context.** At least one content word in the output was not in the teaching sentence/definition, and the object/target is a **distinct graph concept**, not the definition string. | `content_tokens(out) \ content_tokens(definition)` has ≥2 items; `target ∈ graph.concepts`; `target != defn`; target is a neighbour of the subject or of a feature slot. | [11][12][18] |
| **C3** | **Echo rejection.** The output is not a verbatim or near-verbatim paraphrase of the definition. | Definition string is not a substring of the output; token Jaccard(out, definition) < 0.45 **and** longest-common-subsequence ratio < 0.5. Thresholds are engineering parameters to calibrate on held-out pairs, not brain-derived constants. | [18][15][16] |
| **C4** | **New participant / role reassignment.** The sentence introduces a second event participant drawn from prior knowledge (agent, instrument, location, or a typed attribute) rather than only restating the word's definition. | The realized `SyntacticFrame` has a non-empty object/target that is not the definition NP, and the frame binds ≥2 distinct content concepts besides the subject. | [7][19][20][13] |
| **C5** | **Cross-frame productivity.** The same word is deployable in ≥2 distinct constructions across two use requests without echoing the definition. | Log the construction ids used for the word; assert `len(set(ids)) >= 2` after two requests. A single novel sentence can be a one-off; productive knowledge generalises. | [5][10][9] |

**Verdict on the current output.** *"It seems to me that zorblat is a small blue device
that measures air pressure."* fails **C1** (copula `is`, definitional frame), **C2** (zero
novel content tokens), **C3** (the definition is a verbatim substring), **C4** (no new
participant), and **C5** (not demonstrated). It is a correct *recall* and an incorrect
*use*.

**Measurement discipline.** Write the probes and the rubric in separate passes: read
the actual outputs before fixing the similarity thresholds, otherwise the instrument
scores RAVANA's own register as success. Report per-criterion counts, not just an
aggregate, and state the denominator. Do not build the gold set from the same pass that
chose the mechanism.

---

## 4. Concrete mechanism in RAVANA's existing symbolic realiser

**Name:** feature-bundle construction deployment (FBCD).
**Constraint:** no new neural component, no retraining, no changes to the subject
normalizer in `engine_generation.py:2102-2104`. All primitives already exist in
`ravana/src/ravana/language/`; only the *plan* handed to them changes.

**D0 — input.** Subject is already resolved to `zorblat`. Keep the definition store
intact: it is the declarative route and must continue to answer `what is X?`.

**D1 — decompose, do not copy.**
Read `_definitions[subject]` and parse it into a
`FeatureBundle(category, attributes, function_relation, source_turn)`. Reuse the
copula/head-noun parser from `docs/DEFINITION_PROPOSITION.md` (right-headed NP, POS
map via `_concept_pos`) but keep **all** slots, not only the `is_a` proposition. For the
example: `category=device`, `attributes=[small, blue]`,
`function_relation=(measures, air pressure)`.

**D2 — choose a non-definitional construction.**
Select a `SyntacticFrame` template whose `relation_type` is in
`{has_property, part_of, used_for, capable_of, causal, temporal}` and whose surface
construction is not the definitional one. Sources of candidates already in the repo:
- `SyntacticCellAssembly.learned_frames` and `find_construction` /
  `allocate_construction` / `reinforce_construction`
  (`syntactic_cell_assembly.py:577-622`) — the usage-based inventory, persisted via
  `get_state`/`set_state`;
- `SurfaceRealizer.MERGE_VARIANTS` (`surface_realizer.py:153-164`) and
  `_select_variant` (`:729-787`) for non-SVO merge orders;
- the typed relations the realiser already honours at
  `surface_realizer.py:444-446` (`is_a`, `has_property`, `part_of`, `used_for`,
  `capable_of`).
Veto `is_a`/`isa` and any frame whose object is the raw definition string. This is the
load-bearing step: it changes the *proposition*, not the wording.

**D3 — bind a fresh, graph-supported target.**
Resolve a target from the feature bundle or the subject's graph neighbours (e.g.
`air pressure` → a known neighbour, or a known agent/location concept), with the
same guard `docs/DEFINITION_PROPOSITION.md` established — **the target must be a graph
concept**, otherwise the system asserts something the graph cannot support. Reject a
target identical to the definition string or composed only of definition tokens.
Call `SyntacticCellAssembly.bind_to_sentence(subject, relation, target, pos_map)`
(`syntactic_cell_assembly.py:207-287`); it supplies articles via `_determine_article`
and a verb phrase via `_pick_verb_phrase` → `VerbLexicon.select_verb`, so the chosen
construction and the chosen verb are kept consistent.

**D4 — realize in a fresh discourse context.**
Call `SurfaceRealizer.realize(frame, DiscourseState(...), ...)` with a non-definitional
`discourse_type` (e.g. `elaborate`/`connect`), not the definitional `explain` mode of
`chain_walker.py:2558`. Keep the typed-relation override (`:444-446`) and the
completeness monitor `_utterance_complete` (`:633-641`, `:696-727`): a realization that
fails the predicate check returns `""` and must not fall back to the definition.

**D5 — echo gate, with retry and fail-closed.**
Before emission, evaluate C1–C4 against the stored definition. On failure, retry with
the next construction/target pair from D2/D3 (relaxing only the *choice*, never the
echo criterion). If all candidates fail, return `""` so the existing honest-uncertainty
fallback handles the turn — **never silently fall through to the definition**. The
definition is the correct answer to `what is X?`, not to `use X in a sentence`.
This is a generation gate, not a subject-routing fix.

**D6 — reinforce on success.**
On acceptance, call `allocate_construction` / `reinforce_construction` with the new
frame and `learn_from_feedback(frame, True)`. One successful use allocates a
construction (M16); frequency then biases later selection, matching the usage-based
account [5][10]. A failed deployment weakens it. This makes the path history-dependent
rather than a fixed rewrite.

**Worked shape, not a promised string.** With the bundle above and a `used_for` frame
bound to a known agent, the plan is
`(subject=zorblat, relation=used_for, target=<known graph concept>)`, realized as a
new clause that deploys the word; the exact surface form is whatever the existing
realiser produces and must pass C1–C5 before emission. No sentence is authored here.

**Invariants.**
- The definition route is never deleted; recall still works.
- Targets are graph concepts, so added claims remain graph-supported by construction.
- The completeness monitor and the metacognitive withholding path stay upstream of
  emission; FBCD adds a plan, it does not weaken a guard.
- Thresholds in C2/C3 are to be swept and measured, not asserted as brain constants.

---

## 5. Falsifiers

1. If disabling the feature-bundle decomposition leaves outputs unchanged, the bundle
   is decorative — the definition string is still being copied downstream.
2. If the echo gate passes but C1 fails on most outputs, the construction veto in D2 is
   not reaching the frame that `bind_to_sentence` actually returns.
3. If only one construction id ever fires across many use requests, C5 fails and the
   usage-based reinforcement step has not changed selection.
4. If ablating the graph-neighbour requirement in D3 leaves unsupported-assertion counts
   unchanged, the "graph-supported by construction" claim is false for this route.
5. If a high-confidence, fully consolidated word still cannot be deployed in two frames,
   the bottleneck is the realiser's frame inventory, not the definition store.

---

## 6. Honest flags

- Construction grammar is a linguistic/cognitive account, not a localized neural
  circuit; the neural evidence for construction-level units is suggestive rather than
  definitive. Cite [7][8][9] as theory plus behavioural evidence, not as fMRI.
- The ATL hub account is contested at the margins; it is the best-supported reading,
  not a settled fact [13][14].
- "One exposure teaches a word" is a misreading of fast mapping: the correct claim is
  *a durable but partial entry*, with the category often right and the referent wrong
  [1][2][3].
- Consolidation is not complete after one night [21]; FBCD is a limited deployment
  route from a fast-mapped bundle, not a claim of full native-like competence.
- No direct neuroscience study was found for the exact task "use a word taught in one
  conversational turn in a fresh sentence". The mechanism is an extrapolation from
  construction learning, recombination and the productive/receptive dissociation, and
  is labelled as such.

---

## References (full citations)

**Empirical / peer-reviewed journal articles**

- **Carey, S., & Bartlett, E.** (1978). Acquiring a single new word. *Papers and Reports
  on Child Language Development* (Stanford University), 15, 17–29.
- **McMurray, B., Horst, J. S., & Samuelson, L. K.** (2012). Word learning emerges from
  the interaction of online referent selection and slow associative learning.
  *Psychological Review*, 119(4), 831–877. doi:10.1037/a0029872
- **Casenhiser, D., & Goldberg, A. E.** (2005). Fast mapping between a phrasal form and
  meaning. *Developmental Science*, 8(6), 500–508. doi:10.1111/j.1467-7687.2005.00441.x
- **Goldberg, A. E., Casenhiser, D. M., & Sethuraman, N.** (2004). Learning argument
  structure generalizations. *Cognitive Linguistics*, 15(3), 289–316.
  doi:10.1515/cogl.2004.011
- **Goldberg, A. E.** (2002). Surface generalizations: An alternative to alternations.
  *Cognitive Linguistics*, 13(4), 327–356. doi:10.1515/cogl.2002.022
- **Bencini, G. M. L., & Goldberg, A. E.** (2000). The contribution of argument structure
  constructions to sentence meaning. *Journal of Memory and Language*, 43(4), 640–651.
  doi:10.1006/jmla.2000.2757
- **Schacter, D. L., & Addis, D. R.** (2007). The cognitive neuroscience of constructive
  memory: Remembering the past and imagining the future. *Philosophical Transactions of
  the Royal Society B*, 362(1481), 773–786. doi:10.1098/rstb.2007.2087
- **Schacter, D. L., Addis, D. R., & Buckner, R. L.** (2007). Remembering the past to
  imagine the future: The prospective brain. *Nature Reviews Neuroscience*, 8(9),
  657–661. doi:10.1038/nrn2213
- **Patterson, K., Nestor, P. J., & Rogers, T. T.** (2007). Where do you know what you
  know? The representation of semantic knowledge in the human brain. *Nature Reviews
  Neuroscience*, 8, 976–987. doi:10.1038/nrn2277
- **Lambon Ralph, M. A., Jefferies, E., Patterson, K., & Rogers, T. T.** (2017). The
  neural and computational bases of semantic cognition. *Nature Reviews Neuroscience*,
  18(1), 42–55. doi:10.1038/nrn.2016.150
- **Henriksen, B.** (1999). Three dimensions of vocabulary development. *Studies in
  Second Language Acquisition*, 21(2), 303–317. doi:10.1017/S0272263199002089
- **Laufer, B., & Nation, P.** (1999). A vocabulary-size test of controlled productive
  ability. *Language Testing*, 16(1), 33–51. doi:10.1177/026553229901600103
- **Ullman, M. T.** (2001). A neurocognitive perspective on language: The
  declarative/procedural model. *Nature Reviews Neuroscience*, 2(10), 717–726.
  doi:10.1038/35094573
- **Kintsch, W.** (1994). Text comprehension, memory, and learning. *American
  Psychologist*, 49(4), 294–303.
- **Gleitman, L. R.** (1990). The structural sources of verb meanings. *Language
  Acquisition*, 1(1), 3–55. doi:10.1207/s15327817la0101_2
- **Naigles, L.** (1990). Children use syntax to learn verb meanings. *Journal of Child
  Language*, 17(2), 357–374. doi:10.1017/S0305000900013817
- **Dumay, N., & Gaskell, M. G.** (2007). Sleep-associated changes in the mental
  representation of spoken words. *Psychological Science*, 18(1), 35–39.
  doi:10.1111/j.1467-9280.2007.01845.x

**Monographs (theory, not journal articles)**

- **Goldberg, A. E.** (1995). *Constructions: A Construction Grammar Approach to
  Argument Structure*. University of Chicago Press.
- **Goldberg, A. E.** (2006). *Constructions at Work: The Nature of Generalization in
  Language*. Oxford University Press.
- **Tomasello, M.** (2003). *Constructing a Language: A Usage-Based Theory of Language
  Acquisition*. Harvard University Press.

---

## Sources

[1] https://languagelog.ldc.upenn.edu/myl/Acquiring_a_Single_New_Word.pdf
[2] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3101498
[3] https://pmc.ncbi.nlm.nih.gov/articles/PMC3632668
[4] https://eric.ed.gov/?id=EJ850198
[5] https://collaborate.princeton.edu/en/publications/learning-argument-structure-generalizations
[6] https://collaborate.princeton.edu/en/publications/surface-generalizations-an-alternative-to-alternations
[7] https://press.uchicago.edu/ucp/books/book/chicago/C/bo3683810.html
[8] https://academic.oup.com/book/32920/chapter/276901889
[9] https://www.sciencedirect.com/science/article/pii/S0749596X00927578
[10] https://www.hup.harvard.edu/books/9780674044395
[11] https://pmc.ncbi.nlm.nih.gov/articles/PMC2429996
[12] https://www.nature.com/articles/nrn2213
[13] https://www.nature.com/articles/nrn2277
[14] https://www.nature.com/articles/nrn.2016.150
[15] https://eric.ed.gov/?id=EJ655710
[16] https://journals.sagepub.com/doi/10.1177/026553229901600103
[17] https://www.nature.com/articles/35094573
[18] https://eric.ed.gov/?id=EJ484395
[19] https://www.coli.uni-saarland.de/~masta/WS15/Gleitman1990.pdf
[20] https://pubmed.ncbi.nlm.nih.gov/2380274
[21] https://journals.sagepub.com/doi/abs/10.1111/j.1467-9280.2007.01845.x
