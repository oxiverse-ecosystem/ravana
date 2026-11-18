# Cue-sensitive construction selection: why a repeat counter cannot select a construction, and what the brain does instead

**Date:** 2026-11-19
**Scope:** The remaining limit after `BRAIN_CROSS_FRAME_PRODUCTIVITY.md`. After teaching "a zorblat is a small blue device that measures air pressure", RAVANA now emits two alternating sentences, but **all phrasings get the same two sentences**: `what does X do?`, `what is X part of?` and `where is X?` select by `self._productive_use_counts[subject] % 2`, not by what was asked. Frame choice must become sensitive to the question cue and the discourse context. This document answers: what does the human brain use to map a cue to a construction, and what is the corresponding mechanism in RAVANA's existing symbolic inventory?

**Method note:** every empirical claim carries an inline author–year citation plus a ledger-backed numbered citation `[n]`; the ledger maps each number to a retrieved URL in the Sources block. Peer-reviewed journal articles, book chapters and monographs are labelled separately and are not conflated. Claims from model knowledge are `[unverified]`. **No code is changed by this document, and no existing document is edited.**

**Companion docs (do not re-read):** `docs/BRAIN_CROSS_FRAME_PRODUCTIVITY.md` (one frame → repertoire; scored competition with recency fatigue, SCC-RF, and criteria O1–O5), `docs/BRAIN_PRODUCTIVE_VOCABULARY.md` (definition → one construction; FBCD feature-bundle decomposition), `docs/PRODUCTIVE_VOCAB_CODEMAP.md` (emitting path), `docs/LANGUAGE_LEARNING_CONFIRMED.md`, `docs/CAPABILITY_LEARNED_CONSTRUCTIONS_PERSIST_REUSE.md` (M16/M16b construction inventory). This document is the **next** gap: the selection criterion inside the repertoire.

---

## 0. Executive summary

Three findings, kept separate:

1. **A question is not a "say anything" trigger; it sets an information-structure target that the answer must meet.** In the question-under-discussion (QUD) account, accepting a question establishes a discourse goal, and a response is congruent only if the focus of the answer corresponds to the wh-phrase of the question (Roberts 2004, in Horn & Ward eds., *Handbook of Contemporary Pragmatic Theory* [10] — chapter, not journal article). Violating that congruence modulates the brain response to the answer within 200–500 ms (Cowles & Kluender 2007, *Brain and Language*, 102(3), 228–242 [7]); a preceding wh-question about the VP versus the object NP shifts the structural interpretation of an ambiguous sentence in a systematic, replicated way (Carlson & Potter 2022, *Language and Speech*, 65(2), 491–512 [8]); and across fourteen languages the *form* of an answer is systematically shaped by the form of the question (Enfield, Stivers, Brown, et al. 2019, *Journal of Linguistics*, 55(2), 277–304 [9]). A question therefore carries a **constructional cue** that selects an answer form, not merely a request for content.

2. **In the brain, form choice is a competition resolved by cue strength, and cue strength is cue validity.** The Competition Model holds that in production *forms compete to express meanings*, and that the winning form's strength is learned from the reliability and availability of the form–function mapping (MacWhinney 2021, "The Competition Model: Past and Future", in *A Life in Cognition*, Springer, pp. 3–16 [3] — chapter; MacWhinney, Bates & Kliegl 1984, *Journal of Verbal Learning and Verbal Behavior*, 23, 127–150 [1]; Bates, McNew, MacWhinney, Devescovi & Smith 1982, *Cognition*, 11, 245–299 [2]). Cue validity is the product of availability and reliability [1][3]. Crucially, the construction itself is a valid predictor of meaning: corpus evidence finds construction cue validity at least as high as verb cue validity, with higher category validity (Goldberg, Casenhiser & Sethuraman 2005, *Journal of Child Language*, 32(2), 407–426 [4]), and speakers sort sentences by argument-structure construction, not only by verb (Bencini & Goldberg 2000, *Journal of Memory and Language*, 43(4), 640–651 [5]). Selecting a frame from its cue is therefore the experimentally supported direction of mapping, not an engineering shortcut.

3. **The selection operation is LIFG/corticostriatal selection among competitors, and the cue-to-item mapping is content-addressable cue-based retrieval.** Left IFG activation tracks selection among competing alternatives rather than semantic retrieval *per se* (Thompson-Schill, D'Esposito, Aguirre & Farah 1997, *PNAS*, 94(26), 14792–14797 [13]), and left-IFG lesions impair only the high-competition cases, establishing necessity for selection and not for retrieval (Thompson-Schill, Swick, Farah, D'Esposito, Kan & Knight 1998, *PNAS*, 95(26), 15855–15860 [14]). Corticostriatal circuits resolve competition by enhancing a target and suppressing alternatives under conflict/uncertainty (Copland, Brownsett, Iyer & Angwin 2021, *Neuropsychology Review*, 31, 472–494 [15]). At the memory level, cues spread activation to all matching items and retrieval probability is proportional to activation, with similarity-based interference when several items match (Lewis & Vasishth 2005, *Cognitive Science*, 29(3), 375–419 [6]). A per-subject repeat counter implements none of this: it ignores the cue, uses recency as the *only* signal, and cannot express cue validity at all.

**One-sentence mechanism:** replace `_productive_use_counts[subject] % 2` with a deterministic **cue-validity-scored competition** over the existing relation/construction inventory, where each candidate's score combines its persisted base strength, its reliability under the question cue, and a recency/givenness term demoted to a tie-break; select by argmax with a stable tie-break, bind a graph-supported target, realize through the existing realiser, and reinforce the winning mapping.

Name: **Cue-Validity Construction Competition (CVCC)**. It is a refinement of the companion's SCC-RF: it keeps SCC-RF's admissible-set, grounding and persistence machinery, but replaces "alternate on every identical request" with "select by cue validity", and demotes fatigue from selection driver to tie-break. No new neural component, no retraining, no RNG.

---

## 1. Code-level diagnosis (read-only inspection of the current source)

| observed behaviour | current code path | why it is the wrong model |
|---|---|---|
| `what does X do?` → `Zorblat measures air pressure.` | `_productive_use_response`, `response_gen.py:7480–7541`; relative-clause extraction at `:7500–7509`; frame index `_idx = int(_counts.get(subject.lower(),0)) % 2` at `:7517–7521`; emission at `:7523–7534`. | The extracted predicate is the *same functional clause* in both frames. Frame 0 is `f"{subject} {predicate}."`; frame 1 is `f"A {_head} called {subject} {predicate}."`. The two options differ only in surface packaging (bare predication vs appositive naming), not in constructional semantics. There is no `part_of` or locative candidate at all. |
| `what is X part of?` / `where is X?` → the same two strings | `is_productive_use_request` (`response_gen.py:187–189`) detects the *directive* with `_PRODUCTIVE_USE_RE` (`:180–184`) but never inspects the question cue; no branch reads the wh-word or preposition. | The cue is discarded before selection. With a fresh counter state all three questions take frame 0; with a warm counter state the assignment is a function of request ordinal, not of the cue. |
| Ten surface variants | `SurfaceRealizer.MERGE_VARIANTS`, `surface_realizer.py:153–164` (`svo`, `left_dislocation`, `pseudo_cleft`, `topic_fronting_for`, `existential`, `cleft`, `as_for_topic`, `adverb_fronted`, `svo_emphatic`, `svo_causal`). | These are tree-merge orders, i.e. *focus/topic realizations*, not constructional relations. They are useful **after** a relation is chosen (a `cleft`/`pseudo_cleft` can mark a cued focus; `as_for_topic` can mark a given subject), but changing them cannot turn a functional answer into a part-whole answer. |
| Construction inventory | `SyntacticCellAssembly.learned_frames` (`syntactic_cell_assembly.py:107`, allocate `:591`, reinforce `:616`, persist `:743–768`), `VerbLexicon.select_verb` (`verb_lexicon.py:572`). | The inventory that CVCC needs already exists and persists; it is simply not consulted by the productive-use path. |

**Measurement caution.** Line numbers are from a read-only inspection at commit `1b8be3d`; the outputs in the problem statement are treated as given data and were not re-run. Before wiring CVCC, reproduce all three phrasings with an instrumented run that logs the selected relation id and cue, per the discipline in the companion doc §5 and the `brain-faithful-switch-rule` skill.

---

## 2. Research question 1 — how does a question cue constrain the answer form?

### 2.1 QUD, focus congruence, and the form of the answer

On the QUD account, an interrogative accepted by the interlocutors becomes an open discourse goal; an assertion is a congruent response only if its focus alternatives match the question alternatives (Roberts 2004 [10] — chapter). The wh-phrase is the visible marker of that focus requirement: the focused constituent in the answer must correspond to the wh-phrase in the question [7][8]. Three independent evidence types support the claim that this is a real, cue-driven constraint rather than a pragmatic afterthought:

- **Electrophysiology.** An inappropriate word occupying the focus position of an answer to a wh-question modulates the ERP waveform 200–500 ms post-onset relative to a focus-appropriate word (Cowles & Kluender 2007 [7]). The answer form is thus monitored against the cue.
- **Behaviour.** In a pair of experiments, a preceding wh-question about the VP raised VP-attachment choices for an ambiguous modifier (73% in Experiment 2) relative to a question about the object noun (61%); the effect was attributed to focus marking by the question, not to perceptual salience (Carlson & Potter 2022 [8]). A question cue therefore changes which structure is selected/interpreted downstream. **Honest boundary:** this is comprehension of an ambiguous sentence, not production of a fresh sentence; it establishes the *direction* cue → structure, not the exact mapping used here.
- **Cross-linguistic conversation.** In a fourteen-language study, polar questions elicit one of two systematically different answer strategies (interjection-type vs repetition-type), with a strong cross-linguistic preference explained by the semantic-pragmatic fit between question form and answer form (Enfield et al. 2019 [9]). Answer *form* is sensitive to question form even where content is identical.

**Consequence for RAVANA.** `what does X do?`, `what is X part of?` and `where is X?` are not interchangeable wrappers around the same directive. They place the requested information in different focus positions and therefore license different answer constructions. A selector that reads only `is_productive_use_request` has thrown away exactly the information the humanlistener uses.

### 2.2 Wh-word → semantic role → construction

Focus congruence is role-level, not word-level: the answer must fill the role the wh-phrase asks about [7][10]. For the three target questions the roles are:

| question cue | requested role | compatible relation types in RAVANA's inventory |
|---|---|---|
| `what does X do?` / `how does X work?` | predicate / function | `capable_of`, functional predicate, `used_for` |
| `what is X part of?` / `what is X made of?` | meronymic whole or stuff | `part_of` (and typed meronyms where the graph supports them) |
| `where is X?` / `where would you find X?` | location | `located_in` / locative PP |

"Part of" is not a single relation. Winston, Chaffin & Herrmann (1987, *Cognitive Science*, 11(4), 417–444 [11]) distinguish six meronymic types (component–integral object, member–collection, portion–mass, stuff–object, feature–activity, place–area), and Chaffin, Herrmann & Winston (1988, *Language and Cognitive Processes*, 3(1), 17–48 [12]) show experimentally that speakers identify the specific relation from context even when the surface form "part of" is the same. **Consequence:** the candidate set must be built from *relation types* carrying semantic tags, and the cue must select a type, not a stored string. A single untyped `part_of` string would be the computational equivalent of the ambiguity the literature shows speakers resolve.

---

## 3. Research question 2 — cue validity is the selection currency

### 3.1 The Competition Model supplies the production rule

The Competition Model treats language knowledge as form–function mappings whose competition is resolved in real time, with cue strengths adjusted by reliability and availability (MacWhinney 2021 [3] — chapter). Cue validity is defined as the product of availability (how often the cue is present when needed) and reliability (how often it leads to the correct interpretation when present) (MacWhinney, Bates & Kliegl 1984 [1]; Bates et al. 1982 [2]). Cross-linguistic work shows speakers weight cues by their information value, with reliability dominating in adults [1][2][3]. In **production**, this means the chosen form is the winner among competing forms for expressing a meaning, weighted by the same validity [3].

Mapping to RAVANA's primitives (no new variables invented):

- **availability** ≈ `learned_frames[f].weight` and the persisted deployment count for `(word, f)` — the `allocate_construction`/`reinforce_construction`/`get_state` machinery at `syntactic_cell_assembly.py:591`, `:616`, `:743–768`. This is also the type-frequency/collostructional side of usage (Bybee 2006, *Language*, 82(4), 711–733 [18]).
- **reliability** ≈ the construction's fit to the current cue: does a `part_of` frame reliably realize a "part of" question in this system's own history/definition bundle? Initially this is a small fixed cue→relation table, later a count learned from accepted turns.

The brain-derived claims are (a) that availability and reliability combine multiplicatively into validity [1], (b) that cues are weighted by validity, with reliability dominating for an adult system [2][3], and (c) that production selects the winning form by that weighting [3]. The exact numeric form of the score is an implementation choice to be swept, not a brain constant (§7).

### 3.2 Constructions themselves are high-validity cues

Construction cue validity is at least as high as verb cue validity in child-directed speech, and constructions have higher category validity, i.e. they are more available when the meaning is present (Goldberg, Casenhiser & Sethuraman 2005 [4]). Speakers sort sentences by argument-structure construction even when a verb-based sort is available (Bencini & Goldberg 2000 [5]); the usage-based position treats such form–meaning pairings, not words-in-slots, as the unit of generalization (Goldberg 2006 [20] — monograph; Tomasello 2003 [21] — monograph). **Consequence:** a construction-level `cue_fit(f, q)` is not a post-hoc feature engineered for this task; it implements the empirically observed fact that the frame predicts the event interpretation.

### 3.3 Cue-based retrieval supplies the cue→item mapping

In the activation-based model of Lewis & Vasishth (2005 [6]), memory items are bundles of feature–value pairs; a retrieval cue derived from the current word and context spreads activation to all matching items, retrieval probability is proportional to activation, and items sharing a cue interfere (similarity-based interference). This is a comprehension model, and applying it to construction selection is an **extrapolation** (§7), but the computable implication is direct and fits the symbolic inventory: give each candidate frame a small feature tag set (`functional`, `used_for`, `part_of`, `locative`, `property`), derive cue features from the user's wh-word + head noun + preposition, and compute match activation from tag overlap. No new storage is needed — the tags are read off `relation_type` at the `VerbLexicon.select_verb` boundary.

A useful guard falls out of the same theory: because activation is shared among items matching the same cue, a cue that matches several frames should not be treated as decisive. Concretely, if two candidates tie on cue fit, fall back to base strength rather than inventing a distinction. This is the competition-model "conflict" case, where reliability on less frequent configurations is what resolves the tie [3].

---

## 4. Research question 3 — the neural selection operation is not a counter

Left IFG activity follows selection demands while semantic retrieval is held constant (Thompson-Schill et al. 1997 [13]); patients with left-IFG lesions fail only on high-competition items, with a large competition × lesion interaction (Thompson-Schill et al. 1998 [14]). The region is therefore a competitor resolver, not a retrieval switch and certainly not an ordinal counter. Corticostriatal loops implement domain-general selection/sequencing under conflict and uncertainty, enhancing a target through the direct pathway and suppressing competitors through the indirect pathway (Copland et al. 2021 [15]); the basal ganglia are thus not a language module and not a retrieve/generate toggle. Interactive alignment adds the discourse-level force: production and comprehension become coupled through automatic priming at lexical, syntactic and semantic levels (Pickering & Garrod 2004, *Behavioral and Brain Sciences*, 27(2), 169–190 [16] — target article), and structural priming biases reuse of a recently produced structure (Bock 1986, *Cognitive Psychology*, 18(3), 355–387 [17]).

**Therefore:** the correct computational analogue is a scored competition with (i) activation from the cue, (ii) base strength from learned history, (iii) suppression of the just-used candidate, and (iv) a single winner. CVCC has exactly these four terms. A per-subject repeat counter has only (iii); it is cue-blind and would still select by ordinal when competition is high — the specific condition under which LIFG/BG selection is *required* [13][14][15].

**Why fatigue must be demoted.** Priming predicts reuse of the just-produced structure [16][17], while givenness/Grice's quantity maxim penalizes re-asserting known content (Grice 1975 [19] — chapter). The companion doc resolved this with a recency penalty large enough to force alternation. CVCC resolves it differently and more faithfully: the **cue decides among competing frames**, and recency only breaks ties between equally cued frames. This is what lets `where is X?` select the locative frame even immediately after a functional answer — a behaviour no counter can produce.

---

## 5. Observable criteria: cue-sensitive selection vs counter-driven alternation (5)

Each criterion is machine-checkable on both the emitted string and the logged frame/relation id. Thresholds and sweeps are engineering parameters, not brain constants. Report counts with denominators (cue × order × subject) and never build the gold set in the same pass that chose the mechanism.

| # | Criterion | Operational test | Counter-driven system fails because | Evidence |
|---|---|---|---|---|
| **C1** | **Cue differential at fixed state.** | From one persisted state for subject `X`, issue `what does X do?`, `what is X part of?`, `where is X?` in that order with no other turns. Assert the logged relation types are respectively functional, `part_of`, locative; three distinct frame ids; three distinct strings. Repeat from the same saved state: byte-identical frame log. | With a fresh counter all three take frame 0; with a warm counter the assignment is a function of ordinal, so at most one of the three can be correct. | [10][7][1] |
| **C2** | **Cue dominates ordinal (permutation invariance).** | Run all 6 permutations of the three questions from identical fresh state. Assert each cue maps to the same relation type in every permutation (6/6 per cue). | The counter always assigns frames by position; permuting cues permutes the wrong answers. This is the sharpest behavioural dissociation from `%2` alternation. | [9][8][7] |
| **C3** | **Cue overrides recency; fatigue is a tie-break only.** | Immediately after a functional answer, ask `where is X?`. Assert the locative frame wins on its first cued occurrence even though it was not the most recently used. Then ask a neutral `use the word X in a sentence` twice: only there may the fatigue term cycle frames. | The counter cannot override; it necessarily returns the other frame. Conversely, a system with fatigue as driver fails the cued part of this test. | [16][17][19] |
| **C4** | **Focus/role fit and grounding under every cue.** | For each cued answer, assert the focused constituent realizes the requested role: `where` answer has a graph-supported locative target; `what … part of` answer has a graph-supported meronymic target; `what … do` answer has a functional predicate. Also apply the companion's echo invariant (definition not a substring; Jaccard < 0.45; target in graph; ≥2 new content tokens). | The current two frames pass the echo invariant but contain no locative or meronymic target, so role fit fails for two of three cues. A naive "add a preposition" fix fails the graph-target half. | [7][11][12] |
| **C5** | **Ablation and reproducibility (no RNG).** | Three ablations on the same probe set: (a) delete cue term → C1 and C2 must collapse; (b) delete fatigue term → cued criteria still pass, neutral repetition reverts to base frame; (c) delete the counter (current code) → cue criteria pass. Then run the same cue sequence twice from identical state: byte-identical log. | A counter-driven system passes reproducibility but fails C1/C2; a `random.choice` fix passes C1/C2 by luck but fails reproducibility. Only a state-dependent, cue-scored rule passes all. | [13][14][6] |

**Verdict on the current implementation.** C1 fails (no cue read). C2 fails (ordinal, not cue). C3 fails (counter cannot be overridden). C4 fails for 2 of 3 cues (no locative/meronymic content exists). C5 passes only the reproducibility half. The system has **zero cue sensitivity**.

---

## 6. Recommended mechanism: Cue-Validity Construction Competition (CVCC)

**Constraint:** no new neural component, no retraining, no weakening of `_sm_response_grounded` / `_is_word_salad` / the completeness gate, no change to the initial-only performative fix at `engine_generation.py:2098–2104` except the coverage extension already recommended in the companion doc. Every primitive below exists in `response_gen.py`, `surface_realizer.py`, `verb_lexicon.py` and `syntactic_cell_assembly.py`; only the **cue reading** and the **selection rule** change.

### S0 — cue extraction (constructional, not a keyword list)

Add `extract_construction_cue(text) -> Cue` beside `is_productive_use_request` (`response_gen.py:187`). Read the *construction* formed by wh-word + head noun + preposition, with the directive still recognized by `_PRODUCTIVE_USE_RE`:

- `FUNCTION`: `what does <X> do`, `what is <X> for`, `how does <X> work`, `what <X> does`.
- `PART_WHOLE`: `what is <X> part of`, `what is <X> a part of`, `what does <X> belong to`, `what is <X> made of`, `what are the parts of <X>`.
- `LOCATIVE`: `where is <X>`, `where would you find <X>`, `where can <X> be found`, `where does <X> occur`.
- `PROPERTY`: `what is <X> like`, `what colour/size is <X>` (only when the definition bundle has a grounded attribute).
- `NONE`: bare `use the word X in a sentence` and any unmatched phrasing.

Use a constructional test (wh-word + selected preposition + object type), not a flat token list, so that `what is the part of the word X` cannot accidentally match. Fail closed to `NONE`, never guess. This is the same speech-act discipline the companion applies to the directive.

### S1 — admissible construction set from relation types

From the definition bundle (FBCD D1 decomposition) plus the subject's graph neighbours, build candidates tagged by relation type: `capable_of`/functional, `used_for`, `part_of` (with a meronym subtype where the graph supports one), `located_in`, `has_property`. Veto `is_a`/`isa` and any candidate whose object would be the definition string. Each candidate must have a graph-supported target (companion D3/S1). If the **cued** relation has no grounded target — e.g. `quindle` = "the feeling of waiting for rain" has no locative or part target — decline to compose and fall through to honest uncertainty. **Do not substitute another construction to satisfy the cue**: cue sensitivity must not become confabulation, and this guard is exactly what the grounding gate enforces.

### S2 — the score (load-bearing step)

For word `w`, request `q`, candidate frame `f`:

```
availability(f, w) = learned_frames[f].weight + deployments[(w, f)]     # existing persisted state
reliability(f, q)  = P(f | cue(q))    # small fixed cue→relation table, later counted from accepted turns
validity(f, q, w)  = availability(f, w) × reliability(f, q)             # Competition Model product [1][3]

score(f) = log(availability(f,w)) + log(reliability(f,q)) + k*tag_fit(f,q) - λ*recency(f,w)

tag_fit(f,q) = +2 exact tag match (PART_WHOLE↔part_of, LOCATIVE↔located_in, FUNCTION↔capable_of)
               +1 compatible match (FUNCTION↔used_for, PROPERTY↔has_property)
                0 otherwise (cue NONE → 0 for all)
recency(f,w) = deployments of f for w in this session (companion SCC-RF term, now tie-break only)
λ            < min positive base-score gap among admissible frames   (cue must dominate fatigue)
```

The log form is just the product expressed additively; keep it monotone in both factors. `tag_fit` implements cue-feature activation as in cue-based retrieval [6], with the tie rule (equal activation → base strength) that similarity-based interference demands. `learned_frames` weights, `deployments[(w,f)]` and `recency` all live in the same persisted state key used by M16b (`get_state`/`set_state`, `syntactic_cell_assembly.py:743–768`). **`cue_fit` introduces no new store.**

**Why this is brain-faithful and not ad-hoc:**

- availability × reliability is cue validity [1][3];
- construction-level reliability is empirically at least as good as verb-level reliability [4][5];
- `tag_fit` is cue-feature activation as in cue-based retrieval [6];
- the argmax with suppression of the recent winner is LIFG/BG selection [13][14][15];
- the cue itself is an information-state constraint on answer form [7][8][10]; cross-linguistic answer-form sensitivity adds converging evidence [9];
- reinforcing an accepted mapping is the usage-based entrenchment of a type–frequency pattern [18].

### S3 — selection and realization

`f* = argmax_f score(f)`, ties broken by the stable `relation_type` string. **Do not call `random.choices`.** Bind with `find_construction`/`bind_to_sentence` (`syntactic_cell_assembly.py:207`, `:581`) and `VerbLexicon.select_verb(relation_type)` (`verb_lexicon.py:572`). Use `SurfaceRealizer.MERGE_VARIANTS` *only after* the relation is fixed: `cleft`/`pseudo_cleft` to mark the cued focus, `as_for_topic` when the subject is given, `svo` as the unmarked default. The ten surface variants are **not** substitutes for the three constructional relations; varying them alone is what the companion doc forbids and what C4 detects.

### S4 — gate, reinforce, update

Apply the echo/grounding gate and C1–C4 before emission. On acceptance: `allocate_construction`/`reinforce_construction`, `learn_from_feedback(frame, True)`, increment `deployments[(w,f)]` and `recency(f,w)`. This raises availability for the cued mapping — the competition-model learning rule and the type-frequency effect in one step [3][18]. On rejection: try the next-ranked admissible frame (relax the *choice*, never the grounding criterion); if none survives, fall through to honest uncertainty. On a neutral cue with no frame above base, allow fatigue to rotate among equally scored frames — this is the only place alternation remains legitimate.

### S5 — what NOT to do

- Do not use `_productive_use_counts[subject] % 2` as the selector; it is the defect being removed.
- Do not add `random.choice`/`random.choices`; it breaks the reproducibility half of C5.
- Do not treat `MERGE_VARIANTS` as the construction inventory; it is a surface realizer, not a relation selector.
- Do not invent a target when the cued relation is ungrounded; fail closed.
- Do not replace constructional cue tests with a flat keyword list; that reintroduces the paraphrase fragility the companion doc documents.
- Do not delete or bypass `_definitions`; `what is X?` must still recall correctly.

### Falsifiers (with measurable predictions)

1. Delete the cue term: C1 and C2 must collapse to one frame per subject. If they still pass, the cue is leaking in through the directive detector or the subject normalizer.
2. Permute the cue→relation table while leaving everything else fixed: the same cue sequence must now select the wrong frames and C4 must fail. If output is unchanged, the table is not load-bearing.
3. Set `k` (tag weight) below the base-score spread: C1 must fail. Sweep to find the crossover; that value is an empirical parameter, not a brain constant.
4. Set `λ` above the minimum base gap so fatigue dominates: C3's cued part must fail while neutral repetition still cycles — the signature of the current counter-driven system.
5. Remove the graph-target guard in S1: unsupported-assertion counts must rise on cued `where`/`part of` probes. If they do not, the guard was never consulted.

---

## 7. Honest flags

- **No localized "construction selector".** Construction grammar is linguistic theory plus behavioural evidence [5][20][21]; LIFG/BG results concern selection and control [13][14][15], not a dedicated construction module. CVCC is an extrapolation from those results to a symbolic architecture.
- **Cue validity is a framework claim, not a neural measurement.** The availability × reliability product is the Competition Model's definition [1][3]; its exact algebraic use in a production scorer, and the weights `k`/`λ`, are implementation choices to sweep. The brain-derived constraints are the *dependence* on both factors, the dominance of reliability in an adult system, and the multiplicative combination.
- **Cue-based retrieval is a comprehension model.** Lewis & Vasishth (2005 [6]) model dependency resolution in working memory; using tag overlap as construction-selection activation is an analogy, labelled as such. The similarity-based-interference tie rule is the part carried over deliberately.
- **Question-form evidence is mostly comprehension/cross-linguistic.** Carlson & Potter (2022 [8]) and Cowles & Kluender (2007 [7]) show that a question cue changes structural interpretation and focus processing; Enfield et al. (2019 [9]) concerns polar answers in conversation. None is a production study of "answer a question about a one-turn-taught word with a fresh construction", so C1–C5 are **designed diagnostics**, not measured human norms.
- **"Part of" is a family of relations.** Winston et al. (1987 [11]) and Chaffin et al. (1988 [12]) show six/eight meronymic types resolved by context. A single `part_of` frame is a deliberate simplification, valid only where a graph-supported typed meronym is unavailable; the design should prefer a typed subtype when present.
- **QUD/focus is a pragmatic theory.** Roberts (2004 [10]) is a handbook chapter, not an experiment; its empirical support here is the ERP and behavioural work [7][8].
- **Priming vs givenness tension remains.** Alignment/priming predicts reuse [16][17]; Gricean givenness predicts avoiding repetition [19]. CVCC's choice — cue first, fatigue as tie-break — is a design decision consistent with both forces, not a value read off the brain.
- **No direct experiment** matches this exact task. Every mapping from the cited literature to a RAVANA primitive is an engineering instantiation of a general mechanism and should be reported as such.

## References (full citations)

**Peer-reviewed journal articles**

- **MacWhinney, B., Bates, E., & Kliegl, R.** (1984). Cue validity and sentence interpretation in English, German, and Italian. *Journal of Verbal Learning and Verbal Behavior*, 23(2), 127–150. doi:10.1016/S0022-5371(84)90093-8
- **Bates, E., McNew, S., MacWhinney, B., Devescovi, A., & Smith, S.** (1982). Functional constraints on sentence processing: A cross-linguistic study. *Cognition*, 11(3), 245–299. doi:10.1016/0010-0277(82)90017-8
- **Goldberg, A. E., Casenhiser, D. M., & Sethuraman, N.** (2005). The role of prediction in construction-learning. *Journal of Child Language*, 32(2), 407–426. doi:10.1017/S0305000904006798
- **Bencini, G. M. L., & Goldberg, A. E.** (2000). The contribution of argument structure constructions to sentence meaning. *Journal of Memory and Language*, 43(4), 640–651. doi:10.1006/jmla.2000.2757
- **Lewis, R. L., & Vasishth, S.** (2005). An activation-based model of sentence processing as skilled memory retrieval. *Cognitive Science*, 29(3), 375–419. doi:10.1207/s15516709cog0000_25
- **Cowles, H. W., & Kluender, R.** (2007). Violations of information structure: An electrophysiological study of answers to wh-questions. *Brain and Language*, 102(3), 228–242. doi:10.1016/j.bandl.2007.04.004
- **Carlson, K., & Potter, D.** (2022). Focus attracts attachment. *Language and Speech*, 65(2), 491–512. doi:10.1177/00238309211033321
- **Enfield, N. J., Stivers, T., Brown, P., Englert, C., Harjunpää, K., Hayashi, M., Heinemann, T., Hoymann, G., Keisanen, T., Rauniomaa, M., Raymond, C. W., Rossano, F., Yoon, K.-E., Zwitserlood, I., & Levinson, S. C.** (2019). Polar answers. *Journal of Linguistics*, 55(2), 277–304. doi:10.1017/S0022226718000336
- **Winston, M. E., Chaffin, R., & Herrmann, D.** (1987). A taxonomy of part-whole relations. *Cognitive Science*, 11(4), 417–444. doi:10.1207/s15516709cog1104_2
- **Chaffin, R., Herrmann, D. J., & Winston, M.** (1988). An empirical taxonomy of part-whole relations: Effects of part-whole relation type on relation identification. *Language and Cognitive Processes*, 3(1), 17–48. doi:10.1080/01690968808402080
- **Thompson-Schill, S. L., D'Esposito, M., Aguirre, G. K., & Farah, M. J.** (1997). Role of left inferior prefrontal cortex in retrieval of semantic knowledge: A reevaluation. *PNAS*, 94(26), 14792–14797. doi:10.1073/pnas.94.26.14792
- **Thompson-Schill, S. L., Swick, D., Farah, M. J., D'Esposito, M., Kan, I. P., & Knight, R. T.** (1998). Verb generation in patients with focal frontal lesions: A neuropsychological test of neuroimaging findings. *PNAS*, 95(26), 15855–15860. doi:10.1073/pnas.95.26.15855
- **Copland, D. A., Brownsett, S., Iyer, K., & Angwin, A. J.** (2021). Corticostriatal regulation of language functions. *Neuropsychology Review*, 31, 472–494. doi:10.1007/s11065-021-09481-9
- **Pickering, M. J., & Garrod, S.** (2004). Toward a mechanistic psychology of dialogue. *Behavioral and Brain Sciences*, 27(2), 169–190. doi:10.1017/S0140525X04000056
- **Bock, J. K.** (1986). Syntactic persistence in language production. *Cognitive Psychology*, 18(3), 355–387. doi:10.1016/0010-0285(86)90004-6
- **Bybee, J. L.** (2006). From usage to grammar: The mind's response to repetition. *Language*, 82(4), 711–733. doi:10.1353/lan.2006.0186

**Book chapters and monographs (cited as theory, not as journal articles)**

- **MacWhinney, B.** (2021). The Competition Model: Past and Future. In *A Life in Cognition* (pp. 3–16). Springer. doi:10.1007/978-3-030-66175-5_1 (chapter; online 2021)
- **Roberts, C.** (2004). Discourse context in dynamic interpretation. In L. Horn & G. Ward (Eds.), *Handbook of Contemporary Pragmatic Theory* (pp. 197–220). Blackwell. (chapter; QUD)
- **Grice, H. P.** (1975). Logic and conversation. In P. Cole & J. L. Morgan (Eds.), *Syntax and Semantics 3: Speech Acts* (pp. 41–58). Academic Press. (chapter)
- **Goldberg, A. E.** (2006). *Constructions at Work: The Nature of Generalization in Language*. Oxford University Press. (monograph)
- **Tomasello, M.** (2003). *Constructing a Language: A Usage-Based Theory of Language Acquisition*. Harvard University Press. (monograph)

## Sources

[1] https://psyling.talkbank.org/years/1984/cueval.pdf
[2] https://doi.org/10.1016/0010-0277%2882%2990017-8
[3] https://link.springer.com/chapter/10.1007/978-3-030-66175-5_1
[4] https://doi.org/10.1017/S0305000904006798
[5] https://doi.org/10.1006/jmla.2000.2757
[6] https://eric.ed.gov/?id=EJ747042
[7] https://pmc.ncbi.nlm.nih.gov/articles/PMC4075179
[8] https://journals.sagepub.com/doi/abs/10.1177/00238309211033321
[9] https://www.cambridge.org/core/journals/journal-of-linguistics/article/abs/polar-answers/7275CC44717121A045B5018977C9C4B1
[10] https://www.asc.ohio-state.edu/roberts.21/Roberts2004.pdf
[11] https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1104_2
[12] https://www.tandfonline.com/doi/abs/10.1080/01690968808402080
[13] https://www.pnas.org/doi/full/10.1073/pnas.94.26.14792
[14] https://www.pnas.org/doi/abs/10.1073/pnas.95.26.15855
[15] https://link.springer.com/article/10.1007/s11065-021-09481-9
[16] https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/toward-a-mechanistic-psychology-of-dialogue/83442BA495E0D5F81BDB615E4109DBD2
[17] https://www.linguisticsnetwork.org/wp-content/uploads/Syntactic-Persistance_Bock-1986-.compressed.pdf
[18] https://www.unm.edu/~jbybee/downloads/Bybee2006FromUsage.pdf
[19] https://www.sfu.ca/~jeffpell/Cogs300/GriceLogicConvers75.pdf
[20] https://books.google.com/books/about/Constructions_at_Work.html?id=LHrcqeZmUN4C
[21] https://www.hup.harvard.edu/books/9780674044395
