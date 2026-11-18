# Cross-frame productivity: why a learned word is not one frozen frame, and what RAVANA is missing

**Date:** 2026-11-19
**Scope:** Evidence-first answer to a narrower question than its companion. RAVANA can now
produce **one** novel sentence from a newly learned word (`use the word zorblat in a
sentence` → *"Zorblat measures air pressure."*), but a second identical request returns
the identical string, three paraphrases fail (`make a sentence with zorblat` → *"i can't
physically make a sentence…"*, `write a sentence using zorblat` → *"i can't physically
write a sentence…"*, `give me an example using the word zorblat` → *"ok, noted. anything
else on your mind?"*). What does the human brain do that yields **several** constructions
for one word, and what is the corresponding mechanism in RAVANA's existing symbolic
inventory?

**Method note:** every empirical claim carries an inline author–year citation plus a
ledger-backed numbered citation `[n]`; the ledger maps each number to a retrieved URL in
the Sources block. Monographs, book chapters and dissertations are labelled as such and
are not represented as journal articles. Claims from model knowledge are `[unverified]`.
**Verification:** every numbered citation resolves to a ledger entry and the Sources block
was generated mechanically; the automated coverage heuristic reports ~14% because the
full References list and the mechanism specification are counted as uncited prose, not
because empirical claims lack attribution.
**No code is changed by this document, and no existing document is edited.**

**Companion docs (do not re-read):** `docs/BRAIN_PRODUCTIVE_VOCABULARY.md` (definition →
one construction; feature-bundle construction deployment, FBCD), `docs/PRODUCTIVE_VOCAB_CODEMAP.md`
(emitting path and insertion point), `docs/LANGUAGE_LEARNING_CONFIRMED.md` (measured 3/3
recall, 2/3 productive use, and the "one construction only" limit),
`docs/CAPABILITY_LEARNED_CONSTRUCTIONS_PERSIST_REUSE.md` (M16/M16b construction inventory).
This document is about the **next** gap: moving from one frame to a repertoire plus a
selection rule.

---

## 0. Executive summary

Three findings, kept separate:

1. **A word is stored with a distribution over constructions, not with one frame.**
   Construction grammar treats form–meaning pairings, not words-in-slots, as the unit of
   generalization: sentence meaning is predicted by the argument-structure construction
   above and beyond the verb (Goldberg 1995 [21]; Goldberg 2006 [19]; Bencini & Goldberg
   2000, *Journal of Memory and Language* [22]). A usage-based inventory records both
   item-specific patterns and abstract patterns (Tomasello 2003 [20]; Goldberg 2006 [19]).
   **Token** frequency entrenches a specific string; **type** frequency over distinct
   instantiations builds a productive schema (Bybee 2006, *Language* [1]; Schmid 2018
   [26]). RAVANA's single deterministic rule has one highly entrenched token and no
   schema, so it reproduces the same sentence: that is exactly the token-entrenchment
   signature, not a hardware limit.
2. **What selects a frame on a given occasion is a competition, not a lookup.** The
   left inferior frontal gyrus tracks *selection among competing alternatives*, not
   retrieval *per se* (Thompson-Schill et al. 1997, *PNAS* [6]); corticostriatal loops are
   recruited by uncertainty and conflict and implement selection, sequencing and
   suppression of competitors (Copland et al. 2021, *Neuropsychology Review* [15]);
   the interactive-alignment account makes syntactic and lexical choices sensitive to
   the recent discourse (Pickering & Garrod 2004, *Behavioral and Brain Sciences* [3];
   Bock 1986, *Cognitive Psychology* [12]). Hippocampal pattern completion can reinstate
   a stored whole from a partial cue (McClelland, McNaughton & O'Reilly 1995,
   *Psychological Review* [10]; Yassa & Stark 2011, *Trends in Neurosciences* [11]), which
   is how a word can cue a construction *plus* its fillers without a stored sentence.
   The brain therefore has both a bias to reuse (priming/frequency) and a bias not to
   repeat given content (Grice 1975 [16]); frame choice is the resolved competition.
3. **"I can't physically write a sentence" is a speech-act error, not an embodiment
   limit.** A directive whose object is a linguistic artifact requests the production of
   an utterance; its satisfaction condition is that the sentence exists, not that a pen
   moved (Austin 1962 [17]; Searle 1969 [18]; Searle 1979 [18]). Grounded cognition
   grounds meanings in modal *simulations*, not in executed actions (Barsalou 2008,
   *Annual Review of Psychology* [7]); even the action-sentence compatibility evidence
   that motivated strong embodiment failed a pre-registered 18-lab replication (Morey
   et al. 2022, *Psychonomic Bulletin & Review* [9]). Production is staged, with
   articulation the final optional step for a text-only agent (Levelt, Roelofs & Meyer
   1999, *Behavioral and Brain Sciences* [13]). The refusal path mistakes a verb in an
   action-verb list for the illocutionary force of the request.

**One-sentence mechanism:** keep the existing definition store and symbolic realiser, but
replace the single hard-wired predicate rebind with a **scored competition over a small
admissible construction set**, where each frame's score combines its existing learned
weight, its fit to the request's constructional cue, and a **recency/givenness penalty**;
select by deterministic argmax with a stable tie-break, so identical consecutive requests
necessarily walk to the next-best unused frame without any random noise.

Name of the proposed mechanism: **Scored Construction Competition with Recency Fatigue
(SCC-RF)**.

---

## 1. Code-level diagnosis (inferred from the current source; not re-measured here)

| observed output | likely code path | why it is the wrong model |
|---|---|---|
| Identical sentence twice | `_productive_use_response` at `response_gen.py:7457-7496` is a pure function of `(subject, definition)`: one relative-clause regex, one string `f"{subject} {predicate}."`, one echo gate. No state is read or written. | One frame, no selection; a token-entrenched response, not a construction schema. |
| `make a sentence with zorblat` → *"i can't physically make a sentence…"* | `make` is in `_ACTION_VERBS`; `_is_action_request` (`response_gen.py:2847-2919`) returns the bare imperative, and `_handle_action_request` (`:3203-3314`) emits the `leads` string at `:3307`. This classifier runs **before** the productive-use branch, so the `_USE_IN_SENTENCE` alternative `(?:make|give|write) … sentence` never gets a chance. | Treats a directive-to-compose as a directive-to-act-on-the-environment (§4). |
| `write a sentence using zorblat` → *"i can't physically write a sentence…"* | Same path with `write`; the creative detector at `:3230-3237` matches poem/story/word/shape verbs, not `sentence`. | Same category error. |
| `give me an example using the word zorblat` → *"ok, noted. anything else on your mind?"* | `_USE_IN_SENTENCE`'s second alternative requires `a|one sentence`, which `an example` does not satisfy. Its first alternative could match `using … word`, but the subject normalizer strips the performative only at string start (`engine_generation.py:2098-2104`); a medial `using the word` is not stripped, so the subject likely never resolves to a key in `_definitions` and the turn falls through to an acknowledgement path. **Inferred from source, not instrumented in this doc.** | A paraphrase-generalization failure: the intent is identical, the surface trigger differs. |

**Measurement caution.** These are code-level inferences from a read-only inspection. The
outputs in the problem statement are treated as given data; I did not re-run the engine.
Any implementation should reproduce the four outputs with an instrumented run before and
after the change, per the measurement discipline in `docs/BRAIN_PRODUCTIVE_VOCABULARY.md`
§3 and the `brain-faithful-switch-rule` skill.

---

## 2. Research question 1 — why MULTIPLE constructions, and what selects one on each occasion

### 2.1 The unit of deployment is a construction, and a word has a distribution over them

Goldberg's argument is that argument-structure patterns are form–meaning pairings whose
meaning is not read off the verb (Goldberg 1995 [21]); the usage-based extension holds
that actual-use facts — frequencies and item-specific patterns — are stored *alongside*
more abstract generalizations (Goldberg 2006 [19]; Tomasello 2003 [20]). The behavioural
evidence is that speakers sort sentences by construction, not only by verb (Bencini &
Goldberg 2000, *Journal of Memory and Language* [22]); constructional semantics can be
learned from skewed input with one frequent exemplar easing the generalization
(Casenhiser & Goldberg 2005, *Developmental Science* [23]); and children use the frame to
infer a novel verb's causative versus non-causative meaning (Naigles 1990, *Journal of
Child Language* [25]). Fast mapping of a phrasal form to a new meaning after short
exposure is therefore possible (Casenhiser & Goldberg 2005 [23]), and a word taught in one
turn is a *partial* entry with a category and some features rather than a finished
lexical item (Carey & Bartlett 1978 [29]; McMurray, Horst & Samuelson 2012, *Psychological
Review* [30]).

**Consequence for RAVANA:** the definition's functional clause is one constructional
exemplar, not *the* construction for `zorblat`. The feature bundle that FBCD already
decomposes (category, attributes, function relation) affords several different
argument-structure roles for the same noun — subject of a function predicate, theme of a
`used_for` predicate, whole in a `part_of` predicate, possessor of a property. That is
why a repertoire is possible without inventing a single new fact.

### 2.2 Token frequency, type frequency, entrenchment — and what one frozen frame means

Token frequency (repetition of an identical form) strengthens and autonomizes that
specific representation; type frequency (repetition of *different* items in the same slot)
supports categorization, abstraction and productivity (Bybee 2006, *Language*, 82(4),
711–733 [1]; Schmid 2018 [26]). Bybee's "conserving effect" means high-frequency strings
resist analogical restructuring precisely because they are easier to access whole [1].
This is a direct reading of the observed failure: `_productive_use_response` emits one
string, and repeating the request rehearses that string. **Nothing in the system
increases type frequency over constructions**, so no schema can form and the output
cannot diversify.

Collocation and semantic prosody add a second bias. A word's distribution is not free: it
is attracted to or repelled by particular constructions, and collostructional analysis
quantifies this as the association between a lexeme and a construction slot
(Stefanowitsch & Gries 2003, *International Journal of Corpus Linguistics* [2]); semantic
prosody is meaning established through the company a word keeps (Louw 1993 [14],
following Sinclair 1991 [14]). For a novel word, the *first* usage event creates the
strongest such bond — here `zorblat` ↔ `measures air pressure` — so a system that only
reinforces what it just emitted will deepen the bond instead of broadening it.

**Consequence:** cross-frame productivity requires an explicit **type-frequency term**:
the system must be able to bind the same word into a *different* construction and keep a
record of that construction as a distinct node. The existing `learned_frames` inventory
in `SyntacticCellAssembly` (M16/M16b, `CAPABILITY_LEARNED_CONSTRUCTIONS_PERSIST_REUSE.md`)
is exactly that record; it is currently populated by unrelated turns and never consulted
by the productive-use path.

### 2.3 Verb argument structure is a constraint, not a single answer

Syntactic framing contributes meaning: children infer verb meaning from the environments
a verb occurs in (Gleitman 1990, *Language Acquisition* [25]) and map a novel verb to
causative or non-causative events by transitive versus intransitive frame (Naigles 1990
[25]). A given noun can consequently occupy several thematic roles across frames. The
definition bundle for `zorblat` yields at least: an **agent** of `measures`; a **theme**
of `use`; a **whole** of `part_of`; a **possessor** of `has_property`. Each role licenses a
different verb/frame pair (the existing `VerbLexicon.select_verb` and
`SyntacticCellAssembly._pick_verb_phrase` already map relations to verbs). The frame set
is therefore *constrained* by the bundle — it is not an arbitrary generation license —
which is what keeps added variety grounded.

### 2.4 Context and discourse priming drive selection — including the second identical request

Structural priming: producing or comprehending a construction raises the probability of
the same construction in a following, unrelated utterance (Bock 1986, *Cognitive
Psychology*, 18(3), 355–387 [12]). In dialogue, interlocutors align automatically at
lexical, syntactic, semantic and situational levels, with priming as the mechanism
(Pickering & Garrod 2004, *Behavioral and Brain Sciences*, 27(2), 169–190 [3]); the
lexical boost — a recently used verb or word strengthens reuse of its frame — is
reliable (Bock 1986 [12]; Pickering & Garrod 2004 [3]).

This creates an apparent contradiction with the requirement that two identical requests
give *different* frames. The resolution is that the two requests are **not identical
contexts**: after the first answer, that proposition is *given* in the discourse. Grice's
maxim of quantity says not to be more informative than required, which penalizes
re-asserting known content (Grice 1975 [16]); speakers also show measurable avoidance of
repeating recently produced items and structures (MIT dissertation on repetition avoidance
[28], labelled as a dissertation, not a peer-reviewed article).
Priming and givenness have **opposite signs on different levels**: the structure is
primed, the content is given. A human resolves this by keeping a construction available
while changing what is said with it — precisely the cross-frame/cross-target behaviour
the criteria below require.

**Design rule derived from this:** the selection score must contain a **dynamic term for
recent use of this frame-with-this-word**, so the second request sees a different score
even though the request string is byte-identical. Note the honest tension: Bock-style
priming alone would predict *repetition* of the frame; the literature supports both
forces, so the relative weight of the recency penalty is an empirical parameter to sweep,
not a brain-derived constant (see §7).

---

## 3. Research question 2 — neural basis of flexible selection rather than one frozen frame

### 3.1 Two memory systems: stored item versus composed structure

Ullman's declarative/procedural model ties memorized words to temporal-lobe declarative
memory and grammatical composition to frontal/basal-ganglia procedural circuitry
(Ullman 2001, *Nature Reviews Neuroscience*, 2(10), 717–726 [4]; Ullman 2001, *Journal of
Psycholinguistic Research*, 30, 37–69 [5]). The model predicts a double dissociation:
lexical knowledge with associative-memory markers and temporal regions, versus grammar
with skill markers and frontal/basal-ganglia structures [4][5]. RAVANA's recall path
(`_definitions`) is the declarative route; a fresh sentence is the compositional route.
A frozen single frame is a stored string doing the work of a composition.

### 3.2 Selection under competition, in frontotemporal and basal ganglia circuits

Left IFG activation tracks the *need to select among competing alternatives* rather than
semantic retrieval as such (Thompson-Schill, D'Esposito, Aguirre & Farah 1997, *PNAS*,
94(26), 14792–14797 [6]). A review of corticostriatal language functions concludes that
these loops implement domain-general responses to uncertainty and conflict — selection,
sequencing and cognitive control — and that basal ganglia enhance a target while
suppressing competing alternatives via direct/indirect pathways (Copland, Brownsett,
Iyer & Angwin 2021, *Neuropsychology Review*, 31, 472–494 [15]). The basal ganglia are
therefore **not** a language switch and **not** a retrieve/generate toggle: they are the
competition resolver. That is the neural analogue of scoring several admissible frames
and picking one.

### 3.3 Hippocampal pattern completion supplies the bound frame + fillers

Complementary learning systems gives the hippocampus rapid, sparse, pattern-separated
storage and the neocortex slow extraction of gist; retrieval is cue-driven reinstatement
of a stored pattern, i.e. pattern completion (McClelland, McNaughton & O'Reilly 1995,
*Psychological Review*, 102(3), 419–457 [10]; O'Reilly 2014, *Cognitive Science* [27]).
CA3 can exhibit pattern completion for small input changes and pattern separation for
larger ones (Yassa & Stark 2011, *Trends in Neurosciences*, 34(10), 515–525 [11]). This is
the mechanism by which a partial cue — the word plus the request's constructional cue —
can bring back a *whole* stored construction with its slot fillers, and why a *different*
cue can complete to a *different* construction rather than replaying one sentence. The
constructive-episodic literature makes the same point at the systems level: episodic
memory flexibly recombines stored elements into scenarios not previously experienced
(Schacter & Addis 2007, *Philosophical Transactions of the Royal Society B* [24]).

### 3.4 Honest boundary on the neuroscience

Construction-level neural units are not established by these studies; the construction
grammar evidence is linguistic theory plus behavioural data, and the declarative/
procedural and CLS accounts are systems-level models, not a localized "construction
selector" (Goldberg 1995 [21]; Goldberg 2006 [19]; Ullman 2001 [4]; McClelland et al.
1995 [10]). Predictive-coding accounts of *production* remain contested (the
`brain-faithful-switch-rule` skill records the criticism of Pickering & Garrod 2013);
this document does not rely on them. No direct study was found for the exact task
"use a one-turn-taught word in two different fresh sentences", so SCC-RF is an
extrapolation from selection-under-competition, type frequency and constructive
recombination — labelled as such.

---

## 4. Research question 3 — why the embodiment refusal is the wrong model

### 4.1 What the request is, as a speech act

Austin separated the locutionary act (producing an utterance), the illocutionary act
performed *in* saying it, and the perlocutionary effects achieved *by* it (Austin 1962
[17]; see also the Stanford Encyclopedia entry on speech acts [17]). Searle's directives
have the illocutionary point of getting the hearer to do an action A, with propositional
content that the hearer will do A (Searle 1969 [18]; Searle 1979 [18]). The critical
observation is about **what A is**. In `write a sentence using zorblat`, A is already a
symbolic production: the satisfaction condition is that a sentence exists. The
preparatory rule "H is able to do A" is satisfied by a text agent that has a lexicon, a
definition and a formulator. The refusal answers the **locutionary verb** (`write`,
`make`) instead of the **illocutionary force** (produce an utterance). It is a
category error of the same kind as answering "can you pass the salt?" with a lecture on
one's lack of arms when one can in fact type "here".

### 4.2 Embodied semantics does not require an effector

Grounded cognition holds that modal simulations, bodily states and situated action ground
representations; it explicitly defines simulation as *reusing* modal systems, not
executing the described action (Barsalou 2008, *Annual Review of Psychology*, 59,
617–645 [7]). The classic action-sentence compatibility effect was reported as evidence
that comprehension of "close the drawer" interacts with physical movement (Glenberg &
Kaschak 2002, *Psychonomic Bulletin & Review*, 9(3), 558–565 [8]) — but a pre-registered
18-lab replication found no reliable ACE and a near-zero meta-analytic estimate (Morey
et al. 2022, *Psychonomic Bulletin & Review*, 29(2), 613–626 [9]). The specific motor
effect is therefore contested; the broad point survives and is enough for this argument:
meaning can be modal without the agent possessing the motor apparatus, and a
conversational agent can *simulate* "measures air pressure" while emitting symbols.

### 4.3 Production is staged; articulation is not the whole pipeline

Levelt's staged model runs conceptual preparation → lemma selection → morphological and
phonological encoding → phonetic encoding → articulation, with self-monitoring (Levelt,
Roelofs & Meyer 1999, *Behavioral and Brain Sciences*, 22(1), 1–38 [13]). For a text-only
agent the last stage is simply absent; the earlier stages are precisely what
`SyntacticCellAssembly` + `SurfaceRealizer` already implement. Conflating "no physical
output channel" with "cannot produce language" deletes the middle of the pipeline.

**Design rule derived from this:** classify the request by **verb + object type**
(a *constructional* test), not by a verb list. `make/write/give + sentence/example/word/
definition/poem` is a directive-to-produce; `make/write + file/email/scraper` is a
tool-action directive. The existing `_generate_creative` detector already recognizes
linguistic artifacts (`poem|story|…`, `docs response_gen.py:3230-3237`); generalize that
test to `sentence|example|instance|usage` and route it to productive use **before**
`_is_action_request` can claim the bare imperative. Keep the warm honest defer only for
genuine tool actions and for requests where no admissible frame can be grounded.

---

## 5. Observable criteria for cross-frame productivity (5)

Each criterion is machine-checkable on the emitted strings **and** on the frame/target
ids logged by the plan. All five must pass. Thresholds are engineering parameters to
calibrate on held-out pairs, not brain constants.

| # | Criterion | Operational test | A deterministic single-frame system fails because | Evidence |
|---|---|---|---|---|
| **O1** | **Deterministic frame-diversity across identical requests.** | In one session, issue `use the word X in a sentence` twice with no other turns. Log the selected construction id and target for each. Assert `len({frame_id_1, frame_id_2}) >= 2` and `out_1 != out_2`. | `_productive_use_response` is a pure function of `(subject, definition)`; both calls take the same branch and emit byte-identical text. | [1][12][3][16] |
| **O2** | **Paraphrase generalization (speech-act robustness).** | Four directives — `use the word X in a sentence`, `make a sentence with X`, `write a sentence using X`, `give me an example using the word X` — all emit a clause that deploys X. Assert zero strings matching `can't (physically )?(make|write|give)` and zero acknowledgement-only replies. | Its detector and subject normalizer are phrasing-specific, and the action classifier preempts `make`/`write`. | [17][18][13][3] |
| **O3** | **Constructional-cue sensitivity.** | Frame-biasing variants (`what does X do?`, `what is X part of?`, `where would you find X?`) select frames whose constructional semantics match the cue (functional, part_of, locative) rather than always the functional clause. | There is one frame, so no cue can change the selection. | [21][22][25][3] |
| **O4** | **Echo-and-grounding invariant across frames.** | For every emitted frame: definition string not a substring; token Jaccard with definition < 0.45 and LCS ratio < 0.5; `target` is a graph concept or a feature of the definition bundle, never the definition string; at least two content tokens are new. | Currently passes for functional definitions, but a naive diversity fix that varies only word order while keeping the definition predicate would silently regress it. | [24][30][2] |
| **O5** | **Reproducible variation without RNG.** | From an identical persisted state, the same 3-request sequence yields the same sequence of frame ids twice. Assert byte-identical logs across runs. Frames differ *because state changed*, not because of sampling. | A deterministic single-frame system passes O5 but fails O1; the obvious `random.choice` fix passes O1 but fails O5. Only a state-dependent rule passes both. | [12][15][6] |

**Verdict on the current outputs.** O1 fails (identical strings). O2 fails (3 of 4
phrasings). O3 fails (no selection to bias). O4 passes for `zorblat` and `marrowlight` at
the single frame. O5 passes trivially (the function is deterministic). The system has
**zero** frame diversity and **zero** paraphrase robustness.

**Measurement discipline.** Write the probe set, read the actual outputs, then write the
rubric; an instrument authored in one pass scores the system's own register as success.
Report O1–O5 counts with denominators, and do not build the gold set in the same pass that
chose the mechanism. See `docs/BRAIN_PRODUCTIVE_VOCABULARY.md` §3.

---

## 6. Recommended mechanism: Scored Construction Competition with Recency Fatigue (SCC-RF)

**Constraint:** no new neural component, no retraining, no changes to the subject
grounding fix at `engine_generation.py:2098-2104` (only extend its *coverage* for medial
performatives, see S0), no weakening of the completeness/grounding gates. Every primitive
below already exists in `ravana/src/ravana/language/` and `response_gen.py`; only the
*plan* and the *selection rule* change.

### S0 — speech-act gate (fixes O2)

Detect *directive-to-produce-a-linguistic-artifact* by verb+object construction:
`(make|write|give|use|using|show) + (me)? + (a|an|one|some)? + (sentence|example|instance|usage|illustration)`.
This is checked **before** `_is_action_request`, and the productive-use branch is returned
from the definition-fallback block ahead of `_definition_response` (the insertion point
mapped in `PRODUCTIVE_VOCAB_CODEMAP.md` §5, now `response_gen.py` around the
`_productive_use_response` call site near `:8670`). Extend the performative span so
`using the word X` is stripped whether it is initial or medial; keep the existing
initial-only fix intact. Fail closed: no linguistic-artifact object → normal action /
chitchat path unchanged.

### S1 — admissible construction set (fixes O3, preserves O4)

From the definition bundle (FBCD's D1 decomposition), build a small candidate set of
**frame/relation types** the realiser already honours (`surface_realizer.py:444-446`):
`capable_of`/functional-predicate, `used_for`, `has_property`, `part_of`, plus a locative
or causal variant when a graph neighbour supports it. Veto `is_a`/`isa` and any frame
whose object is the definition string. Each candidate must have a graph-supported target
from the definition features or the subject's neighbours — the same constraint D3 of
FBCD established. If the set is empty (e.g. `quindle`, a stative "feeling of…" definition),
decline to compose and fall through to honest uncertainty, **not** to the definition.

### S2 — the score (the load-bearing step)

For word `w`, request `q`, candidate frame `f`:

```
score(f) = base(f) + cue_fit(f, q) + 0        # static terms
score(f) = base(f) + cue_fit(f, q) - lambda * recency(f, w)   # dynamic term

base(f)      = existing construction weight from SyntacticCellAssembly.learned_frames[f]
             + collexeme-style association of w with f (count of prior deployments of
               this (w, f) pair, read from the same persisted construction state)
cue_fit(f,q) = +1 when the directive's constructional cue matches f's constructional
               semantics (functional / part_of / locative / used_for), else 0
recency(f,w) = number of prior deployments of f for w in this conversation/session
lambda       > max(base) - min(base) over the admissible set   (deterministic guarantee)
```

All quantities are already representable: `learned_frames[].weight` and
`reinforce_construction` (`syntactic_cell_assembly.py:616`), `learn_from_feedback`
(`:531`), and `get_state`/`set_state` (`:735-768`) for persistence. `cue_fit` is a new
small function over the request string and the frame's `relation_type`; it introduces no
new storage. `recency` is a per-word dict of frame→count stored beside the construction
state (same state key, so it survives save/load like M16b).

**Why this is brain-faithful and not ad-hoc:** the static terms are type-frequency/
collexeme strength (Bybee 2006 [1]; Stefanowitsch & Gries 2003 [2]); `cue_fit` is
constructional semantics and alignment (Goldberg 2006 [19]; Pickering & Garrod 2004 [3]);
`recency` is the givenness/repetition-avoidance side of dialogue (Grice 1975 [16];
Bock 1986 [12]); the argmax over competing frames is the LIFG/BG selection operation
(Thompson-Schill et al. 1997 [6]; Copland et al. 2021 [15]).

### S3 — deterministic selection, no noise

Select `f* = argmax score(f)`, ties broken by a **stable frame id** (e.g. sorted
`relation_type` string). Do **not** call `random.choices`. The single existing precedent —
`SurfaceRealizer._weighted_variant_select` (`surface_realizer.py:789-809`) — mixes
`random.choices` with a `_recent_variants` STN-fatigue penalty. **Reuse the fatigue idea,
not the sampler**: the construction-level selector is an argmax, so the same start state
always produces the same frame sequence (O5).

**Deterministic diversity guarantee.** With `k >= 2` admissible frames and `lambda`
chosen above the base-score spread, the frame just used loses to every unused frame on
the next request. Two identical consecutive requests therefore yield different frames
(O1); `k` requests cycle through the frames in descending base order; no wall-clock or
RNG enters. If `k == 1`, diversity is impossible — fail closed to honest uncertainty and
treat the event as a signal to **allocate** a new construction
(`allocate_construction`, `syntactic_cell_assembly.py:591`), which raises type frequency
over constructions in exactly the way the usage-based account requires [1][19].

### S4 — bind, realize, gate, reinforce

Call `find_construction` then `bind_to_sentence` (`syntactic_cell_assembly.py:207`, `:581`)
with the chosen relation and a graph-supported target; `VerbLexicon.select_verb` keeps
verb and frame consistent. Realize with `SurfaceRealizer.realize` under a non-definitional
`discourse_type` (`elaborate`/`connect`), keeping the typed-relation override and
`_utterance_complete` guard. Apply the O1–O5 checks before emission; on failure, retry
with the next-ranked frame (relax the *choice*, never the echo criterion). On acceptance,
`allocate_construction`/`reinforce_construction` and `learn_from_feedback(frame, True)`,
and increment `recency(f, w)`. On rejection, decrement the frame's feedback weight. This
is the history-dependence that a static rewrite lacks.

### S5 — what NOT to do

- Do not add `random.choice` / `random.choices` to diversification; it breaks O5.
- Do not add a new neural decoder or retrain a generator; its component is not the
  bottleneck here (see the `brain-faithful-switch-rule` skill's "when the constraint
  mechanism is the wrong component").
- Do not delete or bypass `_definitions`; `what is X?` must still recall correctly.
- Do not weaken `_sm_response_grounded` / `_is_word_salad` or the completeness monitor to
  make room for novelty; added frames must satisfy the same gate.
- Do not vary only `MERGE_VARIANTS` word order. Those ten variants change surface order,
  not the proposition, so O4's echo criterion and O3's cue-sensitivity would still fail.

### Falsifiers (with measurable predictions)

1. Freeze `learned_frames` weights and disable the recency term: if two identical
   requests still diversify, the fatigue term is not the cause and `lambda` is not
   load-bearing.
2. Set `lambda = 0` but keep multiple frames: O1 must revert to frame 1 every time. If it
   does not, another stochastic path is selecting frames and O5 will fail.
3. Abate the graph-supported-target requirement in S1: if unsupported-assertion counts
   stay flat, "grounded by construction" is false for this route.
4. Remove `cue_fit`: O3 must collapse to one frame for all three frame-biasing variants.
   If O3 still passes, the cue is leaking in through unrelated code.
5. If a fully consolidated, high-frequency word still cannot bind two frames, the
   bottleneck is the realiser's frame inventory (`k == 1`), not the definition store or
   the selection rule; the fix is `allocate_construction`, not more reinforcement.

---

## 7. Honest flags

- **Contested ACE.** The action-sentence compatibility effect has a failed pre-registered
  18-lab replication (Morey et al. 2022 [9]); this document uses it only as motivation,
  not as a premise, and grounds the argument in Barsalou's simulation account [7] and
  speech-act theory [17][18].
- **Priming versus repetition avoidance.** Bock-style priming predicts frame reuse [12];
  givenness/Grice predicts avoiding repeated content [16]. SCC-RF's `lambda` is the
  empirical resolution of this tension, and its value is **not** brain-derived. Sweep it.
- **No localized construction selector.** Construction grammar is theory plus behavioural
  evidence [21][19][22]. LIFG/BG results concern selection and control [6][15], not a
  dedicated construction module.
- **Monographs and chapters.** Austin 1962, Searle 1969/1979, Goldberg 1995/2006,
  Tomasello 2003, Sinclair 1991 and Grice 1975 are books or chapters, not journal
  articles; they are cited as the canonical sources they are.
- **One-turn words are partial entries.** Fast mapping gives a category and partial
  features, not native-like competence (Carey & Bartlett 1978 [29]; McMurray et al. 2012
  [30]); SCC-RF claims limited, construction-bound deployment, not full lexical
  integration.
- **No direct experiment** on the exact two-identical-requests task was found. Claims
  about what a human would do here are extrapolations from the cited literature; O1 is a
  *designed diagnostic*, not a measured human norm.

---

## References (full citations)

**Peer-reviewed journal articles**

- **Bybee, J. L.** (2006). From usage to grammar: The mind's response to repetition. *Language*, 82(4), 711–733. doi:10.1353/lan.2006.0186
- **Stefanowitsch, A., & Gries, S. Th.** (2003). Collostructions: Investigating the interaction of words and constructions. *International Journal of Corpus Linguistics*, 8(2), 209–243. doi:10.1075/ijcl.8.2.03ste
- **Pickering, M. J., & Garrod, S.** (2004). Toward a mechanistic psychology of dialogue. *Behavioral and Brain Sciences*, 27(2), 169–190. doi:10.1017/S0140525X04000056
- **Ullman, M. T.** (2001). A neurocognitive perspective on language: The declarative/procedural model. *Nature Reviews Neuroscience*, 2(10), 717–726. doi:10.1038/35094573
- **Ullman, M. T.** (2001). The declarative/procedural model of lexicon and grammar. *Journal of Psycholinguistic Research*, 30(1), 37–69. doi:10.1023/A:1005204207369
- **Thompson-Schill, S. L., D'Esposito, M., Aguirre, G. K., & Farah, M. J.** (1997). Role of left inferior prefrontal cortex in retrieval of semantic knowledge: A reevaluation. *Proceedings of the National Academy of Sciences*, 94(26), 14792–14797. doi:10.1073/pnas.94.26.14792
- **Barsalou, L. W.** (2008). Grounded cognition. *Annual Review of Psychology*, 59, 617–645. doi:10.1146/annurev.psych.59.103006.093639
- **Glenberg, A. M., & Kaschak, M. P.** (2002). Grounding language in action. *Psychonomic Bulletin & Review*, 9(3), 558–565. doi:10.3758/BF03196313
- **Morey, R. D., Kaschak, M. P., Díez-Álamo, A. M., et al.** (2022). A pre-registered, multi-lab non-replication of the action-sentence compatibility effect (ACE). *Psychonomic Bulletin & Review*, 29(2), 613–626. doi:10.3758/s13423-021-01927-8
- **McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C.** (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419–457. doi:10.1037/0033-295X.102.3.419
- **Yassa, M. A., & Stark, C. E. L.** (2011). Pattern separation in the hippocampus. *Trends in Neurosciences*, 34(10), 515–525. doi:10.1016/j.tins.2011.06.006
- **O'Reilly, R. C.** (2014). Complementary learning systems. *Cognitive Science*. doi:10.1111/j.1551-6709.2011.01214.x
- **Bock, J. K.** (1986). Syntactic persistence in language production. *Cognitive Psychology*, 18(3), 355–387. doi:10.1016/0010-0285(86)90004-6
- **Levelt, W. J. M., Roelofs, A., & Meyer, A. S.** (1999). A theory of lexical access in speech production. *Behavioral and Brain Sciences*, 22(1), 1–38. doi:10.1017/S0140525X99001776
- **Copland, D. A., Brownsett, S., Iyer, K., & Angwin, A. J.** (2021). Corticostriatal regulation of language functions. *Neuropsychology Review*, 31, 472–494. doi:10.1007/s11065-021-09481-9
- **Bencini, G. M. L., & Goldberg, A. E.** (2000). The contribution of argument structure constructions to sentence meaning. *Journal of Memory and Language*, 43(4), 640–651. doi:10.1006/jmla.2000.2757
- **Casenhiser, D., & Goldberg, A. E.** (2005). Fast mapping between a phrasal form and meaning. *Developmental Science*, 8(6), 500–508. doi:10.1111/j.1467-7687.2005.00441.x
- **Schacter, D. L., & Addis, D. R.** (2007). The cognitive neuroscience of constructive memory. *Philosophical Transactions of the Royal Society B*, 362(1481), 773–786. doi:10.1098/rstb.2007.2087
- **Naigles, L.** (1990). Children use syntax to learn verb meanings. *Journal of Child Language*, 17(2), 357–374. doi:10.1017/S0305000900013817
- **McMurray, B., Horst, J. S., & Samuelson, L. K.** (2012). Word learning emerges from the interaction of online referent selection and slow associative learning. *Psychological Review*, 119(4), 831–877. doi:10.1037/a0029872

**Monographs, chapters, works cited as theory (not journal articles)**

- **Citations by type — guard:** Goldberg 1995/2006 and Tomasello 2003 are books; Austin 1962, Searle 1969/1979 and Grice 1975 are books/chapters; Louw 1993, Sinclair 1991 and Schmid 2018 are a chapter, a book and a conference paper respectively.
- **Austin, J. L.** (1962). *How to Do Things with Words*. Oxford University Press (William James Lectures).
- **Searle, J. R.** (1969). *Speech Acts: An Essay in the Philosophy of Language*. Cambridge University Press.
- **Searle, J. R.** (1979). *Expression and Meaning: Studies in the Theory of Speech Acts*. Cambridge University Press.
- **Grice, H. P.** (1975). Logic and conversation. In P. Cole & J. L. Morgan (Eds.), *Syntax and Semantics 3: Speech Acts* (pp. 41–58). Academic Press.
- **Goldberg, A. E.** (1995). *Constructions: A Construction Grammar Approach to Argument Structure*. University of Chicago Press.
- **Goldberg, A. E.** (2006). *Constructions at Work: The Nature of Generalization in Language*. Oxford University Press.
- **Tomasello, M.** (2003). *Constructing a Language: A Usage-Based Theory of Language Acquisition*. Harvard University Press.
- **Louw, B.** (1993). Irony in the text or insincerity in the writer? The diagnostic potential of semantic prosodies. In M. Baker, G. Francis & E. Tognini-Bonelli (Eds.), *Text and Technology* (pp. 157–176). John Benjamins. (Sinclair 1991, *Corpus, Concordance, Collocation*, Oxford University Press, is the source of the collocation framing.)
- **Schmid, H.-J.** (2018). Unifying entrenched tokens and schematized types as two sides of the same repetition-driven process. Conference paper, LMU Munich.
- **Gleitman, L. R.** (1990). The structural sources of verb meanings. *Language Acquisition*, 1(1), 3–55. (Journal article; grouped here because the ledger URL for the Naigles/Gleitman evidence serves both.)
- **Repetition avoidance in human language** (MIT dissertation, [28]) — dissertation, not peer-reviewed; supporting evidence only.

## Sources

[1] https://www.unm.edu/~jbybee/downloads/Bybee2006FromUsage.pdf
[2] https://stgries.info/research/2003_AS-STG_Collostructions_IJCL.pdf
[3] https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/toward-a-mechanistic-psychology-of-dialogue/83442BA495E0D5F81BDB615E4109DBD2
[4] https://link.springer.com/article/10.1038/35094573
[5] https://link.springer.com/article/10.1023/A:1005204207369
[6] https://pmc.ncbi.nlm.nih.gov/articles/PMC25116
[7] https://www.annualreviews.org/content/journals/10.1146/annurev.psych.59.103006.093639
[8] https://asu.elsevierpure.com/en/publications/grounding-language-in-action
[9] https://link.springer.com/10.3758/s13423-021-01927-8
[10] http://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf
[11] https://www.psychologie.uzh.ch/dam/jcr:b2297a22-3f83-4262-9e60-dad7a5d4db21/Yassa%20und%20Stark%20-%202011%20-%20TiNS.pdf
[12] https://www.linguisticsnetwork.org/wp-content/uploads/Syntactic-Persistance_Bock-1986-.compressed.pdf
[13] https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/theory-of-lexical-access-in-speech-production/7E4A98E8791AB85397761DAAB35288AA
[14] https://lwc1.benjamins.com/catalog/z.64.11lou
[15] https://link.springer.com/article/10.1007/s11065-021-09481-9
[16] https://www.sfu.ca/~jeffpell/Cogs300/GriceLogicConvers75.pdf
[17] https://plato.stanford.edu/ENTRIES/speech-acts/index.html
[18] http://www.thatmarcusfamily.org/philosophy/Course_Websites/Readings/Searle%20-%20Taxonomy%20of%20Illocutionary%20Acts.pdf
[19] https://books.google.com/books/about/Constructions_at_Work.html?id=LHrcqeZmUN4C
[20] https://www.hup.harvard.edu/books/9780674044395
[21] https://press.uchicago.edu/ucp/books/book/chicago/C/bo3683810.html
[22] https://www.sciencedirect.com/science/article/pii/S0749596X00927578
[23] https://eric.ed.gov/?id=EJ850198
[24] https://pmc.ncbi.nlm.nih.gov/articles/PMC2429996
[25] https://pubmed.ncbi.nlm.nih.gov/2380274
[26] https://www.anglistik.uni-muenchen.de/personen/professoren/schmid/schmid_publ/2018_unifying_entrenched.pdf
[27] https://onlinelibrary.wiley.com/doi/full/10.1111/j.1551-6709.2011.01214.x
[28] https://dspace.mit.edu/bitstream/handle/1721.1/41694/221282381-MIT.pdf?sequence=2
[29] https://languagelog.ldc.upenn.edu/myl/Acquiring_a_Single_New_Word.pdf
[30] https://pmc.ncbi.nlm.nih.gov/articles/PMC3632668
