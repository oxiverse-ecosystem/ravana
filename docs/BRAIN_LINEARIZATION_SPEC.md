# Linearizing a pre-lexical message into surface word order

**Date:** 2026-11-14
**Status:** research + implementation spec. No code changed.
**Question:** RAVANA's GRU decoder emits fluent word salad
("Such change like species one used") and, when a logit boost is applied to the
right content words, emits a *word list* with zero function words
(`docs/ANCHOR_BOOST_DESTROYS_SYNTAX.md`). The missing capability is not word
selection — it is **linearization**: turning a pre-lexical relational message
into a grammatical surface order. This document establishes what the brain does
at that stage, with citations, and ends with a concrete, falsifiable design.

**Companion docs (do not re-read):** `docs/BRAIN_MESSAGE_PREPARATION.md`
(the message layer and lemma competition), `docs/BRAIN_NOVEL_SUBJECT_GROUNDING.md`
(the ATL hub, the angular-gyrus combination step, the commitment gate),
`docs/NEUROLANGUAGE_PRODUCTION_MAPPING.md` and `docs/LANGUAGE_PRODUCTION_PIVOT.md`
(the mechanism→module map), `docs/ANCHOR_BOOST_DESTROYS_SYNTAX.md`,
`docs/CONDITIONING_COLLAPSES_TO_A_POINT.md`,
`docs/DECODER_FIXED_FILLER_REGISTER.md`, `docs/BOOST_STRENGTH_NOT_THE_LEVER.md`,
`docs/RELATION_GATING_SETTLED.md`.

---

## 0. Executive summary

The brain does not select words and then order them. It **builds a relational
message first, assigns syntactic functions, and only then fixes a left-to-right
order** — and the order is fixed *before* articulation, by a mechanism
separable from the words that fill it. Three findings carry the argument:

1. **The message is non-linear and role-bearing.** Levelt's pre-verbal message
   is a propositional structure with thematic-role↔concept bindings
   (`AGENT ↔ DOG`), not a word sequence (Levelt 1989; Bock & Ferreira 2014).
   Linearization is the separate, later operation that imposes a serial order
   on it (Levelt 1981; Kempen & Hoenkamp 1987; Bock & Levelt 1994).

2. **Word order is a two-step computation, not a retrieval.** Thematic roles are
   mapped onto syntactic functions (agent→subject, patient→object) in
   *functional processing*; the functions are then ordered in *positional
   processing* (Bock & Levelt 1994). The ordering step is sensitive to
   conceptual accessibility (Bock & Warren 1985; McDonald, Bock & Kelly 1993)
   but **not** to the identity of the function words (Bock 1989) — which is
   exactly why a content-word logit boost destroys syntax: it supplies the
   open-class fillers and leaves the closed-class frame with nothing to be
   computed from.

3. **The brain both retrieves multi-word units and computes novel ones, and the
   two are not in conflict.** Pawley & Syder's (1983) "nativelike selection /
   nativelike fluency" puzzle is real, but the modern resolution is a
   *dual* route: frequent sequences are chunked (Arnon & Christiansen 2017;
   McCauley & Christiansen 2017), while novel sentences are produced by
   slot-filling into a construction (Goldberg 2006; Tomasello 2003). The
   construction *is* the linearization template; the slot-filler is the
   selected lemma. A system that only does one of the two is broken in a
   characteristic way — and RAVANA's decoder is broken in the "no template"
   way.

**The one-sentence version:** *Linearization is the step that converts a
role-bearing relational message into an ordered syntactic frame; it is
computed, not retrieved, it is separable from the words that fill it, and a
small model must condition on the frame — not on the word set — to emit
grammatical order.*

---

## 1. Levelt's grammatical encoding: from message to surface order

### 1.1 The linearization problem is stated before the model

Levelt (1981), *The speaker's linearization problem*, *Phil. Trans. R. Soc.
Lond. B* 295:305–315, is the founding statement. Speech is a serial channel;
a knowledge structure is not. "The speaker will have to decide on what to say
first, what to say next, and so on." Two constraints are named: **mutual
knowledge** (pragmatic, discourse-dependent) and **working memory** (biological,
domain-independent). The memory constraint is the one that matters for a small
model: the speaker's linearization strategy "will minimize memory load during
the process of formulating," by maximizing the connectivity of the discourse and
backtracking to stored "return addresses" first-in-last-out.

Ferreira & Henderson (1998), *Memory & Cognition* 26:88–96, tested this
directly with branching networks: speakers describe the shorter branch before
the longer, and the linear branch before the choice branch, and they compute
load **per branch, not for the whole network**. That is the *minimal-load /
easy-first* principle, and it is the behavioural signature of an incremental
linearizer rather than a whole-utterance planner.

### 1.2 Incrementality: fragments, not sentences

Kempen & Hoenkamp (1987), *An incremental procedural grammar for sentence
formulation*, *Cognitive Science* 11(2):201–258, is the classic implementation.
Their Incremental Procedural Grammar builds an utterance out of **syntactic
fragments**, each rendered from a fragment of conceptual content, handed to the
articulator while the rest is still being planned. The mechanism is
"Procedures + Stack": a team of syntactic procedures working in parallel on
small parts of the sentence, with a stack as their only communication channel.
Their own framing is the one this spec adopts:

> "A conceptual structure is linear-ized by splitting it up into a sequence of
> messages each of which is expressible in a complete or partial sentence."

Note the direction: the conceptual structure is split into a sequence of
messages; the linearization is a property of that split, not of a later
ordering pass over a finished sentence.

**The strategic qualifier.** Ferreira & Swets (2002), *J. Memory and Language*
46:57–84, showed that incrementality is **not architectural**: speakers plan
the whole utterance when they can, and only interleave planning and
articulation when a deadline forces them to. "Incrementality is not an
architectural property of the language production system; instead, it is a
parameter of production that is under speaker control." The design consequence
is a *variable planning grain*, not a fixed one-clause window.

### 1.3 Functional then positional: the two stages that produce order

Bock & Levelt (1994), *Language production: Grammatical encoding*, in
Gernsbacher (ed.), *Handbook of Psycholinguistics*, pp. 945–984, is the
canonical stage decomposition. Grammatical encoding has two sub-stages:

- **Functional processing** — *lexical selection* (identify the lemmas) and
  *function assignment* (assign grammatical functions: subject/nominative,
  object/accusative, dative, main verb). "The primary problem of function
  assignment is to specify which elements will serve as the subject of the
  incipient utterance and which, if any, will serve as objects of various
  kinds." The verb's argument structure organizes this: "A verb's specification
  of its normally expressed arguments may serve to organize function assignment
  around a unit that is roughly equivalent to a clause" (p. 966).
- **Positional processing** — *constituent assembly* (create the phrase
  structure and fix the linear order) and *inflection*. This is where word
  order is actually determined.

The load-bearing point for RAVANA: **function assignment precedes and is
separable from ordering.** A system that has the right content words but no
function assignment has no subject to put first and no object to put second.
That is the word-list output exactly.

### 1.4 The order is committed before articulation

Two independent lines of evidence say the commitment to order happens *before*
speech onset, not during:

- **Planning-scope studies.** Konopka & Meyer (2014), *Priming sentence
  planning*, *Cognitive Psychology* 73:1–40, manipulated how easy the event gist
  was to encode. When the gist was easy, speakers planned *both* characters
  before speaking (hierarchical incrementality); when it was hard, they planned
  one referent at a time (linear incrementality). The structural scaffold is
  built first and drives lexical retrieval, not the reverse: "the creation of a
  conceptual representation of the message allows the assembly of a structural
  scaffold, which will guide the selection of lexical elements in order."
  Wheeldon, Smith & Apperly (2011), *JEPLMC* 37(5):1051–1064, and Smith &
  Wheeldon (1999) converge: the minimally obligated planning unit is phrasal
  and lexically independent.
- **Intracranial recording.** Morgan, Devinsky, Doyle, Dugan, Friedman &
  Flinker (2024), *Decoding words during sentence production with ECoG
  reveals syntactic role encoding and structure-dependent temporal dynamics*,
  *Communications Psychology* (preprint doi:10.1101/2024.10.30.621177), is the
  cleanest production-side evidence. In sensorimotor cortex, words are activated
  in the order they are spoken. In IFG and MFG, **the order in which words are
  processed depends on the syntactic structure**: in passives, both nouns are
  encoded *simultaneously and sustained*, with subjects in IFG and objects in
  MFG. "Sentence production is not simply a sequence of serially executed
  single-word productions." The authors read this as a subtle pressure explaining
  why nearly all the world's languages place subjects before objects.

**Implication.** A decoder that is a pure next-token predictor over a linear
left context is architecturally the sensorimotor-cortex pattern — words in
spoken order — with no prefrontal structure-dependent stage in front of it.
That is not a training problem; it is a missing stage.

---

## 2. How word order is determined

### 2.1 Thematic role → syntactic function mapping

The mapping is many-to-many, not one-to-one. Bock & Levelt (1994) are explicit
that the correspondence is often close (`AGENT→subject`, `PATIENT→object`) but
has systematic deviations — "the verb *receive* requires the event role
RECIPIENT to be assigned to the grammatical function subject and not to the
indirect object." The mapping is therefore **at least partly verb-specific**,
which is why the verb lemma is retrieved before function assignment in the
lexically-driven model.

Two competing accounts of *what drives* the mapping:

- **Function mapping / direct mapping** (Bock & Levelt 1994; Bock, Loebell &
  Morey 1992): concepts compete for syntactic functions; animate and agent-like
  concepts win subject.
- **Argument selection / proto-roles** (Dowty 1991; tested for Italian by
  ... 2019, *Cognitive Processing*): thematic roles decompose into proto-agent /
  proto-patient features, and the concept with more proto-agent entailments
  becomes the subject.

Bock & Warren (1985), *Cognition* 21:47–67, is the foundational result on the
input side: **conceptual accessibility** (operationalized as imageability)
systematically affects grammatical function assignment — more accessible
referents get higher functions — and does *not* affect the order of nouns
inside a conjunction. McDonald, Bock & Kelly (1993), *Cognitive Psychology*
25(2):188–230, extended this: animacy affects role assignment but not
conjunction order; word length and prosody had no reliable effect on ordering.
Their conclusion is the one to design against: conceptual (meaning-based)
accessibility drives *role assignment*, and lexical (form-based) accessibility
does **not** drive ordering.

**This is the precise reason a content-word boost cannot linearize.** It
changes the activation of the concepts (which is the input to role assignment)
but supplies no role assignment and no function order. The measured result —
on-topic 1.00, unsupported 0, zero function words — is what an intact concept
set with a destroyed frame looks like.

### 2.2 Is order computed or retrieved as a construction?

Both, at different granularities, and the boundary is now fairly well drawn.

- **Computed.** The dual-path model (Chang, Dell & Bock 2006, *Psychological
  Review* 113(2):234–272) separates a **Meaning System** (event semantics plus
  temporary Role↔Concept bindings) from a **Sequencing System** (an SRN that
  learns abstract syntactic category sequences). Communication between them is
  restricted; the sequencing system learns syntax by predicting its own next
  input, error-driven. The model accounts for structural priming, the
  comprehension–production priming symmetry, and the development of abstract
  syntax. Crucially, the sequencing system is linked to *abstract event-role
  slots*, "but not to the word-meanings bound to those event roles" — which is
  how the model keeps lexical semantics out of syntax and still produces
  ordered output.
- **Retrieved.** Thematic-role *order* and *argument structure* are separately
  retrievable and separately primable. Fukumura & Yang (2024), *Interactive
  structure building in sentence production*, *Cognitive Psychology*
  148:101616, used cleft constructions to show that the thematic-role order of
  the prime (role→position mappings) can persist independently of its argument
  structure (role→function mappings), and that the two interact when the verb
  and construction repeat. Their **interactive structure-building account**
  replaces the staged model: "speakers build a sentence structure by choosing a
  thematic role order and argument structure interactively based on their prior
  co-occurrence." This is a direct challenge to the strict Bock & Levelt
  staging, and it is the strongest recent evidence that order is a *learned,
  retrievable* representation, not a pure computation over functions.

**The synthesis that matters for design:** the *frame* (which function comes
first, which second) is a learned, retrievable, primable object; the *fillers*
are selected competitively and slotted in. A model needs both, and the frame is
the thing RAVANA's decoder lacks.

### 2.3 The left IFG and the posterior STS

The neural evidence is comprehension-heavy and must be flagged as such, but
the convergence is real.

- **Left IFG (BA44/45) is the most reliable syntactic-encoding region in
  production.** Indefrey, Brown, Hellwig, Amunts, Herzog, Seitz & Hagoort
  (2001), *A neural correlate of syntactic encoding during speech production*,
  *PNAS* 98(10):5933–5936, used a restrictive scene-description paradigm with
  three levels of syntactic complexity (sentences > noun phrases > word lists)
  at constant conceptual demand. The left anterior Rolandic operculum,
  caudally adjacent to Broca's area, showed a **graded** response: sentences >
  noun phrases > isolated words (Z = 4.79, P = 0.019 corrected). Replicated
  with fMRI (Indefrey et al. 2004). This is the cleanest production-side
  dissociation of *syntactic encoding* from conceptual and phonological load.
- **The IFG's role is sequencing/linearization, not hierarchy.** Matchin &
  Hickok (2020), *The cortical organization of syntax*, *Cerebral Cortex*
  30(3):1481–1498, propose a two-process parcellation: **posterior middle
  temporal gyrus (pMTG)** does hierarchical lexical-syntactic structuring for
  both comprehension and production; **posterior IFG (pIFG, especially pars
  triangularis)** transforms those hierarchies into **morpho-syntactic
  sequences, primarily for production**. Matchin, Hickok and colleagues
  (2020), *Cerebral Cortex Communications* 1(1):tgaa029, confirmed the
  asymmetry: pIFG activation is driven more by production demands, PTL more by
  perception. This is the strongest current statement that *linearization is a
  distinct neural function from structure building* — which is the claim this
  spec is built on.
- **Word-order permutations recruit LIFG + pSTS.** Kinno, Kouno and
  colleagues (fMRI, Japanese scrambling; see the 2025 *Human Brain Mapping*
  study, doi:10.1002/hbm.70604) found that non-canonical orders elicit higher
  activation than canonical ones in the **left inferior frontal gyrus and the
  posterior superior temporal sulcus**, regardless of whether an argument or an
  adjunct is scrambled — a unified mechanism for word-order permutation.
  Kristensen et al. (2013), *Journal of Neurolinguistics*, found the same
  L-IFG/BA44–47 increase for object-initial vs subject-initial Danish clauses.
  Ben-Shachar et al. (2004) and Shetreet & Friedmann (2014) report the same
  LIFG + posterior temporal pattern for Hebrew OSV.
- **Production-side context effects land in the same region.** Funasaki &
  Yano (2024), *Language, Cognition and Neuroscience*, used fNIRS over F7
  (≈L-IFG) in an OSV production task and found the context × word-order
  interaction in L-IFG — the same region where the parallel comprehension study
  found it.

**Honest flag.** The LIFG is not a monolith: Fedorenko & Blank (2020),
*Broca's area is not a natural kind*, *Trends in Cognitive Sciences*
24(4):270–284, show that language-selective and domain-general multiple-demand
regions sit side by side within Broca's area, and that claims about "Broca's
area" must be recast in terms of those components. The Indefrey et al. (2001)
focus is anterior Rolandic operculum / BA6-adjacent, not classic BA44 — cite it
as *a* syntactic-encoding focus, not as "Broca's area does syntax."

---

## 3. Usage-based construction grammar: form–meaning pairings as the unit

### 3.1 The unit is the pairing, not the rule

Goldberg (2006), *Constructions at Work: The Nature of Generalization in
Language*, Oxford University Press, argues that the basic unit of language is
the **construction** — a form–meaning pairing, at any granularity from
morpheme to clause — and that a usage-based model stores both item-specific
information and the generalizations abstracted over it. The argument relevant
here is her treatment of argument structure: the *construction* carries the
relational meaning, and the verb's own semantics is integrated with it. This
inverts the Bock & Levelt priority: the frame is not derived from the verb's
argument structure alone; the frame is a stored object with its own meaning.

Tomasello (2003), *Constructing a Language: A Usage-Based Theory of Language
Acquisition*, Harvard University Press, gives the acquisition mechanism:
children start with **holophrases**, move to **pivot schemas** ("More __"),
then to **item-based constructions** ("Where's the X?", "I wanna X"), and only
later to abstract constructions. The operations are **usage-based syntactic
operations** (Langacker's "symbolic integration"): the child "cuts and pastes"
already-constructed pieces of language together, filling a new constituent into
a slot. The slot is defined by the communicative function of the schema, and
the filler is constrained by that function — "the communicative function of the
item in a slot is thus constrained by the overall communicative function of the
schema, but it is still somewhat open."

**This is the mechanism RAVANA needs and does not have.** The decoder has no
schema with a slot. It has a vocabulary and a next-token distribution.

### 3.2 Novel sentences are produced by slot-filling, not rule application

The strongest evidence that slot-filling is real and fast:

- **Goldberg (2006), Ch. 4:** novel phrasal constructions can be learned and
  generalized after ~3 minutes of exposure. The mechanism is
  **superimposition / insert** — "filling a slot in a frame (or construction)
  with a phrase that fits the semantic and syntactic requirements of that slot."
- **Statistical preemption and partial productivity** (Goldberg 2006, Ch. 6;
  modeled by ... 2016, *AAAI*): whether a slot is productive depends on the
  openness of the slot (the range of witnessed fillers) and on statistical
  preemption (a competing construction that already covers the meaning). This is
  a *graded, learned* constraint on slot-filling — exactly the kind of thing a
  small model can implement as a weight.
- **Arnon & Christiansen (2017), *Topics in Cognitive Science* 9(3):621–636**,
  and McCauley & Christiansen (2017), *Topics in Cognitive Science*
  9(3):637–652: multiword units are real and are learned by chunking and
  undersegmentation; children rely on them more than adult L2 learners. The
  Chunk-Based Learner model reproduces corpus statistics from chunk counts
  alone.

### 3.3 The honest counter-evidence

Construction grammar is not a settled theory of production, and the spec should
not pretend otherwise. A formal review (van Trijp & ... 2008, ILLC report
X-2008-05) found that *Constructions at Work* "proposes no model or theory that
can produce an open-ended number of novel utterances based on a finite amount
of input" — no input–output procedure, no definition of the productive unit, no
combination operation. That criticism is largely answered by the computational
usage-based literature (Bannard, Lieven & Tomasello; Bod's SPL; McCauley &
Christiansen's Chunk-Based Learner; Steels' Fluid Construction Grammar), but it
is a real gap in the *foundational* text. Grade the strong claim **[M]**, not
[S].

---

## 4. Retrieved multi-word sequences vs. computed sentences

### 4.1 Pawley & Syder's two puzzles

Pawley & Syder (1983), *Two puzzles for linguistic theory: Nativelike
selection and nativelike fluency*, in Richards & Schmidt (eds.), *Language and
Communication*, pp. 191–227, Longman, is the classical statement. Two
capacities resist a rule-based account:

1. **Nativelike selection** — "the ability of the native speaker routinely to
   convey his meaning by an expression that is not only grammatical but also
   nativelike," selecting the natural variant from among grammatical
   paraphrases.
2. **Nativelike fluency** — "the native speaker's ability to produce fluent
   stretches of spontaneous connected discourse," despite severely limited
   capacity for encoding novel speech in advance.

Their answer is a **lexicalized sentence stem**: "a unit of clause length or
longer whose grammatical form and lexical content is wholly or largely fixed,"
retrieved "as wholes or as automatic chains from the long term memory," which
"minimizes the amount of clause-internal encoding work." They also note the
"one clause at a time constraint" on fully novel encoding, from the distribution
of dysfluencies.

### 4.2 The modern resolution: a dual route, not a victory for either side

The retrieved-vs-computed dichotomy has been substantially dissolved by
evidence that both routes exist and that the empirical criteria used to
distinguish them are unreliable:

- **Frequency effects without a threshold.** Frequent four-word phrases are
  processed faster than less frequent ones with no evidence of a frequency
  threshold, and idioms prime their construction as well as non-idiomatic
  phrases — so compositional phrases behave like stored ones and non-
  compositional ones show internal structure (see the review in *Journal of
  Memory and Language* / *Language Teaching* research agendas; Arnon &
  Christiansen 2017).
- **Chunking as the learning mechanism.** Arnon & Christiansen (2017) and
  McCauley & Christiansen (2017) argue multiword representations *emerge* from
  chunking and undersegmentation rather than being stipulated. This is a
  single-system account: the same statistical mechanism produces both the
  chunks and the generalizations.
- **The brain evidence is compatible with both.** Morgan et al. (2024) found
  that words share cortical representations across single-word and sentence
  tasks — the lexical route is shared — while prefrontal cortex additionally
  encodes structure-dependent order. A low-activity cortical syntax network
  (Morgan et al. 2024, *bioRxiv* doi:10.1101/2024.06.20.599931) processes
  syntax "mostly to the exclusion of words and meaning," which is a distinct
  syntactic system, not a retrieval system.

**The design-relevant conclusion:** the debate is not "retrieved or computed."
It is "retrieved *and* computed, at different granularities, with the frame
retrieved/learned and the fillers selected." A model that retrieves whole
sentences cannot generalize; a model that computes everything cannot be fluent.
RAVANA's decoder is at the "computes everything, retrieves nothing" pole, and
its output is the predicted consequence.

---

## 5. The neural basis of syntactic ordering specifically

### 5.1 Friederici's model and its stages

Friederici (2002), *Towards a neural basis of auditory sentence processing*,
*Trends in Cognitive Sciences* 6(2):78–84, proposes three functionally and
temporally separable phases:

| Phase | Window | Process | ERP correlate |
|---|---|---|---|
| 1 | ~100–300 ms | Word-category access and **initial syntactic structure building** (local phrase structure) | ELAN |
| 2 | ~300–500 ms | **Lexical-semantic and morphosyntactic processes; thematic role assignment** | N400 |
| 3 | ~500–1000 ms | **Integration, re-analysis, repair** | P600 |

The anatomical claim is a division of labour: **temporal regions identify**
word-level syntactic and semantic information (retrieval of memorized forms);
**frontal regions construct** syntactic and semantic relations. The pathway
account (Friederici 2011, *Nature Reviews Neuroscience*; Friederici 2017,
*Language in Our Brain*) adds two dorsal pathways — SLF/AF for
auditory-to-motor mapping and pSTG↔BA44 for complex syntax — and two ventral
pathways, with the uncinate/fronto-orbital route for "elementary combinatorics."

**What transfers to production.** The phase ordering is a *comprehension*
model. But the anatomical division of labour — temporal = retrieve, frontal =
construct — is stated as a general property of the language system, and
Hagoort (2013, *Frontiers in Psychology* 4:416) explicitly notes that "when
speaking the recruitment of these networks mostly operates in the reversed
temporal order." The retrieval-then-construction separation is the transferable
part; the millisecond windows are not.

### 5.2 MUC: Memory, Unification, Control

Hagoort (2005, *Trends in Cognitive Sciences* 9(9):416–423; 2013, *Frontiers in
Psychology* 4:416; 2019, *Annual Review of Neuroscience* 42:425–446) is the
most directly implementable neural model for this spec:

- **Memory** — temporal cortex and parts of parietal cortex store word form,
  word meaning, and **syntactic templates**. "These areas store information
  about word form, word meanings and the syntactic templates."
- **Unification** — left inferior frontal cortex (BA44/45/47/6) "generates
  larger structures from the building blocks that are retrieved from memory."
  There is a spatial gradient: **semantic unification recruits BA47/45;
  syntactic unification has its focus in BA45/44**; phonological processes
  recruit BA44 and ventral BA6.
- **Control** — prefrontal cortex and ACC manage target language, turn-taking,
  register, attention.

**This is the cleanest statement of the architecture RAVANA needs.** The
"syntactic templates" in Memory are the linearization frames. Unification is
the operation that binds a retrieved template to a message. And the gradient
gives a concrete prediction: a system that only has semantic unification (a
content-word bias) and no syntactic unification (no frame) will produce
precisely the observed output — right concepts, no syntax.

### 5.3 Word-order processing specifically

Beyond the scrambling literature in §2.3:

- **Bornkessel & Schlesewsky (2006), *Psychological Review* 113(4):787–821**,
  the extended Argument Dependency Model: thematic roles are assigned in two
  phases — an early **compute prominence / compute linking** stage using
  morphosyntactic and prominence cues (animacy, definiteness, case, linear
  order) linked to the verb's argument structure, and a later **generalized
  mapping** stage integrating plausibility. The model is explicitly
  cross-linguistic and predicts that *linear order is one input to role
  assignment*, not the output of it. (Note: this is a comprehension model; the
  production-side analogue is the interactive structure-building account of
  Fukumura & Yang 2024.)
- **Zaccarella & Friederici (2015), *Frontiers in Psychology* 6:1818**:
  two-word phrase structure (a determiner + pseudo-noun) activates a specific
  sub-part of BA44, while mere word accumulation activates the frontal
  operculum and anterior insula. Hierarchical combination and accumulation are
  neurally dissociable.
- **Morgan et al. (2024)** is the best production-side evidence for a *distinct
  syntactic system*: 125 syntax-sensitive electrodes, only 6 of which were
  found by the classic sentence-vs-list contrast, and syntax encoded
  independent of overall activity level ("low-activity coding"). Their
  interpretation: "syntax is at some level represented independent of words and
  meaning."

### 5.4 What is *not* established

- **No production-side temporal map.** The Friederici windows are
  comprehension. Indefrey & Levelt (2004), *Cognition* 92:101–144, give
  production windows for *single words* (conceptual ~0–200 ms, lemma
  ~200–275 ms, phonological code ~250–330 ms, syllabification ~330–455 ms,
  phonetic ~455–600 ms) but not for the ordering of a multi-word utterance.
- **The exact cortical locus of linearization is contested.** Matchin & Hickok
  (2020) pIFG; Indefrey et al. (2001) anterior Rolandic operculum; Hagoort
  BA45/44; Friederici BA44. These are adjacent but not identical, and
  Fedorenko & Blank (2020) warn that the anatomical labels are coarser than the
  functional reality.
- **The "syntax is independent of semantics" claim is stronger than the data.**
  Morgan et al.'s low-activity coding result is a single ECoG study in ten
  patients. It is suggestive, not settled.

---

## 6. What this means for RAVANA's decoder

### 6.1 The diagnosis, stated in the literature's terms

The decoder is a **Sequencing System with no Meaning System and no frame**.
Chang, Dell & Bock (2006) predict exactly this failure: their model's
sequencing system learns abstract syntactic category sequences from its own
prediction error, but it can only produce ordered output when the meaning
system supplies Role↔Concept bindings and the message supplies event semantics.
Without them, the sequencing system still produces *a* sequence — the most
probable token sequence under its training distribution — and that sequence is
the fixed filler register measured in
`docs/DECODER_FIXED_FILLER_REGISTER.md`:

```
such 24, one 23, used 22, other 20, there 20, unfolds 16, often 16, ...
top-10 tokens = 39.4% of all tokens; 133 distinct / 457 total (TTR 0.29)
```

That is not a broken model. It is a model doing next-token prediction with no
message to predict *from*. The brain's equivalent is an intact formulator with
an empty semantic store — the semantic-dementia signature
(`docs/BRAIN_NOVEL_SUBJECT_GROUNDING.md` §1).

### 6.2 Why the logit boost was guaranteed to fail

`docs/ANCHOR_BOOST_DESTROYS_SYNTAX.md` measured the anchor boost at strengths
2.0–12.0: on-topic 1.00, unsupported 0, and every output a word list with zero
function words. The literature explains this exactly:

- The boost raises the activation of **content words** — the input to
  *functional processing* (lexical selection) and, via conceptual
  accessibility, to *function assignment* (Bock & Warren 1985).
- It does not supply a **function assignment** and does not supply a
  **positional frame** (Bock & Levelt 1994). So there is no subject slot and no
  object slot; there is a bag of concepts.
- The closed-class frame is *computed from the frame*, not retrieved
  (Levelt 1989; Bock 1989). With no frame, there is nothing to compute the
  determiners and prepositions from. Hence zero function words.
- The ordering that does appear is the decoder's own learned bigram
  statistics, which have no relation to the message. McDonald, Bock & Kelly
  (1993) showed that lexical/form-based accessibility does **not** drive
  ordering — so a token-level bias cannot create order even in principle.

`docs/BOOST_STRENGTH_NOT_THE_LEVER.md` reached the same conclusion from the
other direction: the logit vector is the wrong level of intervention. The
literature now says why: **the logit vector is the output of lexical selection;
linearization is upstream of it.**

### 6.3 The two things RAVANA already has, and the one it does not

| Brain mechanism | RAVANA module | Status |
|---|---|---|
| Message with Role↔Concept bindings | `ravana/src/ravana/learn/macroplanning.py` — `Message`, `RoleBinding`, `build_message`, `select_lexical_concepts` (Luce ratio), `_role_for_relation` | **Implemented, tested, imported by no production module** |
| Committed proposition `(subject, relation, target)` | `ravana/src/ravana/learn/committed_proposition.py` — `commit`, `decide`, `CPGDecision`, `CPGConfig.relation_words` | Implemented; `_cpg_enabled` defaults `False` (`engine.py:1291`) |
| Syntactic frame with function order | `ravana/src/ravana/language/syntactic_cell_assembly.py` — `SyntacticFrame`, `bind_to_sentence`, `SEQUENCE_PRIORS` | Implemented; `SEQUENCE_PRIORS` is a static dict, not learned from the message |
| Surface realization from the frame | `ravana/src/ravana/language/surface_realizer.py` — `realize`, `_build_noun_phrase`, `_apply_variant`, `_apply_agreement` | Implemented and **used by the narrative path** (`response_gen.py:1114`) |
| Competitive lemma selection | `VerbLexicon.select_verb` (`verb_lexicon.py:500`), `BasalGangliaGate.select_concept` | Implemented; the gate selects concepts, not tokens |
| **Linearization conditioned on the message** | — | **Absent** |

The narrative path (`_generate_narrative_paragraph`, `response_gen.py:987`)
already does the right thing for the subjects it handles: it calls
`bind_to_sentence` → `SurfaceRealizer.realize` and produces grammatical
sentences. It scores on-topic 0.242 vs the decoder's 0.069
(`docs/DECODER_ONLY_NUMBERS.md`). The decoder path does not do this; it calls
`NeuralDecoder.generate` with a token boost.

**The gap is not a missing component. It is a missing conditioning
representation.** The decoder is never given the frame.

---

## 7. Concrete, falsifiable design

### 7.1 The representation the decoder must condition on

**Design claim.** A small decoder emits grammatical order if and only if it is
conditioned on an explicit **ordered role sequence** derived from the committed
proposition — not on a bag of content words, and not on a single subject
vector.

**This is now implemented** (`ravana/src/ravana/learn/linearization.py`, commit
`057bbae`) and wired behind `_plan_prefix_enabled` (`engine.py:1304`,
`response_gen.py:1726-1767`), with a prototype measurement harness
(`scripts/prototype_linearization.py`, `scripts/measure_plan_vs_boost.py`).
The representation it builds is a `SurfacePlan`:

```
SurfacePlan:
    subject       : str          # the message's topic
    construction  : str          # categorize | cause | attribute | locate | relate
    tokens        : List[str]    # the construction's frame, slots filled, IN ORDER
    roles         : List[str]    # the parallel role sequence
    fill          : Dict[str,str] # role -> filler, for auditability
```

`tokens` is the load-bearing field, and it is a **construction** in the
usage-based sense (Goldberg 2006; Tomasello 2003): a form–meaning pairing with
slots, where the fillers come from the message. The construction inventory is
five entries covering the graph's typed relation vocabulary:

| relation_type | construction | plan |
|---|---|---|
| `is_a`, `part_of`, `member_of` | categorize | `<theme> is a <theme>` |
| `causes`, `leads_to`, `affects` | cause | `<agent> leads to <ground>` |
| `has_property`, `used_for` | attribute | `<theme> has <relation>` |
| `located_in`, `part`, `time` | locate | `<theme> is located in <ground>` |
| anything else | relate | `<theme> is related to <ground>` |

The chain is `macroplanning.build_message` (spreading activation → Luce-ratio
selection → role assignment) → `linearization.message_to_plan` (relation
semantics → construction → slot filling) → the decoder **continues from** the
plan. The subject always fills slot 0 (it is the utterance's topic). A
construction with no binding for a slot is dropped rather than rendered with a
hole, so the caller falls back.

The general form, of which `SurfacePlan` is the first cut:

```
LinearizationFrame:
    subject_concept   : str            # the message's topic
    relation          : str            # the committed edge's relation_type
    target_concept    : str            # the committed edge's target
    role_order        : List[RoleSlot] # ORDERED; this is the linearization
    surface_skeleton  : List[str]      # function words + slot markers, in order
    focus_role        : str            # which role is in focus (default: theme)

RoleSlot:
    role       : str   # agent | theme | relation | ground
    concept    : str   # the bound concept, or "" for a function-word slot
    is_content : bool  # True = open-class filler; False = closed-class frame word
```

The key property is that **the closed-class frame words are part of the plan,
not selected from the message**. `is`, `a`, `leads`, `to`, `has`, `located`,
`in` are the construction's own words, computed from the frame (Levelt 1989;
Bock 1989). That is precisely what a content-word boost cannot supply.

### 7.2 How the decoder consumes it

Three changes, in order of cost. All are additive; none requires retraining the
GRU. **Change (a) is implemented** (commit `057bbae`, uncommitted wiring in
the working tree); (b) and (c) are not.

**(a) Seed the GRU with the plan, not with `<bos>`.**
`NeuralDecoder.generate` accepts a `prefix: Optional[List[int]]`
(`neural_decoder.py:409-421`, added in `057bbae`). The prefix is fed through
the GRU teacher-forced, so the hidden state at the first free step encodes the
plan's grammatical frame; the prefix tokens are not re-emitted, and the caller
concatenates them back onto the continuation (`response_gen.py:1853-1859`).
This is a **prefix continuation**, not a bias — which is the whole point.
The prior `initial_emb` / `persistent_emb` path (`response_gen.py:1762-1793`)
derives the seed from `W_sm @ attribute_vector(GloVe(subject))`, which
`docs/CONDITIONING_COLLAPSES_TO_A_POINT.md` measured as near-null (pairwise
cosine 0.9990 across subjects; `W_sm` effective rank 9 of 65). The plan prefix
replaces the *content* of the seed with a categorical, ordered structure that
the collapsed signature cannot represent.

Two implementation details worth preserving:
- The `step == 0` "force first word to be a content word" mask
  (`neural_decoder.py:653`) is **disabled when a prefix is present** — the
  prefix already opened with the subject, so masking the first free step to
  content words would fight the frame.
- The prefix is gated on `plan_is_committed(plan)`, which rejects a vacuous
  `relate` construction and any plan whose filler is a numeral or function
  word. This is the guard against "fluent, grammatical, false" output, and it
  is the direct consequence of the prototype's blocking finding (below).

**(b) Constrain the candidate set to the open slots, in order.** *(not
implemented)* At step *t*, the admissible token set is:

```
admissible(t) = { token whose word == skeleton[t].concept }        if skeleton[t].is_content
               { tokens in the closed-class set for skeleton[t] }   otherwise
```

Combined with the existing `content_word_ids` / `token_boost` machinery
(`neural_decoder.py:516-539`), this is a **mask**, not a boost. The measured
lesson of `docs/ANCHOR_BOOST_DESTROYS_SYNTAX.md` is that a boost at any
strength that moves the output replaces the distribution; a mask does not
replace the distribution, it restricts the support. The decoder still chooses
*which* determiner and *which* inflection, from its own distribution, within
the frame. The prefix path already does the weaker version of this: it fixes
the frame's own words and leaves the free steps open.

**(c) Condition the transition prior on the role sequence, not on the last
word.** *(not implemented)* `CerebellarNgram.predict_next(from_word, to_word)`
is a word bigram prior. Add a role-transition prior `P(next_role |
previous_role, relation_type)` — which is exactly
`SyntacticCellAssembly.SEQUENCE_PRIORS` (`syntactic_cell_assembly.py:75-86`),
currently a static dict that is never consulted during decoding. This is the
"learned syntactic relationships in the SRN" of Chang et al. (2006), and it is
the mechanism that makes the order *learned* rather than hard-coded.

### 7.3 The blocking finding: the plan is only as good as the message

`docs/LINEARIZATION_PROTOTYPE.md` records the first measurement of the plan
path, and its result is the most important thing this spec has to say about
sequencing the work.

On the real persisted graph, the plans are **grammatical and semantically
wrong**:

```
gravity    gravity is related to two
ocean      ocean is located in sea
species    species is related to birds
time       time is related to making
sky        sky is related to blue
music      music is a sound          <-- correct, from a typed is_a edge
```

`gravity is related to two` and `time is related to making` are well-formed
clauses about nothing. The cause is upstream: `macroplanning.build_message`
selects by spreading activation over edge weight and confidence, and the
top-scoring edges for `gravity` and `time` point at `two` and `making`. The
linearization is doing its job — **it faithfully realizes a bad message.**

**This is the strongest argument in the whole document for fixing the message
before the formulator.** A perfect formulator over a bad message produces
fluent, grammatical, false sentences, which is *worse* than word salad because
it is harder to detect. Word salad is visibly broken; "gravity is related to
two" is not. The prototype's own conclusion is that the next cut must score the
plan before using it: reject vacuous relations, prefer typed edges over
activation rank, and reject non-concept fillers. `plan_is_committed` implements
the first and third; the second (prefer typed edges) is not yet implemented.

Note the resonance with the measured decoder failure: the decoder emits a word
list because it has no message; the plan path emits grammatical falsehoods
because it has a bad one. Both are upstream of the formulator. Neither is a
decoder-capacity problem.

### 7.4 The minimal falsifiable prediction

**Prediction P1.** Conditioning the decoder on the plan prefix (a) will raise
the **function-word share** and the **frame rate** of decoder output from the
baseline, *without* lowering on-topic below the CPG baseline (0.227), and will
reduce unsupported assertions below the relation-gated 1.118.

`scripts/measure_plan_vs_boost.py` is the instrument. It runs three arms on the
same persisted engine and subjects — BASELINE (the decoder's own distribution),
WORD_BOOST (logit boost at the 9.0 knee), PLAN (the prefix path) — and reports
paired deltas on `on_topic`, `unsupported`, `func_share`, `has_verb`, and
`frame`. The `frame` score is `has_verb × min(1, func_share / (len/3))`: a
clause needs both grammatical glue and a predicate. **This is the axis the
original 9.0-knee sweep could not see**, because its metric
(`scripts/train.py:559`: `in-vocab × distinct-1 × on-topic`) uses the type-token
ratio, which is 1.0 for both a word list and a sentence.

**Falsifier F1.** If function-word share rises but on-topic collapses below
0.10, the frame is over-constraining: the prefix is rejecting the content
words the graph supports. Fix by relaxing the prefix to a weighted prior rather
than a fixed continuation.

**Falsifier F2.** If function-word share does not rise, the decoder is ignoring
the conditioning — which is the `W_sm` collapse all over again, and the fix is
to bypass the attribute encoder entirely (condition on the raw GloVe vector +
the categorical frame, per `docs/CONDITIONING_COLLAPSES_TO_A_POINT.md`
candidate fix 1).

**Falsifier F3.** If the output becomes grammatical but *templated* (the same
construction for every subject, no variation), the model has over-fit to the
canonical order and the construction-retrieval route (§3, §4) is missing. The
fix is to add the learned-frame route: `SyntacticCellAssembly.learned_frames`
(already implemented, `syntactic_cell_assembly.py:107`, capped at 512) supplies
a retrieved frame when one exists, and the computed plan is the fallback.

**Falsifier F4.** If the model produces grammatical order but the *relation*
word is wrong (e.g. "gravity is a kind of force" when the graph says `causes`),
the function assignment is right and the relation binding is wrong — which is
the `CPGConfig.relation_words` failure mode, not a linearization failure. Keep
the two measurements separate.

**Falsifier F5 (the prototype's finding, and the one most likely to fire).**
If the plan is grammatical but its filler is a non-concept (`gravity is related
to two`) or rests on a vacuous `related_to` edge, the failure is in
`build_message`, not in `message_to_plan`. The fix is upstream: gate the plan
on the edge being **typed**, not on its activation rank. `plan_is_committed`
rejects the vacuous relation and the non-concept filler, so this falsifier
should be caught before emission — but only if the guard is not bypassed.

### 7.5 What to measure, and how to avoid the anchor-boost mistake

`docs/ANCHOR_BOOST_DESTROYS_SYNTAX.md` is the cautionary tale: the 9.0 knee was
fitted on `in-vocab ratio × distinct-1 × on-topic` (`scripts/train.py:559`),
and `distinct-1` (type-token ratio) is 1.0 for both a word list and a sentence.
**Any new metric must include a syntactic term.** Minimum instrument:

| Metric | Definition | Why |
|---|---|---|
| `func_share` | fraction of tokens in a closed-class set | the anchor boost drove this to 0 |
| `frame` | `has_verb × min(1, func_share/(len/3))` | direct test of clause-vs-list |
| `plan_skeleton_match` | fraction of outputs whose token sequence matches the plan's `tokens` (content slots wildcarded) | direct test of linearization |
| `on-topic` | existing | must not regress below 0.227 |
| `unsupported` | existing | must fall below 1.118 |
| `distinct-2` | bigram type-token ratio | the metric that should have caught the anchor failure |

Report all six. A result that improves `on-topic` and `distinct-2` while
leaving `func_share` at 0 is the anchor-boost result and must be rejected. A
result that improves `func_share` and `frame` while `plan_skeleton_match` is
at chance means the decoder is ignoring the plan and the framing is
decorative.

### 7.6 What NOT to do

- **Do not raise the logit boost.** Measured flat-to-negative at 2.5 → 5.0
  (`docs/BOOST_STRENGTH_NOT_THE_LEVER.md`) and syntax-destroying at 2.0–12.0
  (`docs/ANCHOR_BOOST_DESTROYS_SYNTAX.md`). The literature explains why:
  lexical-selection activation is not a linearization signal.
- **Do not add logit suppression.** Measured to collapse output into worse
  salad (`docs/NOGO_SUPPRESSION_FAILS.md`); the brain's NoGo inhibits
  *competing plans*, not lexical items.
- **Do not condition on a denser subject vector.** The conditioning path is
  already near-null (`docs/CONDITIONING_COLLAPSES_TO_A_POINT.md`); more of a
  null signal is more noise.
- **Do not retrieve whole sentences.** Pawley & Syder's fluency puzzle is real,
  but a retrieved sentence cannot be grounded to a novel subject, and RAVANA's
  constraint set forbids a sentence lookup table. Retrieve *frames*, not
  sentences.
- **Do not treat this as a decoder-capacity problem.** The GRU is 256-hidden,
  vocab 15 000, embed 75; the narrative path produces grammatical text with no
  neural network at all. The deficit is upstream.

---

## 8. Settled vs. contested

### Settled (replicated, convergent, or mechanistically forced)

1. **There is a pre-lexical message, and it is not a word sequence.** Levelt
   1989; Bock & Ferreira 2014; Konopka & Brown-Schmidt 2014. The message is a
   relational structure with thematic-role↔concept bindings.
2. **Linearization is a distinct operation from message construction.** Levelt
   1981; Kempen & Hoenkamp 1987; Ferreira & Henderson 1998. It imposes a serial
   order on a non-linear structure, under a working-memory/minimal-load
   constraint.
3. **Grammatical encoding has a functional stage (lexical selection + function
   assignment) and a positional stage (constituent assembly + inflection).**
   Bock & Levelt 1994. Function assignment precedes ordering.
4. **Function words and inflections are computed from the frame, not retrieved
   as lexical items.** Levelt 1989; Bock 1989 (closed-class immanence, with the
   Lapointe & Dell modification for free-standing function words).
5. **Conceptual accessibility affects role assignment; lexical/form accessibility
   does not drive ordering.** Bock & Warren 1985; McDonald, Bock & Kelly 1993.
6. **Incrementality is real but strategic, not architectural.** Kempen &
   Hoenkamp 1987; Ferreira & Swets 2002; Konopka & Meyer 2014.
7. **The order is committed before articulation, and is structure-dependent.**
   Konopka & Meyer 2014; Wheeldon, Smith & Apperly 2011; Morgan et al. 2024
   (ECoG: prefrontal word-processing order depends on syntactic structure).
8. **Multiword sequences are represented and used, and novel sentences are
   produced by slot-filling into a learned schema.** Pawley & Syder 1983;
   Goldberg 2006; Tomasello 2003; Arnon & Christiansen 2017; McCauley &
   Christiansen 2017.
9. **Syntactic encoding during production has a neural correlate in the left
   inferior frontal / Rolandic opercular region, dissociable from conceptual
   and phonological load.** Indefrey et al. 2001 (graded sentences > phrases >
   words), replicated 2004.
10. **Word-order permutation recruits LIFG and pSTS.** Kinno et al.;
   Kristensen et al. 2013; Ben-Shachar et al. 2004; Shetreet & Friedmann 2014;
   Funasaki & Yano 2024.
11. **Linearization/sequencing is neurally dissociable from hierarchical
    structuring.** Matchin & Hickok 2020; Matchin et al. 2020 (pIFG production-
    side sequencing; pMTG hierarchical structuring for both).

### Contested (live disputes, or evidence thinner than the claim)

1. **Whether order is computed or retrieved.** The staged Bock & Levelt (1994)
   account vs. Fukumura & Yang's (2024) interactive structure-building account,
   in which thematic-role order and argument structure are selected
   interactively from prior co-occurrence. The evidence currently favours a
   hybrid: the frame is learned/retrievable, the fillers are selected.
2. **Whether construction grammar is a complete theory of production.** The
   van Trijp review's criticism — no input–output procedure, no combination
   operation — is largely answered by computational usage-based models but not
   by the foundational text. Grade [M].
3. **The strong "syntax is independent of semantics" claim.** Morgan et al.
   (2024) low-activity coding is one ECoG study in ten patients. Suggestive,
   not settled.
4. **The exact cortical locus of linearization.** pIFG (Matchin & Hickok 2020)
   vs. anterior Rolandic operculum (Indefrey et al. 2001) vs. BA45/44 (Hagoort
   2013) vs. BA44 (Friederici 2002). Adjacent, not identical; and Fedorenko &
   Blank (2020) show the anatomical labels are coarser than the functional
   reality.
5. **Whether the IFG's role is sequencing or syntactic working memory.**
   Matchin & Hickok (2020) sequencing; the classic account (Friederici) complex
   syntax; others working memory. The production/perception asymmetry is
   supported, the specific computation is not.
6. **Whether the message is cast in a language of thought or in lexical
   concepts.** Levelt 1989 vs. the "linguistic creatures think directly in
   lexical concepts" alternative. Unresolved, and it does not change the design.
7. **Whether Friederici's phase windows transfer to production.** They are a
   comprehension model. The anatomy transfers; the milliseconds do not.
8. **Whether the decoder's failure is a capacity failure or a conditioning
   failure.** The evidence here (`docs/CONDITIONING_COLLAPSES_TO_A_POINT.md`,
   `docs/WHAT_THE_DECODER_ROWS_CONTAIN.md`) strongly favours conditioning, but
   the two are not fully dissociated: a 256-hidden GRU trained on 512 short
   definitions may also lack the capacity to represent a frame. The design in §7
   is testable either way — if the frame conditioning fails, capacity is the
   next hypothesis.

---

## 9. Citations

- **Arnon, I., & Christiansen, M. H.** (2017). The role of multiword building
  blocks in explaining L1–L2 differences. *Topics in Cognitive Science*, 9(3),
  621–636. doi:10.1111/tops.12271
- **Ben-Shachar, M., Hendler, T., Kahn, I., Ben-Bashat, D., & Grodzinsky, Y.**
  (2004). The neural reality of syntactic transformations: Evidence from
  fMRI. *Psychological Science*, 15(11), 806–814.
- **Bock, J. K.** (1982). Toward a cognitive psychology of syntax: Information
  processing contributions to sentence formulation. *Psychological Review*,
  89(1), 1–47.
- **Bock, J. K.** (1986). Syntactic persistence in language production.
  *Cognitive Psychology*, 18(3), 355–387.
- **Bock, J. K.** (1989). Closed-class immanence in sentence production.
  *Cognition*, 31(2), 163–186. doi:10.1016/0010-0277(89)90022-X
- **Bock, J. K., & Ferreira, V. S.** (2014). Syntactically speaking. In
  M. Goldrick, V. S. Ferreira, & M. Miozzo (Eds.), *The Oxford Handbook of
  Language Production* (pp. 21–46). Oxford University Press.
  doi:10.1093/oxfordhb/9780199735471.013.008
- **Bock, J. K., & Levelt, W. J. M.** (1994). Language production: Grammatical
  encoding. In M. A. Gernsbacher (Ed.), *Handbook of Psycholinguistics*
  (pp. 945–984). Academic Press.
- **Bock, J. K., Loebell, H., & Morey, R.** (1992). From conceptual roles to
  structural relations: Bridging the syntactic cleft. *Psychological Review*,
  99(1), 150–171.
- **Bock, J. K., & Warren, R. K.** (1985). Conceptual accessibility and
  syntactic structure in sentence formulation. *Cognition*, 21(1), 47–67.
  doi:10.1016/0010-0277(85)90023-X
- **Bornkessel, I., & Schlesewsky, M.** (2006). The extended Argument
  Dependency Model: A neurocognitive approach to sentence comprehension across
  languages. *Psychological Review*, 113(4), 787–821.
  doi:10.1037/0033-295X.113.4.787
- **Chang, F., Dell, G. S., & Bock, K.** (2006). Becoming syntactic.
  *Psychological Review*, 113(2), 234–272. doi:10.1037/0033-295X.113.2.234
- **Christiansen, M. H., & Chater, N.** (2016). *Creating Language: Integrating
  Evolution, Acquisition, and Processing*. MIT Press.
- **Dell, G. S.** (1986). A spreading-activation theory of retrieval in sentence
  production. *Psychological Review*, 93(3), 283–321.
  doi:10.1037/0033-295X.93.3.283
- **Dowty, D. R.** (1991). Thematic proto-roles and argument selection.
  *Language*, 67(3), 547–619.
- **Fedorenko, E., & Blank, I. A.** (2020). Broca's area is not a natural kind.
  *Trends in Cognitive Sciences*, 24(4), 270–284.
  doi:10.1016/j.tics.2020.01.001
- **Ferreira, F., & Henderson, J. M.** (1998). Linearization strategies during
  language production. *Memory & Cognition*, 26(1), 88–96.
  doi:10.3758/BF03211372
- **Ferreira, F., & Swets, B.** (2002). How incremental is language production?
  Evidence from the production of utterances requiring the computation of
  arithmetic sums. *Journal of Memory and Language*, 46(1), 57–84.
  doi:10.1006/jmla.2001.2797
- **Friederici, A. D.** (2002). Towards a neural basis of auditory sentence
  processing. *Trends in Cognitive Sciences*, 6(2), 78–84.
- **Friederici, A. D.** (2011). The brain basis of language processing: From
  structure to function. *Physiological Reviews*, 91(4), 1357–1392.
- **Friederici, A. D.** (2017). *Language in Our Brain: The Origins of a Uniquely
  Human Capacity*. MIT Press.
- **Fukumura, K., & Yang, F.** (2024). Interactive structure building in sentence
  production. *Cognitive Psychology*, 148, 101616.
  doi:10.1016/j.cogpsych.2023.101616
- **Funasaki, N., & Yano, M.** (2024). Producing non-basic word orders in
  (in)felicitous contexts: Evidence from pupillometry and fNIRS. *Language,
  Cognition and Neuroscience*. doi:10.1080/23273798.2024.2404178
- **Goldberg, A. E.** (2006). *Constructions at Work: The Nature of
  Generalization in Language*. Oxford University Press.
- **Hagoort, P.** (2005). On Broca, brain, and binding: A new framework.
  *Trends in Cognitive Sciences*, 9(9), 416–423.
- **Hagoort, P.** (2013). MUC (Memory, Unification, Control) and beyond.
  *Frontiers in Psychology*, 4, 416. doi:10.3389/fpsyg.2013.00416
- **Hagoort, P.** (2019). The neurobiology of language beyond single words.
  *Annual Review of Neuroscience*, 42, 425–446.
- **Hagoort, P., & Indefrey, P.** (2014). The neurobiology of language beyond
  single words. *Annual Review of Neuroscience*, 37, 347–362.
- **Indefrey, P., Brown, C. M., Hellwig, F., Amunts, K., Herzog, H., Seitz,
  R. J., & Hagoort, P.** (2001). A neural correlate of syntactic encoding during
  speech production. *PNAS*, 98(10), 5933–5936.
  doi:10.1073/pnas.101118098
- **Indefrey, P., & Levelt, W. J. M.** (2004). The spatial and temporal
  signatures of word production components. *Cognition*, 92(1–2), 101–144.
  doi:10.1016/j.cognition.2002.06.001
- **Kempen, G., & Hoenkamp, E.** (1987). An incremental procedural grammar for
  sentence formulation. *Cognitive Science*, 11(2), 201–258.
- **Kinno, R., et al.** (2008). Neural correlates of syntactic processing in
  non-canonical word order: An fMRI study of Japanese. (Cited via Kristensen
  et al. 2013 and the 2025 *Human Brain Mapping* scrambling study.)
- **Konopka, A. E., & Brown-Schmidt, S.** (2014). Message encoding. In
  M. Goldrick, V. S. Ferreira, & M. Miozzo (Eds.), *The Oxford Handbook of
  Language Production*. Oxford University Press.
- **Konopka, A. E., & Meyer, A. S.** (2014). Priming sentence planning.
  *Cognitive Psychology*, 73, 1–40. doi:10.1016/j.cogpsych.2014.04.001
- **Kristensen, L. B., Engberg-Pedersen, E., Nielsen, A. H., & Wallentin, M.**
  (2013). An fMRI study of canonical and noncanonical word order in Danish.
  *Journal of Neurolinguistics*. (BA44/45/47 increase for object-initial vs
  subject-initial clauses.)
- **Levelt, W. J. M.** (1981). The speaker's linearization problem.
  *Philosophical Transactions of the Royal Society of London. Series B,
  Biological Sciences*, 295(1077), 305–315. doi:10.1098/rstb.1981.0142
- **Levelt, W. J. M.** (1989). *Speaking: From Intention to Articulation*. MIT
  Press.
- **Levelt, W. J. M., Roelofs, A., & Meyer, A. S.** (1999). A theory of lexical
  access in speech production. *Behavioral and Brain Sciences*, 22(1), 1–75.
- **Matchin, W., & Hickok, G.** (2020). The cortical organization of syntax.
  *Cerebral Cortex*, 30(3), 1481–1498. doi:10.1093/cercor/bhz180
- **Matchin, W., et al.** (2020). Syntax-sensitive regions of the posterior
  inferior frontal gyrus and the posterior temporal lobe are differentially
  recruited by production and perception. *Cerebral Cortex Communications*,
  1(1), tgaa029.
- **McCauley, S. M., & Christiansen, M. H.** (2017). Computational
  investigations of multiword chunks in language learning. *Topics in Cognitive
  Science*, 9(3), 637–652.
- **McDonald, J. L., Bock, J. K., & Kelly, M. H.** (1993). Word and world order:
  Semantic, phonological, and metrical determinants of serial position.
  *Cognitive Psychology*, 25(2), 188–230. doi:10.1006/cogp.1993.1005
- **Momma, S.** (2021). Filling the gap in gap-filling: Long-distance dependency
  formation in sentence production. *Cognitive Psychology*, 129, 101411.
- **Momma, S.** (2021). *The Production of Linguistic Knowledge*. MIT Press.
  (TAG-based grammatical encoding; elementary trees activated by conceptual
  structure.)
- **Morgan, A. M., Devinsky, O., Doyle, W. K., Dugan, P., Friedman, D., &
  Flinker, A.** (2024). A low-activity cortical network selectively encodes
  syntax. *bioRxiv*. doi:10.1101/2024.06.20.599931
- **Morgan, A. M., Devinsky, O., Doyle, W. K., Dugan, P., Friedman, D., &
  Flinker, A.** (2024/2025). Decoding words during sentence production with
  ECoG reveals syntactic role encoding and structure-dependent temporal
  dynamics. *Communications Psychology*. doi:10.1101/2024.10.30.621177
- **Pawley, A., & Syder, F. H.** (1983). Two puzzles for linguistic theory:
  Nativelike selection and nativelike fluency. In J. C. Richards & R. W.
  Schmidt (Eds.), *Language and Communication* (pp. 191–227). Longman.
- **Shetreet, E., & Friedmann, N.** (2014). The processing of different
  syntactic structures: Word order vs. verb movement. (Hebrew OSV/VSO; LIFG and
  bilateral posterior temporal.)
- **Smith, M., & Wheeldon, L.** (1999). High level scope planning in spoken
  sentence production. *Cognition*.
- **Sikora, et al.** (2025). Unified brain mechanisms underlying the processing
  of word order permutations of arguments and adjuncts in sentences. *Human
  Brain Mapping*. doi:10.1002/hbm.70604
- **Tomasello, M.** (2003). *Constructing a Language: A Usage-Based Theory of
  Language Acquisition*. Harvard University Press.
- **van Trijp, R., & ...** (2008). *Constructions at Work or at Rest?* ILLC
  report X-2008-05. (Review criticizing CxG for lacking a production
  procedure.)
- **Wheeldon, L. R., Smith, M. C., & Apperly, I. A.** (2011). Repeating words
  in sentences: Effects of sentence structure. *Journal of Experimental
  Psychology: Learning, Memory, and Cognition*, 37(5), 1051–1064.
- **Zaccarella, E., & Friederici, A. D.** (2015). Merge in the human brain: A
  sub-region based functional investigation in the left pars opercularis.
  *Frontiers in Psychology*, 6, 1818. doi:10.3389/fpsyg.2015.01818

---

## 10. One-paragraph answer

The brain linearizes by **building a non-linear, role-bearing message first and
then imposing a serial order on it in a separate, later operation**. Levelt's
message is a propositional structure with thematic-role↔concept bindings
(Levelt 1989); the linearization problem is the imposition of order on that
structure under a working-memory constraint (Levelt 1981; Kempen & Hoenkamp
1987; Ferreira & Henderson 1998). Grammatical encoding splits into a
**functional** stage — lexical selection plus function assignment, where
thematic roles become subject/object — and a **positional** stage, where the
functions are ordered and inflected (Bock & Levelt 1994). Conceptual
accessibility drives role assignment, not ordering (Bock & Warren 1985;
McDonald, Bock & Kelly 1993), and the order is committed before articulation
and is structure-dependent (Konopka & Meyer 2014; Morgan et al. 2024, ECoG).
Word order is neurally dissociable from hierarchical structuring — pIFG does
morpho-syntactic sequencing for production, pMTG does hierarchical structuring
for both (Matchin & Hickok 2020) — and word-order permutation recruits LIFG and
pSTS (Kinno et al.; Kristensen et al. 2013; Sikora et al. 2025). The brain also
retrieves multiword units and produces novel sentences by slot-filling into
learned constructions (Pawley & Syder 1983; Goldberg 2006; Tomasello 2003;
Arnon & Christiansen 2017), so the frame is learned and the fillers are
selected. RAVANA's decoder has neither: it is a next-token predictor with no
message, no function assignment, and no frame, and the logit boost fails
because it raises content-word activation — the input to lexical selection —
while leaving the frame empty, so there is nothing to compute determiners and
prepositions from. The design is to condition the decoder on an explicit
**ordered role sequence** (`LinearizationFrame`): seed the GRU with a
frame embedding rather than the collapsed subject signature, mask the
candidate set to the open slots in order, and condition the transition prior on
the role sequence. The falsifiable prediction is that function-word ratio and
skeleton match rise while on-topic stays at or above the CPG baseline and
unsupported assertions fall — and the metric must include a syntactic term,
because the anchor-boost sweep was fitted on one that could not see syntax.
