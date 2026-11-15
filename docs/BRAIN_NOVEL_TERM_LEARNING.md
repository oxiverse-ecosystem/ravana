# How the human brain learns new terminology and new words from conversation — and what RAVANA should implement

**Date:** 2026-11-15
**Scope:** Evidence-first review of incidental and explicit word/term learning, with
derived architectural mechanisms for RAVANA.
**Method note:** Every empirical claim below carries an inline citation `[n]` into the
ledger-backed Sources list at the end. Claims I could not verify from an abstract or
full text are marked `[unverified]` and collected in §7. Where the popular account of a
finding differs from what the paper actually reports, I say so explicitly — this happens
repeatedly in the fast-mapping literature and is the single most important correction in
this document.

**Companion docs:** `docs/BRAIN_NOVEL_SUBJECT_GROUNDING.md` (ATL hub, commitment,
"I don't know" route), `docs/BRAIN_LEXICAL_CATEGORY.md`, `docs/BRAIN_TO_RAVANA_LEARNING.md`.

---

## 0. Executive summary

Learning a new word from conversation is not one process. It is at least four, on four
different timescales, and they fail in four different ways:

1. **Referent selection** — in the moment, from the current scene and the current
   utterance. Fast, competitive, and largely *not* memory: children can pick the right
   referent and forget the mapping five minutes later [2].
2. **Initial encoding / fast mapping** — creating a durable lexical entry from very few
   exposures. Real, but the popular "one exposure" version is a misreading: Carey's own
   retrospective says "Never did anybody believe that children typically create full
   lexical representations upon just one or even a few exposures" [16], and in her
   original data only 2 of 34 children had fully mapped the word by the end of testing
   [16].
3. **Extended mapping** — the slow alignment of the new entry with the adult concept,
   which is where most of the work is, and which sometimes requires *creating new
   semantic primitives* rather than selecting among existing ones [16].
4. **Consolidation / lexical integration** — the offline process that makes the new
   entry behave like an existing word (competing with neighbours, priming related
   meanings). This is sleep-dependent for explicitly learned words [7][13] and is *not*
   the same thing as remembering the word [21].

The architectural consequence for RAVANA: a term-learner needs a **separate, explicitly
modelled stage for each of these**, plus a **commitment gate** that refuses to produce
terminology it has only "fast mapped" but not consolidated. The failure mode of a
fluent decoder that invents terminology is precisely the failure of a system that does
referent selection without extended mapping.

**The one-sentence version:** *The brain does not learn a word in one exposure; it
creates a fragile partial entry in one exposure, and then spends days to weeks aligning
it with an existing concept — and a system that skips the alignment step will produce
plausible terminology with no semantics behind it.*

---

## 1. Incidental / unsupervised word learning from context

### 1.1 Fast mapping: what Carey & Bartlett (1978) actually showed

The study. Carey & Bartlett (1978) exposed 3- and 4-year-olds to a novel colour word,
"chromium," referring to olive green, in a single naturalistic teaching event: the
teacher asked the child to "bring me the chromium tray, not the blue one, the chromium
one." The critical design feature is the **lexical contrast** — two trays identical
except in colour, with one colour named — which makes the *category* of the word
(a colour word) inferable even if the exact referent is not [30][16].

What was found, in Carey's own later account [16]:

- Over half of the children showed evidence of having added "chromium" to their
  lexicon, remembering it 6–10 weeks after a few introducing events, and demonstrating
  the knowledge in a *different* context from the learning one.
- All of the children whose lexicon had been affected had learned that "chromium" was
  **a colour word** — i.e. they got the grammatical/semantic category right.
- But **only 2 of 34 children had fully mapped "chromium" to olive green** by the end of
  testing, differentiating it from the superordinate category (green, or the
  achromatics) into which they had placed olive green at baseline [16].
- The remaining children showed three intermediate patterns: some treated "chromium" as
  a synonym of "green"; more assimilated it to the achromatic region alongside "brown"
  and "grey"; others forgot the word but had begun to *reorganise the colour space*
  itself [16].

And Carey's explicit framing, which is the correction to the popular claim [16]:

> "Never did anybody believe that children typically create full lexical
> representations upon just one or even a few exposures to a new word."

So the honest summary of Carey & Bartlett (1978) is: **a single contrastive exposure
can create a durable lexical entry with the correct grammatical category, but the
referential content of that entry is partial and takes weeks to align with the adult
concept.** The "one exposure" claim is a compression of "one exposure is enough to
create an entry that survives," not "one exposure is enough to learn the word."

Swingley's (2010) reanalysis of the same study is blunter: "Fewer than one in ten of
the 3-year-olds appeared to have linked the word to its intended meaning (olive
green)" [1]. He also notes that the children in the naturalistic teaching context were
"scarcely more likely than controls" to pick the correct referent from an array [1].
Carey's own interest, he points out, was in **extended mapping**, not the fast mapping
the paper is remembered for [1][16].

### 1.2 Markson & Bloom (1997) and Bloom & Markson (1998): the "one exposure" claim

Markson & Bloom (1997), *Nature* 385:813–815 [4], taught 3- and 4-year-olds and adults
a novel name and a novel *fact* about an object (e.g. "let's use the koba to measure
which is longer"), with the information presented a small number of times in a
20-minute session, and tested retention immediately, after 1 week, and after 1 month.
Their findings:

- Children and adults retained the object–label mapping above chance even at the
  1-month delay [4][38].
- Crucially, they retained a **novel fact** presented the same way just as well —
  which is the actual theoretical point of the paper: fast mapping is **not a
  language-specific mechanism** but a general learning-and-memory capacity [4].

The popular version of this result — "children learn a word from one exposure and
remember it for a month" — is not what the paper shows. Two corrections:

1. The exposure was not one event. In the object-label condition the label was presented
   three times during the session; the fact was presented once in one experiment and
   twice in another [38]. The paper's own definition of fast mapping is "learning from
   a few incidental exposures," and the field's convention is that "minimal exposure"
   means up to four encounters [38].
2. Retention was **above chance, not good**. Independent replications put long-term
   retention of a newly mapped object label at roughly 65–67% [38], and Vlach &
   Sandhofer (2012) found rapid forgetting across immediate, 1-week and 1-month tests,
   with forgetting following a curvilinear (standard forgetting-curve) pattern [3].

Bloom & Markson (1998), *Trends in Cognitive Sciences* 2(2):67–73 [12], is the
theoretical companion. Its claim is that word learning is **not** the product of a
general associative learning mechanism, nor of dedicated word-learning constraints, but
of a set of capacities — conceptual biases about the world, the ability to infer
referential intentions, and (by about age two) sensitivity to syntactic cues — some
language-specific and unique to humans, others shared with other species [12]. For
RAVANA this is an important negative result: **co-occurrence statistics alone are
claimed to be insufficient**, and the learner must bring intentional/syntactic structure
to the mapping problem [12].

### 1.3 Cross-situational statistical learning

The alternative to "one-shot" learning is **cross-situational word learning (CSWL)**:
the learner sees many individually ambiguous word–scene pairings and resolves them by
tracking co-occurrence statistics *across* situations. Yu & Smith (2007) showed adults
can acquire word–object mappings from distributed statistics despite in-the-moment
uncertainty; Smith & Yu (2008) showed the same for 12- and 14-month-old infants [5][6].

The mechanism, as characterised in a 15-year review [5]:

- Across trials, associations between co-occurring words and referents are gradually
  strengthened, while non-co-occurring associations stay weak.
- The indeterminacy problem "is solved not in a single trial but across trials, not for
  a single word and its referent but for a data set of many words and referents" [6].
- The learner necessarily also forms **spurious correlations** between a word and the
  distractor objects that co-occurred with it in other trials [5].

The critical caveat, and the reason CSWL is not a complete account of conversation-based
learning: CSWL is a *laboratory* demonstration that a statistical mechanism *can* solve
the mapping problem. It does not show that this is the mechanism conversation uses.
Indeed, detailed behavioural analysis of the paradigm found that in-the-moment
mechanisms (hypothesis testing, attentional cueing) augment the statistical learning,
and that a pure cross-situational statistical learner does not fully explain performance
[5]. For a system like RAVANA the practical reading is: **a co-occurrence accumulator is
necessary but not sufficient, and it will generate spurious associations that must be
pruned by something else.**

### 1.4 How many exposures are needed, and what is extracted

The exposure-count literature gives a wide range, and the range itself is informative:

- **Two exposures** in a naturalistic story context gave 38.5% correct cued recall of a
  new meaning for a familiar word, rising linearly to 63.5% at eight exposures, with no
  significant forgetting after one week [17][25]. The increment was ~8.3% per additional
  two exposures [17].
- Estimates in the educational literature range from **10 exposures** ("some chance of
  recognising the meaning") to **40–200** [unverified — see §7; the 10-exposure figure is
  a secondary citation to Laufer & Nation via a non-peer-reviewed blog, and the 40/200
  figures are secondary citations to Clay et al. via a policy document. Treat as
  ballpark, not established.]
- Carey's "chromium" data show that even after *weeks* and multiple encounters, full
  referential alignment was achieved by only 2 of 34 children [16].

What the learner extracts from an exposure is not a definition. From a single strongly
constrained sentence, adults extract enough to (a) judge which verbs can plausibly take
the novel word as object, and (b) show N400 priming to semantically related words — but
**only if the learning context was strongly constraining** [11]. In weakly constraining
contexts, the same single exposure produced no such effect [11]. This is the cleanest
demonstration that *contextual constraint*, not exposure count alone, determines what
is learned from an incidental encounter.

The developmental literature adds a crucial asymmetry: **referent selection and
retention are dissociable**. Horst & Samuelson (2008) found 24-month-olds were excellent
at referent selection but at chance on retention after a 5-minute delay; retention
improved only when referent selection was augmented with ostensive naming [2]. Vlach &
Sandhofer (2012) extended this: short-term retention does not guarantee long-term
retention, and learners forget word mappings rapidly, with a standard curvilinear
forgetting function [3]. A later large-scale study (354 children) found that after five
minutes, retention was above chance for object labels *and* colour/shape/texture words,
but after one week **only object labels** remained above chance [38].

**Summary of §1.** Incidental learning from conversation is real and fast at the
*referent-selection* level, fragile at the *retention* level, and slow at the
*concept-alignment* level. The popular "one exposure" claim conflates the first with
the third. A system that only does the first will be fluent and empty.

---

## 2. Supervised / explicit definition learning

### 2.1 What makes a definition usable

The empirical literature on definition learning is thinner than the incidental-learning
literature, and much of it is in L2/vocabulary-acquisition rather than cognitive
neuroscience. What is established:

- **Definitions work, and they work better than incidental exposure alone**, but the
  effect is on *explicit* knowledge. In a direct comparison, adults learned new meanings
  for familiar words both incidentally (story reading) and intentionally (definition
  training with multiple-choice meaning-to-word matching); the intentional condition
  produced higher definition recall, and an immediate test after training boosted
  retention [14].
- **A definition is a compressed feature list, and its usefulness depends on the
  features being the right ones.** Clark's (1973) semantic feature hypothesis — that
  early word meanings are partial feature sets that are progressively added to — is the
  classic statement, and it is supported by the *pattern* of children's errors (§5):
  over-extension is driven by a single salient feature (e.g. [+four-legged] for *dog*),
  which is exactly what a partial feature set predicts [9][38].
- **Images/definitions interact.** Adding images to verbal definitions improved
  behavioural outcomes (definition recall, relatedness judgements) but did **not**
  change the ERP signatures of learning and consolidation [24]. That is a dissociation
  worth noting: a manipulation can improve performance without changing the underlying
  consolidation process.

For RAVANA the design implication is that a "definition" should be treated as a
**structured feature/relation set with a stated referent**, not as a string to be
memorised. The evidence for this framing is Clark's feature hypothesis plus the
error-pattern data (§5), not a direct neuroimaging result on definitions.

### 2.2 The anterior temporal lobe / semantic hub

The hub-and-spoke model (Patterson, Nestor & Rogers 2007; Rogers et al. 2004) holds that
modality-specific association cortices are the spokes and the **bilateral anterior
temporal lobe (ATL)** is an amodal hub that distills them into coherent, generalisable
concepts [10][15]. The key evidence for the *learning* (not just representation) role:

- **Hoffman, Evans & Lambon Ralph (2014)**, *Cortex* 50:19–31 [15], studied acquisition
  of new concepts in semantic dementia patients with atrophy centred on the
  ventrolateral ATL. Patients learned to assign abstract visual stimuli to two
  categories with a **family-resemblance structure** in which no single feature was
  diagnostic. Patients were **unable to integrate the features into a coherent
  concept**, instead basing category judgements on a single dimension [15]. The paper's
  conclusion: "integrating disparate sources of information into novel coherent
  concepts is a critical computational function of the ATLs" [15].
- Single-cell recordings in monkeys suggest ATL involvement in visual associative
  learning, but human work had focused on existing knowledge rather than learning
  [10].
- The ATL's role in *representing* meaning is also supported by rTMS: offline 1 Hz
  rTMS over ATL in healthy participants selectively slowed semantic processing
  (see `docs/BRAIN_NOVEL_SUBJECT_GROUNDING.md` for the full citation set).

This is the strongest neuroscience anchor for the document's central claim: **forming a
new coherent concept out of disparate features is a distinct computational operation,
localised to the ATL, and it is not the same as storing a form or a co-occurrence
vector.**

### 2.3 Embedding from co-occurrence vs. binding to an existing concept

Two competing accounts of what a novel term's representation *is*:

**(a) Co-occurrence / distributional embedding.** The term's vector is built from the
contexts in which it appears. This is the CSWL/associative account (§1.3) and is
supported by the finding that a single strongly constrained exposure can yield N400
priming to related words [11] — i.e. *something* semantic is extracted immediately from
context. But the same study shows the extraction is **constraint-gated**, and CSWL
reviewers note the mechanism generates spurious associations [5].

**(b) Binding to an existing concept.** The novel term is not given a fresh vector but
is *attached to* an existing conceptual representation. This is Bloom's position:
"word learning is usually a process of mapping new words to pre-existing concepts"
[33], and the ATL-hub evidence supports it — the hub's job is to bind features into
coherent concepts, and damage to it specifically impairs *integrating* features into a
novel coherent concept [15].

The evidence suggests **both are needed, on different timescales**, and this is the
most important mechanistic claim in the document:

- **Immediately**, a constrained context supports a rapid, prediction-based binding
  event: the reader generates a prediction from the context, the novel word arrives,
  and the mismatch/resolution updates the lexicon — evidenced by N400 modulation and
  priming after a *single* strongly constrained exposure [11].
- **Over time**, the entry is consolidated and integrated with existing neighbours;
  this is the sleep-dependent process of §4, and it is *not* complete after one night
  [20][21].
- **Concept formation** (as opposed to concept binding) is the ATL's distinctive
  contribution and is what fails in semantic dementia [15].

A useful framing for RAVANA: **a novel term should be represented as (form, bound-concept,
feature-set, co-occurrence-trace, consolidation-state)** — five separable fields, not a
single embedding. The consolidation-state field is what a fluent decoder is missing.

---

## 3. Word form vs. concept; the hard problem of empty concepts

### 3.1 The form/concept dissociation

Learning the **form** (phonological/orthographic) and learning the **concept** are
separable, and the neuroscience separates them cleanly:

- **Form learning is fast and hippocampal.** Davis, Di Betta, Macdonald & Gaskell
  (2009) found that learners could distinguish newly learned spoken words from
  similar-sounding foils after only **three intact presentations**, and that hippocampal
  responses were elevated for entirely unfamiliar novel words, with the elevation
  correlating with later behavioural measures of learning [8]. Neocortical responses for
  words learned the previous day were activated to the same level as known English
  words [8].
- **Meaning learning is slower and depends on the amount of associated information.**
  Words learned *with* meanings were remembered better than words learned *without*
  meanings, and both episodic (hippocampus-dependent) and semantic (neocortical)
  systems were engaged during recognition — with *more* involvement of both for
  meaningful words after a one-week delay [23]. The representation "does not shift
  completely from the episodic memory system to the semantic memory system" within the
  week tested [23].
- **Consolidation is not one thing.** The meta-analysis of sleep effects found
  moderate effects for recall (g = 0.57) and recognition (g = 0.52) but only a **small**
  effect for lexical integration (g = 0.39) [21]. Remembering a word and integrating it
  into the lexicon are different outcomes with different effect sizes.

For RAVANA: a term can be **form-known** (recognisable, producible) without being
**concept-known** (usable in a true statement). The decoder's failure mode is
form-known/concept-unknown. Any mechanism proposal must therefore track the two
separately and gate production on the concept side.

### 3.2 The hard problem of empty concepts

A word can be learned without a referent. The philosophical literature on "empty
concepts" (concepts satisfied by nothing in the world — *unicorn*, *phlogiston*,
*Vulcan*) is long-standing, and the key point for a learning system is that **an empty
concept is descriptive, not referential**: its content is fixed by the description that
introduces it, not by a worldly object [unverified — the philosophical sources found
were *Mind & Language* and *Philosophical Studies* papers on empty concept dependency
and concept referentialism; I verified their abstracts/topics but did not verify a
specific empirical claim about how the brain learns empty terms. See §7.]

The empirically grounded version of this problem is **fast mapping with no referent
present**. In the original chromium study the referent *was* present. But much
conversation-based learning has no present referent — the term is introduced by
definition or by description. The relevant evidence:

- **Words can be learned from context with the referent absent.** Borovsky, Elman &
  Fernald (2012) showed that vocabulary skill (not age) predicts anticipatory
  incremental sentence interpretation in 3–10-year-olds and adults — i.e. the ability
  to use combinatory context to predict an upcoming word is a *vocabulary-driven*
  skill, and it is what lets a learner bind a novel term to a described rather than
  displayed referent [unverified — I verified the abstract via ERIC [39] but not the
  full text; the DOI in the citing literature is 10.1016/j.jecp.2012.01.005, which
  differs from the 2012.09.010 cited in one secondary source. Treat the citation
  details as provisional.]
- **Novel words learned with meanings engage both memory systems more than form-only
  words** [23], which is the closest neural evidence that "having a concept" changes
  the representation rather than just adding an association.

The honest statement: **there is no direct neuroscientific study I could verify on how
the brain learns a term that has no referent at all.** The philosophical literature
exists but is not experimental. This is a genuine gap, and RAVANA's mechanism for
empty terms should be treated as a design hypothesis, not a brain-derived certainty.

---

## 4. Consolidation: making a novel term durable

### 4.1 The two-stage / complementary learning systems account

The dominant framework is the **Complementary Learning Systems (CLS)** account
(McClelland, McNaughton & O'Reilly 1995; Davis & Gaskell 2009): a fast-learning
hippocampal system encodes new episodes, and a slow-learning neocortical system
gradually acquires the statistical structure; sleep provides the opportunity for
hippocampal information to be fed into long-term neocortical memory [7][8][19].

The specific, well-replicated findings:

- **Dumay & Gaskell (2007)**, *Psychological Science* 18(1):35–39 [7]: words learned at
  8 p.m. did **not** induce lexical competition immediately, but did after a 12-hour
  interval including a night's sleep, and continued to do so after 24 hours. Words
  learned at 8 a.m. showed no competition immediately or after 12 hours of wakefulness,
  but did after 24 hours — i.e. after sleep had occurred. The authors' interpretation:
  "sleep provides an opportunity for hippocampal information to be fed into long-term
  neocortical memory" [7].
- **Tamminen, Payne, Stickgold, Wamsley & Gaskell (2010)**, *Journal of Neuroscience*
  30(43):14356–14360 [13]: sleep spindle activity was associated with overnight
  **lexical integration** (the emergence of lexical competition) but **not** with gains
  in recall rate or recognition speed of the novel words themselves. Spindles appear
  "particularly important for overnight integration of new memories with existing
  neocortical knowledge" [13].
- **Meta-analysis (Schimke et al., 2021)**: sleep beneficial for novel word learning
  vs wakefulness (g = 0.50); moderate for recall (0.57) and recognition (0.52), small
  for lexical integration (0.39) [21].
- **Hippocampal–neocortical shift**: novel words encountered on the day of scan showed
  elevated hippocampal activation vs words learned the previous day; neocortical
  responses for previous-day words matched known English words [8][21].

### 4.2 What is established vs. inferred

**Established (multiple replications, meta-analysis):**
- Sleep benefits novel word retention and lexical integration more than an equivalent
  wake interval [7][13][21].
- Lexical competition is a behavioural marker of integration and emerges after
  consolidation, not immediately [7][13][19].
- Spindle activity correlates with integration specifically [13].

**Inferred / contested:**
- **That consolidation is *complete* after one night.** Gaskell (2024) explicitly warns
  against this inference: "it might be tempting to infer from sleep studies such as
  Dumay and Gaskell (2007) that the consolidation process can take place over the
  course of a single night, leading to full lexical integration the following morning"
  [19]. The evidence does not support "one night = done."
- **That fast mapping bypasses the hippocampus.** The meta-analysis notes this was
  *theorised* to explain why fast-mapped words did not show a sleep benefit in one
  study (Himmer et al. 2017), but this is a theoretical interpretation, not an
  established finding [21].
- **That sleep is *necessary*.** A 2024 study found **no** sleep benefit for
  *contextual* word learning in younger adults — comparable recognition for sleep and
  wake groups at both initial and delayed sessions — leading the authors to conclude
  that "not all forms of word learning appear to benefit from sleep for consolidation"
  and that encoding method is critical [37]. This is a direct counterexample to a
  strong sleep-necessity claim and should be taken seriously.
- **That the hippocampal→neocortical transfer is the mechanism.** This is the CLS
  interpretation; the behavioural data are consistent with it but the specific
  transfer mechanism is a model, not a measurement [19].

**For RAVANA:** the safe, evidence-backed design principle is that **durable
integration is a separate, offline, repetition-dependent process that is behaviourally
dissociable from initial encoding** — not that "sleep" is the required implementation.
A replay/consolidation stage that runs offline and strengthens the link between a term
and its concept (and between the term and its neighbours) is well motivated; a claim
that it must be tied to a sleep-like state is not.

---

## 5. Developmental evidence and informative errors

### 5.1 How children learn words from conversation

Children learn words from conversation in ways that are *social* and *incremental*:

- **Incremental prediction is vocabulary-driven.** Borovsky, Elman & Fernald (2012)
  found that 3–10-year-olds and adults make anticipatory eye movements to a target
  object after hearing "The pirate hides the…", and that **receptive vocabulary size,
  not age**, predicted anticipatory looking [39]. The interpretation is that knowing
  more words *is* what enables rapid incremental interpretation — a positive feedback
  loop between vocabulary and processing speed.
- **Word learning is a social act.** Bloom's account: children use inferences about the
  referential intentions of others to create the word→concept mapping, and "much of word
  learning involves mapping words into pre-existing concepts" [33]. Bloom & Markson
  (1998) argue the required capacities are conceptual biases, intentional understanding
  and syntactic knowledge — "not only necessary… but also sufficient" [12].
- **Statistical covariation is neither necessary nor sufficient.** Bloom's review
  argues that statistical covariation between word and percept is neither necessary nor
  sufficient for word learning, and that children do not need ostensive naming [33].
  This is a strong claim and it is contested (see §1.3); the honest reading is that
  *both* social-intentional and statistical mechanisms are implicated, and the field
  has not settled their relative weight.

### 5.2 Over-extension and under-extension

Children's word-use errors are systematic, and they are the best window into the
representation:

- **Over-extension** — using a term for a broader range than the adult. Clark (1973)
  documented this and explained it with the **semantic feature hypothesis**: the child's
  entry contains only one or a few salient features (e.g. [+four-legged] for *dog*), so
  the word is over-extended to cats and cows [9]. Clark's account is that children
  "move from attention to individual features, to attention to configuration or
  structured relations between features" [9].
- **Under-extension** — using a term for a narrower range (e.g. *blanky* for one
  blanket only; *cow* only for the child's cuddly toy) [9][38]. Under-extension is
  "usually viewed as a lexical phenomenon… motivated by different strategies than those
  underlying overextensions" [9].
- **Both occur, and which dominates depends on the task.** A study that deliberately
  created opportunities for both found under-extension at least as pronounced as
  over-extension, and concluded that whether a child makes more of one kind "depends
  upon (1) the child in question, (2) the concepts being investigated, and (3) the
  nature of the instances and non-instances" [unverified — this is an ERIC document
  (ED089341) I found via search but did not extract; treat as a lead, not a verified
  finding. See §7.]
- **Comprehension vs production diverge.** In comprehension tasks children are
  excessively liberal (over-extension); in production, both over- and under-extension
  appear, with under-extension only for words that entered the vocabulary later [38].
  This is a performance-constraint effect: the child may *know* more than they *say*.

**Why this matters for RAVANA.** The error pattern is the signature of a **partial
feature set**: over-extension = too few constraints; under-extension = too narrow a
context model. A term-learner that represents a new term as a feature set with
confidence weights will naturally produce both error types as a function of how many
features it has bound — and that is a *falsifiable* prediction, not just an analogy.

---

## 6. Derived mechanisms for RAVANA

Eight proposals. Each states: **what to implement**, **what measurement confirms it**,
**what would falsify it**, and **the specific finding it derives from**.

### M1. Two-stage term entry: referent selection ≠ term acquisition

**Implement.** Separate the in-conversation *referent selection* step (which entity in
the current discourse the novel term picks out) from the *term acquisition* step
(whether a durable lexical entry is created). Maintain a `term_status` field with at
least: `unseen → selected → encoded → consolidated`. Production of the term is gated on
`encoded` or higher, never on `selected`.

**Confirm.** Log the transition counts. If `selected → encoded` is near 1.0, the gate is
not doing anything and the mechanism is vacuous. The predicted (and brain-consistent)
pattern is a large drop from selection to retention.

**Falsify.** If a system with no such gate performs identically on delayed term
retention, the dissociation is not load-bearing for RAVANA.

**Derives from.** Horst & Samuelson (2008): excellent referent selection, chance
retention after 5 minutes [2]; Vlach & Sandhofer (2012): rapid forgetting of mappings
[3].

### M2. Contextual-constraint gate on incidental encoding

**Implement.** Score the constraint of the context in which a novel term appears (e.g.
the entropy reduction / predictability of the term given the preceding discourse). Only
create or strengthen a term entry if the constraint exceeds a threshold. Low-constraint
encounters are logged as exposures but do not create an entry.

**Confirm.** A dose-response: entries created from high-constraint contexts should
survive delayed tests better than entries from low-constraint contexts, at matched
exposure counts.

**Falsify.** If low-constraint single exposures produce entries that survive as well as
high-constraint ones, the gate is wrong.

**Derives from.** Borovsky, Elman & Kutas (2012): N400 priming and plausibility effects
after a single exposure *only* in strongly constraining sentences [11].

### M3. Feature-set representation with confidence, not a point embedding

**Implement.** Represent a novel term as a set of (feature, weight, source) triples
derived from the contexts/definitions it appears in — not as a single vector. Include
the grammatical category as a first-class field (Carey's subjects got the *category*
right even when the referent was wrong).

**Confirm.** The system should reproduce the *signature error pattern*: over-extension
when few features are bound (term applied to feature-similar but wrong referents) and
under-extension when the context model is narrow. Ablating the feature set to a point
embedding should destroy this pattern.

**Falsify.** If the system's errors are not feature-structured (e.g. random, or purely
frequency-driven), the feature-set hypothesis is not the right model.

**Derives from.** Clark (1973) semantic feature hypothesis and the over/under-extension
pattern [9][38]; Carey's finding that children got "chromium" *was a colour word* right
while getting the referent wrong [16].

### M4. ATL-analogue: a concept-integration operator distinct from the embedding store

**Implement.** A distinct module that takes the feature set (M3) plus any co-occurrence
trace and *integrates* them into a single coherent concept representation, and which is
the only route by which a term becomes `consolidated`. This module should be capacity
limited and slow relative to the embedding store.

**Confirm.** A double dissociation in ablation: damaging the integration module should
impair learning *new* concepts while sparing retrieval of *existing* ones. That is the
exact pattern in semantic dementia [15].

**Falsify.** If ablating the integration module impairs new and old concepts equally,
it is not doing the ATL's job and the analogy fails.

**Derives from.** Hoffman, Evans & Lambon Ralph (2014): SD patients could not integrate
features into a coherent novel concept, though they could learn individual feature
associations [15]; hub-and-spoke model [10].

### M5. Offline consolidation with replay, gated on the hippocampal/neocortical distinction

**Implement.** An offline stage that reactivates recently encoded term entries and
strengthens (a) the term↔concept link and (b) the term↔neighbour links. Model the
two-system distinction: a fast episodic store for recent encounters, a slow
consolidated store. Do **not** claim the offline stage requires a sleep-like state; the
evidence does not support that (see §4.2).

**Confirm.** A `lexical competition` analogue: after consolidation, the presence of the
new term should *slow* recognition/processing of a neighbouring existing term. This is
the behavioural marker used in the sleep literature [7][13] and is a strong, specific
prediction.

**Falsify.** If the system shows improved recall but no competition/neighbour effect,
consolidation is not producing integration — it is only strengthening item memory.
That dissociation is exactly what Tamminen et al. (2010) found for spindles vs recall
[13].

**Derives from.** Dumay & Gaskell (2007) [7]; Tamminen et al. (2010) [13]; CLS account
[8][19]; with the counterexample of Schimke et al. (2024) [37] bounding the claim.

### M6. Empty-term handling: descriptive content without a referent

**Implement.** Allow a term entry to be created with `referent = null` but
`descriptive_content` populated from the definition/description. Such a term is
*usable in statements about its description* but must be flagged as referentially empty
when the discourse requires a referent. Distinguish "I know what this word means" from
"I know what this word picks out."

**Confirm.** The system should be able to (a) use the term correctly in sentences that
are true-by-definition, and (b) decline/flag when asked to identify a referent.

**Falsify.** If the system cannot represent a term with no referent at all (i.e. it
either invents a referent or refuses the term entirely), the descriptive-content route
is not implemented.

**Derives from.** The philosophical empty-concept literature (unverified empirically —
see §3.2 and §7) plus the general finding that meaning and reference are separable
[23][33]. **This is the weakest-grounded mechanism in the document and should be
labelled as a design hypothesis.**

### M7. Exposure-count bookkeeping with a forgetting function

**Implement.** Track exposure count per term and model retention as a curvilinear
forgetting function (rapid initial decay, slowing over time), with the decay rate
modulated by whether the term is an object label vs a property word.

**Confirm.** Fit the forgetting curve: retention should drop fastest early and flatten
later, and object-label terms should decay more slowly than property terms at matched
exposures.

**Falsify.** If retention is flat (no forgetting) or linear, the forgetting-function
model is wrong. Note Hulme, Barsky & Rodd (2019) found *no* significant forgetting over
one week for story-learned meanings [17][25] — so the prediction is contested and the
mechanism should be parameterised, not hard-coded.

**Derives from.** Vlach & Sandhofer (2012) curvilinear forgetting [3]; the object-label
vs property-word retention asymmetry [38]; Hulme et al. (2019) as the counterexample
[17].

### M8. Metacognitive gate: withhold terminology that is form-known but concept-unknown

**Implement.** Before producing a term, evaluate the consolidation state and the
strength of the concept binding. If the term is `selected` or weakly `encoded` but not
`consolidated`, either (a) use it with an explicit hedge/attribution ("the term X, which
I take to mean…"), or (b) withhold and ask. This is the term-learning analogue of the
"I don't know" route in `docs/BRAIN_NOVEL_SUBJECT_GROUNDING.md`.

**Confirm.** The system should produce *fewer* confident false statements about
novel-term referents than an ungated baseline, at a measurable cost in coverage.

**Falsify.** If gating does not reduce false statements, the consolidation signal is not
predictive of correctness and the gate is decorative.

**Derives from.** The form/concept dissociation [8][23]; the finding that meaningful
words are remembered better and engage both memory systems [23]; the metacognitive
withholding literature reviewed in `docs/BRAIN_NOVEL_SUBJECT_GROUNDING.md`.

---

## 7. Verification ledger: what was checked and what was not

**Verified from abstract or full text (30 sources registered):**

| # | Source | What was verified |
|---|---|---|
| [1] | Swingley (2010), *LLD* 6:179–183 | Full text: "fewer than one in ten" 3-year-olds linked chromium to olive green |
| [2] | Horst & Samuelson (2008), *Infancy* 13:128–157 | Abstract: referent selection good, 5-min retention at chance |
| [3] | Vlach & Sandhofer (2012), *Front. Psychol.* 3:46 | Abstract/full text: rapid forgetting, curvilinear |
| [4] | Markson & Bloom (1997), *Nature* 385:813–815 | Abstract: words and facts retained, not language-specific |
| [5] | Roembke et al. (2023), *Front. Psychol.* 14:1175272 | Full text: CSWL mechanisms, spurious correlations |
| [6] | Smith & Yu (2008), *Cognition* 106:1558–1568 | Full text: infants learn via cross-situational statistics |
| [7] | Dumay & Gaskell (2007), *Psych. Sci.* 18:35–39 | Abstract: sleep-dependent lexical competition |
| [8] | Davis, Di Betta, Macdonald & Gaskell (2009) | Full text: hippocampal elevation, 3 presentations, overnight consolidation |
| [9] | "Logical word learning: the case of kinship" (PMC) | Full text: Clark's feature hypothesis, over/under-extension |
| [10] | Manchester research page (Kumaran/Hoffman ATL) | Abstract: ATL and novel concept acquisition |
| [11] | Borovsky, Elman & Kutas (2012), *LLD* 8:278–302 | Full text: N400 priming only in strongly constraining contexts |
| [12] | Bloom & Markson (1998), *TiCS* 2:67–73 | Abstract: capacities underlying word learning |
| [13] | Tamminen et al. (2010), *J. Neurosci.* 30:14356–14360 | Abstract: spindles ↔ integration, not recall |
| [14] | "Learning new word meanings from story reading" (PMC) | Full text: incidental vs intentional, immediate testing |
| [15] | Hoffman, Evans & Lambon Ralph (2014), *Cortex* 50:19–31 | Abstract/full text: SD patients fail feature integration |
| [16] | Carey (2010), *LLD* 6:184–205 | Full text: "Never did anybody believe…", 2 of 34 fully mapped |
| [17] | Hulme, Barsky & Rodd (2019), *Language Learning* 69:18–43 | Abstract: 2 exposures → 38.5%, 8 → 63.5%, no 1-week forgetting |
| [19] | Gaskell (2024), *QJEP* 77:2365–2384 | Full text: warns against "one night = full integration" |
| [20] | "Consolidation improves learning of new meanings" (PMC) | Abstract: sleep benefits memory but meanings not integrated in 24h |
| [21] | Schimke et al. (2021) meta-analysis | Abstract: g = 0.50 omnibus; 0.39 lexical integration |
| [23] | "Interaction between episodic and semantic memory" (MPG) | Full text: meaningful > form-only; both systems engaged |
| [24] | Liu & Van Hell (2022) definition+image study | Abstract: images help behaviour, not ERPs |
| [29] | "Consolidation of vocabulary is associated with sleep in children" | Abstract only (Wiley PDF) |
| [37] | Schimke et al. (2024), *QJEP* 77 | Full text: **no** sleep benefit for contextual word learning |
| [38] | Holland thesis (London Met) | Full text: Markson & Bloom exposure counts, 65–67% retention |
| [39] | Borovsky, Elman & Fernald (2012), *JECP* 112:417–436 | Abstract (ERIC): vocabulary, not age, predicts anticipation |

**Not verified / flagged:**

1. **[unverified] The 10-exposure and 40–200-exposure figures.** Found only as secondary
   citations in a non-peer-reviewed blog and a policy document. Not used as evidence.
2. **[unverified] The under-extension/over-extension relative-frequency claim** from
   ERIC ED089341 — found via search snippet, not extracted.
3. **[unverified] The philosophical empty-concept literature** — abstracts/topics verified,
   but no empirical claim about brain learning of empty terms. M6 rests on this gap.
4. **[unverified] The exact DOI for Borovsky, Elman & Fernald (2012)** — sources cite
   both 10.1016/j.jecp.2012.01.005 and 10.1016/j.jecp.2012.09.010. The ERIC record
   gives 112(4):417–436; the DOI should be confirmed before formal citation.
5. **[unverified] Whether CSWL is the mechanism conversation uses** — CSWL is a
   laboratory demonstration; the review itself notes in-the-moment mechanisms augment it
   [5].
6. **Not searched:** direct neuroimaging of *definition* learning (as opposed to word
   learning). The definition-learning literature found was largely L2/vocabulary
   acquisition, not neuroscience. This is a genuine coverage gap.

---

## Sources

[1] https://www.sas.upenn.edu/~swingley/papers/swingley_LLD10_fastmap.pdf
[2] https://onlinelibrary.wiley.com/doi/abs/10.1080/15250000701795598
[3] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3286766
[4] https://www.nature.com/articles/385813a0
[5] https://pmc.ncbi.nlm.nih.gov/articles/PMC10400455
[6] https://sites.socsci.uci.edu/~lpearl/courses/readings/SmithYu2008_WordLearningCrossSituational.pdf
[7] https://journals.sagepub.com/doi/abs/10.1111/j.1467-9280.2007.01845.x
[8] https://pmc.ncbi.nlm.nih.gov/articles/PMC2832742
[9] https://pmc.ncbi.nlm.nih.gov/articles/PMC9166873
[10] https://research.manchester.ac.uk/en/publications/the-anterior-temporal-lobes-are-critically-involved-in-acquiring-
[11] https://pmc.ncbi.nlm.nih.gov/articles/PMC3484686
[12] https://pubmed.ncbi.nlm.nih.gov/21227068
[13] https://pmc.ncbi.nlm.nih.gov/articles/PMC2989532
[14] https://pmc.ncbi.nlm.nih.gov/articles/PMC8362668
[15] https://pmc.ncbi.nlm.nih.gov/articles/PMC3884130
[16] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3101498
[17] https://onlinelibrary.wiley.com/doi/abs/10.1111/lang.12313
[19] https://pmc.ncbi.nlm.nih.gov/articles/PMC11607850
[20] https://pmc.ncbi.nlm.nih.gov/articles/PMC11219009
[21] https://journals.sagepub.com/doi/full/10.1177/17470218231179459
[23] https://pure.mpg.de/rest/items/item_2300819_16/component/file_2430258/content
[24] https://onlinelibrary.wiley.com/doi/full/10.1111/lang.12502
[25] https://eric.ed.gov/?id=EJ1205945
[29] https://onlinelibrary.wiley.com/doi/10.1111/j.1467-7687.2012.01172.x
[30] https://languagelog.ldc.upenn.edu/myl/Acquiring_a_Single_New_Word.pdf
[33] https://minddevlab.yale.edu/sites/default/files/files/Precis%20of%20How%20children%20learn%20the%20meanings%20of%20words.pdf
[37] https://pmc.ncbi.nlm.nih.gov/articles/PMC10960315/pdf/10.1177_17470218231179459.pdf
[38] https://repository.londonmet.ac.uk/909/1/HollandAmanda_WordsAndFacts.pdf
[39] https://eric.ed.gov/?id=EJ970158

---

## References (full citations)

**Primary empirical sources**

- **Carey, S., & Bartlett, E.** (1978). Acquiring a single new word. *Papers and
  Reports on Child Language Development* (Stanford University), 15, 17–29.
  [Primary study; accessed via the Language Log archive. Not DOI-indexed.]
- **Swingley, D.** (2010). Fast mapping and slow mapping in children's word learning.
  *Language Learning and Development*, 6(3), 179–183.
  doi:10.1080/15475441.2010.484412
- **Carey, S.** (2010). Beyond fast mapping. *Language Learning and Development*,
  6(3), 184–205. doi:10.1080/15475441.2010.484379
- **Markson, L., & Bloom, P.** (1997). Evidence against a dedicated system for word
  learning in children. *Nature*, 385, 813–815. doi:10.1038/385813a0
- **Bloom, P., & Markson, L.** (1998). Capacities underlying word learning.
  *Trends in Cognitive Sciences*, 2(2), 67–73. doi:10.1016/S1364-6613(98)01121-8
- **Horst, J. S., & Samuelson, L. K.** (2008). Fast mapping but poor retention by
  24-month-old infants. *Infancy*, 13(2), 128–157.
  doi:10.1080/15250000701795598
- **Vlach, H. A., & Sandhofer, C. M.** (2012). Fast mapping across time: Memory
  processes support children's retention of learned words. *Frontiers in Psychology*,
  3, 46. doi:10.3389/fpsyg.2012.00046
- **Smith, L., & Yu, C.** (2008). Infants rapidly learn word-referent mappings via
  cross-situational statistics. *Cognition*, 106(3), 1558–1568.
  doi:10.1016/j.cognition.2007.06.010
- **Roembke, T. C., Simonetti, M. E., Koch, I., & Philipp, A. M.** (2023). What have
  we learned from 15 years of research on cross-situational word learning? A focused
  review. *Frontiers in Psychology*, 14, 1175272. doi:10.3389/fpsyg.2023.1175272
- **Borovsky, A., Elman, J. L., & Kutas, M.** (2012). Once is enough: N400 indexes
  semantic integration of novel word meanings from a single exposure in context.
  *Language Learning and Development*, 8(3), 278–302.
  doi:10.1080/15475441.2011.614893
- **Borovsky, A., Elman, J. L., & Fernald, A.** (2012). Knowing a lot for one's age:
  Vocabulary skill and not age is associated with anticipatory incremental sentence
  interpretation in children and adults. *Journal of Experimental Child Psychology*,
  112(4), 417–436. [Abstract verified via ERIC EJ970158; DOI provisional — two
  different DOIs circulate in the citing literature.]
- **Hulme, R. C., Barsky, D., & Rodd, J. M.** (2019). Incidental learning and
  long-term retention of new word meanings from stories: The effect of number of
  exposures. *Language Learning*, 69(1), 18–43. doi:10.1111/lang.12313
- **Liu, Y., & Van Hell, J. G.** (2022). Novel word learning with verbal definitions
  and images: Tracking consolidation with behavioral and event-related potential
  measures. *Language Learning*. doi:10.1111/lang.12502
- **Hoffman, P., Evans, G. A. L., & Lambon Ralph, M. A.** (2014). The anterior
  temporal lobes are critically involved in acquiring new conceptual knowledge:
  Evidence for impaired feature integration in semantic dementia. *Cortex*, 50,
  19–31. doi:10.1016/j.cortex.2013.10.006
- **Dumay, N., & Gaskell, M. G.** (2007). Sleep-associated changes in the mental
  representation of spoken words. *Psychological Science*, 18(1), 35–39.
  doi:10.1111/j.1467-9280.2007.01845.x
- **Tamminen, J., Payne, J. D., Stickgold, R., Wamsley, E. J., & Gaskell, M. G.**
  (2010). Sleep spindle activity is associated with the integration of new memories
  and existing knowledge. *Journal of Neuroscience*, 30(43), 14356–14360.
  doi:10.1523/JNEUROSCI.3028-10.2010
- **Davis, M. H., Di Betta, A. M., Macdonald, M. J., & Gaskell, M. G.** (2009).
  Learning and consolidation of novel spoken words. *Journal of Cognitive
  Neuroscience*, 21(4), 803–820. doi:10.1162/jocn.2009.21059
- **Schimke, E. A. E., et al.** (2021). The effect of sleep on novel word learning in
  healthy adults: A systematic review and meta-analysis. *Psychonomic Bulletin &
  Review*. doi:10.3758/s13423-021-01980-3
- **Schimke, E. A. E., Copland, D. A., Gomersall, S. R., & Angwin, A. J.** (2024). To
  sleep or not to sleep? No effect of sleep on contextual word learning in younger
  adults. *Quarterly Journal of Experimental Psychology*, 77(4).
  doi:10.1177/17470218231179459
- **Gaskell, M. G.** (2024). EPS mid-career prize: An integrated framework for the
  learning, recognition and interpretation of words. *Quarterly Journal of
  Experimental Psychology*, 77(12), 2365–2384. doi:10.1177/17470218241284289
- **Consolidation of vocabulary is associated with sleep in children** (2012).
  *Developmental Science*. doi:10.1111/j.1467-7687.2012.01172.x

**Theoretical / review sources**

- **Bloom, P.** (2000). *How Children Learn the Meanings of Words*. MIT Press.
  [Précis verified; monograph not read in full.]
- **Clark, E. V.** (1973). What's in a word? On the child's acquisition of semantics
  in his first language. In T. E. Moore (Ed.), *Cognitive Development and the
  Acquisition of Language* (pp. 65–110). New York: Academic Press.
  [Verified via secondary sources [9][38]; the primary chapter was not accessed.]

**Secondary sources used for specific figures (flagged as such in §7)**

- Holland, A. (thesis, London Metropolitan University). *Words and facts: children's
  fast mapping, retention and extension.* [Source of the Markson & Bloom exposure
  counts and the 65–67% retention figure; a thesis, not a peer-reviewed article.]
- "Logical word learning: The case of kinship" (2022), PMC9166873. [Secondary
  exposition of Clark's feature hypothesis.]
