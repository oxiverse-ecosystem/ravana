# Grounding a novel subject: what the brain does, and what a small decoder should do instead of free-generating

**Date:** 2026-11-14
**Scope:** Research + implementation recommendation for RAVANA's neural decoder
fallback. Question: the decoder is fluent and grammatical but emits word salad
when asked about a subject it has no memorised definition for. What does the
brain do that prevents fluent-but-empty speech, and what should a small model
implement?

**Companion docs (do not re-read):** `docs/GROUNDING_MECHANISM.md` (the CPG
mechanism and the ACC conflict monitor), `docs/BRAIN_MESSAGE_PREPARATION.md`
(the pre-lexical message and lemma competition), `docs/CPG_MEASURED_EFFECT.md`,
`docs/RELATION_GATING_SETTLED.md`, `docs/NOGO_SUPPRESSION_FAILS.md`,
`docs/THETA_LOWER_IS_BETTER.md`, `docs/ESCALATION_INERT.md`,
`docs/DECODER_ONLY_NUMBERS.md`. This document adds the four pieces those do
not cover: the anterior temporal lobe, concept-to-predicate binding, the
predictive-coding account of production, and the "I don't know" route.

---

## 0. Executive summary

The brain does not produce fluent text and then check it. It **commits to a
lexical-semantic plan before it encodes form**, and the commitment is what
makes the sentence about something. Three separable mechanisms do this, and
they fail in three different, clinically distinguishable ways:

1. **The anterior temporal lobe (ATL) is the transmodal semantic hub.** It
   distills modality-specific features into amodal conceptual representations
   that support *combination* — the ability to say something specific about a
   concept rather than something generic. Bilateral ATL atrophy (semantic
   dementia) destroys this and produces exactly the decoder's signature:
   **fluent, grammatical, empty speech with preserved syntax and prosody**
   (Snowden, Goulding & Neary 1989; Hodges, Patterson, Oxbury & Funnell 1992;
   Patterson, Nestor & Rogers 2007). The formulator is intact; the semantic
   store that constrains it is not.
2. **Binding a concept to its predicate is a separate step from retrieving the
   concept.** Levelt's model has a pre-verbal **message** (macro/microplanning)
   whose terminal elements are *lexical concepts*, then **lemma selection** —
   competitive, incremental, Luce-ratio-gated (Levelt 1989; Levelt, Roelofs &
   Meyer 1999; Roelofs 1992). The **angular gyrus** and **middle temporal lobe**
   are the heteromodal convergence zones that integrate individual concepts
   into coherent *combinations* (Binder & Desai 2011; Price, Bonner, Peelle &
   Grossman 2015; Hagoort 2013, 2017). A model that retrieves a concept and
   then lets a next-token predictor fill the predicate has skipped this step.
3. **Production is predictive, not reactive.** A top-down semantic prediction
   is compared against the candidate form before articulation; the comparison
   is the constraint (Friston & Kiebel 2009; Bastos et al. 2012; Hickok 2012;
   Ylinen et al. 2015). Prediction error is the signal that the plan is not
   being realised — and it should **raise** the bar for speaking, not lower it.

And when the brain does **not** know, it does not free-associate. It computes a
**feeling-of-knowing** from the accessibility of partial information (Koriat
1993; Koriat & Levy-Sadot 2001), compares it against a report criterion, and
**withholds** — the "I don't know" response, which is a controlled output
decision, not a failure (Koriat & Goldsmith 1996). The anterior prefrontal
cortex (BA10) evaluates internally generated information and supports the
metacognitive verdict (Christoff, Ream, Geddes & Gabrieli 2003; Baird et al.
2013); the orbitofrontal/anterior-limbic reality filter suppresses activated
but currently irrelevant memory traces *before* they are articulated
(Schnider 2003).

**The one-sentence version:** *Fluent-but-empty speech is what a formulator
does when nothing upstream commits it to a proposition. The fix is not a
better formulator or a stricter post-hoc filter — it is a commitment step that
either binds the subject to a graph-supported predicate or declines to speak.*

---

## 1. The anterior temporal lobe: the hub that makes speech *about* something

### 1.1 The syndrome

Semantic dementia (SD) was named by **Snowden, Goulding & Neary (1989)**,
*Behavioural Neurology* 2:167–182, and given its definitive clinical description
by **Hodges, Patterson, Oxbury & Funnell (1992)**, *Brain* 115(6):1783–1806.
The diagnostic core is a **selective impairment of semantic memory** with:

- fluent, well-articulated spontaneous speech; **normal syntax, phonology,
  prosody and rate**;
- severe anomia, reduced vocabulary, impaired single-word comprehension;
- semantic paraphasias and pervasive deictic circumlocution ("that thing");
- **surface dyslexia**, preserved repetition, preserved perceptual and
  visuospatial abilities, preserved day-to-day episodic memory;
- **focal anterior/inferior temporal lobe atrophy** on imaging.

The theoretical statement of the hub view is **Patterson, Nestor & Rogers
(2007)**, *Nature Reviews Neuroscience* 8:976–987: semantic memory is
**distributed-plus-hub**. Modality-specific association cortices are the
spokes; the **bilateral ATL is the amodal hub** that distills them into
representations supporting *generalisation and inference on the basis of
conceptual structure rather than modality-specific features*. **Lambon Ralph &
Patterson (2008)**, *Annals of the NY Academy of Sciences* 1124:61–76, make the
same case from SD: the ATL "forms amodal semantic representations through the
distillation of the multimodal information projected to it."

The causal evidence is not only correlational. **Pobric, Jefferies & Lambon
Ralph (2007)**, *PNAS* 104(50):20137–20141, used offline 1 Hz rTMS over the
ATL in healthy participants and **mimicked SD**: selective slowing of
specific-level picture naming and word comprehension, with no effect on
equally demanding non-semantic tasks. A laterality study (Xie et al. 2017,
*Brain Structure and Function*) found the naming decrement for **left** ATL
only, consistent with the left ATL's stronger connection to left-lateralised
production areas.

### 1.2 Why the speech is *fluent and empty* rather than broken

This is the part that matters for the decoder. In SD the **formulator is
intact**. What degrades is the *content* the formulator is given. Patients
produce real words in grammatical frames; they simply have no specific concept
to bind to the predicate, so they substitute high-frequency, abstract,
generic vocabulary and deictic phrases. The clinical literature describes this
as "lexical impoverishment with overreliance on high-frequency, abstract words
replacing low-frequency, concrete vocabulary" (Hodges et al. 1992; reviewed in
*Clinical Interventions in Aging* 2025) and, in later stages, as speech
"meaningless and verbose."

**The dissociation that makes the localisation argument strong:** Wernicke's
aphasia (posterior superior temporal lesion) is *also* fluent, but it comes
with semantic and phonemic paraphasias, neologisms, jargon, and — critically —
**no error awareness**, because the comprehension system that would evaluate
the output is damaged. SD patients' output is *more relevant* than Wernicke's
(real words, empty ones) and their comprehension deficit is a *semantic* one.
Two different lesions, two different fluent syndromes. The constraint on
production lives in the **semantic store**, not in the formulator and not in
the monitor.

### 1.3 What this means for RAVANA's decoder

The decoder's failure signature is SD's, not Wernicke's: grammatical local
transitions, real vocabulary, no semantic commitment, and — because the
post-hoc gate *does* catch it — intact error awareness. The measured baseline
is the same shape: on-topic fraction **0.069** on the decoder path
(`docs/DECODER_ONLY_NUMBERS.md`), i.e. fewer than 7% of emitted content words
are graph-neighbours of the subject, with **~1.18 unsupported assertions per
output** (`docs/RELATION_GATING_SETTLED.md`).

**The implication is a reallocation of effort.** The formulator is not the
defective component. Improving the GRU, the attention head, or the training
corpus will not fix a deficit that is upstream of form. The two things that
would fix it are (a) a semantic store rich enough to commit to a specific
proposition about the subject, and (b) a commitment step that refuses to hand
the formulator an unbound subject. RAVANA has the second in prototype
(`ravana/src/ravana/learn/committed_proposition.py`) and the first only
partially.

**Honest flag.** SD is a *degeneration* of a previously intact semantic system;
RAVANA's decoder is a *developmental* system that never had a grounded store.
The failure signature is structurally apt, the etiology is not. Also, the ATL
literature is contested at the margins: anterior temporal *resection* for
epilepsy does not reliably produce domain-general semantic impairment (Drane
et al. 2008), and rTMS "virtual lesions" are not anatomically specific (Rounis
et al. 2006). The hub account is the best-supported reading, not a settled
fact. Grade: **[M]** for "ATL is necessary for amodal semantic combination";
**[S]** for "ATL damage produces fluent empty speech with preserved syntax."

---

## 2. Binding a concept to its predicate: the message, the lemma, and the angular gyrus

### 2.1 Levelt: there is a stage before words

Levelt's production model (Levelt 1989; **Levelt, Roelofs & Meyer 1999**,
*Behavioral and Brain Sciences* 22(1):1–75) is staged and feed-forward:
**conceptual preparation → lexical selection → morphological/phonological
encoding → phonetic encoding → articulation**. The MEG/EEG meta-analysis of
**Indefrey & Levelt (2004)**, *Cognition* 92:101–144, puts approximate windows
on the first two: conceptual preparation ~0–200 ms, lemma retrieval
~200–275 ms.

The load-bearing distinction for this problem:

- **Macroplanning** decides *what to say*: it elaborates the communicative
  intention into subgoals, selects and orders information, linearises.
- **Microplanning** decides *how to say it*: it gives each speech act
  propositional shape and maps states of affairs onto **lexical concepts** —
  concepts that have words in the language.

The output of microplanning is Levelt's **message**: a *non-linear,
propositional/relational structure* with **thematic role–concept bindings**
(`AGENT ↔ DOG`), amodal in format, whose terminal elements are lexical
concepts. It is not a word sequence.

**Lexical selection** then retrieves a **lemma** (the syntactic word:
category, gender, subcategorisation) given a lexical concept. In WEAVER++ this
is a **competitive statistical mechanism** — the Luce-ratio hazard rule:

```
P(select target per time unit) = a(target) / Σ a(all lemmas)
```

with activation spreading and decaying (`a(m, t+Δt) = a(m,t)(1−d) + Σ r·a(n,t)`).
Competitor activation *increases* selection time (Nozari, Dell & Oppenheim
2013). Selection is incremental, and function words and inflections are
*computed from the frame*, not retrieved.

The **Chang–Dell–Bock dual-path model (2006)**, *Psychological Review*
113(2):234–272, makes the same separation computationally: a **Meaning System**
holding event semantics and temporary Role↔Concept bindings, and a
**Sequencing System** (an SRN) that predicts the next word from left context —
with restricted communication between them.

**This is the single most important architectural fact for the decoder.** RAVANA's
GRU is a Sequencing System. It is trained to predict the next token from a
linear left context. It has **no message layer** between the concept graph and
the word readout, so it is asked to do microplanning *and* lexical selection in
one step. The brain does not do this: the relational structure exists first,
and words are selected *from* it. `docs/BRAIN_MESSAGE_PREPARATION.md` reaches
the same conclusion and notes that `ravana/src/ravana/learn/macroplanning.py`
was written to supply the missing stage.

### 2.2 Where the binding happens: angular gyrus and middle temporal lobe

Retrieving a concept and *combining* it with a predicate are dissociable, and
the combination step has a localisation.

- **Binder & Desai (2011)**, *Trends in Cognitive Sciences* 15(11):527–536:
  semantic memory consists of modality-specific representations plus
  **supramodal convergence zones** in the inferior parietal lobe (angular and
  supramarginal gyri), large parts of the middle and inferior temporal gyri,
  and anterior fusiform. Crucially, *"the prefrontal regions control top-down
  activation and selection of the content in posterior stores"* — posterior
  stores hold the content, frontal regions select it. This is the
  content/control split RAVANA needs: the graph is the posterior store, the
  commitment gate is the control.
- **Price, Bonner, Peelle & Grossman (2015)**, *Journal of Neuroscience*
  35(7):3276–3284: three converging experiments (fMRI in healthy adults,
  individual differences in AG structure, and AG atrophy in neurodegenerative
  disease) show that **conceptual combination** — integrating individual
  concepts into coherent semantic combinations — specifically modulates the
  **left angular gyrus**, independent of the modality of the content combined,
  and that AG atrophy predicts impaired combinatorial processing. Ease of
  combination is well predicted by lexical co-occurrence statistics.
- **Hagoort (2013)**, *Frontiers in Psychology* 4:416, and **Hagoort (2017)**:
  the **MUC** model — **Memory** (temporal cortex + angular gyrus store lexical
  and syntactic frames), **Unification** (left IFG, BA44/45/47, binds the
  retrieved frames into a whole-utterance structure), **Control** (prefrontal
  + ACC, manages speaker–listener context). Semantic unification recruits
  BA47/45; syntactic unification BA45/44. Unification "is characterized by a
  high-speed temporal profile including both **prediction and integration** of
  retrieved lexical elements."

**Implication for a small model.** "Binding a concept to its predicate" is not
a matter of retrieving the right noun. It is a **combinatorial operation over
retrieved frames**, and it is the operation RAVANA's decoder never performs.
The graph already stores the typed relation — `(subject, relation_type,
target)` — which *is* the bound predicate. The missing step is that the
decoder is not required to *realise* it. `docs/BOOST_STRENGTH_NOT_THE_LEVER.md`
records the measured consequence: CPG constrains the **argument** (the target)
but leaves the **relation** free, and an unsupported assertion is precisely a
relation the graph does not support. Adding relation gating moved supported
assertions 0.110 → 0.142 and unsupported 1.193 → 1.118
(`docs/RELATION_GATING_SETTLED.md`) — the first intervention that moved the
unsupported count at all.

### 2.3 What a small model should implement here

1. **A message object, not a string.** `ravana/src/ravana/learn/macroplanning.py`
   already builds one: `Message(subject, goal, bindings=[RoleBinding(role,
   concept, activation, relation)], competitors, selection_ratio)`, with roles
   assigned by *relation semantics* (`_role_for_relation`), not by activation
   order. **It is currently not imported by any production module** — only
   `tests/unit/test_macroplanning.py` imports it. That is a wiring gap, not a
   missing feature.
2. **Competitive lemma selection with a Luce ratio, not an argmax readout.**
   `macroplanning.select_lexical_concepts` implements this. The decoder's
   `output_proj` readout is an argmax/sample over a 15k-way softmax; there is
   no competition signal anywhere in the live path. The conflict measure
   (inverse of the top-two activation gap) that `docs/GROUNDING_MECHANISM.md`
   §1.2 borrows from **Nozari, Dell & Schwartz (2011)**, *Cognitive Psychology*
   63(1):1–33, is therefore not computable from the live decoder's outputs
   today.
3. **Require the relation, not just the target.** Already implemented as
   `CPGConfig.relation_words` (`committed_proposition.py:96`), and it is the
   only measured win on unsupported assertions.

---

## 3. Predictive coding: the top-down prediction *is* the constraint

### 3.1 The framework

**Friston & Kiebel (2009)**, *Philosophical Transactions of the Royal Society B*
364(1521):1211–1221: the brain inverts a hierarchical generative model of the
causes of its sensory input; under simplifying assumptions this is the
**suppression of prediction error at every level of the cortical hierarchy**,
and it "rests upon a balance between bottom-up (prediction error) and
top-down (empirical prior) influences." **Bastos, Usrey, Adams, Mangun, Fries
& Friston (2012)**, *Neuron* 76(4):695–711, give the microcircuit: **superficial
pyramidal cells (L2/3) encode prediction error and drive feedforward
connections; deep pyramidal cells (L5/6) encode predictions and drive feedback
connections**; L4 receives the feedforward input and mediates the comparison.
Prediction errors are **precision-weighted** — scaled by inverse variance,
which is what makes the gain term an uncertainty signal.

### 3.2 Applied to speech production

- **Hickok (2012)**, *Nature Reviews Neuroscience* 13(2):135–145: the **state
  feedback control (SFC)** model. The auditory phonological system defines a
  **sensory target**; the lemma activates the motor phonological
  representation; the two interact so the motor plan *hits the sensory target*.
  Forward prediction is instantiated as an **inhibitory motor-to-sensory
  signal**. If the wrong motor program is activated, it inhibits a non-target,
  leaving the correction signal active — so **error correction is part of
  motor planning from the start**, not a post-hoc check.
- **Okada, Matchin & Hickok (2018)**, *Psychonomic Bulletin & Review*
  25(1):423–430: silent articulation (but not imagined speech) activates
  auditory cortex even with no auditory input — the forward-prediction signal.
  A follow-up study implicated **left posterior middle temporal gyrus** in
  internal error correction.
- **Ylinen, Nora, Leminen, Hakala, Huotilainen, Shtyrov, Mäkelä & Service
  (2015)**, *Cerebral Cortex* 25(6):1576–1586: auditory-cortical suppression
  during covert rehearsal is **content-specific** — it occurs only when the
  heard item matches the rehearsed item. That specificity is what makes
  forward-prediction-based monitoring *accurate* rather than a blanket
  attenuation. Two auditory-motor circuits operate at different timescales:
  finer grain in the left hemisphere, coarser in the right.
- **Ylinen, Bosseler, Junttila & Huotilainen (2017)**, *Developmental Science*
  20(6):e12472: in 12- and 24-month-olds, a predictable word context speeds
  and strengthens the electrophysiological response, while a violated word
  expectancy generates a **prediction-error response whose magnitude
  correlates with vocabulary score**. Predictive coding accelerates word
  recognition *and* supports the mapping of novel word forms to meanings.
  (This is the Ylinen work most relevant to a *learning* system; Ylinen's
  2019 output is on prosodic/lexical processing and rhythmic learning —
  Zora, Riad & Ylinen 2019, *Journal of Neurolinguistics* 52:100856;
  Suppanen, Huotilainen & Ylinen 2019, *Infant Behavior and Development*
  57:101346 — not a production-prediction paper. Cite the 2015 and 2017 papers
  for the production claim.)

### 3.3 The computational implication: precision is a gate, and it should close

The Bastos et al. microcircuit makes one thing explicit that the CPG design got
backwards. Prediction error is **precision-weighted**. A high-precision
(low-variance) top-down prediction *suppresses* prediction error and licenses
the current plan; a low-precision prediction leaves error unsuppressed and
should **raise the bar** for emission.

`docs/GROUNDING_MECHANISM.md` §3.1 step 2 encodes this as
`θ = θ_base + α·pe + β·free_energy` — high prediction error raises the
criterion. That is the right direction. But `docs/THETA_LOWER_IS_BETTER.md`
measured the opposite outcome on the live instrument: **lowering** `theta_base`
from 0.30 to 0.15 improved on-topic (0.207 → 0.221), supported (0.142 → 0.155)
*and* unsupported (1.118 → 1.087) simultaneously.

The resolution is a **level-of-abstraction error**, and it is worth stating
precisely because it is the same error the decoder makes:

- In the brain, the ACC/conflict criterion gates **whether to commit to a plan
  at all** — a decision *upstream* of word selection.
- In RAVANA's CPG, `theta` gates **which words are admissible** — a decision
  *downstream* of commitment, applied to the token boost.

Raising `theta` therefore does not make the decoder more careful; it makes it
**quieter**, and the unguided model fills the silence with its own
distribution. `docs/THETA_LOWER_IS_BETTER.md` says this exactly: "The criterion
is not a conservatism knob in this implementation. It is a **coverage** knob."

**So the predictive-coding account does not recommend a higher threshold. It
recommends moving the threshold to the right place.** The prediction error
should gate *speech*, not *vocabulary*. That is the escalation path
(`_cpg_escalate`), which is implemented and verified on a genuinely absent
subject (`docs/ESCALATION_VERIFIED.md`: OFF → "I believe word is opposite to
definition."; ON → "honestly, xqzvwnonalword is a bit outside what i know right
now…") but **measured inert at scale** — bit-identical output on the 480-subject
instrument, because the instrument's subjects are graph-known by construction
and `_cpg_token_boost` returns `{}` (fail-open) for the few that are not
(`docs/ESCALATION_INERT.md`).

**Concrete recommendation:** keep `theta_base = 0.15` (measured best), and make
the *escalation* path the precision gate — but fix its reachability first. Two
specific defects, both in `response_gen.py`:

- `_cpg_token_boost` returns `{}` early when the subject has no node or no
  edges (`response_gen.py:7560-7561`, `7586-7587`). The caller's
  `if _cpg is not None` treats `{}` as a valid decision, `_cpg.committed`
  raises, and the `except` sets `_cpg_committed = True` — **fail-open**. So
  "no graph presence" and "graph present but nothing cleared" are
  indistinguishable, and only the second can escalate. That is why
  `docs/ESCALATION_INERT.md` measured `decline=0` on 20 probes.
- The escalation branch (`response_gen.py:1664-1672`) is gated on
  `_cpg_enabled and _cpg_escalate`, and **both default to `False`**
  (`engine.py:1291`, `1296`) and **neither is persisted** in the save payload
  (`engine.py:12415-12425` persists `use_linggen` but not the CPG flags). A
  promoted `use_linggen` survives a reload; a promoted CPG does not.

---

## 4. What the brain does when it does NOT know

### 4.1 Feeling-of-knowing is computed from accessibility, not read off

**Koriat (1993)**, *Psychological Review* 100(4):609–639, the **accessibility
model**: even when recall fails, people can judge whether the target is
available in memory. FOK is *"parasitic on the processes involved in attempting
to retrieve the target, relying on the accessibility of pertinent
information."* Partial clues come to mind — fragments of the target, semantic
attributes, episodic information, and activations from *neighbouring* targets.
People cannot monitor the accuracy of what comes to mind, so **both correct and
wrong partial clues contribute to FOK**; the judgments are accurate on average
because of the high output-bound accuracy of memory.

**Koriat & Levy-Sadot (2001)**, *JEPLMC* 27(1):34–53, integrate the competing
cue-familiarity account (Reder 1987; Metcalfe 1993) into a **cascaded model**:
cue familiarity is assessed *early*; accessibility is assessed *later* and
**only when cue familiarity is sufficiently high** to drive the memory
interrogation. Both contribute; the accessibility effect is mostly confined to
high-familiarity cues.

**Koriat (1997)**, *JEP:General* 126(4):349–370, the **cue-utilization**
framework: three cue classes — intrinsic (item properties), extrinsic
(learning conditions), and **mnemonic** (subjective experience: retrieval
fluency, accessibility). Practice shifts the basis of judgments from
theory-based cues toward mnemonic ones, which is when accuracy improves.

### 4.2 "I don't know" is a controlled output decision

**Koriat & Goldsmith (1996)**, *Psychological Review* 103(3):490–517: when
people may volunteer or withhold, they **selectively withhold low-confidence
answers**, trading quantity for accuracy. The framework is explicit that this
is a three-step process: *generate a best-candidate answer → monitor its
quality → decide whether to output it against a report criterion.* The
withholding decision tracks **subjective confidence**, and control sensitivity
(the correlation between confidence and correctness) is the quantity that
determines whether the "don't know" option helps.

This is the piece RAVANA's fallback gets *almost* right and then loses.
`_human_like_uncertainty` (`response_gen.py:6875`) is a genuine abstention with
a graded, state-modulated surface form — it varies openers by valence and
arousal, tracks recent openers to avoid cycling, and hands the floor back with
a question. What it lacks is the **monitoring step**: it fires when an
upstream path fails, not when a measured confidence crosses a criterion.
RAVANA's `Metacognition.fok_confidence` / `should_assert`
(`ravana/src/ravana/chat/metacognition.py:45-73`) *is* that monitor — a
saturating `1 − exp(−support/4)` curve over graph/retrieval support, gated at
`theta_withhold = 0.30`, with a learned shift from `EpistemicCalibrator`
(`docs/CAPABILITY_EPISTEMIC_CALIBRATION.md`). It is wired into the
metacognitive realizer (`response_gen.py:3641`) and the numeric-claim honesty
gate (`engine_web_search.py:546`) — **but not into the decoder path.** The
decoder's output is gated by `_sm_response_grounded`, a *resemblance* check,
and by `_final_emit_guard`, a *degeneracy* check. Neither asks "how much
support does the graph actually have for this subject?"

### 4.3 The prefrontal "don't know"

- **Christoff, Ream, Geddes & Gabrieli (2003)**, *Behavioral Neuroscience*
  117(6):1161–1168: rostrolateral PFC (lateral BA10) is selectively active
  during the **evaluation of internally generated information** — information
  that must be generated rather than perceived. The increase is specific to the
  *evaluation* phase, not generation or maintenance.
- **Baird, Smallwood, Gorgolewski & Margulies (2013)**, *Journal of
  Neuroscience* 33:16657–16665: medial and lateral anterior prefrontal networks
  support metacognitive ability for memory and perception.
- **Schnider (2003)**, *Nature Reviews Neuroscience* 4:662–671: the **reality
  filter**. Spontaneous confabulation — the production of fictitious stories
  with a strong feeling that they are true — follows lesions of the posterior
  orbitofrontal cortex / anterior limbic structures. The mechanism is a
  **failure to suppress activated but presently irrelevant memory traces**,
  and the suppression is *preconscious*: the correct rejection of an
  irrelevant item has an electrocortical correlate at **200–300 ms**, before
  recognition and re-encoding (which modulate at 400–480 ms). By the time an
  item is consciously recognised, its cortical representation has already been
  adjusted for whether it pertains to ongoing reality.

**This is the strongest single citation for RAVANA's design.** Confabulation is
not "the language model is bad." It is **a failure of suppression at the point
where an activated trace meets the current communicative goal** — and the
brain does that suppression *before* articulation, at a stage that has no
conscious access to the content. RAVANA's `_sm_response_grounded` is a
post-articulation content check. Schnider's result says the faithful
implementation is a **pre-articulation goal-relevance filter**: does the
activated proposition pertain to *this* query, or is it an intrusion from a
neighbouring node?

Note the resonance with the measured failure: `docs/DECODER_GROUNDING_VETO.md`
records that the gate failed a subject whose *real* associates clear the
threshold easily (`cos(rumour, gossip) = 0.506`) because the query's own
modifier dragged in *its* neighbours (`cos(rumour, heat) = 0.263`). That is
exactly an intrusion of presently irrelevant activated traces — the same
structure as Schnider's confabulation, at graph scale.

### 4.4 The failure mode if the model just generates freely

Stated plainly, because it is the deliverable's central claim:

**A small neural language model with no commitment step will always produce
fluent word salad for a subject it has no grounded proposition about.** Not
sometimes — always, and for a structural reason:

1. **The training objective has no truth term.** CE rewards the *next token
   given the left context*. It is a model of *fluency*, and it is doing its
   job. `docs/SESSION_GOAL_grounded_decoder.md` and
   `docs/GROUNDING_PASS_TOO_WEAK.md` record the attempts: a grounding pass
   exists (`scripts/train.py:187`, `train_decoder_grounded`) and was measured
   to be **empty** — the fitted `W_sm` projection is indistinguishable from
   noise (Frobenius 2.44 vs 1.25 for a random matrix of the same scale;
   `docs/WSM_IS_NOISE.md`). Rebalancing cannot help a null signal.
2. **The conditioning pathway is architecturally incoherent.**
   `docs/CODEBASE_DECODER_AUDIT.md` finds `ConceptAttentionHead` built with
   `vocab_size=hidden_dim=256`, so its output is a 256-D hidden-state bias
   blended 50/50 into `h` and then projected to vocab a second time; the head
   is trained only through `output_proj` and its shipped weights are ~12×
   smaller than random init — it has learned to be a near-no-op. The settle
   loop's `err` is computed and discarded; the update is a fixed interpolation
   toward `h_star`, so the emitted word is essentially the readout of the
   attractor's own state.
3. **The logit vector is the wrong level of intervention.** Measured:
   `token_boost` 2.5 → 5.0 is flat-to-negative (`docs/BOOST_STRENGTH_NOT_THE_LEVER.md`);
   NoGo suppression collapses output into worse salad
   (`docs/NOGO_SUPPRESSION_FAILS.md`). Both act on the logit vector, and the
   logit vector is not where the grounding decision lives.
4. **The post-hoc gate cannot create grounding, only withhold.** It rejects
   43% of outputs (`docs/DECODER_GROUNDEDNESS_BASELINE.md`) and raises
   on-topic 0.151 → 0.384, but the survivors still carry ~0.67 unsupported
   assertions each (`docs/DECODER_ONLY_NUMBERS.md`). A filter that can only
   say "no" leaves the model rarer, not better.

The result is a system that is *fluent by construction and ungrounded by
construction*, and the clinical name for that is semantic dementia. The
decoder is not malfunctioning. It is doing exactly what an intact formulator
does when the semantic hub gives it nothing to bind.

---

## 5. Recommendation: what a small model should implement

Ordered by (evidence strength × cost). Items 1–3 are cheap and already
half-built; item 4 is the load-bearing one.

### 5.1 Route the decoder through a commitment gate, and make "no commitment" a first-class outcome

**Do not** let the decoder run when the graph has no proposition for the
subject. Today the decoder runs and the gate withholds — which is the SD
signature with a monitor bolted on.

- Fix the fail-open in `_cpg_token_boost`: distinguish *"no node / no edges"*
  from *"node and edges exist but nothing cleared θ"* and return a decision
  object with `committed=False` in **both** cases, so the escalation branch is
  reachable. `docs/ESCALATION_INERT.md` measured `decline=0` on 20 probes
  because of exactly this.
- Persist `_cpg_enabled` / `_cpg_escalate` in the save payload
  (`engine.py:12415-12425`), as `use_linggen` already is. A promoted flag that
  does not survive a reload is not promoted.
- Keep `theta_base = 0.15` (measured best on all three axes). Do **not** raise
  it under uncertainty — that is the coverage/conservatism confusion
  (`docs/THETA_LOWER_IS_BETTER.md`). The precision gate belongs on *whether to
  speak*, not on *which words are permitted*.

### 5.2 Bind the relation, not just the target — and wire the message layer

`CPGConfig.relation_words` is implemented and is the only measured win on
unsupported assertions (supported 0.110 → 0.142, unsupported 1.193 → 1.118).
It is off by default. Turn it on and populate it from the graph's
`relation_type` vocabulary rather than a hand-authored table.

Then wire `ravana/src/ravana/learn/macroplanning.py`. It is a complete,
tested, pure implementation of Levelt's message layer — `build_message`
(spreading activation → Luce-ratio selection → role assignment by relation
semantics) and `message_to_boost` — and **no production module imports it.**
A module nothing imports is a paper exercise. The cheapest correct wiring is:
`build_message` → `message_to_boost` → the existing `token_boost` argument of
`NeuralDecoder.generate`, replacing the CPG target-only boost.

### 5.3 Add the pre-articulation goal-relevance filter (the reality filter)

Before decoding, ask: **does the activated proposition pertain to this query?**
Concretely, a subject's committed propositions should be scored against the
*query's* relation preference (RAVANA already computes `spread_pref` from the
question decomposer and `_relation_modulation_for_word`), and propositions
whose relation type does not match the queried relation should be demoted —
not emitted with a confidence equal to a matching one.

This is Schnider's reality filter at graph scale, and it is the direct fix for
the measured `heavy rumour` failure, where the gate was dragged down by
`heavy`'s neighbours. The current `_grounding_subject` head-noun reduction
(`response_gen.py:4204`) is a good patch on the lookup key; the filter is the
mechanism.

### 5.4 Give the decoder a real "don't know" that is monitored, not authored

`_human_like_uncertainty` is a good surface form with no monitor behind it.
Wire `Metacognition.fok_confidence` into the decoder path: compute support
from the graph (typed-edge count, retrieval success — the same inputs the
module already takes), and when `fok_confidence < theta_withhold`, **do not
decode**. That is Koriat & Goldsmith's output decision, and it is graded
rather than binary — the brain withholds *selectively*, by confidence, not by
a global switch.

The `EpistemicCalibrator` ledger (`docs/CAPABILITY_EPISTEMIC_CALIBRATION.md`)
already learns the threshold from experience. Its measured limitation is
explicit and worth respecting: it calibrates *response quality*, not *truth*.
Do not present a quality-calibrated threshold as a truth gate.

### 5.5 What NOT to do

- **Do not retrain the decoder with more CE.** The grounding pass is already
  outweighed and its projection is already null (`docs/WSM_IS_NOISE.md`).
  More of a null signal is more noise.
- **Do not raise the logit boost.** Measured flat-to-negative at 2.5 → 5.0.
- **Do not add logit suppression.** Measured to collapse output into worse
  salad; the brain's NoGo inhibits *competing plans*, not lexical items
  (`docs/NOGO_SUPPRESSION_FAILS.md`).
- **Do not tighten `_sm_response_grounded` further.** It is already rejecting
  43% of output and raising on-topic to 0.384. It is a good filter and the
  wrong tool for creating grounding.
- **Do not treat the decoder as the fix for a graph-coverage problem.** The
  honest reading of `docs/DECODER_GROUNDING_VETO.md` is that most subjects the
  decoder is asked about are not in the graph at all. As RAVANA learns
  concepts, more queries reach the decoder and more of its output passes —
  which makes concept acquisition the prerequisite, not a parallel workstream.

---

## 6. What would falsify this recommendation

1. **If wiring `macroplanning` into the live path does not move the decoder's
   unsupported-assertion count**, the message-layer hypothesis is wrong for
   this model — the deficit would then be in the decoder's readout, not the
   missing message.
2. **If the escalation path, once made reachable, fires on a large fraction of
   real queries**, then the graph is too sparse to ground generation and the
   correct response is concept acquisition, not a better gate. (Report the
   firing rate; do not enable on a 2-query probe — that is the mistake
   `docs/ESCALATION_VERIFIED.md` explicitly avoided.)
3. **If relation gating's gain does not replicate** on a fresh instrument run,
   the 6.3% unsupported-assertion reduction is noise. It was measured once, at
   n=204, with a stochastic decoder.
4. **If a subject with a rich, high-confidence graph neighbourhood still
   produces salad**, the failure is in the decoder's conditioning pathway
   (the `W_sm` null projection and the mis-wired attention head), and no
   commitment mechanism will help. That is the one case where decoder work is
   the right lever.
5. **Neural falsifiability:** the ATL hub account predicts that ATL disruption
   should impair *combination* more than *single-concept* retrieval. Pobric et
   al. (2007) support it (specific-level naming slowed, number naming
   unaffected); Price et al. (2015) support it for the AG. A clean virtual-lesion
   study isolating semantic *combination* at the ATL is still missing.

---

## 7. Honest flags: where the neuroscience is unsettled

1. **Predictive processing of *production* is less settled than of perception.**
   The Hickok SFC model is a well-specified computational proposal; the
   forward-prediction evidence (Okada et al. 2018; Ylinen et al. 2015) is
   real but is about *phonological/motor* prediction. Direct evidence for
   *semantic* forward models in production is thin. The strongest evidence
   remains lesion/behavioural.
2. **The ATL hub account is contested at the margins.** Anterior temporal
   resection does not reliably produce domain-general semantic impairment
   (Drane et al. 2008); rTMS virtual lesions are not anatomically specific
   (Rounis et al. 2006); and three competing accounts of ATL function
   (domain-general hub, unique-entity store, social-concept store) remain live
   (see the review in *Brain* 2010, PMC2791360). Grade **[M]**.
3. **The angular gyrus is over-claimed in the literature.** Price et al. (2015)
   is a strong three-experiment convergence, but the AG is also implicated in
   attention, episodic retrieval and number processing; "conceptual combination"
   is a narrow claim and should not be inflated into "the AG is the semantic
   hub." The hub claim belongs to the ATL; the AG claim is *combination*.
4. **Koriat's accessibility model is a model of *feeling*-of-knowing, not of
   accuracy.** FOK is computed from accessibility regardless of source, so it
   is systematically fallible — which is why Koriat & Goldsmith need a separate
   report criterion. A system that implements only the accessibility
   heuristic will be overconfident exactly where partial clues are plentiful.
5. **Schnider's reality filter is a lesion account of a pathological state.**
   Using it to justify a pre-articulation gate in a healthy system is an
   extrapolation. It is a good extrapolation — the mechanism (suppress
   irrelevant activated traces before articulation) is exactly what a
   confabulation-free system needs — but it is not a direct test.
6. **The decoder's "SD analogy" is functional, not etiological** (§1.3).
7. **"Grounding" is not one thing.** The brain grounds utterances in semantics,
   episodic context, interoception and social common ground simultaneously.
   RAVANA's ConceptGraph captures the first two. This is a partial model.

---

## 8. Citations

- **Baird, B., Smallwood, J., Gorgolewski, K. J., & Margulies, D. S.** (2013).
  Medial and lateral networks in anterior prefrontal cortex support
  metacognitive ability for memory and perception. *Journal of Neuroscience*,
  33, 16657–16665.
- **Bastos, A. M., Usrey, W. M., Adams, R. A., Mangun, G. R., Fries, P., &
  Friston, K. J.** (2012). Canonical microcircuits for predictive coding.
  *Neuron*, 76(4), 695–711. doi:10.1016/j.neuron.2012.10.038
- **Binder, J. R., & Desai, R. H.** (2011). The neurobiology of semantic
  memory. *Trends in Cognitive Sciences*, 15(11), 527–536.
  doi:10.1016/j.tics.2011.10.001
- **Binder, J. R., Desai, R. H., Graves, W. W., & Conant, L. L.** (2009).
  Where is the semantic system? A critical review and meta-analysis of 120
  functional neuroimaging studies. *Cerebral Cortex*, 19, 2767–2796.
- **Chang, F., Dell, G. S., & Bock, K.** (2006). Becoming syntactic.
  *Psychological Review*, 113(2), 234–272.
- **Christoff, K., Ream, J. M., Geddes, L. P. T., & Gabrieli, J. D. E.** (2003).
  Evaluating self-generated information: Anterior prefrontal contributions to
  human cognition. *Behavioral Neuroscience*, 117(6), 1161–1168.
- **Friston, K., & Kiebel, S.** (2009). Predictive coding under the free-energy
  principle. *Philosophical Transactions of the Royal Society B*, 364(1521),
  1211–1221. doi:10.1098/rstb.2008.0300
- **Hagoort, P.** (2013). MUC (Memory, Unification, Control) and beyond.
  *Frontiers in Psychology*, 4, 416. doi:10.3389/fpsyg.2013.00416
- **Hagoort, P.** (2017). The neurobiological infrastructure for syntax and
  semantics. See also Hagoort, P. (2019), "Unification" in *The Core and Beyond
  in the Language-Ready Brain*.
- **Hickok, G.** (2012). Computational neuroanatomy of speech production.
  *Nature Reviews Neuroscience*, 13(2), 135–145. doi:10.1038/nrn3158
- **Hodges, J. R., Patterson, K., Oxbury, S., & Funnell, E.** (1992). Semantic
  dementia: Progressive fluent aphasia with temporal lobe atrophy. *Brain*,
  115(6), 1783–1806. doi:10.1093/brain/115.6.1783
- **Indefrey, P., & Levelt, W. J. M.** (2004). The spatial and temporal
  signatures of word production components. *Cognition*, 92, 101–144.
- **Koriat, A.** (1993). How do we know that we know? The accessibility model
  of the feeling of knowing. *Psychological Review*, 100(4), 609–639.
- **Koriat, A.** (1997). Monitoring one's own knowledge during study: A
  cue-utilization approach to judgments of learning. *Journal of Experimental
  Psychology: General*, 126(4), 349–370.
- **Koriat, A., & Goldsmith, M.** (1996). Monitoring and control processes in
  the strategic regulation of memory accuracy. *Psychological Review*, 103(3),
  490–517.
- **Koriat, A., & Levy-Sadot, R.** (2001). The combined contributions of the
  cue-familiarity and accessibility heuristics to feelings of knowing.
  *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 27(1),
  34–53.
- **Lambon Ralph, M. A., & Patterson, K.** (2008). Generalization and
  differentiation in semantic memory: Insights from semantic dementia.
  *Annals of the New York Academy of Sciences*, 1124, 61–76.
- **Levelt, W. J. M.** (1989). *Speaking: From Intention to Articulation*. MIT
  Press.
- **Levelt, W. J. M., Roelofs, A., & Meyer, A. S.** (1999). A theory of lexical
  access in speech production. *Behavioral and Brain Sciences*, 22(1), 1–75.
- **Nozari, N., Dell, G. S., & Schwartz, M. F.** (2011). Is comprehension
  necessary for error detection? A conflict-based account of monitoring in
  speech production. *Cognitive Psychology*, 63(1), 1–33.
- **Okada, K., Matchin, W., & Hickok, G.** (2018). Neural evidence for
  predictive coding in auditory cortex during speech production.
  *Psychonomic Bulletin & Review*, 25(1), 423–430.
- **Patterson, K., Nestor, P. J., & Rogers, T. T.** (2007). Where do you know
  what you know? The representation of semantic knowledge in the human brain.
  *Nature Reviews Neuroscience*, 8, 976–987. doi:10.1038/nrn2277
- **Pobric, G., Jefferies, E., & Lambon Ralph, M. A.** (2007). Anterior
  temporal lobes mediate semantic representation: Mimicking semantic dementia
  by using rTMS in normal participants. *PNAS*, 104(50), 20137–20141.
- **Price, A. R., Bonner, M. F., Peelle, J. E., & Grossman, M.** (2015).
  Converging evidence for the neuroanatomic basis of combinatorial semantics
  in the angular gyrus. *Journal of Neuroscience*, 35(7), 3276–3284.
- **Roelofs, A.** (1992). A spreading-activation theory of lemma retrieval in
  speaking. *Cognition*, 42, 107–142.
- **Schnider, A.** (2003). Spontaneous confabulation and the adaptation of
  thought to ongoing reality. *Nature Reviews Neuroscience*, 4, 662–671.
  doi:10.1038/nrn1179
- **Snowden, J. S., Goulding, P. J., & Neary, D.** (1989). Semantic dementia: A
  form of circumscribed cerebral atrophy. *Behavioural Neurology*, 2(3),
  167–182.
- **Xie, W., et al.** (2017). Laterality of anterior temporal lobe repetitive
  transcranial magnetic stimulation. *Brain Structure and Function*.
  (Cited for the left-ATL naming-specificity finding; verify before relying on
  the exact statistics.)
- **Ylinen, S., Nora, A., Leminen, A., Hakala, T., Huotilainen, M., Shtyrov,
  Y., Mäkelä, J. P., & Service, E.** (2015). Two distinct auditory-motor
  circuits for monitoring speech production as revealed by content-specific
  suppression of auditory cortex. *Cerebral Cortex*, 25(6), 1576–1586.
  doi:10.1093/cercor/bht351
- **Ylinen, S., Bosseler, A., Junttila, K., & Huotilainen, M.** (2017).
  Predictive coding accelerates word recognition and learning in the early
  stages of language development. *Developmental Science*, 20(6), e12472.
  doi:10.1111/desc.12472

---

## 9. One-paragraph answer

The brain produces grounded sentences because it **commits to a lexical-semantic
plan before it encodes form**: the anterior temporal lobe distills
modality-specific features into amodal concepts that can be *combined*
(Patterson et al. 2007; Snowden et al. 1989; Hodges et al. 1992 — its atrophy
gives fluent, grammatical, empty speech), Levelt's message layer binds
concepts to thematic roles and selects lemmas competitively (Levelt 1989;
Levelt et al. 1999; Roelofs 1992), the angular gyrus and middle temporal lobe
perform the conceptual combination (Binder & Desai 2011; Price et al. 2015;
Hagoort 2013), and a top-down prediction is compared against the candidate
form before articulation, with precision-weighted prediction error gating the
plan (Friston & Kiebel 2009; Bastos et al. 2012; Hickok 2012; Ylinen et al.
2015). When the brain does not know, it computes a feeling-of-knowing from the
accessibility of partial information (Koriat 1993; Koriat & Levy-Sadot 2001),
compares it to a report criterion, and **withholds** (Koriat & Goldsmith 1996),
with anterior prefrontal cortex evaluating internally generated information
(Christoff et al. 2003) and an orbitofrontal reality filter suppressing
irrelevant activated traces before articulation (Schnider 2003). RAVANA's
decoder fails because it is an intact formulator with no commitment step: it
retrieves a subject, then lets a next-token predictor fill the predicate. The
fix is to route it through a commitment gate that either binds the subject to
a graph-supported `(subject, relation, target)` proposition — relation
included, not just target — or declines to decode, with the decline driven by
a monitored confidence rather than an upstream path failure. Free generation
for an ungrounded subject is not a calibration problem; it is the predicted
output of the architecture, and no amount of logit biasing or post-hoc
filtering will change that.
