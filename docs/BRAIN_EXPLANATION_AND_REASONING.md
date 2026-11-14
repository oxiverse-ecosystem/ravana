# Brain Explanation and Reasoning

**Status:** research summary + implementation proposal. No code changed.
**Scope:** how a human brain builds a multi-step causal *explanation* of a familiar
topic, how it decides the explanation is adequate, and what a small (~20 W, no-LLM,
online-learning) architecture should implement instead of the current
association-first behaviour.

**The measured defect this addresses.** Asked `why do you think it rains`, RAVANA
answers `i'm drawn to rains. it sits close to people, which i care about` — an
affective/associative reply. Asked `explain how a battery works`, it answers
`i don't have a clean definition for battery, but it is tied to plasma and cells`.
Both are *one-hop* outputs from the graph's `semantic` edges. The graph already
contains multi-step causal structure (`chain_walker._causal_forward_simulate`,
`_causal_chain_search`, `_seed_physics_causal`, `core/causal_schema.py`) and the
question router already labels `why`/`how` as `causal`
(`language/prefrontal_workspace.py:1013-1014`), but nothing composes that
structure into an explanation and nothing checks the result for adequacy.

---

## 0. Executive summary of the mechanism to build

Five stages, each with a brain analogue and a concrete RAVANA hook:

| # | Stage | Brain analogue | RAVANA hook |
|---|-------|----------------|-------------|
| 1 | **Route**: decide the query is a *causal-explain* request, not a recall/definition/social turn | LIFG unification + PFC task-set | `PrefrontalWorkspace.detect_question_type` → `why`/`how` → `QTYPE_PRIMARY_RELATION["causal"]` |
| 2 | **Assemble**: find a 2–4 step *causal* path from the subject, forward (effects) and/or backward (causes) | MD network + RLPFC relational integration | new `_explanation_chain(subject, target=None, k=2..4)` over `relation_type == "causal"` edges only |
| 3 | **Adequacy gate**: score the chain on coverage, typed-link fraction, provenance, and *unexplained-gap* count; accept/reject/hedge | aPFC/BA10 metacognitive monitoring; Rozenblit & Keil illusion-of-explanatory-depth | new `ExplanationMonitor` reading `metacognition.fok_confidence` + `provenance.admit_as_fact` |
| 4 | **Realize**: linearize the accepted chain into language with causal connectives and graded modality | LIFG unification → Broca hierarchy | reuse `_RELATION_PREDICATE["causal"]`, `hedges.modality_from_support`, `DiscourseType.CAUSAL_EXPLAIN` |
| 5 | **Learn online**: harvest causal links from the user's utterance and from web text into typed edges with provenance | causal induction gated by structure beliefs (Dorfman 2021) | extend `_extract_and_store_causal_relations` (conversation) and `web/web_to_graph.py` (web) |

The one-sentence design rule: **an explanation is a typed causal path plus an
adequacy verdict, and the verdict is computed from the path's own structure —
never from the affective salience of its endpoints.**

---

## 1. The neuroscience of causal reasoning and explanation

### 1.1 Causal vs. associative judgment is a real neural dissociation

The cleanest single experiment is **Satpute, Fenker, Waldmann, Tabibnia,
Holyoak & Lieberman (2005), *An fMRI study of causal judgments*, European
Journal of Neuroscience 22(5):1233–1238.** Identical word pairs (e.g.
`moon–tide`, `wind–erosion`) were judged in blocked runs as either *causally
related* or *associatively related*. Because the stimuli were held constant, any
difference is attributable to the *judgment type*, not the content.

- **Causal > associative** activated **left dorsolateral prefrontal cortex
  (DLPFC), BA 46** (Talairach −30, 40, 18 and −36, 48, 20; Z ≈ 3.8–4.0) and
  **right precuneus (BA 7/19)**.
- **Associative > causal** activated **right superior temporal gyrus (BA 22)**.
- The effect survived covarying out both reaction time and error rate, so it is
  not a difficulty artefact. Behaviourally, causal judgments were *harder*
  (15% vs. 8% errors).

Satpute et al.'s interpretation is the important part for us: an associative
judgment can be made by **spreading activation** between the two concepts. A
causal judgment additionally requires **role binding** — holding the two
referents in working memory *bound to the roles `cause` and `effect`*, because
causal relations are asymmetric (`smoking → lung cancer`, not the reverse). They
cite Hummel & Holyoak's (2003) relational-inference framework and the DLPFC
role-binding literature (Goel et al. 1997; Waltz et al. 1999; Christoff et al.
2001; Kroger et al. 2002).

A meta-analysis broadens the picture: **Feng, Wang, Liu, Wang, Tian & Fan
(2021), *Neural Correlates of Causal Inferences in Discourse Understanding and
Logical Problem-Solving: A Meta-Analysis Study*, Frontiers in Human Neuroscience
15:666179** (ALE over 19 discourse experiments and 20 problem-solving
experiments) found:

- **Discourse-level causal inference** recruits a **left-lateralized
  frontotemporal system**: left inferior frontal gyrus (LIFG), left middle
  temporal gyrus, bilateral medial prefrontal cortex.
- **Logical-problem-solving causal inference** recruits a **non-overlapping**
  frontal-parietal system: LIFG, bilateral middle frontal gyri, dorsal MPFC,
  left inferior parietal lobule.
- Both were related to memory and executive-function terms; only the discourse
  form was related to theory-of-mind terms.

**Design consequence.** RAVANA's `semantic` edges are the associative substrate
(spreading activation, `chain_walker._walk_chain`). Its `causal` edges are the
role-bound substrate. The defect — `i'm drawn to rains` — is what happens when a
`why` query is answered from the associative substrate. The fix is not "add more
causal edges"; it is **route the `why`/`how` query to the causal substrate and
forbid a one-hop associative answer from satisfying it.**

### 1.2 The causal-model tradition: Sloman, Lagnado, Tenenbaum

The computational claim behind the neural one is that people reason over
**causal models** (directed structures with a privileged intervention operator),
not over association strengths.

- **Sloman (2005), *Causal Models: How People Think about the World and Its
  Alternatives*, Oxford University Press.** Lays out the causal-model view:
  causal relations are directional, support counterfactuals, and are distinct
  from similarity or covariation.
- **Sloman & Lagnado (2005), *Do we "do"?*, Cognitive Science 29(1):5–39.**
  Shows that people distinguish *observational* inference from *interventional*
  inference — the `do(X)` operator of Pearl's calculus — and that this
  distinction is not recoverable from associative strength.
- **Sloman & Lagnado (2015), *Causality in Thought*, Annual Review of
  Psychology 66:223–247.** The mature review: causal learning and causal
  reasoning are intertwined and recruit similar representations; covariation
  accounts are insufficient.
- **Lagnado (2011), *Causal thinking*, in Illari, Russo & Williamson (eds.),
  *Causality in the Sciences*, Oxford University Press.** Argues explicitly
  against the covariation-based (associative) account of causal learning.
- **Tenenbaum, Kemp & Griffiths (2011), *How to grow a mind: Statistics,
  structure, and abstraction*, Science 331:1279–1285.** The framework paper:
  human inference is Bayesian inference over *structured* hypotheses, and causal
  relations are the paradigm case.
- **Griffiths & Tenenbaum (2009), *Theory-based causal induction*, Psychological
  Review 116(4):661–716.** Formalises causal induction as Bayesian inference
  over a hypothesis space constrained by a *theory* (e.g. causal power,
  causal linearity). People's causal judgments deviate from pure contingency
  models in exactly the way a theory-constrained Bayesian model predicts.
- **Goodman, Ullman & Tenenbaum (2011), *Learning a theory of causality*,
  Psychological Review 118(1):110–119.** Shows that the abstract causal
  *framework* itself (the `do` operator, screening-off) can be *learned* from
  data — relevant to RAVANA's online-learning requirement.
- **Kemp & Tenenbaum (2008), *The discovery of structural form*, PNAS
  105(31):10687–10692.** Structure (graph shape) is learned jointly with
  instance-level parameters; a small architecture should not hardcode its causal
  topology.
- **Kemp, Goodman & Tenenbaum (2010), *Learning to learn causal models*,
  Cognitive Science 34(7):1185–1243.** Higher-level "overhypotheses" about the
  *form* of causal relations are learned across domains and then constrain new
  learning — the mechanism behind RAVANA's `CausalSchemaLearner` innate priors.
- **Meder, Mayrhofer & Waldmann (2014), *Structure induction in diagnostic
  causal reasoning*, Psychological Review 121(3):277–301.** Diagnostic
  (effect→cause) reasoning is modelled as structure induction over causal Bayes
  nets, not as a single associative lookup.
- **Holyoak & Cheng (2011), *Causal learning and inference as a rational
  process: the new synthesis*, Annual Review of Psychology 62:135–163.** The
  synthesis of the Bayesian and associative traditions: associative processes
  are real but are *constrained* by causal structure.
- **Pearl (2009), *Causality: Models, Reasoning, and Inference* (2nd ed.),
  Cambridge University Press** and **Glymour (2001), *The Mind's Arrows*, MIT
  Press** supply the formalism (`do`-calculus, structural causal models).

**Design consequence.** RAVANA's graph is already a causal-model substrate
(directed typed edges, `do`-like counterfactual simulation in
`_causal_forward_simulate`). What it lacks is the *inference discipline*: a
causal answer must be a path through `causal` edges with a direction, and the
system must be able to say "I have no causal path here" rather than substitute an
association. Note the honest counter-evidence in §6: **Bisra et al. (2018)** and
**the "No Causal Self-Explanation Effect for Factual Knowledge" paper (Applied
Cognitive Psychology, 2025)** show the self-explanation effect is weaker than
textbook accounts claim — see §4.4.

---

## 2. How the brain builds a *multi-step* chain rather than a single association

### 2.1 Relational integration and the rostrolateral prefrontal cortex

A single association is one relation. A causal explanation is *several relations
considered simultaneously* — the defining task of **relational integration**.

- **Christoff, Prabhakaran, Dorfman, Zhao, Kroger, Holyoak & Gabrieli (2001),
  *Rostrolateral prefrontal cortex involvement in relational integration during
  reasoning*, NeuroImage 14(5):1136–1149.** The foundational study. Participants
  solved Raven-style problems under three conditions: 1-relation (no
  integration), 2-relation (integration), 3-relation (integration + further
  processing). **RLPFC (lateral BA 10)** was disproportionately engaged by the
  2-relation condition; DLPFC responded to relational complexity more
  parametrically.
- **Kroger, Sabb, Fales, Bookheimer, Cohen & Holyoak (2002), *Recruitment of
  anterior dorsolateral prefrontal cortex in human reasoning: a parametric study
  of relational complexity*, Cerebral Cortex 12(5):477–485.** Confirms a
  *parametric* DLPFC response as the number of relations to be integrated rises.
- **Smith, Keramatian & Christoff (2007), *Localizing the rostrolateral
  prefrontal cortex at the individual level*, NeuroImage 36(4):1387–1396.**
  Provides an individual-level localizer — the effect is not a group-average
  artefact.
- **Smith, Keramatian, Smallwood & Christoff (2013), *Left, but not right,
  rostrolateral prefrontal cortex meets a stringent test of the relational
  integration hypothesis*, Neuropsychologia 51(13):2724–2733.** The lateralized
  result: it is specifically **left** RLPFC that survives stringent controls.
- **Waltz, Knowlton, Holyoak, Boone, Mishkin, de Menezes Santos, Thomas &
  Miller (1999), *A system for relational reasoning in human prefrontal cortex*,
  Psychological Science 10(2):119–125.** Relational reasoning (not just working
  memory) recruits prefrontal cortex; patients with prefrontal damage are
  selectively impaired.
- **Crone, Wendelken, van Leijenhorst, Honomichl, Christoff & Bunge (2009),
  *Neurocognitive development of relational reasoning*, Developmental Science
  12(1):55–66** and **Wendelken, Chung & Bunge (2012)** — visuospatial and
  semantic relational matching tasks show *overlapping* left RLPFC activation,
  supporting a **domain-general** relational-integration function.
- **Badre & D'Esposito (2009), *Is the rostro-caudal axis of the frontal lobe
  hierarchical?*, Nature Reviews Neuroscience 10(9):659–669** and **Christoff,
  Keramatian, Gordon, Smith, Mädler, Evans, Marois & Hester (2009), *Prefrontal
  organization of cognitive control according to levels of abstraction*, Brain
  Research 1286:94–105.** A rostro-caudal gradient: progressively more rostral
  PFC for progressively more abstract control. Relational integration sits at
  the rostral end.
- **Koechlin & Summerfield (2007), *An information theoretical approach to
  prefrontal executive function*, Trends in Cognitive Sciences 11(6):229–235**
  and **Koechlin (2014), *An evolutionary computational theory of prefrontal
  executive function in decision-making*, Phil. Trans. R. Soc. B 369:20130474.**
  The cascade model: posterior→anterior LPFC implements contextual → episodic →
  branching control. A multi-step chain is exactly an *episodic* control program
  (a sequence of sub-goals), not a single selection.

**Design consequence.** A 2–4 step explanation is a *relational-integration*
task, not a retrieval task. The architecture should (a) build the chain
explicitly as a sequence of bound `(cause, effect)` role pairs, and (b) treat
chain length as a control parameter with a cost, not as "walk until you run out
of edges".

### 2.2 The multiple-demand (MD) network: assembling sub-goals into a program

- **Duncan (2010), *The multiple-demand (MD) system of the primate brain: mental
  programs for intelligent behaviour*, Trends in Cognitive Sciences 14(4):172–179.**
  The MD network (lateral frontal, insular, dorsomedial frontal, parietal) is
  active across *diverse* demanding tasks and correlates with fluid
  intelligence. Its proposed function: **assembling a series of sub-tasks into a
  structured mental program**.
- **Duncan (2013), *The structure of cognition: attentional episodes in mind and
  brain*, Neuron 80(1):35–50.** Cognition as a sequence of *attentional
  episodes*: select → act → reset. A chain of reasoning is a sequence of such
  episodes.
- **Woolgar, Parr, Cusack, Thompson, Nimmo-Smith, Torralva, Roca, Antoun,
  Manes & Duncan (2010), *Fluid intelligence loss linked to restricted regions
  of damage within frontal and parietal cortex*, PNAS 107(33):14899–14902.**
  Lesion mapping: damage to a restricted frontal-parietal set predicts
  fluid-intelligence loss.
- **Woolgar, Duncan et al. (2018), *The multiple-demand system but not the
  language system supports fluid intelligence*, Nature Human Behaviour 2(3):200–204.**
  A clean dissociation: MD supports fluid intelligence; the language-selective
  network does not. This matters for RAVANA because it means *explanation
  structure* and *explanation wording* are separable — see §4.
- **Assem, Glasser, Van Essen & Duncan (2020), *A domain-general cognitive core
  defined in multimodally parcellated human cortex*, Cerebral Cortex
  30(8):4361–4380.** A 360-parcel definition of the MD core; confirms
  domain-generality.
- **Fedorenko & Duncan (2019)** and **Fedorenko, Ivanova & Regev (2024), *The
  language network as a natural kind within the broader landscape of the human
  brain*, Nature Reviews Neuroscience** — the language network is a distinct,
  left-lateralized frontotemporal system, separable from MD and DMN.

**Design consequence.** The MD analogue in RAVANA is the *chain assembler plus
its control loop*: a bounded search that selects the next causal hop
(sub-goal), commits it, and resets. It should be **separable from the
language-production path** (`surface_realizer`, `syntactic_assembly`), because
the brain's language network is not what does the reasoning. Concretely: the
explanation module should return a *structured chain object* (list of typed hops
+ a verdict), and the realizer should consume that object. Today
`_plan_causal_explain` (`prefrontal_workspace.py:1513`) picks associations by a
`_has_causal_hint` *string match on the label* (`"cause" in label`) — that is a
lexical proxy for a relation type the graph already stores explicitly. It
should read `edge.relation_type == "causal"` instead.

---

## 3. How the brain knows an explanation is complete or satisfactory

### 3.1 Metacognitive monitoring and the anterior prefrontal cortex

- **Nelson & Narens (1990), *Metamemory: a theoretical framework and new
  findings*, in Bower (ed.), *The Psychology of Learning and Motivation* 26.**
  The monitoring/control distinction: a meta-level monitors an object-level and
  can direct it. This is the architectural template — the explanation monitor
  must be a *separate* reader over the explanation, not a property of the chain
  builder.
- **Hart (1965)** introduced the recall-judge-recognition (RJR) paradigm for the
  feeling of knowing; **Koriat (1993), *How do we know that we know? The
  accessibility model of the feeling of knowing*, Psychological Review
  100(4):609–639** showed FOK is based on the *accessibility* of partial
  information, not direct access to the trace — i.e. it is a computed
  confidence from cues, which is exactly implementable.
- **Fleming & Dolan (2012), *The neural basis of metacognitive ability*, Phil.
  Trans. R. Soc. B 367(1594):1338–1349** and **Fleming, Huijgen & Dolan (2012),
  *Prefrontal contributions to metacognition in perceptual decision making*,
  Journal of Neuroscience 32(18):6117–6125.** Prefrontal cortex, especially
  **lateral frontopolar cortex (BA 10)**, supports metacognitive evaluation.
- **Vaccaro & Fleming (2018), *Thinking about thinking: a coordinate-based
  meta-analysis of neuroimaging studies of metacognitive judgements*, Brain and
  Neuroscience Advances 2.** Meta-analysis of 47 studies: consistent activation
  in medial and lateral PFC, including **right rostrolateral PFC**, with
  functional connectivity to task-specific sensory areas during metacognitive
  reports.
- **Bang & Fleming (2018), *Distinct encoding of decision confidence in human
  medial prefrontal cortex*, PNAS 115(23):6082–6087.** Confidence is encoded
  distinctly from its components (evidence certainty, task demands) — a
  confidence signal can be *about the explanation* rather than about the
  retrieval.
- **Morales, Lau & Fleming (2018), *Domain-general and domain-specific patterns
  of activity supporting metacognition in human prefrontal cortex*, Journal of
  Neuroscience 38(14):3534–3546.** Both domain-general (frontopolar) and
  domain-specific patterns exist. A small architecture should implement the
  domain-general kind: one monitor, many content types.
- **Fleming (2024), *Metacognition and confidence: a review and synthesis*,
  Annual Review of Psychology 75:241–268.** The current synthesis; distinguishes
  metacognitive *bias*, *sensitivity*, and *efficiency* — a small system should
  at minimum track *sensitivity* (does confidence track actual chain quality?).
- **Ackerman & Thompson (2017), *Meta-reasoning: Monitoring and control of
  thinking and reasoning*, Trends in Cognitive Sciences 21(8):607–617.** The
  direct analogue: reasoners terminate because they are *satisfied* or because
  they hit a resource limit. Satisficing is the stopping rule.

### 3.2 The "explanatory gap" feeling and the illusion of explanatory depth

- **Rozenblit & Keil (2002), *The misunderstood limits of folk science: an
  illusion of explanatory depth*, Cognitive Science 26(5):521–562.** People rate
  their understanding of familiar devices (zippers, toilets, helicopters) as
  high — until they are asked to *produce a step-by-step explanation*, at which
  point confidence drops sharply. The feeling of understanding is largely
  *unearned*; the act of explaining is what exposes the gap.
- **Keil (2006), *Explanation and understanding*, Annual Review of Psychology
  57:227–254.** Reviews the evidence that understanding is not the same as
  having a description; explanations are graded and people are sensitive to
  their own explanatory limits.
- **Lombrozo (2006), *The structure and function of explanations*, Trends in
  Cognitive Sciences 10(10):464–470.** Explaining changes the probability one
  assigns to causal claims and drives inference; explanation has a functional
  role in learning.
- **Lombrozo (2007), *Simplicity and probability in causal explanation*,
  Cognitive Psychology 55(3):232–257** and **Vrantsidis & Lombrozo (2022),
  *Simplicity as a cue to probability: multiple roles for simplicity in
  evaluating explanations*, Cognitive Science.** People prefer explanations with
  fewer unexplained causes; simplicity is used as a cue to probability.
- **Vasilyeva, Wilkenfeld & Lombrozo (2017), *Contextual utility affects the
  perceived quality of explanations*, Psychonomic Bulletin & Review
  24(4):1070–1077.** Explanations are judged better when they support the
  inferences the task requires — adequacy is *instrumental*, not intrinsic.
- **Zemla, Sloman, Bechlivanidis & Lagnado (2017), *Evaluating everyday
  explanations*, Psychonomic Bulletin & Review 24(5):1488–1500.** Concrete
  explanations are evaluated differently from abstract ones; evaluation is
  sensitive to the abstraction level.
- **Lombrozo & Liquin (2023), *Explanation is effective because it is
  selective*, Current Directions in Psychological Science 32(3).** The
  mechanism: explanation works by *selecting* the right information, not by
  adding volume. A 2-hop explanation that is on-mechanism beats a 5-hop
  explanation that is off-mechanism.
- **Pekrun, Goetz, Titz & Perry (2002), *Academic emotions in students'
  self-regulated learning and achievement*, Educational Psychologist 37(2):91–105.**
  Epistemic emotions (curiosity, confusion, surprise, frustration) are triggered
  by cognitive discrepancy — the *feeling* of a gap is an affective signal that
  learning should continue. RAVANA already has a VAD/emotion module; the gap
  signal should feed it.

**Design consequence — this is the core of the deliverable.** The brain's
adequacy signal is *not* "did I retrieve something". It is "does the produced
chain actually cover the explanandum with typed, directional, sourced links, and
how many *unexplained gaps* remain". Rozenblit & Keil's result implies a
specific, falsifiable implementation: **confidence must be computed *after*
chain construction, from chain structure, and must drop when the chain is
short, untyped, or unsourced.** RAVANA's current `metacognition.fok_confidence`
(`chat/metacognition.py`) is a saturating function of *support count* — it
cannot express "this 2-hop chain is adequate" vs. "this 5-hop chain is a
ramble". It needs a chain-level companion.

---

## 4. Explanation vs. description in the brain, and whether explaining changes learning

### 4.1 The dissociation

- **Monti, Parsons & Osherson (2012), *Thought beyond language: neural
  dissociation of algebra and natural language*, Psychological Science
  23(8):914–921** and **Monti, Parsons & Osherson (2012), *Language and thought
  are not the same thing*, in Forbus, Gentner & Regier (eds.)** — reasoning
  (algebra) and language comprehension recruit partially distinct systems.
- **Fedorenko & Duncan (2019)** / **Fedorenko et al. (2024)** — the language
  network is a natural kind, distinct from MD and DMN.
- **Hagoort (2005), *On Broca, brain, and binding: a new framework*, Trends in
  Cognitive Sciences 9(9):416–423** and **Hagoort (2019), *The neurobiology of
  language beyond single-word processing*, Science 366(6461):55–58.** The MUC
  (Memory–Unification–Control) framework: LIFG does *unification* — binding
  retrieved elements into a structured whole. **Unification is the bridge
  between a causal structure and a sentence.** RAVANA's `PrefrontalWorkspace`
  docstring already cites MUC.
- **Mason & Just (2004), *How the brain processes causal inferences in text: a
  theoretical account of generation and integration component processes
  utilizing both cerebral hemispheres*, Psychological Science 15(1):1–7.**
  Causal inference in text has *two* components: **generation** (producing
  candidate inferences) and **integration** (fitting them into the evolving
  representation). A description needs only retrieval; an explanation needs
  both.
- **Kuperberg, Lakshmanan, Caplan & Holcomb (2006), *Making sense of discourse:
  an fMRI study of causal inferencing across sentences*, NeuroImage
  33(1):343–361.** Causal inferencing across sentences recruits a bilateral
  network for unstated information — the brain *builds* the missing links.
- **Trabasso & van den Broek (1985), *Causal thinking and the representation of
  narrative events*, Journal of Memory and Language 24(5):612–630.** The
  causal-network representation of narrative: events are represented by their
  causal relations, and causal connectedness predicts importance and
  memorability. This is the classic basis for "causal structure is what makes a
  chain explanatory rather than a list".
- **Graesser, Singer & Trabasso (1994), *Constructing inferences during
  narrative text comprehension*, Psychological Review 101(3):371–395.**
  Constructionist theory: readers generate causal antecedent and consequence
  inferences to maintain coherence.

**Design consequence.** *Description* = retrieve a stored predicate for the
subject (`battery is a device that stores energy`). *Explanation* = a directed
path with at least one intermediate node that the subject does not already
name. RAVANA's `_causal_chain_from_graph` (`response_gen.py:3103`) already
implements a 2-clause description-shaped form (`X causes S; S leads to Y`) but
requires only ≥2 causal edges *anywhere* around the subject — it does not
require that the clauses form a *connected path*, and it is called only from one
site (`response_gen.py:8745`). This is the closest existing primitive and the
right place to build.

### 4.2 The self-explanation effect — and its limits

- **Chi, Bassok, Lewis, Reimann & Glaser (1989), *Self-explanations: how
  students study and use examples in learning to solve problems*, Cognitive
  Science 13(2):145–182.** The seminal study: students who spontaneously
  self-explained worked examples learned more, and their self-explanations were
  dominated by *inferences* about principles, not restatements.
- **Chi (2000), *Self-explaining expository texts: the dual processes of
  generating inferences and repairing mental models*, in Glaser (ed.),
  Advances in Instructional Psychology 5:161–238.** The dual-process account:
  self-explanation = (a) generate inferences, (b) detect and repair mismatches
  with the existing mental model. **This is the mechanism RAVANA should
  implement: explaining is a *learning* act because it surfaces the gap.**
- **VanLehn, Jones & Chi (1992), *A model of the self-explanation effect*,
  Journal of the Learning Sciences 2(1):1–59.** The Cascade model: learners
  self-explain at *impasses* — when a step does not follow from what they know.
  Self-explanation is triggered by a *comprehension failure*, not by a timer.
- **Ainsworth & Loizou (2003), *The effects of self-explaining when learning
  with text or diagrams*, Cognitive Science 27(4):669–681.** Self-explanation
  helps more when the material supports inference generation.
- **Roy & Chi (2005), *The self-explanation principle in multimedia learning*,
  in Mayer (ed.), Cambridge Handbook of Multimedia Learning.** Self-explanation
  promotes *deep* processing measurable in transfer.
- **Bisra, Liu, Nesbit, Salimi & Winne (2018), *Inducing self-explanation: a
  meta-analysis*, Educational Psychology Review 30(3):703–725.** 64 studies,
  ~6000 participants: overall **g = 0.55** (95% CI 0.45–0.65) in favour of
  prompted self-explanation. A real, moderate effect.
- **Counter-evidence (important):** *No Causal Self-Explanation Effect for
  Factual Knowledge*, Applied Cognitive Psychology (2025) — across immediate and
  delayed tests, self-explanation produced **no** benefit over re-reading for
  factual recall, though the *number of correct inferences* generated during
  self-explanation predicted performance. And **Bisra et al. (2018)** report
  heterogeneity: the effect is reliable for *problem-solving/transfer* outcomes
  and much weaker for *factual* outcomes. See §6.
- **Developmental evidence:** **Legare & Lombrozo (2014), *Selective effects of
  explanation on learning during infancy*, Cognition 131(2):256–266** —
  explanation promotes causal learning and generalization but *not* memory for
  causally irrelevant perceptual detail. **Walker, Lombrozo & Williams (2017),
  *Explaining constrains causal learning in childhood*, Child Development
  88(1):229–246** — explaining narrows hypothesis space. **Williams &
  Lombrozo (2010), *The role of explanation in discovery and generalization:
  evidence from category learning*, Cognitive Science 34(5):776–806** —
  explanation recruits comparison and helps discovery of broad patterns.
- **Explanation as exploration constraint:** **Gweon, Tenenbaum & Schulz
  (2010)** and **Bonawitz, Shafto, Gweon, Goodman, Spelke & Schulz (2011),
  *The double-edged sword of pedagogy: instruction and pedagogy both limit
  spontaneous exploration and discovery*, Cognition 120(3):322–330** —
  explanation and exploration work in tandem; explaining constrains future
  exploration to the relevant region.

**Design consequence.** Explaining should be *wired to learning*, not just to
answering. Concretely: when the adequacy gate rejects a chain (gap detected),
that rejection should (a) trigger a targeted web read on the missing link, and
(b) record the attempted chain as an *unresolved gap* in the belief store —
exactly Chi's "repair the mental model" step and VanLehn's impasse-driven
learning. RAVANA already has the machinery: `web_learning.queue_background_search`,
`KnowledgeGap` in `web/web_to_graph.py`, and `self_directed_learning.py`.

---

## 5. Recommended mechanism for RAVANA

### 5.1 What exists today (verified by reading the code)

| Primitive | Location | What it does | Gap |
|---|---|---|---|
| `_causal_forward_simulate(start, max_steps=4, top_k=3)` | `chat/chain_walker.py:2193` | BFS/A* over `relation_type == "causal"` edges from a start node; returns up to 3 paths as `"a → b → c"` strings | No adequacy check; returns *all* reachable paths up to a cost bound; no target; no provenance filter |
| `_causal_chain_search(start, end, max_steps=5)` | `chat/chain_walker.py:2153` | A* from A to B with a GloVe heuristic | **Dead code** — requires a pre-specified end node, so it bails for open-ended "why" questions |
| `_causal_chain_from_graph(target)` | `chat/response_gen.py:3103` | Reads causal in/out neighbours of `target`; emits `"X causes S; S leads to Y."` if ≥2 causal edges | Does not require a *connected path*; single-hop neighbours suffice; called from one site |
| `_causal_restructure(text, rel)` | `chat/response_gen.py:3070` | Splits a retrieved snippet on a causal connective and reorders the clauses | Works on retrieved text, not on graph structure |
| `_seed_physics_causal()` | `chat/engine_reasoning.py:3185` | Seeds 24 authored causal edges (sun→light→photosynthesis→plants→animals; tree→fall→vibrate→air→sound) | Hand-authored; not learned |
| `_extract_and_store_causal_relations(text)` | `chat/chain_walker.py:749` | Splits a user utterance on a causal-connective word, creates/strengthens `causal` edges between content words on each side | **Skips all questions** (line ~783: `if sentence.endswith("?") ... continue`) — so a user asking "why does it rain" teaches nothing |
| `_create_causal_edge` | `chat/chain_walker.py:847` | Creates a `causal` edge with `weight=0.75, confidence=0.7`; boosts existing | No provenance tag; no convergence counting |
| `metacognition.fok_confidence` / `should_assert` | `chat/metacognition.py:45,64` | Saturating confidence from support count + retrieval flag; assert/hedge gate at θ=0.30 | Support-count only; no chain structure |
| `provenance.admit_as_fact` / `provenance_adjusted_confidence` | `chat/provenance.py:117,126` | Requires verifiable provenance or ≥2 converging sources; caps unprovenanced confidence | Not applied to chains |
| `_causal_narrative_degenerate(text)` | `chat/response_gen.py:9221` | Rejects placeholder/tautological/junk-endpoint causal narratives | Post-hoc string filter; not a structural check |
| `PrefrontalWorkspace.detect_question_type` | `language/prefrontal_workspace.py:1031` | Labels `why`/`how`; `QTYPE_PRIMARY_RELATION` maps both to `"causal"` | Correct routing, unused by the causal path |
| `_plan_causal_explain` | `language/prefrontal_workspace.py:1513` | Plans 3 `CAUSAL_EXPLAIN` intents | Picks targets by `_has_causal_hint` — a **substring match on the label** (`"cause" in label`), not the edge's relation type |
| `WebToGraph` | `web/web_to_graph.py:49` | Maps OpenIE relations to edge types; `"causes" → "causes"` | `RELATION_TO_EDGE` has no `"causal"` key; writes `causes` which is not in `ConceptEdge`'s relation-type table |
| `CausalSchemaLearner` | `core/causal_schema.py:54` | Learns `(state_a, condition, state_b)` triples; innate priors | Not connected to the explanation path |

### 5.2 The proposed pipeline

**Stage 1 — Route.** Reuse `detect_question_type`. `why` and `how` →
`causal_explain`. Add a guard: a `causal_explain` route must not be satisfied by a
`semantic`-only answer. (Today `_is_conditional_query` at
`engine_reasoning.py:2708` already excludes definitional/explanation shapes from
the counterfactual simulator — the same discipline is needed in the other
direction.)

**Stage 2 — Assemble.** New method `_explanation_chain(subject, target=None,
min_steps=2, max_steps=4)`:

1. Resolve `subject` to node ids via `_concept_keywords`.
2. Build a **snapshot** of the causal subgraph reachable within `max_steps`
   (copy the pattern at `chain_walker.py:2215-2242` — the B5 stability fix:
   simulate on a frozen snapshot, not the live graph).
3. Enumerate simple paths of length `min_steps..max_steps` using **only**
   `relation_type == "causal"` edges. Prefer:
   - **Backward** (cause→…→subject) when the query is `why <subject>`;
   - **Forward** (subject→…→effect) when the query is `how does <subject> work`
     or `what does <subject> do`;
   - **Bidirectional** (cause→…→subject→…→effect) when both directions have
     support and the query is `explain <subject>`.
4. Score each candidate path:
   `score = Σ_hop [ log(w_hop · c_hop) ] − λ·len(path) − μ·(unprovenanced_hops)`
   where `w_hop`/`c_hop` are the edge weight/confidence and `λ` is a simplicity
   penalty (Lombrozo 2007; Lombrozo & Liquin 2023 — explanation is effective
   because it is *selective*).
5. Return the top path as a structured `ExplanationChain` (list of
   `(from_label, relation_type, to_label, weight, confidence, provenance_class)`
   hops) plus the runner-up, so the monitor can compute a *margin*.

**Stage 3 — Adequacy gate.** New `ExplanationMonitor` (a reader, per Nelson &
Narens 1990 — separate from the builder). Inputs: the chain, the query type, the
subject. Computes:

```
coverage      = 1 if the chain's terminal node is the explanandum (or a named
                effect of it), else 0
typed_fraction= (# hops with relation_type == "causal") / len(hops)
provenance    = min over hops of provenance_adjusted_confidence(hop, converging)
gap_count     = # of adjacent hop pairs where the link is missing or inverted
margin        = score(best) - score(runner_up)
adequacy      = w1*coverage + w2*typed_fraction + w3*provenance
                - w4*gap_count + w5*min(1, margin)
```

Verdicts:
- `adequacy ≥ θ_assert` → assert the chain flatly.
- `θ_hedge ≤ adequacy < θ_assert` → assert with `hedges.modality_from_support`
  (already maps to `likely`/`possible`/`unknown`).
- `adequacy < θ_hedge` → **do not explain**. Emit an honest gap statement and
  trigger Stage 5 learning. This is the Rozenblit & Keil behaviour: the system
  should feel the gap when asked to produce the mechanism.

Calibration: `θ_assert`, `θ_hedge`, and the `w_i` are **fit, not frozen** — reuse
the existing `calibrator.theta_withhold()` ledger pattern from
`response_gen.py:3914-3919` so an overconfident engine raises its own bar.

**Stage 4 — Realize.** Consume the `ExplanationChain` object, not a string:
- Linearize with `_RELATION_PREDICATE["causal"]` = `"leads to"`
  (`response_gen.py:6886`) — or better, a per-hop connective chosen from the
  hop's own relation type.
- Attach the modality from Stage 3 via `hedges.modality_from_support`.
- Keep it to 2–4 clauses. Lombrozo & Liquin (2023): selectivity beats volume.
- Route through the existing `DiscourseType.CAUSAL_EXPLAIN` intent and the
  nesting pass in `engine_generation.py:319-360` so a 2-hop chain becomes a
  single nested sentence (`S leads to B, because A causes S`) rather than two
  unrelated sentences.

**Stage 5 — Learn online.**
- *From conversation:* fix `_extract_and_store_causal_relations`
  (`chain_walker.py:749`) to **stop skipping questions**. A user asking
  `why does it rain` is *revealing* that `rain` has an unexplained cause in the
  graph — record the gap. A user *asserting* `rain happens because water
  condenses` should create/strengthen `condense → rain` with a provenance tag
  (`user_assertion`) and a convergence counter.
- *From web:* `WebToGraph.RELATION_TO_EDGE` (`web/web_to_graph.py:57`) maps
  `"causes" → "causes"`, but `ConceptEdge`'s relation-type table
  (`ravana_ml/src/ravana_ml/graph.py:168-181`) has no `"causes"` key — it has
  `"causal"`. **This is a live bug for online causal learning**: web-extracted
  causal facts are written under a relation type the rest of the system does not
  recognise. Add `"causes" → "causal"` (and `"caused_by" → "causal"` with
  reversed endpoints).
- *Gate the learning:* Dorfman, Tomov, Cheung, Clarke, Gershman & Hughes
  (2021), *Causal inference gates corticostriatal learning*, Journal of
  Neuroscience 41(32):6892–6904 — beliefs about causal structure modulate
  prediction-error-driven learning, and those beliefs are represented in
  **anterior insula and inferior frontal gyrus**. The architectural translation:
  a new causal edge should only *strengthen* (not be created at full weight)
  when it is consistent with the existing local causal structure; a structurally
  surprising edge gets a low prior and must earn its weight through convergence.
  RAVANA's `_create_causal_edge` currently sets `weight=0.75, confidence=0.7`
  on first sight — too high for an unprovenanced single mention.

### 5.3 Worked examples (what the mechanism should produce)

`why do you think it rains`
- Route: `why` → `causal_explain`.
- Assemble: backward search from `rain`. If the graph has
  `evaporate → condense → rain` (learned), return it. If not, return `[]`.
- Gate: `[]` → `adequacy < θ_hedge` → **withhold**, say the gap, queue a web
  read on `rain cause`. *Not* `i'm drawn to rains`.
- After learning: `rain forms because water evaporates, condenses into droplets,
  and falls when the droplets grow heavy` — 3 hops, all `causal`, all sourced.

`explain how a battery works`
- Route: `how` → `causal_explain`, forward direction.
- Assemble: forward from `battery`. If the graph has
  `battery → chemical_reaction → electron_flow → current`, return it.
- Gate: if only `battery —semantic— plasma` exists (today's behaviour),
  `typed_fraction = 0` → `adequacy < θ_hedge` → withhold and say
  `i don't have a clean causal chain for battery yet` — which is *honest*, and
  is what the current system already half-says, but now for the right reason and
  with a learning trigger attached.

---

## 6. Falsifiability

Each claim below is testable against RAVANA's own behaviour. A claim that
survives is evidence for the mechanism; a claim that fails is evidence against
it and the mechanism should be revised.

**F1 — Routing claim.** *A `why`/`how` query answered from `semantic` edges only
is the defect.* **Falsified if:** adding the causal-substrate requirement makes
no measurable difference on a held-out set of `why`/`how` queries (i.e. the
graph's `causal` edges are no better than its `semantic` edges for these
queries). *Measure:* blind A/B on 50 `why`/`how` queries, rated for "names a
mechanism" by a rubric that does not know which system produced the answer.

**F2 — Adequacy claim.** *Chain-structure confidence predicts explanation
quality better than support-count confidence.* **Falsified if:** on a labelled
set of (chain, quality) pairs, `ExplanationMonitor.adequacy` does not beat
`metacognition.fok_confidence` on AUC or calibration (ECE). *Measure:* fit both
on 70% of the data, report held-out AUC/ECE. This is the single most important
test — if it fails, the whole §5.3 gate is decorative.

**F3 — Gap claim.** *Withholding on a short/untyped chain is perceived as more
honest than asserting an associative answer.* **Falsified if:** human raters
prefer the associative answer, or rate the withheld answer as less helpful with
no honesty benefit. *Measure:* forced-choice plus 7-point honesty/helpfulness
scales.

**F4 — Learning claim.** *Online extraction of causal links from conversation and
web improves subsequent explanation adequacy.* **Falsified if:** after N turns of
conversation-driven causal edge creation, held-out `why`/`how` adequacy does not
improve, or improves no more than a control that adds `semantic` edges at the
same rate. *Measure:* longitudinal held-out adequacy, with the causal-edge
extractor ablated.

**F5 — Self-explanation claim.** *Triggering a targeted web read on a detected
gap improves later answers more than answering from the current graph.*
**Falsified if:** gap-triggered reading produces no delayed improvement, or the
improvement is explained by sheer exposure (control: read on a random topic).
*This is the claim most at risk given the counter-evidence in §6.1.*

**F6 — Simplicity claim.** *Among adequate chains, the shorter one is preferred
and is rated as better.* **Falsified if:** raters prefer longer chains, or
length has no effect once adequacy is controlled. *Note:* Lombrozo (2007) and
Liquin & Lombrozo (2022) support this, but Vasilyeva et al. (2017) show
preferences are instrumentally malleable — the effect may be weak.

**F7 — Directionality claim.** *Backward chains answer `why` and forward chains
answer `how`; a chain in the wrong direction is judged non-explanatory.*
**Falsified if:** direction has no effect on ratings. *Note:* Satpute et al.
(2005) provide the neural basis for direction sensitivity (role binding), so
this should hold; a failure would suggest the rating rubric is not measuring
explanation.

### 6.1 Explicit unsettled evidence

State these plainly; they are reasons not to over-commit.

1. **The self-explanation effect is not robust for factual knowledge.**
   Bisra et al. (2018) find g = 0.55 overall but with substantial heterogeneity,
   and *No Causal Self-Explanation Effect for Factual Knowledge* (Applied
   Cognitive Psychology, 2025) finds **no** benefit of self-explanation over
   re-reading for factual recall at immediate or delayed test — only the
   *number of correct inferences* generated predicted performance. The effect is
   reliable for problem-solving and transfer, weak or absent for facts. RAVANA's
   `why`/`how` queries are often factual, so the "explaining drives learning"
   premise is **not** safely generalisable. Design consequence: implement
   gap-triggered learning, but *measure* it (F5) rather than assuming it.
2. **Causal-model vs. associative accounts are not settled.** Holyoak & Cheng
   (2011) call their view a "new synthesis" precisely because pure causal-model
   and pure associative accounts both fail. Griffiths & Tenenbaum (2009) is a
   rational model, not a process model. RAVANA should not assume the graph's
   `causal` edges are *correct* — only that they are *typed*.
3. **The RLPFC/relational-integration result has a laterality dispute.** Smith
   et al. (2013) find only *left* RLPFC survives stringent tests; other studies
   report bilateral. Do not build a claim on "RLPFC does X" alone.
4. **The MD network's necessity is contested.** Woolgar et al. (2018) show MD
   supports fluid intelligence and the language network does not, but lesion
   evidence (Woolgar et al. 2010) is correlational with respect to causality;
   Fedorenko et al. (2024) argue the language network is a natural kind while
   disputing how segregated it is. Treat "MD = reasoning, language network =
   wording" as a useful design heuristic, not established fact.
5. **Metacognitive confidence is not a single system.** Fleming (2024) reviews
   dissociations between metacognitive bias, sensitivity, and efficiency, and
   between perceptual and memory metacognition (Baird et al. 2013; McCurdy et
   al. 2013, cited in Fleming 2024). A single scalar `adequacy` is a
   simplification. It is the right simplification for a 20 W system, but it
   should be documented as one.
6. **The illusion-of-explanatory-depth effect may be task-specific.**
   Rozenblit & Keil (2002) used mechanical devices. Whether the same collapse
   occurs for *social* and *biological* explanations (RAVANA's typical topics)
   is not established. RAVANA's `why do people lie` may not behave like
   `how does a zipper work`.
7. **Explanation adequacy is instrumentally malleable.** Vasilyeva et al.
   (2017) show perceived explanation quality shifts with the task's required
   inferences. There may be no context-free adequacy function to implement.
   Design consequence: make the adequacy weights `w_i` context-dependent (query
   type, user model) and fit them, rather than fixing them.
8. **Online causal learning from text is an open problem.** RAVANA's
   `_extract_and_store_causal_relations` splits on a causal-connective word and
   links *all* content words on each side — a precision disaster for complex
   sentences. The literature (Meder et al. 2014; Goodman et al. 2011) uses
   structured hypothesis spaces, not connective matching. Expect low precision
   and plan for convergence-based filtering.

---

## 7. References

**Causal reasoning and explanation — neuroscience**

- Satpute, A. B., Fenker, D. B., Waldmann, M. R., Tabibnia, G., Holyoak, K. J., &
  Lieberman, M. D. (2005). An fMRI study of causal judgments. *European Journal
  of Neuroscience, 22*(5), 1233–1238.
- Feng, W., Wang, W., Liu, J., Wang, Z., Tian, L., & Fan, L. (2021). Neural
  correlates of causal inferences in discourse understanding and logical
  problem-solving: a meta-analysis study. *Frontiers in Human Neuroscience, 15*,
  666179.
- Mason, R. A., & Just, M. A. (2004). How the brain processes causal inferences
  in text: a theoretical account of generation and integration component
  processes utilizing both cerebral hemispheres. *Psychological Science, 15*(1),
  1–7.
- Kuperberg, G. R., Lakshmanan, B. M., Caplan, D. N., & Holcomb, P. J. (2006).
  Making sense of discourse: an fMRI study of causal inferencing across
  sentences. *NeuroImage, 33*(1), 343–361.
- Operskalski, J. T., & Barbey, A. K. (2017). Cognitive neuroscience of causal
  reasoning. In M. R. Waldmann (Ed.), *The Oxford Handbook of Causal Reasoning*
  (pp. 217–242). Oxford University Press.
- Dorfman, H. M., Tomov, M. S., Cheung, B., Clarke, D., Gershman, S. J., &
  Hughes, B. L. (2021). Causal inference gates corticostriatal learning.
  *Journal of Neuroscience, 41*(32), 6892–6904.
- Lee, S. W., O'Doherty, J. P., & Shimojo, S. (2015). Neural computations
  mediating one-shot learning in the human brain. *PLoS Biology, 13*(4), e1002137.
- Boorman, E. D., Behrens, T. E. J., Woolrich, M. W., & Rushworth, M. F. S.
  (2009). How green is the grass on the other side? Frontopolar cortex and the
  evidence in favor of alternative courses of action. *Neuron, 62*(5), 733–743.

**Relational reasoning, RLPFC, and the multiple-demand network**

- Christoff, K., Prabhakaran, V., Dorfman, J., Zhao, Z., Kroger, J. K., Holyoak,
  K. J., & Gabrieli, J. D. E. (2001). Rostrolateral prefrontal cortex involvement
  in relational integration during reasoning. *NeuroImage, 14*(5), 1136–1149.
- Kroger, J. K., Sabb, F. W., Fales, C. L., Bookheimer, S. Y., Cohen, M. S., &
  Holyoak, K. J. (2002). Recruitment of anterior dorsolateral prefrontal cortex
  in human reasoning: a parametric study of relational complexity. *Cerebral
  Cortex, 12*(5), 477–485.
- Smith, R., Keramatian, K., & Christoff, K. (2007). Localizing the rostrolateral
  prefrontal cortex at the individual level. *NeuroImage, 36*(4), 1387–1396.
- Smith, R., Keramatian, K., Smallwood, J., & Christoff, K. (2013). Left, but not
  right, rostrolateral prefrontal cortex meets a stringent test of the relational
  integration hypothesis. *Neuropsychologia, 51*(13), 2724–2733.
- Waltz, J. A., Knowlton, B. J., Holyoak, K. J., Boone, K. B., Mishkin, F. S., de
  Menezes Santos, M., Thomas, C. R., & Miller, B. L. (1999). A system for
  relational reasoning in human prefrontal cortex. *Psychological Science,
  10*(2), 119–125.
- Crone, E. A., Wendelken, C., van Leijenhorst, L., Honomichl, R. D., Christoff,
  K., & Bunge, S. A. (2009). Neurocognitive development of relational reasoning.
  *Developmental Science, 12*(1), 55–66.
- Badre, D., & D'Esposito, M. (2009). Is the rostro-caudal axis of the frontal
  lobe hierarchical? *Nature Reviews Neuroscience, 10*(9), 659–669.
- Christoff, K., Keramatian, K., Gordon, A. M., Smith, R., Mädler, B., Evans, K.,
  Marois, R., & Hester, R. (2009). Prefrontal organization of cognitive control
  according to levels of abstraction. *Brain Research, 1286*, 94–105.
- Koechlin, E., & Summerfield, C. (2007). An information theoretical approach to
  prefrontal executive function. *Trends in Cognitive Sciences, 11*(6), 229–235.
- Koechlin, E. (2014). An evolutionary computational theory of prefrontal
  executive function in decision-making. *Philosophical Transactions of the Royal
  Society B, 369*(1655), 20130474.
- Duncan, J. (2010). The multiple-demand (MD) system of the primate brain: mental
  programs for intelligent behaviour. *Trends in Cognitive Sciences, 14*(4),
  172–179.
- Duncan, J. (2013). The structure of cognition: attentional episodes in mind and
  brain. *Neuron, 80*(1), 35–50.
- Woolgar, A., Parr, A., Cusack, R., Thompson, R., Nimmo-Smith, I., Torralva, T.,
  Roca, M., Antoun, N., Manes, F., & Duncan, J. (2010). Fluid intelligence loss
  linked to restricted regions of damage within frontal and parietal cortex.
  *PNAS, 107*(33), 14899–14902.
- Woolgar, A., Duncan, J., et al. (2018). The multiple-demand system but not the
  language system supports fluid intelligence. *Nature Human Behaviour, 2*(3),
  200–204.
- Assem, M., Glasser, M. F., Van Essen, D. C., & Duncan, J. (2020). A
  domain-general cognitive core defined in multimodally parcellated human cortex.
  *Cerebral Cortex, 30*(8), 4361–4380.
- Fedorenko, E., Ivanova, A. A., & Regev, T. I. (2024). The language network as a
  natural kind within the broader landscape of the human brain. *Nature Reviews
  Neuroscience*.

**Metacognition, adequacy, and the explanatory gap**

- Nelson, T. O., & Narens, L. (1990). Metamemory: a theoretical framework and new
  findings. In G. Bower (Ed.), *The Psychology of Learning and Motivation*
  (Vol. 26, pp. 125–173). Academic Press.
- Hart, J. T. (1965). Memory and the feeling-of-knowing experience. *Journal of
  Educational Psychology, 56*(4), 208–216.
- Koriat, A. (1993). How do we know that we know? The accessibility model of the
  feeling of knowing. *Psychological Review, 100*(4), 609–639.
- Koriat, A. (2000). The feeling of knowing: some metatheoretical implications
  for consciousness and control. *Consciousness and Cognition, 9*(2), 149–171.
- Fleming, S. M., & Dolan, R. J. (2012). The neural basis of metacognitive
  ability. *Philosophical Transactions of the Royal Society B, 367*(1594),
  1338–1349.
- Fleming, S. M., Huijgen, J., & Dolan, R. J. (2012). Prefrontal contributions to
  metacognition in perceptual decision making. *Journal of Neuroscience, 32*(18),
  6117–6125.
- Vaccaro, A. G., & Fleming, S. M. (2018). Thinking about thinking: a
  coordinate-based meta-analysis of neuroimaging studies of metacognitive
  judgements. *Brain and Neuroscience Advances, 2*.
- Bang, D., & Fleming, S. M. (2018). Distinct encoding of decision confidence in
  human medial prefrontal cortex. *PNAS, 115*(23), 6082–6087.
- Morales, J., Lau, H., & Fleming, S. M. (2018). Domain-general and
  domain-specific patterns of activity supporting metacognition in human
  prefrontal cortex. *Journal of Neuroscience, 38*(14), 3534–3546.
- Fleming, S. M. (2024). Metacognition and confidence: a review and synthesis.
  *Annual Review of Psychology, 75*, 241–268.
- Ackerman, R., & Thompson, V. A. (2017). Meta-reasoning: monitoring and control
  of thinking and reasoning. *Trends in Cognitive Sciences, 21*(8), 607–617.
- Rozenblit, L., & Keil, F. (2002). The misunderstood limits of folk science: an
  illusion of explanatory depth. *Cognitive Science, 26*(5), 521–562.
- Keil, F. C. (2006). Explanation and understanding. *Annual Review of Psychology,
  57*, 227–254.
- Lombrozo, T. (2006). The structure and function of explanations. *Trends in
  Cognitive Sciences, 10*(10), 464–470.
- Lombrozo, T. (2007). Simplicity and probability in causal explanation.
  *Cognitive Psychology, 55*(3), 232–257.
- Vrantsidis, T. H., & Lombrozo, T. (2022). Simplicity as a cue to probability:
  multiple roles for simplicity in evaluating explanations. *Cognitive Science,
  46*(8), e13169.
- Vasilyeva, N., Wilkenfeld, D., & Lombrozo, T. (2017). Contextual utility
  affects the perceived quality of explanations. *Psychonomic Bulletin & Review,
  24*(4), 1070–1077.
- Zemla, J. C., Sloman, S., Bechlivanidis, C., & Lagnado, D. A. (2017). Evaluating
  everyday explanations. *Psychonomic Bulletin & Review, 24*(5), 1488–1500.
- Lombrozo, T., & Liquin, E. G. (2023). Explanation is effective because it is
  selective. *Current Directions in Psychological Science, 32*(3).
- Pekrun, R., Goetz, T., Titz, W., & Perry, R. P. (2002). Academic emotions in
  students' self-regulated learning and achievement. *Educational Psychologist,
  37*(2), 91–105.

**Causal-model tradition and causal learning**

- Sloman, S. A. (2005). *Causal Models: How People Think about the World and Its
  Alternatives*. Oxford University Press.
- Sloman, S. A., & Lagnado, D. A. (2005). Do we "do"? *Cognitive Science, 29*(1),
  5–39.
- Sloman, S. A., & Lagnado, D. A. (2015). Causality in thought. *Annual Review of
  Psychology, 66*, 223–247.
- Lagnado, D. A. (2011). Causal thinking. In P. M. Illari, F. Russo, & J.
  Williamson (Eds.), *Causality in the Sciences* (pp. 125–149). Oxford University
  Press.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.).
  Cambridge University Press.
- Glymour, C. (2001). *The Mind's Arrows: Bayes Nets and Graphical Causal Models
  in Psychology*. MIT Press.
- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and
  Search* (2nd ed.). MIT Press.
- Cheng, P. W. (1997). From covariation to causation: a causal power theory.
  *Psychological Review, 104*(2), 367–405.
- Cheng, P. W., & Buehner, M. J. (2012). Causal learning. In K. J. Holyoak & R. G.
  Morrison (Eds.), *The Oxford Handbook of Thinking and Reasoning* (pp.
  210–233). Oxford University Press.
- Holyoak, K. J., & Cheng, P. W. (2011). Causal learning and inference as a
  rational process: the new synthesis. *Annual Review of Psychology, 62*,
  135–163.
- Tenenbaum, J. B., Kemp, C., & Griffiths, T. L. (2011). How to grow a mind:
  statistics, structure, and abstraction. *Science, 331*(6022), 1279–1285.
- Tenenbaum, J. B., Griffiths, T. L., & Kemp, C. (2006). Theory-based Bayesian
  models of inductive learning and reasoning. *Trends in Cognitive Sciences,
  10*(7), 309–318.
- Griffiths, T. L., & Tenenbaum, J. B. (2005). Structure and strength in causal
  induction. *Cognitive Science, 29*(4), 649–675.
- Griffiths, T. L., & Tenenbaum, J. B. (2009). Theory-based causal induction.
  *Psychological Review, 116*(4), 661–716.
- Goodman, N. D., Ullman, T. D., & Tenenbaum, J. B. (2011). Learning a theory of
  causality. *Psychological Review, 118*(1), 110–119.
- Kemp, C., & Tenenbaum, J. B. (2008). The discovery of structural form. *PNAS,
  105*(31), 10687–10692.
- Kemp, C., Goodman, N. D., & Tenenbaum, J. B. (2010). Learning to learn causal
  models. *Cognitive Science, 34*(7), 1185–1243.
- Meder, B., Hagmayer, Y., & Waldmann, M. R. (2009). The role of learning data in
  causal reasoning about observations and interventions. *Memory & Cognition,
  37*(3), 249–264.
- Meder, B., Mayrhofer, R., & Waldmann, M. R. (2014). Structure induction in
  diagnostic causal reasoning. *Psychological Review, 121*(3), 277–301.
- Hagmayer, Y., & Mayrhofer, R. (2013). Hierarchical Bayesian models as formal
  models of causal reasoning. *Argument & Computation, 4*(1), 36–45.
- Waldmann, M. R., & Hagmayer, Y. (2005). Seeing versus doing: two modes of
  accessing causal knowledge. *Journal of Experimental Psychology: Learning,
  Memory, and Cognition, 31*(2), 216–227.
- Gerstenberg, T. (2024). Counterfactual simulation in causal cognition. *Trends
  in Cognitive Sciences, 28*(10), 924–936.
- Byrne, R. M. J. (2005). *The Rational Imagination: How People Create
  Alternatives to Reality*. MIT Press.
- Gopnik, A., & Wellman, H. M. (2012). Reconstructing constructivism: causal
  models, Bayesian learning mechanisms, and the theory theory. *Psychological
  Bulletin, 138*(6), 1085–1108.
- Schulz, L. E., Gopnik, A., & Glymour, C. (2007). Preschool children learn about
  causal structure from conditional interventions. *Developmental Science,
  10*(3), 322–332.
- Kushnir, T., & Gopnik, A. (2005). Young children infer causal strength from
  probabilities and interventions. *Psychological Science, 16*(9), 678–683.
- Sobel, D. M., & Legare, C. H. (2014). Causal learning in children. *Wiley
  Interdisciplinary Reviews: Cognitive Science, 5*(4), 413–427.
- Bramley, N. R., Gerstenberg, T., Mayrhofer, R., Lagnado, D. A., & Tenenbaum,
  J. B. (2018). Time in causal structure learning. *Cognitive Psychology, 101*,
  54–82.

**Explanation and learning**

- Chi, M. T. H., Bassok, M., Lewis, M. W., Reimann, P., & Glaser, R. (1989).
  Self-explanations: how students study and use examples in learning to solve
  problems. *Cognitive Science, 13*(2), 145–182.
- Chi, M. T. H. (2000). Self-explaining expository texts: the dual processes of
  generating inferences and repairing mental models. In R. Glaser (Ed.),
  *Advances in Instructional Psychology* (Vol. 5, pp. 161–238). Lawrence Erlbaum.
- VanLehn, K., Jones, R. M., & Chi, M. T. H. (1992). A model of the
  self-explanation effect. *Journal of the Learning Sciences, 2*(1), 1–59.
- Ainsworth, S., & Loizou, A. T. (2003). The effects of self-explaining when
  learning with text or diagrams. *Cognitive Science, 27*(4), 669–681.
- Roy, M., & Chi, M. T. H. (2005). The self-explanation principle in multimedia
  learning. In R. E. Mayer (Ed.), *Cambridge Handbook of Multimedia Learning*
  (pp. 287–298). Cambridge University Press.
- Bisra, K., Liu, Q., Nesbit, J. C., Salimi, F., & Winne, P. H. (2018). Inducing
  self-explanation: a meta-analysis. *Educational Psychology Review, 30*(3),
  703–725.
- *No causal self-explanation effect for factual knowledge.* (2025). *Applied
  Cognitive Psychology.*
- Legare, C. H., & Lombrozo, T. (2014). Selective effects of explanation on
  learning during infancy. *Cognition, 131*(2), 256–266.
- Walker, C. M., Lombrozo, T., & Williams, J. J. (2017). Explaining constrains
  causal learning in childhood. *Child Development, 88*(1), 229–246.
- Williams, J. J., & Lombrozo, T. (2010). The role of explanation in discovery and
  generalization: evidence from category learning. *Cognitive Science, 34*(5),
  776–806.
- Bonawitz, E., Shafto, P., Gweon, H., Goodman, N. D., Spelke, E., & Schulz, L.
  (2011). The double-edged sword of pedagogy: instruction and pedagogy both
  limit spontaneous exploration and discovery. *Cognition, 120*(3), 322–330.
- Fonseca, B. A., & Chi, M. T. H. (2011). Instruction based on self-explanation.
  In R. E. Mayer & P. A. Alexander (Eds.), *Handbook of Research on Learning and
  Instruction* (pp. 296–321). Routledge.
- Hagoort, P. (2005). On Broca, brain, and binding: a new framework. *Trends in
  Cognitive Sciences, 9*(9), 416–423.
- Hagoort, P. (2019). The neurobiology of language beyond single-word processing.
  *Science, 366*(6461), 55–58.
- Monti, M. M., Parsons, L. M., & Osherson, D. N. (2012). Thought beyond
  language: neural dissociation of algebra and natural language. *Psychological
  Science, 23*(8), 914–921.
- Trabasso, T., & van den Broek, P. (1985). Causal thinking and the
  representation of narrative events. *Journal of Memory and Language, 24*(5),
  612–630.
- Graesser, A. C., Singer, M., & Trabasso, T. (1994). Constructing inferences
  during narrative text comprehension. *Psychological Review, 101*(3), 371–395.

---

## 8. How to read this document

- §1–§4 are the research summary. §5 is the implementation proposal and is the
  only section that should be acted on directly.
- §6 is the falsifiability contract. **F2 is the load-bearing test** — if
  chain-structure confidence does not beat support-count confidence, the
  adequacy gate is not doing anything and §5.3 should be abandoned.
- §6.1 lists what is *not* settled. Do not cite the self-explanation effect as
  established for factual learning; the 2025 null result is real.
- No file other than this one was modified.
