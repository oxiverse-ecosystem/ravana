# RAVANA — Social / Affective / Humor Layer: Neuroscience Grounding Brief

Goal: specify a brain-faithful HUMOR, AFFECT, and SOCIAL-INTENT layer for a
**non-LLM** cognitive architecture. All behaviour must be composed from RAVANA's
existing primitives — the ConceptGraph (typed edges), GloVe 64-D embeddings, and
the VAD (valence-arousal-dominance) emotion engine. No generative language model
is permitted; responses are assembled algorithmically.

Each mechanism is reported in the mandated shape:
- **(a)** leading brain model (2-3 sentences)
- **(b)** canonical papers (author-year + core finding)
- **(c)** the key insight that makes a naive implementation wrong
- **(d)** what an algorithmic, non-LLM model must capture

Citation precision is prioritized over prose.

---

## 1. HUMOR

### (a) Leading brain model
Humor is a **two-stage incongruity-detection → resolution** process: a percept
violates an active expectation (incongruity), then a reframe under a second
interpretive schema resolves it (bisociation). Detection/resolution is carried
by temporo-parietal and prefrontal cortex (semantic reframing), while the
"mirth" reward is a mesolimbic/dopaminergic signal (nucleus accumbens,
amygdala, VTA). The best current unifying account frames the funny moment as the
pleasurable **debugging of a covertly-committed error** in a just-in-time belief.

### (b) Canonical papers
- **Koestler 1964, _The Act of Creation_** — *bisociation*: humor arises when a
  single situation is perceived in two self-consistent but incompatible frames
  of reference ("matrices") that collide.
- **Suls 1972 (incongruity-resolution model)** — two-stage: (1) a punchline
  violates the expectation set up by the setup, (2) a cognitive rule is found
  that makes the incongruity fit; humor requires *both*, not incongruity alone.
- **Mobbs et al. 2003, _Neuron_** — funny cartoons activate the mesolimbic
  reward system (nucleus accumbens, amygdala, VTA); funniness ratings scale with
  reward-region activation → humor is literally rewarding.
- **Berthier et al. 2009 / 2013** — lesion & clinical work: humor comprehension
  vs. appreciation dissociate; right-hemisphere / frontal damage impairs the
  *resolution* stage while leaving incongruity detection partly intact.
- **Chan et al. 2013** — separates **detection** (temporo-parietal junction,
  posterior temporal) from **appreciation** (amygdala, midbrain, parahippocampal)
  — two neural stages matching Suls' two cognitive stages.
- **Amir & Biederman 2015** — humor engages both a *cognitive* network (for
  getting the joke) and a *reward/affective* network (for enjoying it), and the
  two are separable.
- **Vrticka, Black & Reiss 2013 (Nat. Rev. Neurosci.)** — developmental &
  social-affective review: humor recruits mentalizing (TPJ/MPFC) + reward; social
  context modulates funniness → humor is inherently social reward.
- **Hurley, Dennett & Adams 2011, _Inside Jokes_** — humor is the emotional
  reward for **detecting and retracting a mistaken covert inference** (a
  just-in-time belief committed during comprehension); mirth reinforces
  epistemic error-correction.
- **Predictive-coding / expectation-violation accounts** — humor as a resolved
  **prediction error**: a high-precision expectation is violated then re-explained
  cheaply; the reward tracks the *drop* in prediction error, not the surprise
  alone.

### (c) Naive-implementation pitfall
Treating humor as **surprise / low-probability = funny** is wrong. Pure
incongruity (a non-sequitur, a random word swap) produces confusion, not mirth.
The reward is contingent on **resolution** — a second frame must be *found* that
retroactively makes the incongruity coherent — and on **stakes**: it must
correct a belief the system had actually (covertly) committed to. Surprise
without a cheap reinterpretation is just error; reinterpretation without a prior
commitment is just a fact.

### (d) What the non-LLM model must capture
1. **Dual-frame retrieval.** For a candidate item, hold the *active* interpretive
   frame (the concept subgraph / vector primed by the setup) AND search for a
   *second* frame that also fits the surface form — an alternate sense, a distant
   graph neighbor, or a low-cosine-but-valid concept sharing the trigger token.
   Bisociation = two frames, one anchor.
2. **Commitment then violation.** Model the setup as raising the *precision*
   (confidence) of one prediction; the punch is scored by how strongly that
   prediction was held and then overturned. No prior commitment → no humor.
3. **Resolution gate.** Only emit "funny" when a second frame *reconciles* the
   incongruity at low cost (a discoverable link exists in the graph or embedding
   space). Incongruity with no resolving edge → tag as confusion/nonsense, not
   humor.
4. **Reward tagging.** On successful resolution, emit a positive VAD spike
   (high valence, high arousal, mid/high dominance) — the mesolimbic "mirth"
   signal — and let it reinforce the reinterpretation path (Hurley/Dennett:
   reward for error-correction).
5. **Social modulation.** Scale the reward by social context (shared frame,
   safe/benign stakes) via the SOCIAL-INTENT layer — funniness is not intrinsic.

---

## 2. AFFECT

### (a) Leading brain model
Emotion is not a fixed set of discrete labels but a **constructed** state built
from two ingredients: a low-dimensional **core affect** (Russell's circumplex:
valence × arousal) and a **conceptual/predictive categorization** of that state
using prior experience and context (Barrett's theory of constructed emotion).
Core affect is grounded in **interoceptive predictive coding** — the brain's
running prediction of bodily/physiological state (insula, anterior cingulate) —
so an emotion label is the brain's best explanation of an interoceptive signal
in context.

### (b) Canonical papers
- **Russell 1980 (circumplex model)** — affect is organized in a 2-D space of
  **valence** and **arousal**; discrete emotion words are points on this circle,
  not primitives.
- **Mehrabian & Russell 1974** — the **PAD** model: **P**leasure, **A**rousal,
  **D**ominance as three near-orthogonal dimensions spanning emotional state
  (dominance added to capture control/potency; this is RAVANA's VAD).
- **Mehrabian 1996 (PAD manual/temperament)** — operationalizes and validates
  the three PAD axes and maps emotion terms into the cube.
- **Barrett 2006, "Solving the emotion paradox"** — discrete emotions are not
  natural kinds; they are **constructed** by conceptual categorization of core
  affect → labels are context-dependent, not innate detectors.
- **Barrett 2017 (theory of constructed emotion / active inference)** — emotions
  are predictions (concepts) the brain generates to explain interoceptive input.
- **Seth 2013 ("interoceptive inference") / Seth & Friston 2016 / Seth 2021** —
  feelings arise from **predictive coding of interoceptive signals**; emotion =
  inferred cause of bodily prediction error.
- **Critchley & Garfinkel 2004 / 2011 / 2017** — insula & ACC integrate
  interoception; **interoceptive accuracy** modulates felt emotion intensity
  (e.g., heartbeat-detection → anxiety coupling).
- **Warriner, Kuperman & Brysbaert 2013** — human VAD norms for **13,915
  English lemmas**; the empirical lexicon-level ground truth for word affect.
- **Mohammad 2018 (NRC-VAD Lexicon)** — reliable valence/arousal/dominance
  ratings for ~20,000 words via best-worst scaling; the go-to computational VAD
  resource.
- **Mohammad & Turney 2013 (NRC EmoLex)** — crowd-sourced associations of words
  to 8 Plutchik emotions + 2 polarities; discrete-category complement to VAD.
- **Recchia & Louwerse 2015 / word2vec-affect work** — distributional vectors
  **reproduce human VAD norms**: affect is recoverable from co-occurrence
  geometry, so embeddings can extend a seed lexicon to unrated words.

### (c) Naive-implementation pitfall
Two traps. **(1)** Treating emotion as a **hard lookup of discrete labels**
("angry"/"happy") — Barrett shows these are constructed and context-dependent,
so a fixed classifier misfires across contexts. **(2)** Assuming **semantic
similarity = affective similarity**. Cosine-near words can be affective opposites
("love"/"hate", "hero"/"villain" sit close distributionally but split on
valence). Distributional geometry captures *topic*, not reliably *valence* —
using raw cosine as an affect proxy inverts sign on antonym pairs.

### (d) What the non-LLM model must capture
1. **Dimensional core, not labels.** Represent affect as continuous **VAD**
   (Russell/Mehrabian), and derive any discrete label as the *nearest region* in
   VAD space — labels are outputs of categorization, never the substrate.
2. **Seeded lexicon + vector extension (carefully).** Ground VAD in real norms
   (Warriner 2013 / NRC-VAD, EmoLex for categories). For unrated words, extend
   via embeddings — but as **supervised regression from GloVe → VAD** trained on
   the seed lexicon, *not* raw cosine. This is the principled answer to the open
   question: distributional semantics is a valid stand-in for affect **only when
   projected onto an affect-supervised axis**, because affect is a *learned
   direction* in embedding space, not the ambient metric.
3. **Constructed categorization.** The emotion assigned to a state = core affect
   (VAD) + context (active concepts / recent graph activation) → best-matching
   emotion concept. Same VAD, different context → different label (Barrett).
4. **Interoception proxy with no body (open question).** Substitute the missing
   physiological signal with **internal cognitive state as the interoceptive
   channel**: prediction-error magnitude, retrieval confidence/fluency, goal
   progress, resource/uncertainty load. Map these to arousal (prediction-error &
   uncertainty), valence (goal congruence / expected-value), dominance
   (controllability / confidence). This is Seth's interoceptive inference with
   *epistemic* interoception replacing visceral interoception — the honest
   principled proxy: the system reads its **own processing signals** as its "body."
5. **Intensity from confidence.** Per Critchley & Garfinkel, scale felt
   intensity by the reliability/precision of the internal signal, not just its
   value.

---

## 3. SOCIAL-INTENT

### (a) Leading brain model
Understanding others' intentions runs on the **mentalizing / Theory-of-Mind
network** — temporo-parietal junction (TPJ) and medial prefrontal cortex (MPFC),
overlapping the **default-mode network** — which infers unobservable mental
states (beliefs, goals) behind observed behavior. In real interaction this is
complemented by **second-person / interactive** engagement and by **forward
(predictive) models** of the interlocutor: comprehension partly works by
covertly *predicting/simulating* what the other will say, using the same
production machinery (motor/language reuse).

### (b) Canonical papers
- **Saxe & Kanwisher 2003** — the **TPJ** is selectively engaged when reasoning
  about another person's **beliefs** (not just about people or scenes) → a
  dedicated mentalizing substrate.
- **Frith & Frith 2006, "The neural basis of mentalizing"** — reviews the core
  **social brain** (MPFC, TPJ, temporal poles) for attributing mental states;
  automatic vs. controlled mentalizing.
- **Spreng, Mar & Kim 2009 (meta-analysis)** — autobiographical memory,
  prospection, and theory-of-mind converge on a **common core = default-mode
  network**; social inference reuses the self-projection/simulation system.
- **Schilbach et al. 2013, "Toward a second-person neuroscience"** — social
  cognition during **active interaction** (being engaged) differs from detached
  observation; the interaction itself is constitutive.
- **Pulvermüller 2003 / 2018** — language is grounded in **distributed
  action-perception circuits** (neural reuse); word meaning activates sensorimotor
  systems → semantics is embodied and reused, not amodal symbols.
- **Pickering & Garrod 2013, "An integrated theory of language production and
  comprehension"** — comprehension uses **forward models**: listeners predict the
  speaker's utterance via their own production system → prediction-by-simulation.
- **Dell 1986 (spreading-activation model of production)** — language production
  as **spreading activation** over a network of semantic/lexical/phonological
  units; selection by activation levels — a directly implementable non-LLM
  production mechanism.

### (c) Naive-implementation pitfall
Treating social intent as **surface pattern-matching on the utterance** (keyword →
canned intent) ignores that intent is an **inferred hidden variable** about the
*other agent's* mental state, computed against a **model of that specific
interlocutor** and the shared context. The same words carry different intent
across speakers, histories, and stakes. Equally wrong: modeling comprehension as
passive parsing — the brain **predicts** the partner (forward model) and treats
the interaction as coupled, not one-directional.

### (d) What the non-LLM model must capture
1. **Explicit other-agent model.** Maintain a separate belief/goal state for the
   interlocutor (a small ToM subgraph: what they know, want, expect) distinct
   from RAVANA's own — mentalizing is inference *over a second mind*, per
   Saxe/Kanwisher & Frith.
2. **Self-projection reuse.** Infer the other's likely goal by **simulating with
   your own machinery** on their inputs (Spreng's DMN reuse; Pickering & Garrod
   forward model): run RAVANA's own goal/response generator on the *estimated*
   partner state and read the prediction.
3. **Predict-then-compare.** Before/while comprehending, generate a forward
   prediction of the partner's intent; the *mismatch* between predicted and
   actual drives updating (and feeds the HUMOR expectation-violation and AFFECT
   prediction-error channels — one shared prediction-error currency across all
   three layers).
4. **Spreading-activation production.** Assemble the response via Dell-style
   spreading activation over the ConceptGraph (semantic → lexical selection by
   activation), grounded in Pulvermüller-style action-perception concept nodes —
   the concrete non-LLM generation path.
5. **Interaction-sensitive.** Weight inference by engagement/relationship state
   (Schilbach second-person): direct address, shared history, and stakes modulate
   how much intent-inference and social reward are applied.

---

## Cross-cutting: one shared prediction-error currency

All three layers reduce to **generate a prediction → measure the error → act on
it**, which is why they should share machinery in RAVANA:

- **HUMOR** = a high-precision prediction violated, then *cheaply resolved* by a
  second frame → reward spike.
- **AFFECT** = interoceptive/epistemic prediction error read as VAD (arousal ∝
  error magnitude & uncertainty; valence ∝ goal congruence; dominance ∝ control).
- **SOCIAL-INTENT** = a forward prediction of the *other agent*, with error
  driving belief update.

Implementation implication: build **one prediction+error core** over the
ConceptGraph and GloVe space, feeding a shared VAD engine, with each layer
differing only in *what* is predicted and *how* the error is interpreted.

## Answers to the two open questions
1. **Interoception with no body** → use **epistemic/processing interoception**:
   treat internal signals (prediction-error magnitude, retrieval confidence &
   fluency, uncertainty, goal-progress, resource load) as the interoceptive
   channel and map them to VAD (Seth's interoceptive inference, generalized).
2. **Is distributional semantics a valid affect proxy?** → **Only via supervised
   projection.** Raw cosine tracks topic and inverts on antonyms; but a
   regression from GloVe → seed VAD norms (Warriner/NRC-VAD) recovers human VAD
   well (Recchia & Louwerse 2015). Affect is a *learned direction* in embedding
   space, not the ambient similarity metric.
