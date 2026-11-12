# How the Brain Constrains Language Production to Be Both Fluent and Grounded — and a Mechanism for RAVANA

**Date:** 2026-11-12
**Scope:** Literature synthesis + concrete mechanism for RAVANA's neural decoder.
**Constraint:** No code change proposed. Mechanism only, in RAVANA's existing primitives.

---

## 0. Executive summary

The brain does **not** produce fluent text and then check it. It produces a **pre-verbal message**, commits to a **lexical-semantic plan**, and only then encodes form. Grounding is enforced at the **lemma/conceptual level before articulation**, by a **production-side conflict monitor** (anterior cingulate) that is *distinct* from the comprehension-based perceptual loop. The perceptual loop (Levelt) handles *post-lexical* error detection; the ACC conflict monitor handles *selection* errors. Both are needed, and they fail in different aphasias — which is the strongest evidence that they are separate systems.

RAVANA's current architecture has the **right shape** (a pre-articulation gate, `_sm_response_grounded`) but the **wrong grain and wrong direction**:

- It gates **after** the whole paragraph is generated (a post-hoc veto), not **per-selection** during generation.
- It gates on **lexical resemblance** (GloVe cosine) and **surface form**, not on **graph-committed propositional content**.
- It **falls through** to a different pathway on failure (template/uncertainty), rather than **re-running selection with a relaxed criterion** — which is what the brain does (conduit d'approche).

The mechanism below re-expresses grounding as a **selection-time constraint**: the decoder may only emit a token whose concept is *already committed* by the graph walk, and the commitment must clear a **conflict threshold** that rises with prediction error. This is the ACC monitor, implemented in RAVANA's own terms.

---

## 1. The neuroscience: what actually constrains production

### 1.1 The core architecture: message → lemma → form, with monitoring at each stage

Levelt's model (Levelt 1989; Levelt, Roelofs & Meyer 1999) is staged and feed-forward: **conceptual preparation → lexical selection → morphological/phonological encoding → phonetic encoding → articulation**. Two monitoring loops run in parallel:

- **Internal loop (pre-articulatory):** the incrementally built phonological word is fed back into the speech-comprehension system *before* articulation. This is the "perceptual loop" (Levelt 1983, 1989). It can intercept errors.
- **External loop (post-articulatory):** the speaker hears their own overt speech and parses it.

Crucially, Levelt (1983) showed from 959 self-repairs that **interruption follows detection promptly**, and that **detection improves toward the end of constituents** — i.e. monitoring is incremental and clause-grain, not whole-utterance. The monitor performs a **matching function**: it compares parsed inner/outer speech against (i) the intended message, (ii) contextual appropriateness, (iii) linguistic well-formedness.

**Implication for RAVANA:** RAVANA's gate is already clause-grained (good — it splits on `[.!?]`). But it runs on the *finished* text. The brain's internal loop runs on the *incremental plan*. The difference matters: a post-hoc veto can only reject; an incremental monitor can *redirect*.

### 1.2 The ACC conflict monitor: the neural analogue of a grounding gate

The most important and most mis-cited result for this problem is **Nozari, Dell & Schwartz (2011)**, *Cognitive Psychology*, 63, 1–33. They showed that error detection in speech production correlates with **production-side conflict**, *not* with comprehension ability — a double dissociation in aphasic patients. Their model:

- At each selection point (lexical layer, then phoneme layer), **conflict** = inverse of the activation difference between the top two candidates.
- A **criterion** is placed on conflict; responses exceeding it are flagged as errors.
- The signal is attributed to the **anterior cingulate cortex (ACC)**, following Botvinick et al. (2001) and Yeung, Botvinick & Cohen (2004).

This is the **production-based monitor**, and it is *complementary* to Levelt's comprehension-based perceptual loop. The two accounts are not rivals; they monitor different things (selection vs. form). See the synthesis in *Journal of Cognition* (2019), "Towards a New Model of Verbal Monitoring."

The ACC's broader role is **conflict monitoring and cognitive control** (Botvinick, Braver, Barch, Carter & Cohen 2001, *Psychological Review*, 108, 624–652). The **error-related negativity (ERN)**, a frontocentral ERP at 50–100 ms post-error, is the putative neural signature, sourced to the rostral ACC (Holmes & Pizzagalli 2008; Mathalon, Whitfield & Ford 2003). On **correct** high-conflict trials, the **N2** component indexes the same system *before* the response — i.e. the ACC signals conflict prospectively, not just retrospectively (van Veen & Carter 2002; Botvinick et al. 2001). This is the key point: **the monitor is predictive, not merely reactive.**

**Implication for RAVANA:** RAVANA's `prediction_free_energy` on nodes and edges is the right *quantity* (it is a prediction-error signal). But it is currently used as a *learning* signal, not as a *selection-time gate*. The ACC result says it should be the latter: **conflict at selection time determines whether a candidate is emitted at all.**

### 1.3 Predictive processing: the brain generates a prediction and verifies it before articulation

The predictive-processing framework (Friston 2010; Clark 2016, *Surfing Uncertainty*) holds that the brain maintains a generative model and minimizes **variational free energy** — an upper bound on surprisal. In speech, this is instantiated as **forward models with efference copies**:

- **Hickok (2012)**, *Nature Reviews Neuroscience*, 13, 135–145: the **state feedback control (SFC) model**. The auditory phonological system defines a **sensory target**; the lemma activates the motor phonological representation; the two interact so the motor plan *hits the sensory target*. Forward prediction is instantiated as an **inhibitory motor-to-sensory signal** (the efference copy). If the wrong motor program is activated, it inhibits a non-target, leaving the correction signal active — so **error correction is part of motor planning from the start**, not a post-hoc check.
- **Pickering & Garrod (2013)**, *Behavioral and Brain Sciences*, 36, 329–347: production and comprehension are **interwoven**; speakers construct **forward models of their own utterances** and use them to **monitor upcoming speech**. This is the "prediction-by-simulation" account. (See also Pickering & Garrod 2007 on action prediction; Garrod & Pickering 2009 on joint action.)

The critical claim for RAVANA: **the brain does not generate a sentence and then verify it. It generates a prediction of the sentence, compares the prediction to the intended message, and the comparison *is* the constraint.** Fluency and grounding are not two systems; they are one system with a prediction and a prediction-error signal.

**Honest flag:** the predictive-processing account of *production* is less empirically settled than its account of *perception*. The Hickok SFC model is a well-specified computational proposal, but direct neural evidence for forward-model-based *semantic* verification (as opposed to phonological/motor verification) is thinner. The strongest evidence is lesion/behavioral (aphasia dissociations), not real-time neural.

### 1.4 Speech errors and conduit d'approche: verification re-runs retrieval

**Conduit d'approche** (CdA) is the successive self-correction behavior in **conduction aphasia**: the patient produces a series of approximations converging on the target ("caterpillar … pattercillar … catapiller … caterpillar"). CdA is theoretically loaded because it shows:

1. The **target representation is intact** — the patient can *evaluate* each attempt against it.
2. The **monitor is intact** — the patient *detects* the mismatch.
3. The **failure is in phonological assembly/sequencing**, not in semantics or monitoring.

This is the clearest evidence that **verification during production re-runs retrieval with a relaxed criterion**, rather than falling through to a different process. The monitor does not say "I don't know, give up" — it says "that's not it, try again, closer."

**Implication for RAVANA:** RAVANA's gate currently **falls through** to `_human_like_uncertainty` on failure. The brain-faithful behavior is to **retry selection with a relaxed criterion** (lower the conflict threshold, re-walk the graph, re-decode), and only escalate to honest uncertainty after N failed attempts. This is the single most important behavioral change the mechanism below recommends.

### 1.5 Aphasia: what happens when the semantic→phonological constraint is damaged

Two syndromes are directly relevant, and they dissociate:

- **Semantic dementia (SD)** (Snowden, Goulding & Neary 1989; Hodges, Patterson, Oxbury & Funnell 1992, *Brain*, 115, 1783–1806; Neary et al. 1998): anterior temporal lobe atrophy destroys **semantic memory**. Patients produce **fluent but empty speech** — "lexical impoverishment with overreliance on high-frequency, abstract words replacing low-frequency, concrete vocabulary" (Hodges et al. 1992; cf. the recent review in *Clinical Interventions in Aging*, 2025). Speech is "meaningless and verbose." **Syntax, phonology, prosody, and fluency are preserved.** This is the *exact* syndrome RAVANA's decoder exhibits: fluent, grammatical, semantically empty.
- **Wernicke's aphasia** (posterior superior temporal lesion): fluent speech with **semantic paraphasias** ("watch" for "clock"), **phonemic paraphasias**, **neologisms**, and **jargon/word salad**. Patients are typically **unaware of errors** (StatPearls, *Wernicke Aphasia*, NBK441951). Here the comprehension system is damaged, so the perceptual-loop monitor cannot evaluate output.

The dissociation is the key: **SD = intact form, degraded semantics → fluent emptiness. Wernicke = degraded comprehension → fluent nonsense + no error awareness.** RAVANA's decoder is an **SD-like** system: the formulator is intact, the semantic store is too thin to constrain it. The fix is therefore **not** a better formulator (the decoder is already fluent) — it is a **stronger semantic commitment before form encoding.**

**Honest flag:** SD is a *degradation* of a previously intact semantic system; RAVANA's decoder is a *developmental* system that never had a grounded semantic store. The analogy is structurally apt (same failure signature) but not etiologically identical. Also, SD patients' speech is *more relevant* than Wernicke's (they use real words, just empty ones) — RAVANA's decoder output is closer to SD than to Wernicke's jargon, which is reassuring.

### 1.6 The dual stream: the dorsal stream constrains what gets articulated

**Hickok & Poeppel (2000, 2004, 2007)** dual-stream model:

- **Ventral stream** (bilateral STG/MTG → anterior MTG): speech **comprehension**, lexical-semantic processing.
- **Dorsal stream** (left-dominant, posterior planum temporale/area Spt → posterior IFG → premotor): **sensorimotor integration**, mapping acoustic/phonological representations to articulatory motor codes. It supports **online error detection and correction** and **forward prediction** (Hickok 2012; Hickok, Houde et al. 2011).

The dorsal stream is the **constraint pathway**: it is what turns a phonological plan into an articulatory gesture, and it does so by *predicting the sensory consequences* of the gesture and correcting on mismatch. In the SFC model, the dorsal stream's forward model is the **efference copy** that inhibits the sensory target when the correct motor program is active.

**Implication for RAVANA:** RAVANA's decoder has **no dorsal stream analogue** — it generates tokens directly from a GRU hidden state with attention over conditioning embeddings. There is no *articulatory prediction* and no *sensory target* to hit. The mechanism below proposes a **functional analogue**: the graph walk produces a **committed proposition** (the "sensory target"), and the decoder's output is scored against it (the "forward prediction"), with the score gating emission.

### 1.7 The hippocampus: binding what + where/context into an utterance

The hippocampus supports **relational/contextual binding** (Cohen & Eichenbaum 1993; Davachi 2006; Eichenbaum et al. 2007) — binding distributed cortical representations of items *and their contexts* into a unified episodic trace. In language:

- **Duff & Brown-Schmidt (2012)** and **Rubin et al. (2011)**: hippocampal amnesia patients have deficits in **on-line referential processing** over short discourse histories — they cannot maintain and integrate linguistic representations as they unfold.
- **MacKay et al. (2011)**: the hippocampus supports **discourse-level integration** through **linguistic binding** — the same binding processes that support episodic memory enable rapid formation of new connections between lexical, semantic, and phonological representations during discourse.

**Implication for RAVANA:** RAVANA's ConceptGraph is a **semantic** store (nodes + typed edges) but has no explicit **episodic/contextual binding** of *which* facts were learned *in which context*. The `ConceptEdge.context_labels` field is the seed of this. A grounded utterance requires not just "what is true of X" but "what is true of X *in this context*" — the hippocampal binding step. The mechanism below uses `context_labels` and the situation vector as the binding substrate.

---

## 2. What the evidence says the brain's actual constraint mechanism is

Synthesizing §1, the brain's constraint on production is **not a post-hoc filter**. It is a **selection-time commitment** with three properties:

1. **Commitment before form.** The speaker commits to a **lexical-semantic plan** (lemma + message) *before* phonological encoding. The plan is constrained by the semantic store (temporal lobe) and by the current context (hippocampal binding). *(Levelt 1989; Hodges et al. 1992; Duff & Brown-Schmidt 2012)*

2. **Conflict-gated selection.** At each selection point, the top candidate is emitted **iff** the activation gap over competitors clears a **criterion**. The criterion is a function of **conflict** (inverse gap) and is implemented by the **ACC**. *(Nozari, Dell & Schwartz 2011; Botvinick et al. 2001; Yeung et al. 2004)*

3. **Prediction-error-modulated threshold.** The criterion is not fixed: it **rises with prediction error / free energy** (uncertainty) and **falls with practice**. High uncertainty → more conservative selection → more retrieval, less generation. *(Friston 2010; Clark 2016; Hickok 2012)*

4. **Incremental, clause-grain monitoring with retry.** The plan is monitored **incrementally** (not post-hoc), and on mismatch the system **re-runs selection with a relaxed criterion** (conduit d'approche), not "give up." *(Levelt 1983; CdA literature)*

5. **Two monitors, two failure modes.** The **ACC conflict monitor** (production-side, selection errors) and the **perceptual loop** (comprehension-side, form errors) are distinct. SD damages the semantic store → fluent emptiness. Wernicke damages the perceptual loop → fluent nonsense + no awareness. *(Nozari et al. 2011; Hodges et al. 1992; StatPearls NBK441951)*

**The one-sentence version:** *The brain says true things because it commits to a semantic plan before it encodes form, and the commitment is gated by a conflict monitor that is modulated by prediction error — not because it generates fluent text and then checks it.*

---

## 3. A concrete mechanism for RAVANA

The mechanism is expressed entirely in RAVANA's existing primitives:

| Brain primitive | RAVANA primitive |
|---|---|
| Semantic store (temporal lobe) | `ConceptGraph` nodes + typed edges |
| Episodic/contextual binding (hippocampus) | `ConceptEdge.context_labels`, `situation_vec`, `node.temporal_context` |
| Lemma / message commitment | A **committed proposition** = (subject node, relation edge, target node) |
| Prediction error / free energy | `node.prediction_free_energy`, `edge.prediction_free_energy`, `FreeEnergyAccumulator` |
| Conflict at selection | Activation gap between the top graph-committed candidate and the next |
| ACC criterion | A threshold on conflict, modulated by prediction error |
| Perceptual loop | `_sm_response_grounded` (existing) |
| Conduit d'approche | **Retry selection with a relaxed criterion** (new) |
| Forward model / efference copy | Score decoder output against the committed proposition |

### 3.1 The mechanism: **Committed-Proposition Gating (CPG)**

**Step 1 — Commit (before decoding).**
For the query subject, the graph walk produces a **committed proposition**:
- a subject node `S` (from `_concept_keywords`),
- a set of **candidate target nodes** `{T_i}` reachable by typed edges from `S`,
- each candidate carries a **commitment strength** `c_i = edge.weight × edge.confidence × node.confidence × node.stability` (all existing fields),
- each candidate carries a **prediction error** `pe_i = max(node.prediction_free_energy(T_i), edge.prediction_free_energy(S→T_i))`.

The commitment is **not** "the subject is in the graph" (the current gate's step 1). It is "the graph asserts a specific relation between S and T_i, with a measurable confidence." This is the lemma-level commitment the brain makes before form encoding.

**Step 2 — Set the conflict threshold (ACC criterion).**
Compute the **conflict** among candidates: `conflict = 1 − (c_best − c_second)/max(c_best, ε)`. Set the threshold:
```
θ = θ_base + α · pe_best + β · FreeEnergyAccumulator.normalized
```
where `θ_base` is a base criterion, `α` scales with the best candidate's prediction error, and `β` scales with the graph's overall free energy. **High prediction error → higher threshold → more conservative emission.** This is the Friston/Clark modulation, and it is the opposite of the current behavior (which *lowers* the bar when uncertain, by falling through to templates).

**Step 3 — Gate each token at selection time (not post-hoc).**
During `NeuralDecoder.generate`, at each step, the candidate token `w` is **admissible** iff:
- `w`'s concept is **bound** to a committed target `T_i` (via `ConceptBinding` / `_concept_keywords`), **and**
- the commitment strength `c_i` clears `θ`, **and**
- the token is not a **function word** (existing `content_word_ids` mask).

If no admissible token exists at a step, the decoder **does not emit** — it **re-runs selection** with a relaxed threshold `θ ← θ − Δ` (conduit d'approche), up to `N` retries. Only after `N` failures does it escalate to honest uncertainty. This is the behavioral change that matters: **retry, don't fall through.**

**Step 4 — Verify the emitted clause against the commitment (perceptual loop).**
After each clause, `_sm_response_grounded` runs as today, but its **reference check** is re-anchored: instead of "does the sentence mention a GloVe-neighbour of the subject," it asks "does the sentence's content **realize** the committed proposition (S, relation, T_i)?" A clause that names `T_i` with the committed relation passes; a clause that names the subject but not the committed target fails. This is the **forward-model check**: the emitted form is compared to the sensory target (the committed proposition).

**Step 5 — Update the commitment (learning).**
If the clause passes, the committed edge's `confidence` and `prediction_count` are reinforced (existing Hebbian/plasticity machinery). If it fails after retries, the edge's `prediction_free_energy` is incremented (existing `apply_free_energy`). This closes the loop: the graph learns which propositions are reliable, and the threshold adapts.

### 3.2 Why this is brain-faithful (and why the current gate is not)

| Property | Brain | Current RAVANA gate | CPG mechanism |
|---|---|---|---|
| **Grain** | Incremental, per-selection | Post-hoc, per-paragraph | Per-token at selection + per-clause verify |
| **Constraint source** | Semantic store (lemma) | GloVe cosine (surface) | Graph-committed proposition (typed edge) |
| **Uncertainty response** | Raise threshold, retry retrieval | Fall through to template | Raise threshold, retry selection |
| **Monitor type** | ACC conflict + perceptual loop | Perceptual-loop only | Conflict gate + existing perceptual loop |
| **Prediction error** | Modulates threshold | Used for learning only | Modulates threshold (step 2) |
| **Failure mode** | CdA (retry, then escalate) | Immediate escalation | Retry N times, then escalate |

### 3.3 What this does NOT require

- **No new training objective.** The decoder's CE training is unchanged. CPG is a *runtime* constraint, not a retraining. (A grounding term in training is a separate, complementary idea — see §4.)
- **No new data structure.** Everything uses existing fields: `ConceptEdge.weight/confidence/prediction_free_energy/context_labels`, `ConceptNode.confidence/stability/prediction_free_energy`, `FreeEnergyAccumulator`, `ConceptBinding`.
- **No change to the decoder's forward pass.** The gate operates on the *candidate set* before sampling, using the existing `content_word_ids` / `token_boost` / `basal_ganglia` machinery.

### 3.4 The complementary training-side idea (flagged, not proposed)

The mechanism above is a **runtime constraint**. The brain's constraint is also **learned**: the semantic store is built over years, and the ACC criterion is calibrated by experience. RAVANA's decoder was trained on **grounded descriptions** (`train_decoder_grounded`) but with a **pure CE objective** — it learned to paraphrase coherent text, not to *select* true propositions. A complementary change would be to add a **grounding term** to the training loss: reward the decoder for emitting tokens whose concepts are graph-committed, penalize tokens that are not. This is the "semantic constraint in the formulator" that the brain has and RAVANA lacks. **This is flagged as a separate work item, not part of the CPG mechanism.**

---

## 4. What would falsify the mechanism

A mechanism is only useful if it can fail. CPG makes specific, testable predictions:

1. **If CPG is correct, gating at selection time should reduce fluent nonsense more than post-hoc gating.** Falsified if: post-hoc `_sm_response_grounded` and CPG produce identical output quality. (They should differ: CPG can *redirect* generation, post-hoc can only *veto*.)

2. **If the conflict threshold is the right variable, raising `θ` should trade fluency for grounding monotonically.** Falsified if: raising `θ` simply withholds more text without improving the truthfulness of what remains (i.e. the gate becomes a random censor).

3. **If retry-with-relaxation is brain-faithful, it should rescue cases the current fall-through loses.** Falsified if: retrying with relaxed `θ` produces *worse* output than immediate escalation to uncertainty (i.e. the retry loop amplifies drift rather than converging).

4. **If the commitment must be propositional (not just subject-presence), then a clause that names the subject but not a committed target should fail.** Falsified if: subject-presence alone predicts grounding as well as propositional commitment (i.e. the current gate's step 1 is sufficient).

5. **If prediction error modulates the threshold, then high-free-energy contexts should produce more conservative (more retrieved, less generated) output.** Falsified if: output verbosity/quality is invariant to `FreeEnergyAccumulator.normalized`.

6. **If the mechanism is SD-analogous, then strengthening the semantic store (more/better edges) should improve grounding more than improving the decoder.** Falsified if: decoder improvements (more training, better architecture) reduce fluent nonsense without any change to the graph.

7. **Neural falsifiability (for the brain claim):** if the ACC conflict monitor is the production-side gate, then ACC lesions should impair error *detection* without impairing comprehension. This is supported by Nozari et al. (2011) in aphasia; a direct ACC-lesion test in healthy speakers is harder. **Flag:** the ACC's role in speech monitoring is inferred from aphasia double-dissociations and ERP source localization, not from a clean lesion study in healthy speakers. The mechanism's neural claim is *consistent with* the evidence, not *proven* by it.

---

## 5. Honest flags: where the neuroscience is unsettled

1. **Predictive processing of production is less settled than of perception.** The Hickok SFC model is a computational proposal; direct neural evidence for *semantic* forward models (vs. phonological/motor) is thin. The strongest evidence is behavioral/lesion.

2. **The ACC conflict monitor vs. perceptual loop debate is not fully resolved.** Nozari et al. (2011) is influential but contested; the "Towards a New Model of Verbal Monitoring" (2019) review argues for a hybrid. The mechanism above uses *both* (conflict gate + perceptual loop), which is the safest reading.

3. **Conduit d'approche is clinical-case-based and not well quantified.** The "retry with relaxed criterion" reading is plausible, not established. The recent *Aphasiology* review (2021, doi:10.1080/02687038.2021.1881349) notes that repeated attempts do not always converge — they can move *away* from the target. So the retry loop must have a bounded N and an escalation path.

4. **SD is a degeneration, RAVANA is a development.** The analogy is structurally apt but not etiologically identical. SD patients had a grounded semantic store that degraded; RAVANA never had one. The mechanism addresses the *functional* deficit (no commitment constraint), not the *etiological* one.

5. **The hippocampus's role in language is real but diffuse.** Duff & Brown-Schmidt (2012) and MacKay et al. (2011) make the case, but the hippocampal contribution to *single-utterance grounding* (vs. discourse integration) is less direct. The mechanism uses `context_labels` as a binding substrate, which is a reasonable but *interpretive* mapping.

6. **"Grounding" is not a single thing.** The brain grounds utterances in semantics, episodic context, interoception, and social common ground simultaneously. RAVANA's ConceptGraph captures only the first two. The mechanism is a *partial* model, not a complete one.

---

## 6. Citations

- **Botvinick, M. M., Braver, T. S., Barch, D. M., Carter, C. S., & Cohen, J. D.** (2001). Conflict monitoring and cognitive control. *Psychological Review*, 108(3), 624–652.
- **Clark, A.** (2016). *Surfing Uncertainty: Prediction, Action, and the Embodied Mind*. Oxford University Press.
- **Cohen, N. J., & Eichenbaum, H.** (1993). *Memory, Amnesia, and the Hippocampal System*. MIT Press.
- **Duff, M. C., & Brown-Schmidt, S.** (2012). The hippocampus and the flexible use and processing of language. *Frontiers in Human Neuroscience*, 6, 742.
- **Friston, K.** (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11, 127–138.
- **Garrod, S., & Pickering, M. J.** (2009). Joint action, interactive alignment, and dialogue. *Topics in Cognitive Science*, 1(2), 292–304.
- **Hickok, G.** (2012). Computational neuroanatomy of speech production. *Nature Reviews Neuroscience*, 13(2), 135–145.
- **Hickok, G., & Poeppel, D.** (2000, 2004, 2007). The cortical organization of speech processing. *Nature Reviews Neuroscience*; dual-stream model.
- **Hodges, J. R., Patterson, K., Oxbury, S., & Funnell, E.** (1992). Semantic dementia: Progressive fluent aphasia with temporal lobe atrophy. *Brain*, 115(6), 1783–1806.
- **Levelt, W. J. M.** (1983). Monitoring and self-repair in speech. *Cognition*, 14(1), 41–104.
- **Levelt, W. J. M.** (1989). *Speaking: From Intention to Articulation*. MIT Press.
- **Levelt, W. J. M., Roelofs, A., & Meyer, A. S.** (1999). A theory of lexical access in speech production. *Behavioral and Brain Sciences*, 22(1), 1–75.
- **MacKay, D. G., James, L. E., Hadley, C. B., & Fogler, K. A.** (2011). On-line lexical processing in discourse: Hippocampal contributions. In *The Handbook of the Neuropsychology of Language*.
- **Neary, D., Snowden, J. S., Gustafson, L., et al.** (1998). Frontotemporal lobar degeneration: A consensus on clinical diagnostic criteria. *Neurology*, 51(6), 1546–1554.
- **Nozari, N., Dell, G. S., & Schwartz, M. F.** (2011). Is comprehension necessary for error detection? A conflict-based account of monitoring in speech production. *Cognitive Psychology*, 63(1), 1–33.
- **Pickering, M. J., & Garrod, S.** (2007). Do people use language production to make predictions during comprehension? *Trends in Cognitive Sciences*, 11(3), 105–110.
- **Pickering, M. J., & Garrod, S.** (2013). An integrated theory of language production and comprehension. *Behavioral and Brain Sciences*, 36(4), 329–347.
- **Snowden, J. S., Goulding, P. J., & Neary, D.** (1989). Semantic dementia: A form of circumscribed cerebral atrophy. *Behavioural Neurology*, 2(3), 167–182.
- **Yeung, N., Botvinick, M. M., & Cohen, J. D.** (2004). The neural basis of error detection: Conflict monitoring and the error-related negativity. *Psychological Review*, 111(4), 931–959.
- **StatPearls.** *Wernicke Aphasia* (NBK441951); *Conduction Aphasia* (NBK537006).
- **Towards a New Model of Verbal Monitoring.** (2019). *Journal of Cognition*, 2(1), 81.
- **Repeated attempts, phonetic errors, and syllabifications in a case study of conduction aphasia.** (2021). *Aphasiology*, doi:10.1080/02687038.2021.1881349.

---

## 7. One-paragraph answer to the question

The brain produces language that is both fluent and grounded because it **commits to a lexical-semantic plan before it encodes form**, and that commitment is **gated by a conflict monitor (ACC)** whose criterion is **modulated by prediction error** — not because it generates fluent text and then checks it. The monitor is **incremental** (clause-grain), **production-side** (distinct from the comprehension-based perceptual loop), and on mismatch it **re-runs selection with a relaxed criterion** (conduit d'approche) rather than giving up. RAVANA's decoder fails because it has the formulator (fluent) but no commitment constraint: it generates tokens that are not bound to any graph-committed proposition. The fix is a **Committed-Proposition Gate**: before decoding, the graph walk commits to a typed proposition (S, relation, T) with a confidence; during decoding, a token is admissible only if it realizes a committed target and clears a conflict threshold that rises with prediction error; on failure, retry with a relaxed threshold, then escalate. This uses only RAVANA's existing primitives (`ConceptEdge.weight/confidence/prediction_free_energy/context_labels`, `ConceptNode.confidence/stability`, `FreeEnergyAccumulator`, `ConceptBinding`, `_sm_response_grounded`) and requires no retraining.
