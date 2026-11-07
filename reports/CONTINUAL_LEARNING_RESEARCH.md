# Continuous learning in the human brain: mechanisms, evidence, and what is computationally replicable

**Research brief for RAVANA** — prepared from primary and review sources (see Sources at end).
Scope: how a brain learns for a lifetime without catastrophic forgetting or collapse, and which mechanisms are online, local, and CPU-plausible.

---

## 0. Executive summary

The brain does not solve catastrophic forgetting with a single mechanism. The converging picture across neuroscience and continual-learning research is a **stack of interacting mechanisms**:

1. **Timescale separation** — a fast, labile episodic system (hippocampal/MTL) and a slow, structured semantic system (neocortex), with the fast system writing to the slow system gradually. This is the Complementary Learning Systems (CLS) framework.[1][2]
2. **Offline replay** — reactivation of recent traces during rest/sleep, *interleaved* with older traces, which lets the slow system learn without overwriting. Replay is prioritised by salience, novelty, reward-prediction error, and expected future relevance.[21][20][22]
3. **Local synaptic consolidation and protection** — fast labile traces are progressively transferred to slow protected variables (cascade/synaptic-tagging models), and importance-weighted protection prevents overwriting. This is the biological counterpart of EWC / Synaptic Intelligence.[15][13][14][16]
4. **Neuromodulatory gating** — dopamine (reward-prediction error, novelty), norepinephrine/arousal, and acetylcholine (encoding vs retrieval mode) decide *which moments are worth plastic change*.[28][30][29][31]
5. **Adaptive forgetting / triage** — forgetting is not a bug; it reduces interference, prevents overfitting, and frees capacity for goal-relevant information.[19][18][21]
6. **Reconsolidation** — retrieval makes a stored trace transiently labile, so it can be *updated* rather than merely appended.[10][11][12]

**For RAVANA, the highest-value mechanisms are local, online, and CPU-cheap:** two-timescale consolidation per weight, local importance-weighted protection, a small episodic buffer with prioritised replay, a scalar salience/novelty write-gate, and rule-based triage. The least CPU-plausible are exact Fisher-information EWC at scale, progressive networks that grow per task, and full generative replay of raw inputs.

---

## 1. Complementary Learning Systems (CLS)

### (a) Mechanism
CLS theory holds that intelligent agents need **two learning systems**. The neocortex gradually acquires structured, parametric knowledge (the statistics of the environment, supporting generalisation), while the hippocampus and medial temporal lobe rapidly learn the specifics of individual items and episodes. New information is first stored in the fast system; replay of those traces during offline states drives gradual, interleaved learning in the slow system.[1][2]

### (b) Strongest evidence + citation
- McClelland, McNaughton & O'Reilly (1995) — the original formulation, derived from connectionist modelling successes and failures.[2]
- Kumaran, Hassabis & McClelland (2016) — the updated review; extends replay's role to goal-dependent weighting of experience statistics, responds to challenges (hippocampal generalisation; rapid neocortical learning for schema-consistent information), and links to ML memory modules.[1]
- Behavioural/neuropsychological evidence: medial temporal lobe damage (patient H.M. and others) produces severe anterograde amnesia with preserved remote/semantic memory, consistent with a fast-encoding, slow-consolidating division.[24]

### (c) Computational implication
- **Why must the neocortex learn slowly?** Two arguments: (i) each experience is a single sample, so a small learning rate aggregates information over many samples and estimates the population statistics more accurately; (ii) in a multilayer network the optimal adjustment of each weight depends on all the others, so before the ensemble is structured the error signals are noisy and weak — learning must be gradual and interleaved.[1]
- **Why does a single fast learner catastrophically forget?** Because the same distributed weights encode old and new knowledge. A large weight change to accommodate a new item (e.g. a dangerous lion at a watering hole) disrupts the representation of other items (other animals). This is catastrophic interference, and it is the stability–plasticity dilemma.[1][3][4]
- **Interleaved replay reconciles the systems:** replaying new hippocampal traces while old neocortical knowledge is still active lets the slow learner integrate new structure without overwriting. In artificial networks, interleaved training is the standard way to avoid catastrophic forgetting, and the same networks can learn many tasks when trained interleaved.[9][8]

### (d) Confidence + evidence type
**Well-established** as a computational framework and as a broad neuroscience account. Evidence types: computational modelling (strong), neuropsychology (strong), systems neuroscience (strong for replay/consolidation). **Contested/qualified:** the strict "hippocampus = only episodic, no generalisation" version has been revised — the hippocampus supports some forms of generalisation, and neocortical learning can be rapid when new information is consistent with existing structure.[1]

---

## 2. Catastrophic forgetting / interference

### (a) Mechanism
Catastrophic interference is the abrupt loss of previously learned information when a network is trained sequentially on new information. It arises from **overlap in distributed representations**: new learning changes the weights that encode old learning.[3][4]

### (b) Strongest evidence + citation
- McCloskey & Cohen (1989): backpropagation networks trained on "ones" addition facts lost them after a single learning trial on "twos" facts; in an A-B/A-C list paradigm, training that produced 50% correct in humans produced near 0% in the network, with C responses intruding on B cues. Interference was catastrophic when learning was sequential but not concurrent.[4]
- Ratcliff (1990): even one learning trial caused significant loss of well-learned information; outputs became blends of old and new; discrimination between studied and unseen items *decreased* with learning, unlike human memory.[4]
- French (1999): review framing catastrophic forgetting as the radical end of the stability–plasticity problem.[6]
- Kirkpatrick et al. (2017), EWC: quantifies the importance of weights to previous tasks via the Fisher information and selectively slows learning on important weights.[5]
- Liu (2020) book chapter on continual learning: taxonomy of continual learning; catastrophic forgetting; rehearsal, EWC, Synaptic Intelligence, MAS, progressive networks, PackNet, iCaRL exemplars, and generative replay approaches.[3]

### (c) Computational implication — remedies and biological plausibility
| Remedy | Core idea | Biological plausibility | Online / local / CPU? |
|---|---|---|---|
| Rehearsal / replay | Store or regenerate old examples and interleave with new | **High** — hippocampal replay is the canonical mechanism | Online if buffer small; local updates; CPU-plausible |
| Pseudo-rehearsal / generative replay | Replay internally generated patterns, not raw data | **Moderate** — brain replays compressed traces, not pixels | CPU-heavy if a deep generator is needed; borderline |
| EWC | Quadratic penalty on changes to important weights (Fisher) | **Moderate** — analogous to consolidation strength | Online diagonal EWC is O(n); exact Fisher is O(n²) and task-boundary dependent |
| Synaptic Intelligence (SI) | Online estimate of each synapse's contribution to past learning; consolidate important synapses | **Moderate-High** — local, online, continuous | Online, local, O(n) memory; CPU-plausible |
| Progressive networks | Add a new sub-network per task with lateral connections | **Low-Moderate** — brain does add structure (e.g. neurogenesis) but not whole networks per task | Not bounded; grows with tasks; not scalable |
| PackNet / freezing | Allocate disjoint weight subsets per task | **Low** — no evidence for hard task-wise partition | Static once allocated; not truly continual |
| Latent-cause / contextual partitioning | Infer contexts and maintain separate parameters per context | **High** — contextual/latent-cause inference is a leading account of biological continual learning | Online; clustering cost; CPU-plausible |

### The reverse problem: catastrophic memorisation / overgeneralisation
The opposite failure is a system that protects or stores so much that it **cannot generalise** (overfitting to specific past events) or that memorises noise. Richards & Frankland argue that forgetting prevents overfitting to specific past events and promotes generalisation.[19] In ML, the same tension appears: lookup tables are perfectly stable but cannot generalise, while backpropagation generalises but is unstable.[4] A system that only appends (never overwrites) eventually either runs out of capacity or becomes a lookup table; a system that only overwrites forgets. **The design target is therefore controlled, weighted updating, not maximum stability or maximum plasticity.**

### (d) Confidence + evidence type
**Well-established** in artificial networks (computational modelling, strong). **Well-established** as a phenomenon to be explained biologically. The specific remedies are mostly computational; biological plausibility varies as tabulated. **Contested:** whether any single artificial remedy matches biology; whether the brain's solution is better described as regularisation, architecture, or memory systems (likely all three).[25]

---

## 3. Replay and consolidation

### (a) Mechanism
Replay is the reactivation of neural patterns resembling past waking experience, most prominently in the hippocampus during sharp-wave ripples in rest and slow-wave sleep, but also in wake and in cortex. Replay is temporally compressed, can be forward or reverse relative to experience, and is modulated by reward, novelty, and behavioural relevance.[21][22][7]

### (b) Strongest evidence + citation
- Wilson & McNaughton (1994): hippocampal place-cell ensembles reactivate during sleep in the same order as during prior experience.[21][24]
- Foster & Wilson (2006): awake replay can run in reverse as well as forward.[22][7]
- Diba & Buzsáki (2007): forward replay starting at the current location and moving along upcoming paths is more common than replay from remote locations.[7]
- Girardeau & Zugaro (2011): review of hippocampal ripples and memory consolidation; selective suppression of ripples impairs spatial memory (Girardeau et al. 2009, reviewed in [32]).[32]
- Rasch & Born (2013): sleep actively maintains memories by reactivation, not merely protects them passively.[33]
- Lewis & Durrant (2011): overlapping replay during sleep builds cognitive schemata by integrating shared features across experiences.[34]
- Bendor and colleagues (2023): sleep replay rate increases proportionally with the number of trajectories run and decreases with familiarity; the cumulative number of *awake* replay events predicts which memories are prioritised for sleep replay — a parsimonious neural correlate of memory triage.[21]
- Kleinman & Foster (2024): spatial localisation of replay depends on dopamine signalling; reverse-ordered replay increases with reward changes; on familiar tracks, ripples track reward-prediction error.[22]

### (c) What does replay DO computationally?
Replay is not a single function. The evidence supports at least these computational roles:
1. **Consolidation / systems transfer** — reactivation strengthens neocortical traces and gradually makes memories hippocampus-independent.[24][21]
2. **Interleaving / interference reduction** — replay of old traces during new learning prevents catastrophic forgetting. In spiking networks, interleaving new training with offline reactivation constrains synaptic weights to the old-task manifold and converges toward the intersection of old and new manifolds.[8]
3. **Credit assignment / valuation** — reverse replay propagates reward information backwards along trajectories (Bellman-like backups). Mattar & Daw (2018) show that a rational agent should access memories in order of utility, formalised as a **gain term × need term**; this reproduces forward, reverse, and offline replay and unifies planning, learning, and consolidation under one rule.[20]
4. **Generalisation / schema building** — overlapping replay across related experiences extracts shared structure.[34][18]
5. **Prioritisation / triage** — replay is biased toward salient, novel, reward-related, and expected-to-be-relevant material; low-priority memories are effectively forgotten.[21][18][22]

### (d) Confidence + evidence type
**Well-established** that replay occurs and is necessary for consolidation (causal ripple-suppression studies; neurobiology, strong). **Moderate** for the specific computational role of reverse replay and for prioritisation mechanisms (behaviour + modelling). **Contested:** whether replay's primary function is consolidation, planning, or valuation — Mattar & Daw's account unifies these but is a normative theory, not a mechanistic account.[20] **Well-established** that replay-inspired mechanisms help artificial continual learning.[7][9]

---

## 4. Reconsolidation

### (a) Mechanism
When a consolidated memory is retrieved, it transiently returns to a labile state and must be restabilised — a process requiring protein synthesis, analogous to initial consolidation. Retrieval therefore opens a window in which a memory can be modified, strengthened, weakened, or erased.[10][11]

### (b) Strongest evidence + citation
- Nader, Schafe & LeDoux (2000): in auditory fear conditioning, a consolidated memory became sensitive to the protein-synthesis inhibitor anisomycin in the basolateral amygdala *only if* the memory was reactivated; anisomycin given without reactivation had no effect, and a 6-hour delay eliminated the effect. This demonstrated that reactivation returns a consolidated memory to a labile state.[11]
- Nader (2015) review: consolidation theory cannot explain the reconsolidation dataset; new and reactivated memories are in an "active state" and stabilise over time into an "inactive" state; remembering returns them to the active state.[10]
- Lee, Nader & Schiller (2017): review of reconsolidation updating, with implications for clinical interventions.[12]

### (c) Computational implication
For a system that must **update a stored fact rather than append a new one**, reconsolidation has a clear implication: retrieval should make the trace labile and re-writable within a bounded window. Concretely:
- A retrieved concept/association should be marked "updateable" for a period, during which new evidence can revise its value/attributes.
- After the window, the trace should be re-protected (consolidated).
- This avoids both (i) endless append-only growth and (ii) immediate overwriting that loses the old value before the new evidence is evaluated.
- In ML terms this resembles a cache-coherence or "read-modify-write" protocol with a soft write-back window, or a transient reduction of a weight's consolidation strength upon retrieval.

### (d) Confidence + evidence type
**Well-established** in fear conditioning and several other paradigms (neurobiology, strong). **Moderate/contested** in humans and for complex declarative material; boundary conditions determine when reconsolidation occurs versus when retrieval simply updates without destabilisation.[10][12]

---

## 5. Neuromodulatory gating of plasticity

### (a) Mechanism
Neuromodulators act as a **write gate**, determining which moments and synapses are eligible for plastic change:
- **Dopamine** encodes reward-prediction error and novelty, and gates LTP/LTD induction in hippocampus and striatum.[28][22][31]
- **Norepinephrine / locus coeruleus** signals arousal and salience; the LC-NE adaptive-gain theory links tonic/phasic LC activity to exploitation versus exploration.[30]
- **Acetylcholine** biases the hippocampus between encoding and retrieval modes; high ACh during novelty suppresses retrieval-mode circuits and promotes encoding.[29]

### (b) Strongest evidence + citation
- Schultz, Dayan & Montague (1997): dopamine neurons signal reward-prediction error.[22]
- Lisman, Grace & Düzel (2011): a neoHebbian framework in which dopamine-dependent late LTP is the mechanism for episodic memory; dopamine shifts the LTD/LTP threshold.[28]
- Shohamy & Adcock (2010): dopaminergic modulation enhances memory encoding for motivationally relevant information — "adaptive memory".[31]
- Aston-Jones & Cohen (2005): LC-NE adaptive gain theory — phasic LC responses signal task-relevant events and promote exploitation; tonic mode promotes exploration.[30]
- Hasselmo (2006): acetylcholine regulates encoding versus retrieval; cholinergic activity during novelty enhances encoding and suppresses competing retrieval.[29]
- Kleinman & Foster (2024): dopamine is required for the spatial localisation of hippocampal replay; reward-prediction error directs replay content.[22]

### (c) Computable signals
The brain's gating signals are **scalar, online, and computable**:
- **Prediction error / surprise:** |observed − predicted|. A large error opens a plasticity window. This is the single most implementable gating signal.
- **Novelty:** distance of current input/representation from stored traces (e.g. low familiarity or high pattern-separation signal). Novelty gates encoding and promotes LTP.[28][29]
- **Reward / value:** signed reward-prediction error; positive errors potentiate, negative errors depress.[28][31]
- **Arousal / salience:** a global gain signal (LC-NE analogue) that scales plasticity and controls exploration.[30]
- **Expected future relevance / test expectancy:** information that will be needed later is prioritised for consolidation.[18][21]

For RAVANA, a scalar **write gate** `g = f(|prediction_error|, novelty, reward, relevance)` multiplying local Hebbian weight changes is a direct, online, local implementation of these findings.

### (d) Confidence + evidence type
**Well-established** that dopamine encodes reward-prediction error and gates plasticity (neurobiology, strong). **Well-established** that acetylcholine separates encoding and retrieval (neurobiology/physiology, strong). **Moderate** for LC-NE adaptive gain as a unified account (theory strongly supported by behavioural and physiological work, but contested in detail).[30] The exact mapping from these signals to a single scalar gate is **computational modelling**, not established neurobiology.

---

## 6. Metaplasticity and the stability–plasticity dilemma

### (a) Mechanism
**Metaplasticity** is activity-dependent regulation of the *capacity* for subsequent plasticity — "plasticity of plasticity". A synapse's history determines whether and how much it can change next. Related mechanisms include **heterosynaptic plasticity** (changes at one synapse affect neighbouring synapses) and **homeostatic synaptic scaling**.[17][13]

### (b) Strongest evidence + citation
- Abraham & Bear (1996): coined metaplasticity; prior activity raises or lowers the LTP induction threshold.[17]
- Heterosynaptic metaplasticity in vivo: conditioning stimulation of medial perforant path afferents raised the LTP induction threshold at neighbouring synapses in a BCM-like, postsynaptically regulated manner.[37]
- Zenke & Gerstner (2017): Hebbian plasticity alone is unstable; **rapid compensatory processes** (e.g. heterosynaptic depression triggered by high postsynaptic firing) stabilise it on short timescales, while slower homeostatic processes fine-tune circuits. Slow synaptic scaling (hours) is not fast enough to stabilise Hebbian learning by itself.[13]
- Benna & Fusi (2016): a cascade of biochemical processes operating on multiple timescales can store memories in fast variables and progressively transfer them to slow variables; with bidirectional interactions, memory lifetime scales almost linearly with the number of synapses and the model reproduces power-law forgetting, metaplasticity, delayed expression of synaptic changes, and spacing effects.[15]
- Synaptic tagging and capture: weak stimulation sets a transient, protein-synthesis-independent "tag"; strong stimulation of an independent input supplies plasticity-related proteins that stabilise the tagged synapse into late-LTP. Demonstrated in vivo in the living rat — the past and future activity of neurons determines the persistence of synaptic change.[16]

### (c) Is there a computable rule for when to change a weight and when to protect it?
Yes — several local, computable rules emerge:
1. **Two-timescale consolidation:** maintain a fast labile weight and a slow protected weight; on each update move a fraction of the fast weight into the slow weight. The slow weight changes only gradually, so old knowledge is protected while new knowledge is still learnable. Memory lifetime scales near-linearly with the number of synapses.[15]
2. **Importance-weighted protection (EWC/SI analogue):** track a running online estimate of how much each weight contributes to past predictions; penalise changes to important weights. Synaptic Intelligence estimates this online and locally.[14][5]
3. **Tag-and-capture:** set a transient "tag" on a weakly supported change; commit it to long-term storage only if a sufficiently strong/salient event supplies consolidation resources within a time window. This gives a computable rule: weak changes are provisional and decay; strong/salient changes rescue them.[16]
4. **Sliding threshold (BCM-like metaplasticity):** the threshold for LTP/LTD slides as a function of recent postsynaptic activity, so a synapse that has been recently potentiated is harder to potentiate again and easier to depress. This prevents runaway potentiation and is a purely local rule.[17][37]
5. **Heterosynaptic competition:** increasing one weight onto a neuron decreases others onto the same neuron (normalisation), implementing a fixed resource budget. Fast normalisation stabilises Hebbian learning.[13]

### (d) Confidence + evidence type
**Well-established** that metaplasticity and heterosynaptic regulation exist (neurobiology, strong; in vivo evidence for BCM-like threshold shifts).[37][17] **Well-established** computationally that Hebbian plasticity needs fast compensatory mechanisms on multiple timescales.[13] **Moderate** for the cascade model's specific scaling claims (computational modelling, strong within its assumptions; the exact biophysical implementation is not settled).[15] Synaptic tagging and capture is **well-established in vitro and in vivo** for the phenomenon; the precise molecular mechanism remains an active research area.[16]

---

## 7. One-shot / episodic learning and schema assimilation

### (a) Mechanism
The hippocampus can store a single episode after one exposure. The **hippocampal indexing theory** proposes that the hippocampus does not store the content of an experience but a compact **index** to the distributed neocortical patterns that constitute it; partial cues reactivate the index, which reactivates the cortical pattern, enabling retrieval and pattern completion.[26] Over time, repeated and overlapping experiences are integrated into neocortical **schemas** — structured knowledge networks that support rapid assimilation of related new information.[35][23]

### (b) Strongest evidence + citation
- Teyler & DiScenna (1986): the original indexing proposal.[26]
- Teyler & Rudy (2007): updating the index; conclude the theory "has aged very well" and its core ideas appear in many contemporary theories.[26]
- Gilboa & Marlatte (2017): neurobiology of schemas and schema-mediated memory; vmPFC and hippocampus support assimilation of related new information.[35]
- Sommer et al. (2022): participants overlearned schemata for 7 weeks, then encoded novel schema-related and control facts; schema-related facts showed enhanced vmPFC–hippocampus coupling at encoding and **more efficient systems consolidation** over 2 weeks.[23]
- Tse et al. (2007): prior knowledge accelerates systems consolidation of novel, expectation-neutral information in animals.[23]
- Preston & Eichenbaum (2013): hippocampus–prefrontal interplay supports rapid encoding and the assimilation of new memories into pre-existing schema networks.[24]
- Piette et al. (2026): one-shot learning in the striatum is supported by a nonclassical plasticity mechanism (endocannabinoid-mediated LTP requiring only a few stimulations), distinct from the NMDAR-LTP that supports slower, repeated-experience learning.[27]

### (c) Computational implication
- Store **compressed indices**, not full copies, of episodes; retrieve via pattern completion. This is CPU-cheap and mirrors the brain's apparent strategy.
- Distinguish **fast one-shot storage** (episodic buffer, sparse/one-shot plasticity) from **slow semantic integration** (schema/graph consolidation).
- **Schema assimilation:** when new information matches existing graph structure, integrate it rapidly (and with less replay); when it does not, treat it as a novel episode requiring consolidation. This directly reduces the load on replay and explains how the neocortex can sometimes learn quickly without forgetting.[1][23]
- The existence of a distinct one-shot plasticity mechanism (eCB-LTP) suggests that **brief, salient events can trigger lasting change with very few spikes**, which matters for CPU-plausible learning: a salient event should be able to write immediately without many repetitions.[27]

### (d) Confidence + evidence type
**Well-established** that the hippocampus supports one-shot episodic learning (neuropsychology and physiology, strong). **Moderate** for indexing theory specifically (a influential framework supported by converging data, but not directly proven). **Well-established** that prior knowledge/schema accelerates memory encoding and consolidation (behaviour + fMRI, strong).[23][35] The eCB-LTP one-shot mechanism is **recent and strong for the striatum**, but its generality across systems is still being established.[27]

---

## 8. Forgetting as adaptive

### (a) Mechanism
Forgetting is not merely decay; it is an active, regulated process that reduces the influence of outdated information and prevents overfitting. Mechanisms include synaptic depotentiation/elimination, active suppression of competing memories, and adult neurogenesis.[19]

### (b) Strongest evidence + citation
- Richards & Frankland (2017): transience (1) enhances flexibility by reducing the influence of outdated information on decision-making and (2) prevents overfitting to specific past events, promoting generalisation. The goal of memory is to optimise decision-making, not to transmit information perfectly.[19]
- Cowan et al. (2021): consolidation is adaptive — goal-relevant information is tagged and prioritised, while mundane information is forgotten; consolidation also transforms memories by integrating across related experiences.[18]
- Bendor et al. (2023): memory triage — sleep replay prioritises salient/novel experiences and deprioritises familiar ones, leading to eventual forgetting of low-priority memories.[21]
- Wixted (2004): recently formed memories that have not consolidated are vulnerable to interference; as traces consolidate they become increasingly resistant to interference even as they fade — suggesting an active, time-dependent process.[36]
- Stickgold & Walker (2013): sleep-dependent memory triage — selective processing that enhances generalisation while allowing forgetting.[19]

### (c) What should be FORGOTTEN, and by what rule?
The evidence points to a small set of computable criteria:
1. **Disuse / recency:** memories not reactivated or referenced over time lose priority (natural decay of the fast trace).
2. **Low usefulness / low reward relevance:** information not associated with reward, threat, or goal relevance is deprioritised and eventually forgotten.[18][21]
3. **Redundancy / familiarity:** information already well represented by the schema (high familiarity) is not repeatedly consolidated.[21]
4. **Dissonance / inconsistency:** information that conflicts with the current model or has been superseded should be weakened or replaced rather than retained alongside the new fact.
5. **Interference reduction:** forgetting competing memories that would otherwise interfere with retrieval of important information.[19]
6. **Capacity pressure:** under limited capacity, retain the subset that maximises expected future decision quality (a rate-distortion / triage objective).

For RAVANA, a practical rule is a **utility score** per concept/association combining recency, frequency, reward/relevance, and consistency with the current graph; periodically consolidate high-utility traces and decay or evict low-utility ones. This is local, online, and CPU-cheap.

### (d) Confidence + evidence type
**Well-established** that forgetting has adaptive functions and is actively regulated (behaviour + neurobiology, strong).[19][18] **Moderate** for the specific claim that forgetting promotes generalisation (computational argument strongly supported; direct causal evidence in humans is harder to obtain). **Contested:** the relative contributions of decay versus interference versus active suppression, and whether forgetting is always adaptive versus sometimes a failure mode.[36][19]

---

## 9. Synthesis: what RAVANA should replicate

### 9.1 Mechanism-to-implementation map

| Mechanism | Biological basis | Computational form for RAVANA | Online? | Local? | CPU-plausible? | Evidence strength |
|---|---|---|---|---|---|---|
| Two-timescale consolidation | Cascade of synaptic processes; synaptic tagging and capture[15][16] | Fast labile weight + slow protected weight per parameter; slow weight integrates a fraction of the fast weight | Yes | Yes | Yes | High |
| Importance-weighted protection | EWC/SI analogue; consolidation strength[5][14] | Running online estimate of each weight's contribution; penalise changes to important weights | Yes | Yes | Yes (diagonal/online) | High (computational); moderate (biological) |
| Prioritised replay | Hippocampal replay; gain × need; salience[20][21][22] | Small episodic buffer; replay with priority = salience × novelty × expected relevance; interleave with new learning | Yes | Yes | Yes (small buffer) | High (phenomenon); moderate (normative rule) |
| Salience/novelty write gate | Dopamine RPE; LC-NE; acetylcholine[28][30][29] | Scalar gate `g = f(|prediction_error|, novelty, reward, relevance)` multiplying local Hebbian updates | Yes | Yes | Yes | High |
| Schema assimilation | vmPFC–hippocampus coupling; schema networks[23][35] | Detect match to existing concept graph → rapid integration; else episodic storage | Yes | Yes | Yes | High |
| Reconsolidation / update | Retrieved memory becomes labile[10][11] | On retrieval, mark trace updateable for a window; allow revision; re-protect afterwards | Yes | Yes | Yes | Moderate |
| Adaptive forgetting / triage | Active forgetting; consolidation triage[19][18][21] | Utility score (recency, frequency, reward, consistency); consolidate high utility, decay/evict low utility | Yes | Yes | Yes | High |
| Pattern separation | Dentate gyrus orthogonality[24] | Orthogonalise overlapping episode representations to reduce interference | Yes | Yes | Yes | Moderate |
| One-shot plasticity | eCB-LTP; BTSP[27] | Salient brief events can write immediately with few exposures | Yes | Yes | Yes | Moderate (recent) |
| Exact Fisher EWC | — | Full Fisher matrix per task | No (task boundaries) | No | No (O(n²)) | High (computational) but not biological |
| Progressive networks | — | New sub-network per task | No | No | No (unbounded growth) | Low (biological plausibility) |
| Full generative replay | — | Deep generator replaying raw inputs | Offline | No | No (CPU-heavy) | Moderate (phenomenologically inspired) |
| Backprop through deep nets | — | Global gradient descent | No | No | No (not local) | High (engineering) but not biological |

### 9.2 Ranked recommendations for RAVANA

1. **Two-timescale consolidation per weight** (fast/slow). This is the single highest-value addition: it gives protection without freezing, is local and online, and has strong theoretical support for near-linear memory lifetime.[15][13][16]
2. **Local importance-weighted protection** (Synaptic-Intelligence-style online consolidation strength). Prevents the most-used weights from being overwritten while leaving others plastic.[14][5]
3. **Scalar salience/novelty write gate** driven by prediction error and novelty. Directly implements the neuromodulatory gating findings and is trivially cheap.[28][30][29]
4. **Small episodic buffer with prioritised replay** during the sleep phase, with priority = salience × novelty × expected relevance, interleaved with new learning. Avoids storing raw data; replay compressed traces or concept-graph states.[21][20][8]
5. **Schema-assimilation fast path**: if new information matches the existing typed concept graph, integrate immediately with low replay cost; otherwise treat as a novel episode.[1][23][35]
6. **Reconsolidation-style update window**: retrieval marks a fact labile for a bounded period so it can be revised rather than duplicated.[10][11]
7. **Utility-based triage/forgetting**: periodic consolidation of high-utility traces and decay/eviction of low-utility ones, using recency, frequency, reward relevance, and consistency.[19][18][21]

### 9.3 Well-established vs contested

**Well-established (strong, converging evidence):**
- Catastrophic interference in distributed networks and the stability–plasticity dilemma.[3][4][6]
- Two-system (hippocampal/neocortical) organisation and the need for slow interleaved cortical learning.[1][2]
- Replay occurs and is necessary for consolidation; replay is prioritised by salience/novelty/reward.[21][32][22]
- Neuromodulators (dopamine, acetylcholine, norepinephrine) gate plasticity and bias encoding.[28][29][30]
- Hebbian plasticity requires fast compensatory processes on multiple timescales; metaplasticity exists in vivo.[13][37]
- Forgetting has adaptive functions and is actively regulated.[19][18]
- Reconsolidation exists (especially in fear conditioning): retrieved memories become labile and re-stabilise.[10][11]

**Contested / qualified:**
- Whether replay's primary function is consolidation, planning, or valuation (likely all three, but their relative importance is debated).[20]
- Whether the hippocampus contributes only episodic memory or also supports generalisation (the strict version has been revised).[1]
- The precise molecular mechanism of synaptic tagging and capture, and how general it is beyond hippocampus/amygdala.[16]
- The relative contributions of decay, interference, and active suppression to forgetting.[36][19]
- Whether any single artificial continual-learning remedy is truly brain-like; the field remains fragmented across regularisation, architectural, and memory-based approaches.[3][25]
- Human reconsolidation and its clinical applicability.[12]

### 9.4 Open questions worth testing in RAVANA
- Does a two-timescale consolidation rule with a local importance trace actually prevent forgetting in the GRU decoder over long non-stationary streams, and what is the memory-lifetime scaling?
- Does a salience/novelty gate change the trade-off between new learning and old retention compared with a uniform learning rate?
- Does prioritised replay from a small compressed buffer match the benefit of full-data rehearsal at a fraction of the compute?
- Does a reconsolidation window reduce duplicate/conflicting facts compared with append-only storage?
- Does utility-based triage improve downstream task performance (decision quality) rather than merely reducing memory size?[19]

---

## Sources

[1] https://web.stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf — Kumaran, Hassabis & McClelland (2016) - What Learning Systems do Intelligent Agents Need? (CLS Updated)
[2] https://psycnet.apa.org/doiLanding?doi=10.1037/0033-295X.102.3.419 — McClelland, McNaughton and O'Reilly (1995) - Why there are complementary learning systems in the hippocampus and neocortex
[3] https://www.cs.uic.edu/~liub/lifelong-learning/continual-learning.pdf — Liu and Yang (2020) - Continual Learning and Catastrophic Forgetting (book chapter)
[4] https://en.wikipedia.org/wiki/Catastrophic_interference — Wikipedia - Catastrophic interference (McCloskey and Cohen 1989; Ratcliff 1990; French 1999)
[5] https://www.pnas.org/doi/10.1073/pnas.1611835114 — Kirkpatrick et al. (2017) - Overcoming catastrophic forgetting in neural networks (EWC)
[6] https://lead.ube.fr/wp-content/uploads/2023/09/000282-catastrophic-forgetting-in-connectionist-networks.pdf — French (1999) - Catastrophic forgetting in connectionist networks
[7] https://pmc.ncbi.nlm.nih.gov/articles/PMC9074752 — Hayes et al. (2021) - Replay in Deep Learning: Current Approaches and Missing Biological Elements
[8] https://pmc.ncbi.nlm.nih.gov/articles/PMC9674146 — Golden et al. (2022) - Sleep prevents catastrophic forgetting in spiking neural networks
[9] https://www.nature.com/articles/s41467-020-17866-2 — van de Ven et al. (2020) - Brain-inspired replay for continual learning with artificial neural networks
[10] https://pmc.ncbi.nlm.nih.gov/articles/PMC4588064 — Nader (2015) - Reconsolidation and the Dynamic Nature of Memory
[11] https://www.nature.com/articles/35021052 — Nader, Schafe and LeDoux (2000) - Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval
[12] https://pure-oai.bham.ac.uk/ws/portalfiles/portal/40975303/LeeNaderSchiller_TICS_JLedits_ds_jl.pdf — Lee, Nader and Schiller (2017) - An update on memory reconsolidation updating
[13] https://zenkelab.org/wp-content/uploads/preprints/2017-zenke_gerstner-preprint.pdf — Zenke and Gerstner (2017) - Hebbian plasticity requires compensatory processes on multiple timescales
[14] https://pmc.ncbi.nlm.nih.gov/articles/PMC6944509 — Zenke, Poole and Ganguli (2017) - Continual Learning Through Synaptic Intelligence
[15] https://www.gatsby.ucl.ac.uk/~pel/tnlectures/papers/benna_fusi.pdf — Benna and Fusi (2016) - Computational principles of synaptic memory consolidation
[16] https://pmc.ncbi.nlm.nih.gov/articles/PMC3705498 — Shires et al. (2012) - Synaptic tagging and capture in the living rat
[17] https://pmc.ncbi.nlm.nih.gov/articles/PMC4639700 — Yger and Gilson (2015) - Models of Metaplasticity: A Review of Concepts
[18] https://schapirolab.org/papers/Cowan_etal_2021_PsychBulletinReview.pdf — Cowan et al. (2021) - Memory consolidation as an adaptive process
[19] https://cbmm.mit.edu/sites/default/files/documents/Week3_RichardsFrankland2017.pdf — Richards and Frankland (2017) - The Persistence and Transience of Memory
[20] https://www.princeton.edu/~ndaw/md18.pdf — Mattar and Daw (2018) - Prioritized memory access explains planning and hippocampal replay
[21] https://www.nature.com/articles/s41467-023-43939-z — Huelin Gorriz, Takigawa and Bendor (2023) - The role of experience in prioritizing hippocampal replay
[22] https://pmc.ncbi.nlm.nih.gov/articles/PMC11185723 — Kleinman and Foster (2024) - Spatial localization of hippocampal replay requires dopamine signaling
[23] https://www.jneurosci.org/content/42/30/5916 — Sommer et al. (2022) - The Assimilation of Novel Information into Schemata and Its Efficient Consolidation
[24] https://pmc.ncbi.nlm.nih.gov/articles/PMC3789138 — Preston and Eichenbaum (2013) - Interplay of hippocampus and prefrontal cortex in memory
[25] https://pmc.ncbi.nlm.nih.gov/articles/PMC9676180 — Pisupati and Niv (2022) - The challenges of lifelong learning in biological and artificial systems
[26] https://onlinelibrary.wiley.com/doi/abs/10.1002/hipo.20350 — Teyler and Rudy (2007) - The hippocampal indexing theory and episodic memory: Updating the index
[27] https://www.nature.com/articles/s41593-026-02392-z — Piette et al. (2026) - Striatal endocannabinoids drive one-shot learning
[28] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3183413/pdf/nihms313421.pdf — Lisman, Grace and Duzel (2011) - A neoHebbian framework for episodic memory; role of dopamine-dependent late LTP
[29] https://pmc.ncbi.nlm.nih.gov/articles/PMC2659740/pdf/nihms95357.pdf — Hasselmo (2006) - The role of acetylcholine in learning and memory
[30] https://pubmed.ncbi.nlm.nih.gov/16254995 — Aston-Jones and Cohen (2005) - An integrative theory of locus coeruleus-norepinephrine function
[31] https://doi.org/10.1016/j.tics.2010.08.002 — Shohamy and Adcock (2010) - Dopamine and adaptive memory
[32] https://girardeaulab.org/wp-content/uploads/2018/08/Girardeau2011.pdf — Girardeau and Zugaro (2011) - Hippocampal ripples and memory consolidation
[33] https://sonar.ch/global/documents/22681 — Rasch and Born (2013) - About sleep's role in memory
[34] https://www.sciencedirect.com/science/article/pii/S1364661311001094 — Lewis and Durrant (2011) - Overlapping memory replay during sleep builds cognitive schemata
[35] https://doi.org/10.1016/j.tics.2017.04.013 — Gilboa and Marlatte (2017) - Neurobiology of schemas and schema-mediated memory
[36] https://cenl.ucsd.edu/Jclub/Wixted_2004.pdf — Wixted (2004) - The Psychology and Neuroscience of Forgetting
[37] https://www.pnas.org/doi/10.1073/pnas.181342098 — Heterosynaptic metaplasticity in the hippocampus in vivo: A BCM-like modifiable threshold for LTP (PNAS)
