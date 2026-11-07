# How the brain acquires and integrates knowledge from reading/text in real time
## Mechanisms, evidence, and implementable design principles for RAVANA

Scope: knowledge acquisition from *text* (reading / streaming), not language production. The production literature (Levelt, Dell, Kempen & Hoenkamp, Goldberg/Tomasello, Chang-Dell-Bock, Pickering & Garrod) is assumed known and is not re-reported. Every mechanism below is stated with its evidence type and a confidence rating, and the computational implication is written for a CPU-only, online, no-retraining system.

---

## 1. What is actually retained after reading: surface, textbase, situation model

### M1. Tripartite representation: surface → textbase → situation model
**What it is.** Readers do not store text as a copy. They construct three representation levels: (i) a *surface* code (exact wording/syntax), (ii) a *textbase* (the propositional semantic content of the text itself), and (iii) a *situation model* (the state of affairs described, elaborated with prior knowledge and integrated into the reader's knowledge). The situation model is what supports inference and problem-solving; the textbase supports reproduction.
**Evidence.** van Dijk & Kintsch (1983), *Strategies of Discourse Comprehension*, Academic Press — the source model.[1] Kintsch (1994), *American Psychologist*, 49(4), 294–303, states the distinction explicitly: "Remembering a text means that one can reproduce it... Learning from a text implies that one is able to use the information... in other ways."[53] Kintsch (1988), *Psychological Review*, 95(2), 163–182, gives the Construction-Integration (CI) model: construction is strictly bottom-up (words → propositions → inferences), then a spreading-activation integration phase settles the incoherent network into a stable structure; "to construct even a single proposition, an appropriate frame must be retrieved from one's store of knowledge, and its slots must be filled in the way indicated by the text."[2][53]
**Computational implication.** Store three separable structures with different lifetimes and functions: a short-lived verbatim/surface cache, a propositional textbase for the current document, and a persistent situation-model graph that is the actual knowledge store. Do **not** let the raw text become the knowledge representation.
**Confidence.** Strong. Human behavioural + computational (CI is a running simulation). The three-level dissociation is one of the most replicated findings in text comprehension.

### M2. Verbatim memory decays within seconds; gist persists
**What it is.** The exact wording of a sentence is retained only for the few seconds needed to comprehend it; once a semantic interpretation is made, the form is dropped and the meaning is stored.
**Evidence.** Sachs (1967), *Perception & Psychophysics*, 2(9), 437–442: after 80–160 syllables of interpolated discourse, recognition of syntactic/voice changes fell to near chance while semantic-change detection stayed high; "the original form of the sentence is stored only for the short time necessary for comprehension to occur."[5] Kintsch, Welsch, Schmalhofer & Zimny (1990), *Journal of Memory and Language*, 29(2), 133–159: across 0–4 day delays, "surface memory was found only on the immediate test. Memory for the text-base was quite strong initially, decreased with delay, but remained above zero even after four days. Situational memory... stayed at a high level, independent of delay."[6] Fuzzy-trace theory generalises this: verbatim and gist traces are stored in parallel and are dissociable; verbatim decays faster and is interference-sensitive, gist is long-lasting and interference-resistant.[7][8]
**Computational implication.** Give verbatim representations an explicit fast exponential decay (seconds-to-minutes half-life) and never treat them as knowledge. Invest storage budget in gist/propositional/situational structure. This directly answers "what should we replicate": a decaying surface buffer, not a text archive.
**Confidence.** Strong. Human behavioural (Sachs; Kintsch et al.) + theoretical (FTT). The *rate* is modality- and task-dependent, so treat the decay constant as a tunable parameter, not a fixed constant.

### M3. Propositions and macropropositions: the unit of knowledge is the idea, not the word
**What it is.** The textbase is a network of propositions (predicate + arguments). Macro-operators — deletion, generalization, construction — reduce the microstructure to a *macrostructure* of macropropositions, i.e. the gist. Frederiksen's propositional network formalism adds logical/causal/algebraic relations between propositions, not just semantic content.
**Evidence.** Kintsch & van Dijk (1978), *Psychological Review*, 85(5), 363–394: macro-operators "reduce the information in a text base to its gist — the theoretical macrostructure."[3] Frederiksen (1975), *Cognitive Psychology*, 7, 371–458: propositional networks with labelled binary relations between concepts, plus inter-propositional logical structure.[4] van Dijk (1979), *Discourse Processes*, 2, 113–126, introduces macropropositions as the topic-defining propositions of a discourse.[1]
**Computational implication.** Represent text as a typed predicate-argument graph with explicit inter-proposition links (causal, temporal, conditional). Build gist by applying deletion/generalization/construction operators over that graph — these are cheap graph operations, no gradient training required. This is the single most implementable idea in the classic literature.
**Confidence.** Strong for the representational claim (behavioural + computational). Moderate for the *online* claim: the classic work models comprehension after reading, not incremental construction under a streaming constraint; CI (M1) is the best-supported online variant.

### M4. Event indexing: the situation model is indexed on five dimensions
**What it is.** During comprehension, each incoming event is indexed on time, space, causation, intentionality/goal, and protagonist/entity. Integration cost and long-term link strength are functions of overlap on these dimensions; a break on any dimension triggers a new situation model.
**Evidence.** Zwaan, Langston & Graesser (1995) and Zwaan & Radvansky (1998), *Psychological Bulletin*, 123(2), 162–185: the event-indexing model; more situational overlap → easier clause processing and stronger long-term connections.[9] Zacks, Speer, Swallow, Braver & Reynolds (2007), *Psychological Bulletin*, 133(2), 273–293: event segmentation is automatic and hierarchically organised, and segmentation has consequences for memory and learning.[10]
**Computational implication.** Give every extracted event a 5-field index (time, space, cause, goal, entity). Use index overlap as the integration key: same index → merge into the current situation model; index break → close the episode and start a new one. This is exactly the "streaming boundary detection" a real-time system needs, and it is a deterministic hash/overlap computation.
**Confidence.** Strong for the behavioural phenomenon; moderate for the exact dimensional set (the model is explicitly provisional; Zwaan & Radvansky note it "is not... a complete model").

---

## 2. Schema theory: how new information is assimilated into existing structure

### M5. Encoding is schema-relative: slots, perspective, and prior knowledge
**What it is.** New information is not stored as a copy; it is matched to the slots of an activated schema and subsumed. What gets encoded at all depends on which schema is active: the same text yields different memories under different reading goals/perspectives.
**Evidence.** Bartlett (1932), *Remembering*: the "War of the Ghosts" serial-reproduction studies show levelling, sharpening and conventionalisation — unfamiliar elements are dropped or transformed to fit existing schemata; memory is reconstructive, not reproductive.[11] Bransford & Johnson (1972), *Journal of Verbal Learning and Verbal Behavior*, 11(6), 717–726: supplying an appropriate context *before* a passage markedly improved comprehension and recall; supplying the same information *after* the passage did not — prior knowledge is a prerequisite for encoding, not a retrieval aid.[12] Pichert & Anderson (1977), *Journal of Educational Psychology*, 69(4), 309–315: readers given a burglar vs homebuyer perspective recalled different idea units from the same text, and shifting perspective after reading recovered previously unrecallable information — "alternative high-level schemata can provide frameworks for assimilating a text, perhaps by providing 'slots' for different types of information."[13]
Rumelhart (1980), "Schemata: The Building Blocks of Cognition," in *Theoretical Issues in Reading Comprehension*: schemata are knowledge structures with variables/slots that are instantiated by input.[14] Chi, Feltovich & Glaser (1981), *Cognitive Science*, 5(2), 121–152: experts categorise physics problems by deep principle, novices by literal surface features — the knowledge structure determines the representation.[15]
**Computational implication.** Knowledge must be stored as typed schemas with named slots, not as flat facts. At ingest time, activate the best-matching schema and fill slots; facts that match no slot are either (a) attached as exceptional instances or (b) buffered for schema revision (see M6). Carry a goal/perspective tag into the encoding step, because it changes what is stored. This is cheap: schema matching is retrieval + slot binding.
**Confidence.** Strong (behavioural). The slot-filling mechanism is well supported; the *restructuring* side (how a schema changes when slots fail) is much weaker — see the "no clean evidence" section.

### M6. Schema-consistent information is learned rapidly; inconsistent information needs interleaving
**What it is.** When new information is congruent with an existing schema, it can be assimilated rapidly (even in one trial) and consolidated quickly; when it is inconsistent, rapid learning causes catastrophic interference and it must be learned slowly and interleaved with related known material.
**Evidence.** Tse et al. (2011), *Science*, 333, 891–895: schema-dependent gene activation in neocortex; rats with an established flavour-location schema learned new associations in a single trial and retained them, and retrieval became hippocampally independent within ~24 h.[18] McClelland (2013), *Journal of Experimental Psychology: General*, 142(4), 1190–1210: simulations extending McClelland et al. (1995) show that schema-*consistent* information is learned rapidly without interference, while schema-*inconsistent* information produces catastrophic interference if learned too fast; "complementary learning systems and interleaved learning are necessary... if that information is not consistent with the structure of known information."[22] van Kesteren, Ruiter, Fernández & Henson (2012), *Trends in Neurosciences*, 35(4), 211–219: schema-congruent information is usually better remembered; MTL–mPFC interactions mediate the congruency-dependent efficiency.[16] Gilboa & Marlatte (2017), *Trends in Cognitive Sciences*, 21(8), 618–631: vmPFC–hippocampus interactions in schema-mediated encoding and consolidation.[17]
**Computational implication.** Implement a **consistency gate** at write time: if the new fact's predicted slots/relations are already present in the graph, merge immediately (fast path). If it conflicts or matches no schema, route it to a replay buffer for slow, interleaved assimilation (slow path). This is the single most important design rule for a no-retraining system, because it is the brain's actual solution to catastrophic interference.
**Confidence.** Strong for the behavioural/neurobiological phenomenon (Tse; van Kesteren; McClelland simulations). Moderate for the exact threshold of "consistency" — the brain's criterion is not specified, and the human data are mostly item-level (flavour-location, word pairs), not text-level.

---

## 3. Integrating a new fact without corrupting, duplicating, or contradicting the graph

### M7. Two learning systems: fast episodic buffer + slow structured semantic store
**What it is.** Effective learning requires two complementary systems: a fast system that learns specifics of individual experiences (hippocampus; sparse, pattern-separated, interference-robust) and a slow system that gradually acquires structured knowledge (neocortex; distributed, overlapping, statistical). Replay of the fast store interleaves new experiences with old, allowing gradual integration without distortion.
**Evidence.** McClelland, McNaughton & O'Reilly (1995), *Psychological Review*, 102(3), 419–457: neocortical synapses change a little on each reinstatement; the hippocampus permits rapid learning of new items "without disrupting this structure."[21] Kumaran, Hassabis & McClelland (2016), *Trends in Cognitive Sciences*, 20(7), 512–534: updates CLS with goal-dependent replay weighting and rapid neocortical learning for schema-consistent information.[20]
**Computational implication.** Architecturally separate a fast, content-addressed episodic buffer (new ingest, exact provenance, short retention) from a slow semantic graph (structured, merged, long retention). Integration happens by replay: reactivate the new item together with its neighbours in the semantic graph and nudge the graph, rather than by direct overwrite. This is implementable on a CPU as a two-tier store with a replay scheduler.
**Confidence.** Strong (neurobiological + computational). The two-system framework is the consensus account of why interleaving works.

### M8. Catastrophic interference is the failure mode to design against
**What it is.** In a single distributed weight store, sequential learning of new patterns can catastrophically erase old ones; the greater the new learning, the greater the disruption.
**Evidence.** McCloskey & Cohen (1989), *Psychology of Learning and Motivation*, 24, 109–165: after learning A-B then A-C associations, the network's A-B knowledge dropped to ~1% within 5 trials of A-C and was gone after 15; interference is catastrophic only under sequential, not concurrent, training.[29]
**Computational implication.** Do not use a single monolithic embedding table that is overwritten on every ingest. Use append-only / sparse-update structures (proposition graph with node merging), and reserve dense parameter changes for an offline consolidation phase. This is a hard constraint for RAVANA: a decoder that retrains on every page will destroy prior knowledge.
**Confidence.** Strong (computational). Directly demonstrated in connectionist networks; the brain avoids it via M7/M6.

### M9. Reconsolidation: retrieval opens a limited window in which a memory can be updated
**What it is.** Reactivating a consolidated memory returns it to a labile state requiring restabilisation; during that window the memory can be strengthened, weakened, or *updated with new information*. This is the brain's mechanism for editing an existing memory without re-learning it from scratch.
**Evidence.** Nader, Schafe & LeDoux (2000), *Nature*, 406, 722–726: anisomycin infused into the amygdala shortly after reactivation of a consolidated fear memory produced amnesia; the same treatment without reactivation left memory intact; the effect was time-limited (~6 h).[23] Nader & Hardt (2009), *Nature Reviews Neuroscience*, 10, 224–234: consolidation and reconsolidation rest on the same operational definitions; boundary conditions inform rather than invalidate the phenomenon.[24] Schiller et al. (2010), *Nature*, 463, 49–53: in humans, extinction training delivered during the reconsolidation window updated the fear memory; the effect lasted ≥1 year and was specific to the reactivated memory.[25] Hupbach, Gomez, Hardt & Nadel (2007), *Learning & Memory*, 14(1–2), 47–53: a reminder before new learning caused new items to intrude into recall of the old list; the effect was time-dependent and "reconsolidation is also a constructive process, one that supports the incorporation of new information."[26]
**Computational implication.** Model every stored fact as having a *labile window* after it is retrieved/replayed. Updates (corrections, refinements, contradictions) are applied only inside that window; outside it, the fact is effectively read-only and a conflicting new fact must spawn a new versioned node. This gives a principled, biologically grounded rule for mutable knowledge without global retraining.
**Confidence.** Moderate. The rodent fear-conditioning evidence is strong and replicated; the human evidence is contested. Elsey, Van Ast & Kindt (2018), *Psychological Bulletin*, 144(8), 797–848: "there are a number of inconsistent findings, and the presence of alternative explanations means that we cannot conclusively infer the presence of reconsolidation at the neurobiological level from current evidence. Reconsolidation remains a viable but hotly contested explanation."[27] Treat the *window* mechanism as a design heuristic with moderate confidence, not as established fact.

### M10. Contradiction handling: belief revision (AGM) and entrenchment
**What it is.** When a new belief contradicts existing beliefs, the belief set must be revised by removing some old beliefs so the set remains consistent. AGM provides the postulates: revision is consistent, inclusive, and minimal-ish; the choice of what to give up is governed by an entrenchment ordering (how dispensable each belief is).
**Evidence.** Alchourrón, Gärdenfors & Makinson (1985), *Journal of Symbolic Logic*, 50(2), 510–530: the AGM framework — contraction, expansion, revision; revision adds a sentence and removes others "if this is needed to ensure that the resulting belief set is consistent"; the Levi identity defines revision as contraction by ¬φ followed by expansion by φ.[19]
**Computational implication.** Every node in the knowledge graph carries an entrenchment score (support count, source reliability, recency, schema centrality). On contradiction, apply an AGM-style revision: keep the new fact if its support exceeds the entrenchment of the old, demote/remove the loser, and record the revision event. This is a deterministic graph operation — no retraining. It also gives you a natural trust model (M12).
**Confidence.** Strong for the formal framework (computational/logical); moderate for the claim that brains implement AGM-like revision. AGM is a rationality benchmark, not a neural model; human belief revision is known to be biased and identity-protective. Use AGM as the *engineering* specification, not as a claim about human cognition.

### M11. Competition and suppression: retrieval-induced forgetting
**What it is.** Retrieving some items actively suppresses related competitors sharing the same retrieval cue, making them less accessible later. Forgetting is partly an inhibitory, competitive process, not just passive decay.
**Evidence.** Anderson, Bjork & Bjork (1994), *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 20(5), 1063–1087: practicing retrieval of half the members of a category impaired delayed recall of the unpractised members of that category; the impairment survived controls for output interference and endured ≥20 min; it was restricted to high-frequency members.[28]
**Computational implication.** Use this as a *deduplication and disambiguation* mechanism: when two nodes compete for the same cue (e.g. two entities with similar names, or two conflicting values for the same attribute), retrieving one should suppress the other. Implement as a per-cue activation with lateral inhibition. This is how you avoid duplicate nodes accumulating for the same concept — repeated co-activation of the winner strengthens it and suppresses the near-duplicate.
**Confidence.** Strong for the behavioural effect; moderate for the inhibitory mechanism (a retrieval-failure/blocking account also fits; the inhibitory account is well supported but not universally accepted).

### M12. Source/provenance is stored and used for trust
**What it is.** People do not retrieve an abstract source tag; they make source attributions during remembering by evaluating the qualitative features of the memory record (perceptual detail, contextual/spatiotemporal information, cognitive operations) against flexible criteria. Reality monitoring distinguishes externally perceived from internally generated information on the same feature dimensions.
**Evidence.** Johnson & Raye (1981), *Psychological Review*, 88(1), 67–85: reality monitoring; memories of perceived events carry more spatial/temporal/sensory detail, internally generated memories carry more cognitive-operation records.[42] Johnson, Hashtroudi & Lindsay (1993), *Psychological Bulletin*, 114(1), 3–28: the source-monitoring framework; "people do not typically directly retrieve an abstract tag or label that specifies a memory's source, rather, activated memory records are evaluated and attributed to particular sources through decision processes performed during remembering"; source memory is critical for evaluating the veridicality of a "fact" (tabloid vs Consumer Reports example).[41]
Neuroscience: source memory engages prefrontal, medial temporal and parietal regions; left lateral PFC is consistently more active for source than old/new judgements, and the anterior PFC is central to reality monitoring.[44][43]
**Computational implication.** Every knowledge node must carry provenance metadata: source document, source type/reliability, ingest time, and the encoding operations used. Do not store a literal "trust tag"; instead compute trust at query time from provenance features (source reliability, corroboration count, recency, consistency with the schema) — this mirrors the attribution process and lets trust be context-dependent. Also flag internally generated inferences (deductions) as distinct from externally read facts, because the brain treats these differently (reality monitoring).
**Confidence.** Strong (behavioural + neurobiological). The framework is well established; the exact decision criteria are flexible and error-prone, so treat trust as a computed estimate.

---

## 4. Retrieval practice and spacing: why one-pass ingestion is not enough

### M13. Retrieval practice beats re-reading for long-term retention
**What it is.** Once an item can be recalled, repeated *studying* adds essentially nothing, while repeated *retrieval* produces large long-term gains. The act of retrieval is itself a learning event.
**Evidence.** Karpicke & Roediger (2008), *Science*, 319(5865), 966–968: after one successful recall, repeated study produced no measurable learning a week later, while repeated testing produced ~80% recall vs ~36% in the drop-from-testing conditions (d = 4.03); ~80 extra study trials produced practically no gain.[32] Roediger & Karpicke (2006), *Psychological Science*, 17(3), 249–255: with prose passages, immediate recall favoured re-reading (81% vs 75% of idea units), but after one week retrieval practice won (56% vs 42%); "retrieval practice... enhances long-term retention of information by slowing the rate of forgetting."[33] Karpicke & Blunt (2011), *Science*, 331(6018), 772–775: retrieval practice beat elaborative studying with concept mapping on inference questions (d = 1.50), and concept mapping was not significantly better than extra reading.[34]
**Computational implication.** A system that ingests a web page once and never revisits it is replicating the *worst* condition in this literature. Schedule self-retrieval: after ingest, generate queries from the stored propositions and attempt to answer them from the graph alone (no source text present), with feedback from the source only on failure. This is the online, CPU-feasible version of retrieval practice — it costs query time, not retraining. Note the interaction with M12: retrieval also suppresses competitors (M11), so it doubles as deduplication.
**Confidence.** Strong (human behavioural, many replications). The effect is robust across materials and delays.

### M14. Spacing and interleaving
**What it is.** Distributing practice over time (spacing) and mixing topics (interleaving) improves long-term retention and transfer relative to massed, blocked practice, even when total study time is equated. The optimal inter-study interval grows with the retention interval.
**Evidence.** Cepeda, Pashler, Vul, Wixted & Rohrer (2006), *Psychological Bulletin*, 132(3), 354–380: meta-analysis of 839 assessments in 317 experiments; ISI and retention interval operate jointly — the ISI producing maximal retention increases as retention interval increases.[35] Cepeda et al. (2008), *Psychonomic Bulletin & Review*: across ~1,354 participants and delays up to 1 year, the optimal gap declined from ~20–40% of a 1-week test delay to ~5–10% of a 1-year test delay.[36] Bjork & Bjork (2011), in *Psychology and the Real World*: desirable difficulties — spacing, interleaving, variation, and testing — "trigger encoding and retrieval processes that support learning, comprehension, and remembering."[37]
**Computational implication.** Maintain a rehearsal schedule with expanding intervals (e.g. review a fact after ~5–10% of its intended retention horizon), and interleave replay of unrelated topics rather than blocked re-ingest. This is the computational counterpart of M7's replay and is cheap: a priority queue over nodes with a due-time.
**Confidence.** Strong (behavioural, meta-analytic). The precise optimal schedule is parameterised by retention horizon, so treat the numbers as a starting prior.

### M15. Contested nuance: is re-reading really useless?
**What it is.** The strong claim "restudying never helps" depends on the design. When spacing is controlled within subjects, restudying *does* improve learning, and learners' awareness of the relative benefits is enhanced.
**Evidence.** Soderstrom, Kerr & Bjork (2016), *Psychological Science*, 23(10), 1326–1333: Experiment 1 replicated Karpicke & Roediger (2008); Experiment 2, controlling spacing within subjects, found both repeated testing and restudying improved learning, and the apparent ineffectiveness of restudying in the between-subjects design was attributable to spacing differences.[38]
**Computational implication.** Do not conclude "never re-read." The correct rule is: *re-reading without retrieval is weak; re-reading plus retrieval is strong.* For RAVANA, a cheap re-scan of a source is fine as long as it is paired with a retrieval attempt; pure passive re-ingest should be deprioritised.
**Confidence.** Moderate. This is a real boundary condition on M13; the between-subjects vs within-subjects discrepancy is well documented.

---

## 5. Elaborative encoding: what makes an encoding durable

### M16. Levels of processing / elaboration
**What it is.** The memory trace is a by-product of perceptual analysis; trace persistence is a positive function of the depth and *elaboration* of processing. Deeper (semantic) processing yields more durable traces than shallow (structural/phonemic) processing, and it is the qualitative nature of the encoding operation — not time spent — that matters.
**Evidence.** Craik & Lockhart (1972), *Journal of Verbal Learning and Verbal Behavior*, 11(6), 671–684: "the memory trace can be understood as a by-product of perceptual analysis and... trace persistence is a positive function of the depth to which the stimulus has been analyzed."[39] Craik & Tulving (1975), *Journal of Experimental Psychology: General*, 104(3), 268–294: ten experiments; semantic encoding questions produced higher retention than physical/phonemic ones; "it is the qualitative nature of a word's encoding which determines retention, not processing time as such"; results are better described by "the degree of elaboration of the encoded trace."[40]
**Computational implication.** Durability should be a function of how much the ingest *elaborated* the item — how many existing graph connections were activated and integrated, whether inferences were generated, whether the item was related to prior knowledge. Compute an "elaboration score" per proposition (number of activated neighbours, depth of integration) and use it as the prior for the node's retention weight. This is a cheap, deterministic computation and gives you a principled forgetting/retention schedule without a learned decay parameter.
**Confidence.** Strong (behavioural). The framework is a *framework*, not a theory (Tulving's own characterisation), and "depth" is descriptive; the operational content is elaboration + distinctiveness, which is what the computational rule above uses.

---

## 6. Sleep and gist abstraction: offline extraction of structure

### M17. Sleep selectively consolidates memories that are relevant, and transforms representations
**What it is.** Sleep is not passive protection from interference; it is an active systems-consolidation process. Memories are reactivated during slow-wave sleep, transformed, and redistributed to neocortical long-term stores; REM sleep may stabilise the transformed memories. Consolidation is *selective*: it preferentially benefits memories expected to be needed.
**Evidence.** Wilhelm, Diekelmann, Molzow, Ayoub, Mölle & Born (2011), *Journal of Neuroscience*, 31(5), 1563–1569: post-learning sleep improved delayed retrieval only when subjects expected a retrieval test; the effect correlated with slow-oscillation activity and spindle count during SWS.[45] Born & Wilhelm (2012), *Psychological Research*, 76(2), 192–203: sleep consolidation is selective and produces qualitative changes in representations, including facilitating extraction of explicit knowledge from implicitly learned material.[46] Rasch & Born (2013), *Physiological Reviews*, 93(2), 681–766: comprehensive review; waking brain is optimised for encoding, sleeping brain for consolidation; SWS reactivation transforms representations for long-term integration.[48]
**Computational implication.** Have an explicit *offline consolidation phase* (not tied to a biological sleep cycle, but a scheduled low-priority background pass). Its job is to replay recent episodic items against the semantic graph, merge consistent ones, and extract shared structure. Make consolidation *selective*: weight replay by expected future relevance (query frequency, goal relevance, recent use), not uniformly.
**Confidence.** Strong for selective consolidation and SWS dependence (human behavioural + neurophysiological). Moderate for the claim that sleep is *necessary* for gist abstraction — see M18.

### M18. Overlapping replay builds schemata; but the abstraction evidence is contested
**What it is.** The proposed mechanism for schema formation: related memories are replayed in overlapping combinations during sleep; the shared elements are selectively strengthened, which progressively builds abstract schematic representations. This is the sleep-based route from episodes to gist.
**Evidence.** Lewis & Durrant (2011), *Trends in Cognitive Sciences*, 15(8), 343–351: "the overlapping replay of related memories selectively strengthens shared elements. Repeated reactivation of memories in different combinations progressively builds schematic representations... This selective strengthening forms the basis of cognitive abstraction."[47] Hennies, Lambon Ralph, Kempkes, Cousins & Lewis (2016), *Journal of Neuroscience*, 36(13), 3799–3810: sleep spindle density predicted the growth of the schema-related memory benefit across 24 h and predicted hippocampal disengagement for schema-related memories — first evidence that sleep is associated with the *interaction* between prior knowledge and long-term memory formation.[49] **Contested:** Lerner & Gluck (2019), *Sleep Medicine Reviews*, 47, 39–50: systematic review of sleep and extraction of hidden regularities — "many null results have been published as well, with no clear consensus emerging regarding the conditions that yield this sleep effect"; sleep facilitates *implicit* detection of regularities more reliably than *explicit* recognition of them, and some claimed abstraction effects can be explained by local memory strengthening.[50]
**Computational implication.** Implement an offline *abstraction pass*: take sets of recently replayed episodes, compute their shared propositional subgraphs (set intersection over activated nodes/relations), and promote the shared structure into a macro-node or schema. This is a graph operation, not learning. But treat it as a heuristic with moderate confidence: the human evidence that sleep is *required* for abstraction is weak and contested, so the same algorithm run online may capture most of the benefit.
**Confidence.** Moderate for the mechanism (Lewis & Durrant is a theoretical proposal); weak-to-moderate for the empirical claim that sleep is necessary for abstraction (Lerner & Gluck). The safest reading: overlap-based abstraction is a plausible and implementable mechanism, but its dependence on a sleep-like offline state is not established.

---

## 7. Directly relevant computational precedents (not human evidence)

### M19. Latent semantic analysis: knowledge induction from co-occurrence
**What it is.** Global semantic knowledge can be induced from local co-occurrence statistics over a large text corpus, without any prior linguistic knowledge, by extracting the right number of dimensions (~300). LSA learned English vocabulary at a rate comparable to schoolchildren, mostly by indirect inference.
**Evidence.** Landauer & Dumais (1997), *Psychological Review*, 104(2), 211–240: "By inducing global knowledge indirectly from local co-occurrence data in a large body of representative text, LSA acquired knowledge about the full vocabulary of English at a comparable rate to schoolchildren... most of its acquired knowledge was attributable to indirect inference rather than direct co-occurrence relations."[51]
**Computational implication.** Distributional co-occurrence is a viable *unsupervised* way to build the semantic neighbourhood structure that schema matching and elaboration scoring need. For a CPU budget, a sparse co-occurrence matrix with incremental updates (rather than a full SVD) is the obvious implementation; it also gives a cheap similarity metric for entity resolution and duplicate detection. This is computational evidence, not human evidence, but it demonstrates the feasibility of the core acquisition loop.

### M20. Continual/lifelong knowledge-graph embedding without retraining
**What it is.** A growing body of work updates KG embeddings incrementally as new facts arrive, using knowledge distillation and regularisation to avoid catastrophic forgetting, explicitly without retraining from scratch.
**Evidence.** Lifelong embedding learning for growing KGs (2022): a masked KG autoencoder with embedding transfer and regularisation, "learning new facts while retaining old knowledge without re-training from scratch."[30] Continual KGE by structural feature distillation: "dynamically updates embeddings without full retraining... effectively mitigates catastrophic forgetting."[30]
**Computational implication.** Confirms that incremental, no-retraining knowledge-graph update is an solved-ish engineering problem. The transferable techniques: (i) initialise new entities from their structural neighbours (so new facts inherit existing geometry), (ii) replay old facts interleaved with new, (iii) regularise against drift of old embeddings. On a CPU, the graph-native version of these (slot-filling from neighbours, replay, node-level regularisation) is cheaper than embedding-space distillation.
**Confidence.** Moderate (computational; benchmark-driven, not yet a settled theory).

### M21. Locate-then-edit: rank-one updates to a single fact
**What it is.** Factual associations in transformers are localised and directly editable: a single rank-one weight change can insert, modify, or delete one factual association while preserving neighbours.
**Evidence.** Meng, Bau, Andonian & Belinkov (2022), *NeurIPS 35*: ROME — causal mediation to locate the feed-forward module mediating a factual prediction, then a rank-one edit; ROME maintained specificity and generalisation on counterfactual assertions where other methods sacrificed one or the other.[31]
**Computational implication.** The general principle — *localise the fact, then apply a minimal, constrained edit* — is exactly the right pattern for RAVANA's no-retraining constraint, and is implementation-agnostic (it does not require a transformer). In a graph store, the analogue is: locate the target node/edge by retrieval, apply a bounded update to that node and its immediate neighbourhood only, and verify that neighbours are unchanged. This operationalises M9's labile-window update safely.
**Confidence.** Moderate (computational; demonstrated in large transformers, not in brain-like systems).

---

## 8. Synthesis: the implementable mechanism stack for RAVANA

Ranked by (evidence strength × CPU-feasibility × relevance to the no-retraining constraint):

1. **Propositional/event graph with typed slots and 5-field event index** (M1, M3, M4) — the core store. Deterministic, incremental, no training.
2. **Consistency-gated write** (M6, M7, M8) — fast merge if schema-consistent; buffer-and-replay if not. This is the brain's actual anti-catastrophic-interference rule.
3. **Two-tier store: fast episodic buffer + slow semantic graph, with scheduled replay** (M7, M14) — replay interleaves new and old; expanding intervals.
4. **Retrieval practice as a first-class operation** (M13, M15) — self-query from the graph after ingest; retrieval beats re-reading, but re-reading is not worthless when paired with retrieval.
5. **AGM-style contradiction revision with entrenchment scores** (M10) — bounded, local, deterministic; gives both consistency and a trust ordering.
6. **Provenance and computed trust on every node** (M12) — source, source type, corroboration, recency; distinguish inferred from read.
7. **Elaboration score as the retention prior** (M16) — durability ∝ integration into existing graph structure.
8. **Verbatim surface cache with fast decay** (M2) — explicitly short-lived; never the knowledge store.
9. **Offline overlap-based abstraction pass** (M17, M18) — promote shared subgraphs to macro-nodes; selective by expected relevance. Medium confidence.
10. **Lateral inhibition / competitive retrieval for deduplication** (M11) — suppress near-duplicate competitors on the same cue.
11. **Sparse distributional co-occurrence layer for similarity/entity resolution** (M19) — incremental, CPU-cheap.
12. **Locate-then-edit bounded updates** (M21) — never global overwrite; update one node + neighbourhood and verify.

### Where the human evidence is weak or absent
- **No clean evidence** that the brain performs explicit logical consistency checking during reading. AGM is a rationality benchmark; human belief revision is biased and identity-protective. Treat consistency maintenance as an *engineering* requirement, not a discovered neural mechanism.
- **No clean evidence** for the mechanism of *schema restructuring/accommodation* (as opposed to slot-filling). Bartlett's distortions show that memory is reconstructive, but the computational process by which a schema itself changes is underspecified in the classic literature.
- **No clean evidence** that sleep is *necessary* for gist abstraction. The overlap-replay mechanism is a well-argued theory (Lewis & Durrant 2011) with supportive correlational evidence (Hennies 2016), but systematic reviews report many null results and no consensus (Lerner & Gluck 2019). The abstraction algorithm is worth implementing; the sleep-dependence is not established.
- **No clean evidence** for human *fact deduplication* as a distinct mechanism. Retrieval-induced forgetting (M11) is the closest well-established analogue, but it is about competition at retrieval, not about merging duplicate records in a store.
- **No clean evidence** for the *time constants* of reconsolidation windows in humans. The 4–6 h window is from rodent fear conditioning; human declarative reconsolidation is contested (Elsey 2018). Use a tunable window, not a fixed biological constant.
- **No clean evidence** that the brain stores propositional graphs in the Kintsch/Frederiksen sense. These are *computational* representational hypotheses that fit behavioural data; the neural implementation is not established. The value for RAVANA is as an engineering representation, not as a claim about neural data structure.

---

## Sources

[1] https://muse.jhu.edu/article/452373/summary
[2] https://psycnet.apa.org/doiLanding?doi=10.1037/0033-295X.95.2.163
[3] https://psycnet.apa.org/record/1979-22783-001
[4] https://files.eric.ed.gov/fulltext/ED179945.pdf
[5] https://link.springer.com/article/10.3758/BF03208784
[6] https://epub.uni-regensburg.de/13636/1/ubr06085_ocr.pdf
[7] https://journals.sagepub.com/doi/full/10.1177/2372732218797143
[8] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4268540
[9] https://cs.uky.edu/~sgware/reading/papers/zwaan1998situation.pdf
[10] https://pmc.ncbi.nlm.nih.gov/articles/PMC2852534
[11] https://bpspsychub.onlinelibrary.wiley.com/doi/pdfdirect/10.1111/bjop.12449?download=true
[12] https://memlab.yale.edu/sites/default/files/files/1972_Bransford_Johnson_JVLVB.pdf
[13] https://psycnet.apa.org/record/1978-08708-001
[14] https://catalog.lib.kyushu-u.ac.jp/opac_download_md/1546597/p113.pdf
[15] https://doi.org/10.1207/s15516709cog0502_2
[16] https://psycnet.apa.org/record/2012-09436-003
[17] https://psycnet.apa.org/record/2017-31703-010
[18] https://pmc.ncbi.nlm.nih.gov/articles/PMC3789138
[19] https://plato.stanford.edu/archives/Spr2012/entries/logic-belief-revision
[20] https://web.stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf
[21] https://psycnet.apa.org/doiLanding?doi=10.1037/0033-295X.102.3.419
[22] https://stanford.edu/~jlmcc/papers/McClellandOF_JEPGRapidNeocorticalLearning.pdf
[23] https://www.nature.com/articles/35021052
[24] https://www.nature.com/articles/nrn2590
[25] https://www.nature.com/articles/nature08637
[26] https://www.pure.ed.ac.uk/ws/files/13950218/Reconsolidation_of_episodic_memories.pdf
[27] https://psycnet.apa.org/record/2018-24701-001
[28] https://pubmed.ncbi.nlm.nih.gov/7931095
[29] https://experts.illinois.edu/en/publications/catastrophic-interference-in-connectionist-networks-the-sequentia
[30] https://arxiv.org/html/2211.15845v2
[31] https://papers.nips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html
[32] https://learninglab.psych.purdue.edu/downloads/2008/2008_Karpicke_Roediger_Science.pdf
[33] https://pmc.ncbi.nlm.nih.gov/articles/PMC6353106
[34] https://learninglab.psych.purdue.edu/downloads/2011/2011_Karpicke_Blunt_Science.pdf
[35] https://psycnet.apa.org/doiLanding?doi=10.1037/0033-2909.132.3.354
[36] https://wixtedlab.ucsd.edu/publications/wixted/Cepeda_Vul_Rohrer_Wixted_Pashler.pdf
[37] https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/04/EBjork_RBjork_2011.pdf
[38] https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/11/soderstorm_kerr.pdf
[39] https://web.stanford.edu/class/psych205/papers/CraikLockhart72.pdf
[40] https://psycnet.apa.org/record/1976-00185-001
[41] https://memlab.yale.edu/sites/default/files/files/1993_Johnson_Hashtroudi_Lindsay_PsychBull.pdf
[42] https://memlab.yale.edu/sites/default/files/files/1981_Johnson_Raye_PsychRev.pdf
[43] https://www.memlab.psychol.cam.ac.uk/pubs/Simons2017%20TICS.pdf
[44] https://pmc.ncbi.nlm.nih.gov/articles/PMC2859897
[45] https://pubmed.ncbi.nlm.nih.gov/21289163
[46] https://wixtedlab.ucsd.edu/publications/Psych%20218/Born_Wilhelm_2012.pdf
[47] https://pubmed.ncbi.nlm.nih.gov/21764357
[48] https://pubmed.ncbi.nlm.nih.gov/23589831
[49] https://orca.cardiff.ac.uk/id/eprint/91981
[50] https://pmc.ncbi.nlm.nih.gov/articles/PMC6779511/pdf/nihms-1531125.pdf
[51] https://psycnet.apa.org/record/1997-03612-001
[53] https://andymatuschak.org/files/papers/Kintsch%20-%201994%20-%20Text%20comprehension,%20memory,%20and%20learning.pdf
