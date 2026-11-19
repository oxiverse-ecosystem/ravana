# Growing causal substrate from sparse experience: brain mechanism and RAVANA recommendation

**Status:** research + design recommendation. **No code changed.** No engine was run for this
document and no instrument was edited. The only file produced is this one.

**Scope.** The THINKING front scores 19/100 offline and 41/100 web at n=27. The *production*
mechanism is known to work when trusted chains exist (`enumerate_chains` → `grade_chain` →
`assertable_chain` in `learn/explanation.py`); the deficit is *acquisition*: only two
subjects (`sun→light→photosynthesis`, `water vapour→clouds→rain`) have a verified 2-hop
trusted causal chain. This document follows the user's standing rule — research the
brain first, cite author/year/journal, then derive the mechanism — and the
`neuroscience-to-architecture` discipline: split acquisition from production, grade the
evidence, map mechanism → existing module → gap, and state what the evidence does not
support.

Evidence grades: **[S]** strong (convergent behavioural + neuroimaging/lesion, or
formal model with repeated behavioural support), **[M]** moderate, **[W]** weak/contested.
Evidence types are labelled **beh** (human behavioural), **neuro** (human or animal
neurobiology), **comp** (computational/rational model). A [comp] result is a claim about
optimal inference, *not* directly about how a brain implements it.

---

## 0. Headline recommendation

**Use verified chain schemas as hypothesis generators and search targets, never as
evidence; make the two-source convergence gate apply to the causal EDGE, not only to the
node; and let a felt, specific gap decide where to look.**

Three findings carry the recommendation:

1. The brain solves sparse causal induction with **domain-general inference constrained
   by prior knowledge** — an abstract causal theory supplies the ontology, link
   plausibility, and functional form that make sparse data sufficient (Griffiths &
   Tenenbaum 2009). Unconstrained covariation does not.
2. A **schema is what makes one-trial learning stick** (Tse et al. 2007), *and* the same
   schema is a principal source of **false memories** because schema-consistent
   inferences get misattributed to an external source (Johnson 1997; Gilboa & Marlatte
   2017). The resolution is not "trust schemas" or "distrust schemas"; it is
   **schema → provisional hypothesis → corroborated edge**.
3. Exploration is driven by a **specific, felt information gap** (Loewenstein 1994) and
   by **expected information gain about hidden causes** (Friston et al. 2015), and the
   curious state itself potentiates encoding (Gruber et al. 2014). RAVANA already has a
   gap signal (`WebToGraph.knowledge_gap().efe`); it is currently a *topic*-level gap,
   not a *missing-causal-relation* gap.

**The single highest-leverage code-level gap** (verified by reading the source, not
inferred): in `web/web_to_graph.py`, `_ensure_node` promotes a NODE only after k distinct
`source_url`s (`_promote_min_sources`, lines 149, 174–198), but `learn_text` stamps a new
causal EDGE with `edge_kind="web_fact"` on **first sighting** (lines 285–296). Since
`provenance_trusted` accepts `web` (`learn/explanation.py:315-316`), one web sentence is
already enough to make a hop assertable. That is one-trial consolidation *without* a
schema and *without* source convergence — precisely the condition under which Tse et al.
and Johnson et al. predict fragility and confabulation.

---

## 1. Research question 1 — How humans build causal models from sparse observation

### 1.1 Causal-Bayesian inference: structure is inferred, not accumulated

The formal account is **structural causal models** and the `do`-calculus: a causal claim
is a directed structure with an intervention operator, and counterfactual dependence is
defined by changing one variable and propagating consequences (Pearl 2009, *Causality*,
2nd ed., Cambridge University Press; **comp**, [S] as a formalism, not a brain claim).
`P(outcome | do(X))` is not recoverable from observational co-occurrence alone; the
distinction is psychological fact, not just formal convenience. Sloman and Lagnado (2005,
"Do we 'do'?", *Cognitive Science* 29(1):5–39; **beh**, [S]) showed that people distinguish
observation from intervention, and Waldmann and Hagmayer (2005, "Seeing versus doing",
*JEP: Learning, Memory, and Cognition* 31(2):216–227; **beh**, [S]) showed the same for
accessing causal knowledge. Sloman (1996, *Psychological Bulletin* 119(1):3–22; **beh**,
[S], working distinction) dissociated an associative system driven by contiguity and
similarity from a rule-based system that captures "logical, hierarchical, and
causal-mechanical" structure and can *explain*.

The direct answer to "how is sparse data enough?" is **Griffiths & Tenenbaum (2009),
"Theory-based causal induction", *Psychological Review* 116(4):661–716 (**comp**, [S] with
 behavioural support)**. Their framing is nearly verbatim the RAVANA problem: identifying
unobservable mechanisms from "sparse and limited data". Causal induction is
"domain-general statistical inference guided by domain-specific prior knowledge, in the
form of an abstract causal theory." They isolate three constraints supplied by prior
knowledge: (i) the **ontology** of entities, properties and relations; (ii) the
**plausibility** of specific causal relationships; (iii) the **functional form** of those
relationships. Without (i)–(iii) the hypothesis space is too large for sparse data to
constrain — this is the computational reason RAVANA's mostly-unconstrained graph yields
one-hop associations rather than mechanisms.

Two extensions matter for acquisition:

- **The abstract framework is itself learnable.** Goodman, Ullman & Tenenbaum (2011,
  "Learning a theory of causality", *Psychological Review* 118(1):110–119; **comp**, [M])
  show that primitives such as the `do` operator and screening-off can be acquired from
  data. RAVANA therefore need not hardcode every relation; it can learn the *shape* of
  causal relations.
- **Structure shapes are reused across domains.** Kemp, Goodman & Tenenbaum (2010,
  "Learning to learn causal models", *Cognitive Science* 34(7):1185–1243; **beh + comp**,
  [S]) found that a learned **causal schema** capturing commonalities across several
  causal models accelerates learning about subsequently encountered objects. Tenenbaum,
  Kemp & Griffiths (2011, "How to grow a mind", *Science* 331:1279–1285; **comp**, [S])
  generalise the point: human inference is Bayesian inference over *structured*
  hypotheses. Holyoak & Cheng (2011, *Annual Review of Psychology* 62:135–163) label the
  reconciliation of associative and causal-model traditions a "new synthesis" — the
  associative system is real but constrained by causal structure.

### 1.2 Intervention and counterfactual reasoning

**Gerstenberg (2024), "Counterfactual simulation in causal cognition", *Trends in
Cognitive Sciences* 28(10):924–936 (**comp + beh**, [S])** gives the operative recipe:
three ingredients are required — a **generative mental model**, the ability to
**intervene** on it, and the capacity to **simulate consequences**. Causal judgment tracks
how certain the agent is that the counterfactual outcome would differ from what happened.
Gerstenberg, Goodman, Lagnado & Tenenbaum (2021, "A counterfactual simulation model of
causal judgments for physical events", *Psychological Review* 128(5):936–975; **comp +
beh**, [S]) implement this and fit human physical-event judgments. The CSM is explicitly a
model of **judgment, not learning** (the authors say so); do not cite it as evidence that
humans *acquire* structure this way — cite it as evidence that a stored structure must
support deletion/simulation to count as causal at all. Meder, Mayrhofer & Waldmann (2014,
"Structure induction in diagnostic causal reasoning", *Psychological Review*
121(3):277–301; **beh + comp**, [S]) show that even backward ("why") reasoning is
structure induction over causal Bayes nets, not a single lookup. Bramley, Gerstenberg,
Mayrhofer & Lagnado (2018, "Time in causal structure learning", *JEP: Learning, Memory,
and Cognition* 44(12):1880–1910; **beh**, [M]) show temporal order constrains which
structures are considered; Gerstenberg's group further shows people weight candidate
structures by implied interval similarity.

**Implication.** RAVANA's counterfactual primitive is only meaningful if it tests
structure: an edge that cannot be removed without breaking the chain is not a
load-bearing causal link. This is criterion **G3** below.

### 1.3 The role of prior knowledge in constraining which links are considered

Prior knowledge is not an optional aid — it is what makes the inverse problem solvable
(Griffiths & Tenenbaum 2009, above). At the neural level, **Dorfman, Tomov, Cheung,
Clarke, Gershman & Hughes (2021), "Causal inference gates corticostriatal learning",
*Journal of Neuroscience* 41(32):6892–6904 (**neuro + beh**, [M])** show that beliefs about
causal structure modulate prediction-error-driven learning, represented in anterior
insula and inferior frontal gyrus. A new link consistent with current structure is
learned fast; a structurally surprising one is not simply written at full strength.

**Implication.** A schema-congruent edge should be *cheap to hypothesise* but still has to
earn trust; a structure-surprising edge should require more corroboration, not less.
RAVANA currently gives every first-sighted web causal edge the same trusted stamp
(`web_fact`), which inverts this gradient.

---

## 2. Research question 2 — How the brain decides to explore for new causal structure

### 2.1 Information-gap theory

**Loewenstein (1994), "The psychology of curiosity: A review and reinterpretation",
*Psychological Bulletin* 116(1):75–98 (**beh/theory**, [S])** reframes curiosity as
"a form of cognitively induced deprivation that arises from the perception of a gap in
knowledge or understanding." Three properties are load-bearing for an explorer:

- the gap must be **specific** — a known unknown, not diffuse ignorance;
- curiosity is **transient** and is discharged when the gap is filled;
- the agent is more curious when it believes the available information **can resolve**
  the gap (Loewenstein 1994, p. 87).

**Implication.** A gap signal of the form "topic X is under-connected" (`efe` from sparse
neighbourhood) opens a weak gap; a gap of the form "the causal link from A to B is
missing, direction `why`" opens the kind of gap Loewenstein describes and can be targeted
by a specific query. RAVANA has the first; it needs to derive the second from the failed
chain assembly.

### 2.2 Active inference and epistemic foraging

**Friston, Rigoli, Ognibene, Mathys, FitzGerald & Pezzulo (2015), "Active inference and
epistemic value", *Cognitive Neuroscience* 6(4):187–214 (**comp**, [M])** decompose
expected free energy into **extrinsic value** (utility/prior preference) and **epistemic
value** (information gain, reduction of uncertainty about hidden causes). Exploration
maximises epistemic value *until there is no further information gain*; once posterior
uncertainty is exhausted, exploitation takes over. Friston, Lin, Frith, Pezzulo, Hobson
& Ondobaka (2017, "Active inference, curiosity and insight", *Neural Computation*
29:2633–2683; **comp**, [M]) extend the account to novelty and uncertainty-resolving
policies. The narrower claim — epistemic value = information gain about hidden causes —
is the usable one; the claim that free energy is a *unified* brain theory is contested in
the commentaries on Friston (2010) and is **not** used here.

Convergent behavioural evidence: **Gong, Gerstenberg, Mayrhofer & Bramley (2023),
"Active causal structure learning in continuous time", *Cognitive Psychology* 140:101542
(**beh**, [M])** found that people choose *when and where to intervene* so as to balance
maximising expected information against minimising inferential complexity, and argue that
metacognitive awareness of one's inferential limits is critical. This is a caution
against "fetch everything": an informative but structurally complex page is a poor target.

### 2.3 Curiosity potentiates the encoding itself

**Gruber, Gelman & Ranganath (2014), "States of curiosity modulate hippocampus-dependent
learning via the dopaminergic circuit", *Neuron* 84(2):486–496 (**neuro**, [S])** found
that high-curiosity states improved memory both for the queried information *and for
incidental material*, with anticipatory activity in SN/VTA, nucleus accumbens and
hippocampus predicting the benefit. Curiosity is therefore not merely a routing signal:
it raises the gain on encoding while the gap is open. **Implication:** the targeted read
should happen while the gap is recorded and specific, not as unrelated background
crawling — a direct argument for gap-triggered rather than schedule-triggered fetching.

---

## 3. Research question 3 — How the brain avoids flooding memory with spurious causal links

This question conflates three distinct literatures; they are answered separately.

### 3.1 One-trial learning vs consolidation

**Tse, Langston, Kakeyama, Bethus, Spooner, Wood, Witter & Morris (2007), "Schemas and
memory consolidation", *Science* 316(5821):76–82 (**neuro**, [S])** is the decisive
result. Rats trained gradually on six flavour–place associations formed a schema; new
associations could then be learned in **one trial** and became hippocampal-independent
within 48 h. In an *inconsistent* arena with no stable schema, one-trial learning failed.
Hippocampal lesions 3 h after learning abolished the new memory but lesions at 48 h
spared it. The authors' conclusion: schemas play a **causal role** in one-trial
consolidation. **Lee, O'Doherty & Shimojo (2015), "Neural computations mediating one-shot
learning in the human brain", *PLoS Biology* 13(4):e1002137 (**neuro + beh**, [M])** show
in humans that uncertainty about a causal association sets the learning rate: the
hippocampus switches on for one-shot causal learning, and ventrolateral PFC encodes the
uncertainty that gates the switch.

**Implication.** One trial is sufficient *inside a learned schema*, not in general. A
first web sighting of a causal predicate is a one-trial event with no established schema
for that subject; the brain-faithful behaviour is a high-learning-rate *hypothesis* held
provisionally, not a consolidated fact. RAVANA's provisional node buffer implements the
hold for **nodes**; the **edge** path bypasses it.

### 3.2 Source monitoring

**Johnson & Raye (1981), "Reality monitoring", *Psychological Review* 88(1):67–85
(**beh**, [S])** established that memories from external perception and from internally
generated thought are discriminated by cues (perceptual/contextual detail, cognitive
operations), and that the discrimination is imperfect. **Johnson, Hashtroudi & Lindsay
(1993), "Source monitoring", *Psychological Bulletin* 114(1):3–28 (**beh**, [S])** generalised
this to source *classes*. **Johnson (1997), "Source monitoring and memory distortion",
*Philosophical Transactions of the Royal Society B* 352(1362):1733–1745 (**beh/review**,
[S])** is the key mechanism statement: attribution uses heuristic processes plus slower,
deliberate checks that "retrieve additional supporting or disconfirming evidence" and
"evaluate plausibility and consistency given general knowledge, schemes, biases and
goals." False memories arise from the *same* processes as true ones; an inference is not
an error until it is **misattributed to perception**. Source guessing falls back on
schematic expectations precisely when no source–item contingency is encoded (Spaniol &
Bayen 2002 probability-matching account, as tested in *JEP: LMC*; **beh**, [M]).

**Implication.** The discriminating variable is the **source class**, not the confidence
or the plausibility of the content. RAVANA's `provenance_trusted` is correctly built on
that grain: `web`/`definition`/`conversation`/`curriculum`/`curated` are accepted,
`co_occurrence` is rejected as "association, not causation" (`learn/explanation.py:268-316`).
`assertable_chain` fails closed — one untrusted hop sinks the whole chain
(`learn/explanation.py:319-328`) — matching the source-monitoring principle that a chain
is only as strong as its weakest attribution.

### 3.3 Schema-driven plausibility gates — the double edge

Schemas are not a pure filter; they are also a **confabulation generator**. Gilboa &
Marlatte (2017, "Neurobiology of schemas and schema-mediated memory", *Trends in
Cognitive Sciences* 21(8):618–631 (**neuro review**, [S])** report that schema
instantiation (vmPFC, hippocampus, angular gyrus, posterior cortices) can *enhance or
distort* memory from encoding onward; schema congruency boosts conceptual memory while
reducing fine detail and increasing false alarms, and strong schematic contexts raise
both correct and incorrect memories. Johnson (1997) makes the same point behaviourally:
schema-consistent inferences are preferentially misattributed to external source.

**Implication.** "It fits the existing schema" must **never** be an admission signal. It
is a *candidate-generation* signal. The admission signal must be an independent,
explicit source statement with corroboration. This is the design constraint that
separates substrate growth from confabulation — criterion **G1/G2** below.

---

## 4. Research question 4 — Do schemas let one observation support a general structure?

**Yes, as a hypothesis with a specific scope: within a domain, a schema supplies the
relation template into which a single new observation can be bound; it does not supply
the truth of the new observation.**

- **Bartlett (1932), *Remembering: A Study in Experimental and Social Psychology*
  (Cambridge University Press; **beh**, foundational monograph)** established that
  remembering is an "effort after meaning": reconstruction guided by organised prior
  reactions (*schemata*), with systematic assimilation of the unfamiliar to the familiar.
  Comprehension can therefore outrun the input — the reader supplies causal links the
  text omitted.
- **Piaget (1952), *The origins of intelligence in children* (International Universities
  Press; **theory**, foundational monograph)** named the two operations: **assimilation**
  (fit new input to an existing structure) and **accommodation** (restructure when it will
  not fit). A schema is reusable because it is abstracted from instances.
- **Tse et al. (2007)** supplies the neurobiological demonstration that an established
  schema is what permits one-trial assimilation and rapid consolidation.
- **Kemp, Goodman & Tenenbaum (2010)** supplies the direct computational/behavioural
  demonstration for causal structure specifically: a learned schema "specifies the
  causal powers and characteristic features of [object] categories and the characteristic
  causal interactions between categories", allowing subsequent causal models to be
  learned rapidly.
- **Gilboa & Marlatte (2017)** supply the caveat: moderate prior knowledge produces
  vmPFC–hippocampal interaction; very strong schematic influence can *inhibit* accurate
  detail encoding. Schema reuse has an optimum, not a maximum.

**Can RAVANA reuse an existing verified chain schema for a new subject?** It should reuse
the **relation shape** (e.g. "`A` drives a state change in `B`, which produces `C`") and
the *search targets* implied by it, but each new subject's hops must be independently
observed and sourced. Concretely: reuse `sun→light→photosynthesis` as a template for a
new energy/growth subject, and `water vapour→clouds→rain` as a template for a new
condensation/precipitation subject — then instantiate as **provisional hypotheses** and
let the web learner corroborate the new edges. Relabelling nodes on an old chain is
confabulation, not transfer (criterion **G4**).

---

## 5. Observable criteria: real substrate growth vs confabulated edges

Each criterion is a check on the **structured chain object plus the graph**, not a regex
over prose and not a string scored in isolation. This follows the existing THINKING-front
rubric (`docs/BRAIN_REASONING_DESIGN.md` §0), which was written after three measurement
failures caused by exactly that kind of proxy.

| # | Criterion and check | Why a confabulated edge fails | Human evidence | RAVANA hook (verified names) |
|---|---|---|---|---|
| **G1** | **Source-class provenance on the whole claim.** Every hop must carry a trusted `edge_kind` in `source_metadata` (`web_fact`, `curated`, `definition`, `curriculum`, `conversation`); any `co_occurrence`, `schema_hypothesis`, or unstamped hop fails the whole chain. | A confabulated link has no source that actually *states* the relation; it is internally generated and misattributed. High weight/confidence does not help — co-occurrence edges at w≥0.80 c=0.90 were measured false. | Johnson & Raye (1981); Johnson (1997) source monitoring; Dorfman et al. (2021) structure gates learning | Already implemented: `_hop_provenance`, `provenance_trusted`, `assertable_chain`, `grade_chain` in `learn/explanation.py:268-366`. `grade_chain` already converts `assert`→`decline` when the chain is untrusted. Keep; do not add a confidence surrogate. |
| **G2** | **Corroboration on the relation, not just the node.** A new causal hop is promoted only when ≥2 **distinct** `source_url` ids state the same `(subject, causal predicate, object)` triple. Re-sightings from one URL do not count. | A single sentence is a one-trial event; without an established schema it is a fragile hypothesis, and a schema-congruent single sighting is exactly the false-memory case. Convergence is the consolidation event. | Tse et al. (2007); Lee et al. (2015); Friston et al. (2015) epistemic value; Spaniol & Bayen (2002) | **Gap:** node promotion uses `_promote_min_sources` (`web_to_graph.py:149,174-198`) but `learn_text` stamps `edge_kind="web_fact"` on first sight (`:285-296`). Extend the distinct-source count to the causal edge; store the source ids in `source_metadata` and assert their count before the trusted stamp. |
| **G3** | **Load-bearing counterfactual on a frozen snapshot.** Delete each hop in turn on a copied graph and re-run `enumerate_chains`; the chain passes only if every deletion breaks the subject↔outcome connection with no equal/short alternate path. | Fabricated edges are decorations; removing one leaves the others intact. A real dependency is necessary. | Gerstenberg (2024) three ingredients; Gerstenberg et al. (2021); Pearl (2009) `do`-calculus; Sloman & Lagnado (2005) | `enumerate_chains` (`learn/explanation.py:156`) on a snapshot — the same discipline as `chain_walker`'s frozen-snapshot simulate. Print the deletion result beside the verdict. |
| **G4** | **Schema transfer is shape-preserving and instance-fresh.** A chain for a new subject must bind *new* node ids with *new* edge objects carrying their own provenance; the relation-type sequence may match a verified template, but no edge may be inherited by relabelling. | Relabelling an old verified chain changes the label string while leaving the evidence for a different subject in place — source misattribution in graph form. | Kemp et al. (2010) (schema specifies categories/powers, not instance identity); Bartlett (1932) assimilation; Gilboa & Marlatte (2017) distortion by strong schemas | Store `edge_kind="schema_hypothesis"` on instantiated templates; that tag is **not** in `provenance_trusted`'s accepted set, so it cannot assert. Compare node ids and provenance, not just labels. |
| **G5** | **Assertion/gap calibration against an independent census.** `grade_chain` verdict must match an independently, correctly iterated census: assert only when a trusted ≥2-hop chain exists in the query direction; otherwise `decline` + record a specific gap + trigger targeted reading. A decline when a chain exists, or an assertion when none exists, is a FAIL. | Confabulation is fluent assertion where the store has no chain; it is also the "I don't know" that hides a chain that was there. Both directions must be checked. | Rozenblit & Keil (2002) illusion of explanatory depth; Koriat (1993) feeling-of-knowing as computed from partial information; Johnson (1997) plausibility-based source guessing | `grade_chain` + `assertable_chain` already produce assert/hedge/decline. Add the census oracle; assert the census iterates nodes/edges correctly (dict → `.values()`/`.items()`) and that the known trusted edge count is non-empty before trusting a zero. |

**Instrument discipline.** These are conditions of a conjunction, not independent scores.
Report per-subject `assert / hedge / decline` counts plus the raw chain and each hop's
provenance beside its verdict — never a single quality average. Before quoting any
accuracy, add unseen `why`/`how` probes with hand-labelled gold chains; a rubric authored
in the same pass as its probes has never seen a real failure. Confirm the harness can
detect a difference by running it once with the provenance gate disabled: if every arm
returns the same number, the instrument is measuring something upstream of the mechanism.

---

## 6. Concrete mechanism recommendation

No new subsystem is required. The pipeline reuses the graph, the provenance record, the
provisional buffer and the web learner; the changes are to **what counts as trusted** and
**what the gap targets**. Keep the current path behind a flag so the change is reversible.

**Stage A — Convert a topic gap into a missing-relation gap (exists, wired).**
`WebToGraph.knowledge_gap(topic)` returns a `KnowledgeGap` with `efe = max(0, max_known -
known_edges)` (`web_to_graph.py:300-333`) — Friston/Loewenstein-shaped, but only about
topic sparsity. When `enumerate_chains(subject_ids)` returns no trusted ≥2-hop chain for a
`why`/`how` query, record the specific gap: `(subject, needed_direction, missing_hop)`
where `missing_hop` names the causal predicate and intermediate node the template needs.
This is the transition from "unknown topic" to a Loewenstein gap the agent believes it can
close.

**Stage B — Retrieve a verified chain schema as a hypothesis (new, small).**
Index the two existing verified chains by relation shape and domain tag in a lightweight
registry beside `core/causal_schema.py::CausalSchemaLearner` (no new infrastructure).
Instantiate the best-matching template for the new subject as **provisional hops** with
`source_metadata["edge_kind"] = "schema_hypothesis"` and no trusted source. Because
`provenance_trusted` does not accept that tag, these hops are *unassertable by
construction* — schema supplies candidate structure and the search target, never
evidence (Kemp et al. 2010; Tse et al. 2007; Johnson 1997).

**Stage C — Targeted epistemic foraging with a causal-verb constraint (exists, wired).**
Feed the missing hop to the existing gap-driven web read, with a query designed to yield
an OpenIE fact whose relation maps through `WebToGraph.RELATION_TO_EDGE` to a causal edge
(`learn_text` at `web_to_graph.py:206-298`; `edge_type = RELATION_TO_EDGE.get(f.relation,
"semantic")`). "Extractable causal verbs" is not a stylistic preference: without an
explicit predicate the extracted edge is `semantic`, which `enumerate_chains` will not
walk. Target the gap while it is open — Gruber et al. (2014) predict the encoding benefit
— and prefer informative, structurally simple sources (Gong et al. 2023).

**Stage D — Promote the EDGE on distinct-source convergence (the real gate change).**
Generalise `_promote_min_sources` from nodes to causal edges: collect distinct
`source_url` ids for the same typed triple; hold first sightings in the existing
provisional structure; stamp `edge_kind="web_fact"` only at the k-th distinct source.
Require the provenance record to retain an explicit causal predicate and the source ids.
Add the two guards a decision gate needs: an **absolute floor** (a weak-but-clearer
hypothesis must not displace a trusted incumbent) and a **source-reliability guard** (one
uncorroborated input displaces nothing) — a relative margin alone is not enough. A
schema-congruent edge still earns trust only by convergence (Dorfman et al. 2021; Tse et
al. 2007); a structure-surprising edge should require *more* corroboration, not less.

**Stage E — Consolidate on reuse, with auditable provenance (exists, partly wired).**
Once a chain has been asserted and traversed, it may be stamped as a reusable verified
schema (`curriculum`/`consolidated`) — but keep every per-edge source id so the chain is
auditable. Verify the `source_metadata` round-trips through the serializer before
believing any provenance-based measurement: a dropped `source_metadata` field makes every
downstream gate read an absent stamp as a property of the world. Tse et al. (2007) makes
"reuse consolidates" the brain-faithful step; the serializer check makes it observable.

**Decision rule in one line:**
`schema → provisional hypothesis (unassertable) → targeted read → 2 distinct sources on
the causal edge → provenance_trusted → assertable_chain → grade_chain → assert / hedge /
decline`, with a census oracle checking the verdict against the graph in both directions.

---

## 7. Falsifiability

- **F1 (schema-transfer claim).** Gap-targeted reading seeded from verified chain schemas
  yields more *new trusted 2-hop chains per subject* than topic-only reading at equal
  pages fetched. **Falsified if** schema seeding adds no trusted chains over a control that
  reads the same number of pages on random topics. Measure counts, not answer quality.
- **F2 (edge-convergence claim).** Applying the 2-source gate to causal edges reduces
  confabulated assertions without collapsing coverage relative to the current first-sight
  stamp. **Falsified if** the trusted-chain count does not rise after the same number of
  reads, or if decline rate rises while assertion correctness is unchanged. Report
  `assert / hedge / decline` counts and the hand-checked correctness of each asserted hop.
- **F3 (gap-specificity claim).** A missing-relation gap closes faster than a topic-level
  gap on the same subject. **Falsified if** both require the same number of reads to reach
  a trusted chain.
- **F4 (provenance-grain claim).** Source-class gating predicts error better than
  confidence gating. **Falsified if** `assertable_chain` does not separate hand-labelled
  true from false chains while confidence/weight does. This is the load-bearing test —
  if it fails, G1 is decorative.

---

## 8. Limits and contested evidence

1. **The schema → one-trial result is strongest within a domain.** Tse et al. (2007) is a
   rat flavour–place task in a consistent arena; the inconsistent-arena condition is the
   important control. Cross-domain schema transfer in humans is supported by Kemp et al.
   (2010) but with artificial objects and a rational model, not a demo that any label can
   be swapped. Do not claim arbitrary relabelling is valid.
2. **Rational/computational models are not process models.** Pearl (2009), Griffiths &
   Tenenbaum (2009), Kemp et al. (2010), Friston et al. (2015) and Gerstenberg (2024) are
   optimality or judgment accounts. They license *design constraints*, not claims that
   RAVANA's graph is a brain.
3. **Free energy as a unified theory is contested.** Only the epistemic-value /
   information-gain decomposition is used, as a proxy for "where to look next"; free
   energy is not computable from a chain string.
4. **Causal-model vs associative accounts are not settled.** Holyoak & Cheng (2011) call
   their synthesis a "new synthesis" precisely because neither pure side works. RAVANA's
   `causal` edges are *typed*, which is not the same as *correct*.
5. **Source monitoring is a theory of episodic/source memory, not of web text.** The
   mapping to provenance classes is functional (source that states the relation vs source
   that merely co-activates), not anatomical or mechanistic. It is the right mapping for
   an architecture with explicit provenance, but it is an analogy.
6. **`curated` is a source class, not a truth guarantee.** `docs/AUTHORED_CAUSAL_SKELETON.md`
   records that the authored priors contain at least one false edge (`gravity → earth`).
   Per-edge review remains a separate task; this document does not certify any edge as
   true.
7. **C1–C5-style structural checks cannot establish truth.** G1–G5 establish that graph-
   supported, source-tracked structure grew; they cannot establish that the world is as
   the graph says. A short generated string is evidence a chain was *realized*, not that
   it is correct.
8. **Gap-triggered learning must be measured, not assumed.** The self-explanation effect
   is weak for factual knowledge (Bisra et al. 2018, g=0.55 with heterogeneity; a 2025
   *Applied Cognitive Psychology* null for factual recall — see
   `docs/BRAIN_EXPLANATION_AND_REASONING.md` §6.1). F1/F3 are the checks, not a premise.

---

## 9. References

All journal articles below are peer-reviewed. Foundational monographs are labelled as
such. DOIs are given where verified against a publisher/Crossref/PsycNet page.

- Bartlett, F. C. (1932). *Remembering: A study in experimental and social psychology*
  (foundational monograph). Cambridge University Press.
- Bramley, N. R., Gerstenberg, T., Mayrhofer, R., & Lagnado, D. A. (2018). Time in causal
  structure learning. *Journal of Experimental Psychology: Learning, Memory, and
  Cognition, 44*(12), 1880–1910. DOI 10.1037/xlm0000548.
- Dorfman, H. M., Tomov, M. S., Cheung, B., Clarke, D., Gershman, S. J., & Hughes, B. L.
  (2021). Causal inference gates corticostriatal learning. *Journal of Neuroscience,
  41*(32), 6892–6904.
- Friston, K., Lin, M., Frith, C. D., Pezzulo, G., Hobson, J. A., & Ondobaka, S. (2017).
  Active inference, curiosity and insight. *Neural Computation, 29*, 2633–2683.
- Friston, K., Rigoli, F., Ognibene, D., Mathys, C., FitzGerald, T., & Pezzulo, G. (2015).
  Active inference and epistemic value. *Cognitive Neuroscience, 6*(4), 187–214.
  DOI 10.1080/17588928.2015.1020053.
- Gerstenberg, T. (2024). Counterfactual simulation in causal cognition. *Trends in
  Cognitive Sciences, 28*(10), 924–936. DOI 10.1016/j.tics.2024.04.012.
- Gerstenberg, T., Goodman, N. D., Lagnado, D. A., & Tenenbaum, J. B. (2021). A
  counterfactual simulation model of causal judgments for physical events. *Psychological
  Review, 128*(5), 936–975. DOI 10.1037/rev0000281.
- Gilboa, A., & Marlatte, H. (2017). Neurobiology of schemas and schema-mediated memory.
  *Trends in Cognitive Sciences, 21*(8), 618–631. DOI 10.1016/j.tics.2017.04.013.
- Gong, T., Gerstenberg, T., Mayrhofer, R., & Bramley, N. R. (2023). Active causal
  structure learning in continuous time. *Cognitive Psychology, 140*, 101542.
  DOI 10.1016/j.cogpsych.2022.101542.
- Goodman, N. D., Ullman, T. D., & Tenenbaum, J. B. (2011). Learning a theory of causality.
  *Psychological Review, 118*(1), 110–119.
- Griffiths, T. L., & Tenenbaum, J. B. (2009). Theory-based causal induction.
  *Psychological Review, 116*(4), 661–716. DOI 10.1037/a0017201.
- Gruber, M. J., Gelman, B. D., & Ranganath, C. (2014). States of curiosity modulate
  hippocampus-dependent learning via the dopaminergic circuit. *Neuron, 84*(2), 486–496.
  DOI 10.1016/j.neuron.2014.08.060.
- Holyoak, K. J., & Cheng, P. W. (2011). Causal learning and inference as a rational
  process: the new synthesis. *Annual Review of Psychology, 62*, 135–163.
  DOI 10.1146/annurev.psych.121208.131634.
- Johnson, M. K. (1997). Source monitoring and memory distortion. *Philosophical
  Transactions of the Royal Society B, 352*(1362), 1733–1745. DOI 10.1098/rstb.1997.0156.
- Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring.
  *Psychological Bulletin, 114*(1), 3–28. DOI 10.1037/0033-2909.114.1.3.
- Johnson, M. K., & Raye, C. L. (1981). Reality monitoring. *Psychological Review, 88*(1),
  67–85. DOI 10.1037/0033-295X.88.1.67.
- Kemp, C., Goodman, N. D., & Tenenbaum, J. B. (2010). Learning to learn causal models.
  *Cognitive Science, 34*(7), 1185–1243. DOI 10.1111/j.1551-6709.2010.01128.x.
- Koriat, A. (1993). How do we know that we know? The accessibility model of the feeling
  of knowing. *Psychological Review, 100*(4), 609–639. DOI 10.1037/0033-295X.100.4.609.
- Lee, S. W., O'Doherty, J. P., & Shimojo, S. (2015). Neural computations mediating
  one-shot learning in the human brain. *PLoS Biology, 13*(4), e1002137.
  DOI 10.1371/journal.pbio.1002137.
- Loewenstein, G. (1994). The psychology of curiosity: A review and reinterpretation.
  *Psychological Bulletin, 116*(1), 75–98. DOI 10.1037/0033-2909.116.1.75.
- Meder, B., Mayrhofer, R., & Waldmann, M. R. (2014). Structure induction in diagnostic
  causal reasoning. *Psychological Review, 121*(3), 277–301. DOI 10.1037/a0035944.
- Pearl, J. (2009). *Causality: Models, reasoning, and inference* (2nd ed.)
  (foundational monograph). Cambridge University Press. DOI 10.1017/CBO9780511803161.
- Piaget, J. (1952). *The origins of intelligence in children* (foundational monograph).
  International Universities Press.
- Rozenblit, L., & Keil, F. (2002). The misunderstood limits of folk science: an illusion
  of explanatory depth. *Cognitive Science, 26*(5), 521–562.
  DOI 10.1207/s15516709cog2605_1.
- Sloman, S. A. (1996). The empirical case for two systems of reasoning. *Psychological
  Bulletin, 119*(1), 3–22. DOI 10.1037/0033-2909.119.1.3.
- Sloman, S. A., & Lagnado, D. A. (2005). Do we "do"? *Cognitive Science, 29*(1), 5–39.
  DOI 10.1207/s15516709cog2901_2.
- Tenenbaum, J. B., Kemp, C., & Griffiths, T. L. (2011). How to grow a mind: statistics,
  structure, and abstraction. *Science, 331*(6022), 1279–1285.
- Tse, D., Langston, R. F., Kakeyama, M., Bethus, I., Spooner, P. A., Wood, E. R., Witter,
  M. P., & Morris, R. G. M. (2007). Schemas and memory consolidation. *Science,
  316*(5821), 76–82. DOI 10.1126/science.1135935.
- Waldmann, M. R., & Hagmayer, Y. (2005). Seeing versus doing: two modes of accessing
  causal knowledge. *Journal of Experimental Psychology: Learning, Memory, and Cognition,
  31*(2), 216–227. DOI 10.1037/0278-7393.31.2.216.

### Verified source URLs

- https://doi.org/10.1037/0033-2909.116.1.75 — Loewenstein (1994)
- https://doi.org/10.1037/0033-295X.88.1.67 — Johnson & Raye (1981)
- https://doi.org/10.1098/rstb.1997.0156 — Johnson (1997)
- https://doi.org/10.1080/17588928.2015.1020053 — Friston et al. (2015)
- https://doi.org/10.1016/j.neuron.2014.08.060 — Gruber et al. (2014)
- https://doi.org/10.1126/science.1135935 — Tse et al. (2007)
- https://doi.org/10.1016/j.tics.2017.04.013 — Gilboa & Marlatte (2017)
- https://doi.org/10.1037/a0017201 — Griffiths & Tenenbaum (2009)
- https://doi.org/10.1111/j.1551-6709.2010.01128.x — Kemp et al. (2010)
- https://doi.org/10.1016/j.cogpsych.2022.101542 — Gong et al. (2023)
- https://doi.org/10.1371/journal.pbio.1002137 — Lee et al. (2015)
- https://cicl.stanford.edu/papers/gerstenberg2024counterfactual.pdf — Gerstenberg (2024)

---

## 10. MEASURED NEGATIVE RESULT: definitions cannot bootstrap substrate

The code map proposed a conservative offline seed: read RAVANA's 583 stored
definitions and write only explicitly stated causal propositions, with both
endpoints already known as graph concepts, stamped `edge_kind='definition'`.

This was implemented and measured, then REVERTED as a no-op. Results:

```
WRITE_N=0
CANDIDATES=0
```

Only **8 of 583** definitions contain any causal-ish connective at all, and
most are false positives for causal structure:

- `cat :: ... kept by humans as a pet and for catching pests`
- `dog :: ... kept by humans as a pet, guard, or worker`
- `music :: ... valued as an art form`
- `sky :: ... appears blue because molecules in the air scatter shorter
  (blue) wavelengths` — a real causal claim, but its endpoint words are not
  graph concepts, so the conservative rule correctly refuses to write it.

**Conclusion:** the stored definition set does not contain enough explicit
causal propositions to grow the causal substrate offline. More substrate must
come from web reading with the network on, or from authored curriculum edges —
not from reinterpreting definitions. A schema can select which text to read;
it cannot substitute for the read. This is consistent with the source-memory
literature: schema-congruence generates hypotheses, but convergence from an
actual source is what admits them (Johnson & Raye 1981; Tse et al. 2007).

The n=60 held-out measurement confirms the gap: offline thinking is **8/100**
(primary 10, sealed 7), versus 19/100 at n=27. No rubric changed between the
two runs — the smaller suite was simply easier.
