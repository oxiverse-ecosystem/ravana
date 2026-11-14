# Typing a relation online: how the brain decides what KIND of relation holds

**Date:** 2026-11-14
**Status:** research + implementation spec. No code changed.
**Question:** RAVANA's graph has 55,441 edges but only ~5,000 carry a non-vacuous
`relation_type`; the rest are `semantic` (mere association). A precision
generation route needs a typed relation to build a sentence. The brain does not
have this problem: it meets a novel concept or a novel concept pair and
immediately knows whether it is *a kind of* something, *a part of* something,
*causes* something. What is that mechanism, and what should a small system do
when it has no stored label?

**Companion docs (do not re-read):** `docs/BRAIN_LINEARIZATION_SPEC.md`
(the message→frame→order pipeline this sits upstream of),
`docs/BRAIN_NOVEL_SUBJECT_GROUNDING.md` (the ATL hub and the commitment gate),
`docs/BRAIN_MESSAGE_PREPARATION.md`, `docs/ONLINE_RELATION_TYPING.md` (the
existing prototype of the mechanism this spec generalizes),
`docs/COVERAGE_TWO_BUGS.md`, `docs/PLAN_NARRATIVE_POSITIVE.md`,
`docs/RELATION_GATING_SETTLED.md`.

**Scope note.** This document is about the *relation-type* decision, not about
which target to pick (`macroplanning.select_target`) and not about how to
linearize the resulting proposition (`linearization.message_to_plan`). It is
the missing middle step: given a pair of concepts and no label, what predicate
holds between them, and how sure are you?

---

## 0. Executive summary

The brain does not retrieve a relation label for a novel pair. It **derives**
one, by a comparison process that is structurally analogous to analogical
mapping. Three findings carry the argument:

1. **Relation typing is a comparison, not a lookup.** The dominant
   computational accounts of how abstract relations are acquired — DORA
   (Doumas, Hummel & Sandhofer 2008; Doumas, Puebla, Martin & Hummel 2022),
   BART (Lu, Chen & Holyoak 2012; Lu, Wu & Holyoak 2019), and
   structure-mapping (Gentner 1983) — all agree that a relation is *isolated by
   comparing two or more situations and intersecting what is invariant*, not by
   matching a feature list against a stored prototype. The behavioural
   signature is that **comparison causes the discovery**: children and adults
   learn a novel relational category from two aligned exemplars when one
   exemplar teaches nothing (Christie & Gentner 2010; Doumas & Hummel 2013;
   Gentner, Anggoro & Klibanoff 2011).

2. **There are dedicated relation categories, and they are few and
   theory-like.** Relational categories (Gentner & Kurtz 2005) are defined by
   *fit to a core relational structure*, not by shared intrinsic features, and
   they are acquired later, remembered worse, and are more mutable than entity
   categories (Gentner 2005; Asmuth & Gentner 2017). The typology that
   survives cross-linguistic and corpus pressure is small: class-inclusion,
   part-whole, cause-effect, contrast, similarity, space, time, and
   object-attribute (Bejar, Chaffin & Embly 1991, as operationalized in
   SemEval-2012 Task 2, Jurgens, Mohammad, Turney & Holyoak 2012; see also
   Popov, Pavlova & Hristova 2020).

3. **The brain does not commit at chance — it hedges, and the hedge is
   graded.** Relational judgments are not all-or-none: exemplars of a
   relation form **typicality gradients** (Chaffin 1992; Popov et al. 2020;
   Ichien, Lu & Holyoak 2022), a single pair can instantiate several relations
   to different degrees (Ichien et al. 2022), and relational reasoning under
   uncertainty has a distinct parietal signature (Ragni, Franzmeier, Maier &
   Knauff 2016). The design consequence is a **calibrated deferral**: commit
   when the margin is large, hedge when it is small, and fall back when it is
   negligible.

**The one-sentence version:** *Online relation typing is structural alignment
against a small inventory of learned relational prototypes, scored with a
calibrated margin, and the system defers rather than commits when the margin
collapses.*

**Status as of this writing.** The mechanism already exists in the repo in
prototype: `chain_walker._infer_relation_type` (`chain_walker.py:1410`)
compares the relational direction vector `target.vector - subject.vector`
against GloVe-seeded relation prototypes (`chain_walker.py:608`) and returns
`(type, confidence)`. `docs/ONLINE_RELATION_TYPING.md` measured it at **75.5%
recovery on known typed edges (40/53)**, with accuracy monotone in confidence
(0.600 below 0.30, 0.839 in 0.30–0.45, 1.000 above 0.45), and a plan-route
coverage of **29/40 subjects at threshold 0.35 with unsupported assertions at
0.000**. This spec's contribution is the *literature grounding* for that
mechanism and a *sharper design* for the weak-evidence case.

---

## 1. Acquiring a new concept's relational structure on first encounter

### 1.1 The three classical accounts, and where each fails

The question "how does a novel noun get categorized online" has three
canonical answers, and the literature has largely moved to a hybrid.

**(a) Prototype / classical theory.** Categories are represented by a summary
prototype (or by necessary-and-sufficient features), and a novel instance is
categorized by similarity to the prototype. The empirical core is real and
replicated: family resemblance and graded typicality (Rosch & Mervis 1975;
Rosch 1978), basic-level privilege (Rosch et al. 1976), and the prototype
effects in speeded categorization. But the strong version fails in a way that
matters directly for relation typing: **it cannot represent a relation as an
entity that can take novel arguments.** A prototype of *cause* is a bundle of
features; it is not a two-place predicate. Doumas et al. (2008) state the
requirement exactly: to think about a relation you need "a representational
element that corresponds specifically to the relation (or, more accurately, to
the roles of the relation)," plus a *dynamic* binding of roles to fillers that
is independent of the role and filler representations. Prototype theory
supplies neither.

**(b) Theory-theory.** Concepts are organized by the theories that embed them,
and categorization is inference to the best explanation rather than feature
matching (Murphy & Medin 1985; Gopnik & Wellman 2012; Carey 2009). This is
the account that predicts relation typing best, because a theory *is* a
relational structure: it specifies which properties are causally or
constitutively connected (Ahn 1995; Sloman, Love & Ahn 1998). The
neuropsychological evidence converges: category-specific deficits and
feature-coherence effects are better explained by theory-like causal
structure than by similarity (Mahon & Caramazza 2009). But theory-theory is
underspecified about the *online* computation — it says what the representation
is, not how a novel pair gets assigned to one.

**(c) Embodied / grounded accounts.** Concepts are simulations in modal
systems (Barsalou 2008; Barsalou 2016), and conceptual combination is
perceptual simulation (Wu & Barsalou 2009). For *entity* concepts this is
well supported. For *relational* concepts it is weaker: Mahon & Caramazza
(2009) review the neuropsychological evidence and conclude that the
sensorimotor grounding of abstract and relational content is not established.
The honest reading is that grounding constrains the *content* of a relation
(what "above" or "grasps" means) but does not supply the *typing* mechanism.

**The hybrid that the evidence supports.** Relation typing is a
**comparison-based abstraction over a theory-like relational inventory**,
grounded where the relation has modal content and symbolic where it does not.
This is the synthesis in Holyoak & Lu (2021): relational reasoning involves
"multiple systems" — early, domain-bound systems for causation, time, space and
mental attribution, plus a later, domain-general system that re-represents
their outputs as explicit role-governed relations. The later system is the one
that types.

### 1.2 What is settled about a novel noun

The following are replicated and convergent enough to design against:

- **The noun advantage is real and has a relational explanation.** Nouns are
  learned before verbs across languages (Gentner 1982; Gentner & Boroditsky
  2001). Gentner's *natural partitions* hypothesis: object concepts carve the
  world at its high-cohesion joints, while relational terms "partition the
  leftovers" — a diffuse set of relational components. The corollary,
  *relational relativity*, is the load-bearing one for this spec: **the
  mapping from experience to relational terms is more variable across languages
  than the mapping to nominal terms** (Gentner 1982; Gentner & Bowerman 2009).
  A system that hard-codes one relation vocabulary is therefore building in an
  English-specific prior, not a universal one.
- **Naming invites the category.** Hearing a novel word for two exemplars
  invites comparison and promotes category formation (Waxman & Markow 1995;
  Gentner & Namy 1999). A label is not a passive tag; it is a *comparison
  prompt*.
- **The taxonomic/thematic distinction is early and robust.** Children honor
  a taxonomic constraint (a novel label extends to same-kind objects) over a
  thematic one (Markman & Hutchinson 1984; Markman 1989, 1990). The two
  relation families are neurally and behaviourally dissociable
  (taxonomic/thematic/integrative: Wu & Barsalou 2009; Santos, Chaigneau,
  Simmons & Tyler 2011).
- **Similarity-based induction is the developmental default, and relational
  induction comes later.** SINC (Sloutsky & Fisher 2004) and the
  "conceptual influences on induction" results (Sloutsky, Kloos & Fisher
  2007) show early induction is featural; conceptual/relational information
  plays a negligible role early and grows with age. This is the *relational
  shift* (Gentner 1988; Rattermann & Gentner 1998).

**Design consequence.** A small system should not expect a single mechanism to
cover both "is this a kind of X" and "does X cause Y". The first is a
similarity computation over entity features; the second is a structural
computation over role bindings. They have different confidence profiles, and
the system should treat them differently (§7.2).

---

## 2. Relation typing specifically: is there a dedicated typology?

### 2.1 Yes — relational categories are a distinct kind of category

Gentner & Kurtz (2005) define a **relational category** as "a category whose
membership is determined by a common relational structure rather than by common
properties" — *bridge* (must connect two entities), *carnivore* (must eat
animals), *robbery* (thief, goods, victim). They divide relational categories
into **role categories** (members play the same role in a schema: *thief*) and
**schema categories** (the relational system itself: *robbery*). Entity
categories are "first-order partitions of the world"; relational categories are
"second-order ways of organizing and linking those first-order partitions"
(Gentner & Kurtz 2005, following Gentner 1982).

The empirical contrasts are systematic:

| Property | Entity categories | Relational categories |
|---|---|---|
| Basis of membership | shared intrinsic features | fit to a core relational structure |
| Representation | dense, richly interconnected | sparse, rule-like |
| Acquisition | early | later (Gentner 2005; Asmuth & Gentner 2017) |
| Memory | better | worse (Asmuth & Gentner 2005, 2017) |
| Mutability | lower | higher (Asmuth & Gentner 2017) |
| Linguistic form | mostly nouns | relational nouns, verbs, prepositions, comparatives |

Approximately **half of the 100 highest-frequency nouns in the British
National Corpus are relational nouns** (Asmuth & Gentner 2005, cited in
Gentner & Kurtz 2005). Relational content is not a marginal addition to
semantic memory; it is half of it.

**The typology.** The most widely used operational typology is Bejar, Chaffin
& Embly (1991), with ten high-level categories, as adopted by SemEval-2012
Task 2 (Jurgens et al. 2012) and used in the fMRI work of Zhang, Han, Worth &
Liu (2020):

```
CLASS-INCLUSION   (flower:tulip, queen:Elizabeth)
PART-WHOLE        (car:engine, forest:tree)
OBJECT-ATTRIBUTE  (lemon:sour, knife:sharp)
CASE RELATIONS    (agent, instrument, beneficiary ...)
SPACE             (above, in, near)
TIME              (before, during, after)
SIMILAR           (car:auto, stream:river)
CONTRAST          (hot:cold, rich:poor)
CAUSE-EFFECT      (virus:flu, joke:laughter)
NON-ATTRIBUTE     (negative cases)
```

This is a *linguistic* typology, and it is contested as a psychological one
(§8). But it is the right *engineering* starting point because (a) it is the
typology the brain-imaging literature actually used, (b) it is small, and (c)
its categories are the ones a sentence can realize.

### 2.2 The neural evidence: dedicated relation representations exist

Three converging lines:

- **Relations are encoded independently of the concepts they connect.**
  Zhang et al. (2020), *Nature Communications* 11:1877, used the SemEval-2012
  word pairs and found that semantic relations are represented as distributed
  cortical patterns that are **independent of the specific concepts** in a
  given instance. A 2026 preprint extends this: "relations are encoded
  independently of the specific concepts that are connected in any particular
  instance of the relation" (bioRxiv 2026.02.19.706815). This is the strongest
  direct evidence for dedicated relation representations.
- **A frontoparietal network encodes relation types, with a rostro-caudal
  gradient.** Chiang, Peng, Lu, Holyoak & Monti (2021), *J. Cogn. Neurosci.*
  33(3):377–389, found that "subareas of parietal cortex appear to support the
  encoding of individual relations, whereas the rostrolateral pFC (RLPFC)
  appears critical in second-order comparisons." A meta-analysis of 47 fMRI
  studies (Wertheim & Ragni 2018, *J. Cogn. Neurosci.* 30(11):1734–1748)
  found frontoparietal activation for both deductive and inductive reasoning,
  with an extended network for induction. The BART model's predictions
  correlated with neural activity in **left posterior parietal cortex during
  relation representation and comparison**, and in RLPFC during comparison
  (Chiang et al. 2021; Lu et al. 2019).
- **The "how"/"what" streams are not the relation-typing mechanism.** The
  dorsal ("where"/"how") and ventral ("what") visual pathways (Kravitz, Saleem,
  Baker & Mishkin 2011, *Nat. Rev. Neurosci.* 12:217–230; Kravitz et al. 2013)
  are about spatial processing for action versus object quality. They are
  frequently invoked in the relational-reasoning literature, but the mapping is
  loose. The honest statement is that the dorsal stream's role in
  "arbitrary and dynamic spatiotemporal relations to guide action" is a
  *domain-specific precursor*, not the domain-general typing mechanism. The
  domain-general mechanism is the frontoparietal/RLPFC one above.

**The "how" vs "what" claim in the task brief.** The premotor-parietal "how"
versus temporal "what" framing is real but should be cited carefully. The
cleaner dissociation for relation typing is **parietal (first-order relation
representation) vs rostrolateral PFC (second-order relation comparison)**
(Wendelken et al. 2012; Bunge, Helskog & Wendelken 2009; Crone, Wendelken,
van Leijenhorst, Honomichl, Christoff & Bunge 2009), not the visual-stream
split. See §8.

### 2.3 The relational shift

Gentner (1988) proposed the **relational shift**: children interpret analogy
and metaphor first in terms of object similarity, then in terms of relational
similarity. Rattermann & Gentner (1998) confirmed it on a causal-mapping task.
The shift is **domain-specific and knowledge-dependent**, not a single maturational
switch: "as children build knowledge in a domain, they move from considering
similarities to considering relations" (Morrison et al., cited in Gentner 2006).
Gentner & Rattermann (1991) and Gentner, Rattermann & Forbus (1993) showed the
same pattern across tasks.

**Design consequence.** A system that has *no* relational knowledge in a domain
should behave like a pre-shift child: it should fall back to entity/feature
similarity and *not* assert a relational type. This is a principled reason for
the confidence gate in §7.3, not just a pragmatic one.

---

## 3. The role of comparative/analogical reasoning in typing the novel

This is the core of the mechanism, and the literature is unusually convergent.

### 3.1 Structure-mapping: relations are mapped, attributes are not

Gentner (1983), *Cognitive Science* 7(2):155–170, is the founding statement.
The central distinction is between **attributes** (one-place predicates:
`LARGE(x)`) and **relations** (higher-order predicates: `COLLIDE(x,y)`).
Structure-mapping's systematicity principle: **a predicate is more likely to
be imported into the target if it belongs to a matching system of
interconnected relations**; isolated predicates are dropped. "Attributes of
objects are dropped; relations between objects are preserved" (Gentner 1983;
Falkenhainer, Forbus & Gentner 1989).

The typing consequence: **a novel pair is typed by finding a known relational
system into which both arguments fit, not by overlapping feature sets.** Two
pairs with no features in common can instantiate the same relation
(`love:hate :: rich:poor`; `blindness:sight :: poverty:money` — Lu et al. 2019).
Feature overlap is neither necessary nor sufficient.

### 3.2 Comparison causes the discovery

The strongest and most directly relevant evidence is that **comparison is the
mechanism that creates the relational abstraction in the first place**:

- **Christie & Gentner (2010)**, *Journal of Cognition and Development*
  11(3):356–373: preschoolers shown one exemplar of a novel spatial relation
  preferred the object match; children shown **two exemplars and prompted to
  compare** were significantly more likely to choose the relational match.
  Seeing two exemplars *without* a comparison prompt did not help. The
  comparison process, not the multiplicity of examples, is the active
  ingredient.
- **Doumas & Hummel (2013)**, *PLoS ONE* 8(6):e63889: participants who
  **mapped** two exemplars (aligned them) rose to ceiling on 50 additional
  category classifications and transferred to novel stimuli with unseen
  dimensional values. Mapping facilitated discovery of the relation.
- **Gentner, Anggoro & Klibanoff (2011)**, *Child Development* 82(4):1173–1188:
  3-, 4-5- and 6-year-olds learned a relational category
  (e.g. *home for*: nest→bird, barn→horse) when given structure-mapping
  support plus relational language. Anggoro, Gentner & Klibanoff (2005) showed
  **progressive alignment** (concrete → abstract) plus relational labels let
  3-year-olds learn the relation.
- **Jung & Hummel (2015)**, *Cognitive Science* 39(6):1259–1291, and Jung &
  Hummel (2015), *Frontiers in Psychology* 6:110: people have great difficulty
  learning **probabilistic** relational categories (family-resemblance
  structure, no single invariant relation), and hierarchical encoding plus
  comparison makes them learnable. The mechanism is "intersection discovery
  akin to schema induction."

### 3.3 The computational accounts

Three models, all converging on the same mechanism:

- **DORA (Discovery Of Relations by Analogy)** — Doumas, Hummel & Sandhofer
  (2008), *Psychological Review* 115(1):1–43; Doumas, Puebla, Martin & Hummel
  (2022), *Psychological Review* 129(5):999–1041. DORA starts from
  **unstructured distributed representations** (flat feature vectors) and
  discovers relational predicates by **comparison and intersection discovery**:
  align two structured representations, intersect the bound semantic
  content, and the intersection becomes an explicit predicate that can be
  dynamically bound to new arguments. The 2022 paper generalizes this to
  cross-domain transfer. The key claim for this spec: **relations are learned
  from non-relational inputs by re-representation, not stipulated.**
- **BART (Bayesian Analogy with Relational Transformations)** — Lu, Chen &
  Holyoak (2012), *Psychological Review* 119(3):617–648; Lu, Wu & Holyoak
  (2019), *PNAS* 116(10):4176–4181. BART takes word embeddings for a *pair*,
  remaps features by the **rank of the differences** between the two words'
  feature values, and learns a weight distribution over relation types. It
  "captures the pattern of human typicality judgments for a broad range of
  abstract semantic relations" and solves verbal analogies at near-human
  accuracy. Crucially, **the converse of any learned relation can be formed
  without additional training** — the representation is genuinely relational,
  not a lookup table.
- **LISA / semantic-relation networks** — Hummel & Holyoak (2003),
  *Psychological Review* 110(2):327–366; Lu, Ichien & Holyoak (2022),
  *Psychological Review* 129(5):1078–1103. Explicit role-filler binding via
  synchrony; mapping is graph matching under cognitive constraints.

**The convergence.** All three say: *type a novel pair by aligning it to a
known relational structure and reading off the invariant.* None says: *look up
a label.* This is exactly what `chain_walker._infer_relation_type` does with
vectors, and it is why that mechanism is brain-faithful rather than a hack.

### 3.4 The honest counter-evidence: retrieval fails

The comparative route is powerful but **not the default**. The sad conclusion
from a large body of research is that **analogical retrieval often fails**:
people fail to retrieve a useful analog even when it is clearly in memory
(Gentner 1998; Gentner, Rattermann & Forbus 1993; Forbus, Gentner & Law 1995).
The failure is of *access*, not storage — a hint that the story is relevant
raises solution rates dramatically. This is why the design in §7 uses a
**closed, small prototype inventory** rather than open-ended retrieval over the
whole graph: it trades the coverage of retrieval for the reliability of a
bounded comparison.

---

## 4. Uncertainty: does the brain commit, hedge, or defer?

**Short answer: it hedges, and the hedge is graded.** The evidence does not
support a binary commit/defer model.

### 4.1 Relational judgments are graded, not categorical

- **Typicality gradients.** "Much like object categories (Rosch 1975),
  examples of semantic relations form typicality gradients rather than simply
  being 'all or none'" (Chaffin 1992; Chaffin & Herrmann 1988; Popov et al.
  2020; Ichien et al. 2022). `hot:cold` is a better example of CONTRAST than
  `warm:cool` (Jurgens et al. 2012).
- **Multi-relation pairs.** "A single pair of words can instantiate multiple
  relations to some degree" — `friend:enemy` is CONTRAST but also, to a degree,
  SIMILAR (Ichien et al. 2022). The representation is **distributed**, not
  atomistic.
- **Relational similarity is separable from role similarity and semantic
  similarity.** Popov, Pavlova & Hristova (2020) found that relational
  similarity predicted exemplar generation, analogical verification
  RT/accuracy, and relational luring in memory; role similarity predicted a
  subset; **semantic similarity predicted none of the tasks after controlling
  for the other two**. This is the decisive result for a design that might be
  tempted to type by feature overlap: *feature overlap is the wrong signal.*
- **Relational similarity predicts neural similarity.** Chiang et al. (2021)
  showed that a model's relational-similarity predictions correlated with
  neural activity during analogical reasoning, in left posterior parietal
  cortex and RLPFC.

### 4.2 Uncertainty has a neural signature

Ragni, Franzmeier, Maier & Knauff (2016), *Brain and Cognition* 104:72–81,
studied relational reasoning with **degrees of uncertainty in both premises and
conclusions** and found a parietal signature for the uncertain case. The
broader point: reasoning under incomplete information is not the same
computation as reasoning under certainty, and it recruits the parietal
relation-representation system differently. A system that knows its evidence
is weak should be running a different computation, not the same one with a
lower threshold.

### 4.3 The design consequence

The brain's response to weak relational evidence is **not** to pick the
argmax and assert it. It is to:

1. **Commit** when the margin between the best and second-best relation is
   large (the relational analog of a high typicality rating).
2. **Hedge** when the best relation is clearly ahead but the absolute evidence
   is weak — realize the relation with a weaker construction, or realize it
   with an explicit epistemic marker.
3. **Defer** when the margin collapses — fall back to a non-relation-bearing
   form (association, enumeration, or no claim).

This is the graded policy in §7.3. It is also exactly what
`docs/ONLINE_RELATION_TYPING.md` measured: below confidence 0.30 the accuracy
is 0.600 (barely better than chance over five types), and the correct action
is to *not fire*.

---

## 5. Learned from statistics, or innate primitives?

**Short answer: a small set of innate *primitives* plus a large, learned,
statistically-derived relational inventory.** The evidence is strong that
relation *content* is learned; it is weaker but real that the *capacity* and a
few *foundational relations* are innate.

### 5.1 The case for innate primitives

- **Core knowledge.** Spelke & Kinzler (2007), *Developmental Science*
  10(1):89–96: "Human cognition is founded, in part, on four systems for
  representing objects, actions, number, and space. It may be based, as well,
  on a fifth system for representing social partners." Each system "guides and
  shapes the mental processes that it serves." Spelke (2022), *What Babies
  Know*, defends these as innate, abstract, interconnected concept clusters.
  The relevance: **object, agent, number, space and social-partner are
  pre-wired relational domains.** A system typing a novel pair should have
  these as its primitive vocabulary.
- **Prelinguistic relational concepts.** 7- and 9-month-old infants detect and
  generalize the abstract **same–different** relation (Ferry, Hespos &
  Gentner 2015, *Child Development*; Gentner 2015). This is the simplest
  possible relational concept and it is present before language. Same–different
  is arguably the *ur*-relation: it is what comparison operates on.
- **The Late System.** Holyoak & Lu (2021) argue that domain-general
  relational reasoning is a human-specific "Late System" coming online around
  age three, *coexisting* with early domain-bound systems. The early systems
  are shared with other animals; the late one is not. The implication is that
  *the capacity to re-represent relations explicitly* is the innate part, and
  the *specific relations* are learned.

### 5.2 The case for statistical learning

- **Relations can be learned from non-relational inputs.** DORA (Doumas et al.
  2008, 2022) and BART (Lu et al. 2012, 2019) both demonstrate this
  computationally: flat feature vectors in, structured relational predicates
  out. Lu et al. (2019) is explicit that this "provides key elements of a
  protosymbolic representation system."
- **Relational language is acquired late and varies cross-linguistically.**
  Verbs and relational nouns are acquired later than entity nouns (Gentner
  1982; Gentner & Boroditsky 2001); relational terms are more mutable
  (Asmuth & Gentner 2017); and the mapping from experience to relational terms
  varies across languages (Gentner & Bowerman 2009; Malt, Sloman, Gennari,
  Shi & Wang 1999). If the relation inventory were innate and fixed, this
  variation would be hard to explain.
- **Comparison is the learning mechanism, and it is domain-general.**
  Christie & Gentner (2010), Doumas & Hummel (2013), Gentner et al. (2011),
  Jung & Hummel (2015) all show that the *same* comparison process acquires
  *different* relations in different domains. That is the signature of a
  general mechanism over variable content.
- **The "theory theory" of care.** Carey (2009), *The Origin of Concepts*:
  the initial state includes rich core-cognition representations, but new
  concepts are created by **Quinean bootstrapping** — placeholder symbols whose
  meaning is filled in by their interrelations, plus structured
  representational change. Gopnik & Wellman (2012), *Psychological Bulletin*
  138(6):1085–1108, recast this as causal models plus Bayesian learning.
  Neither is a blank slate; neither is a fixed inventory.

### 5.3 The synthesis, and its design consequence

**Settled:** the *capacity* for relational representation and a handful of
foundational relational domains (object, agent, space, time, number, social)
are early and probably innate; the *specific* relation inventory is learned by
comparison over experience and varies with language and culture.

**Design consequence.** A small system should:
- have a **small, fixed primitive tier** (the core-knowledge domains) that is
  never learned and always available;
- have a **learned inventory tier** (the Bejar ten, or a subset) whose
  prototypes are estimated from data and can be revised;
- and should **not** treat a learned prototype as ground truth — it is a
  hypothesis with a confidence (§7.3).

---

## 6. What this means for RAVANA

### 6.1 The existing mechanism, and why it is the right shape

`chain_walker._infer_relation_type` (`chain_walker.py:1410-1444`) does:

```
rel_vec = (target.vector - subject.vector); normalize
for rtype, proto_vec in self._relation_prototypes:
    conf = dot(rel_vec, proto_vec)
    pick argmax
confidence = 0.2 + 0.7 * max(0, best_conf)
```

This is a **relational direction vector compared against learned prototypes**.
In the literature's terms it is closest to BART's relation vector (Lu et al.
2019) and to the difference-vector baseline that BART was designed to beat
(Ichien et al. 2022). It is *not* feature overlap; it is a directional,
relational comparison. That is the correct family.

**Its measured performance** (`docs/ONLINE_RELATION_TYPING.md`, 40 subjects,
ground truth = the graph's own stored typed edges):

```
recovery on known typed edges: 40/53 = 0.755
  inferred confidence   n     accuracy
  0.00 - 0.30          20     0.600
  0.30 - 0.45          31     0.839
  0.45 - 1.00           2     1.000
```

**Accuracy is monotone in confidence.** That is the single most important
measurement in this document, because it is the empirical license for the
graded policy in §7.3. A mechanism whose confidence does not track accuracy
cannot be gated; this one can.

### 6.2 The gap

Three things are missing, and they are the subject of §7.

1. **The prototype inventory is hand-seeded and too small.**
   `_init_relation_prototypes` (`chain_walker.py:608-658`) seeds five types —
   `causal`, `contrastive`, `temporal`, `analogical`, `semantic` — from
   GloVe seed words. The graph's typed vocabulary
   (`graph_typing.TYPED_RELATION_TYPES`: `isa`, `has_property`, `capable_of`,
   `used_for`, `part_of`) is *not* in that inventory at all. So the two
   vocabularies do not meet: an `isa` edge can only be typed as whatever
   `isa`'s direction vector happens to align with. This is the same class of
   bug as the `_construction_for` vocabulary mismatch that
   `docs/COVERAGE_TWO_BUGS.md` fixed.
2. **The confidence is not a margin.** `0.2 + 0.7 * best_conf` is a monotone
   transform of the best dot product. It is high when the direction is
   aligned with *some* prototype, even if the second-best is nearly as high.
   It therefore cannot express "the evidence is ambiguous between two
   relations," which is precisely the case the brain hedges on (§4).
3. **There is no weak-evidence policy.** The current callers either use the
   inferred type or don't. There is no hedge form and no deferral form.

### 6.3 The measured context

The plan route is the right consumer for a typed relation.
`docs/PLAN_NARRATIVE_POSITIVE.md` measured the plan-driven narrative formulator
at on-topic 0.357 / unsupported 0.000 on the 7 subjects where a plan commits,
vs the baseline's 0.195 / 1.975. `docs/COVERAGE_TWO_BUGS.md` raised plan-route
firings from 6 to 15 and unsupported from 0.143 to 0.067.
`docs/ONLINE_RELATION_TYPING.md` raised firings to **29/40 at threshold 0.35
with unsupported 0.000**. The typed relation is the input to that route; the
only question is how to type it well and what to do when the typing is weak.

---

## 7. Concrete, falsifiable design

### 7.1 The representation

```
RelationHypothesis:
    subject        : str
    target         : str
    ranking        : List[(relation_type, score)]   # sorted, ALL types
    best           : str
    margin         : float   # score[best] - score[second]
    confidence     : float   # calibrated, see 7.3
    evidence_class : str     # 'stored' | 'aligned' | 'weak'
    construction   : str     # from linearization._construction_for(best)
```

The load-bearing change from the current `_infer_relation_type` is that the
output is a **full ranking with a margin**, not a single argmax with a
monotone transform of the best score. The margin is what the brain's hedging
behaviour requires (§4).

### 7.2 The typing algorithm (online, no retraining)

**Step 1 — Stored label first.** If the edge carries a non-vacuous
`relation_type` (not in `VACUOUS_RELATIONS`), use it with
`evidence_class='stored'` and `confidence=1.0`. This is the high-precision
route and it is what `docs/COVERAGE_TWO_BUGS.md` and
`docs/PLAN_NARRATIVE_POSITIVE.md` already exploit. **Do not re-infer a stored
label** — a stored label is a fact; an inference is a hypothesis.

**Step 2 — Structural alignment against the prototype inventory.** Compute the
relational direction vector and score every prototype:

```
rel_vec  = normalize(target.vector - subject.vector)
score[r] = dot(rel_vec, proto[r])       for each r in inventory
```

This is the existing mechanism, retained.

**Step 3 — Two-tier inventory.** The inventory must have two tiers:

```
PRIMITIVE_TIER  (never learned; core-knowledge domains, Spelke & Kinzler 2007)
    object      agent       space       time        number      social

LEARNED_TIER    (estimated from the graph's own typed edges; Bejar et al. 1991)
    class_inclusion   part_whole   cause_effect   contrast
    similar           space       time           object_attribute
```

The learned tier's prototypes should be **estimated from the graph's typed
edges**, not hand-seeded. Concretely: for each typed relation type *r* in the
graph, average `normalize(target.vector - subject.vector)` over all edges
labelled *r*. This makes the prototype a *learned* summary of the graph's own
relational usage, which is the DORA/BART claim (relations are learned from
non-relational inputs) applied to the graph. The hand-seeded `proto_map` in
`chain_walker.py:622` is a reasonable cold-start prior and should be the
fallback when a relation type has too few edges.

**Step 4 — Normalize the scores before comparing them.** Dot products against
different prototypes are not on a common scale. Use a softmax over the
inventory (temperature fitted on the graph's typed edges) so that `margin` and
`confidence` are comparable across subjects and relation types. This is a
calibration step, and it is required for the threshold in Step 5 to mean
anything.

**Step 5 — Gate, and branch.**

```
if evidence_class == 'stored':        → COMMIT
elif margin >= M_COMMIT and conf >= C_COMMIT:  → COMMIT
elif margin >= M_HEDGE and conf >= C_HEDGE:    → HEDGE
else:                                          → DEFER
```

with starting values `M_COMMIT = 0.15`, `C_COMMIT = 0.45`, `M_HEDGE = 0.05`,
`C_HEDGE = 0.30`. **These are starting values, not fitted ones.** The
measured anchor is `docs/ONLINE_RELATION_TYPING.md`: accuracy 0.839 in the
0.30–0.45 band and 0.600 below 0.30, so `C_HEDGE = 0.30` is the empirically
supported floor and `C_COMMIT = 0.45` is conservative. Both must be re-fitted
on a held-out set of typed edges (§7.5).

### 7.3 The three policies

**COMMIT** — build the sentence from the typed relation, using the existing
`linearization.message_to_plan` → `SurfaceRealizer.realize` path. This is the
measured precision route (unsupported 0.000).

**HEDGE** — build the sentence with an **explicitly weaker construction**.
The inventory already contains a `relate` construction
(`<theme> is related to <ground>`) which `plan_is_committed`
(`linearization.py:248`) rejects as vacuous. The hedge policy should instead
use a **marked** form that asserts the relation type but flags it as inferred:

```
"<subject> appears to be a kind of <target>"      (class_inclusion, hedged)
"<subject> seems to contribute to <target>"       (cause_effect, hedged)
"<subject> is apparently part of <target>"        (part_whole, hedged)
```

This is the linguistic realization of a graded judgment (§4.1) and it is the
brain's hedge: the relation is asserted, but the epistemic status is on the
surface. **The hedge form must not be counted as a committed plan** by
`plan_is_committed` — it is a distinct evidence class.

**DEFER** — do not assert a relation at all. Fall back to the baseline
narrative path, or emit a non-relation-bearing form (enumeration, or an
explicit "I don't know how X and Y are related"). This is the empirically
supported choice below confidence 0.30, where accuracy is 0.600 — barely
better than chance over five types, and not worth asserting
(`docs/ONLINE_RELATION_TYPING.md`).

**The deferral must be query-time only.** Do not persist the inferred type back
onto the edge. `docs/ONLINE_RELATION_TYPING.md` is right: persisting would let
a 60%-confidence guess become a stored fact, and the graph would then be
unable to distinguish fact from inference. The inferred type is a hypothesis
about a query, not a property of the graph.

### 7.4 What NOT to do

- **Do not type by feature overlap.** Popov et al. (2020) showed semantic
  similarity predicts none of the relational tasks after controlling for
  relational and role similarity. A cosine between `subject.vector` and
  `target.vector` is the wrong signal.
- **Do not use the raw dot product as the confidence.** It is not calibrated
  and it has no margin. Use the softmax margin (Step 4).
- **Do not infer over a stored label.** Stored labels are facts; inferring
  over them would let a bad inference overwrite a good fact.
- **Do not let the hedge form count as committed.** It would inflate the
  precision metric while asserting a relation the system is not sure of —
  the "fluent, grammatical, false" failure of `docs/BRAIN_LINEARIZATION_SPEC.md`
  §7.3.
- **Do not treat the primitive tier as learned.** The core-knowledge domains
  are the fixed vocabulary; making them learnable would let the system talk
  itself out of object and agent.

### 7.5 The falsifiable predictions

**P1 (coverage).** With the two-tier inventory and the COMMIT gate at
`C_COMMIT = 0.45`, plan-route firings on 40 graph-discovered subjects will be
**≥ 29/40** (the `docs/ONLINE_RELATION_TYPING.md` number) and unsupported
assertions will remain **≤ 0.067**.

*Falsifier F1:* if firings fall below 29, the learned-tier prototypes are
degrading the alignment — the hand-seeded `proto_map` was better. Fix by
estimating prototypes only from relation types with ≥ *k* typed edges and
falling back to `proto_map` otherwise.

**P2 (precision of the hedge).** HEDGED outputs will have a **higher
unsupported rate than COMMITTED outputs but a lower one than the baseline**,
and will be **grammatical** (`func_share > 0`).

*Falsifier F2:* if hedged outputs are ungrammatical, the hedge construction is
not being realized through `SurfaceRealizer` and the hedge is being emitted as
a raw string. Fix by routing the hedge through the same realizer with a
marked construction.

**P3 (calibration).** Accuracy on known typed edges will be **monotone in the
softmax confidence**, and the 0.30–0.45 band will have accuracy **≥ 0.80**.

*Falsifier F3:* if accuracy is flat in confidence, the softmax temperature is
wrong or the prototypes are not discriminative. Fix by fitting the temperature
on a held-out set of typed edges, or by adding the missing relation types to
the inventory.

**P4 (deferral is correct).** On the subjects where the system DEFERS, the
baseline path will outperform the plan route on unsupported assertions.

*Falsifier F4:* if deferred subjects do better on the plan route, the
threshold is too conservative and `C_HEDGE` should be lowered.

### 7.6 The instrument

Reuse `scripts/measure_plan_narrative.py` and the `docs/ONLINE_RELATION_TYPING.md`
harness. Minimum report:

| Metric | Definition | Why |
|---|---|---|
| `firings` | fraction of subjects with a COMMITTED plan | coverage |
| `hedged` | fraction with a HEDGED plan | the new middle band |
| `deferred` | fraction with neither | the new deferral |
| `unsupported` | existing | must not rise above 0.067 |
| `on_topic` | existing | must not regress below 0.32 |
| `typing_accuracy` | accuracy on known typed edges, by confidence band | calibration |
| `margin_hist` | histogram of `margin` for correct vs incorrect typings | does the margin discriminate? |

**The margin histogram is the key new instrument.** If `margin` does not
separate correct from incorrect typings, the gate is not measuring what it
claims to measure, and P3 is falsified regardless of the accuracy number.

---

## 8. Settled vs. contested

### Settled (replicated, convergent, or mechanistically forced)

1. **Relational categories are a distinct kind of category**, defined by fit to
   a relational structure rather than shared features, and they are acquired
   later, remembered worse, and are more mutable than entity categories.
   Gentner & Kurtz 2005; Gentner 2005; Asmuth & Gentner 2017.
2. **Comparison/structural alignment is the mechanism that creates relational
   abstractions.** Christie & Gentner 2010; Doumas & Hummel 2013; Gentner et
   al. 2011; Jung & Hummel 2015. One exemplar teaches less than two aligned
   exemplars, and comparison — not mere multiplicity — is the active
   ingredient.
3. **Relations can be learned from non-relational inputs by
   re-representation.** Doumas et al. 2008, 2022; Lu et al. 2012, 2019.
4. **Relational judgments are graded, not categorical.** Typicality gradients
   (Chaffin 1992; Popov et al. 2020), multi-relation pairs (Ichien et al.
   2022), and a distinct uncertainty computation (Ragni et al. 2016).
5. **Relational similarity is dissociable from semantic similarity.**
   Popov et al. 2020: semantic similarity predicts none of the relational
   tasks after controlling for relational and role similarity.
6. **Relations are neurally encoded independently of the concepts they
   connect.** Zhang et al. 2020; Chiang et al. 2021.
7. **A frontoparietal network with a rostro-caudal gradient supports
   relational reasoning**: parietal for first-order relation representation,
   RLPFC for second-order comparison. Wendelken et al. 2012; Bunge et al.
   2009; Crone et al. 2009; Wertheim & Ragni 2018; Chiang et al. 2021.
8. **There is a developmental relational shift** from object/feature similarity
   to relational similarity, and it is domain-specific and
   knowledge-dependent. Gentner 1988; Rattermann & Gentner 1998; Gentner 2006.
9. **A small set of core-knowledge relational domains is early and probably
   innate** (objects, actions, number, space, social partners). Spelke &
   Kinzler 2007; Spelke 2022; Ferry et al. 2015 (same–different in infancy).
10. **The specific relation inventory is learned and varies across languages.**
    Gentner 1982; Gentner & Bowerman 2009; Malt et al. 1999; Asmuth &
    Gentner 2017.
11. **RAVANA's existing `_infer_relation_type` mechanism works, and its
    confidence is monotone in accuracy** (0.600 / 0.839 / 1.000 by band;
    75.5% recovery). `docs/ONLINE_RELATION_TYPING.md`. Measured, n=53.

### Contested (live disputes, or evidence thinner than the claim)

1. **The exact typology.** Bejar et al. (1991) ten categories are a linguistic
   taxonomy adopted for convenience by SemEval-2012 and the fMRI literature.
   Whether the brain's relational categories carve the world at those joints
   is not established. Gentner & Kurtz (2005) offer a different cut (role vs
   schema categories). Popov et al. (2020) show relational similarity is
   separable from role and semantic similarity, but the factor structure is
   not settled. **Grade [M].**
2. **The "how"/"what" stream framing.** The premotor-parietal "how" vs
   temporal "what" dissociation (Kravitz et al. 2011, 2013) is real for
   visuospatial processing, but its extension to relation typing is loose. The
   cleaner dissociation is parietal vs RLPFC (first- vs second-order
   relations). The task brief's framing should be cited as a *precursor*
   dissociation, not the typing mechanism. **Grade [M].**
3. **Whether relational reasoning is one system or several.** Holyoak & Lu
   (2021) argue for multiple systems (early domain-bound + a late
   domain-general one); a unitary account is still defended. The behavioural
   and neural evidence favours multiple systems, but the late system's
   autonomy is not settled. **Grade [M].**
4. **How much is innate.** Core knowledge is well supported for objects,
   agents, number and space; the claim that the *relation inventory* is
   learned is supported by computational models (DORA, BART) and by
   cross-linguistic variation, but there is no direct test that a specific
   relation (e.g. CAUSE-EFFECT) is learned rather than innate. **Grade [M].**
5. **Whether the hedge is a distinct computation or a threshold on the same
   one.** Ragni et al. (2016) show a parietal signature for uncertain
   relational reasoning, but whether hedging is a separate process or a
   graded output of the same process is not resolved. **Grade [M].**
6. **Whether RAVANA's confidence threshold generalizes.** The 0.35 threshold
   was measured on 40 subjects with the graph's own typed edges as ground
   truth. Those edges are themselves partly produced by the same prototype
   mechanism (`graph/engine.py:_correct_relation_types`), so the ground truth
   is not fully independent. **This is the most important caveat in the
   document.** The 75.5% recovery is an upper bound on the mechanism's
   accuracy against a clean label set. **Grade [M].**
7. **Whether the learned-tier prototypes should be estimated from the graph's
   typed edges or from an external corpus.** Estimating from the graph is
   self-consistent but risks circularity (the typed edges were partly produced
   by the prototypes). Estimating from an external corpus (e.g. ConceptNet
   relation vectors) breaks the circularity but introduces a distribution
   shift. Not settled; §7.2 Step 3 recommends graph-estimated with a
   cold-start prior, and §7.5 P3 is the test.

---

## 9. Citations

- **Ahn, W.-K.** (1995). The role of causal status in feature centrality.
  *Cognitive Science* 19(4). (Causal status hypothesis; cause features weighted
  more than effect features.)
- **Ahn, W.-K., & Bailenson, J.** (1996). Mechanism knowledge in
  categorization. *Memory & Cognition*.
- **Anggoro, F. K., Gentner, D., & Klibanoff, R. S.** (2005). Children's
  learning of relational categories. In *Proceedings of the 27th Annual
  Conference of the Cognitive Science Society*.
- **Asmuth, J., & Gentner, D.** (2005). Context sensitivity of relational
  nouns. *Proceedings of the 27th Annual Conference of the Cognitive Science
  Society*.
- **Asmuth, J., & Gentner, D.** (2017). Relational categories are more mutable
  than entity categories. *Quarterly Journal of Experimental Psychology*,
  70(10), 2007–2025. doi:10.1080/17470218.2016.1219752
- **Barsalou, L. W.** (1983). Ad hoc categories. *Memory & Cognition*, 11(3),
  211–227. doi:10.3758/BF03196968
- **Barsalou, L. W.** (2008). Grounded cognition. *Annual Review of
  Psychology*, 59, 617–645. doi:10.1146/annurev.psych.59.103006.093639
- **Barsalou, L. W.** (2016). On staying grounded and avoiding quixotic dead
  ends. *Psychonomic Bulletin & Review*, 23(4), 1122–1142.
  doi:10.3758/s13423-016-1028-3
- **Bejar, I. I., Chaffin, R., & Embly, D. C.** (1991). A taxonomy of semantic
  relations. In *Semantic Relations and Lexical Semantics*. (Adopted as the
  SemEval-2012 Task 2 typology; see Jurgens et al. 2012.)
- **Binder, J. R., & Desai, R. H.** (2011). The neurobiology of semantic
  memory. *Trends in Cognitive Sciences*, 15(11), 527–536.
  doi:10.1016/j.tics.2011.10.001
- **Binder, J. R., Conant, L. L., Humphries, C. J., Fernandino, L., Simons,
  S. B., Aguilar, M., & Desai, R. H.** (2016). Toward a brain-based
  componential semantic representation. *Cognitive Neuropsychology*, 33(3–4),
  130–174. doi:10.1080/02643294.2016.1147426
- **Bloom, P.** (1996). Intention, history, and artifact concepts. *Cognition*,
  60(1), 1–29. doi:10.1016/0010-0277(95)00699-0
- **Bunge, S. A., Helskog, E. H., & Wendelken, C.** (2009). Left, but not
  right, rostrolateral prefrontal cortex meets a stringent test of the
  relational integration hypothesis. *NeuroImage*, 46, 338–342.
- **Carey, S.** (2009). *The Origin of Concepts*. Oxford University Press.
  doi:10.1093/acprof:oso/9780195367638.001.0001
- **Carey, S., & Spelke, E. S.** (1996). Science and core knowledge.
  *Philosophy of Science*, 63(4), 515–533. doi:10.1086/289971
- **Chaffin, R.** (1992). The typicality of semantic relations. *Memory &
  Cognition*.
- **Chiang, J. N., Peng, Y., Lu, H., Holyoak, K. J., & Monti, M. M.** (2021).
  Distributed code for semantic relations predicts neural similarity during
  analogical reasoning. *Journal of Cognitive Neuroscience*, 33(3), 377–389.
  doi:10.1162/jocn_a_01620
- **Christie, S., & Gentner, D.** (2010). Where hypotheses come from: Learning
  new relations by structural alignment. *Journal of Cognition and
  Development*, 11(3), 356–373. doi:10.1080/15248371003700015
- **Crone, E. A., Wendelken, C., van Leijenhorst, L., Honomichl, R. D.,
  Christoff, K., & Bunge, S. A.** (2009). Neurocognitive development of
  relational reasoning. *Developmental Science*, 12(1), 55–66.
  doi:10.1111/j.1467-7687.2008.00743.x
- **Doumas, L. A. A., & Hummel, J. E.** (2013). Comparison and mapping
  facilitate relation discovery and predication. *PLoS ONE*, 8(6), e63889.
  doi:10.1371/journal.pone.0063889
- **Doumas, L. A. A., Hummel, J. E., & Sandhofer, C. M.** (2008). A theory of
  the discovery and predication of relational concepts. *Psychological
  Review*, 115(1), 1–43. doi:10.1037/0033-295X.115.1.1
- **Doumas, L. A. A., Puebla, G., Martin, A. E., & Hummel, J. E.** (2022). A
  theory of relation learning and cross-domain generalization. *Psychological
  Review*, 129(5), 999–1041. doi:10.1037/rev0000346
- **Falkenhainer, B., Forbus, K. D., & Gentner, D.** (1989). The
  structure-mapping engine. *Artificial Intelligence*, 41(1), 1–63.
- **Ferry, A. L., Hespos, S. J., & Gentner, D.** (2015). Prelinguistic
  relational concepts: Investigating analogical processing in infants. *Child
  Development*, 86(5), 1386–1405. doi:10.1111/cdev.12381
- **Forbus, K. D., Gentner, D., & Law, K.** (1995). MAC/FAC: A model of
  similarity-based retrieval. *Cognitive Science*, 19(2), 141–205.
- **Gentner, D.** (1981). Some interesting differences between verbs and
  nouns. *Cognition and Brain Theory*, 4(2), 161–178.
- **Gentner, D.** (1982). Why nouns are learned before verbs: Linguistic
  relativity vs. natural partitions. In *Proceedings of the 2nd International
  Conference on the Study of Child Language*. (Natural partitions hypothesis.)
- **Gentner, D.** (1983). Structure-mapping: A theoretical framework for
  analogy. *Cognitive Science*, 7(2), 155–170.
  doi:10.1207/s15516709cog0702_3
- **Gentner, D.** (1988). Metaphor as structure mapping: The relational shift.
  *Child Development*, 59(1), 47–59. doi:10.1111/j.1467-8624.1988.tb03194.x
- **Gentner, D.** (1998). Analogy. In *A Companion to Cognitive Science*.
  (Analogical retrieval often fails.)
- **Gentner, D.** (2005). The development of relational category knowledge. In
  *Categorization Inside and Outside the Laboratory*.
- **Gentner, D.** (2006). Why verbs are hard to learn. In *Action Meets Word*.
- **Gentner, D.** (2016). Language as cognitive tool kit: How language
  supports relational thought. *American Psychologist*, 71(8), 750–757.
- **Gentner, D., Anggoro, F. K., & Klibanoff, R. S.** (2011). Structure
  mapping and relational language support children's learning of relational
  categories. *Child Development*, 82(4), 1173–1188.
  doi:10.1111/j.1467-8624.2011.01599.x
- **Gentner, D., & Boroditsky, L.** (2001). Individuation, relativity, and
  early word learning. In *Language Acquisition and Conceptual Development*.
- **Gentner, D., & Bowerman, M.** (2009). Why some spatial semantic categories
  are harder to learn than others: The typological prevalence hypothesis. In
  *Crosslinguistic Approaches to the Psychology of Language*.
- **Gentner, D., & Christie, S.** (2010). Mutual bootstrapping between
  language and analogical processing. *Language and Cognition*, 2(2),
  261–283. doi:10.1515/langcog.2010.011
- **Gentner, D., & Hoyos, C.** (2017). Analogy and abstraction. *Topics in
  Cognitive Science*, 9(3), 672–693. doi:10.1111/tops.12278
- **Gentner, D., & Kurtz, K. J.** (2005). Relational categories. In *Categorization Inside and Outside the Laboratory*.
- **Gentner, D., & Markman, A. B.** (1997). Structure mapping in analogy and
  similarity. *American Psychologist*, 52(1), 45–56.
  doi:10.1037/0003-066X.52.1.45
- **Gentner, D., & Namy, L. L.** (1999). Comparison in the development of
  categories. *Cognitive Development*, 14(4), 487–513.
- **Gentner, D., & Rattermann, M. J.** (1991). Language and the career of
  similarity. In *Perspectives on Language and Thought*.
- **Gentner, D., Rattermann, M. J., & Forbus, K. D.** (1993). The roles of
  similarity in transfer: Separating retrievability from inferential
  soundness. *Cognitive Psychology*, 25(4), 524–575.
- **Gopnik, A., & Wellman, H. M.** (2012). Reconstructing constructivism:
  Causal models, Bayesian learning mechanisms and the theory theory.
  *Psychological Bulletin*, 138(6), 1085–1108. doi:10.1037/a0028044
- **Gray, M. E., & Holyoak, K. J.** (2020). Individual differences in
  relational reasoning. *Memory & Cognition*, 48(1), 96–110.
  doi:10.3758/s13421-019-00964-y
- **Holyoak, K. J., & Lu, H.** (2021). Emergence of relational reasoning.
  *Current Opinion in Behavioral Sciences*, 37, 118–124.
  doi:10.1016/j.cobeha.2020.11.012
- **Holyoak, K. J., & Thagard, P.** (1995). *Mental Leaps: Analogy in Creative
  Thought*. MIT Press.
- **Hummel, J. E., & Holyoak, K. J.** (2003). A symbolic-connectionist theory
  of relational inference and generalization. *Psychological Review*, 110(2),
  327–366.
- **Ichien, N., Lu, H., & Holyoak, K. J.** (2022). Predicting patterns of
  similarity among abstract semantic relations. *Journal of Experimental
  Psychology: Learning, Memory, and Cognition*, 48(1), 108–121.
  doi:10.1037/xlm0001010
- **Jung, W., & Hummel, J. E.** (2015). Making probabilistic relational
  categories learnable. *Cognitive Science*, 39(6), 1259–1291.
- **Jung, W., & Hummel, J. E.** (2015). Revisiting Wittgenstein's puzzle:
  Hierarchical encoding and comparison facilitate learning of probabilistic
  relational categories. *Frontiers in Psychology*, 6, 110.
  doi:10.3389/fpsyg.2015.00110
- **Jurgens, D., Mohammad, S., Turney, P., & Holyoak, K. J.** (2012).
  SemEval-2012 Task 2: Measuring degrees of relational similarity. In
  *Proceedings of the 6th International Workshop on Semantic Evaluation*,
  356–364. (Ten relation categories from Bejar et al. 1991.)
- **Kelemen, D., & Carey, S.** (2009). The development of artifact cognition.
  In *The Oxford Handbook of Evolutionary Psychology*.
- **Kemp, C., Chang, K. K., & Lombardi, L.** (2010). Category and feature
  identification. *Acta Psychologica*, 133(3), 216–233.
  doi:10.1016/j.actpsy.2009.11.012
- **Kemp, C., & Jern, A.** (2014). A taxonomy of inductive problems.
  *Psychonomic Bulletin & Review*, 21(1), 23–46.
  doi:10.3758/s13423-013-0467-3
- **Kemp, C., & Regier, T.** (2012). Kinship categories across languages
  reflect general communicative principles. *Science*, 336(6084), 1049–1054.
  doi:10.1126/science.1218811
- **Kemp, C., Xu, Y., & Regier, T.** (2018). Semantic typology and efficient
  communication. *Annual Review of Linguistics*, 4, 109–128.
  doi:10.1146/annurev-linguistics-011817-045406
- **Kravitz, D. J., Saleem, K. S., Baker, C. I., & Mishkin, M.** (2011). A new
  neural framework for visuospatial processing. *Nature Reviews Neuroscience*,
  12(4), 217–230. doi:10.1038/nrn3008
- **Kravitz, D. J., Saleem, K. S., Baker, C. I., & Mishkin, M.** (2013). The
  ventral visual pathway: An expanded neural framework for the processing of
  object quality. *Trends in Cognitive Sciences*, 17(1), 26–49.
- **Lu, H., Chen, D., & Holyoak, K. J.** (2012). Bayesian analogy with
  relational transformations. *Psychological Review*, 119(3), 617–648.
  doi:10.1037/a0028719
- **Lu, H., Ichien, N., & Holyoak, K. J.** (2022). Probabilistic analogical
  mapping with semantic relation networks. *Psychological Review*, 129(5),
  1078–1103. doi:10.1037/rev0000358
- **Lu, H., Wu, Y. N., & Holyoak, K. J.** (2019). Emergence of analogy from
  relation learning. *PNAS*, 116(10), 4176–4181.
  doi:10.1073/pnas.1814779116
- **Mahon, B. Z., & Caramazza, A.** (2009). Concepts and categories: A
  cognitive neuropsychological perspective. *Annual Review of Psychology*, 60,
  27–51. doi:10.1146/annurev.psych.60.110707.163532
- **Malt, B. C., Sloman, S. A., Gennari, S., Shi, M., & Wang, Y.** (1999).
  Knowing versus naming: Similarity and the linguistic categorization of
  artifacts. *Journal of Memory and Language*, 40(2), 230–262.
  doi:10.1006/jmla.1998.2593
- **Markman, E. M.** (1989). *Categorization and Naming in Children: Problems
  of Induction*. MIT Press.
- **Markman, E. M.** (1990). Constraints children place on word meanings.
  *Cognitive Science*, 14(1), 57–77.
- **Markman, E. M., & Hutchinson, J. E.** (1984). Children's sensitivity to
  constraints on word meaning: Taxonomic versus thematic relations. *Cognitive
  Psychology*, 16(1), 1–27. doi:10.1016/0010-0285(84)90002-1
- **Markman, A. B., & Gentner, D.** (1993). Structural alignment during
  similarity comparisons. *Cognitive Psychology*, 25(4), 431–467.
- **Murphy, G. L.** (2002). *The Big Book of Concepts*. MIT Press.
- **Murphy, G. L., & Medin, D. L.** (1985). The role of theories in conceptual
  coherence. *Psychological Review*, 92(3), 289–316.
  doi:10.1037/0033-295X.92.3.289
- **Popov, V., Pavlova, M., & Hristova, P.** (2020). The internal structure of
  semantic relations: Effects of relational similarity and typicality.
  *PsyArXiv* preprint. doi:10.31234/osf.io/fqd4b
- **Ragni, M., Franzmeier, I., Maier, S., & Knauff, M.** (2016). Uncertain
  relational reasoning in the parietal cortex. *Brain and Cognition*, 104,
  72–81. doi:10.1016/j.bandc.2016.02.006
- **Rattermann, M. J., & Gentner, D.** (1998). Relational shift in the
  development of analogy: Children's performance on a causal-mapping task.
  *Cognitive Development*, 13(4), 453–478.
- **Rosch, E.** (1978). Principles of categorization. In *Cognition and
  Categorization*, 27–48. Erlbaum.
- **Rosch, E., & Mervis, C. B.** (1975). Family resemblance: Studies in the
  internal structure of categories. *Cognitive Psychology*, 7(4), 573–605.
  doi:10.1016/0010-0285(75)90024-9
- **Rosch, E., Simpson, C., & Miller, R. S.** (1976). Structural bases of
  typicality effects. *Journal of Experimental Psychology: Human Perception
  and Performance*, 2(4), 491–502.
- **Santos, A., Chaigneau, S. E., Simmons, W. K., & Tyler, L. K.** (2011).
  Property generation reflects word meaning. *Cognitive Science*.
- **Sloman, S. A., Love, B. C., & Ahn, W.-K.** (1998). Feature centrality and
  conceptual coherence. *Cognitive Science*, 22(2), 189–228.
- **Sloutsky, V. M., & Fisher, A. V.** (2004). Induction and categorization in
  young children: A similarity-based model. *Journal of Experimental
  Psychology: General*, 133(2), 166–188. doi:10.1037/0096-3445.133.2.166
- **Sloutsky, V. M., Kloos, H., & Fisher, A. V.** (2007). Conceptual
  influences on induction: A case for a late onset. *Cognitive Science*.
- **Spelke, E. S.** (2022). *What Babies Know: Core Knowledge and Composition*
  (Vol. 1). Oxford University Press. doi:10.1093/oso/9780190618247.001.0001
- **Spelke, E. S., & Kinzler, K. D.** (2007). Core knowledge. *Developmental
  Science*, 10(1), 89–96. doi:10.1111/j.1467-7687.2007.00569.x
- **Tversky, A.** (1977). Features of similarity. *Psychological Review*,
  84(4), 327–352. doi:10.1037/0033-295X.84.4.327
- **Waxman, S. R., & Markow, D. B.** (1995). Words as invitations to form
  categories: Evidence from 12- to 13-month-old infants. *Cognitive
  Psychology*, 29(3), 257–302. doi:10.1006/cogp.1995.1016
- **Wendelken, C., et al.** (2012). Rostrolateral prefrontal cortex: Domain
  general or domain sensitive? *NeuroImage*. (Left RLPFC and left IPS most
  strongly engaged during relational comparison.)
- **Wertheim, J., & Ragni, M.** (2018). The neural correlates of relational
  reasoning: A meta-analysis of 47 functional magnetic resonance studies.
  *Journal of Cognitive Neuroscience*, 30(11), 1734–1748.
  doi:10.1162/jocn_a_01311
- **Wu, L.-L., & Barsalou, L. W.** (2009). Perceptual simulation in conceptual
  combination: Evidence from property generation. *Acta Psychologica*,
  132(2), 173–189.
- **Xu, F., & Carey, S.** (1996). Infants' metaphysics: The case of numerical
  identity. *Cognitive Psychology*, 30(2), 111–153.
  doi:10.1006/cogp.1996.0005
- **Xu, F., & Kushnir, T.** (2013). Infants are rational constructivist
  learners. *Current Directions in Psychological Science*, 22(1), 28–32.
  doi:10.1177/0963721412469396
- **Xu, F., & Tenenbaum, J. B.** (2007). Word learning as Bayesian inference.
  *Psychological Review*, 114(2), 245–272.
- **Zhang, Y., Han, K., Worth, R., & Liu, Z.** (2020). Connecting concepts in
  the brain by mapping cortical representations of semantic relations.
  *Nature Communications*, 11, 1877. doi:10.1038/s41467-020-15804-w

---

## 10. One-paragraph answer

The brain does not retrieve a relation label for a novel concept pair; it
**derives** one by **structural alignment against a small inventory of learned
relational prototypes**, and it **hedges or defers when the alignment is
ambiguous**. The mechanism is comparison-based abstraction: DORA (Doumas et
al. 2008, 2022) and BART (Lu et al. 2012, 2019) both learn explicit relational
predicates from non-relational inputs by intersecting what is invariant across
aligned exemplars, and the behavioural literature shows that comparison — not
feature overlap, not mere multiplicity of examples — is what creates a
relational abstraction (Christie & Gentner 2010; Doumas & Hummel 2013;
Gentner et al. 2011). Relational categories are a distinct, theory-like kind
of category (Gentner & Kurtz 2005) with a small cross-linguistically stable
typology (Bejar et al. 1991; Jurgens et al. 2012), and they are neurally
encoded independently of the concepts they connect (Zhang et al. 2020;
Chiang et al. 2021). The judgments are graded — typicality gradients,
multi-relation pairs, a distinct uncertainty computation (Popov et al. 2020;
Ichien et al. 2022; Ragni et al. 2016) — and semantic similarity predicts
none of the relational tasks once relational and role similarity are
controlled. A small system should therefore: use a stored label when it has
one; otherwise align the relational direction vector against a two-tier
inventory (innate core-knowledge primitives plus learned relation prototypes
estimated from the graph's own typed edges); gate on a **softmax margin**, not
a raw dot product; and commit, hedge with an explicitly marked construction,
or defer — never persisting an inferred type, because a 60%-confidence guess
must not become a stored fact. RAVANA's `_infer_relation_type` is already
this mechanism, and its measured 75.5% recovery with monotone
confidence-accuracy is the empirical license for the gate; what is missing is
the two-tier inventory, the margin, and the hedge/defer policy.
