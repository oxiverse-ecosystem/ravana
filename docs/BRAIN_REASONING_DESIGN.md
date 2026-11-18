# RAVANA THINKING front — brain-grounded design specification

**Status:** research + design recommendation. No code changed.
`scripts/measure_fronts.py` is owned by another agent and was not touched, and no
engine was run for this document.
**Scope:** what the THINKING front *should* measure, grounded in peer-reviewed
cognitive neuroscience and cognitive psychology. This answers the five design
questions (a)–(e) and specifies probe criteria that a definition with a relative
clause cannot pass.

## The measurement failure this spec must survive

Three results from this session were probe bugs, not system bugs:

1. a causal regex counted the relativizers `that` / `which` as causal, so
   "a device **that** stores chemical energy" scored as reasoning ("100 fixed");
2. a `multi` criterion required a period, so a correct 2-hop mechanism scored BAD
   ("honestly 0, architecture broken");
3. a census iterated `graph.nodes` (a **dict**) as if it yielded nodes, so labels
   read as `''` ("learning does not persist").

The system was unchanged in all three cases. The consequence for this document
is a hard constraint:

> **No criterion below is a regex over punctuation or a single connective, and
> no criterion scores a string in isolation.** Every criterion is a check on a
> structured causal-chain object cross-validated against the graph. The first run
> of any new rubric is a *draft of the instrument*, not a measurement; write the
> probes, read the answers, then fix the rubric, and print every chain beside its
> verdict (neuroscience-to-architecture skill, "measurement discipline").

---

## 0. Headline recommendation

**Measure the decision, not the prose.**

THINKING should score: *was a grounded, direction-correct, role-bound causal
chain constructed and asserted when a trusted one exists in the graph — or was
a gap honestly declared when none exists?* A definition is a FAIL in both cases.

A passing answer must rest on a structured chain with all six of these
properties:

1. **≥2 causal hops**, not one retrieved clause;
2. a **shared intermediate entity** handed from hop *i* to hop *i+1*;
3. a **process / state-change predicate** on every hop, not a stative copula;
4. **trusted provenance** on every hop (the string is only the realization of
   the chain, not the evidence for it);
5. **direction** matching the query (`why` → chain ends at subject; `how` → chain
   starts at subject);
6. **load-bearing hops**: deleting any one edge must break the chain.

Where no such chain exists, the only passing output is an explicit gap
statement, accepted **only** because an independent (correctly iterated) graph
census confirms no trusted chain exists. A system that declines everything is
caught by the symmetric guard: *a decline when a chain does exist is a FAIL*.

This changes what the front means: from "produced mechanism-sounding text" to
"made the right mechanism-or-gap decision." The two currently failing probes
(`explain how a battery works`, `what would happen if the moon disappeared`) fail
today because they emit definitions; under this spec they still fail until they
either build a real chain **or** honestly decline — which is the correct
behaviour when the graph has no trusted chain.

---

## (a) How the brain decides between retrieving a stored fact and constructing a causal mechanism

**There is no binary switch. Two mechanisms must be distinguished.**

### a.1 The retrieve-vs-construct decision is a thresholded race, not a dispatch flag

The decision is a race whose threshold is set by at least three computable
signals. Retrieval strength of the best stored candidate rises with practice and
consistency: Logan's (1988) instance theory states that automatization is a
shift from algorithm-based performance to **retrieval of stored instances**, with
practice increasing the amount retrieved and consistency making retrieved
instances useful (*Psychological Review*, 95(4):492–527). The competing signals
are prediction error / free energy (Friston, 2010) and competitive density
(lateral inhibition in dual-route and competitive-selection accounts: Coltheart
et al., 2001; Ullman, 2001). The threshold rises with prediction error and falls
with practice — so **high prediction error makes retrieval *more* likely, not
less** (conservative response to uncertainty); construction is reached when
retrieval *fails*, not simply when uncertainty is high. The sign of this term is
the single most important thing to get right if the rule is ever wired in.

The behavioural signature of the two routes is Sloman's (1996) classical
dissociation: the associative system reflects similarity and temporal contiguity
and gives "quick and dirty" answers; the rule-based system operates on symbolic
structures with variables, captures *"logical, hierarchical, and causal-mechanical"*
structure, and **can provide explanations of the environment** (*Psychological
Bulletin*, 119(1):3–22). Note that this distinction was itself contested in
commentaries in the same issue — treat "two systems" as a useful and well-
supported working distinction, not a settled anatomical fact.

### a.2 Construction is recombination of retrieved material, not invention ex nihilo

Schacter and Addis's (2007) **constructive episodic simulation hypothesis** is
the direct answer to "what does constructing consist of?": the brain retrieves
details of past experience and **flexibly recombines** them into a novel
scenario; remembered past and imagined future events rely on many of the same
processes (*Philosophical Transactions of the Royal Society B*, 362:773–786;
DOI 10.1098/rstb.2007.2087; see also Schacter, Addis & Buckner, 2007,
*Nature Reviews Neuroscience*, 8:657–661). This is why a pure retrieval account
and a pure invention account are both wrong: construction is retrieval *plus*
recombination under a goal.

At the neural level, a single association and a causal judgement are
dissociable with **identical stimuli**: Satpute, Fenker, Waldmann, Tabibnia,
Holyoak and Lieberman (2005) had participants judge the same word pairs
(`moon–tide`, `wind–erosion`) as causally or associatively related; causal >
associative recruited left DLPFC (BA 46) and right precuneus, associative >
causal recruited right superior temporal gyrus (*European Journal of
Neuroscience*, 22(5):1233–1238). The proposed operation is **role binding**:
hold the referents bound to the asymmetric roles `cause` and `effect`. A
multi-step mechanism additionally requires **relational integration** — several
bound pairs held simultaneously. Christoff, Prabhakaran, Dorfman, Zhao, Kroger,
Holyoak and Gabrieli (2001) found lateral BA10 (RLPFC) disproportionately engaged
by 2-relation over 1-relation problems (*NeuroImage*, 14(5):1136–1149); Kroger,
Sabb, Fales, Bookheimer, Cohen and Holyoak (2002) found a parametric DLPFC
response to relational complexity (*Cerebral Cortex*, 12(5):477–485); Bunge,
Helskog and Wendelken (2009) found the stringent effect to be **left** RLPFC
(*NeuroImage*, 46(1):338–342). Duncan's (2010) multiple-demand system supplies
the control structure that assembles sub-tasks into a mental program (*Trends in
Cognitive Sciences*, 14(4):172–179).

### a.3 The decision is not sufficient: production must commit to a proposition

Getting the route right does not make output true. Nozari, Dell and Schwartz
(2011) showed that error detection in speech production correlates with
**production-side conflict**, not comprehension, and dissociates from
comprehension in aphasia (*Cognitive Psychology*, 63(1):1–33). The brain commits
to a lexical-semantic plan *before* form encoding (Levelt, 1989; Levelt, Roelofs
& Meyer, 1999). The implication for THINKING: a fluent causal-sounding string
with no committed, graph-supported proposition is confabulation, and the correct
response is to withhold. An "honest I don't know" from the retrieval path beats
fluent confabulation.

**Measurable RAVANA state (already present, by name):** `relation_type == "causal"`
edges, `enumerate_chains` / `grade_chain` (`learn/explanation.py`), provenance
classes (`web_fact`, `curated`), and the question router's `why`/`how` → causal
mapping. The rule must use these names; it must not invent a new lexical proxy
such as `"cause" in label`.

---

## (b) Why evolution selected mechanism construction over pure retrieval or pure association

Four convergent reasons, each with an identifiable literature.

1. **Intervention and counterfactual control.** Pure association encodes
   covariation; it cannot represent that causal relations are *asymmetric* and
   support `do(X)`. Sloman and Lagnado (2005, "Do we 'do'?", *Cognitive Science*,
   29(1):5–39) and Waldmann and Hagmayer (2005, "Seeing versus doing", *Journal
   of Experimental Psychology: Learning, Memory, and Cognition*, 31(2):216–227)
   show that people distinguish observation from intervention, and that this
   distinction is **not recoverable from associative strength**. An organism
   that can simulate `do(X)` can choose an action to change the world; one that
   only retrieves associations cannot.
2. **Flexible novel behaviour at acceptable cost.** Daw, Niv and Dayan (2005)
   formalize two controllers: model-free caching (cheap, rigid, insensitive to
   outcome devaluation) and **model-based tree search** (costly, but simulates
   potential trajectories through a learned model of task structure)
   (*Nature Neuroscience*, 8(12):1704–1711). They locate goal-directed control
   in prefrontal cortex and habitual control in dorsolateral striatum. Pure
   retrieval is the caching end of this spectrum: selected when the environment
   is stable and practice is high (Logan, 1988), but it fails when the
   contingency changes. The cost of construction buys generalization to cases
   never experienced.
3. **Free-energy / survival imperative.** Friston (2010, "The free-energy
   principle: a unified brain theory?", *Nature Reviews Neuroscience*,
   11:127–138) argues that adaptive agents must minimize long-term surprise and
   do so by maintaining an **internal generative model of the causes** of their
   sensations; perception is inference about hidden causes, and action changes
   the sensory input to match prediction. A generative model is exactly a
   mechanism-construction capacity: retrieval alone cannot generate the
   predictions needed to keep the agent within its viable repertoire. (Friston's
   claim to be a *unified* theory is contested by the commentaries published
   with it; the narrower claim — inference over hidden causes under a bound on
   surprise — is the load-bearing one.)
4. **Calibration of confidence.** Rozenblit and Keil (2002, "The misunderstood
   limits of folk science: an illusion of explanatory depth", *Cognitive
   Science*, 26(5):521–562) showed that people rate their understanding of
   familiar devices as high until asked for a **step-by-step explanation**, at
   which point confidence collapses. The feeling of knowing from retrieval is
   cheap and often unearned; the *act of constructing the mechanism* is what
   exposes the gap. Keil (2006, *Annual Review of Psychology*, 57:227–254) and
   Lombrozo (2006, *Trends in Cognitive Sciences*, 10(10):464–470) develop the
   functional account: explanation drives learning and inference, not just
   report. A system that never constructs a mechanism cannot detect its own
   illusions of understanding — a direct fitness cost for acting on false
   knowledge.

**Honest limit:** this is a functional explanation, not a demonstrated
evolutionary sequence. No cited study directly compared the fitness of
mechanism-construction versus retrieval over generations. The claim is that the
*computational benefits* above are what the capacity buys; label it as a
functional inference, not a documented selection event.

---

## (c) Observable behavioural signatures that distinguish a genuine mechanism from a definition with a relative clause

A definition with a relative clause ("a battery is a device **that** stores
chemical energy"; "rain is water **that** falls from the sky") is a stative
copula plus a subordinate modifier. It has one relation, no time, no state
change, no intermediate, and no dependency that could be removed. A mechanism
has all five. The literature gives the following discriminating signatures.

| Signature | Definition | Mechanism | Source |
|---|---|---|---|
| **Role-bound asymmetric relations** | one undirected `is-a`; no cause/effect binding | ≥2 directed `cause→effect` pairs held simultaneously | Satpute et al. (2005); Christoff et al. (2001) |
| **Intermediate entity** | none; predicate modifies the subject directly | an entity that is the *effect* of one step and the *cause* of the next | Pacer & Lombrozo (2017) |
| **Process, not state** | copula + relative clause ("is a Y that…") | change-of-state verbs (evaporates, condenses, flows, oxidizes) | Zemla, Sloman, Bechlivanidis & Lagnado (2023); Sloman (1996) |
| **Counterfactual dependence** | removing any clause leaves the definition intact | removing a step breaks the outcome; `do(step)` changes the terminal state | Gerstenberg (2024); Gerstenberg, Goodman, Lagnado & Tenenbaum (2021); Sloman & Lagnado (2005) |
| **Temporal / ordered transition** | atemporal | steps are ordered; duration and order affect judgment | Bramley, Gerstenberg, Mayrhofer & Lagnado (2018), *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 44(12):1880–1910 |
| **Root simplicity / explained intermediates** | no intermediate to explain | preference tracks *unexplained* causes, not total causes | Pacer & Lombrozo (2017), *Journal of Experimental Psychology: General*, 146(12):1761–1780 |
| **Honest gap on absence** | asserts knowledge fluently | confidence drops when asked to unpack; says "I can't construct this" | Rozenblit & Keil (2002) |

The critical point for probe design: **the first row alone is not enough.** A
causal connective (`because`, `leads to`) can be appended to a definition by a
surface generator. The discriminating signatures are the *relational ones* —
shared intermediate, load-bearing step, provenance — because producing them
requires the system to actually have a graph-supported chain.

---

## (d) What "good explanation" means, and what is measurable in a short generated string

| Author | "Good explanation" means | Measurable in a short string? |
|---|---|---|
| **Lombrozo** (2006, *TiCS* 10(10):464–470; 2007, *Cognitive Psychology* 55(3):232–257; Lombrozo & Liquin 2023, *Current Directions in Psychological Science* 32(3)) | causal-mechanical, **selective**, and simpler explanations carry higher prior probability; explanation changes inference and promotes learning. "Explanation is effective because it is selective." | **Partly.** Mechanism-bearing hop count and absence of off-path hops: yes, on the chain object. Simplicity preference itself: no — Vasilyeva, Wilkenfeld & Lombrozo (2017, *Psychonomic Bulletin & Review* 24(4):1436–1450) show it is instrumentally malleable. |
| **Rozenblit & Keil** (2002, *Cognitive Science* 26(5):521–562; Keil 2006, *Annual Review of Psychology* 57:227–254) | step-by-step mechanism visibility; confidence should be calibrated to whether the mechanism can actually be produced. | **Yes — this is the core probe.** Chain length + every intermediate explained + honest gap when absent. |
| **Gerstenberg** (2024, *Trends in Cognitive Sciences* 28(10):924–936; Gerstenberg et al. 2021, *Psychological Review* 128(5):936–975) | counterfactual simulation: a generative model + intervention operator + consequence simulator; an explanation must support "what if this step had not occurred". | **Yes, via graph cross-check** (edge deletion), not via a string marker. Surface "if…then" words are weak evidence. |
| **Sloman** (1996, *Psychological Bulletin* 119(1):3–22; Sloman & Lagnado 2005, *Cognitive Science* 29(1):5–39; 2015, *Annual Review of Psychology* 66:223–247) | a causal model with an asymmetric `do` operator, not covariation or similarity; directional, supports intervention. | **Yes, on the chain object:** direction, role binding, intervention survival. |
| **Friston** (2010, *Nature Reviews Neuroscience* 11:127–138; Friston 2006, *Journal of Physiology—Paris* 100(1–3):70–79) | an explanation is the generative model's inference about hidden causes; "good" means it reduces prediction error / free energy, and uncertainty is admitted when the model lacks the cause. | **No, not directly.** Free energy is not computable from a string. The *observable proxy* is the decision: assert only model-supported causes and decline when none exists. Report this as a proxy, not a measurement of free energy. |

**Summary of measurability.** Directly checkable in the answer: number of causal
steps, shared intermediate, process verbs versus copula, absence of a
relative-clause-only definition, explicit gap statement. Checkable only with a
graph cross-check: provenance, load-bearing hops, root simplicity, direction
against query type, absence of fabricated edges. **Not** checkable from a string:
truth of the causal claim, free-energy reduction, and quality of the underlying
generative model. A short generated string is evidence that a chain was
*realized*; it is not evidence that the chain is *true*.

---

## (e) Probe criteria a definition cannot pass, with the neuro/cognitive citation for each

Each criterion is stated as an **observable check on the structured chain object
plus the graph**, with an explicit note on why a definition cannot pass it and
how a surface-fake is blocked.

### C1 — Causal chaining with a shared intermediate (chain closure)

**Check.** The answer parses into ≥2 causal hops; the effect entity of hop *i* is
the same graph node (or the same coreferring string) as the cause entity of hop
*i+1*; that shared entity is **not** the subject itself. Reject 2-cycles and any
chain that returns to its start node.

**Why a definition fails.** A copula + relative clause has one relation and no
effect→cause handoff, so the check has nothing to bind. Appending "X leads to Y"
to a definition does not create a shared intermediate between two steps.

**Fake blocked.** Arbitrary "A leads to B, and B leads to C" chains are blocked by
C4 (provenance) and C3 (load-bearing), not by string form. Correct implementation
must iterate the graph's nodes/edges correctly (dict → `.items()`/`.values()`),
assert that the causal-edge census is non-empty, and must **not** require a period
or any punctuation mark.

**Citation.** Satpute et al. (2005) — role binding of asymmetric cause/effect in
DLPFC; Christoff et al. (2001) and Bunge, Helskog & Wendelken (2009) —
relational integration of ≥2 relations in RLPFC; Pacer & Lombrozo (2017) — the
intermediate must itself be explained.

### C2 — Process predicate on every hop, not a stative copula or relative clause

**Check.** Every hop's predicate is a change-of-state / process verb (evaporates,
condenses, falls, flows, converts, oxidizes, discharges) or an equivalent typed
causal relation from the graph. Reject an answer whose only relations are
`is/are` + a subordinate clause introduced by `that`, `which`, or `who`. `that`
and `which` are **never** causal connectives.

**Why a definition fails.** By construction its main relation is a stative copula
and its modifier is a relative clause; there is no change of state to detect.
This is the direct guard against measurement error #1.

**Fake blocked.** A process verb pasted onto a definition still fails C1, C3 and
C4. The check is one condition of a conjunction, never a sufficient condition.

**Citation.** Sloman (1996) — the rule-based system captures causal-mechanical,
not merely statistical, structure; Zemla, Sloman, Bechlivanidis & Lagnado
(2023), *Cognition* 239:105551 — supplying a mechanism *increases* preference for
complex explanations, so length alone is not the discriminator; Trabasso & van
den Broek (1985), *Journal of Memory and Language* 24(5):612–630 — causal
connectedness of events, not a list of predicates, is what makes a sequence
explanatory.

### C3 — Counterfactual / deletion test: every hop is load-bearing

**Check.** For each hop, delete that edge on a **frozen snapshot** of the graph
and re-run the chain search. The answer passes only if removing any single hop
breaks the subject→outcome connection (no alternate path of equal or shorter
length survives). A string-level counterfactual marker ("if X had not happened…")
may be reported but is **not** sufficient evidence.

**Why a definition fails.** A definition asserts no dependency, so no deletion
changes anything: there is no load-bearing step. A list of independent
associations also fails, because deleting one link leaves the others intact.

**Fake blocked.** To pass, the graph must actually contain the edges and the
stored structure must make each one necessary. Fabricating the string cannot
fabricate the deletion result.

**Citation.** Gerstenberg (2024) — counterfactual simulation requires a generative
model, an intervention operator and a consequence simulator; Gerstenberg et al.
(2021) — counterfactual simulation model of causal judgment for physical events;
Sloman & Lagnado (2005) — `do`-operator distinction; Pearl (2009), *Causality*,
Cambridge University Press — formal deletion / `do` calculus.

### C4 — Provenance on every hop: the chain object, not the prose, is the evidence

**Check.** Each emitted causal step corresponds to a stored edge with
`relation_type == "causal"` and a trusted provenance class (`web_fact` with the
required source convergence, or reviewed `curated`). Unprovenanced steps fail the
chain. Print the chain with each hop's provenance beside the verdict.

**Why a definition fails.** A retrieved definition is a sentence, not a typed,
provenance-stamped causal edge. "Water vapour leads to clouds" with no backing
edge is a fluent claim, not a grounded inference.

**Fake blocked.** This is the criterion that makes C1–C3 non-gameable by regex,
and it is orthogonal to punctuation (error #2) and to relativizers (error #1).
It also enforces the production constraint: commit to a proposition before form
encoding; if the store lacks the proposition, decline rather than generate.

**Citation.** Friston (2010) — inference must be licensed by an internal
generative model; Nozari, Dell & Schwartz (2011) and Levelt (1989) — commitment
to a plan before form encoding; Rozenblit & Keil (2002) — confidence must track
mechanism visibility, not retrieval fluency.

### C5 — Query-direction correctness, and honest gap as the symmetric pass

**Check.** `why X` → the chain must **end at X** (backward, cause→X); `how does X
work` → the chain must **start at X** (forward); `explain X` → either, or a
cause→X→effect sandwich. If no trusted ≥2-hop chain exists in the required
direction, the passing output is an explicit gap statement **and** a recorded
gap. A decline is accepted **only** when an independent, correctly iterated
graph census confirms the absence; a decline when a chain exists is a FAIL.
Conversely, a definition when no chain exists is a FAIL.

**Why a definition fails.** A definition is directionless and asserts knowledge;
it can satisfy neither "end at X via ≥2 typed hops" nor "detect and report the
absence of a mechanism." This is what the two currently failing probes should be
judged against.

**Fake blocked.** The census oracle prevents both gaming directions: declining
everything and asserting everything. It is also the guard that would have caught
measurement error #3 — assert the census is over real nodes/edges and that the
known causal-edge count is non-empty before trusting any zero result.

**Citation.** Meder, Mayrhofer & Waldmann (2014), *Psychological Review*
121(3):277–301 — diagnostic (backward) reasoning is structure induction, not a
single Bayesian lookup; Gerstenberg (2024) — forward simulation under `do(X)`;
Rozenblit & Keil (2002) — the honest gap is the correct response when the
mechanism cannot be produced; Ackerman & Thompson (2017), *Trends in Cognitive
Sciences* 21(8):607–617 — meta-reasoning terminates by satisficing or resource
limits; Koriat (1993), *Psychological Review* 100(4):609–639 — feeling-of-knowing
is a computed signal from accessible partial information, not direct access.

---

## 6. Scoring rule implied by C1–C5

For each probe, score **1** iff:

```
chain_exists = correctly_iterated_census finds a trusted >=2-hop causal chain
               in query_direction(subject)

if chain_exists:
    pass = chain_ok(C1,C2,C3,C4) and direction_ok(C5)
else:
    pass = honest_gap_statement and gap_recorded and not_definition

AND no emitted relation is a relativizer (that/which/who) treated as causal,
AND the verdict does not depend on punctuation.
```

Report the per-probe chain, its hop provenance, and the verdict. Report the
denominator (3 probes) and label the score a **judgement for movement between
runs**, never a capability ranking. Add ≥10 unseen `why`/`how` probes with
hand-labelled gold chains before quoting any accuracy — a 3-probe scorecard is
an anecdote, and a rubric authored in the same pass as its probes has never seen
a real failure (neuroscience-to-architecture skill, measurement discipline).

---

## 7. Limits and contested evidence

- **Two-system accounts are contested.** Sloman (1996) was criticised in
  commentaries in the same *Psychological Bulletin* issue; Holyoak and Cheng
  (2011, *Annual Review of Psychology* 62:135–163) call their reconciliation a
  "new synthesis." Treat the associative/rule distinction as a working
  distinction.
- **Free energy as a unified brain theory is contested** (commentaries on
  Friston 2010). Only the inference-over-hidden-causes claim is used here, and
  only as a proxy for honest assertion.
- **RLPFC laterality is disputed** — Bunge et al. (2009) find left only; other
  studies report bilateral. Do not build a criterion on "RLPFC does X" alone.
- **Root simplicity is a single-lab result** (Pacer & Lombrozo 2017) and
  explanation-quality preferences are context-dependent (Vasilyeva et al. 2017;
  Johnson, Valenti & Keil 2019, *Cognitive Psychology* 113:101222). The
  simplicity term should be a tunable weight, not a fixed constant.
- **The self-explanation effect is weak for factual knowledge**: Bisra et al.
  (2018, *Educational Psychology Review* 30(3):703–725) report g = 0.55 with
  heterogeneity, and Harders and Ebersbach (2026, *Applied Cognitive Psychology*
  40(3), DOI 10.1002/acp.70174) find no causal self-explanation benefit for
  factual recall. Do not assume "explaining drives learning" for RAVANA's
  factual `why`/`how` queries; measure it.
- **A short string cannot establish truth.** C1–C5 establish that a
  graph-supported mechanism was constructed and realized; they cannot establish
  that the graph's causal edges are correct. Per-edge provenance review remains a
  separate task (`docs/AUTHORED_CAUSAL_SKELETON.md`).

---

## 8. References (all peer-reviewed; DOIs verified against Crossref / publisher pages)

- Ackerman, R., & Thompson, V. A. (2017). Meta-reasoning: Monitoring and control of thinking and reasoning. *Trends in Cognitive Sciences, 21*(8), 607–617. DOI 10.1016/j.tics.2017.05.004
- Bunge, S. A., Helskog, E. H., & Wendelken, C. (2009). Left, but not right, rostrolateral prefrontal cortex meets a stringent test of the relational integration hypothesis. *NeuroImage, 46*(1), 338–342. DOI 10.1016/j.neuroimage.2009.01.064
- Bramley, N. R., Gerstenberg, T., Mayrhofer, R., & Lagnado, D. A. (2018). Time in causal structure learning. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 44*(12), 1880–1910. DOI 10.1037/xlm0000548
- Christoff, K., Prabhakaran, V., Dorfman, J., Zhao, Z., Kroger, J. K., Holyoak, K. J., & Gabrieli, J. D. E. (2001). Rostrolateral prefrontal cortex involvement in relational integration during reasoning. *NeuroImage, 14*(5), 1136–1149. DOI 10.1006/nimg.2001.0922
- Coltheart, M., Rastle, K., Perry, C., Langdon, R., & Ziegler, J. (2001). DRC: a dual route cascaded model of visual word recognition and reading aloud. *Psychological Review, 108*(1), 204–256.
- Daw, N. D., Niv, Y., & Dayan, P. (2005). Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control. *Nature Neuroscience, 8*(12), 1704–1711. DOI 10.1038/nn1560
- Duncan, J. (2010). The multiple-demand (MD) system of the primate brain: mental programs for intelligent behaviour. *Trends in Cognitive Sciences, 14*(4), 172–179. DOI 10.1016/j.tics.2010.01.004
- Friston, K. (2006). A free energy principle for the brain. *Journal of Physiology—Paris, 100*(1–3), 70–79. DOI 10.1016/j.jphysparis.2006.10.001
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience, 11*, 127–138. DOI 10.1038/nrn2787
- Gerstenberg, T. (2024). Counterfactual simulation in causal cognition. *Trends in Cognitive Sciences, 28*(10), 924–936. DOI 10.1016/j.tics.2024.04.012
- Gerstenberg, T., Goodman, N. D., Lagnado, D. A., & Tenenbaum, J. B. (2021). A counterfactual simulation model of causal judgments for physical events. *Psychological Review, 128*(5), 936–975. DOI 10.1037/rev0000281
- Harders, B., & Ebersbach, M. (2026). No causal self-explanation effect for factual knowledge. *Applied Cognitive Psychology, 40*(3). DOI 10.1002/acp.70174
- Holyoak, K. J., & Cheng, P. W. (2011). Causal learning and inference as a rational process: the new synthesis. *Annual Review of Psychology, 62*, 135–163. DOI 10.1146/annurev.psych.121208.131634
- Johnson, S. G. B., Valenti, J. J., & Keil, F. C. (2019). Simplicity and complexity preferences in causal explanation: an opponent heuristic account. *Cognitive Psychology, 113*, 101222. DOI 10.1016/j.cogpsych.2019.05.004
- Keil, F. C. (2006). Explanation and understanding. *Annual Review of Psychology, 57*, 227–254. DOI 10.1146/annurev.psych.57.102904.190100
- Koriat, A. (1993). How do we know that we know? The accessibility model of the feeling of knowing. *Psychological Review, 100*(4), 609–639. DOI 10.1037/0033-295X.100.4.609
- Kroger, J. K., Sabb, F. W., Fales, C. L., Bookheimer, S. Y., Cohen, M. S., & Holyoak, K. J. (2002). Recruitment of anterior dorsolateral prefrontal cortex in human reasoning: a parametric study of relational complexity. *Cerebral Cortex, 12*(5), 477–485. DOI 10.1093/cercor/12.5.477
- Levelt, W. J. M. (1989). *Speaking: From Intention to Articulation*. MIT Press.
- Levelt, W. J. M., Roelofs, A., & Meyer, A. S. (1999). A theory of lexical access in speech production. *Behavioral and Brain Sciences, 22*(1), 1–38.
- Lombrozo, T. (2006). The structure and function of explanations. *Trends in Cognitive Sciences, 10*(10), 464–470. DOI 10.1016/j.tics.2006.08.004
- Lombrozo, T. (2007). Simplicity and probability in causal explanation. *Cognitive Psychology, 55*(3), 232–257. DOI 10.1016/j.cogpsych.2006.09.006
- Lombrozo, T., & Liquin, E. G. (2023). Explanation is effective because it is selective. *Current Directions in Psychological Science, 32*(3). DOI 10.1177/09637214231156106
- Logan, G. D. (1988). Toward an instance theory of automatization. *Psychological Review, 95*(4), 492–527. DOI 10.1037/0033-295X.95.4.492
- Meder, B., Mayrhofer, R., & Waldmann, M. R. (2014). Structure induction in diagnostic causal reasoning. *Psychological Review, 121*(3), 277–301. DOI 10.1037/a0035944
- Nozari, N., Dell, G. S., & Schwartz, M. F. (2011). Is comprehension necessary for error detection? A conflict-based account of monitoring in speech production. *Cognitive Psychology, 63*(1), 1–33. DOI 10.1016/j.cogpsych.2011.05.001
- Pacer, M., & Lombrozo, T. (2017). Ockham's razor cuts to the root: Simplicity in causal explanation. *Journal of Experimental Psychology: General, 146*(12), 1761–1780. DOI 10.1037/xge0000318
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. DOI 10.1017/cbo9780511803161
- Rozenblit, L., & Keil, F. (2002). The misunderstood limits of folk science: an illusion of explanatory depth. *Cognitive Science, 26*(5), 521–562. DOI 10.1207/s15516709cog2605_1
- Satpute, A. B., Fenker, D. B., Waldmann, M. R., Tabibnia, G., Holyoak, K. J., & Lieberman, M. D. (2005). An fMRI study of causal judgments. *European Journal of Neuroscience, 22*(5), 1233–1238.
- Schacter, D. L., & Addis, D. R. (2007). The cognitive neuroscience of constructive memory: remembering the past and imagining the future. *Philosophical Transactions of the Royal Society B, 362*(1481), 773–786. DOI 10.1098/rstb.2007.2087
- Schacter, D. L., Addis, D. R., & Buckner, R. L. (2007). Remembering the past to imagine the future: the prospective brain. *Nature Reviews Neuroscience, 8*, 657–661. DOI 10.1038/nrn2213
- Sloman, S. A. (1996). The empirical case for two systems of reasoning. *Psychological Bulletin, 119*(1), 3–22. DOI 10.1037/0033-2909.119.1.3
- Sloman, S. A., & Lagnado, D. A. (2005). Do we "do"? *Cognitive Science, 29*(1), 5–39. DOI 10.1207/s15516709cog2901_2
- Sloman, S. A., & Lagnado, D. A. (2015). Causality in thought. *Annual Review of Psychology, 66*, 223–247. DOI 10.1146/annurev-psych-010814-015135
- Trabasso, T., & van den Broek, P. (1985). Causal thinking and the representation of narrative events. *Journal of Memory and Language, 24*(5), 612–630. DOI 10.1016/0749-596X(85)90049-X
- Ullman, M. T. (2001). A neurocognitive perspective on language: the declarative/procedural model. *Nature Reviews Neuroscience, 2*, 717–726.
- Vasilyeva, N., Wilkenfeld, D., & Lombrozo, T. (2017). Contextual utility affects the perceived quality of explanations. *Psychonomic Bulletin & Review, 24*(4), 1436–1450. DOI 10.3758/s13423-017-1275-y
- Waldmann, M. R., & Hagmayer, Y. (2005). Seeing versus doing: two modes of accessing causal knowledge. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 31*(2), 216–227. DOI 10.1037/0278-7393.31.2.216
- Zemla, J. C., Sloman, S. A., Bechlivanidis, C., & Lagnado, D. A. (2023). Not so simple! Causal mechanisms increase preference for complex explanations. *Cognition, 239*, 105551. DOI 10.1016/j.cognition.2023.105551

## Sources

- https://doi.org/10.1038/nrn2787 — Friston (2010)
- https://doi.org/10.1098/rstb.2007.2087 — Schacter & Addis (2007)
- https://doi.org/10.1038/nrn2213 — Schacter, Addis & Buckner (2007)
- https://doi.org/10.1038/nn1560 — Daw, Niv & Dayan (2005)
- https://doi.org/10.1016/j.cogpsych.2011.05.001 — Nozari, Dell & Schwartz (2011)
- https://doi.org/10.1037/0033-295X.95.4.492 — Logan (1988)
- https://doi.org/10.1037/0033-2909.119.1.3 — Sloman (1996)
- https://doi.org/10.1016/j.tics.2024.04.012 — Gerstenberg (2024)
- https://doi.org/10.1037/rev0000281 — Gerstenberg et al. (2021)
- https://doi.org/10.1207/s15516709cog2605_1 — Rozenblit & Keil (2002)
- https://doi.org/10.1016/j.tics.2006.08.004 — Lombrozo (2006)
- https://doi.org/10.1016/j.cogpsych.2006.09.006 — Lombrozo (2007)
- https://doi.org/10.1016/j.cognition.2023.105551 — Zemla et al. (2023)
- https://doi.org/10.1037/xge0000318 — Pacer & Lombrozo (2017)
- https://doi.org/10.1207/s15516709cog2901_2 — Sloman & Lagnado (2005)
- https://doi.org/10.1037/0033-2909.119.1.3 — Sloman (1996), two-system dissociation
