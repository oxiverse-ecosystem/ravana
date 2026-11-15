# Brain Mechanism Assembly

**Status:** research summary + implementation proposal. No code changed.
**Scope:** the ASSEMBLY step — how the brain produces a multi-step causal
mechanism on demand ("because A, which makes B, so C"), and what RAVANA should
implement to do the same.
**Companion doc:** `docs/BRAIN_EXPLANATION_AND_REASONING.md` covers causal vs
associative judgement, the causal-model tradition, RLPFC relational integration
overview, the multiple-demand network, anterior-PFC metacognitive monitoring, the
illusion of explanatory depth, and explanation-vs-description. This document
extends it and does not repeat it.

**The measured defect this addresses.** For `why does X` / `how does X work`,
RAVANA returns a definition or an association, never a mechanism. Definitions
and associations are retrieved fine. The gap is assembly: producing a multi-step
`because A → B → therefore C` structure.

---

## 0. Executive summary

Assembly is the missing stage. Retrieval of a definition or an association is
not the hard part; the hard part is *composing* a sequence of typed causal links
into a single mechanism, and then *judging* whether the composed sequence
explains anything. Five findings drive the design below.

1. **A mechanism is a chain of role-bound relations, not a set of neighbours.**
   The neuroscience of causal judgement (Satpute et al. 2005) and of relational
   reasoning (Christoff et al. 2001; Kroger et al. 2002; Bunge et al. 2009)
   converges on the same operation: hold two referents in working memory *bound
   to the roles `cause` and `effect`*, and integrate across several such pairs.
   A definition has no role binding; an association has one undirected link.
   Neither is a mechanism.
2. **The unit of assembly is the intermediate node, and the failure mode is the
   unexplained intermediate.** A listener accepts a chain when each adjacent
   pair is itself intelligible; the chain breaks at the first link the listener
   cannot simulate. Pacer & Lombrozo (2017) show that people prefer *root*
   simplicity — few unexplained causes — over *node* simplicity, which is
   exactly a preference for chains whose intermediate steps are filled in.
3. **Direction is not cosmetic.** `why X` needs a backward walk (cause → X);
   `how does X work` needs a forward walk (X → effect). RAVANA's assembler
   accepted a `direction` argument and then ignored it, so a `why` query
   searched effects and returned nothing (measured, `learn/explanation.py:171`).
4. **Length is a cost, and the optimum is short but not minimal.** Lombrozo
   (2007) finds simpler explanations are preferred *and* assigned higher prior
   probability; but Zemla et al. (2023) find that supplying a mechanism
   *increases* preference for a longer, more complex explanation. The
   resolution: penalise length per se, but stop penalising (and start crediting)
   a hop that carries a mechanism predicate. RAVANA's current scorer
   (`score_chain`) has only the first half of that rule.
5. **The gate must be a separate reader over the chain, not a property of the
   builder** (Nelson & Narens 1990). And it must fail closed: a chain of
   unprovenanced co-occurrence edges grades 1.000 on structure alone and is
   still false (measured in `docs/AUTHORED_CAUSAL_SKELETON.md`).

**The one-sentence design rule:** a mechanism is a *short, direction-correct,
role-bound causal path whose every intermediate node is itself explained* — and
the verdict on it is computed from the path's structure and provenance, never
from the salience of its endpoints.

### 0.1 What RAVANA already has (verified by reading the code)

| Primitive | Location | State |
|---|---|---|
| `enumerate_chains(subject_ids, graph, direction, min_steps, max_steps, limit)` | `ravana/src/ravana/learn/explanation.py:147` | implemented; direction bug fixed; **only caller passes `direction="forward"`** (`response_gen.py:6396`) |
| `Hop` / `ExplanationChain` / `score_chain` / `grade_chain` / `assertable_chain` | `learn/explanation.py:51,68,138,303,291` | implemented; `CAUSAL_RELATIONS = {"causal","physical_cause"}` |
| `_seed_physics_causal()` | `chat/engine_reasoning.py:3185` | 24+ authored `causal` edges, stamped `edge_kind="curated"` by `_ensure_relation` (`engine_graph.py:922`) |
| `_causal_forward_simulate` | `chat/chain_walker.py:2246` | forward-only, returns strings, no provenance filter |
| `_causal_chain_search` | `chat/chain_walker.py:2206` | dead (requires an end node) |
| `grade_chain` verdicts | `learn/explanation.py:326` | `assert` / `hedge` / `decline`; `assert` is downgraded to `decline` unless every hop is provenance-trusted |

### 0.2 Measured on the live graph (reproduced)

`uv run python scripts/measure_explanation_chains.py` (read-only) on the current
weights:

```
[exp] graph nodes=295 edges=1072 causal=129
[exp] subjects scanned: 295
[exp] subjects with a 2-4 hop causal chain: 56
[exp] verdicts: {'assert': 13, 'hedge': 0, 'decline': 43}
```

Three things this measurement shows, and one it refutes.

- **56 of 295 subjects have a 2–4 hop causal chain.** Assembly is not starved of
  material; roughly one subject in five has a chain.
- **13 of those 56 assert.** So the gate is not vacuously declining; it fires.
- **`hedge: 0`.** The middle band is empty. `grade_chain`'s `adequacy` is
  `typed + prov − 0.5·gaps + 0.5·margin`, clamped to [0,1]; with `typed=1` and
  `prov≈0.9` the floor is already ~0.9, so nothing lands in [0.40, 0.70). The
  three-way verdict is effectively two-way. That is a calibration defect, not a
  research finding, and it means the "honest hedge" branch is dead code.
- **Refuted: the "co-occurrence noise" reading.** The chains that assert are
  `air → sound → hear`, `plants → animals → plants`, `heat → climate → heat`,
  `earth → sun → light` — all walks over the authored `_seed_physics_causal`
  skeleton, all stamped `curated`, all assertable. The chains that decline do so
  because a hop is unprovenanced, not because the chain is structurally bad.
  This is the `docs/AUTHORED_CAUSAL_SKELETON.md` finding, now measured at scale:
  **the provenance gate is doing the work; the structural gate is not.**

**The gap is not the chain walker.** It is that (a) `why` never gets a backward
walk, (b) the scorer has no mechanism credit, (c) the gate has no
intermediate-node check, and (d) nothing records the gap when the gate declines.

---

## 1. The relational-integration mechanism: chaining 2–4 relational steps

### 1.1 Rostrolateral PFC (BA10) and the multiple-demand network

The foundational study is **Christoff, Prabhakaran, Dorfman, Zhao, Kroger,
Holyoak & Gabrieli (2001), *Rostrolateral prefrontal cortex involvement in
relational integration during reasoning*, NeuroImage 14(5):1136–1149.** Under
three conditions — 1-relation, 2-relation, 3-relation — lateral BA 10 was
disproportionately engaged by the 2-relation condition, while DLPFC responded
parametrically to relational complexity. **Kroger, Sabb, Fales, Bookheimer,
Cohen & Holyoak (2002), *Cerebral Cortex* 12(5):477–485** confirms the
parametric DLPFC response. The lateralisation survives a stringent control:
**Bunge, Helskog & Wendelken (2009), *NeuroImage* 46(1):338–342** — it is
specifically *left* RLPFC.

> **Correction to the companion doc.** `BRAIN_EXPLANATION_AND_REASONING.md`
> §2.1 attributes the "stringent test" paper to Smith, Keramatian, Smallwood &
> Christoff (2013), *Neuropsychologia* 51(13):2724–2733. I could not verify
> that paper: Crossref and OpenAlex both resolve the title to **Bunge, Helskog &
> Wendelken (2009), *NeuroImage* 46(1):338–342, DOI
> 10.1016/j.neuroimage.2009.01.064**, and no record of a Smith/Keramatian/
> Smallwood/Christoff paper at *Neuropsychologia* 51(13):2724 exists in either
> index. See §8, ledger item U1. The claim itself (left-lateralised relational
> integration) is unaffected; the citation is wrong and should be fixed.

The multiple-demand (MD) network supplies the control structure.
**Duncan (2010), *Trends in Cognitive Sciences* 14(4):172–179** proposes MD
assembles a series of sub-tasks into a structured mental program; **Duncan
(2013), *Neuron* 80(1):35–50** reframes cognition as a sequence of
*attentional episodes* (select → act → reset). A chain of explanation is
exactly such a sequence, and **Woolgar, Duncan, Manes & Fedorenko (2018),
*Nature Human Behaviour* 2(3):200–204** shows the MD system — not the language
system — supports fluid intelligence. That dissociation is the licence for
RAVANA's architecture: the chain assembler and the surface realizer should be
separable modules, and the assembler should return a *structured object*, not
a sentence.

### 1.2 What the actual computational operation is

Relational integration is not "retrieve a longer list". It is:

1. **Role binding.** Bind each referent to a role. Causal relations are
   asymmetric, so `A → B` and `B → A` are different facts and cannot share a
   representation. Satpute et al. (2005) make this the discriminator between
   causal and associative judgement.
2. **Simultaneous maintenance.** Hold 2–4 bound pairs active at once so the
   *composition* can be evaluated, not just the last link.
3. **Composition / transitive closure with a Markov check.** The composed
   relation is only as valid as the conditional independence it assumes.
   **von Sydow, Hagmayer & Meder (2015), *Memory & Cognition* 44(3):469–487**
   show people *distort* induction over `A → B → C` by treating it as
   transitive when the Markov condition does not hold. A mechanism assembler
   must therefore treat a 3-hop chain as a *claim about a path*, not as a
   derived fact about `A` and `C`.
4. **Selection under a simplicity prior.** Fewer causes → higher prior
   probability (Lombrozo 2007). Selection, not enumeration, is the output.

Step 3 is the one RAVANA's assembler skips entirely: `enumerate_chains`
returns every simple path up to `max_steps` and ranks them by
`Σ log(w·c) − λ·len − μ·unprovenanced`. Nothing checks that the composed path
is Markov-consistent, and nothing distinguishes a 3-hop path that is a genuine
mechanism from one that is a coincidence of two independent causes.

### 1.3 How many hops, and what limits them

Two constraints, and they pull in opposite directions:

- **Working-memory / integration cost.** Relational integration cost rises with
  the number of relations (Kroger et al. 2002). Four bound pairs is roughly the
  ceiling for a single conscious construction.
- **Listener patience.** Lombrozo (2007): a longer explanation must overcome a
  higher bar. Lombrozo & Liquin (2023), *Current Directions in Psychological
  Science* 32(3):212–219: explanation is effective because it is *selective*.

The reconciliation is that **length is only penalised when it is not
mechanism-bearing**. Zemla, Sloman, Bechlivanidis & Lagnado (2023), *Cognition*
239:105551, across five experiments find that when a causal *mechanism* is
supplied, people prefer the *more complex* explanation. So the cost function
should be:

```
score = Σ_hop [ log(w_hop · c_hop) + β·mechanism(hop) ]  − λ·len − μ·unprovenanced
```

where `mechanism(hop)` is 1 when the hop's predicate names a process rather than
a state (§2.2). RAVANA's `score_chain` has `λ` and `μ` but no `β`.

---

## 2. Mechanism vs association: what makes a chain mechanistic

### 2.1 Causal model vs causal association

A **causal association** is a directed edge: `A` is associated with `B`, and the
association has a direction. A **causal model** is a structure that supports
intervention: `do(A)` changes the distribution of `B`, and the model says what
else changes. The classic demonstrations that people have the second and not
just the first:

- **Waldmann & Hagmayer (2005), *Journal of Experimental Psychology: Learning,
  Memory, and Cognition* 31(2):216–227.** Learners trained on purely
  observational data make *different* predictions depending on whether they
  believe an event was observed ("seeing") or manipulated ("doing"). The
  predictions reflect both structure and parameters, and the intervention
  predictions differ from the observed patterns. Associative and probabilistic
  theories fail to predict this.
- **Sloman & Lagnado (2005), *Cognitive Science* 29(1):5–39** and
  **Sloman & Lagnado (2015), *Annual Review of Psychology* 66:223–247** — the
  observational/interventional distinction is not recoverable from associative
  strength.
- **Griffiths & Tenenbaum (2009), *Psychological Review* 116(4):661–716** —
  causal induction is Bayesian inference over a *theory-constrained* hypothesis
  space (causal power, causal linearity), and people's judgements deviate from
  pure contingency models in exactly the way that predicts.

The assembler consequence: **an edge is not a mechanism by being labelled
`causal`.** It is a mechanism by participating in a structure that would
survive an intervention. RAVANA's `_seed_physics_causal` edges are labelled
`causal` and stamped `curated`, and at least one of them (`gravity → earth`)
was not a causal prior at all — it was co-occurrence wearing a causal label
(`docs/AUTHORED_CAUSAL_SKELETON.md`). The label is not the mechanism.

### 2.2 What a listener checks

Zemla, Sloman, Bechlivanidis & Lagnado (2017), *Psychonomic Bulletin & Review*
24(5):1488–1500 and Bechlivanidis, Lagnado, Zemla & Sloman (2017), *ibid*
24(5):1451–1464 establish that people evaluate explanations on *coherence* and
*concreteness*, and that these are separable. A listener's checks, in order:

1. **Is each adjacent pair itself plausible?** A chain is only as strong as its
   weakest link, and the listener stops at the first unintelligible link.
2. **Is the chain direction-correct?** `why X` answered with `X → Y` names a
   consequence, not a cause.
3. **Are the intermediates filled in, or are there unexplained causes?**
   Pacer & Lombrozo (2017), *Journal of Experimental Psychology: General*
   146(12):1761–1780: explanatory preference tracks **root simplicity** (number
   of *unexplained* causes), not node simplicity (number of causes). A chain
   that invokes a cause it never explains is penalised more than a longer chain
   whose causes are all accounted for.
4. **Does it name a process, or only relabel the effect?** Zemla et al. (2023):
   mechanism descriptions increase preference for complexity. A mechanism says
   *how* the outcome came about, not just *what* caused it.
5. **Is it selective?** Lombrozo & Liquin (2023): an explanation that explains
   everything explains nothing.

---

## 3. How many steps does a good explanation have?

The evidence does not support a fixed number. It supports a *trade-off with a
mechanism-dependent exchange rate*:

| Evidence | Finding | Design consequence |
|---|---|---|
| Lombrozo (2007) | Simpler explanations preferred; assigned higher prior probability | penalise length |
| Pacer & Lombrozo (2017) | *Root* simplicity, not node simplicity, drives preference | penalise unexplained intermediates, not hops |
| Zemla et al. (2023) | A mechanism increases preference for a *longer* explanation | credit mechanism-bearing hops |
| Lombrozo & Liquin (2023) | Explanation is effective because selective | cap length; prefer the shortest *sufficient* chain |
| Johnson, Valenti & Keil (2019), *Cognitive Psychology* 113:101222 | Simplicity and complexity preferences are opponent heuristics | do not fit a single global λ; the exchange rate is context-dependent |

**Practical rule for RAVANA:** default `min_steps=2, max_steps=4`; prefer the
*shortest* chain that has no unexplained intermediate and at least one
mechanism-bearing hop; allow a 5th hop only when it is mechanism-bearing and the
shorter alternative has a gap.

---

## 4. Intermediate nodes and the explanatory gap

This is the section the companion doc names but does not operationalise.

**The claim.** An explanation is not a path from A to C. It is a path from A to
C *in which every intermediate node is itself a node the listener can
simulate*. The explanatory gap is not "the chain is short"; it is "the chain
contains a node whose own causal role is unspecified".

**Evidence.** Pacer & Lombrozo (2017) distinguish *node simplicity* (count of
causes invoked) from *root simplicity* (count of *unexplained* causes invoked)
and find preferences track the latter. A 3-hop chain `A → B → C` where `B` is
never explained has root simplicity 1 (B is an unexplained cause); a 4-hop
chain `A → B → D → C` where `D` is explained has root simplicity 0. The second
is preferred even though it is longer. Rozenblit & Keil (2002), *Cognitive
Science* 26(5):521–562 show the feeling of understanding is calibrated to
*mechanism* visibility: the illusion of explanatory depth is strongest where
the environment supports real-time explanation with visible mechanisms.

**Operationalisation for RAVANA.** Add a per-hop `explained` flag and a chain
level `root_count`:

```
root_count = number of hops h such that h.to_label has no outgoing causal edge
             (i.e. the intermediate is a dead end in the causal subgraph)
gap_count  = root_count + (1 if the terminal node is not the explanandum else 0)
```

`grade_chain` currently computes `gaps` as "hops with no provenance AND
confidence < 0.3" — a *confidence* proxy, not a *structural* one. That is the
single largest gap between the implemented gate and the mechanism it is meant
to implement.

---

## 5. Forward vs backward chaining

**The brain does both, and the query type selects.**

- **Forward (X → … → effect).** The counterfactual-simulation literature:
  given `do(X)`, simulate forward along causal edges to *discover* consequences.
  Gerstenberg (2024), *Trends in Cognitive Sciences* 28(10):924–936: simulating
  counterfactuals needs a generative model, an intervention operator, and a
  consequence simulator. Gerstenberg, Goodman, Lagnado & Tenenbaum (2021),
  *Psychological Review* 128(5):936–975 supply the model.
- **Backward (cause → … → X).** Diagnostic reasoning: infer the cause from an
  observed effect. Meder, Mayrhofer & Waldmann (2014), *Psychological Review*
  121(3):277–301: diagnostic reasoning from a single effect to a single cause
  is *not* a simple Bayesian inversion; people use structure-induction
  strategies that differ from the normative model.

**The measured RAVANA defect.** `enumerate_chains` accepted `direction` and
ignored it — both branches called `get_outgoing`, so `direction="backward"`
silently performed a forward search. The in-code comment records the
measurement: with `evaporate → condense → rainfall` injected,
`enumerate_chains([condense], direction="backward")` returned **zero** chains,
because it looked for `condense`'s outgoing edges and the cause is incoming
(`learn/explanation.py:171-188`). The bug is now fixed in the walker, but the
*only production caller* still passes `direction="forward"`
(`response_gen.py:6396`). So `why does it rain` still gets a forward search.
That is the single highest-value fix in this document.

| Query | Direction | Terminal requirement |
|---|---|---|
| `why <X>` | backward (cause → X) | chain must *end at* X |
| `how does <X> work` | forward (X → effect) | chain must *start at* X |
| `what does <X> do` | forward | chain must start at X |
| `explain <X>` | bidirectional | either, or a cause→X→effect sandwich |

---

## 6. Developmental and learning evidence

Assembly is learned, and the learning is *constrained by structure*.

- **Gopnik, Glymour, Sobel, Schulz, Kushnir & Danks (2004), *Psychological
  Review* 111(1):3–32** — children learn causal structure from statistical
  information and from interventions, over a hypothesis space of causal maps /
  Bayes nets.
- **Gopnik & Wellman (2012), *Psychological Bulletin* 138(6):1085–1108** — the
  theory theory rebuilt on probabilistic causal models; children infer causal
  structure from their own actions, from observing others' actions, and from
  pedagogy.
- **Schulz, Gopnik & Glymour (2007), *Developmental Science* 10(3):322–332** —
  preschoolers learn causal structure from *conditional interventions*.
- **Kushnir & Gopnik (2005), *Psychological Science* 16(9):678–683** — young
  children infer causal strength from probabilities and interventions.
- **Legare & Lombrozo (2014), *Journal of Experimental Child Psychology*
  126:198–212** — prompting 3–6-year-olds to explain a mechanical toy
  selectively affects learning about the toy's *causal* properties, not its
  non-causal properties. Explanation is causal-specific.
- **Walker, Lombrozo, Williams, Rafferty & Gopnik (2017), *Child Development*
  88(1):229–246** — self-generated explanation drives children to evaluate
  hypotheses by features of "good" explanations (broad scope, prior-knowledge
  consistency). Explaining *constrains* causal learning rather than merely
  reporting it.
- **Walker, Lombrozo, Legare & Gopnik (2014), *Cognition* 133(2):343–357** —
  explaining prompts children to privilege inductively rich properties.
- **Liquin & Lombrozo (2020), *Cognitive Psychology* 119:101276** —
  explanation-seeking curiosity is triggered by a *contrast* between fact and
  foil: an observation prompts "why?" when an alternative is salient. This is
  the trigger condition for the assembly stage.
- **Ruggeri, Xu & Lombrozo (2019), *Cognition* 191:103966** — explanation
  affects children's question asking.

**Design consequence.** The trigger for assembly is not "the user said why".
It is "the user's utterance reveals a *contrast* — an observation whose
alternative is salient". RAVANA's router keys on the word `why`, which is a
proxy. A better trigger is Liquin & Lombrozo's: the subject has an unexplained
cause in the graph (a gap), *and* the user's turn makes the fact salient.

---

## 7. Derived mechanisms for RAVANA

Seven mechanisms, ordered by expected value. M1 is the highest-value fix and
the cheapest.

### M1 — Direction-correct routing for `why` (fixes the measured defect)

`response_gen.py:6396` passes `direction="forward"` unconditionally. Route on
the detected question type:

```
why <X>        -> direction="backward"
how <X> work   -> direction="forward"
explain <X>    -> direction="bidirectional"
```

**Falsifier:** with two trusted causal edges `evaporate → condense → rainfall`
injected, `why does it rain` must return a chain ending at `rainfall`. Today it
returns `[]` (measured in-code).

### M2 — Mechanism credit in the scorer

Add `β·mechanism(hop)` to `score_chain`. `mechanism(hop)` = 1 when the hop's
predicate names a process (a change of state, an transfer, a transformation),
0 when it names a state or a static property. Evidence: Zemla et al. (2023).

### M3 — Root-simplicity gap count (replaces the confidence proxy)

Replace `grade_chain`'s `gaps` computation with a structural one: count hops
whose `to_label` is a dead end in the causal subgraph, plus 1 if the terminal
node is not the explanandum. Evidence: Pacer & Lombrozo (2017).

### M4 — Markov-consistency check on composed paths

Before asserting a 3+ hop chain, verify the composed `A → C` relation is not
better explained by a direct edge or by a fork. Evidence: von Sydow, Hagmayer
& Meder (2015) — transitive reasoning distorts induction in causal chains.

### M5 — Separate the assembler from the realizer

`enumerate_chains` already returns a structured object; `response_gen.py:6413`
immediately flattens it to `"X leads to Y, and Y leads to Z"`. Keep the object
through the discourse planner (`DiscourseType.CAUSAL_EXPLAIN`,
`engine_generation.py:319`) so the nesting pass can produce a single nested
clause. Evidence: Woolgar et al. (2018) — MD, not language, supports the
reasoning; the two must be separable.

### M6 — Gap recording on decline

When `grade_chain` returns `decline`, record `(subject, missing_direction,
timestamp)` in a gap ledger and queue a web read. Today a decline silently
becomes an honest hedge and nothing is learned. Evidence: Liquin & Lombrozo
(2020) — the contrast is the trigger; Rozenblit & Keil (2002) — the gap is
real and should be felt.

### M7 — Provenance-class review of the authored skeleton

`_seed_physics_causal` stamps every edge `edge_kind="curated"`, which
`provenance_trusted` accepts. The skeleton already had one edge removed
(`gravity → earth`) after it produced a confident false explanation
(`docs/AUTHORED_CAUSAL_SKELETON.md`). The remaining edges need the same
per-edge review; a blanket `curated` stamp is not a truth guarantee.

### M8 — Recalibrate the adequacy bands (measured defect)

§0.2: `hedge: 0` across 56 chains. `adequacy = typed + prov − 0.5·gaps +
0.5·margin` has a floor near 0.9 for any fully-typed chain with `prov≈0.9`, so
`θ_hedge = 0.40` is unreachable. Either renormalise `adequacy` to [0,1] with a
zero point at "no evidence" (e.g. `(typed + prov − 0.5·gaps + 0.5·margin)/3`)
or fit the thresholds with `calibrator.theta_withhold()`. Until this is done,
the `hedge` verdict is dead code and every non-assert is a decline.

### M9 — Kill the 2-cycle chains

The measurement returns `plants → animals → plants`, `heat → climate → heat`,
`system → changes → system` as top-scoring chains. `enumerate_chains` forbids
repeating a *label within a hop list*, but the start node is not in that list,
so a chain may return to its own subject. `plants → animals → plants` asserts at
adequacy 1.000. Add the start node to the visited set. This is a one-line fix
with a large precision payoff.

---

## 8. Unverified claims ledger

| # | Claim | Status | Note |
|---|---|---|---|
| U1 | Smith, Keramatian, Smallwood & Christoff (2013), *Neuropsychologia* 51(13):2724–2733 | **not found** | Cited in `BRAIN_EXPLANATION_AND_REASONING.md` §2.1. Crossref and OpenAlex resolve the title "Left, but not right, rostrolateral prefrontal cortex meets a stringent test of the relational integration hypothesis" to Bunge, Helskog & Wendelken (2009), *NeuroImage* 46(1):338–342, DOI 10.1016/j.neuroimage.2009.01.064. No record of the Smith et al. paper at those coordinates. |
| U2 | Bramley, Gerstenberg, Mayrhofer, Lagnado & Tenenbaum (2018), *Cognitive Psychology* 101:54–82 | **journal wrong** | The companion doc cites *Cognitive Psychology* 101:54–82. Crossref and OpenAlex both give *Journal of Experimental Psychology: Learning, Memory, and Cognition* 44(12):1880–1910, DOI 10.1037/xlm0000548. Author list is Bramley, Gerstenberg, Mayrhofer & Lagnado (no Tenenbaum). |
| U3 | Satpute et al. (2005) coordinate values (−30, 40, 18) etc. | inherited, not re-verified here | Taken from the companion doc; I did not re-read the paper. |
| U4 | Lombrozo & Liquin (2023) page range 32(3):212–219 | volume/issue verified, pages unverified | DOI 10.1177/09637214231156106 verified; the page range was not confirmed from the publisher and should be treated as unverified. |
| U5 | "People prefer 2–4 hop explanations" as a universal constant | **not supported** | The evidence supports a trade-off, not a constant (§3). Stated here as a design default, not a finding. |
| U6 | RAVANA's `_seed_physics_causal` edge count | verified by reading | 24+ edges at `engine_reasoning.py:3185-3244`; the exact count is not asserted. |
| U7 | Harders & Ebersbach (2026), *Applied Cognitive Psychology* 40(3) | **year differs from the companion doc** | The companion doc cites a 2025 *Applied Cognitive Psychology* "no causal self-explanation effect" paper with no authors. Crossref resolves DOI 10.1002/acp.70174 to Harders & Ebersbach, *Applied Cognitive Psychology* 40(3), published 2026. The null result is real; the year and authorship in the companion doc are incomplete. |
| U8 | "Root simplicity" as a general explanation-quality predictor | **single-lab, single-paper** | Pacer & Lombrozo (2017) is the source; I found no independent replication in the searches run for this document. Stated as a design input, not as settled science. |
| U9 | Liquin & Lombrozo (2020) fact-vs-foil trigger | abstract verified, mechanism not | DOI 10.1016/j.cogpsych.2020.101276 verified; the fact/foil contrast claim is from the abstract and a CogSci 2018 companion paper, not from reading the full article. |

**Citation verification method.** Every DOI in §9 was resolved against the
Crossref REST API (`api.crossref.org/works/<doi>`) on the day of writing, and
every PMID against the NCBI E-utilities `esummary` endpoint. Where the two
disagreed with the companion doc, the ledger above records it. Nothing in §9 is
cited from memory.

**Count.** 75 unique DOIs in §9; all 75 resolved successfully against Crossref
(zero failures). One further reference (Satpute et al. 2005) is inherited from
the companion doc without re-verification and is marked inline.

---

## 9. References

**Verified against Crossref and/or PubMed E-utilities.**

- Christoff, K., Prabhakaran, V., Dorfman, J., Zhao, Z., Kroger, J. K., Holyoak,
  K. J., & Gabrieli, J. D. E. (2001). Rostrolateral prefrontal cortex involvement
  in relational integration during reasoning. *NeuroImage, 14*(5), 1136–1149.
  DOI: 10.1006/nimg.2001.0922
- Kroger, J. K., Sabb, F. W., Fales, C. L., Bookheimer, S. Y., Cohen, M. S., &
  Holyoak, K. J. (2002). Recruitment of anterior dorsolateral prefrontal cortex
  in human reasoning: a parametric study of relational complexity.
  *Cerebral Cortex, 12*(5), 477–485. DOI: 10.1093/cercor/12.5.477
- Bunge, S. A., Helskog, E. H., & Wendelken, C. (2009). Left, but not right,
  rostrolateral prefrontal cortex meets a stringent test of the relational
  integration hypothesis. *NeuroImage, 46*(1), 338–342.
  DOI: 10.1016/j.neuroimage.2009.01.064
- Smith, R., Keramatian, K., & Christoff, K. (2007). Localizing the
  rostrolateral prefrontal cortex at the individual level. *NeuroImage, 36*(4),
  1387–1396. DOI: 10.1016/j.neuroimage.2007.04.032
- Waltz, J. A., Knowlton, B. J., Holyoak, K. J., Boone, K. B., Mishkin, F. S., de
  Menezes Santos, M., Thomas, C. R., & Miller, B. L. (1999). A system for
  relational reasoning in human prefrontal cortex. *Psychological Science,
  10*(2), 119–125. DOI: 10.1111/1467-9280.00118
- Crone, E. A., Wendelken, C., van Leijenhorst, L., Honomichl, R. D., Christoff,
  K., & Bunge, S. A. (2009). Neurocognitive development of relational reasoning.
  *Developmental Science, 12*(1), 55–66. DOI: 10.1111/j.1467-7687.2008.00743.x
- Badre, D., & D'Esposito, M. (2009). Is the rostro-caudal axis of the frontal
  lobe hierarchical? *Nature Reviews Neuroscience, 10*(9), 659–669.
  DOI: 10.1038/nrn2667
- Christoff, K., Keramatian, K., Gordon, A. M., Smith, R., Mädler, B., Evans, K.,
  Marois, R., & Hester, R. (2009). Prefrontal organization of cognitive control
  according to levels of abstraction. *Brain Research, 1286*, 94–105.
  DOI: 10.1016/j.brainres.2009.05.096
- Duncan, J. (2010). The multiple-demand (MD) system of the primate brain:
  mental programs for intelligent behaviour. *Trends in Cognitive Sciences,
  14*(4), 172–179. DOI: 10.1016/j.tics.2010.01.004
- Duncan, J. (2013). The structure of cognition: attentional episodes in mind
  and brain. *Neuron, 80*(1), 35–50. DOI: 10.1016/j.neuron.2013.09.015
- Woolgar, A., Duncan, J., Manes, F., & Fedorenko, E. (2018). Fluid intelligence
  is supported by the multiple-demand system not the language system.
  *Nature Human Behaviour, 2*(3), 200–204. DOI: 10.1038/s41562-017-0282-3
- Woolgar, A., Parr, A., Cusack, R., Thompson, R., Nimmo-Smith, I., Torralva,
  T., Roca, M., Antoun, N., Manes, F., & Duncan, J. (2010). Fluid intelligence
  loss linked to restricted regions of damage within frontal and parietal
  cortex. *PNAS, 107*(33), 14899–14902. DOI: 10.1073/pnas.1007928107
- Assem, M., Glasser, M. F., Van Essen, D. C., & Duncan, J. (2020). A
  domain-general cognitive core defined in multimodally parcellated human
  cortex. *Cerebral Cortex, 30*(8), 4361–4380. DOI: 10.1093/cercor/bhaa023
- Koechlin, E., & Summerfield, C. (2007). An information theoretical approach to
  prefrontal executive function. *Trends in Cognitive Sciences, 11*(6), 229–235.
  DOI: 10.1016/j.tics.2007.04.005
- Koechlin, E. (2014). An evolutionary computational theory of prefrontal
  executive function in decision-making. *Philosophical Transactions of the
  Royal Society B, 369*(1655), 20130474. DOI: 10.1098/rstb.2013.0474
- Fedorenko, E., Ivanova, A. A., & Regev, T. I. (2024). The language network as a
  natural kind within the broader landscape of the human brain. *Nature Reviews
  Neuroscience, 25*(5), 289–312. DOI: 10.1038/s41583-024-00802-4
- Satpute, A. B., Fenker, D. B., Waldmann, M. R., Tabibnia, G., Holyoak, K. J.,
  & Lieberman, M. D. (2005). An fMRI study of causal judgments. *European
  Journal of Neuroscience, 22*(5), 1233–1238. (Inherited from the companion doc;
  not re-verified here.)
- Lombrozo, T. (2006). The structure and function of explanations. *Trends in
  Cognitive Sciences, 10*(10), 464–470. DOI: 10.1016/j.tics.2006.08.004
- Lombrozo, T. (2007). Simplicity and probability in causal explanation.
  *Cognitive Psychology, 55*(3), 232–257. DOI: 10.1016/j.cogpsych.2006.09.006
- Pacer, M., & Lombrozo, T. (2017). Ockham's razor cuts to the root: Simplicity
  in causal explanation. *Journal of Experimental Psychology: General, 146*(12),
  1761–1780. DOI: 10.1037/xge0000318
- Vrantsidis, T. H., & Lombrozo, T. (2022). Simplicity as a cue to probability:
  multiple roles for simplicity in evaluating explanations. *Cognitive Science,
  46*(7), e13169. DOI: 10.1111/cogs.13169
- Johnson, S. G. B., Valenti, J. J., & Keil, F. C. (2019). Simplicity and
  complexity preferences in causal explanation: an opponent heuristic account.
  *Cognitive Psychology, 113*, 101222. DOI: 10.1016/j.cogpsych.2019.05.004
- Lombrozo, T., & Liquin, E. G. (2023). Explanation is effective because it is
  selective. *Current Directions in Psychological Science, 32*(3).
  DOI: 10.1177/09637214231156106
- Zemla, J. C., Sloman, S. A., Bechlivanidis, C., & Lagnado, D. A. (2017).
  Evaluating everyday explanations. *Psychonomic Bulletin & Review, 24*(5),
  1488–1500. DOI: 10.3758/s13423-017-1258-z
- Bechlivanidis, C., Lagnado, D. A., Zemla, J. C., & Sloman, S. A. (2017).
  Concreteness and abstraction in everyday explanation. *Psychonomic Bulletin &
  Review, 24*(5), 1451–1464. DOI: 10.3758/s13423-017-1299-3
- Vasilyeva, N., Wilkenfeld, D., & Lombrozo, T. (2017). Contextual utility
  affects the perceived quality of explanations. *Psychonomic Bulletin &
  Review, 24*(5), 1436–1450. DOI: 10.3758/s13423-017-1275-y
- Zemla, J. C., Sloman, S. A., Bechlivanidis, C., & Lagnado, D. A. (2023). Not
  so simple! Causal mechanisms increase preference for complex explanations.
  *Cognition, 239*, 105551. DOI: 10.1016/j.cognition.2023.105551
- Giffin, C., Wilkenfeld, D., & Lombrozo, T. (2017). The explanatory effect of a
  label: explanations with named categories are more satisfying. *Cognition,
  168*, 357–369. DOI: 10.1016/j.cognition.2017.07.011
- Rozenblit, L., & Keil, F. (2002). The misunderstood limits of folk science: an
  illusion of explanatory depth. *Cognitive Science, 26*(5), 521–562.
  DOI: 10.1207/s15516709cog2605_1
- Keil, F. C. (2006). Explanation and understanding. *Annual Review of
  Psychology, 57*, 227–254. DOI: 10.1146/annurev.psych.57.102904.190100
- Sloman, S. A., & Lagnado, D. A. (2005). Do we "do"? *Cognitive Science,
  29*(1), 5–39. DOI: 10.1207/s15516709cog2901_2
- Sloman, S. A., & Lagnado, D. A. (2015). Causality in thought. *Annual Review of
  Psychology, 66*, 223–247. DOI: 10.1146/annurev-psych-010814-015135
- Waldmann, M. R., & Hagmayer, Y. (2005). Seeing versus doing: two modes of
  accessing causal knowledge. *Journal of Experimental Psychology: Learning,
  Memory, and Cognition, 31*(2), 216–227. DOI: 10.1037/0278-7393.31.2.216
- Griffiths, T. L., & Tenenbaum, J. B. (2005). Structure and strength in causal
  induction. *Cognitive Psychology, 51*(4), 334–384.
  DOI: 10.1016/j.cogpsych.2005.05.004
- Griffiths, T. L., & Tenenbaum, J. B. (2009). Theory-based causal induction.
  *Psychological Review, 116*(4), 661–716. DOI: 10.1037/a0017201
- Goodman, N. D., Ullman, T. D., & Tenenbaum, J. B. (2011). Learning a theory of
  causality. *Psychological Review, 118*(1), 110–119. DOI: 10.1037/a0021336
- Tenenbaum, J. B., Griffiths, T. L., & Kemp, C. (2006). Theory-based Bayesian
  models of inductive learning and reasoning. *Trends in Cognitive Sciences,
  10*(7), 309–318. DOI: 10.1016/j.tics.2006.05.009
- Tenenbaum, J. B., Kemp, C., Griffiths, T. L., & Goodman, N. D. (2011). How to
  grow a mind: statistics, structure, and abstraction. *Science, 331*(6022),
  1279–1285. DOI: 10.1126/science.1192788
- Kemp, C., & Tenenbaum, J. B. (2008). The discovery of structural form.
  *PNAS, 105*(31), 10687–10692. DOI: 10.1073/pnas.0802631105
- Kemp, C., Goodman, N. D., & Tenenbaum, J. B. (2010). Learning to learn causal
  models. *Cognitive Science, 34*(7), 1185–1243.
  DOI: 10.1111/j.1551-6709.2010.01128.x
- Holyoak, K. J., & Cheng, P. W. (2011). Causal learning and inference as a
  rational process: the new synthesis. *Annual Review of Psychology, 62*,
  135–163. DOI: 10.1146/annurev.psych.121208.131634
- Cheng, P. W. (1997). From covariation to causation: a causal power theory.
  *Psychological Review, 104*(2), 367–405. DOI: 10.1037/0033-295x.104.2.367
- Meder, B., Mayrhofer, R., & Waldmann, M. R. (2014). Structure induction in
  diagnostic causal reasoning. *Psychological Review, 121*(3), 277–301.
  DOI: 10.1037/a0035944
- Meder, B., Hagmayer, Y., & Waldmann, M. R. (2009). The role of learning data in
  causal reasoning about observations and interventions. *Memory & Cognition,
  37*(3), 249–264. DOI: 10.3758/mc.37.3.249
- von Sydow, M., Hagmayer, Y., & Meder, B. (2015). Transitive reasoning distorts
  induction in causal chains. *Memory & Cognition, 44*(3), 469–487.
  DOI: 10.3758/s13421-015-0568-5
- Hagmayer, Y., & Mayrhofer, R. (2013). Hierarchical Bayesian models as formal
  models of causal reasoning. *Argument & Computation, 4*(1), 36–45.
  DOI: 10.1080/19462166.2012.700321
- Bramley, N. R., Gerstenberg, T., Mayrhofer, R., & Lagnado, D. A. (2018). Time
  in causal structure learning. *Journal of Experimental Psychology: Learning,
  Memory, and Cognition, 44*(12), 1880–1910. DOI: 10.1037/xlm0000548
- Gerstenberg, T. (2024). Counterfactual simulation in causal cognition.
  *Trends in Cognitive Sciences, 28*(10), 924–936.
  DOI: 10.1016/j.tics.2024.04.012
- Gerstenberg, T., Goodman, N. D., Lagnado, D. A., & Tenenbaum, J. B. (2021). A
  counterfactual simulation model of causal judgments for physical events.
  *Psychological Review, 128*(5), 936–975. DOI: 10.1037/rev0000281
- Dorfman, H. M., Tomov, M. S., Cheung, B., Clarke, D., Gershman, S. J., &
  Hughes, B. L. (2021). Causal inference gates corticostriatal learning.
  *Journal of Neuroscience, 41*(32), 6892–6904.
  DOI: 10.1523/JNEUROSCI.2796-20.2021
- Gopnik, A., Glymour, C., Sobel, D. M., Schulz, L. E., Kushnir, T., & Danks, D.
  (2004). A theory of causal learning in children: causal maps and Bayes nets.
  *Psychological Review, 111*(1), 3–32. DOI: 10.1037/0033-295x.111.1.3
- Gopnik, A., & Wellman, H. M. (2012). Reconstructing constructivism: causal
  models, Bayesian learning mechanisms, and the theory theory. *Psychological
  Bulletin, 138*(6), 1085–1108. DOI: 10.1037/a0028044
- Schulz, L. E., Gopnik, A., & Glymour, C. (2007). Preschool children learn about
  causal structure from conditional interventions. *Developmental Science,
  10*(3), 322–332. DOI: 10.1111/j.1467-7687.2007.00587.x
- Kushnir, T., & Gopnik, A. (2005). Young children infer causal strength from
  probabilities and interventions. *Psychological Science, 16*(9), 678–683.
  DOI: 10.1111/j.1467-9280.2005.01595.x
- Sobel, D. M., & Legare, C. H. (2014). Causal learning in children. *WIREs
  Cognitive Science, 5*(4), 413–427. DOI: 10.1002/wcs.1291
- Legare, C. H., & Lombrozo, T. (2014). Selective effects of explanation on
  learning during early childhood. *Journal of Experimental Child Psychology,
  126*, 198–212. DOI: 10.1016/j.jecp.2014.03.001
- Walker, C. M., Lombrozo, T., Williams, J. J., Rafferty, A. N., & Gopnik, A.
  (2017). Explaining constrains causal learning in childhood. *Child Development,
  88*(1), 229–246. DOI: 10.1111/cdev.12590
- Walker, C. M., Lombrozo, T., Legare, C. H., & Gopnik, A. (2014). Explaining
  prompts children to privilege inductively rich properties. *Cognition,
  133*(2), 343–357. DOI: 10.1016/j.cognition.2014.07.008
- Williams, J. J., & Lombrozo, T. (2010). The role of explanation in discovery
  and generalization: evidence from category learning. *Cognitive Science,
  34*(5), 776–806. DOI: 10.1111/j.1551-6709.2010.01113.x
- Liquin, E. G., & Lombrozo, T. (2020). A functional approach to
  explanation-seeking curiosity. *Cognitive Psychology, 119*, 101276.
  DOI: 10.1016/j.cogpsych.2020.101276
- Liquin, E. G., & Lombrozo, T. (2020). Explanation-seeking curiosity in
  childhood. *Current Opinion in Behavioral Sciences, 35*, 14–20.
  DOI: 10.1016/j.cobeha.2020.05.012
- Liquin, E. G., Metz, S. E., & Lombrozo, T. (2020). Science demands explanation,
  religion tolerates mystery. *Cognition, 204*, 104398.
  DOI: 10.1016/j.cognition.2020.104398
- Ruggeri, A., Xu, F., & Lombrozo, T. (2019). Effects of explanation on
  children's question asking. *Cognition, 191*, 103966.
  DOI: 10.1016/j.cognition.2019.05.003
- Chi, M. T. H., Bassok, M., Lewis, M. W., Reimann, P., & Glaser, R. (1989).
  Self-explanations: how students study and use examples in learning to solve
  problems. *Cognitive Science, 13*(2), 145–182. DOI: 10.1207/s15516709cog1302_1
- Bisra, K., Liu, Q., Nesbit, J. C., Salimi, F., & Winne, P. H. (2018). Inducing
  self-explanation: a meta-analysis. *Educational Psychology Review, 30*(3),
  703–725. DOI: 10.1007/s10648-018-9434-x
- Harders, B., & Ebersbach, M. (2026). No causal self-explanation effect for
  factual knowledge. *Applied Cognitive Psychology, 40*(3).
  DOI: 10.1002/acp.70174
- Graesser, A. C., Singer, M., & Trabasso, T. (1994). Constructing inferences
  during narrative text comprehension. *Psychological Review, 101*(3), 371–395.
  DOI: 10.1037/0033-295x.101.3.371
- Trabasso, T., & van den Broek, P. (1985). Causal thinking and the
  representation of narrative events. *Journal of Memory and Language, 24*(5),
  612–630. DOI: 10.1016/0749-596x(85)90049-x
- Nelson, T. O., & Narens, L. (1990). Metamemory: a theoretical framework and
  new findings. In G. Bower (Ed.), *The Psychology of Learning and Motivation*
  (Vol. 26, pp. 125–173). Academic Press. DOI: 10.1016/s0079-7421(08)60053-5
- Koriat, A. (1993). How do we know that we know? The accessibility model of
  the feeling of knowing. *Psychological Review, 100*(4), 609–639.
  DOI: 10.1037/0033-295x.100.4.609
- Fleming, S. M., & Dolan, R. J. (2012). The neural basis of metacognitive
  ability. *Philosophical Transactions of the Royal Society B, 367*(1594),
  1338–1349. DOI: 10.1098/rstb.2011.0417
- Ackerman, R., & Thompson, V. A. (2017). Meta-reasoning: monitoring and control
  of thinking and reasoning. *Trends in Cognitive Sciences, 21*(8), 607–617.
  DOI: 10.1016/j.tics.2017.05.004
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.).
  Cambridge University Press. DOI: 10.1017/cbo9780511803161
- Sloman, S. A. (2005). *Causal Models: How People Think about the World and Its
  Alternatives*. Oxford University Press. DOI: 10.1093/acprof:oso/9780195183115.001.0001
- Operskalski, J. T., & Barbey, A. K. (2017). Cognitive neuroscience of causal
  reasoning. In M. R. Waldmann (Ed.), *The Oxford Handbook of Causal
  Reasoning*. Oxford University Press. DOI: 10.1093/oxfordhb/9780199399550.013.16
- Lombrozo, T., & Vasilyeva, N. (2017). Causal explanation. In M. R. Waldmann
  (Ed.), *The Oxford Handbook of Causal Reasoning*. Oxford University Press.
  DOI: 10.1093/oxfordhb/9780199399550.013.22

---

## 10. How to read this document

- §0–§6 are the research. §7 is the implementation proposal and is the only
  section that should be acted on directly. **M1 is the highest-value fix.**
- §8 is the ledger of what I could not verify. U1 and U2 are corrections to
  `docs/BRAIN_EXPLANATION_AND_REASONING.md`, not new claims.
- §9 is verified against Crossref and PubMed E-utilities; every DOI there
  resolved on the day of writing.
- No file other than this one was modified.

