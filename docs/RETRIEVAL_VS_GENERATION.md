# Retrieval vs. Generation: What the Brain Actually Does, and a Switching Rule for RAVANA

**Status:** literature review + design proposal. Every claim below is either cited to a real
paper or explicitly flagged as unsettled. Nothing here is a fabricated consensus.

**Scope note:** RAVANA's "retrieval paths" (KB definition, template, decomposition, web
direct answer) are not the same thing as the brain's "retrieval" (whole-form lexical
lookup). The mapping is *functional*, not anatomical: RAVANA's retrieval paths are
**stored-answer paths** (a stored string or a stored subgraph is surfaced), and its neural
decoder is the **compositional/generative path**. The brain literature is about the
latter distinction. The rule below is therefore a *functional* translation, and says so
where the analogy is loose.

---

## 1. The short answer

The brain does **not** have a clean retrieve-vs-generate switch. It has a **race** between
a fast stored-representation pathway and a slower compositional pathway, and the winner is
decided by **three quantities that are already computable in RAVANA**:

1. **Prediction error / free energy** of the current context (how well the stored model
   already accounts for what is being asked).
2. **Retrieval strength / familiarity** of the specific target (frequency, recency,
   associative strength — the "instance" strength of Logan's race).
3. **Competition / conflict** among candidate stored representations (NoGo / lateral
   inhibition), which is the basal-ganglia contribution.

The switch is **not** "retrieve if confident, else generate." It is closer to:

> **Generate compositionally only when the stored-representation pathway's best candidate
> fails to clear a threshold that itself rises with prediction error.**

That is: high prediction error does *not* trigger generation — it makes generation
*harder to reach*, because the brain's conservative response to uncertainty is to fall
back on the most-practiced stored form, not to compose something novel. Generation is the
path taken when **no stored form clears the bar**, and the bar is set by how much
uncertainty the system is currently tolerating.

This is the opposite of the intuitive "uncertain → be creative" rule, and it matters for
RAVANA: RAVANA's current ordering is *already* "stored first, decoder last," which is
brain-faithful in direction — the problem is that the decoder is **unreachable**, not that
the ordering is wrong. See §5.

---

## 2. Levelt: where retrieval and generation sit

Levelt's model (Levelt, 1989; Levelt, Roelofs & Meyer, 1999) splits production into
**conceptualization → formulation → articulation**.

- **Conceptualizer** produces a *preverbal message*. This is where "what to say" is
  decided, drawing on situational knowledge and the discourse model. It is not linguistic.
- **Formulator** does **grammatical encoding** (lemma selection + syntactic building) and
  **phonological encoding** (retrieving the word's sound form, building the syllable plan).
- **Articulator** executes the phonetic plan from stored gestural scores.

The retrieve/generate distinction lives **inside the Formulator**, and Levelt's own
architecture is *two-step within lexical access*:

1. **Lemma selection** — a *retrieval* step. The concept activates a lemma (a
   syntactic-semantic word specification) from the mental lexicon. This is stored.
2. **Phonological encoding** — a *retrieval* step for the word form, but a
   **compositional/assembly** step for the syllabic frame and the phonetic plan.

So even in the most "generative" part of Levelt's model, the *word* is retrieved and only
the *form* is assembled. This is why the brain's language production is often described as
"retrieval-heavy": the unit of retrieval is the lemma and the morphophonological form, and
the unit of assembly is the syntactic frame and the phonetic plan.

**Key consequence for RAVANA:** the brain's "generation" is *not* free-form generation from
a blank slate. It is **assembly over retrieved parts**. RAVANA's neural decoder is closer
to "assembly over a learned distribution" than to "retrieval of a stored string" — which
is the right analogy, and it means the decoder should be *conditioned* on retrieved
material (which RAVANA already does via `conditioning_embs`), not run cold.

**Citation:** Levelt, W. J. M. (1989). *Speaking: From Intention to Articulation*. MIT
Press. Levelt, W. J. M., Roelofs, A., & Meyer, A. S. (1999). A theory of lexical access in
speech production. *Behavioral and Brain Sciences, 22*(1), 1–38.

---

## 3. Dual-route models: when stored whole-form beats compositional assembly

The cleanest evidence for a retrieve-vs-assemble switch comes from **reading aloud**, where
the dual-route model (Coltheart, 1978; Coltheart, Rastle, Perry, Langdon & Ziegler, 2001)
posits:

- a **lexical route** — whole-word orthography→phonology lookup, used for known words and
  for **exception words** (yacht, colonel) that cannot be assembled;
- a **nonlexical route** — grapheme→phoneme assembly, used for novel words and nonwords.

The two routes **race**, and the lexical route wins for high-frequency words because it is
faster. The nonlexical route is the fallback for anything the lexical route cannot supply.

**The critical detail:** the switch is **not** "known vs unknown." It is **frequency- and
neighbourhood-dependent**. A high-frequency word with many orthographic neighbours (a
"dense neighbourhood") is read more slowly and more error-prone than a high-frequency word
with few neighbours — because the lexical route faces more competition. This is the
**neighbourhood-density effect** (Coltheart, Davelaar, Jonasson & Besner, 1977; see also
Andrews, 1992). The brain's retrieve-vs-assemble decision is therefore sensitive to
**local competitive density**, not just to whether a representation exists.

**Applied to production:** the same logic is carried by the **dual-route cascade** and
**interactive-activation** models of production. Dell's interactive two-step model (Dell,
1997; Dell, Schwartz, Martin, Saffran & Gagnon, 1997) has lexical selection and phonological
encoding as two interactive retrieval steps, with **frequency implemented as connection
weight** — high-frequency words have stronger connections and win the race. The model's
speech-error patterns (semantic paraphasias, phonemic paraphasias, blends) are explained by
**competition among partially-activated candidates**, not by a clean retrieve/generate
switch.

**Citation:** Coltheart, M., Rastle, K., Perry, C., Langdon, R., & Ziegler, J. (2001). DRC:
A dual route cascaded model of visual word recognition and reading aloud. *Psychological
Review, 108*(1), 204–256. Dell, G. S., Schwartz, M. F., Martin, N., Saffran, E. M., &
Gagnon, D. A. (1997). Lexical access in aphasic and nonaphasic speakers. *Psychological
Review, 104*(4), 801–838.

---

## 4. Proceduralization / automatization: how generation becomes retrieval

This is the most important section for RAVANA, because it explains **why the switch moves
with practice** — and therefore why a static dispatch order is wrong.

**Logan's instance theory of automatization (1988):** every encounter with a stimulus
encodes an "instance" (a memory trace of the whole processing episode). On each trial,
there is a **race** between:
- an **algorithmic/procedural** process (the slow, compositional route), and
- **direct retrieval** of the earliest instance that matches the stimulus.

At first the race is won by the procedure (there are no instances). With practice,
instances accumulate and retrieval wins — **not because the procedure got faster, but
because retrieval got faster and its latency distribution shifted below the procedure's.**
The transition is **quantitative and continuous**, not a discrete stage change.

**Anderson's ACT-R (1982, 1993; Anderson & Lebiere, 1998):** knowledge starts
**declarative** (interpreted by a slow procedural interpreter) and becomes **procedural**
via **knowledge compilation** — production rules are formed, then composed into larger
units. The same shift: novel compositional performance becomes fluent retrieval.

**Siegler's distribution-of-associations model (Siegler & Shrager, 1984) and the adaptive
strategy choice model (Siegler & Shipley, 1995):** strategy selection depends on the
**relative efficiency (speed × accuracy)** of each strategy, and on the **associative
strength** of the problem–answer pair. Retrieval is used when its associative strength
exceeds a **confidence criterion**; otherwise a procedural strategy runs. Practice
strengthens associations, so retrieval is used more — but **even skilled adults do not
always retrieve** (Imbo & Vandierendonck, 2007, showed adults still use counting and
transformation strategies on simple arithmetic after practice).

**The brain-faithful consequence:** the retrieve/generate boundary is **not fixed**. It is
a **moving frontier determined by practice history**. A system that hard-codes "KB first,
decoder last" will, in the brain, eventually have the decoder *win* for familiar inputs —
because the decoder's output for a familiar input becomes a stored instance. RAVANA's
decoder is currently never used, so it never gets the practice that would make it win.
This is a **bootstrapping problem**, and it is the deepest reason the current dispatch
order is stuck.

**Citations:** Logan, G. D. (1988). Toward an instance theory of automatization.
*Psychological Review, 95*(4), 492–527. Anderson, J. R. (1982). Acquisition of cognitive
skill. *Psychological Review, 89*(4), 369–406. Anderson, J. R., & Lebiere, C. (1998). *The
Atomic Components of Thought*. Erlbaum. Siegler, R. S., & Shrager, J. (1984). Strategy
choices in addition and subtraction. In C. Sophian (Ed.), *Origins of Cognitive Skills*.
Erlbaum. Imbo, I., & Vandierendonck, A. (2007). Practice effects on strategy selection and
strategy efficiency in simple mental arithmetic. *Psychological Research, 72*, 528–541.

---

## 5. The basal ganglia: the selection mechanism, not the switch itself

The basal ganglia (BG) are **not** a retrieve-vs-generate switch. The evidence supports a
more specific role: **response selection under competition**, gated by dopamine.

**Anatomy:** cortico-striato-thalamo-cortical (CSTC) loops. The **direct pathway** (Go)
facilitates the selected action; the **indirect pathway** (NoGo) suppresses competitors.
Both converge on a **dynamic threshold** (Graybiel, 2008; Mink, 1996).

**Evidence from Parkinson's disease:** PD damages the BG and produces language deficits
*without* a primary linguistic deficit — reduced sentence complexity, impaired
morphosyntactic production, and impaired **language control** (the ability to utter in one
language without interference from another). The deficit is **selection/sequencing**, not
storage (Colman, 2011; Copland, Brownsett & Iyer, 2021; Lee & Tomblin, 2015).

**Ullman's declarative/procedural model (2001):** the **mental lexicon** (memorized words
and their forms) depends on **declarative memory** (temporal lobe); the **mental grammar**
(rules for sequential and hierarchical composition) depends on **procedural memory**
(frontal cortex + basal ganglia). This is the closest thing in the literature to a
retrieve-vs-generate neural split — but note the direction: **grammar (composition) is the
procedural system**, and the lexicon (stored forms) is declarative. The BG are implicated
in the *compositional* side, not the retrieval side.

**The honest reading:** the BG implement **competitive selection with a threshold that
dopamine modulates**. They do not decide "retrieve vs generate"; they decide **which
candidate wins**, given a threshold. The threshold is the switch's knob, and it is set by
arousal, novelty, prediction error, and dopamine tone — exactly the parameters RAVANA's
`BasalGangliaGate` already exposes (`set_prediction_error`, `set_arousal`,
`set_novelty`, `set_dopamine_tone`).

**Citations:** Ullman, M. T. (2001). The declarative/procedural model of lexicon and
grammar. *Journal of Psycholinguistic Research, 30*(1), 37–69. Copland, D. A., Brownsett,
S., & Iyer, K. (2021). Corticostriatal regulation of language functions. *Neuropsychology
Review, 31*, 472–494. Lee, J. C., & Tomblin, J. B. (2015). Procedural learning and
individual differences in language. *Language Learning and Development, 11*(3), 215–236.
Graybiel, A. M. (2008). Habits, rituals, and the evaluative brain. *Annual Review of
Neuroscience, 31*, 359–387.

---

## 6. Predictive processing / active inference: the free-energy account

**Friston's free-energy principle (Friston, 2010, 2018):** the brain minimizes variational
free energy — a bound on surprise — by either **updating its generative model** (perception)
or **acting to change its input** (action). In language, this is often cashed out as
**hierarchical predictive coding**: higher levels generate predictions, lower levels
compute **prediction error**, and only the error is passed up (Rao & Ballard, 1999;
Friston, 2005).

**Clark (2013, 2016):** the brain is a "prediction machine"; perception and action are both
in the service of prediction-error minimization. *Surfing Uncertainty* (2016) extends this
to the embodied, situated case.

**What this predicts for retrieve-vs-generate:** the brain should **prefer the pathway that
minimizes expected free energy**. A stored representation that predicts the context well
(low prediction error) is cheap — use it. Composition is expensive (it requires building a
new model of the message), so it is used when the stored model's prediction error is high
*and* the system has the resources to compose.

**The subtlety that matters:** high prediction error does **not** automatically trigger
composition. In the active-inference framing, high prediction error can be resolved either
by **updating the model** (perceptual inference) or by **acting** (sampling the world to
confirm a prediction). Composition is a *third* option — and it is the most expensive.
The brain's observed bias is toward the cheapest resolution. So the rule is:

> **Compose only when (a) no stored representation predicts the context adequately, AND
> (b) the cost of composition is acceptable given current precision/arousal.**

**Where this is unsettled:** the application of predictive coding to *language production*
specifically (as opposed to comprehension) is **contested**. Pickering & Garrod (2013)
argue production uses forward models (predictions of the sensory consequences of one's own
speech); their BBS target article drew multiple commentaries disputing whether the
evidence supports forward models in production at all (e.g., the commentary by
... — see Pickering & Garrod, 2013, *Behavioral and Brain Sciences, 36*(4), 329–347,
and the accompanying commentaries). The comprehension literature (N400, P600, hierarchical
predictive coding) is much better established than the production literature. **Flag:
do not treat "the brain predicts its own utterance" as settled.**

**Citations:** Friston, K. (2010). The free-energy principle: a unified brain theory?
*Nature Reviews Neuroscience, 11*(2), 127–138. Friston, K. (2018). Active inference and
cognitive consistency. *Psychological Inquiry, 29*(2), 67–73. Clark, A. (2013). Whatever
next? Predictive brains, situated agents, and the future of cognitive science.
*Behavioral and Brain Sciences, 36*(3), 181–204. Clark, A. (2016). *Surfing Uncertainty:
Prediction, Action, and the Embodied Mind*. Oxford University Press. Rao, R. P. N., &
Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience,
2*(1), 79–87. Pickering, M. J., & Garrod, S. (2013). An integrated theory of language
production and comprehension. *Behavioral and Brain Sciences, 36*(4), 329–347.

---

## 7. Speech errors, paraphasias, and *conduit d'approche*: what they reveal

The speech-error literature is the strongest evidence that production is **retrieval with
competition**, not clean assembly.

- **Semantic paraphasias** (saying "dog" for "cat") and **phonemic paraphasias** (saying
  "cag" for "cat") are the two dominant error types. Both are explained by **spreading
  activation + competition** in a network where multiple candidates are partially
  activated (Dell, 1997; Levelt et al., 1999). The error is the *losing* candidate
  surfacing — evidence that retrieval is a race, not a lookup.
- **Blends** ("c" from "cat" + "g" from "dog" → "cag") show that two candidates can be
  co-activated and merged — again, competition, not a switch.
- **Tip-of-the-tongue (TOT) states** are a *partial* retrieval failure: the lemma is
  selected but the phonological form is not. This is direct evidence for Levelt's
  two-step retrieval (lemma → form) and for the claim that **retrieval is staged**
  (Brown & McNeill, 1966; Levelt et al., 1999).

**Conduit d'approche (CdA):** the repetitive self-corrective behaviour in which a speaker
approaches the target word through successive approximations ("... it's a... it's a kind
of... it's a *bicycle*"). CdA is found in **conduction aphasia** and some aphasic
populations, and it is the clearest window onto the retrieve/generate question:

- CdA is **not** free generation. The speaker is *constrained* by the target — each attempt
  is a partial retrieval that is rejected and retried. This is **retrieval with a
  monitoring loop**, not compositional assembly.
- Conduction aphasia is classically a **phonological encoding deficit** (Levelt's
  Formulator, second step): comprehension is preserved, but phonological sequencing and
  repetition of polysyllabic words are impaired (Bastiaanse & den Ouden, see
  *Phonological encoding and conduction aphasia*, in the Levelt-model literature;
  StatPearls, *Conduction Aphasia*, NBK537006).
- The existence of CdA implies the brain has a **monitor** that compares the produced form
  against the target and **re-runs retrieval** when they mismatch. This is Levelt's
  **perceptual loop theory** (Levelt, 1983, 1989): the speaker monitors their own *inner
  speech* via the comprehension system, and repairs on mismatch.

**The implication for RAVANA:** RAVANA's `_sm_response_grounded` gate and the
ACC/ERN-style `_topic_grounded` monitor are the right *shape* — a monitor that vetoes
ungrounded output. But the brain's monitor **re-runs retrieval**, it does not fall through
to a different pathway. RAVANA's monitor currently falls through to
`_human_like_uncertainty` or the decoder. A more brain-faithful monitor would **retry
retrieval with a relaxed criterion** before escalating to composition.

**Citations:** Levelt, W. J. M. (1983). Monitoring and self-repair in speech. *Cognition,
14*(1), 41–104. Brown, R., & McNeill, D. (1966). The "tip of the tongue" phenomenon.
*Journal of Verbal Learning and Verbal Behavior, 5*(4), 325–337. Dell, G. S. et al.
(1997), *Psychological Review, 104*(4), 801–838. On *conduit d'approche* specifically,
see the recent review of psycholinguistic variables influencing CdA (*Aphasiology*,
2026, doi:10.1080/02687038.2026.2613004) and the classic aphasiological literature on
conduction aphasia (e.g., StatPearls, *Conduction Aphasia*, NBK537006; den Ouden &
Bastiaanse on phonological encoding in conduction aphasia). **Flag:** the CdA literature
is largely clinical-case-based, and the phenomenon is not well quantified — treat the
"re-run retrieval with a relaxed criterion" reading as a plausible interpretation, not
an established mechanism.

---

## 8. Formulaic language / prefabs: how much of speech is retrieved?

This is the strongest *quantitative* evidence that retrieval dominates ordinary speech.

- **Pawley & Syder (1983):** native speakers have "nativelike selection" and "nativelike
  fluency" because they store **hundreds of thousands of preconstructed clauses** —
  memorized multiword units — and retrieve them whole. Grammar is the *fallback* for
  novel combinations, not the primary engine of fluent speech.
- **Wray (2002):** formulaic sequences are "stored and retrieved whole from memory at the
  time of use, rather than being subject to generation or analysis by the language
  grammar" (Wray, 2002, p. 9). Wray argues the lexicon is dominated by multiword units.
- **Erman & Warren (2000):** in a corpus study, **58.6%** of the spoken text was composed
  of formulaic sequences. This is the number most often cited, and it is **corpus-based
  and contested** — the definition of "formulaic" drives the count, and different
  definitions give different numbers (see the discussion in Wray, 2013, *Language
  Teaching, 46*(3), 316–334, which warns explicitly against treating repeated claims as
  established).
- **Sinclair (1991):** the "idiom principle" — the speaker has available a large number of
  preconstructed multiword units, and uses them wherever possible; the "open-choice
  principle" (grammar) is the fallback.
- **Siyanova-Chanturia, Conklin & Schmitt (2011):** idioms are processed faster than
  novel phrases in eye-tracking, and **phrasal frequency** (not just word frequency)
  modulates processing — evidence that multiword units are stored and retrieved as wholes.

**The honest range:** estimates of formulaic prevalence vary widely (roughly **30–80%**
of spoken discourse, depending on definition and corpus). The defensible claim is
**not** "80% of speech is retrieved chunks" — it is **"a substantial and possibly
majority share of ordinary speech is retrieved multiword material, and the exact
proportion is definition-dependent and unsettled."** Wray (2013) explicitly cautions
against the "Bellman's Rule of Three" — repeated claims becoming received wisdom.

**Implication for RAVANA:** RAVANA's KB definitions, templates, and decomposition paths
are **prefabs**. The brain-faithful design is to make them *the majority path* — which
RAVANA already does. The decoder is the "open-choice principle" — the fallback for novel
combinations. The problem is not that the decoder is a fallback; it is that the fallback
is **unreachable**, so the system has no way to produce novel combinations at all.

**Citations:** Pawley, A., & Syder, F. H. (1983). Two puzzles for linguistic theory:
Nativelike selection and nativelike fluency. In J. C. Richards & R. W. Schmidt (Eds.),
*Language and Communication*. Longman. Wray, A. (2002). *Formulaic Language and the
Lexicon*. Cambridge University Press. Wray, A. (2013). Formulaic language. *Language
Teaching, 46*(3), 316–334. Erman, B., & Warren, B. (2000). The idiom principle and the
open choice principle. *Text, 20*(1), 29–62. Sinclair, J. McH. (1991). *Corpus,
Concordance, Collocation*. Oxford University Press. Siyanova-Chanturia, A., Conklin, K.,
& Schmitt, N. (2011). Adding more fuel to the fire: An eye-tracking study of idiom
processing by native and non-native speakers. *Second Language Research, 27*(1), 1–22.

---

## 9. What the evidence says the switch criterion is

Synthesizing §2–§8, the brain's actual criterion is **not** a binary retrieve/generate
flag. It is a **thresholded race** with three inputs:

| Input | Brain correlate | RAVANA primitive |
|---|---|---|
| **Retrieval strength** of the best stored candidate | instance strength (Logan 1988); connection weight (Dell 1997); frequency/AoA | `node.confidence`, `node.stability`, `edge.weight`, `_definitions` membership, `_concept_sources` membership |
| **Prediction error / free energy** of the context | hierarchical predictive coding (Friston 2010; Clark 2013) | `node.prediction_free_energy`, `edge.prediction_free_energy`, `_free_energy`, `_mean_prediction_error` |
| **Competition / neighbourhood density** | NoGo lateral inhibition (Mink 1996; Graybiel 2008); orthographic neighbourhood effects | `len(get_outgoing(nid))`, `len(get_incoming(nid))`, BG gate `no_go_strength` |

**The criterion, stated plainly:**

> Retrieve the best stored representation **iff its strength clears a threshold that
> increases with prediction error and with competitive density.** Otherwise, compose.
> Composition is itself **conditioned on retrieved material** (the retrieved concepts
> constrain the generative model), and is **monitored** against the target; on mismatch,
> the system **re-runs retrieval with a relaxed criterion** before composing again.

Two non-obvious consequences:

1. **High prediction error makes retrieval *more* likely, not less** — because the
   threshold rises and the system falls back on the most-practiced form. (This is the
   conservative/routine response to uncertainty.) Generation is reached when retrieval
   *fails*, not when uncertainty is high.
2. **The boundary moves with practice.** A pathway that is generative at first becomes
   retrievable with use (Logan 1988; Anderson 1982). A static dispatch order cannot
   capture this; the rule must be **history-dependent**.

---

## 10. The implementable rule for RAVANA

Expressed entirely in RAVANA's existing vocabulary. This is a **proposal**, not code that
has been run.

### 10.1 Define the retrieval-strength score

For the current subject `s` and the current turn, compute a **retrieval strength**
`R(s) ∈ [0, 1]` from the signals RAVANA already has:

```
R(s) = w1 * has_definition(s)          # _definitions membership — strongest prefab
     + w2 * has_web_knowledge(s)       # _concept_sources membership
     + w3 * node_confidence(s)         # ConceptNode.confidence
     + w4 * node_stability(s)          # ConceptNode.stability
     + w5 * assoc_strength(s)          # top association score from the spread
     + w6 * familiarity(s)             # user_model.infer_user_knows(s) — already exists
```

with `w1 > w2 > w3 ≈ w4 ≈ w5 ≈ w6` and `Σw = 1`. Suggested starting weights:
`w1=0.30, w2=0.20, w3=0.15, w4=0.10, w5=0.15, w6=0.10`.

**Why these:** `has_definition` is the closest thing to a stored whole-form
representation (a prefab). `node.confidence` and `node.stability` are RAVANA's existing
retrieval-strength analogues. `familiarity` captures the FOK signal RAVANA already
computes — the **cue-familiarity heuristic** of Reder (1987) and the accessibility
account of Koriat (1993), both already cited in RAVANA's code (`engine.py:10986`).
Citations: Reder, L. (1987). Strategy selection in question answering. *Cognitive
Psychology, 19*(1), 90–138. Koriat, A. (1993). How do we know that we know? The
accessibility model of the feeling of knowing. *Psychological Review, 100*(4), 609–639.

### 10.2 Define the prediction-error / free-energy term

```
PE(s) = max( node.prediction_free_energy(s),
             mean(edge.prediction_free_energy for edges incident to s),
             self._mean_prediction_error )
```

All three are already computed and persisted (`engine.py:12481`,
`chain_walker.py:931`, `graph/engine.py:1012`).

### 10.3 Define the competitive-density term

```
D(s) = (out_degree(s) + in_degree(s)) / max_degree_in_graph
```

using `graph.get_outgoing(nid)` and `graph.get_incoming(nid)` — both exist
(`ravana_ml/src/ravana_ml/graph.py:1522`). Normalise by the graph's max degree so the
term is comparable across graph sizes. **High density = more competition = harder for
any single candidate to win** (the neighbourhood-density effect, §3).

### 10.4 The threshold

```
T(s) = T_base * (1 + k_pe * PE(s)) * (1 + k_d * D(s)) * (1 - k_f * familiarity(s))
```

- `T_base` — base retrieval threshold (start at 0.35, tune).
- `k_pe` — how strongly prediction error raises the bar (start at 0.5).
- `k_d` — how strongly competitive density raises the bar (start at 0.3).
- `k_f` — how strongly user familiarity lowers the bar (start at 0.2).

**The familiarity term is subtracted** because a familiar *user* makes the system more
willing to retrieve (the user's own words are a strong cue — Reder's cue-familiarity
hypothesis, already in RAVANA's code at `engine.py:10986`).

### 10.5 The decision

```
if R(s) >= T(s):
    RETRIEVE          # KB / template / decomposition / web — the existing paths
else:
    COMPOSE           # the neural decoder, conditioned on retrieved material
```

**Critical:** `COMPOSE` must still be **conditioned** on the retrieved concepts. RAVANA
already does this (`conditioning_embs` built from the situation vector and top active
concepts in `_generate_with_situation_model`). The rule does not change that; it changes
**when** the decoder is allowed to run.

### 10.6 The monitor-and-retry loop (the CdA analogue)

After composition, run the existing grounding gate (`_sm_response_grounded`). On failure:

1. **Relax the retrieval criterion** and re-run retrieval: `T(s) ← T(s) * 0.8`, retry the
   KB/template paths once. (This is the *conduit d'approche* behaviour — re-run retrieval
   with a relaxed criterion, do not immediately give up.)
2. If retrieval still fails, **compose again** with a lower temperature (more conservative
   sampling) — the brain's conservative response to persistent prediction error.
3. If composition still fails the grounding gate, **escalate to honest uncertainty**
   (`_human_like_uncertainty`) — the current terminal path, which is correct.

### 10.7 The practice term (the Logan/Anderson term)

The rule must be **history-dependent**, or it cannot capture proceduralization. Add a
**use counter** to each retrieval path and to the decoder:

```
N_retrieve(s)  — how many times a stored path answered for subject s
N_compose(s)   — how many times the decoder answered for subject s
```

Then modulate the threshold:

```
T(s) ← T(s) * (1 - k_p * tanh(N_compose(s) / N_practice))
```

where `N_practice` is the practice scale (start at 20). **As the decoder answers more
often for a subject, the threshold drops and the decoder is reached more easily** —
exactly Logan's instance-theory prediction that practice shifts the race. This is the
term that fixes RAVANA's bootstrapping problem: the decoder can *earn* reachability
through use, rather than needing a hand-set `use_linggen` flag.

**Optional, more brain-faithful version:** cache the decoder's accepted output for a
subject as a new `_definitions` entry after `k` successful grounded compositions. This is
literally proceduralization — a generated form becomes a stored prefab. It also gives the
system a natural "learning" signal that is not a separate training pass.

### 10.8 Where to insert it in the dispatch order

The rule is a **gate**, not a reordering. Insert it at the top of `_generate_response`
(`response_gen.py:7189`), *before* the existing chain, and use it to decide whether the
existing chain runs at all:

```
R, T = compute_retrieval_strength(ctx.subject), compute_threshold(ctx.subject)
if R >= T:
    run the existing chain unchanged (KB → web → decomposition → reflective → ...)
else:
    run the decoder path first (_generate_with_situation_model)
    then the monitor-and-retry loop (§10.6)
```

This preserves the existing safety hierarchy (the ACC/ERN `_topic_grounded` monitor still
vetoes) while making the decoder reachable. It also means the decoder is tried **on its
merits**, not only after every other path has failed — which is the specific defect
documented in `docs/DECODER_REACHABILITY.md`.

**Note on `use_linggen`:** the existing `use_linggen` flag is a *promotion* flag that
flips only when the decoder wins a CE comparison. The rule above **subsumes** it: the
practice term (§10.7) makes the decoder's reachability a *measured, earned* quantity
rather than a hand-set flag. `use_linggen` can remain as a global kill-switch.

---

## 11. What would falsify the rule

A rule is only worth proposing if it can be wrong. These are the concrete falsifiers.

**F1 — The decoder never wins even with the practice term.**
If, after `N_practice` successful grounded compositions for a subject, the decoder still
loses to the KB path on a held-out probe set, then the practice term is not doing what
Logan's instance theory predicts — either the decoder's outputs are not being cached as
stored forms, or the retrieval paths are simply better for those subjects (in which case
the decoder should *not* win, and the rule is correctly conservative).

**F2 — High prediction error correlates with *more* retrieval, not less.**
The rule predicts that raising `PE(s)` raises `T(s)`, making retrieval *more* likely.
Falsifier: measure the correlation between `PE(s)` and the fraction of turns that use a
retrieval path. If the correlation is **negative** (high PE → more generation), the
"conservative response to uncertainty" claim (§6, §9) is wrong for RAVANA's inputs, and
the sign of `k_pe` should flip.

**F3 — Neighbourhood density does not predict retrieval difficulty.**
The rule predicts that high `D(s)` raises `T(s)` and makes retrieval *harder* (more
composition). Falsifier: regress `D(s)` against retrieval success rate. If the coefficient
is ~0 or positive, the neighbourhood-density analogy (from reading aloud) does not
transfer to RAVANA's graph, and `k_d` should be dropped.

**F4 — The monitor-and-retry loop does not improve groundedness.**
The rule predicts that relaxing `T(s)` and retrying retrieval (§10.6) recovers more
grounded answers than falling straight through to uncertainty. Falsifier: A/B the
retry-once loop against the current fall-through. If the retry loop does not increase the
grounded-answer rate, the CdA analogy is wrong and the loop should be removed.

**F5 — The practice term does not shift the boundary.**
The rule predicts that `N_compose(s)` reduces `T(s)` over time. Falsifier: track
`T(s)` against `N_compose(s)` across a long run. If `T(s)` does not fall, the
proceduralization analogy is not implemented correctly (or the decoder's outputs are not
being cached).

**F6 — The formulaic-language majority claim does not hold for RAVANA's inputs.**
The rule assumes retrieval paths should handle the majority of turns (§8). Falsifier:
measure the fraction of real user turns that a retrieval path can answer grounded. If it
is **below ~50%**, the prefab assumption is wrong for RAVANA's domain and the decoder
should be the *primary* path, not the fallback.

**F7 — The decoder's output is not improvable by conditioning on retrieved material.**
The rule assumes composition is conditioned on retrieval (§2, §10.5). Falsifier: compare
groundedness of decoder output with vs. without `conditioning_embs`. If conditioning does
not help, the "assembly over retrieved parts" analogy is wrong and the decoder should be
run cold (or the conditioning mechanism is broken).

---

## 12. Honest flags: where the neuroscience is unsettled

1. **Predictive coding in production is contested.** The strong evidence is in
   comprehension (N400, P600, hierarchical predictive coding). Pickering & Garrod's
   (2013) forward-model account of production drew significant criticism. Do not treat
   "the brain predicts its own utterance" as established.
2. **The formulaic-language prevalence numbers are definition-dependent.** The 58.6%
   figure (Erman & Warren, 2000) is widely cited but the range across studies is roughly
   30–80%. Wray (2013) explicitly warns against treating repeated claims as established.
3. **The BG's role in language is "regulatory," not "linguistic."** Copland et al. (2021)
   conclude the striatum responds to *uncertainty and conflict* demanding selection and
   sequencing — a domain-general control role, not a language-specific switch. The
   retrieve/generate framing may be an over-reading of a selection mechanism.
4. **Ullman's declarative/procedural split is a model, not a settled fact.** It is
   influential and well-motivated, but the claim that grammar is *procedural* and the
   lexicon *declarative* remains a theoretical commitment with dissenting accounts
   (e.g., connectionist single-route models).
5. **The "conservative response to uncertainty" claim is an inference, not a direct
   measurement.** It follows from the predictive-processing framing and from the
   arithmetic-strategy literature (Imbo & Vandierendonck, 2007), but there is no direct
   neuroimaging study showing "high PE → more retrieval, less composition" in language
   production. Falsifier F2 is the test.
6. **The mapping from brain to RAVANA is functional, not anatomical.** RAVANA's "KB" is
   not the brain's "mental lexicon," and RAVANA's decoder is not the brain's "grammar."
   The rule is a *design analogy* justified by convergent evidence, not a claim about
   neural implementation.

---

## 13. Bottom line for RAVANA

- The brain's switch is a **thresholded race**, not a binary flag. The threshold rises
  with **prediction error** and **competitive density**, and falls with **practice**.
- RAVANA's current ordering (stored first, decoder last) is **directionally
  brain-faithful** — the defect is that the decoder is **unreachable**, so the system
  cannot produce novel combinations and the decoder never gets the practice that would
  make it win.
- The fix is a **gate** at the top of `_generate_response` that computes
  `R(s)` vs `T(s)` from signals RAVANA already has, plus a **practice term** that lets
  the decoder earn reachability, plus a **retry loop** that mirrors *conduit d'approche*.
- The rule is falsifiable in at least seven concrete ways (§11). The most important are
  F2 (does high PE really mean more retrieval?) and F6 (is the prefab assumption right
  for RAVANA's inputs?).
