# Opinion Formation in the Brain: Mechanisms and a Concrete Architecture

Status: design note for RAVANA. Scope: how a brain forms, holds, and *expresses* an opinion
on a topic it has no stored answer for, without either deferring to the interlocutor or
confabulating certainty. Citations are real; the architecture in §5 is a proposal.

---

## 0. Problem statement

RAVANA is asked *"what do you think about artificial intelligence"* and answers:

> "i'm still forming a view on artificial intelligence. i don't have a fixed stance on
> artificial intelligence yet — what's your take?"

Asked *"what is your opinion on privacy"* it answers:

> "i care deeply about privacy. it is a basic right"

The stance machinery demonstrably exists; the live path deflects on some topics. The
defect is not "no opinion available" — it is a **gate that fires on the wrong signal**.
The brain's answer to the same situation is instructive: it does not treat "no stored
answer" as a reason to abstain. It treats it as a reason to *build* an answer from
whatever is available, and to *mark the product with its provenance*.

Two failure modes to avoid, and they pull in opposite directions:

- **Deference / abstention** — the system returns the question instead of a position.
  Cost: no identity, no learning signal, the user's view is never engaged.
- **Confabulated certainty** — the system emits a position with no basis and full
  conviction. Cost: false beliefs get committed, then perseverate (§2), and the
  system cannot revise them.

The design goal is a *substantive calibrated stance*: a real position, with an honest
strength tag, that is revisable.

---

## 1. Attitudes are constructed, not retrieved

### 1.1 Constructive preference

Payne, Bettman and Johnson's **adaptive decision maker** framework holds that people
have a *vocabulary of preference primitives* — evaluative fragments, learned attribute
evaluations, hard-wired likes/dislikes — and assemble them into an evaluation on the
spot, contingent on task and context (Payne, Bettman & Johnson, 1992, 1993; Bettman,
Luce & Payne, 1998). The decisive line is theirs: preference measurement is
**architecture, not archaeology** — you are building a value, not uncovering one
(Payne, Bettman & Schkade, 1999). Strategy selection is itself constructed and
*opportunistic*: the system changes processing mid-decision as it learns the problem
structure (Payne, Bettman & Johnson, 1993).

Schwarz's construal model makes the same point for attitudes: judgments are formed on
the spot from whichever inputs are accessible — features of the object, one's own
behaviour toward it, and one's apparent affective reaction ("how do I feel about
this?") — and stability across occasions is *emergent*, not stored (Schwarz & Bohner,
2001; Schwarz, 2007). Feldman and Lynch (1988) add the mechanism that makes this
matter operationally: an input is used to answer a later question in proportion to its
**accessibility × diagnosticity**, relative to alternatives — and merely *asking* the
question can create the construct ("self-generated validity").

### 1.2 Neural value representation

- **vmPFC/OFC** represents subjective value on a **common scale** across reward types,
  integrating multiple dimensions (Peters & Büchel, 2010; Levy & Glimcher, 2012). It
  encodes value independently of the motor response, i.e. it is a valuation region
  rather than an action-selection region (Kosik et al., 2020). OFC carries
  identity-specific value; vmPFC carries identity-general value (Howard et al., 2015;
  Klein-Flügge et al., 2016). Value is an *emergent pattern of connectivity*, not a
  scalar in a neuron (Padoa-Schioppa, 2011).
- **Amygdala** is not a "negativity detector". It tracks **motivational salience /
  relevance to current goals**, and its valence tuning shifts with the evaluative goal
  in play (Cunningham, Raye & Johnson, 2005; Cunningham, Van Bavel & Johnsen, 2008).
  It responds to novel, ambiguous, and extreme stimuli, and it is engaged even by
  implicit, unrequested evaluation of novel faces (Adolphs, Tranel & Damasio, 1998;
  Winston et al., 2002; Engell, Haxby & Todorov, 2007). A negativity bias exists but
  modulation is stronger for positive than negative information (Cunningham et al.,
  2008).
- **Iterative reprocessing** (Cunningham & Zelazo, 2007; Cunningham, Zelazo, Packer &
  Van Bavel, 2007): evaluation is a *loop*. Fast, automatic valence is re-processed by
  increasingly reflective, context-sensitive and goal-sensitive processes; each
  iteration can recruit finer detail and can *change* the evaluation. Ambivalence
  arises when iterations fail to resolve competing tendencies.

**Design consequence.** "No stored stance on X" is not an empty set. The primitives
are there: values, prior beliefs, past positions on neighbouring topics, the user's
expressed views, the system's own past utterances. Construction means *retrieving a
set of primitives and combining them under a current goal*, then tagging the result.

---

## 2. Commitment and belief perseverance

### 2.1 Why a formed view sticks

Once a view is formed it is hard to dislodge. **Belief perseverance**: after the
evidence for a belief is fully discredited, the belief survives — and it survives
*more* if the person generated causal explanations for it (Ross, Lepper & Hubbard,
1975; Anderson, Lepper & Ross, 1980; Anderson, 1982). The mechanism is
**explanation availability**: the relative number of supporting vs. conflicting
explanations generated before invalidation predicts perseverance (Anderson, New &
Speer, 1985). **Biased assimilation**: Lord, Ross and Lepper (1979) gave
pro- and anti-capital-punishment participants the *same* mixed studies; each side rated
the agreeing study as better conducted and both sides became *more* extreme.

Attitude **representation theory** (Lord & Lepper, 1999) is the constructive
counterpart: an attitude report is the combination of activated assumptions about the
object plus the immediate situation, and *exemplar stability* predicts attitude
stability and resistance to change (Lord, Paulson, Sia, Thomas & Lepper, 2004;
Sia, Lord, Blessum, Ratcliff & Lepper, 1997).

**Commitment is not all bad.** Kolling, Wittmann, Behrens, Boorman, Mars and
Rushworth (2016) show dACC carries **search value** (average value of alternatives)
and **engage/persistence value**, and signals the need to update internal models —
i.e. the same region supports both sticking and switching, and the signal that
instigates change "need not itself be a conflict or difficulty signal".

### 2.2 ACC conflict monitoring

Botvinick, Braver, Barch, Carter and Cohen (2001) and Botvinick, Cohen and Carter
(2004): ACC detects conflict between simultaneously active competing representations
and engages DLPFC to resolve it. Empirically, ACC activation tracks *decision* conflict,
not merely response conflict, including in tasks requiring integration of beliefs and
preferences (Pochon, Riis, Sanfey, Nystrom & Cohen, 2008; Botvinick, Cohen & Carter,
2004; van Veen & Carter, 2005). Brown (2013) recasts ACC as predicting and evaluating
planned action outcomes.

**Design consequence.** Conflict is a *signal with a direction*, not a stop sign.
High conflict between candidate positions should (a) raise processing depth, and
(b) either resolve toward one side or produce an explicitly *ambivalent* stance with
low certainty — but it should never produce abstention. Abstention is what the
current RAVANA path does, and it is the one output the brain does not select here.

---

## 3. Metacognitive awareness of attitude strength

### 3.1 Attitude certainty and ambivalence

**Attitude certainty** is the subjective sense of confidence or conviction about one's
attitude (Tormala & Rucker, 2007, 2018). It decomposes into **clarity** ("I know what
I think") and **correctness** ("I think I'm right") (Petrocelli, Tormala & Rucker,
2007). Certainty is not a by-product of elaboration: equal amounts of thinking can
yield different certainty (Rucker & Petty, 2004, 2006; Tormala & Petty, 2002, 2004).

Rucker, Tormala, Petty and Briñol (2014) propose certainty is built from a **finite
set of appraisals**: accuracy, completeness, relevance, legitimacy, importance, and the
subjective experience of thinking about the attitude. Impressed → certain;
unimpressed → uncertain; neutral → unchanged. Crucially, this requires *motivation
and ability* to appraise; without them certainty is simply not updated.

**Ambivalence** is the simultaneous activation of positive and negative components
(Priester & Petty, 1996; van Harreveld, van der Pligt & de Liver, 2009). It is
experienced as aversive **specifically when a choice must be made** — ambivalence
predicts physiological arousal only under commitment pressure (van Harreveld, Rutjens,
Rotteveel, Nordgren & van der Pligt, 2009). The **metacognitive model** (Petty, 2006;
Petty & Briñol, 2006) tags evaluations as valid/invalid, producing univalent,
explicitly ambivalent, and *implicitly ambivalent* states; after attitude change the
old attitude lingers and drives behaviour outside awareness (Petty, Tormala, Briñol
& Jarvis, 2006).

### 3.2 The monitoring/control gate

Koriat and Goldsmith (1996) separate memory performance into **retrieval**,
**monitoring effectiveness** (does assessed probability separate right from wrong?),
**control sensitivity**, and **response criterion**. Accuracy improves only if
monitoring is good; a criterion set too low buys quantity at the cost of accuracy.
Koriat's **accessibility model** of feeling-of-knowing (1993) is the key warning:
FOK is driven by the *amount and ease* of partial information, **regardless of its
correctness** — hence illusions of knowing when accessibility is not diagnostic
(Koriat, 1993, 1998). Confabulation is the clinical extreme: associative retrieval
runs without strategic monitoring, so the first idea that comes to mind is emitted as
fact (Moscovitch, 1989, 1995; Gilboa et al., 2006; Koriat & Goldsmith, 1996; see also
Burgess & Shallice, 1996).

**Design consequence.** Certainty must be *computed from appraisals of the evidence
actually used*, not from fluency, not from topic familiarity, and not from the fact
that a sentence was produced. And there must be a genuine **withhold** option that is
*not* the same as the current deflect — withholding is legitimate when monitoring
effectiveness is near zero, and it must be visibly rare and explicitly labelled.

---

## 4. Calibrated expression and perspective-taking

### 4.1 Hedging vs conviction

Expression style tracks certainty. In messages to a disagreer, high-certainty writers
produced more arguments, clearer beliefs and values, more emotional and moral
language, and a more judgmental tone; low-certainty writers used **more hedges,
qualifiers and questions** and expressed more interest in the opposing view
(Cheatham & Tormala, 2017). The certainty–advocacy relationship is **J-shaped**:
advocacy peaks at high certainty, bottoms at moderate, and rises again at low
certainty (Cheatham & Tormala, 2017) — low certainty drives *inquiry*, not silence.

Hedges are not merely politeness: hedge **perspective** (personal vs general) and
**likelihood** both affect persuasion through *perceived confidence* (Pezzuti,
Leonhardt & Warren, 2025). And certainty is **incongruent-sensitive**: a non-expert
expressing certainty, or an expert expressing uncertainty, violates expectancies,
raises involvement, and persuades more (Karmarkar & Tormala, 2010).

### 4.2 Keeping your view while taking theirs

The **right temporoparietal junction** is the hub for representing another's temporary
mental state — goals, intentions, desires — *even when they are false or opposed to
one's own*; enduring dispositions of self and other are handled more by mPFC (Van
Overwalle, 2009; Van Overwalle & Baetens, 2009). rTPJ is specifically implicated in
**switching between self and other representations**, and causal (TMS/tDCS) work
supports a role in self–other distinction and embodied perspective-taking (Sowden &
Catmur, 2015; Wang et al., 2016; Martin et al., 2019; Besharati et al., 2016).
Perspective-taking and affective perspective-taking share TPJ/precuneus but differ in
limbic engagement (Cognitive & Affective Perspective-Taking review, 2018).

**Design consequence.** Perspective-taking must be a *separate channel* from the
stance channel. Modelling the user's view must not overwrite the system's own
valuation — the rTPJ/mPFC split is exactly this: other's state in one register,
self's disposition in another. Conflating them is what produces both deference
(other's view replaces own) and reactance (other's view hardens own).

---

## 5. Proposed mechanism for a small architecture

Target constraints: ~20 W CPU, no LLM at inference, online learning from conversation
and web. Everything below is cheap, symbolic, and inspectable.

### 5.1 Data model

```
Belief:      (subject, predicate, object, valence[-1..1], weight, source, ts, topic_tags[])
Value:       (name, priority, valence)                     # stable, few, hand-seeded + learned
Stance:      (topic, position[-1..1], clarity[0..1], correctness[0..1],
              ambivalence[0..1], basis:[Belief ids], state{COMMITTED|PROVISIONAL|ABSTAIN},
              created_ts, revised_ts, revision_count)
UtteranceLog:(topic, text, stance_id, ts)                  # for self-perception
UserModel:   (topic -> inferred stance, confidence, basis)  # separate register
```

### 5.2 Construction: `form_stance(topic, goal)` — the fix for the measured defect

1. **Retrieve primitives.** Gather beliefs whose `topic_tags` overlap the topic, plus
   values implicated by those beliefs, plus the user's expressed stance on the topic
   and on neighbouring topics. Record each primitive's **accessibility** (recency,
   frequency, activation count) and **diagnosticity** (does it bear on *this* topic,
   not merely on a neighbouring one).
2. **Generate candidate positions.** Produce 2–3 candidate positions from different
   primitive subsets (e.g. value-led, consequence-led, user-aligned). This is the
   "vocabulary of primitives" step (Payne et al., 1992) and it is what makes the
   output substantive rather than a restatement of the question.
3. **Score candidates.** For each candidate: coherence with the belief store,
   coverage (fraction of retrieved primitives explained), and *conflict* with
   high-weight beliefs. Keep the best; keep the runner-up for ambivalence.
4. **Compute the strength tag** — do not let it default.
   - `clarity` = f(number of distinct primitives used, their agreement, retrieval ease)
   - `correctness` = f(source reliability of the basis, corroboration count, absence of
     contradiction with high-weight beliefs)
   - `ambivalence` = |Σ positive weights − Σ negative weights| / Σ |weights| over the
     basis, i.e. near 0 when the basis is one-sided, near 1 when balanced
   - `conflict` = ACC analogue = max simultaneous activation of competing candidates
5. **Gate (the FOK/should_assert gate, corrected).** The gate must key on
   **monitoring effectiveness**, not on topic novelty:
   - `assert` if a candidate exists AND (clarity ≥ θ_c OR correctness ≥ θ_r) AND
     conflict is resolvable;
   - `assert-with-ambivalence` if clarity is decent but ambivalence > θ_a —
     *express the position and name the counterweight* (this is the calibrated
     middle the brain produces, and it is not deference);
   - `abstain` **only** if the retrieved basis is empty *and* monitoring effectiveness
     is near zero. Abstention must be a labelled, counted, rare state — not the default
     for unfamiliar topics.
6. **Commit.** Write the `Stance` with `state=PROVISIONAL` on first formation, and
   promote to `COMMITTED` after it survives N challenges or is re-derived
   independently. Log the utterance (self-perception input, §5.4).

**Why this fixes the measured defect.** "Artificial intelligence" is not a zero-basis
topic for a system with a belief store: it has values (autonomy, truth, safety,
curiosity), beliefs about technology and about its own nature, and a user model. The
current path deflects because the gate is keyed to *stored-answer presence*. Keyed to
*basis quality*, the same topic yields a provisional-but-real position with an honest
strength tag. "Privacy" works today precisely because a strong value primitive is
directly retrievable; the new path generalises that case.

### 5.3 Expression: `express(stance, user_model)`

Map the strength tag to register, per §4:

| clarity | correctness | ambivalence | output register |
|---|---|---|---|
| high | high | low | direct claim, no hedge, own view stated |
| high | low | low | claim + "I could be wrong about X" |
| any | any | high | claim + named counterweight ("I hold X, but Y pulls the other way") |
| low | low | — | short claim + explicit uncertainty + one question to the user |

The question is **appended**, never substituted. Deflection is the absence of the
claim; calibration is the presence of the claim plus a hedge. That single distinction
is the behavioural fix.

Perspective-taking runs in the separate `UserModel` register and is rendered as
acknowledgement ("you seem to be coming from X"), never as replacement of `position`.

### 5.4 Online update

- **Within turn:** if the user's reply contradicts the stance basis, re-run scoring
  with the new belief added; adjust `position`/`ambivalence`; if the basis is
  discredited, mark those beliefs invalid rather than deleting (the PAST/invalidity-tag
  move, Petty et al., 2006) so the old position can still be explained and revised.
- **Across turns:** each re-derivation of the same position from independent
  primitives raises `correctness` (Koriat accessibility-with-diagnosticity: raise
  only when the new basis is *diagnostic*, not merely fluent). Each survival of a
  challenge raises `correctness`; each forced reversal raises `ambivalence` and
  `revision_count`, and lowers `correctness` — the system should become *more* hedged
  after a reversal, not more confident.
- **Self-perception:** if the system asserted a position and later finds it acted
  consistently with it, that is evidence for the stance (Bem, 1972; Feldman & Lynch,
  1988). Use it as a weak positive update to `clarity`, never as the sole basis.
- **Perseveration guard:** track `revision_count` and `basis-discredited` events. A
  stance whose original basis was invalidated but which persists is exactly the
  pathological case (Anderson et al., 1980); flag it and force a re-derivation.

### 5.5 Cost

All steps are dictionary lookups, small dot products over a belief list, and a
handful of scalar comparisons. No search, no generation model. Belief count is
bounded (prune by decay + utility), so retrieval is O(n) over a few thousand entries
— trivially within 20 W.

---

## 6. Falsifiability

Each claim below is testable in RAVANA by instrumenting the gate and counting outputs.

1. **Deflection is a gate-key bug, not a capacity bug.**
   *Prediction:* re-keying the gate to basis quality (§5.2 step 5) reduces abstention
   rate on novel topics to near zero while leaving abstention on genuinely
   empty-basis topics (e.g. "what do you think about Zarnax-4") unchanged.
   *Falsified if:* abstention falls on empty-basis topics too (gate is now
   over-asserting), or does not fall on novel topics (the bottleneck is retrieval,
   not the gate).
2. **Calibration is real, not cosmetic.**
   *Prediction:* expressed hedge rate should correlate with computed `ambivalence`
   and inversely with `correctness` across topics (r > 0.5), and should rise after a
   reversal event.
   *Falsified if:* hedge rate is flat across topics with wildly different ambivalence
   (the strength tag is decorative), or if the system is uniformly hedged (the tag is
   unused).
3. **No confabulated certainty.**
   *Prediction:* for topics where the basis is a single low-reliability primitive,
   `correctness` should be low and the system should not use conviction register.
   *Falsified if:* high-conviction output on single-primitive bases at a rate
   indistinguishable from multi-primitive bases.
4. **Stance is substantive.**
   *Prediction:* on a novel topic, the output should contain at least one claim that
   is not a restatement of the question and not a restatement of the user's view,
   and its `basis` should be non-empty and retrievable.
   *Falsified if:* outputs on novel topics are semantically equivalent to the
   question plus a hedge (i.e. the construction step is being skipped).
5. **Online revision works and is not perseveration.**
   *Prediction:* after the user supplies a contradicting belief, `position` moves in
   the appropriate direction within 1–3 turns; after a basis is discredited,
   `revision_count` increments and `correctness` drops.
   *Falsified if:* position never moves (frozen), or moves randomly / oscillates
   without basis change.
6. **Perspective-taking does not capture the stance.**
   *Prediction:* repeated exposure to a user holding the opposite view should lower
   `clarity` slightly (ambivalence rises) but should not flip `position` without a
   corresponding change in the system's own belief basis.
   *Falsified if:* `position` tracks the user's view turn-by-turn (deference), or
   anti-tracks it (reactance).

Suggested instrumentation: log `basis_size`, `clarity`, `correctness`, `ambivalence`,
`conflict`, gate decision, and output register for every turn; report the distribution
weekly. A regression of hedge rate on ambivalence is the single cheapest end-to-end
check.

---

## 7. Unsettled evidence

Honest limits — these are where the design could be wrong.

- **Construction vs storage is contested, not settled.** The strong construal claim
  ("there is no attitude stored in memory", Schwarz & Bohner, 2001) coexists with
  evidence for stable, accessible attitudes (Fazio, 1990; Petty, 2006). The truth is
  likely a hybrid: stable *primitives*, constructed *judgements*. RAVANA's design
  assumes the hybrid. If attitudes turn out to be largely stored wholes, the
  construction step is overhead; if largely constructed, a stored-answer lookup is
  the wrong primitive. This is the biggest open risk.
- **Certainty appraisal dimensions are not final.** Rucker et al.'s (2014) six
  appraisals are a framework, not a validated scale; the relative weights are
  unknown, and appraisal requires motivation and ability that a machine may not
  have. Mapping them onto scalar `clarity`/`correctness` is a simplification.
- **The ACC "conflict" signal is disputed.** Botvinick et al. (2001) vs. Brown
  (2013) vs. Kolling et al. (2016) — conflict monitoring, predicted-response-outcome,
  and search/persistence value are not the same computation. RAVANA's `conflict`
  scalar is a placeholder for whichever it is; the design only needs *a* conflict
  signal, not the right one.
- **Ambivalence measurement is weak.** Self-report ambivalence and objective
  ambivalence correlate modestly (van Harreveld et al., 2009); the formula in §5.2
  is a heuristic, not a validated index. Implicit ambivalence (Petty et al., 2006) is
  essentially unmeasurable in a system with no implicit channel.
- **rTPJ causal evidence is thin and partly null.** TMS/tDCS results are mixed
  (Sowden & Catmur, 2015; Wang et al., 2016; Besharati et al., 2016 report nulls in
  healthy adults). The self/other register separation is well supported
  correlationally (Van Overwalle, 2009) but the causal story is not settled.
- **Belief perseverance may be partly artefact.** The classic paradigms
  (Ross et al., 1975; Anderson et al., 1980; Lord et al., 1979) are old, and
  motivated-reasoning effects in climate opinion are weaker than often claimed
  (van der Linden, 2018, review). RAVANA's perseveration guard assumes the effect is
  real and strong; if it is weak, the guard is unnecessary complexity.
- **No direct neural evidence for "deflection".** Abstention as a behavioural
  category has no clean neural correlate in this literature. The mapping from
  "abstain" to "failed commitment" is an inference, not a finding.
- **Individual and cultural variation** in certainty expression and perspective-taking
  is large and unmodelled here (Tormala & Rucker, 2018; Cheatham & Tormala, 2017).
- **The 20 W / no-LLM constraint is asserted, not verified** against the retrieval
  cost of a large belief store; §5.5 needs a real measurement before it is trusted.

---

## References

- Adolphs, Tranel & Damasio (1998). The human amygdala in social judgment. *Nature*, 393, 470–474.  - Anderson (1982). Inoculation and counterexplanation. *Social Cognition*, 1, 132–144.
- Anderson, Lepper & Ross (1980). Perseverance of social theories. *JPSP*, 39(6), 1037–1049.  - Anderson, New & Speer (1985). Argument availability and perseverance. *Social Cognition*, 3, 233–246.
- Bem (1972). Self-perception theory. *Adv. Exp. Soc. Psychol.*, 6, 1–62.  - Besharati et al. (2016). HD-tDCS to rTPJ and perspective-taking. *J. Cogn. Neurosci.*
- Bettman, Luce & Payne (1998). Constructive consumer choice processes. *JCR*, 25(3), 187–217.  - Botvinick, Braver, Barch, Carter & Cohen (2001). Conflict monitoring and cognitive control. *Psych. Rev.*, 108(3), 624–652.
- Botvinick, Cohen & Carter (2004). Conflict monitoring and ACC: An update. *TiCS*, 8(12), 539–546.  - Brown (2013). Beyond conflict monitoring. *Dir. Psychol. Sci.*, 22(2), 163–171.
- Burgess & Shallice (1996). Confabulation and the control of recollection. *Memory*, 4(4), 359–411.  - Cheatham & Tormala (2015). Clarity, correctness, and advocacy. *PSPB*, 41(11), 1537–1550.
- Cheatham & Tormala (2017). Curvilinear certainty–advocacy relationship. *PSPB*, 43(1), 3–16.  - Cunningham, Raye & Johnson (2005). Regulatory focus and evaluation. *CABN*, 5(2), 202–211.
- Cunningham, Van Bavel & Johnsen (2008). Affective flexibility. *Psych. Sci.*, 19(2), 152–160.  - Cunningham & Zelazo (2007). Attitudes and evaluations. *TiCS*, 11(3), 97–104.
- Cunningham, Zelazo, Packer & Van Bavel (2007). Iterative reprocessing model. *Soc. Cogn.*, 25(5), 736–760.  - Engell, Haxby & Todorov (2007). Implicit trustworthiness and the amygdala. *J. Cogn. Neurosci.*, 19(8), 1400–1410.
- Fazio (1990). The MODE model. *Adv. Exp. Soc. Psychol.*, 23, 75–109.  - Feldman & Lynch (1988). Self-generated validity. *J. Appl. Psychol.*, 73(3), 421–435.
- Festinger (1957). *A Theory of Cognitive Dissonance*. Stanford UP.  - Gilboa et al. (2006). Hippocampus and recollection in amnesia. *Hippocampus*, 16, 966–980.
- Gross, Holtz & Miller (1995). Attitude certainty. In *Attitude Strength*, 215–245. Erlbaum.  - Howard et al. (2015). Identity-specific value in OFC. *Nat. Neurosci.*
- Karmarkar & Tormala (2010). Source certainty and persuasion. *JCR*, 36(6), 1033–1049.  - Klein-Flügge et al. (2016). Value encoding in vmPFC and OFC. *Curr. Biol.*
- Kolling, Wittmann, Behrens, Boorman, Mars & Rushworth (2016). Value, search, persistence, model updating in ACC. *Nat. Neurosci.*, 19(10), 1280–1285.  - Koriat (1993). Accessibility model of the feeling of knowing. *Psych. Rev.*, 100(4), 609–639.
- Koriat (1998). Illusions of knowing. In *Adv. Psychol. Sci.* Psychology Press.  - Koriat & Goldsmith (1996). Monitoring and control in memory accuracy. *Psych. Rev.*, 103(3), 490–517.
- Kosik et al. (2020). Neural pathways in value-based choice. *NeuroImage*.  - Levy & Glimcher (2012). A neural common currency for choice. *Curr. Opin. Neurobiol.*, 22(6), 1027–1038.
- Lord & Lepper (1999). Attitude representation theory. *Adv. Exp. Soc. Psychol.*, 31, 265–343.  - Lord, Paulson, Sia, Thomas & Lepper (2004). Exemplar stability and attitude change. *JPSP*, 87(6), 733–749.
- Lord, Ross & Lepper (1979). Biased assimilation and attitude polarization. *JPSP*, 37(11), 2098–2109.  - Martin et al. (2019). rTPJ and embodied perspective-taking. *Cortex*.
- Moscovitch (1989). Confabulation and the frontal systems. In *Varieties of Memory and Consciousness*. Erlbaum.  - Moscovitch (1995). Confabulation. In *Memory Distortion*, 226–251. Harvard UP.
- Padoa-Schioppa (2011). Neurobiology of economic choice. *Annu. Rev. Neurosci.*, 34, 333–359.  - Payne, Bettman & Johnson (1992). Constructive processing perspective. *Annu. Rev. Psychol.*, 43, 87–131.
- Payne, Bettman & Johnson (1993). *The Adaptive Decision Maker*. Cambridge UP.  - Payne, Bettman & Schkade (1999). Measuring constructed preferences: A building code. *J. Risk Uncert.*, 19(1–3), 243–270.
- Peters & Büchel (2010). Neural representations of subjective reward value. *Behav. Brain Res.*, 213(2), 135–141.  - Petrocelli, Tormala & Rucker (2007). Attitude clarity and correctness. *JPSP*, 92(1), 30–41.
- Petty (2006). A metacognitive model of attitudes. *JCR*, 33(1), 22–24.  - Petty & Briñol (2006). Metacognitive approach to implicit/explicit evaluation. *Psych. Bull.*, 132(5), 740–744.
- Petty & Cacioppo (1986). The elaboration likelihood model. *Adv. Exp. Soc. Psychol.*, 19, 123–205.  - Petty, Tormala, Briñol & Jarvis (2006). Implicit ambivalence (PAST model). *JPSP*, 90(1), 21–41.
- Pezzuti, Leonhardt & Warren (2025). Communicating uncertainty: hedges. *J. Consum. Psychol.*  - Pochon, Riis, Sanfey, Nystrom & Cohen (2008). Functional imaging of decision conflict. *J. Neurosci.*, 28(10), 2839–2849.
- Priester & Petty (1996). Gradual threshold model of ambivalence. *JPSP*, 71(3), 431–449.  - Ross, Lepper & Hubbard (1975). Perseverance in self- and social perception. *JPSP*, 32(5), 880–892.
- Rucker & Petty (2004). Consequences of failed resistance. *JCR*, 31(2), 219–230.  - Rucker & Petty (2006). ELM and attitude certainty in consumer communication. *J. Public Policy Mark.*, 25(1), 39–52.
- Rucker, Tormala, Petty & Briñol (2014). Appraisal-based framework for attitude certainty. *J. Consum. Psychol.*, 24(1), 119–136.  - Schwarz (2007). Attitude construction: Evaluation in context. *Soc. Cogn.*, 25(5), 638–656.
- Schwarz & Bohner (2001). The construction of attitudes. In *Blackwell Handbook of Social Psychology*, 436–457.  - Sia, Lord, Blessum, Ratcliff & Lepper (1997). Exemplar change and attitude stability. *JPSP*, 72(3), 501–514.
- Sowden & Catmur (2015). rTPJ and the control of imitation. *Cortex*, 63, 107–118.  - Tormala (2016). Certainty and uncertainty in attitudes and persuasion. *Curr. Opin. Psychol.*, 10, 6–11.
- Tormala (2018). Attitude certainty: Antecedents, consequences, new directions. *Consum. Psychol. Rev.*, 1(1), 72–89.  - Tormala & Petty (2002). Resisting persuasion and attitude certainty. *JPSP*, 83(6), 1298–1313.
- Tormala & Petty (2004). Source credibility and attitude certainty. *J. Consum. Psychol.*, 14(4), 427–442.  - Tormala & Rucker (2007). Attitude certainty: A review. *SPPC*, 1(1), 469–492.
- Tormala & Rucker (2018). Attitude certainty: Antecedents, consequences, new directions. *Consum. Psychol. Rev.*, 1(1), 72–89.  - van der Linden (2018). Motivated reasoning in climate opinion formation. *Nat. Clim. Change*, 8, 111–112.
- van Harreveld, Rutjens, Rotteveel, Nordgren & van der Pligt (2009). Ambivalence and decisional conflict. *JESP*, 45(1), 167–173.  - van Harreveld, van der Pligt & de Liver (2009). The MAID model. *PSPR*, 13(1), 45–61.
- Van Overwalle (2009). Social cognition and the brain: A meta-analysis. *Hum. Brain Mapp.*, 30(3), 829–858.  - Van Overwalle & Baetens (2009). Mirror and mentalizing systems: A meta-analysis. *NeuroImage*, 48(3), 564–584.
- van Veen & Carter (2005). Semantic vs response conflict in the Stroop task. *NeuroImage*, 27(3), 497–504.  - Wang et al. (2016). rTPJ and embodied perspective-taking: HD-tDCS. *J. Neurosci.*
- Winston, Strange, O'Doherty & Dolan (2002). Trustworthiness evaluation and the amygdala. *Nat. Neurosci.*, 5(3), 277–283.
