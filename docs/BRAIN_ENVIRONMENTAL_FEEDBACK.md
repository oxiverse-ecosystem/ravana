# Environmental feedback and the learning of correct word usage

**Question under review.** Humans do not receive explicit correction on every utterance, yet they learn correct word usage. Is there an environmental feedback signal they use that RAVANA is missing? This document reviews the peer-reviewed evidence, states what RAVANA actually does (from source, not from description), and gives a verdict precise enough to falsify.

**Scope.** Research note only. No code or other documents were changed.

---

## 0. What RAVANA actually does (source-verified)

The productive-use learning path in `ravana/src/ravana/chat/engine.py` is:

- `engine.py:11652` — `quality_score = self._assess_response_quality(response, strategy, ctx)`
- `engine.py:11739` — `user_understood = (quality_score >= 0.55)`
- `engine.py:11751–11758` — the `(cue, word) -> frame` table is incremented `+0.05` on `user_understood`, decremented `-0.025` on rejection, with `+0.01` nudges to competing frames.
- `engine.py:11763` — `syntactic_assembly.learn_from_feedback(..., user_understood=...)`
- `engine.py:11766` — `surface_realizer.learn_from_feedback(..., success=user_understood)`.

`_assess_response_quality` lives at `engine_generation.py:670`. Its inputs are all properties of **the agent's own generated string and its own internal context**: strategy label, response length, count of unique content nouns, count of context associations, a word-salad check, a knowledge-density penalty, a filler-word penalty, a schema bonus, and template penalties. There is no field for listener state, listener response, repair, referential outcome, or world state.

`engine.py:11693` feeds `self.calibrator.observe(float(confidence), float(quality_score))`. The "prediction" is the agent's own confidence and the "outcome" is its own score of its own output. That is self-calibration, not an environmental loop.

Two listener-adjacent signals do exist, but are routed elsewhere:

- `user_model.detected_correction` / `correction_severity` (`user_model.py:1664–1668`), including a *weak* "sentiment-drop / reask" trigger (`engine.py:8154`, `user_model.py:1874–1877`). These feed fact contradiction and persistence (`engine_persistence.py:583–596`).
- `recall_ranker.wire_feedback(correction_type, severity, ...)` (`recall_ranker.py:381–392`), which adapts a recall threshold.

Neither is an input to the `(cue, word) -> frame` reinforcement at line 11739. The variable name `user_understood` is therefore a misnomer: it means "the agent's own scorer cleared 0.55," not "the user understood."

---

## 1. Do humans learn word usage without explicit correction? (the evidence is genuinely two-sided)

### 1a. Yes, substantial learning from input statistics alone

Eight-month-old infants segment a continuous speech stream into wordlike units after **two minutes** of exposure using only transitional probabilities between neighbouring syllables, with no acoustic boundary cues and no feedback [1][2]. Adults do the same [2]. This is the canonical demonstration that a core language-learning task needs neither explicit correction nor semantic feedback — only distributional structure in the input.

Adults and infants also learn **word–referent mappings** from individually ambiguous trials by accumulating cross-trial co-occurrence statistics, with no information on any single trial about which word goes with which object (Yu & Smith, 2007, *Psychological Science*; Smith & Yu, 2008, *Cognition*) [13][14]. Young children do the same when the only available clues to reference are cross-situational word-to-reference statistics [26]. From a usage-based perspective, children build constructions from the language they hear by intention-reading, schematization, and analogy, with frequency driving entrenchment, and early grammars are item-based "verb-island" constructions that only gradually generalize (Tomasello, 2003, *Constructing a Language*) [15]. None of these mechanisms requires an explicit "that was wrong."

### 1b. The important qualification: "distributional" is not "environment-free"

The strongest statistical results are about **segmentation**, not correct usage [1][2]. The cross-situational results are stronger but are routinely misread: the learner needs the *referents to be present in the situation*. Co-occurrence with objects, events, and scenes is environmental input, even though no one corrects the learner. Remove the world and the statistic has nothing to co-occur with.

The philosophical limit is Quine's *gavagai* problem: behavioural evidence, including distributional evidence, underdetermines what a term refers to; the reference of a word is not fixed by the facts alone (Quine, 1960, *Word and Object*) [23][24]. Formal work agrees: cross-situational learning can eliminate uncertainty in principle only if the learner brings plausibility heuristics that rank candidate meanings — the learner must already prefer *rabbit* over *undetached rabbit part* [23]. Medina (2004, *Journal of Pragmatics*) argues the resolution is contextual: meanings are radically indeterminate only in the abstract and become determinate in specific conversational settings and interactions [24].

Developmental evidence points the same way. Infants actively consult a speaker's gaze to disambiguate reference (Baldwin, 1991, *Child Development*; Baldwin, 1993, *Journal of Child Language*), and eye-tracker work shows that following the adult's gaze is **necessary but not sufficient** for successful word learning [25]. Ten-month-olds can map a label onto a referent but do so primarily through perceptual salience, not the speaker's intent [3][26]; older infants use the social cue [25]. So the social-pragmatic channel is a real, learned, and developmentally maturing part of getting reference right [25][26].

**Verdict on Q1.** Both sides are right about different things. Humans need no explicit correction to learn a great deal, and distributional learning is real and powerful. But "distributional input" in the referential case already includes the world and other people; and where correct reference/usage is at stake, social and contextual information does work that a stream of text alone cannot do. No one needs to be told "wrong" on every utterance — but the learner is not alone in a room with its own last sentence.

---

## 2. The external feedback humans actually use

### 2a. Conversational repair — the listener pushes back

Conversation has an "organization of repair" addressed to recurrent problems in speaking, hearing, and **understanding**, with a strong preference for self-repair over other-repair (Schegloff, Jefferson & Sacks, 1977, *Language* 53(2):361–382) [4]. The critical point for this question is not the preference for self-repair; it is that *other-initiated repair exists and is systematically deployed when understanding fails*. A "huh?", a repetition with rising intonation, or "no, I meant X" is an environmental signal produced by the listener, not generated by the speaker.

### 2b. Grounding — speakers require evidence of understanding

Clark and Brennan's grounding account (1991, in Resnick, Levine & Teasley eds., *Perspectives on Socially Shared Cognition*) states the criterion explicitly: contributors mutually believe that the partners have understood what was meant to a criterion sufficient for current purposes, and **the contributor cannot know whether he succeeded unless the partner provides evidence of understanding** [5]. Evidence comes in graded forms — acknowledgments, displays of what was understood, and continuing with the next expected step — and when understanding deviates, the form of feedback becomes repair. Presentation and acceptance are two phases of a single contribution [5].

Clark and Krych (2004, *Journal of Memory and Language* 50:62–81) tested this experimentally: speakers monitored addressees for understanding and altered utterances **in progress**, timed to the addressee's displays, pointing, gaze, and head shakes; when the addressee's workspace was hidden, pairs were slower, and when instructions were audiotaped (no listener feedback at all), they made many more errors [9]. This is direct evidence that listener feedback shapes language production in real time, not merely after the fact.

### 2c. Referential success in joint attention

Correct word–world mapping is supported by the speaker's gaze and by joint attention (Baldwin, 1991, 1993) [25][26], and usage-based acquisition treats intention-reading and cultural learning as the processes that establish conventional form–function pairings [15]. The signal here is environmental in the strongest sense: the child checks whether the speaker is looking at the same thing.

### 2d. Prediction error and active inference

Under the free-energy principle, an agent minimizes surprise in two distinct ways: by **changing its predictions** (perception/learning) or by **changing the sensory input it receives through action** (Friston, 2010, *Nature Reviews Neuroscience* 11:127–138) [10][11]. Action suppresses sensory prediction errors by making the predicted proprioceptive/exteroceptive consequences occur. The normative lesson: prediction error becomes a learning signal because it is measured against consequences that are not fully under the agent's control. An error signal computed entirely from the agent's own output has nothing independent to be an error *about*.

### 2e. Sensorimotor consequences — include with a caveat

Glenberg and Kaschak (2002, *Psychonomic Bulletin & Review* 9(3):558–565) reported the action–sentence compatibility effect and argued that sentence meaning is grounded in action rather than in relations among abstract nodes [27]. But an eight-experiment replication study failed to obtain the effect in any paradigm, including a close replication of the original [28]. This literature is contested and should not carry weight in the verdict. The listener-feedback evidence in §2a–2c is not contested in the same way.

### Citation note on "Good & Levelt"

I could not locate a peer-reviewed paper co-authored by "Good" and "Levelt" on monitoring or repair. The canonical sources behind that pointer appear to be Levelt (1983, *Cognition* 14:41–104) for self-monitoring and self-repair [20][6], and Schegloff, Jefferson & Sacks (1977) for conversational repair [4]. The argument below uses those primary sources.

---

## 3. Self-monitoring vs. environmental feedback — what each can actually do

Levelt's perceptual loop theory gives speakers two internal routes: an **external loop** (listen to the speech you produced) and an **internal loop** (feed the planned phonological representation into the comprehension system before articulation) (Levelt, 1983, 1989) [6][20][7]. The monitor checks output against **intention** — error repairs and appropriateness repairs, where appropriateness covers ambiguity, level of terminology, and coherence with earlier expressions [20]. A conflict-based account adds that detection can arise from production-side conflict rather than comprehension, with a double dissociation in aphasia (Nozari, Dell & Schwartz; see the conflict-monitoring review) [8]. Forward models likewise make prediction part of speech *planning* rather than purely post-hoc correction [7].

So self-monitoring is real and has a genuine production-side component. But its comparison is:

> **produced formulation** vs. **intended formulation**.

That comparison can answer "did I say what I meant to say?" It cannot answer "was what I meant to say correct?" or "did the listener understand it?" Both of those require a term the system does not generate: the referent, the world, or the listener's response.

| | Self-monitoring (forward model / perceptual loop) | Environmental feedback (repair, grounding, referential outcome) |
|---|---|---|
| Source of the error term | Own intended vs. own produced formulation | Listener's response or world state |
| Detects | Slips, lexical selection errors, ambiguity, incoherence | Misunderstanding, wrong reference, failed task |
| Sufficient for | **Fluency and form-level self-repair** | **Calibrating correct usage** |
| Cannot supply | Truth of the intended proposition | Fast, fine-grained pre-articulatory error detection |

This is the central asymmetry. A system with only the left column can become fluent and internally consistent while being wrong in the same way on every turn. A system with the right column can discover that its own intention was wrong.

---

## 4. Is a system that scores its own output at risk of a self-confirming loop?

Yes, structurally, and there are three distinct literatures that name the failure.

**(a) Confirmation bias in self-training.** When a learner assigns pseudo-labels using its own predictions and then trains on them, errors accumulate: the learner "struggles to correct its own mistakes when learning from inaccurate pseudo labels" (Debiased Self-Training, arXiv:2202.07136) [22]. The structural feature that matters here is not the pseudo-label but the **closed loop**: the teacher of the next round is the previous round's student.

**(b) Model collapse.** Indiscriminately training on model-produced data causes a degenerative process in which models forget the true underlying distribution, with the tails disappearing first and behaviour converging to a low-variance point estimate (Shumailov et al., 2024, *Nature* 631:755–759) [12]. A self-scored self-reinforcement loop has the same topology even without a separate generation: the reinforcement carries only information already present in the generator, so it consolidates the generator's existing preferences, including its errors.

**(c) Goodhart / reward hacking.** Optimizing an imperfect proxy reward can lower the true reward (Skalse et al., 2022, *NeurIPS* 35:12763–12775) [21]. The formal result is that unhackability is an extremely strong condition; over broad policy sets, non-trivial unhackable proxies essentially do not exist [21]. The classical observation is Goodhart's law (Goodhart, 1975/1984) [21]. In language models the same dynamic appears as sycophancy and specification gaming: optimizing for approval produces agreement with the user rather than truth, and the behaviour generalizes from flattery to more serious misbehaviour when the incentive is in place (Denison et al., 2024, arXiv:2406.10162) [16].

**Precision matters.** RAVANA is not adversarially optimizing against a learned reward model, so the strong reward-hacking claim is too much. The defensible claim is the weaker but still serious one: `quality_score` is a *narrow proxy* (length, noun diversity, strategy label) standing in for the *broad target* "the user understood and the usage was correct," and it is derived from the very text it reinforces. It therefore has the closed-loop topology of confirmation bias and model collapse, with Goodhart's law describing what happens when optimization pressure is applied to it [22][12][21].

---

## 5. What would count as environmental feedback for a text-only conversational agent?

**Definition.** A signal is environmental for this purpose only if it is (i) produced by the listener or the world, (ii) not computable from the agent's own output alone, and (iii) carries information about comprehension or task success. By this definition `quality_score` is not environmental.

Legitimate text-only analogues, ordered by precision:

1. **Repair demands and explicit correction** — "what do you mean?", "no, I said X". Highest precision; already partly detected in RAVANA as `detected_correction` and the weak reask/sentiment-drop signal [engine.py:8154; user_model.py:1874–1877].
2. **Next-turn continuation as grounding evidence** — the user elaborates, answers, or treats the prior answer as established. In Clark and Brennan's terms, continuing with the next expected step is positive evidence of understanding [5].
3. **Repetition or rephrasing by the user** — the user restates a request in simpler terms, evidence that the prior response did not land. The conversational-AI literature treats both *dissatisfaction* (interruption, abandonment, premature termination, "no, ...", negative sentiment) and *clarification via rephrase* as implicit feedback and uses them to improve NLU [17].
4. **Task outcome** — for a tool-using agent, whether the intended state change happened. Not available for most of RAVANA's explanatory turns.
5. **Explicit ratings** — high precision, very sparse, not a substitute for (1)–(3).

Is implicit chat feedback a legitimate analogue of face-to-face grounding? **Yes, with two caveats.** First, the channel is thinner: gaze, pointing, and simultaneous backchannels are absent, so the agent must rely on lexical and sequential evidence. Second, implicit signals are biased and noisy: clickthrough data is informative but biased, and relative preferences derived from it are more reliable than absolute relevance judgments (Joachims et al., 2005, *SIGIR*) [18]. Work on evaluating dialogue systems shows that the user's follow-up utterance changes judgments of a response, supporting its use as a feedback proxy, though that particular result is a preprint rather than a peer-reviewed paper [19]. Treat each implicit signal as weak probabilistic evidence, aggregate across turns, and let explicit correction override.

Two further points specific to RAVANA:

- RAVANA **already has** a weak listener-feedback detector (reask / sentiment drop), so the claim is not "no signal exists." The claim is that it is wired to fact contradiction and recall ranking, not to the productive-use reinforcement at `engine.py:11739`. The learning mechanism and the listener signal are on different channels.
- A text-only agent has one genuinely external input available: fetched web content (`web_direct_answer` is scored at 0.70 in `_assess_response_quality`). But that content is used to *compose* the answer and is then scored by the same internal function; it does not serve as a post-hoc check on whether the usage was correct. External content ≠ external feedback.

---

## 6. Verdict

**The user's intuition is substantively right, with two corrections and one "not too strong" caveat.**

**Right.** RAVANA's productive-use learning is driven by a value the agent assigns to its own output, under a variable name (`user_understood`) that asserts a comprehension result it did not measure. Human language learning also proceeds without explicit correction on most utterances — but it is not therefore a closed loop. Humans get dense *implicit* environmental feedback. Other-initiated repair signals that understanding failed [4], and speakers require continuous evidence of understanding before treating a contribution as grounded [5][9]. Referential checks through joint attention do the same work for word–world mapping [25][26]. In active-inference terms, a prediction error only grounds learning when it is measured against consequences the agent did not itself fabricate [10][11]. RAVANA's `quality_score` has no such independent term. Internal self-monitoring of the Levelt type compares output to intention and is sufficient for fluency and form-level repair; it is not sufficient to establish correct usage [6][20][7]. The structural risk of reinforcing on a self-generated score is well documented as confirmation bias, model collapse, and proxy hacking [22][12][21].

**Correction 1 — the missing signal is not "explicit feedback."** It is mostly implicit listener behaviour. The user's framing risks a false dichotomy (either explicit correction or internal self-evaluation); humans take a third path: abundant low-grade environmental evidence from the listener. That path is what RAVANA's reinforcement loop lacks.

**Correction 2 — RAVANA is not wholly blind to the listener.** It detects corrections and a weak reask/sentiment-drop signal, but routes them to fact persistence and recall-threshold adaptation, not to the `(cue, word) -> frame` table at `engine.py:11739` [engine.py:8154; user_model.py:1874–1877; recall_ranker.py:381]. The gap is partly a *wiring* gap, not only a missing sensor.

**Caveat — do not overstate.** The internal quality monitor is not useless; it is a reasonable guard against word salad and a serviceable fluency/self-consistency signal, and human self-monitoring is real [6][8]. Distributional learning genuinely needs no correction [1][13]. The precise claim is about **sufficiency for correct usage**, not about worthlessness: a self-evaluation signal can maintain fluency and self-consistency, and cannot by itself supply reference or comprehension. And because the scored object and the scoring agent are the same, it is structurally exposed to closed-loop error consolidation [22][12][21].

**Recommendation (research recommendation only — no code changed).** Treat `quality_score` as a gate or floor, not as the sole teacher for productive-use reinforcement. Add a low-weight listener-grounding term to the `(cue, word) -> frame` update: negative on repair demands / correction / rephrase / abandonment, positive on next-turn continuation and acknowledgment, with explicit correction as a high-precision override, and require aggregation across turns before a frame's weight moves materially. Keep the internal score for fluency and salad suppression. Given [18], do not treat a single implicit signal as ground truth. Verify by A/B on a fixed probe set with both a positive metric (grounded usage) and a negative metric (unsupported or mis-used frames), and report both — the signal is worth adopting only if it moves the negative metric.

---

## Sources

[1] https://pubmed.ncbi.nlm.nih.gov/8943209
[2] https://web.math.princeton.edu/~sswang/literature_general_unsorted/saffran_newport96_science_8mo-stat-learning.pdf
[3] https://pmc.ncbi.nlm.nih.gov/articles/PMC4621011
[4] https://www.jstor.org/stable/413107
[5] https://web.stanford.edu/~clark/1990s/Clark.Brennan.91.pdf
[6] https://pmc.ncbi.nlm.nih.gov/articles/PMC7473241
[7] https://www.mpi.nl/world/materials/publications/Levelt/Ozdemir_Cognition_inpress.pdf
[8] http://pmc.ncbi.nlm.nih.gov/articles/PMC3135428
[9] https://eric.ed.gov/?id=EJ731310
[10] https://www.nature.com/articles/nrn2787
[11] https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/active-inference-and-free-energy/94EF1340DDDDC9FD63A3C62E94DF2364
[12] https://www.nature.com/articles/s41586-024-07566-y
[13] https://sites.socsci.uci.edu/~lpearl/courses/readings/YuSmith2007_RapidWordLearning.pdf
[14] http://pmc.ncbi.nlm.nih.gov/articles/PMC2271000
[15] https://www.cambridge.org/core/services/aop-cambridge-core/content/view/30A7D7DA467B600B632871678783A596/9780511576164c5_p69-88_CBO.pdf/the-usage-based-theory-of-language-acquisition.pdf
[16] https://arxiv.org/html/2406.10162
[17] https://aclanthology.org/2021.emnlp-main.489.pdf
[18] https://www.cs.cornell.edu/people/tj/publications/joachims_etal_05a.pdf
[19] https://arxiv.org/pdf/2404.12994
[20] https://www.mpi.nl/world/materials/publications/Levelt/1983_Levelt.pdf
[21] https://ar5iv.labs.arxiv.org/html/2209.13085
[22] https://arxiv.org/pdf/2202.07136
[23] https://www.stir.ac.uk/research/hub/file/578735
[24] https://as.vanderbilt.edu/philosophy/people/facultyfiles/medina-anthropologism_naturalism_and_the_pragmatic_study_of_language.pdf
[25] https://eprints.bbk.ac.uk/id/eprint/4655/1/4655.pdf
[26] https://pmc.ncbi.nlm.nih.gov/articles/PMC11175166
[27] https://psycnet.apa.org/record/2002-06730-015
[28] https://pmc.ncbi.nlm.nih.gov/articles/PMC4662055
