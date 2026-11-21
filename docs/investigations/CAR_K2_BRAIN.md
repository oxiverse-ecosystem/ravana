# CAR K2 — why two spaced repetitions fail under load/interference, and what brain mechanism would fix it

**Status:** final. **Method:** READ-ONLY literature investigation. No engine, pytest, measurement
script, or source file was run or modified; this document is the only artifact owned by this task.
**Question:** why do some newly taught associations fail after only TWO spaced repetitions under
high item load / interference while other items with the same exposure count succeed, and what
brain mechanism would make 2-exposure acquisition robust **without changing the benchmark's
thresholds**?

**Verification convention.** Every citation below was checked against a publisher landing page or
PubMed/Europe PMC record in this task; author, year, journal, volume/pages, DOI and (where
available) PMID/PMCID are given in §4. Citations already verified in the repo are labelled
*(repo-verified)*; citations newly verified here carry their PMID/PMCID or publisher record.
Claims that could not be sourced are marked `[unverified]` and are not used to carry a fix.
Findings are mapped to RAVANA primitives only as an engineering analogy — no animal/human result
is claimed to be literally implemented by a graph edge.

**Observed pattern being explained (from `CAR_ROOT_CAUSE.md`, `LEARN5_CAR_MEASURED.md`,
`LEARN5_CAR_POSTFIX.md`, read-only):** at k=2 the primary curve is non-monotonic (worst adjacent
reversal −0.10 primary, −0.20/−0.30 sealed); ~7/10 sealed k=2 items have `raw_chains=0`
in-process; the failures are *item-composition-sensitive*, clustering on weak-weight taught hops
(w≈0.19–0.50, c≈0.39–0.54), on items whose tokens are shared with other items (`fossil`,
`refraction`, `glow`, `causes`, `happens`), and on curriculum-known subjects whose pre-existing
chain caps the measurable delta. The harness teaches a 2-hop causal chain in spaced rounds and
asks a why-question after each teaching rep (`scripts/measure_learn5_car.py`;
`mechanism_intent()` in `response_gen.py:233-250`).

---

## 1. Mechanism map: observed pattern → brain findings

### 1.1 Two exposures are a consolidation window, not an exposure count

The decisive variable is not "how many times was the item shown" but "what happened to the trace
between and after showings." A first teaching rep writes a **fast, labile, episode-tagged**
association; a second rep either (a) successfully retrieves and re-stabilizes it, or (b) fails to
address it and leaves it available for overwrite. Synaptic tagging & capture gives the cellular
form: a weak stimulus sets a local, time-limited tag that can capture plasticity-related proteins
supplied by a strong event, converting early into late LTP (Frey & Morris 1997, *repo-verified*).
Reconsolidation gives the systems form: reactivation returns a consolidated memory to a labile
state requiring renewed stabilization (Nader, Schafe & LeDoux 2000, *repo-verified*; Hardt,
Einarsson & Nader 2010). Complementary learning systems supplies the architecture: a fast
hippocampal one-shot store and a slow neocortical statistical store, with interleaved replay
preventing the fast traces from being dissolved into the slow ones (McClelland, McNaughton &
O'Reilly 1995, *repo-verified*; Liao & Losonczy 2024).

**RAVANA mapping.** `edge_kind='curated'` + `source`/`evidence`/`acquired_by` is the synaptic tag;
`weight`/`confidence` at w≈0.19–0.50 / c≈0.39–0.54 is the early-LTP analogue; the why-question
probe is the reactivation event; cold save/load is the consolidation window. The measured
load-time `curated→curriculum` re-stamp (`engine_graph.py:930-938`, CAR_ROOT_CAUSE §1.2) destroys
the tag at exactly the moment the trace should be captured. A count of two reps is therefore
compatible with either robust retention or complete loss depending on whether the tag survived
and whether the second rep was a retrieval.

### 1.2 Why shared-token items fail: cue overlap, initial strength, competitor count

All three candidate determinants named in the question are supported, at different levels:

- **Cue overlap.** The A-B, A-C paradigm is the direct demonstration: when the same cue A is
  later paired with C, recall of the original B response is reduced even when competition is
  removed on an MMFR test — i.e. there is genuine unlearning of the first association, not just
  response competition at test (Barnes & Underwood 1959). Transfer magnitude varied with response
  similarity and with degree of first-list learning (Underwood 1951): similar competitors and a
  weak original association are the vulnerable combination.
- **Initial strength.** The fan effect shows that retrieval of a fact slows as more facts attach
  to its concept (Anderson 1974), and that overlearning reduces the fan effect (Anderson 1983).
  A weak trace with a high fan-out cue loses the activation race; a well-learned trace resists it.
- **Number of competing traces.** Prior knowledge about a person speeds learning of new facts
  about that person, slows their retrieval, and *shrinks* fan effects when the new facts are
  integrated into an existing network (Anderson 1981). The number of traces on the cue is thus a
  real determinant, but its sign depends on whether the traces were integrated or merely piled up.
- **Retrieval-side competition.** Retrieving one member of a competing set suppresses other
  members (retrieval-induced forgetting; Anderson & Spellman 1995, *repo-verified*) — which is why
  the brain-faithful repair is to inhibit *untrusted competitors* at selection, never to weaken the
  taught edge.

**RAVANA mapping.** Items whose tokens recur across teaching sentences (the `fossil`,
`refraction`, `glow`, `causes`/`happens` family in the audit JSON) are the A-B/A-C analogue: the
shared node is cue A. `score_chain` ranks by `sum(log(w*c)) − 0.35·len − 0.25·unprovenanced`
(`explanation.py:147`), so a high-weight untrusted co-occurrence chain outranks a low-weight
trusted taught chain, and `enumerate_chains` truncates at limit 64 *before* grading
(`explanation.py:261-265`). That is a retrieval-competition failure with the same computational
shape as cue overload, even though the measured cause of the primary k=2 reversal was the
load-path overwrite and truncation, not AB-AC unlearning (see §5).

### 1.3 What the why-question after each rep does: consolidate AND destabilize

The why-probe is not neutral. Three effects are in play and they pull in opposite directions:

1. **Testing consolidates.** Retrieval practice produces more durable learning than restudying
   (Roediger & Karpicke 2006, *repo-verified*), and beats elaborative concept mapping on a later
   inference test by ~50% (d=1.50), with the advantage attributed to retrieval-specific rather
   than elaborative processes (Karpicke & Blunt 2011). Mechanistically, retrieval has been
   proposed to act as a *rapid consolidation event* that creates adaptive hippocampal–neocortical
   representations through online reactivation of associative information (Antony, Ferreira,
   Norman & Wimber 2017 — a Forum proposal, not a new experiment).
2. **Even a failed test potentiates the next encoding.** Unsuccessful retrieval attempts enhance
   subsequent learning (Kornell, Hays & Bjork 2009); conditional-probability analyses isolate a
   genuine *test-potentiated learning* effect in which failed retrieval attempts improve the
   effectiveness of the following restudy, distinct from the direct testing effect (Arnold &
   McDermott 2013).
3. **Reactivation opens a lability window.** A subtle reminder is sufficient to return an episodic
   memory to a state in which new information can be incorporated; the incorporated items intrude
   into recall of the original list, and the effect is time-dependent (Hupbach, Gomez, Hardt &
   Nadel 2007). This is reconsolidation as a *constructive* process and therefore as a risk: if a
   later similar item with a shared cue is active during the window, it can be written into the
   earlier trace.

**RAVANA mapping.** The harness's why-question is simultaneously (a) a retrieval-practice event
that should capture the tag, (b) a test-potentiation event that makes the *next* teaching rep more
effective even when the current answer fails, and (c) a reconsolidation window in which the
load-time re-stamp can substitute schema defaults for the episode record. Whether two reps help
or hurt depends on which of these three dominates — and all three are decided by the distinctness
and provenance of the trace, not by the exposure count.

### 1.4 Why high-pre-existing-structure items are harder

The question's contrast — `bruise/impact/rupture` (rich existing structure) versus
`hiccup/spasm/interruption` (novel arbitrary pairs) — maps onto a real dissociation between
**assimilation** and **pattern separation**:

- A schema lets schema-consistent paired associates be assimilated in one trial and become
  rapidly hippocampal-independent (Tse et al 2007, *repo-verified*); the updated CLS account
  likewise holds that schema-consistent information can be learned rapidly by the slow system
  (Kumaran, Hassabis & McClelland 2016, *repo-verified*). Prior knowledge therefore *can* be a
  one-shot accelerator.
- But prior knowledge can equally *hinder* new learning. When famous (highly familiar) faces are
  paired with novel faces, successful associative memory depends on the hippocampus **separating**
  the two representations; in cortex the novel face is instead **assimilated** toward the famous
  one, and separation specifically mediated memory when interference from previous associations
  had to be overcome (Bein, Reggev & Maril 2020). Without separation, the new item collapses onto
  the incumbent and the teaching adds no distinct route.
- The sign of the prior-knowledge effect is load-dependent. Schema benefit and proactive
  interference coexist neurally (mPFC vs left vlPFC/PCC); under **high attentional load**, greater
  vlPFC functional connectivity was associated with *greater* proactive interference (Oren et al
  2017). This is the closest human analogue of "high item load" flipping an item from benefit to
  cost.
- Probe with a cue that already has rich structure and the incumbent wins the conflict; source
  monitoring is what should identify the new episode as the source of the current claim
  (Johnson, Hashtroudi & Lindsay 1993, *repo-verified*).

**RAVANA mapping.** The curriculum scaffold is RAVANA's schema. Curriculum-known subjects
(`rust`, `earthquake`) retained best because the taught chain could attach to an existing trusted
chain; `amber`'s "immediate trusted chain" was itself a composite borrowing `fossil`'s hop. But
`_seed_curriculum_causal` currently *relabels* taught conversation edges as `curriculum` on every
cold load, i.e. it performs cortical assimilation **without** the hippocampal separation step that
Bein et al. show is required for prior knowledge to be helpful under interference. The repair is
"schema proposes, source monitoring disambiguates, never overwrite the episode tag" — not
"remove the schema."

### 1.5 Which mechanism would let TWO repetitions suffice

Of the three options in the question, the evidence ranks them as follows:

| Candidate | Verdict from literature | Why |
|---|---|---|
| One-trial learning from a reliable teaching source | **Supported as the encoding half, with a caveat** | Fast/binding plasticity can form a memory after a single brief experience (Liao & Losonczy 2024); adults retain a single hypothesized word–referent pairing and abandon it only if a later instance disconfirms it — "propose but verify" (Trueswell, Medina, Hafri & Gleitman 2012); a schema lets one-trial assimilation (Tse et al 2007). Caveat: no verified study shows that a *trusted-source tag alone* sets confidence; the variable in fast-mapping work is referential uncertainty/corroboration, not source prestige `[unverified]`. |
| Elaboration / rehearsal | **Not the lever** | Deeper semantic encoding does produce more durable traces (Craik & Lockhart 1972), but when pitted against retrieval practice on a later inference test, elaborative concept mapping was *inferior* despite equating initial learning (Karpicke & Blunt 2011). Rehearsal is not what the second rep should be doing. |
| Retrieval-practice scheduling | **Best supported as the consolidation half** | Testing effect (Roediger & Karpicke 2006), retrieval-specific advantage (Karpicke & Blunt 2011), retrieval-as-consolidation (Antony et al 2017), test-potentiated encoding after failed tests (Kornell et al 2009; Arnold & McDermott 2013), expanding-interval spacing whose optimum grows with retention interval (Cepeda et al 2006, *repo-verified*). |

**The mechanism that satisfies the "two reps, no threshold change" constraint:** rep 1 writes a
fast, high-confidence, distinctly tagged trace from the trusted teaching sentence (one-shot /
propose-but-verify); rep 2 is an *effortful retrieval*, not a re-teach, which re-stabilizes the
trace if successful and potentiates the next encoding if it fails; and the two reps are
interleaved with distinct episode context tags so that shared tokens do not collapse onto one
cue. This changes **what is stored and how it is addressed**, leaving `grade_chain`,
`assertable_chain`, `THETA_ASSERT` and the trust set untouched. Confidence at first sighting is a
change to stored edge data, not to the decision threshold — but it must be A/B'd with a k=0
control to prove it did not become a de facto trust bypass (§5).

---

## 2. Fix proposals in RAVANA primitives

All three are hypotheses with falsifiers, not results. Execution belongs to the execution-owning
agent; A/Bs must use identical copied state, one process, fixed hash seed, offline mode, matched
compute, and unchanged `grade_chain`/`assertable_chain`/`THETA_ASSERT`/trust set.

### K2-A — Confidence target at first sighting for trusted teaching

- **Targets:** `learn/intake.py` source map (`:80-81`); `KnowledgeIntake.novel_write`'s deliberately
  low `0.15 + 0.35·retention_prior` (`BRAIN_TO_RAVANA_LEARNING.md:304-310`); the `curated` edge's
  `edge_kind`/`source`/`evidence`/`acquired_by` stamp; `source_metadata` as the capture tag.
- **Change in primitives:** for a *trusted teaching* source (user/seed or authored-scaffold with
  an episode stamp), write the first causal hop at a confidence target at or above the trusted
  floor instead of the novel-write floor, stamp `acquired_by='user_assertion'` and
  `source='conversation:<turn>'`, and treat it as **propose-but-verify**: retain the high
  confidence unless a later rep disconfirms it. Do not add a trust bypass and do not touch the
  grading gate.
- **Measurable prediction:** on the identical CAR bank at k=2, the weak-weight taught hops
  (amber/magnify/stalactite class) are assertable immediately and survive cold reload at a higher
  rate than the current k=2 baseline; answer reach on the k=0 control stays 0; known-bad probe
  rejections unchanged.
- **Falsifier:** if cold assertability is unchanged *while the chain exists in the census*, the
  binding constraint is rank/truncation or provenance loss, not initial strength — demote K2-A to
  a no-effect. If the k=0 control becomes assertable or bad probes are accepted, the high-confidence
  write leaked a trust bypass — revert immediately.
- **Brain basis:** fast single-shot hippocampal binding (Liao & Losonczy 2024; McClelland et al
  1995); propose-but-verify one-trial retention (Trueswell et al 2012); synaptic tag must be set
  to be capturable (Frey & Morris 1997); prior-knowledge integration accelerates learning
  (Anderson 1981); Tse et al 2007 for one-trial assimilation.

### K2-B — Make repetition 2 a retrieval event with expanding-interval scheduling

- **Targets:** `KnowledgeIntake.due_for_review()` / `record_review(recalled=...)` expanding interval
  (`BRAIN_TO_RAVANA_LEARNING.md:316-318`); the why-question path (`mechanism_intent()`,
  `_why_how_causal_chain`); replay/reconsolidation on save-load.
- **Change in primitives:** rep 2 is a retrieval probe graded by the *unchanged* `assertable_chain`.
  On `assert`, reinforce exactly the traversed hops' `weight`/`confidence` and restore
  `source_metadata` from the item's own record (never a blanket stamp). On failure, do not
  strengthen any competitor; schedule the next probe at a shorter interval (test-potentiated
  encoding). Match cycles, triples and wall clock across arms; never buy the gain with extra
  compute. If a later shared-cue item is active in the lability window, separate it before
  reactivation rather than after.
- **Measurable prediction:** k=2 cold retention rises over a restudy-matched control at matched
  compute; items whose first probe failed show a larger second-rep gain than items whose first
  probe succeeded (the Kornell/Arnold signature); untrusted-edge count and known-bad rejections
  are flat.
- **Falsifier:** if failed-probe items benefit no more than passive restudy, test potentiation is
  absent in this graph — record the negative. If the gain appears only with a compute overrun, it
  is "more replay", not consolidation — revert. If shared-cue intrusion appears after the probe
  (Hupbach 2007 signature), the reconsolidation risk dominates and the schedule must be changed.
- **Brain basis:** testing effect (Roediger & Karpicke 2006; Karpicke & Blunt 2011); retrieval as
  fast consolidation (Antony et al 2017); test-potentiated learning (Arnold & McDermott 2013;
  Kornell et al 2009); reconsolidation and its lability (Nader et al 2000; Hupbach et al 2007;
  Hardt et al 2010); synaptic capture (Frey & Morris 1997); spacing law (Cepeda et al 2006).

### K2-C — Token-overlap-aware interleaving with distinct episode context tags

- **Targets:** the CAR teaching schedule (`scripts/measure_learn5_car.py` spacing rounds); the
  CLI interleaved positive control; `score_chain`/`enumerate_chains` candidate selection
  (`explanation.py:147,261-265`); `_seed_curriculum_causal` overwrite guard
  (`engine_graph.py:930-938`).
- **Change in primitives:** compute token overlap across the teaching triples; schedule
  high-overlap items non-adjacently (interleaved rather than blocked) and stamp each item's edges
  with a distinct episode context tag so selection can prefer the route matching the query's
  context. Keep the traversal budget at 64 and all thresholds fixed. The schema may still CREATE
  absent hops but may not relabel a conversation-stamped `curated` edge. Inhibiting an untrusted
  competitor must never weaken the taught edge (cf. the rank-first literature mapping in
  CAR_ROOT_CAUSE §2.2).
- **Measurable prediction:** at fixed k=2 the cold-assertability gap between high-overlap and
  low-overlap items shrinks; measured fan-out on shared tokens falls; the CLI interleaved positive
  control (concurrent A+B ≈0.875/1.000) is preserved; the A-only vs sequential A→B interference
  contrast shrinks at matched compute.
- **Falsifier:** if token overlap does not predict which k=2 items fail in the audit JSON, the
  interference variable is absent in CAR and the fix is inert — record the negative. If
  interleaving degrades immediate acquisition without a delayed gain (the contextual-interference
  cost reported for interleaved acquisition), revert. If unrelated fronts regress, the tag-aware
  selection leaked — revert that half.
- **Brain basis:** pattern separation resolves interference from prior associations (Bein et al
  2020); cue overload / fan effect (Anderson 1974; Anderson 1983); A-B/A-C retroactive interference
  (Barnes & Underwood 1959; Underwood 1951); proactive interference under high load (Oren et al
  2017); interleaved CLS replay (McClelland et al 1995; McCloskey & Cohen 1989, *repo-verified*);
  retrieval-induced forgetting (Anderson & Spellman 1995); source monitoring (Johnson et al 1993).

---

## 3. Ranking and sequencing

K2-B has the strongest direct support for "what the second rep should be" and the cheapest
falsifier (compute-matched ON/OFF). K2-A has the strongest support for "why a first rep can fail"
but the largest safety risk (a hidden trust bypass) and must be run with a k=0 control. K2-C is the
only proposal that addresses the *high item load / interference* half of the question directly,
but it also has the strongest prior that the measured CAR reversal is not an interference effect
(§5) — so it must first be checked against the audit JSON before any code is written. Order:
instrument the overlap/weight census → K2-B A/B → K2-A A/B with k=0 guard → K2-C only if overlap
predicts failures.

---

## 4. Verified references

**Already verified in the repo (full details in `CAR_ROOT_CAUSE.md` §2 and `BRAIN_RETRIEVAL_AND_REPLAY.md` §7); re-checked names/DOIs against the cited records here:**
- Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. *Psychological Science, 17*(3), 249–255. doi:10.1111/j.1467-9280.2006.01693.x.
- Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin, 132*(3), 354–380. doi:10.1037/0033-2909.132.3.354.
- Frey, U., & Morris, R. G. M. (1997). Synaptic tagging and long-term potentiation. *Nature, 385*(6616), 533–536. doi:10.1038/385533a0.
- Nader, K., Schafe, G. E., & LeDoux, J. E. (2000). Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. *Nature, 406*(6797), 722–726. doi:10.1038/35021052.
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review, 102*(3), 419–457. doi:10.1037/0033-295X.102.3.419.
- Tse, D., Langston, R. F., Kakeyama, M., Bethus, I., Spooner, P. A., Wood, E. R., Witter, M. P., & Morris, R. G. M. (2007). Schemas and memory consolidation. *Science, 316*(5821), 76–82. doi:10.1126/science.1135935.
- Anderson, M. C., & Spellman, B. A. (1995). On the status of inhibitory mechanisms in cognition: Memory retrieval as a model case. *Psychological Review, 102*(1), 68–100. doi:10.1037/0033-295X.102.1.68.
- Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring. *Psychological Bulletin, 114*(1), 3–34.
- Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? *Trends in Cognitive Sciences, 20*(7), 512–534. doi:10.1016/j.tics.2016.05.004.
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks. *Psychology of Learning and Motivation, 24*, 109–165. doi:10.1016/S0079-7421(08)60536-8.

**Newly verified in this task (publisher / PubMed / PMC record given):**
1. Barnes, J. M., & Underwood, B. J. (1959). "Fate" of first-list associations in transfer theory. *Journal of Experimental Psychology, 58*(2), 97–105. doi:10.1037/h0047507 (APA PsycNET record 1960-03903-001; Europe PMC PMID 13796886).
2. Underwood, B. J. (1951). Associative transfer in verbal learning as a function of response similarity and degree of first-list learning. *Journal of Experimental Psychology, 42*(1), 44–53. doi:10.1037/h0059484 (Europe PMC PMID 14880655).
3. Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. *Cognitive Psychology, 6*(4), 451–474. doi:10.1016/0010-0285(74)90021-8 (publisher/Crossref record; paradigm source).
4. Anderson, J. R. (1983). Retrieval of information from long-term memory. *Science, 220*(4592), 25–30. doi:10.1126/science.6828877 (PubMed PMID 6828877 — "Fanning of multiple paths from a node dissipates the activation … and increases retrieval time"; "Fan effects are reduced as subjects overlearn the material").
5. Anderson, J. R. (1981). Effects of prior knowledge on memory for new information. *Memory & Cognition, 9*(3), 237–246. doi:10.3758/BF03196958 (Springer publisher PDF).
6. Hardt, O., Einarsson, E. Ö., & Nader, K. (2010). A bridge over troubled water: Reconsolidation as a link between cognitive and neuroscientific memory research traditions. *Annual Review of Psychology, 61*, 141–167. doi:10.1146/annurev.psych.093008.100455 (PubMed PMID 19575608; Annual Reviews vol. 61 record).
7. Hupbach, A., Gomez, R., Hardt, O., & Nadel, L. (2007). Reconsolidation of episodic memories: A subtle reminder triggers integration of new information. *Learning & Memory, 14*(1–2), 47–53. doi:10.1101/lm.365707 (PMC1838545).
8. Antony, J. W., Ferreira, C. S., Norman, K. A., & Wimber, M. (2017). Retrieval as a fast route to memory consolidation. *Trends in Cognitive Sciences, 21*(8), 573–576. doi:10.1016/j.tics.2017.05.001 (PMC5912918; Cell Press record).
9. Arnold, K. M., & McDermott, K. B. (2013). Test-potentiated learning: Distinguishing between direct and indirect effects of tests. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 39*(3), 940–945. doi:10.1037/a0029199 (PMC3764602, PMID 22774852).
10. Kornell, N., Hays, M. J., & Bjork, R. A. (2009). Unsuccessful retrieval attempts enhance subsequent learning. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 35*(4), 989–998. doi:10.1037/a0015729 (PubMed PMID 19586265; APA PsycNET record).
11. Karpicke, J. D., & Blunt, J. R. (2011). Retrieval practice produces more learning than elaborative studying with concept mapping. *Science, 331*(6018), 772–775. doi:10.1126/science.1199327 (PubMed PMID 21252317; Science publisher page).
12. Bein, O., Reggev, N., & Maril, A. (2020). Prior knowledge promotes hippocampal separation but cortical assimilation in the left inferior frontal gyrus. *Nature Communications, 11*, 4590. doi:10.1038/s41467-020-18364-1 (PMC7490707).
13. Oren, N., Shapira-Lichter, I., Lerner, Y., Tarrasch, R., Hendler, T., Giladi, N., & Ash, E. L. (2017). Schema benefit vs. proactive interference: Contradicting behavioral outcomes and coexisting neural patterns. *NeuroImage, 158*, 271–281. doi:10.1016/j.neuroimage.2017.07.006 (PubMed PMID 28689053).
14. Craik, F. I. M., & Lockhart, R. S. (1972). Levels of processing: A framework for memory research. *Journal of Verbal Learning and Verbal Behavior, 11*(6), 671–684. doi:10.1016/S0022-5371(72)80001-X (APA PsycNET record 1973-20189-001; CiNii record).
15. Liao, Z., & Losonczy, A. (2024). Learning, fast and slow: Single- and many-shot learning in the hippocampus. *Annual Review of Neuroscience, 47*, 187–209. doi:10.1146/annurev-neuro-102423-100258 (PubMed PMID 38663090; Annual Reviews vol. 47 record).
16. Trueswell, J. C., Medina, T. N., Hafri, A., & Gleitman, L. R. (2012/2013). Propose but verify: Fast mapping meets cross-situational word learning. *Cognitive Psychology, 66*(1), 126–156. doi:10.1016/j.cogpsych.2012.10.001 (PMC3529979; ERIC EJ995549).

---

## 5. What the evidence does NOT support

- **No study manipulates the exact benchmark.** Nothing in the literature combines "two spaced
  repetitions", a why-question after each rep, high item load, and a semantic graph with fixed
  assertion thresholds. §1 is a mechanistic analogy, not a demonstration in this instrument.
- **The measured primary k=2 reversal was not attributed to AB-AC retroactive interference.**
  CAR_ROOT_CAUSE §RC-3 explicitly rules retroactive interference out as the cause of the primary
  k=2 reversal; the located causes are the load-time `curated→curriculum` provenance overwrite,
  phrase-node sanitization, and rank/truncation (Drop points A–D). Barnes & Underwood and Oren et
  al. are therefore hypotheses about what *would* make a high-interference item fail, not the
  established cause of this benchmark's dip. K2-C must be checked against the audit JSON before it
  is believed.
- **Retrieval practice is not unconditionally protective.** Reactivation opens a lability window
  in which later similar material can be incorporated (Hupbach et al 2007; Hardt et al 2010). The
  human sleep/interference literature is mixed: `BRAIN_RETRIEVAL_AND_REPLAY.md` §4 reports
  Deliens et al. 2013 (retroactive interference only in the sleep group), Ellenbogen et al.
  2006/2009 (protection) and Abel et al. 2023 ("still unclear"). Those specific records were not
  re-verified in this task and are cited here as *repo-reported, not independently checked*.
- **"A trusted-source tag alone sets confidence" is not established.** Fast mapping and
  propose-but-verify show one-trial retention under referential ambiguity; they do not show that
  source prestige or an `edge_kind` label is sufficient to write a high-confidence causal edge.
  Epistemic-trust/testimony data were not verified here `[unverified]`. K2-A is an engineering
  extrapolation and needs the k=0 control.
- **Schema congruence does not make arbitrary novel pairs intrinsically easy.** Tse et al. 2007
  shows rapid assimilation of schema-*consistent* pairs; it does not show that items without a
  schema are easier to restructure. The predicted direction for `bruise/impact/rupture` versus
  `hiccup/spasm/interruption` rests on pattern-separation and proactive-interference work
  (Bein et al 2020; Oren et al 2017; Anderson 1981), which was not run on those stimuli. Whether
  those particular tokens have "rich existing structure" in RAVANA's graph is an empirical
  question for the audit census, not a fact from the literature.
- **Elaboration is not the supported lever.** Craik & Lockhart 1972 supports deeper encoding, but
  Karpicke & Blunt 2011 found elaborative concept mapping inferior to retrieval practice on the
  later test. "More rehearsal" would be the wrong second rep.
- **One-trial evidence is mostly word/perceptual learning.** Whether a single trusted causal
  teaching sentence can become a durable, assertable two-hop graph route is untested. Liao &
  Losonczy 2024 is a review of plasticity mechanisms; Antony et al 2017 is a theoretical proposal.
- **Changing confidence is not literally threshold-neutral.** K2-A leaves `THETA_ASSERT` and
  `grade_chain` unchanged but alters their inputs. Until the k=0 control and known-bad probes are
  measured, it may be a disguised threshold change. It must not be reported as "no threshold
  change" without that evidence.
- **A gain from any of these fixes is not evidence for the brain mechanism.** Only the predicted
  *pattern* (failed-probe benefit, overlap-dependent failure, tag-dependent retrieval) would
  support the mapping; a flat improvement in score would not.
