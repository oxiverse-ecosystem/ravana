# How the brain specifies an information need, and what RAVANA is missing

**Status:** research basis + design recommendation. No code changed. Repository at
`6a2b5fa`; the measured failure is `[gap->web] status=no_chain sources=2 chain_hops=0-1`
with network on, and the query-reformulation-plus-retry A/B was reverted because
chain hops went 1 -> 0 and answers degraded.

**Thesis.** The defect is not query phrasing and not retrieval volume. RAVANA detects an
*anomaly* but never builds an *information need*. In the human literature these are
different states: a bare "gap" flag is the visceral level of awareness, whereas search is
driven by a typed, question-shaped representation with a known anchor, an expected answer
form, and a stopping rule. A system that cannot say *what kind of thing* would close the
gap cannot select a source that contains it, and will happily ingest dictionary prose
that is topically relevant but cognitively useless.

---

## Evidence grading used below

| Grade | Meaning |
|---|---|
| **A** | Peer-reviewed empirical article, experiments with behavioural and/or neural data. |
| **B** | Peer-reviewed theoretical/review article or canonical proposal; mechanism plausible, not a direct experimental demonstration in the exact domain. |
| **C** | Working paper, dissertation, conference abstract or secondary summary; flagged when used. |
| **CONTESTED** | Peer-reviewed but actively debated; both readings stated. |

Each numbered `[n]` is a retrieved page in the Sources block; the graded reference list
gives authors, year, journal, volume/pages and DOI where verified. Claims without a
numbered citation are either author-year references graded C in the reference list, or
design inferences explicitly labelled as such.

---

## 1. Metacognition: what is in a feeling-of-knowing that is not in a gap flag

Flavell's (1979, *American Psychologist* 34(10):906-911) framework separates
**metacognitive knowledge** (what I know about tasks and strategies) from
**metacognitive experiences** — the conscious feeling that accompanies an enterprise,
e.g. "the sudden feeling that you do not understand something another person just
said" [31] (**B**). A gap flag is one bit of a metacognitive experience with neither the
task model nor the strategy knowledge attached. Flavell's model has four interacting
classes: knowledge, experience, goals/tasks, and actions/strategies; RAVANA currently
carries only the experience.

Koriat's (1993, *Psychological Review* 100(4):609-639) accessibility model is the key
mechanistic result: "the computation of FOK is parasitic on the processes involved in
attempting to retrieve the target, relying on the accessibility of pertinent
information" [1] (**A**). There is no privileged monitor that reads a memory trace
directly. The FOK judgment is built from (i) *how much* partial information came to
mind, (ii) *how easily* it came, and (iii) *cue familiarity* (Reder 1987; Koriat &
Levy-Sadot's cascaded familiarity-then-accessibility account, both **A/B**). Because
correct and incorrect fragments both raise accessibility, FOK can be confidently
wrong [1]. A bare flag throws away all three cues and keeps only the failure outcome.

Two distinctions matter for this failure:

- **Lexical vs factual vs structural ignorance.** The tip-of-the-tongue state is a
  *temporary inaccessibility of a known word* with phonological/semantic fragments and
  a feeling of imminence (Brown & McNeill 1966; Schwartz & Metcalfe 2011, *Memory &
  Cognition* 39(5):737-749, **A**); FOK is the estimated likelihood of later
  *recognition*. Neither is the same as "I do not know the causal mechanism". RAVANA's
  retrieved dictionary definitions answer a **definition/lexical** need, not a
  **causal-structure** need — a category error the current flag cannot represent.
- **Recall vs insight.** Metcalfe (1986, *Journal of Experimental Psychology: Learning,
  Memory, and Cognition* 12:288-294) showed that FOK predicts later recognition but
  "the feeling-of-knowing rank ordering of insight problems did not relate to problem
  solution" [7] (**A**). Mechanism discovery is an insight-type problem: feeling that you
  do not know it does not tell you which missing link to look for. The system needs a
  *structural* description of the unsolved step, not a confidence value.

**Contested.** Koriat's inference/accessibility account is not universally accepted;
Metcalfe (2000, *Consciousness and Cognition* 9(2):149-166) argues that the rejection of
direct access is premature and distinguishes information-based from experience-based
judgments [CONTESTED]. Nothing below depends on which side is right: both agree FOK is
graduated, cue-bound and target-specific, which a boolean gap flag is not.

**What the FOK carries that the gap flag lacks:** target identity; gap *type*
(lexical / definitional / factual-recall / causal-structure / causal-strength);
an inventory of accessible partial cues (what *is* known); a calibrated magnitude; and
an implicit stopping rule ("I would know it if I saw X").

---

## 2. Information-gap theory: a gap is a specific question, not a generic ignorance

Loewenstein (1994, *Psychological Bulletin* 116(1):75-98) defines curiosity as "a form
of cognitively induced deprivation that arises from the perception of a gap in
knowledge or understanding" [3] (**B**), and distinguishes **specific curiosity** — the
desire for a particular piece of information, as in solving a puzzle — from diffuse
exploration. A gap opens when a *question becomes activated* and its answer is not known
with certainty; its intensity depends on the salience ofthe missing content and on the
estimated probability that a given piece of information will close it. This section is
the direct lineage of `docs/GAP_DRIVEN_WEB_LEARNING.md`: a perceived gap should *drive*
search. But the same theory says the drive is selective, not a generic "fetch anything
on-topic".

The formal question-answer rendering makes the requirement concrete: an information gap
is an activated question plus uncertainty over its answer set (Golman & Loewenstein,
"Information Gaps: A Theory of Preferences...", manuscript, **C**). That is, a gap is
closable exactly when the seeker can represent **(a)** the question, **(b)** the space
of admissible answers, and **(c)** a test for whether a candidate answer resolves it.
Vague curiosity fails all three: it has a topic but no question, no answer set and no
stopping rule. RAVANA's `why/how` + no-2-hop-chain condition supplies (a) weakly and
supplies neither (b) nor (c), which is why any two retrieved sources count as "sources"
and nothing can count as a satisfying answer.

Belkin (1980, *Canadian Journal of Information Science* 5:133-143) arrives at the same
place from IR: an information need arises from a recognised **anomalous state of
knowledge (ASK)**, and the paper is explicitly framed around "the problem of
nonspecifiability of information need" [13] (**B**). Anomalies are of different kinds —
"gaps or lacks, uncertainty, or incoherence" — and Belkin, Oddy & Brooks (1982,
*Journal of Documentation* 38(2):61-71; 38(3):145-164, **B**) argue that the *type* of
anomaly must determine the retrieval strategy. Applying one strategy (generic template
query + `causes`-only OpenIE extraction) to a mechanism-type anomaly is precisely the
mismatch this literature predicts.

Taylor (1968, *College & Research Libraries* 29(3):178-194) supplies the developmental
ladder: seekers pass through "the visceral, conscious, formalized, and compromised
needs" [15] (**A/B**) — an unexpressed unease, a conscious description of an ill-defined
area of indecision, a qualified rational statement, and finally the question posed to
the system. His five negotiation filters are: subject, motivation/purpose, personal
characteristics, relation to the file organisation, and **anticipated answers**. RAVANA
jumps from visceral (weak response + `why`) straight to compromised (a template query),
skipping the formalized statement and the anticipated-answer filter. That is the level
collapse the reverted retry could not fix: it rephrased the compromised question, it did
not formalize the need.

---

## 3. Curiosity, epistemic value and encoding: what selects the next observation

Friston's active-inference formulation decomposes expected free energy into extrinsic
value and epistemic value; agents "maximize simultaneously 'extrinsic value' (e.g.,
expected reward) and 'epistemic value' (e.g., information gain or the resolution of
uncertainty implicit in exploration or curiosity)" [5] (**B**) (Friston, Rigoli,
Ognibene, Mathys, Fitzgerald & Pezzulo 2015, *Cognitive Neuroscience* 6(4):187-214;
Pezzulo, Rigoli & Friston 2016, *Learning & Memory* 23(7):322-338, DOI
10.1101/lm.041780.116). The operational consequence for *selection* is that the
observation chosen is the one with the largest expected reduction of uncertainty about
hidden states. A source that restates what is already known has zero epistemic value;
so does a source whose content cannot touch the hidden variable. A Wikipedia navigation
line and a dictionary definition both fail this test for a mechanism question even
though both are topically on-string.

Kang, Hsu, Krajbich, Loewenstein, McClure, Wang & Camerer (2009, *Psychological Science*
20(8):963-973) found that "curiosity arises from an incongruity, or information
gap—a discrepancy between what one knows and what one wants to know" [9] (**A**), that
curiosity follows an inverted-U in confidence (maximum at intermediate confidence),
that participants spent tokens or waiting time to obtain the answer, and that
high-curiosity answers were recalled better 1-2 weeks later. The design implications
are: (i) search should be triggered by a *calibrated intermediate* confidence, not by a
binary failure; (ii) the decision to spend a read is made from the FOK magnitude; (iii)
"what would close this gap" must be represented *before* the read, because the
curiosity state alters encoding of what follows. Boundary conditions on the inverted-U
(importance of the information, task demands) are still being worked out — Dubey &
Griffiths (2020, *Psychological Review* 127(3):455-476) and later replications support
the general shape but qualify it [**CONTESTED** on exact form, not on selectivity].

Gruber, Gelman & Ranganath (2014, *Neuron* 84(2):486-496) show that "activity in the
midbrain and the nucleus accumbens was enhanced during states of high curiosity" [10]
and that anticipatory midbrain-hippocampal activity and connectivity predict later
memory, including for incidental material encountered in the same window. This is
evidence that the *pre-read* state is functional — it is not decoration to write an
expected-answer field into the gap object. It is **not** evidence about URL selection:
the paper studies trivia-question anticipation, so any claim that it directly picks
documents would be an over-extension (design inference, not cited fact).

---

## 4. Turning a felt gap into a query: four levels and two mappings

Query formulation is not a paraphrase problem. Marchionini's formulation analysis (as
summarised by Riehm & Vakkari, *Information Processing & Management* 2006, **B**)
requires two mappings: a **semantic mapping** from the vocabulary of the problem to the
system's vocabulary, and an **action mapping** to operations the search system
recognises. RAVANA performed only a surface paraphrase of the question string.

- **Belkin** [13] (**B**): represent the ASK, do not demand a fully specified query;
  choose the strategy from the anomaly type.
- **Taylor** [15] (**A/B**): negotiate visceral -> conscious -> formalized before the
  compromised query; explicitly use "anticipated answers" as a filter.
- **Ingwersen** (1996, *Journal of Documentation* 52(1):3-50; 1992 *Information
  Retrieval Interaction*, Taylor Graham) states that polyrepresentation should encode
  the information need, problem state and domain task "in a structure of causality"
  [17] (**B**). Operationally: represent the need with *several cognitively different*
  signals (subject, relation, missing hop, expected answer form, genre constraint)
  rather than one template string; cognitive overlap across representations reduces
  retrieval uncertainty.
- **Wilson** (1997, *Information Processing & Management* 33(4):551-572; 1981,
  *Journal of Documentation* 37(1):4-15, **B**) treats information seeking as
  goal-driven and inserts **intervening variables** — including *source
  characteristics* — between the need and the search act. Source genre is therefore
  part of the search decision, not an afterthought. Query reformulation, in the
  interactive-IR literature (Ingwersen 1996; Belkin's episode model; Saracevic's
  stratified model), is driven by the cognitive/situational state of the searcher, not
  by lexical substitution. The measured failure of RAVANA's one retry is consistent
  with this and is **not** evidence against reformulation in general.

---

## 5. Source selection: how a human can reject a document before reading it

Saracevic's relevance framework (1996, "Relevance reconsidered '96", CoLIS; 2007
review, *JASIST*, **B**) distinguishes system/algorithmic, **topical**, **cognitive
(pertinence)**, situational/utility and motivational relevance. Cognitive relevance is
the relation between the user's *state of knowledge / cognitive need* and the object,
judged by informativeness, novelty and information quality. RAVANA's two sources were
almost certainly **topically** relevant (they named the subject) and **cognitively
irrelevant** (they added no new causal structure). Topical match cannot be the selection
criterion for a mechanism gap.

Source monitoring adds the pre-reading mechanism. Johnson, Hashtroudi & Lindsay (1993,
*Psychological Bulletin* 114(1):3-28) define "source monitoring" as "the set of
processes involved in making attributions about the origins of memories, knowledge, and
beliefs" [20] (**A**) and show that people do **not** retrieve an abstract source tag:
source is *inferred* from qualities of experience — semantic detail, perceptual and
contextual information, affect, and cognitive operations — by decision processes with
flexible criteria. Familiarity alone is insufficient for a source attribution. Translated
to documents: a hit's lexical overlap with the subject is familiarity; its *structure*
(ordered event clauses, causal connectives, process verbs) is the cue that it can carry
a causal chain. The current pipeline uses only `hit.title + hit.content` as text and
never evaluates this cue.

Causal cognition constrains the expected answer shape. Griffiths & Tenenbaum (2005,
*Cognitive Psychology* 51(4):334-384) formalize "the distinction between causal
structure and causal strength: The difference between asking whether a causal
relationship exists and asking how strong that causal relationship might be"
[21] (**A**); their causal-support model handles the *structure* question, which is what
a "why" gap asks for. Gerstenberg, Goodman, Lagnado & Tenenbaum (2015, "How, whether,
why: Causal judgments as counterfactual contrasts", *Proc. 37th CogSci*, **A/B**) show
that these are genuinely different judgments; Gerstenberg (2024, *Trends in Cognitive
Sciences* 28(10):924-936, **B**) argues counterfactual simulation requires a generative
model, interventions on it, and simulation of consequences. Sloman & Lagnado (2015,
*Annual Review of Psychology* 66:223-247, **B**) review how prior domain knowledge and
expertise filter causal inference — so prior graph knowledge legitimately gates which
candidates are worth reading. A definition or a bare association answers "what is X" or
"how strong", not "through what intermediate event does X produce Y".

Finally, text comprehension itself is causal-chain construction. Trabasso & van den
Broek (1985, *Journal of Memory and Language* 24(5):612-630) found that "whether or not
an event was in a causal chain and the number of its causal connections were both found
to account for substantial proportions of common and unique variance" in recall,
summarisation and importance [22] (**A**); Trabasso & Sperry (1985, same volume,
595-611, **A**) showed the same for judged importance, using necessity/counterfactual
tests. Fletcher et al.'s on-line studies (**A**) show readers hold the likeliest causal
antecedent in working memory and search for a missing antecedent when that fails. This
is the cognitive reason a 2-hop chain is the right closure criterion *and* the reason
the source must be process prose: chain-bearing prose is the only genre whose surface
form reliably carries the structure the reader is trying to build.

---

## 6. Observable criteria: "RAVANA now has a well-formed information need"

All five are checkable **before** a search is issued, from state RAVANA already has
(`ctx.subject`, `ctx.raw_input`, graph nodes/edges, `fok_confidence`, the gap registry):

1. **Typed gap.** A gap object exists with non-empty `subject_node`, `question_type`
   (`why`/`how`) and `gap_type`. If `gap_type` is `lexical` or `definition`, the need is
   not a mechanism need: route to the KB/definition path and **do not** open a web read.
2. **Bounded missing hop.** Exactly one structural link is unresolved and its role is
   named (`cause`, `effect`, or `intermediate`) with a `from`/`to` anchor where at least
   one endpoint is a resolved graph node. Zero anchors = topic sweep, not an information
   need; more than one unresolved link = decompose first.
3. **Expected answer shape + closure test declared.** Before searching, the system can
   state what a satisfying snippet would contain — an ordered chain of >=2 event
   clauses joined by causal connectives, each hop attributable to a source — and can
   test a candidate snippet against that shape.
4. **Calibrated FOK in the actionable band.** A numeric confidence (existing
   `fok_confidence(support_count, retrieval_succeeded, base)` from
   `chat/metacognition.py`) is neither ~0 (nothing to anchor) nor ~1 (already known),
   matching the intermediate-confidence window in which curiosity and information
   search peak [9]. A flat gap flag cannot express this.
5. **Source-admissibility predicate declared.** The search request carries a genre
   constraint (process/mechanism prose required) and a veto list (dictionary entry,
   glossary, navigation/chrome, "jump to", reference list). This is the action mapping
   and the "anticipated answers" filter made explicit before the query is compiled.

If any of 1-5 fails, the correct behaviour is to **not search** and to refine the need,
not to spend one of the four session reads.

---

## 7. Concrete recommendation (design only — no code changed)

### 7.1 The structured object the gap should carry

Replace the current `(subject: str, raw_input: str)` call to `close_mechanism_gap` with
one typed object, assembled from existing signals at the gap-detection site in
`engine.py` (where `_mechanism_gap` is already computed):

```
InformationNeed {
    subject_node:      graph_node_id,        # RESOLVED node, not lowercased string
    question_type:     "why" | "how",
    gap_type:          "lexical" | "definition" | "causal_structure"
                     | "causal_strength",
    relation_type:     "causal" | "enable" | "inhibit",   # typed, not just "causes"
    known_anchor:      graph_node_id,        # observed effect OR known cause
    missing_hop:       { from: node_id | "unknown",
                         to:   node_id | "unknown",
                         role: "cause" | "effect" | "intermediate" },
    missing_hops:      1,                    # bounded; decompose if >1
    expected_answer_shape: "ordered chain >= 2 causal clauses between anchors",
    closure_test:      "chain_hops >= 2 AND both endpoints resolve to graph nodes
                        AND every hop has provenance",
    fok:               float in (0,1),       # from metacognition.fok_confidence
    source_genre:      "process_prose_required",
    query_constraints: { include: [subject lemmas, relation cue words,
                                   "mechanism", "process", "how ... works"],
                         exclude: ["definition", "dictionary", "glossary",
                                   "wikipedia navigation"] },
    budget:            { max_sources: 2, max_reads_per_session: 4 },
}
```

Why each field exists, in one line: `gap_type` stops definition needs from reaching the
web; `relation_type` widens extraction beyond the single `causes` verb; `known_anchor`
and `missing_hop` turn "I don't know why X" into "I need the link between A and B";
`expected_answer_shape` and `closure_test` restore the answer set that gap theory says
a closable gap must have [3][13]; `fok` supplies the calibrateable control signal [9];
`source_genre` and `query_constraints` implement Taylor's anticipated-answer filter and
Ingwersen's polyrepresentation [15][17]. Store the object in the existing provenance
record alongside `gap_subject` and `gap_query`, so a later session can audit what the
system *thought it was looking for*.

### 7.2 Filtering candidates to process prose

1. **Compile the query from `InformationNeed`, not from `ctx.raw_input`.** Add the
   relation/genre constraints; do not merely reword the original question (the
   reverted retry shows that is the wrong lever).
2. **Screen each hit before `openie.extract` / `learn_text`.** Reject on *positive
   evidence of incompatibility* (veto-only, matching source-monitoring practice [20]):
   title or snippet is a definition/glossary entry; high ratio of navigation/list
   markers ("jump to", "contents", "references"); no causal connective (`because`,
   `leads to`, `results in`, `due to`, `drives`, `triggers`, `via`, `through`); fewer
   than two distinct event clauses; no token overlapping `known_anchor` or the known
   side of `missing_hop`. Admit everything else — do not require positive proof of
   "process prose", because surface markers of genre are unreliable [20].
3. **Extract typed relations, not only `relation == "causes"`.** The object's
   `relation_type` names what to accept (`enable`, `inhibit`, `produce`, `drive` are
   mechanism relations too), then normalise to a `causal` graph edge with provenance.
4. **Keep the existing >=2-hop closure gate, but split the status.** If no candidate
   survives screening, return `no_eligible_source` (genre/genre-filter problem); if
   candidates survive but no 2-hop chain forms, keep `no_chain` (world-knowledge
   problem). Today both look like `no_chain`, which is why the failure was
   misdiagnosed as a phrasing problem.
5. **Re-check `gap_type` before consuming the result.** A source that closes a
   `causal_structure` need must yield ordered hops, not a stronger association —
   Griffiths & Tenenbaum's structure/strength distinction [21].

### 7.3 Falsifiable predictions / how to verify before enabling

- **P1:** with the query and budget held constant, adding the step-2 veto increases
  `chain_hops` and the `learned` rate, and lowers `no_chain` + `no_eligible_source`;
  measure on the same subject set with network on.
- **P2:** adding `missing_hop` + genre constraint to the query raises the fraction of
  hits containing >=1 causal connective, compared with the current
  `_GAP_CAUSAL_QUERY` template.
- **P3:** routing `gap_type in {lexical, definition}` to the KB rather than the web
  removes `sources=2, chain_hops<=1` cases without lowering answer quality.
- **P4:** searches fired at intermediate calibrated `fok` close more gaps than
  searches fired at the extremes; report both closure rate and answer-quality delta.
- **P5:** the status split changes the diagnosis on the measured case: if the
  dictionary/Wikipedia hits are vetoed, the status becomes `no_eligible_source`, not
  `no_chain`.

Report both the positive metric (chain_hops >= 2, learned) **and** the negative one
(answer quality), and re-run the unchanged baseline before each A/B — the earlier retry
showed a hop-count gain can coexist with worse answers, and a stochastic generator can
manufacture a one-sample win.

---

## References (graded; full bibliographic detail)

**A — peer-reviewed empirical**
- Koriat, A. (1993). How do we know that we know? The accessibility model of the feeling of knowing. *Psychological Review, 100*(4), 609-639. DOI 10.1037/0033-295X.100.4.609.
- Metcalfe, J. (1986). Feeling of knowing in memory and problem solving. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 12*, 288-294.
- Metcalfe, J., Schwartz, B. L., et al. (1993). The cue-familiarity heuristic in metacognition. *Journal of Experimental Psychology: Learning, Memory, and Cognition*.
- Kang, M. J., Hsu, M., Krajbich, I. M., Loewenstein, G., McClure, S. M., Wang, J. T., & Camerer, C. F. (2009). The wick in the candle of learning: Epistemic curiosity activates reward circuitry and enhances memory. *Psychological Science, 20*(8), 963-973. DOI 10.1111/j.1467-9280.2009.02402.x.
- Gruber, M. J., Gelman, B. D., & Ranganath, C. (2014). States of curiosity modulate hippocampus-dependent learning via the dopaminergic circuit. *Neuron, 84*(2), 486-496. DOI 10.1016/j.neuron.2014.08.060.
- Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring. *Psychological Bulletin, 114*(1), 3-28. DOI 10.1037/0033-2909.114.1.3.
- Griffiths, T. L., & Tenenbaum, J. B. (2005). Structure and strength in causal induction. *Cognitive Psychology, 51*(4), 334-384. DOI 10.1016/j.cogpsych.2005.05.004.
- Trabasso, T., & van den Broek, P. (1985). Causal thinking and the representation of narrative events. *Journal of Memory and Language, 24*(5), 612-630. DOI 10.1016/0749-596X(85)90049-X.
- Trabasso, T., & Sperry, L. L. (1985). Causal relatedness and importance of story events. *Journal of Memory and Language, 24*(5), 595-611.
- Brown, R., & McNeill, D. (1966). The "tip of the tongue" phenomenon. *Journal of Verbal Learning and Verbal Behavior, 5*, 325-337. (As cited by the TOT literature; primary not re-extracted this pass.)
- Schwartz, B. L., & Metcalfe, J. (2011). Tip-of-the-tongue (TOT) states: Retrieval, behavior, and experience. *Memory & Cognition, 39*(5), 737-749. DOI 10.3758/s13421-010-0066-8.

**B — peer-reviewed theoretical / review / canonical**
- Flavell, J. H. (1979). Metacognition and cognitive monitoring: A new area of cognitive-developmental inquiry. *American Psychologist, 34*(10), 906-911. DOI 10.1037/0003-066X.34.10.906.
- Loewenstein, G. (1994). The psychology of curiosity: A review and reinterpretation. *Psychological Bulletin, 116*(1), 75-98. DOI 10.1037/0033-2909.116.1.75.
- Friston, K., Rigoli, F., Ognibene, D., Mathys, C., Fitzgerald, T., & Pezzulo, G. (2015). Active inference and epistemic value. *Cognitive Neuroscience, 6*(4), 187-214. DOI 10.1080/17588928.2015.1020053.
- Pezzulo, G., Rigoli, F., & Friston, K. J. (2016). Active inference, epistemic value, and vicarious trial and error. *Learning & Memory, 23*(7), 322-338. DOI 10.1101/lm.041780.116.
- Belkin, N. J. (1980). Anomalous states of knowledge as a basis for information retrieval. *Canadian Journal of Information Science, 5*, 133-143.
- Belkin, N. J., Oddy, R. N., & Brooks, H. M. (1982). ASK for information retrieval: Parts I and II. *Journal of Documentation, 38*(2), 61-71; 38*(3), 145-164.
- Taylor, R. S. (1968). Question-negotiation and information seeking in libraries. *College & Research Libraries, 29*(3), 178-194.
- Ingwersen, P. (1992). *Information Retrieval Interaction*. London: Taylor Graham.
- Ingwersen, P. (1996). Cognitive perspectives of information retrieval interaction: Elements of a cognitive IR theory. *Journal of Documentation, 52*(1), 3-50. DOI 10.1108/eb026960.
- Wilson, T. D. (1981). On user studies and information needs. *Journal of Documentation, 37*(1), 4-15.
- Wilson, T. D. (1997). Information behaviour: An interdisciplinary perspective. *Information Processing & Management, 33*(4), 551-572. DOI 10.1016/S0306-4573(97)00028-9.
- Saracevic, T. (1996). Relevance reconsidered '96. In *Information science: Integration in perspective* (CoLIS 2), 201-218.
- Saracevic, T. (2007). Relevance: A review of the literature and a framework for thinking on the notion in information science. Part II. *Journal of the American Society for Information Science and Technology*.
- Sloman, S. A., & Lagnado, D. (2015). Causality in thought. *Annual Review of Psychology, 66*, 223-247. DOI 10.1146/annurev-psych-010814-015135.
- Gerstenberg, T. (2024). Counterfactual simulation in causal cognition. *Trends in Cognitive Sciences, 28*(10), 924-936. DOI 10.1016/j.tics.2024.04.012.
- Riehm, U., & Vakkari, P. (2006). Query formulation and reformulation in information retrieval interaction. *Information Processing & Management* (IPM 2006 proceedings).

**C — working paper / abstract-only / secondary (labelled, not load-bearing)**
- Golman, R., & Loewenstein, G. (2015). *Curiosity, information gaps, and the utility of knowledge* (manuscript). Carnegie Mellon University.
- Gerstenberg, T., Goodman, N. D., Lagnado, D. A., & Tenenbaum, J. B. (2015). How, whether, why: Causal judgments as counterfactual contrasts. *Proc. 37th Annual Meeting of the Cognitive Science Society*, 782-787.
- Metcalfe, J. (2000). Feelings and judgments of knowing: Is there a special noetic state? *Consciousness and Cognition, 9*(2), 149-166. [CONTESTED: direct access vs inference]

**Design inferences in this document (not claims from the literature):** the five
observable criteria, the `InformationNeed` field set, the veto-only source filter, the
`no_eligible_source` status split, and predictions P1-P5. Each is derived from the cited
mechanisms and must be measured in RAVANA before being treated as effective.

## Sources

[1] https://newiipdm.haifa.ac.il/wp-content/uploads/2015/05/1993-Koriat-PR-FOK.pdf
    > "The model postulates that the computation of FOK is parasitic on the processes involved in attempting to retrieve the target, relying on the accessibility of pertinent information."
[3] https://www.cmu.edu/dietrich/sds/docs/loewenstein/PsychofCuriosity.pdf
    > "The new account interprets curiosity as a form of cognitively induced deprivation that arises from the perception of a gap in knowledge or understanding."
[5] https://pmc.ncbi.nlm.nih.gov/articles/PMC4918783
    > "tive Inference ( [Friston et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4918783/#PEZZULOLM041780C28)), in which agents maximize simultaneously “extrinsic value” (e.g., expected reward) and “epistemic value” (e.g., information gain or the resolution of uncertainty implicit in exploration or curiosity).[4](https://pmc.ncbi.nlm.nih.gov/articles/PMC4918783/#FN1) In this paper, we use this Active Inference scheme to consider an important empirical phenomenon underlying VTE behavior; namely, t"
[7] https://columbia.edu/cu/psychology/metcalfe/PDFs/Metcalfe%201986%20Feeling.pdf
    > "In contrast to the positive correlation found in the memory task, the feeling-of-knowing rank ordering of insight problems did not relate to problem solution."
[9] https://neuroecon.berkeley.edu/public/papers/Psychol%20Sci%202009%20Kang.pdf
    > "A theory guiding our research holds that curiosity arises from an incongruity, or information gap—a discrepancy between what one knows and what one wants to know (Loewenstein, 1994)."
[10] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4252494
    > "FMRI results revealed that activity in the midbrain and the nucleus accumbens was enhanced during states of high curiosity."
[13] https://wp.comminfo.rutgers.edu/belkin/wp-content/uploads/sites/75/2015/03/Belkin_ASK_CJIS.pdf
    > "This paper presents an alternative view of information retrieval, based on the problem of nonspecifiability of information need and within a context known as the cognitive viewpoint."
[15] https://crl.acrl.org/index.php/crl/article/viewFile/12027/13473
    > "When they go through librarians they must develop their questions through four levels of need, referred to here as the visceral, conscious, formalized, and compromised needs."
[17] https://vlex.co.uk/vid/cognitive-perspectives-of-information-846710993
    > "The concept seeks to represent the current user's information need, problem state, and domain work task or interest in a structure of causality."
[20] https://memlab.yale.edu/sites/default/files/files/1993_Johnson_Hashtroudi_Lindsay_PsychBull.pdf
    > "Source monitoring refers to the set of processes involved in making attributions about the origins of memories, knowledge, and beliefs (Hashtroudi, Johnson, & Chrosniak, 1989; Johnson, 1988a, 1988b; Lindsay & Johnson, 1987; Lindsay, Johnson, & Kwon, 1991)."
[21] https://collaborate.princeton.edu/en/publications/structure-and-strength-in-causal-induction
    > "This framework makes precise the distinction between causal structure and causal strength: The difference between asking whether a causal relationship exists and asking how strong that causal relationship might be."
[22] https://cs.uky.edu/~sgware/reading/papers/trabassovandenbroek1985causal.pdf
    > "Whether or not an event was in a causal chain and the number of its causal connections were both found to account for substantial proportions of common and unique variance in all four measures."
[31] https://jwilson.coe.uga.edu/EMAT7050/Students/Wilson/Flavell%20%281979%29.pdf — Flavell 1979
    > "A Model of Cognitive Monitoring I believe that the monitoring of a wide variety of cognitive enterprises occurs through the actions of and interactions among four classes of phenomena: (a) metacognitive knowledge, (b) metacognitive experiences, (c) goals (or tasks), and (d) actions (or strategies)."

---

## MEASURED NEGATIVE: typed need + source veto did not close the gap

The recommendation above was implemented at `close_mechanism_gap`: a typed
`InformationNeed` (direction, gap type, anchor, missing hop, closure test),
a widened candidate pool of 8 with the QUERY unchanged, and a veto-only
admissibility predicate rejecting definition/glossary/navigation snippets
and snippets with no process connective. Rejected URLs never reached
`openie.extract` or provenance stamping. 12 unit tests passed.

A/B with IntentForge on, identical state, no `eng.save()`:

```
iron rust       candidates=8 admitted=2 causal_hops=0 chain_hops=0
ocean tides     candidates=8 admitted=2 causal_hops=0 chain_hops=0
answers: unchanged definitions; no improvement
```

The filter changed *which* sources were read but did not produce a single
causal hop. The weaker undirected proxy also fell from 1 to 0 on tides. The
code was **REVERTED**; no capability gain, so no code shipped.

**What this establishes:** an information need can be well-formed and a
source can look like process prose, yet the retrieval surface still does not
contain two connected, extractable causal relations for these subjects. The
bottleneck is the available corpus and extraction, not the specification of
the need. Next work must change the INPUT: authored curriculum chains with
provenance, or full-page extraction instead of search snippets. Do not
re-attempt query phrasing or veto lists alone.
