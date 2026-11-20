# Brain mechanisms of retrieval failure and replay failure

**Status:** decision-ready research note. **Owner:** this file only. **Scope:** explains two
measured RAVANA-LEARN-5 failures with peer-reviewed mechanisms, states observable
acceptance criteria with negative controls, and names the narrowest mechanism change for
each. No code and no other document is modified here.

**Evidence base (measured, not hypothetical):**

| Failure | Measured fact | Source |
|---|---|---|
| F1 CAR | Teaching creates provenance-trusted >=2-hop chains that survive a cold reload: primary k=8 `Δ_cold = +0.80` (8/10 vs 0/10, Fisher p=0.00071). But on the **held-out paraphrase** the emitted answer surfaces a chain in only 0.00-0.10 of arms; on the **identical-surface** probe 5/10 primary k=8 answers echo the taught mechanism words. | `docs/LEARN5_CAR_MEASURED.md` (headline table; "Answer reach"; Limitations 1) |
| F2 CLI | Real retroactive interference: A-only retention 0.900 (PRIMARY) / 0.833 (SEALED) vs sequential A->B 0.700 / 0.833. Interleaving protects: concurrent A+B 0.875 / 1.000. Sleep replay on-off = **0.000** on both pools at matched compute (wall clock 0.313/0.309 s and 0.240/0.243 s). Replay-strengthening control does move an edge 0.20 -> 0.22, but the trusted-chain verdict is a binary provenance-and-typing gate, so the bump cannot cross it. | `docs/LEARN5_CLI_MEASURED.md` (section 3 table; section 4) |

**One sentence each failure:** F1 is a *cue-to-trace addressing* failure — the chain exists but
no cue in the paraphrase question addresses it. F2 is an *encoding-overlap* failure that is
being treated as a *consolidation-strength* failure — interleaving removes the overlap at
encoding, while replay only re-strengthens whatever was already encoded.

---

## 1. Encoding specificity and transfer-appropriate processing (F1)

**Sources.** Tulving & Thomson (1973, *Psychological Review*) defined the encoding
specificity principle: what is stored is determined by what was perceived and how it was
encoded, and *what is stored determines which retrieval cues are effective*. Their
recognition-failure data showed that a word can be reliably recalled by a cue yet fail to be
recognized when the cue is changed — memory is not a stored object with fixed accessibility,
it is a cue-dependent operation. Morris, Bransford & Franks (1977, *Journal of Verbal
Learning and Verbal Behavior*) added transfer-appropriate processing: semantic encoding beat
rhyme encoding on a standard recognition test, but rhyme encoding beat semantic encoding on
a rhyming test. "Depth" is not the cause; *congruence between encoding operations and
retrieval operations* is.

**Why storing in one format does not make it retrievable by a differently phrased question.**
The CAR teaching channel encodes the chain together with the teaching utterance's surface
form: the specific words, ordering, and conversational frame in which `a -> b -> x` was
asserted. The held-out paraphrase preserves the proposition but changes the surface tokens
and often the interrogative frame (`why does X happen?` becomes `what is the mechanism
behind X?`). Under encoding specificity the changed question is a *different cue*, and
there is no principle requiring it to overlap the encoded trace. Under TAP the paraphrase
also demands a different operation: it asks for a relational/inferential report, whereas
teaching trained a surface-bound assertion. The graph-level metric correctly measures that
the trace exists; it cannot measure that the trace is addressable from the new cue. A cold
reload that preserves `edge_kind` provenance is exactly what the theory predicts: storage
survives, access does not.

**Retrieval cues that must be present at answer time (all three, conjunctive):**

1. **Encoding-overlap cue.** At least one feature of the question must be a feature that was
   actually encoded with the chain — for CAR, a *relation* (cause/enables/produces) or the
   participant nodes `a`, `b`, `x`, not merely the topic label of the probe.
2. **Diagnostic / low-fan-out cue.** The cue must single this chain out from candidates.
   CAR's cold graph carries hundreds of untrusted chains per subject (631/640 rejected for
   primary k=8), so a bare subject node such as `amber` or `lightning` is massively
   overloaded and functionally useless as an address.
3. **Operation-match cue.** The retrieval operation requested by the question must match an
   operation practiced at encoding. If teaching never required "given the relation type,
   emit the chain," a question that requires exactly that is out of transfer.

None of the three is supplied by "make the edge weight stronger." Strength changes
*availability given a cue*; it does not create a cue or change its fan-out.

---

## 2. Tip-of-the-tongue and cue overload (F1)

**Sources.** Brown & McNeill (1966, *Journal of Verbal Learning and Verbal Behavior*) showed
that in the TOT state participants know the target's first letter, syllable count, stress
pattern, and similar-sounding/similar-meaning neighbours, yet cannot produce the word:
partial, veridical knowledge coexists with failed production. Roediger (1973, *Journal of
Verbal Learning and Verbal Behavior*, "Inhibition in recall from cueing with recall targets")
gave the two-factor account: cues *help* when they open access to a higher-order unit that
unaided recall could not reach, but *hurt* when they force retrieval of more information
than is needed to gain that access. Roediger & Neely (1982, *Canadian Journal of Psychology*,
"Retrieval blocks in episodic and semantic memory") extended retrieval blocking to the A-B /
A-D interference paradigm and to the TOT situation. Watkins (1975, *Journal of Verbal
Learning and Verbal Behavior*, "Inhibition in recall with extralist 'cues'") and Watkins &
Watkins (1975, *Journal of Experimental Psychology: Human Learning and Memory*) stated the
cue-overload principle: a cue's effectiveness falls as more traces are subsumed under it.

> **Citation note (do not propagate the error).** The requested attribution "Roediger and
> Debuse" could not be verified as a TOT or cue-overload source. The nearest peer-reviewed
> matches are the three above. Curran, DeBuse, Woroch & Hirshman (2006, *Journal of
> Neuroscience*) is a real "DeBuse" paper but concerns familiarity vs recollection, not TOT
> or overload. Cite Roediger (1973) / Roediger & Neely (1982) / Watkins (1975) instead.

**Why knowing is not producing in conversation.** CAR's headline endpoint is a *feeling of
knowing* implemented as a structured verdict: `grade_chain(...).verdict == 'assert' AND
assertable_chain(...)` on the `ExplanationChain` object. That is the machine analogue of
Brown & McNeill's generic recall — the system can partially specify the answer (it "knows"
the first letters `a`, `b`, the relation shape, the trusted provenance) without being able
to output it. Production requires a further step: selecting one candidate chain from the
candidate set and committing it to the generation context. With an overloaded cue the
selection step fails even though the trace is intact. The 0.50 identical-surface echo is
the complementary error: when the exact teaching surface *is* supplied, a different,
surface-matched sentence is emitted — production is driven by whichever surface cue won,
not by the trusted chain. Both observations are the same TOT architecture seen from two
sides: access and selection are distinct from storage.

**What makes a cue diagnostic (testable definition).** A cue `q` is diagnostic for chain `c`
iff (i) `q` is a feature actually bound to `c` at encoding, and (ii) the number of candidate
chains in the active candidate set carrying `q` is small — ideally one. Formally, diagnostic
power tracks `P(c | q)`, not `P(c)`. A topic word shared by hundreds of chains has high
`P(q | c)` and low `P(c | q)`: it is a good *description* and a bad *address*. Concretely for
CAR: the ordered relation signature `CAUSE(CAUSE(a,b), (b,x))` plus the node identities is
diagnostic; the bare subject string is not. This is why the retrieval side, not the strength
side, is the lever.

---

## 3. Schema instantiation and analogical retrieval (F1)

**Sources.** Gentner (1983, *Cognitive Science*, "Structure-mapping: a theoretical framework
for analogy") distinguishes object attributes from relations and from higher-order relations;
analogy maps relational structure, and systematicity — connected higher-order relations —
determines what is mapped. Gentner, Rattermann & Forbus (1993, *Cognitive Psychology*),
Holyoak & Koh (1987) and Ross (1989) established the asymmetry: *similarity-based retrieval
is dominated by surface similarity even when structural similarity is what mapping needs* —
people retrieve surface-similar cases rather than the relationally similar case that would
support transfer. Gentner, Loewenstein & Thompson (2003, *Journal of Educational Psychology*)
showed the remedy: **analogical encoding** — explicitly comparing two cases that instantiate
the same schema — abstracts the common relational structure and raises later transfer.

**Why surface overlap dominates unless a relational cue is explicit.** Retrieval and mapping
are different mechanisms. Retrieval is a content-addressable, surface-driven bounce; mapping
is structural and can operate once both analogs are in working memory. A held-out paraphrase
removes the surface features that would have driven the bounce, and it does not supply an
explicit relational cue that could replace them. The trusted chain is therefore *inert
knowledge*: available, structurally correct, spontaneously unretrieved. This predicts the
exact CAR number — at most 0.10 answer reach on paraphrase while graph-level chain presence
is +0.80. It also predicts that the 0.50 exact-surface echo will *not* generalize: surface
cues produce parroting, not schema use.

**Schema instantiation requirement.** Before a chain can be emitted on a novel phrasing, the
relation pattern must have been instantiated as a schema across at least two surface forms,
with the invariant explicitly marked. A single teaching surface leaves the chain indexed by
that surface; two compared surfaces force the common relational skeleton to become the index.
There is a useful bridge to F2: the same property that makes relational cues diagnostic also
makes them resistant to interference — a chain keyed by its relation signature does not
collide with a new chain that merely shares a node.

---

## 4. Sleep replay, sharp-wave ripples, and retroactive interference (F2)

**Sources.** Wilson & McNaughton (1994, *Science*) recorded hippocampal place-cell ensembles
and found that cells co-active during behaviour increased their co-firing in subsequent
slow-wave sleep, strongest during sharp-wave/ripple bursts — the first direct evidence that
experience is re-expressed offline. Diekelmann & Born (2010, *Nature Reviews Neuroscience*)
synthesise the two functions: SWS slow oscillations, spindles and ripples coordinate
hippocampo-neocortical **system consolidation** (redistribution, abstraction, generalisation)
plus synaptic consolidation; global synaptic downscaling preserves encoding capacity.
Wixted (2004, *Annual Review of Psychology*) places RI in this frame: a fresh trace is labile
and new learning disrupts its stabilisation; an A-B / A-C design measures that disruption.
McCloskey & Cohen (1989, in *Psychology of Learning and Motivation*, vol. 24) showed the
canonical pattern: catastrophic interference under **sequential** A->B training, largely
absent under **concurrent** A+B training — precisely RAVANA's 0.700 vs 0.875 result.

**Why replay alone may not fix interference.**

1. **Different level of description.** Ripple replay re-activates and redistributes the
   representation that exists; it does not dissolve a competing binding. In CLI the loss is
   not weak A weights, it is a lost or non-selectable trusted chain after B introduced
   competing routes on shared/generic nodes. Replaying A with the same cue that B now
   dominates re-strengthens the losing route, not the distinction.
2. **Replay can reopen lability.** Reactivation returns a consolidated memory to a labile
   state. Deliens, Schmitz, Caudron, Mary, Leproult & Peigneux (2013, *PLoS ONE*) found
   retroactive interference **only in the sleep group**, interpreted as reconsolidation:
   reactivating A while learning a related AC set made A sensitive again. So replay is not
   monotonically protective; its sign depends on what is co-active.
3. **The human literature is explicitly mixed.** Ellenbogen, Hulbert, Stickgold, Dinges &
   Thompson-Schill (2006, *Current Biology*) and Ellenbogen, Hulbert, Jiang & Stickgold
   (2009, *PLoS ONE*) reported sleep-conferred resistance to interference; Abel, Nickl,
   Reßle, Unger & Bäuml (2023, *Psychonomic Bulletin & Review*) found partial protection and
   concluded it is "still unclear" whether sleep protects against retroactive interference.
   A mechanism whose human effect is conditional cannot be the sole explanation for a
   binary engineering gate's failure.
4. **The measured null is a metric/mechanism mismatch, not a small effect.** In CLI, replay
   demonstrably moves an edge weight 0.20 -> 0.22, yet `on - off = 0.000` because the
   endpoint is a binary trust-and-typing decision. Adding compute or weight magnitude cannot
   cross a categorical gate. A "replay gain" that appears only when the replay is allowed to
   write provenance/typing would be a *different mechanism* wearing replay's name.

**What distinguishes interleaving from replay.** Interleaving is an **encoding-side,
retrieval-based** manipulation: each A item is reconstructed afresh in varying contexts,
which creates multiple distinct retrieval routes and reduces the dominance of any single
overloaded cue (contextual interference; Shea & Morgan, 1979, *Journal of Experimental
Psychology: Human Learning and Memory* — worse acquisition, better delayed retention and
transfer). Replay is a **post-encoding, consolidation-side** manipulation: it re-presents
the encoded routes and increases their stability/generality. Interleaving changes *what
there is to retrieve* (distinct, context-tagged routes); replay changes *how strongly the
existing routes are held*. That is why interleaving protects in CLI (0.875/1.000) while
weight-only replay does nothing (0.000). In hippocampal terms: interleaving approximates
pattern separation at input; replay without distinguishing context cannot create separation
after the fact.

**Brain-faithful way to reduce retroactive interference.** Make A and B *distinct at
encoding*, then let consolidation preserve the distinction:

1. **Distinct context/provenance tags** on A and B traces (the computational analogue of
   dentate-gyrus pattern separation) so shared node tokens do not collapse onto one cue.
2. **Interleaved rather than sequential A+B acquisition**, with A **retrieved** (
   effortful reconstruction) rather than restudied — retrieval practice is what makes the
   route durable (Roediger & Karpicke, 2006, *Psychological Science*; Karpicke & Blunt,
   2011, *Science*).
3. **Replay only the distinguishing features** — the A edge *together with* its provenance
   and relation type — so replay reinforces the separation instead of the overlap. If a
   replay step cannot see the tag, it cannot be the RI mechanism and should not be credited.
4. Where replay is co-active with new learning, treat the resulting lability as a real risk
   (Deliens et al., 2013) and schedule separation before reactivation, not after.

---

## 5. Observable criteria before claiming success

### 5.1 Failure 1 — "the answer path selects the trusted chain on a held-out paraphrase"

Claim success only if **all** of the following hold, measured with the existing graph-level
detector plus a separate answer-level endpoint (never a prose regex):

| # | Criterion | Pass threshold |
|---|---|---|
| C1 | Held-out-paraphrase **answer reach** on primary (and sealed) items | `has_chain` in emitted answer >= 0.60, and primary - sealed gap <= 0.15 |
| C2 | Cold-reload survival of the *answer-level* gain | paraphrase reach after cold reload >= 0.80 x immediate paraphrase reach, and >= 0.60 absolute |
| C3 | Paraphrase-invariance: two independent paraphrases of the same item | both >= 0.60; per-item disagreement <= 0.20 |
| C4 | Cue diagnosticity: probing with the relation signature alone, subject node removed | reach >= 0.50 (proves the route is relational, not topic-bound) |
| C5 | Cue-overload control: add N=20 untrusted chains on the same subject node before the probe | reach drop <= 0.15 (proves low fan-out, not brute-force search budget) |

**Negative controls (a fail regardless of C1-C5):**

- **Exact-surface parroting.** If gain on the taught surface exceeds gain on held-out
  paraphrases by > 0.15, or if the answer merely echoes taught sentences while the chain is
  absent, the result is parroting, not retrieval. The current 0.50 echo / <=0.10 paraphrase
  profile is exactly this signature.
- **Traversal-budget artefact.** If success requires raising `TRAVERSAL_LIMIT` after seeing
  the result, or if reach falls as budget falls from 64 to 8, the gain is a search artefact.
- **Graph-only success.** A cold-reload `Δ_cold = +0.80` with answer reach <= 0.10 is storage,
  not retrieval. It must not be reported as closing F1.
- **Weight bump.** Any intervention whose only measurable effect is confidence/weight
  change with an unchanged binary gate verdict is out of scope for F1.

### 5.2 Failure 2 — "sleep replay reduces retroactive interference"

Claim success only if **all** of the following hold:

| # | Criterion | Pass threshold |
|---|---|---|
| D1 | Sleep-on vs sleep-off `ret_A` on primary | `on - off >= 0.15` (hard fail < 0.10), sealed same direction and `on - off >= 0.10` |
| D2 | Compute-matched ablation | ON and OFF replay the same number of triples/cycles; wall-clock ratio in [0.9, 1.1]; OFF pads with out-of-range edge ids |
| D3 | Interference reinstatement | A-only retention exceeds sequential A->B by >= 0.10 on at least one pool, showing there is RI left to repair |
| D4 | Interleaving positive control preserved | concurrent A+B >= 0.85 on primary and >= 0.90 on sealed, and >= sequential A->B on both |
| D5 | Mechanism specificity | replay-on gain survives when OFF is given the *same* weight updates with wrong provenance/context tags — i.e. gain depends on *what* is replayed, not on how much |

**Negative controls (a fail regardless of D1-D5):**

- **Gain disappears at matched compute.** If `on - off` collapses to 0.000 when cycles,
  triples and wall clock are matched, the "gain" was extra processing, not consolidation.
  This is the current result and must be reported as FAIL, as it is in `LEARN5_CLI_MEASURED.md`.
- **Weight-only replay.** A 0.20 -> 0.22 edge bump with an unchanged trust/typing verdict is
  not consolidation of the retention endpoint; it is a parameter change in a different
  variable.
- **No interference to remove.** If A-only and A->B are equal (floor or ceiling), any
  replay gain is uninformative. Report the interference contrast, not just `ret_A`.
- **Fresh-engine saturation.** Transfer structural = 1.000 for the never-taught engine
  (G6): a 0.60 floor met by curated knowledge is not learning. Report trained-minus-fresh;
  current value is 0.000.
- **No cold reload.** For an RI/consolidation claim, require a saved-graph cold reload before
  the final retest. The current CLI run omits this; absence of observed gain does not change
  the FAIL direction, but a future positive must include it.

---

## 6. Narrowest mechanism change per failure (no code edited here)

### 6.1 F1 — change the *address*, not the strength

**Recommendation:** add a single **relation-signature retrieval index** over trusted chains
at encoding, and require the answer path to query it with a signature derived from the
question.

- At teaching time, store for each trusted chain its paraphrase-invariant key: the ordered
  relation types and their argument roles (e.g. `CAUSE(CAUSE(a,b), (b,x))`) plus a small set
  of sanctioned relational phrases ("mechanism", "because", "leads to") — not the literal
  teaching sentence.
- At answer time, derive the same key from the question and use it as the *primary* address
  into the candidate set, with node tokens as a secondary filter. The candidate set should
  be scored on `P(chain | cue)` (diagnosticity), not on edge weight alone.
- Keep the existing trust gate, `grade_chain`, and `assertable_chain` unchanged. This is an
  indexing/selection change, not a threshold change.
- Why this is narrowest: it touches exactly the step that fails (cue -> chain selection) and
  leaves storage, provenance, and the cold-reload behaviour that already works alone. It is
  directly predicted by Tulving & Thomson (1973), Morris et al. (1977), Watkins (1975), and
  Gentner et al. (2003). Do **not** solve it by strengthening edges, raising traversal
  budget, or tuning the paraphrase set.

### 6.2 F2 — change *what is replayed*, and stop assigning RI repair to replay

**Recommendation:** make interleaved, context-tagged A+B acquisition with A retrieval the
default RI defence; restrict replay to re-activating A **together with its distinguishing
provenance/relation tag**, and treat weight-only replay as a non-contribution.

- Preserve the interleaved positive control (0.875/1.000) as the default teaching schedule —
  it is the manipulation with measured protective effect and a direct literature basis
  (McCloskey & Cohen, 1989; Shea & Morgan, 1979).
- Give A and B distinct context/provenance tags so they are not subsumed by one retrieval
  cue; this attacks cue overload at encoding, which is where the measured 0.200 drop occurs.
- If `SleepConsolidation` / `HippocampalReplay` is retained, require the replayed triple to
  carry the A tag and relation type; a replay that cannot see the tag must not be credited
  with an RI gain. This converts replay from a strength operation into a separation-
  preserving one, consistent with Wilson & McNaughton (1994) and Diekelmann & Born (2010),
  and respects the lability risk in Deliens et al. (2013).
- Do **not** add cycles, compute, or weights to force `on - off > 0`: with matched compute
  that is the exact failure mode being tested against. The next positive result must pass
  D1-D5 and the matched-compute and cold-reload negative controls above.

---

## 7. Verified references

- Brown, R., & McNeill, D. (1966). The "tip of the tongue" phenomenon. *Journal of Verbal
  Learning and Verbal Behavior*, 5(4), 325-337. doi:10.1016/S0022-5371(66)80040-3.
- Deliens, G., Schmitz, R., Caudron, I., Mary, A., Leproult, R., & Peigneux, P. (2013). Does
  recall after sleep-dependent memory consolidation reinstate sensitivity to retroactive
  interference? *PLoS ONE*, 8(7), e68727. doi:10.1371/journal.pone.0068727.
- Diekelmann, S., & Born, J. (2010). The memory function of sleep. *Nature Reviews
  Neuroscience*, 11(2), 114-126. doi:10.1038/nrn2762.
- Ellenbogen, J. M., Hulbert, J. C., Stickgold, R., Dinges, D. F., & Thompson-Schill, S. L.
  (2006). Interfering with theories of sleep and memory: sleep, declarative memory, and
  associative interference. *Current Biology*, 16(13), 1290-1294.
  doi:10.1016/j.cub.2006.05.024.
- Ellenbogen, J. M., Hulbert, J. C., Jiang, Y., & Stickgold, R. (2009). The sleeping brain's
  influence on verbal memory: boosting resistance to interference. *PLoS ONE*, 4(1), e4117.
- Gentner, D. (1983). Structure-mapping: a theoretical framework for analogy. *Cognitive
  Science*, 7(2), 155-170.
- Gentner, D., Loewenstein, J., & Thompson, L. (2003). Learning and transfer: a general role
  for analogical encoding. *Journal of Educational Psychology*, 95(2), 393-408.
  doi:10.1037/0022-0663.95.2.393.
- Gentner, D., Rattermann, M. J., & Forbus, K. D. (1993). The roles of similarity in transfer:
  separating retrievability from inferential soundness. *Cognitive Psychology*, 25(4),
  524-575.
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks:
  the sequential learning problem. In G. H. Bower (Ed.), *The Psychology of Learning and
  Motivation* (Vol. 24, pp. 109-165). Academic Press.
- Morris, C. D., Bransford, J. D., & Franks, J. J. (1977). Levels of processing versus
  transfer appropriate processing. *Journal of Verbal Learning and Verbal Behavior*, 16(5),
  519-533. doi:10.1016/S0022-5371(77)80016-9.
- Abel, M., Nickl, A. T., Reßle, A., Unger, C., & Bäuml, K.-H. T. (2023). The role of sleep for
  memory consolidation: does sleep protect memories from retroactive interference?
  *Psychonomic Bulletin & Review*, 30(6), 2296-2304. doi:10.3758/s13423-023-02264-8.
- Roediger, H. L. (1973). Inhibition in recall from cueing with recall targets. *Journal of
  Verbal Learning and Verbal Behavior*, 12, 644-657.
- Roediger, H. L., & Karpicke, J. D. (2006). Test-enhanced learning: taking memory tests
  improves long-term retention. *Psychological Science*, 17(3), 249-255.
- Roediger, H. L., & Neely, J. H. (1982). Retrieval blocks in episodic and semantic memory.
  *Canadian Journal of Psychology*, 36(2), 213-242. doi:10.1037/h0080640.
- Shea, J. B., & Morgan, R. L. (1979). Contextual interference effects on the acquisition,
  retention, and transfer of a motor skill. *Journal of Experimental Psychology: Human
  Learning and Memory*, 5(2), 179-187. doi:10.1037/0278-7393.5.2.179.
- Tulving, E., & Thomson, D. M. (1973). Encoding specificity and retrieval processes in
  episodic memory. *Psychological Review*, 80(5), 352-373. doi:10.1037/h0020071.
- Watkins, M. J. (1975). Inhibition in recall with extralist "cues". *Journal of Verbal
  Learning and Verbal Behavior*, 14, 294-303.
- Watkins, O. C., & Watkins, M. J. (1975). Buildup of proactive inhibition as a cue-overload
  effect. *Journal of Experimental Psychology: Human Learning and Memory*, 1(4), 442-452.
  doi:10.1037/0278-7393.1.4.442.
- Wixted, J. T. (2004). The psychology and neuroscience of forgetting. *Annual Review of
  Psychology*, 55, 235-269. doi:10.1146/annurev.psych.55.090902.141555.
- Wilson, M. A., & McNaughton, B. L. (1994). Reactivation of hippocampal ensemble memories
  during sleep. *Science*, 265(5172), 676-679. doi:10.1126/science.8036517.
