# Web-driven learning loop + saturated components — root cause & brain-grounded design

**Status:** investigation/design doc (read-only analysis + literature; no engine
run). Branch `fix/weight-store-durability`. Author: parent (the delegated web
investigator was rate-limited twice before writing; this consolidates the
measured facts from `docs/WEB_NODE_ADMISSION_BLOCKED.md`,
`docs/WEB_LEARNING_CHAIN_VERIFIED.md`, the five negative-result docs, and the
`ravana-debugging` skill §7a/§7b, plus verified literature).

---

## 1. Web loop: the measured bottleneck is the INPUT, not the read schedule

Five attempts to improve web-driven learning were each implemented and A/B
tested; every one failed (0 causal hops) or produced extra reads with zero score
change, and was reverted:

| # | tweak | variable it tuned | result |
|---|---|---|---|
| 1 | query phrasing | when/what to read | 0 hops |
| 2 | bounded retry | when to read | 0 hops |
| 3 | typed InformationNeed + source veto | what to read | 0 hops |
| 4 | full-page extraction | how to read | failed A/B, reverted |
| 5 | gap widening | when to read | 86 extra reads, 0 score change |

**All five tuned the same axis: when/how/what to READ.** The measured input
statistics say that axis is not the binding constraint:

- Only **14.6% of 1,087 frozen queries** contain ≥2 extractable causal triples
  (the existing OpenIE extractor over the frozen corpus). 85.4% of queries
  cannot supply a connected 2-hop causal chain **no matter how the read is
  scheduled**.
- The gap-read template fired **4 times in 434 probes** — the read path almost
  never engages, so even a perfect extractor has nothing to do.
- The only measured learning gain came from **authored curriculum provenance
  (53 edges)**, not from the web loop.

**Root-cause verdict:** the loop is bottlenecked at (a) **corpus poverty** for
the *specific endpoint being measured* (trusted ≥2-hop causal chains) and (b)
**read-path rarity** (gap detection rarely fires). Tuning extraction/retry/
phrasing cannot fix either. This is the skill §7a rule: *locate the bottleneck
before building more machinery; measure both sides (read-path firing rate AND
corpus relation density).* Both were measured; both are low.

### What the corpus CAN support
Single facts and definitions are far more abundant than connected causal
pairs. A loop that acquires **single provenance-stamped facts/definitions**
(not 2-hop chains) has a realistic target; a loop scored only on
`assertable_chain` 2-hop chains is measuring something the corpus mostly lacks.
`chain_hops` (undirected component proxy) is NOT a valid endpoint — only
`assertable_chain` + `grade_chain` are.

---

## 2. Brain-grounded mechanism (DIFFERENT from the five failed tweaks)

The brain does not learn by reading more whenever a gap is flagged. Encoding is
**gated by a motivational/prediction-error signal** and **accelerated by schema
congruence**:

- **Epistemic curiosity = information-gap reward signal.** Kang et al. (2009),
  *Psychological Science* 20(8):963–973, doi:10.1111/j.1467-9280.2009.02402.x —
  curiosity about a trivia answer activates reward circuitry (SN/VTA, nucleus
  accumbens) and enhances later memory for the answer. Curiosity is an
  *information-gap* state: a specific unresolved question, not a broad "I know
  little about this topic".
- **Curiosity gates hippocampal encoding via dopamine.** Gruber, Gelman &
  Ranganath (2014), *Neuron* 84(2):486–496, doi:10.1016/j.neuron.2014.08.060 —
  high-curiosity states enhance hippocampus-dependent learning of BOTH the
  target answer and incidental material, via anticipatory SN/VTA–hippocampus
  connectivity. The motivational state, not mere exposure, determines what is
  encoded.
- **Schema-congruent facts encode rapidly.** Tse et al. (2007), *Science*
  316:76–82, doi:10.1126/science.1135935 — once a neocortical schema exists,
  congruent new associations are consolidated in hours, not weeks. A new fact
  that fits an existing structure is cheap to learn; an isolated fact is not.

### Proposed mechanism in RAVANA primitives (the untried axis)
The five tweaks tuned the READ schedule. The untried axis is the **ENCODING
GATE and its signal source**:

1. **Curiosity gate from an EXTERNAL signal, not a self-rated gap.** Per skill
   §7, a quality score derived from the system's own output is a self-confirming
   loop. The external signals already exist but are routed elsewhere: the next
   turn's **re-ask, repair, correction, or acknowledgment**. Define a
   `curiosity_need(subject)` that fires when the system *failed to answer* a
   subject AND the user re-asked/corrected it (an observed information gap),
   not when `detect_gaps` self-flags a low-degree node. This is the Kang/Gruber
   information-gap reward analogue: a specific unresolved question the
   environment has marked as worth resolving.
2. **Encode SINGLE provenance-stamped facts, schema-anchored.** When the gate
   fires and a read succeeds, write the extracted fact as a `web_fact` edge OR a
   verified definition anchored to an existing graph neighbour (Tse schema
   congruence: prefer facts whose target is already a neighbour of the subject).
   Stamp `source_metadata.edge_kind='web_fact'` + URL + evidence span at WRITE
   time (skill §6: stamp provenance, never loosen the gate).
3. **Score the endpoint the corpus can supply.** Measure **single-fact
   acquisition** (verified-definition count, `web_fact` edge count with
   provenance, and emitted-answer reach on those subjects) — NOT only 2-hop
   `assertable_chain`. Report the 2-hop chain metric separately as the
   corpus-limited ceiling it is.

**Prediction:** on the frozen corpus, the curiosity-gated single-fact loop
writes >0 provenance-stamped facts on subjects the system previously failed
(the 4/434 read-path firing rises because the gate is driven by observed
failures, not self-flagged gaps), and emitted-answer reach on those subjects
rises from ~0. **Falsifier:** if the gate fires but the extractor still yields
no usable single fact (corpus has neither pairs NOR singles for those subjects),
the corpus is the hard ceiling — record the negative and stop tuning the loop
(skill §7a). **Negative metric:** untrusted-edge count and known-bad probe
rejections must NOT rise (the gate must not admit co-occurrence as fact).

**Honest flag:** this is a hypothesis with a falsifier, not a result. Ship behind
a default-OFF flag, A/B on identical frozen state, revert on no gain (skill §7a).
The five negatives are strong evidence that the *read-schedule* axis is
exhausted; the encoding-gate axis is genuinely untried but unproven.

---

## 3. Live web n=60 arm: resumable checkpointed runner

The live arm died at 195/434 probes (~57 s/web turn). Per the frozen-snapshots
reference, a wide web arm needs **frozen snapshots, not a larger timeout**.
Design for `scripts/measure_fronts_web.py` (or a new `measure_fronts_web_resumable.py`):

1. **Append-only per-probe JSONL** (`scratch/front_probes_web_live.jsonl`): write
   each probe's record + flush immediately after it completes.
2. **Resume from last completed index:** on start, read the JSONL, skip probes
   already recorded; continue from `max(probe_index)+1`. A killed run loses at
   most one probe.
3. **Capture-on-first-fetch:** the first time a probe fetches a URL, save the
   page bytes to `scratch/web_fixtures/<hash>.html` and record the hash in the
   probe row. On replay (or resume after the fixture exists), serve the fixture
   and **block + count** live fetches (the frozen-arm tripwire).
4. **EXIT line + summary mtime check:** write `EXIT=<code>` only on completion;
   before reporting a mean, check probe-row count == expected total AND summary
   mtime > run start (a timed-out run leaves a stale smaller summary — skill §2).
5. **Re-measure the frozen arm post-intent-fix:** the frozen n=60 mean 77 was
   measured BEFORE `527f401`; re-run `measure_fronts_web_frozen.py` on the
   repaired weights + F1/F3/F5 to refresh it (it is stale).

This makes the live arm reproducible and resumable without raising the budget.

**Implemented 2026-11-21:** both `measure_fronts_web.py` and
`measure_fronts_web_frozen.py` now support `--resume`. Each probe row carries a
`state_id` = `commit:weight-pkl-mtime`; on resume, only rows from the SAME code
and weight state are reused (their ~57 s web turn is skipped), rows from any
other state are ignored as stale evidence, and the log is opened in append mode.
`tests/unit/test_web_front_resume.py` = 3 passed.

**Measured limitation found while re-measuring:** the frozen arm's old mean-77
summary (preserved as `scratch/front_probes_web_frozen_summary_STALE.json`) was
recorded on the **degraded cold-start graph** (209 PMI concepts), not on the
repaired learned graph. On the repaired 18,984-edge store the same harness runs
~2 min per probe (vs 161 s for the whole 434-probe run before repair) — a ~100×
slowdown from graph-size-dependent turn processing. A full frozen re-measure is
therefore infeasible in-session; the resume harnesses make it possible to
continue it across sessions. The interrupted re-run truncated the old per-probe
`scratch/front_probes_web_frozen.jsonl` (the summary and the Nov-20 full run log
`scratch/front_probes_web_frozen_run.log` preserve the evidence). **Treat mean 77
as stale and not a current result.**

---

## 4. Unsaturated WFA / PDR / RSI redesigns (G6-invalid passes)

All three PASS at 1.00 → invalid as progress signals (G6). Rebuild with harder
held-out items that cannot saturate on the current system. Keep the hash-split
sealing discipline (`sha256(front|difficulty|subject)` fixed before runs).

### WFA — web fact attribution (currently tautological)
- **Defect:** gold claim = source sentence (entailment trivially true);
  answer-only arm is an empty parametric store, not a live engine; n=32/pool.
- **Redesign:** (a) gold claim is a **paraphrase** of the source span, not the
  span itself, so entailment requires real matching; (b) answer-only arm is the
  **live engine with web disabled** (its actual parametric/graph answer), so the
  attribution-vs-answer gap measures real added value; (c) include
  **near-miss decoy sources** (same topic, wrong fact) so attribution must
  discriminate; (d) n≥60 gap + 30 closed-web + 30 decoy per pool (design §4.5).
- **Cannot saturate because:** paraphrase entailment + live-engine baseline +
  decoys introduce real error modes the current 1.00 hides.

### PDR — poison resistance (currently in-memory token substitution)
- **Defect:** poison is in-memory token substitution; never survives a real
  save/reload, so it tests a transient state, not durable resistance.
- **Redesign:** inject poison through the **real intake path**, `eng.save()`,
  cold reload, THEN measure intrusion at p=5. Include **provenance-forged
  poison** (a false fact stamped `web_fact`) so the trust gate must reject by
  content/consistency, not just by missing tag. Add a **sleeper poison** that is
  benign at write and activated by a later cue (tests consolidation-time
  re-evaluation).
- **Cannot saturate because:** durable + forged + sleeper poison exercise
  failure modes the in-memory substitution never reaches.

### RSI — rule induction (currently no engine boot, flat across k)
- **Defect:** no engine boot; flat k=1/2/4 → a mechanism check, not learning
  growth.
- **Redesign:** boot the **live engine**, teach the rule through conversation,
  and grade demonstration count k∈{1,2,4,8} with **graded difficulty** (near
  transfer → far transfer → foil). Require a **monotonic learning curve** across
  k (the CAR-style gate), not a flat 1.00. Add **negative rules** (the system
  must NOT induce a rule from counterexamples).
- **Cannot saturate because:** far transfer + foils + negative rules + a
  monotonicity gate introduce error modes a flat exact-match 1.00 hides.

---

## 5. Verified citations
- Kang, Hsu, Krajbich, Loewenstein, McClure, Wang & Camerer (2009). *Psychol
  Sci* 20(8):963–973. doi:10.1111/j.1467-9280.2009.02402.x (verified PubMed
  19619181 / SAGE).
- Gruber, Gelman & Ranganath (2014). *Neuron* 84(2):486–496.
  doi:10.1016/j.neuron.2014.08.060 (verified Cell/PMC4252494).
- Tse et al. (2007). *Science* 316:76–82. doi:10.1126/science.1135935 (verified
  in `CAR_ROOT_CAUSE.md` §2.6).
- McClelland, McNaughton & O'Reilly (1995). *Psychol Rev* 102:419–457 (CLS;
  verified in `CLI_ROOT_CAUSE.md`).

## 6. What this doc does NOT claim
- No web-loop gain is claimed; the proposed encoding-gate mechanism is a
  hypothesis with a falsifier, to be A/B tested default-OFF.
- The corpus ceiling (14.6% pairs) is a measured limit on the 2-hop endpoint,
  not a claim that the system cannot learn single facts.
- The redesigned WFA/PDR/RSI are designs, not yet implemented or measured.