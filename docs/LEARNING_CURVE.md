# Within-session learning curve — does RAVANA actually learn?

**Deliverable:** `scripts/measure_learning_curve.py` → `scratch/learning_curve_audit.json`
**Measured at:** commit `2766a80`, `RAVANA_OFFLINE=1`, ONE process, scratch `data_dir` copy, NO `eng.save()`.
**Status:** measured. **Verdict: no within-session learning is demonstrable through the default teaching path.** The instrument is valid (its positive control fires), so the flat curve is a real property of the current write path, not a probe artifact.

---

## 1. Why this metric, grounded in how human learning is measured

The claim under test is not "RAVANA scores well" but "RAVANA changes what it can explain because of what it just experienced." Four peer-reviewed sources define the design; none of the criteria below are ad hoc.

| Source | What it establishes | How the design implements it |
|---|---|---|
| **Ebbinghaus, H. (1885/1913). *Memory: A Contribution to Experimental Psychology*** (Ruger & Bussenius, trans., Teachers College, Columbia University; original *Über das Gedächtnis*, 1885). The savings/relearning method: retention is the time *saved* relearning relative to first learning, `savings = (L − L_t)/L`; Ebbinghaus was his own subject and replicated the whole procedure. | Learning is a **within-subject change from before to after an intervening episode**, never a single test score. | Each subject is asked the same mechanism question before and after teaching, on identical graph state in one process. The headline is the **delta** in trusted-chain rate, not a post score. (Our analogue of "savings" is a rate difference, not relearning time — stated plainly, see Limitations.) |
| **Roediger, H. L., III, & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. *Psychological Science, 17*(3), 249–255.** doi:10.1111/j.1467-9280.2006.01693.x | The post-teaching measurement must be a **retrieval test**, not re-reading the taught material; testing beats restudy. | The post question is a **held-out paraphrase** ("can you explain why X occurs?") of the baseline ("why does X happen?"), never the taught assertion. The no-teach control is the restudy-free baseline; the definition-only control is "study that does not supply the mechanism". |
| **Loewenstein, G. (1994). The psychology of curiosity: A review and reinterpretation. *Psychological Bulletin, 116*(1), 75–98.** doi:10.1037/0033-2909.116.1.75 | Curiosity is a felt **information gap** between what is known and what is wanted; a small "bite" of knowledge drives closure of that gap. | A `why/how` mechanism question creates the gap; teaching supplies the two missing causal links; the chain is the observable gap-closure. |
| **Schacter, D. L., & Addis, D. R. (2007). The cognitive neuroscience of constructive memory: Remembering the past and imagining the future. *Philosophical Transactions of the Royal Society B: Biological Sciences, 362*(1481), 773–786.** doi:10.1098/rstb.2007.2087 | **Constructive episodic simulation**: memory flexibly extracts and recombines elements of past episodes into new scenarios; the unit is a recombination, not a verbatim replay. | The learned unit is a **2-hop chain** assembled from a conversational episode and two web-reading episodes — a new recombination of A→B and B→X, not the stored sentence. A single fact is explicitly *not* counted as learning. |

**Property under test (one sentence):** after being taught a 2-hop causal mechanism, RAVANA's answer to a held-out paraphrase contains a provenance-trusted 2+ hop causal chain that was absent at baseline — and this does not happen with no teaching or with definition-only teaching.

---

## 2. Instrument (validated before the curve)

A "trusted chain" is the **structured chain object**, never a regex over prose (per `ravana-debugging/references/probe-validation.md`, re-read before reporting):

```
enumerate_chains(nids, graph, min_steps=2, max_steps=4, direction="backward")
  -> grade_chain(c)["verdict"] == "assert"
  -> assertable_chain(c)          # every hop has trusted provenance
```

This is the identical gate the engine uses in `ResponseGenMixin._why_how_causal_chain`. A trace hook on that method records whether the gate fired *during generation* and whether the emitted chain string reached the final answer, so "chain exists in the graph" is distinguished from "chain reached the user".

**Detector self-check, run before any curve** (all passed):

| Check | Result |
|---|---|
| positive: injected `a→b→x`, two `web_fact` causal edges | detected, labels `['zza','zzb','zzx']`, adequacy 1.0, verdict `assert` |
| negative: definition-shaped `is_a` edges | rejected (no chain) |
| direction: forward walk from `a` ends at `x` | correct |

**Controls mirror the human-learning controls the sources motivate:**

- **NO_TEACH_CONTROL (n=6)** — the Ebbinghaus no-episode baseline. A curve that rises here is a probe artifact. Measured delta: **0.000**.
- **DEF_ONLY_CONTROL (n=6)** — study without mechanism, analogous to restudy without the to-be-retrieved relation. Definition text via the same web channel, two URLs, no causal predicate. Measured delta: **0.000**; zero causal edges added.
- **WEB_UNGATED_POSITIVE (n=6)** — positive end-to-end control: same web channel, write gate disabled for the teaching call only. Confirms the instrument can detect learning when trusted edges actually land. Measured delta: **+0.167**.

---

## 3. Results (exact, from `scratch/learning_curve_audit.json`)

| Arm | n | baseline graph chain | post graph chain | baseline answer chain | post answer chain | new trusted causal edges | Δ chain rate |
|---|---|---|---|---|---|---|---|
| **TEACH_BOTH** (conversational assertion + 2-URL web teaching) | 12 | 0.000 | 0.000 | 0.000 | 0.000 | **0** | **0.000** |
| NO_TEACH_CONTROL | 6 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0.000 |
| DEF_ONLY_CONTROL | 6 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0.000 |
| WEB_UNGATED_POSITIVE | 6 | 0.000 | 0.167 | 0.000 | 0.167 | 6 | **+0.167** |

### Teaching-channel diagnosis (the reason the main arm is flat)

- **Web teaching wrote 0 of 24 facts** in TEACH_BOTH (12 subjects × 2 source URLs, all `[0, 0]`). The identical prose on a bare `WebToGraph` (no engine intake) writes **2** `causal`/`web_fact` edges. The engine's `KnowledgeIntake` schema-consistency gate rejects every causal teaching fact.
- **Definition teaching wrote 12 of 12 facts** (`[1, 1]` for each of 6 subjects) through the same gated channel. So the gate is not rejecting web facts generally — it rejects the causal ones specifically.
- **Conversational assertion added 10–14 causal edges per subject** (10, 13 or 14) — but **all untrusted**, so `assertable_chain` declines them and `new_trusted_causal_edges` stays 0. A naive "count causal edges" metric would have reported a large spurious gain here (~13 edges/subject); the provenance gate is what prevents that artifact.
- With the gate bypassed, only 1/6 subjects (`glow`) received **both** hops in the same session. The other five received one hop each — OpenIE extraction / cross-subject dedup, not the chain gate, is the remaining blocker. `glow`'s post answer was the exact gated chain: `excitation leads to emission, and emission leads to glow`, backed by graph chain `['excitation', 'emission', 'glow']`.

---

## 4. Hand-labelled before/after pairs (13 pairs, all probe verdicts agree)

Criterion applied by hand: does the answer state a 2+ hop causal mechanism for the subject (agent → intermediate → subject)? "Hedge/deflection" = no. Strings are verbatim from the audit.

| # | Subject | BEFORE answer | AFTER answer (held-out paraphrase) | Hand label | Probe |
|---|---|---|---|---|---|
| 1 | fog | `honestly, fog is a bit outside what i know right now.` | `hmm, i'm not totally sure about fog yet, and that's okay i'd love to hear your take first.` | no chain → no chain | agree |
| 2 | dew | `hmm, i'm not totally sure about dew yet, and that's okay what's your sense of it.` | `honestly, dew is a bit outside what i know right now i wonder what the real answer is — any theories.` | no → no | agree |
| 3 | rust | `honestly, rust is a bit outside what i know right now.` | `i don't really have a solid grasp on rust so far.` | no → no | agree |
| 4 | echo | `hmm, i'm not totally sure about echo yet, and that's okay.` | `honestly, echo is a bit outside what i know right now what do you think about it.` | no → no | agree |
| 5 | rainbow | `honestly, rainbow is a bit outside what i know right now i'd love to hear your take first.` | `hmm, i'm not totally sure about rainbow yet, and that's okay i wonder what the real answer is — any theories.` | no → no | agree |
| 6 | shadow | `honestly, shadow is a bit outside what i know right now.` | `i don't really have a solid grasp on shadow so far what do you think about it.` | no → no | agree |
| 7 | hiccup | `i don't really have a solid grasp on hiccup so far.` | `honestly, hiccup is a bit outside what i know right now i'd love to hear your take first.` | no → no | agree |
| 8 | blush | `i don't really have a solid grasp on blush so far.` | `hmm, i'm not totally sure about blush yet, and that's okay.` | no → no | agree |
| 9 | sunburn | `hmm, i'm not totally sure about sunburn yet, and that's okay what made you wonder about that.` | `honestly, sunburn is a bit outside what i know right now.` | no → no | agree |
| 10 | callus | `hmm, i'm not totally sure about callus yet, and that's okay i wonder what the real answer is — any theories.` | `i don't really have a solid grasp on callus so far what do you think about it.` | no → no | agree |
| 11 | bruise | `honestly, bruise is a bit outside what i know right now i'd love to hear your take first.` | `i don't really have a solid grasp on bruise so far what made you wonder about that.` | no → no | agree |
| 12 | sweat | `i don't really have a solid grasp on Sweat so far i wonder what the real answer is — any theories.` | `hmm, i'm not totally sure about Sweat yet, and that's okay what do you think about it.` | no → no | agree |
| 13 | glow *(ungated positive)* | `hmm, i'm not totally sure about glow yet, and that's okay.` | `excitation leads to emission, and emission leads to glow` | **no chain → TRUSTED 2-hop chain** | agree |

All 13 hand labels match the automated verdict. Zero false positives on the two negative controls; one true positive on the positive control.

---

## 5. Verdict

**RAVANA does not demonstrably learn within a conversation under the shipped teaching path.** On 12 held-out mechanism subjects, baseline and post trusted-chain rates are both **0/12**, with **zero** new trusted causal edges. The no-teach and definition-only controls are also flat at **0**, so the absence of a rise is **not** a measurement artifact — the probe rejects a chain where none exists and accepts one where it exists (`glow`, ungated positive control).

The blocker is localized, not architectural: the schema-consistency write gate rejects **24/24** causal teaching facts while accepting **12/12** definition facts; the conversational channel writes many causal edges but none with trusted provenance; and even with the gate bypassed, OpenIE/dedup delivered both hops for only **1/6** subjects. The architecture *can* emit a learned, provenance-trusted 2-hop explanation — the `glow` answer is exactly that — but the production ingestion path does not currently deliver the substrate for it. Answering the user's goal ("self improvement like human brain, learn with conversations and reading websites") therefore requires a source-level fix in the ingestion path, which is outside this deliverable's file ownership (`ravana/src/` is owned by other agents).

### Limitations, stated honestly

- **n is small.** 12 taught + 6 + 6 + 6 controls, not ~27; the main arm is a hypothesis test at this size, but the effect is exactly zero and the controls are clean, so the direction is not in doubt.
- **Surface answers are stochastic.** An earlier run of the same harness produced associative hedges instead of the deflections shown above for the same baseline questions; the structural chain gate and graph-edge counts are deterministic given edges, and only those carry the headline. Do not over-read the exact hedge wording.
- **"Savings" analogue is a rate delta**, not Ebbinghaus's relearning-time measure; RAVANA has no trial-count-to-criterion observable here.
- **The ungated positive control changes the teaching channel** (gate disabled), not the measurement. It is evidence that the instrument works and that the gate is the blocker — not evidence that the shipped system learns.
- No `eng.save()` was called; state lived only in memory on a scratch copy, so nothing persisted into `weights/`.
- **Concurrent edits:** other agents own `ravana/src/` and were editing it during this measurement window. At session end `git status` showed uncommitted changes to `engine.py`, `response_gen.py` and `web_learning.py` (including a new `close_mechanism_gap` method). The numbers above are from the on-disk code at run time, not a clean `2766a80` checkout. That new method is the natural follow-up teaching channel to measure once it is merged; it was deliberately not used here because the specified teaching channels were the conversational assertion and `WebToGraph.learn_text`.

## 7. Post-fix update (branch `fix/conversational-causal-trust`)

The flat curve above was real, but the localised cause was **not** a blanket
schema-gate rejection of causal facts. The conversational channel
(`chain_walker.py::_create_causal_edge`) wrote causal edges with no
`source_metadata` (`edge_kind='unknown'`), so `assertable_chain()` declined
them; it never called `KnowledgeIntake` at all. The fix routes the *asserted*
propositions through the existing gate with a user-source provenance record
and leaves the gate's thresholds untouched.

Re-run on the same unmodified instrument (branch source, commit `47bfe04`
baseline): **TEACH_BOTH Δ = +0.833** (10/12), 22 new trusted causal edges,
answer-chain rate 0.750; NO_TEACH Δ = 0.000 and DEF_ONLY Δ = 0.000.
Positive control Δ = +0.500 (shares intermediates with the teach arm — see
caveats). Full root cause, brain-grounded design, raw before/after strings,
tests and caveats: `docs/LEARNING_CURVE_CONSOLIDATION.md`.

```bash
cd C:/Users/Likhith/Documents/projects/ravana
uv run python scripts/measure_learning_curve.py
# full raw I/O per probe: scratch/learning_curve_after2.log
# machine audit (all 30 records, exact strings): scratch/learning_curve_audit.json
```

One cold boot (~3 min), all arms in one process, no concurrent engine runs.
