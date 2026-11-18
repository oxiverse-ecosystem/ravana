# Benchmarking survey and recommended suite for RAVANA

**Status:** research + recommendation. No code, `README.md`, or other docs were changed.
**Repo surveyed:** `C:/Users/Likhith/Documents/projects/ravana`, commit `1b8be3d`, tree clean.
**Method:** literature first (authors, year, peer-reviewed venue), recommendation derived second, per the standing rule. External claims carry inline ledger ids; the `Sources` block is generated mechanically from the retrieval ledger, not retyped.

---

## 0. Headline recommendation

Adopt a **tiered, pre-registered suite: RAVANA-7 v1**, in four layers:

1. **Tier 0 — instrument gates.** No score is reported unless the rubric passes known-good/known-bad pairs, a control-task selectivity check, a symmetric decline/answer guard, and web-on/web-off parity.
2. **Tier 1 — held-out probe suite.** **60 items per front** (30 primary + 30 sealed replication), replacing the current ~27 as the headline number while keeping the old 27 as provenance-only.
3. **Tier 2 — reference tracks, run quarterly.** Small, cheap slices of BIG-bench reasoning tasks, MMLU/MMLU-Pro as a *forgetting check only*, a live-web grounded-QA track, and locally executed code items — graded with WebArena/SWE-bench-style functional correctness and reported with HELM's multi-metric schema.
4. **Tier 3 — learning track.** Four curves per release: acquisition vs teaching events, continual A→B→A retention, Ebbinghaus relearning savings, and lexically disjoint transfer, plus an information-gap closure probe.

Five principles drive this design. Measure the behaviour, not the proxy string [26], and measure change, not a single score [13][16]. Treat every rubric as an instrument that must be validated before use [21][26]. Never reward disabling a capability [27], and report uncertainty, because 27 items cannot resolve small effects [28][29].

---

## 1. What "fit" means for this system

| Constraint | Consequence for benchmark choice |
|---|---|
| No LLM; ~20 W CPU | LM log-likelihood scoring, LLM-as-judge, and full agent environments are not usable as *headline scores*. They are usable as data sources or design patterns only. Energy must be a first-class reported metric, as in the NeuroBench system track, where power and time must both be reported on every run [30][31]. |
| Must learn online from conversation and web | Static accuracy is insufficient by construction. The unit of evaluation must be a learning curve or a savings/transfer measure [13][16][18]. |
| Seven capability fronts | Seven separately reported numbers with confidence intervals; a bare MEAN hides regressions and invites aggregate gaming. |
| Known measurement defects | Two defects are already documented in-repo: a causal regex counted the relativizers `that`/`which` as causal, so *"a device that stores chemical energy"* scored as reasoning (`docs/BRAIN_REASONING_DESIGN.md`, lines 11–30); and a synthetic "+20 web verified separately" bonus awarded points for a capability the harness had disabled, so an `OFFLINE` run scored 100 while an identical `ONLINE` run scored 80 (`scripts/measure_fronts.py`, lines 620–625; `docs/PLAN_ROUTE_REPLICATION.md`). Both are instrument failures, not system failures, and both map directly onto the probe-validity and Goodhart literature below [26][27]. |

**Definition of fit used here.** A suite fits if it (a) does not require an LLM or a large tool stack to score, (b) can be administered to a cold-booted, seeded, offline-or-online engine in one process, (c) yields a graded *change* signal, and (d) cannot be gamed by switching a capability off.

---

## 2. RQ1 — Established benchmarks: what fits and what does not

### 2.1 Static NLU and knowledge benchmarks

**GLUE / SuperGLUE.** GLUE is nine sentence- or sentence-pair classification tasks with a single-number aggregate (Wang et al., 2018, BlackboxNLP@EMNLP; ICLR 2019 version) [2]. SuperGLUE is eight harder NLU tasks with a toolkit and leaderboard, created because GLUE performance had surpassed non-expert human level, "suggesting limited headroom for further research" (Wang et al., 2019, NeurIPS 32) [1]. *Fit:* partial — a small, fixed slice is usable as a language-understanding floor check. *Misfit:* classification format, not generative conversation; saturated; no online-learning signal; the single-number aggregate is exactly the kind of metric that hides per-front regressions.

**BIG-bench.** 204 tasks from 442 authors at 132 institutions, spanning linguistics, childhood development, math, commonsense, physics, social bias, and software development (Srivastava et al., 2022, Transactions on Machine Learning Research) [3]. *Fit:* high as a *task donor* — the causal-judgement and logical-deduction subtasks make good held-out reasoning probes, and the paper's finding that multi-step tasks show brittle, "breakthrough" behaviour while memorization-heavy tasks improve smoothly is a direct argument for probing mechanisms rather than facts [3]. *Misfit:* the full suite is far beyond a 20 W budget and many items assume an LM response format.

**HELM.** HELM evaluates 30 models over 42 scenarios and measures seven metrics — accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency — for each of 16 core scenarios where possible, explicitly releasing prompts and completions (Liang et al., 2022, NeurIPS) [4]. *Fit:* high as a *reporting schema*, not a task list. Adopting efficiency as a first-class metric is the right pattern for a 20 W system [4][30]. *Misfit:* its scenario taxonomy is LM-centric; importing its task list would import the wrong substrate.

**MMLU.** 57 academic subjects, 15,908 four-option questions, evaluated zero- or five-shot (Hendrycks et al., 2020; ICLR 2021) [5]. *Fit:* very low. It measures accumulated declarative knowledge, not conversation or learning [5], and is now saturated: Wang et al. (2024, NeurIPS Datasets & Benchmarks) report that "rapid progress ... quickly led to performance saturation on MMLU", with frontier models clustered at 86–87% and only ~11% headroom, and show that MMLU-Pro's ten-option, reasoning-heavy version drops accuracy by 16–33 points and cuts prompt sensitivity from 4–5% to ~2% [34]. *Misfit:* use at most a tiny frozen slice as a catastrophic-forgetting check, never as a headline score; never optimize toward it.

### 2.2 Agent and interactive benchmarks

**AgentBench.** Eight environments (OS, database, knowledge graph, digital card game, lateral-thinking puzzles, plus household, web-shopping, web-browsing), multi-turn open-ended generation, tested on 27–29 models (Liu et al., 2023; ICLR 2024) [6]. *Fit:* the design principle — multi-turn, environment-grounded, per-environment scores with a failure taxonomy [6]. *Misfit:* requires API/tool stacks and presumes an LLM agent; unusable wholesale at 20 W.

**WebArena.** A self-hostable, reproducible web environment with 812 long-horizon tasks from 241 templates, graded by **functional correctness** (programmatic validators on end state or returned information) rather than action-sequence matching; at release the best GPT-4 agent reached 14.41% versus 78.24% human (Zhou et al., 2023; ICLR 2024) [7]. *Fit:* high as a grading philosophy and as the model for a small live-web grounded-QA track. Functional correctness is precisely what a no-LLM architecture can supply: check the answer against an evidence span, not against a model judge. *Misfit:* the full environment is too heavy; RAVANA's web use is reading and learning, not multi-site actuation.

**SWE-bench.** 2,294 real GitHub issues paired with pull requests from 12 Python repositories, graded by fail-to-pass tests; the original best configuration resolved 1.96% (Jimenez et al., 2023; ICLR 2024) [8][9]. *Fit:* the verification principle — hidden tests decide, so code generation can be graded without an LLM judge [8][9]. A local, scaled-down version fits the code-generation front. *Misfit:* repository-scale context and patching are beyond the current code front.

**GAIA.** 466 conceptually simple questions requiring reasoning, tool use, browsing, and multimodality, in three difficulty levels; humans scored 92% versus 15% for GPT-4 with plugins (Mialon et al., 2023; ICLR 2024) [10][11]. *Fit:* the three-level structure is a good template for stratifying RAVANA probes by number of required hops, and short-string answers are cheap to grade [10][11]. *Misfit:* the tasks require tools and multimodal input, and the headline agent result used manually selected plugins [11], i.e. an oracle configuration — a caution against reporting a best-case arm as the system score.

### 2.3 Loewenstein-style information-gap measures

Loewenstein's information-gap theory treats curiosity as a cognitively induced deprivation arising from the perception of a gap between what is known and what is wanted (Loewenstein, 1994, *Psychological Bulletin*, 116(1):75–98) [12]. This is the one item on the list that is a *theory of measurement*, not an off-the-shelf suite [12]. It fits RAVANA unusually well: a mechanism question opens a gap, a web query is the attempt to close it, and the measurable quantity is the reduction in gap — not the fluency of the final sentence. The existing `scripts/measure_learning_curve.py` already cites this theory in its design header [12]. The same logic predicts a failure mode to guard against: a system can close a gap by *declaring* one, so gap closure must be scored against an independently verified state change [12][26].

### 2.4 Verdict table

| Benchmark | Adopt as data | Adopt design/reporting | Small frozen slice only | Do not adopt |
|---|---|---|---|---|
| GLUE / SuperGLUE [1][2] | | | ✓ | |
| BIG-bench [3] | ✓ (reasoning subtasks) | | | |
| HELM [4] | | ✓ (7-metric schema) | | |
| MMLU / MMLU-Pro [5][34] | | | ✓ (forgetting check) | |
| AgentBench [6] | | ✓ (multi-turn, per-env) | | |
| WebArena [7] | | ✓ (functional correctness) | | |
| SWE-bench [8][9] | | ✓ (hidden-test grading) | | |
| GAIA [10][11] | | ✓ (level stratification) | | |
| Information-gap [12] | ✓ (custom probes) | | | |
| NeuroBench [30][31] | | ✓ (energy as first-order) | | |

---

## 3. RQ2 — Measuring learning rather than static competence

**Few-shot and acquisition curves.** GPT-3 was evaluated as k examples specified purely in text, with no gradient updates, and the paper's central move is to report performance as a function of k rather than as a single score (Brown et al., 2020, NeurIPS 33) [13]. For RAVANA, k is the number of teaching events in the session, not the number of prompt demonstrations [13]. Report an acquisition curve (slope to criterion and area under the curve), not just the final pass rate.

**Continual learning and forgetting.** Lifelong learning is defined as accommodating new information without catastrophic forgetting, with retention and transfer as joint requirements (Parisi et al., 2019, *Neural Networks*, 113:54–71) [14][15]. The classic intervention is to protect parameters important for earlier tasks — elastic weight consolidation slows learning on those weights (Kirkpatrick et al., 2017, *PNAS*, 114(13):3521–3526) [20]. The measurement consequence is an A→B→A protocol: teach A, teach B, retest A [14][20]. Report backward transfer and forgetting as separate numbers, and run the consolidation-on/off ablation; RAVANA's sleep cycle is the local analogue of the protected-weight mechanism.

**Ebbinghaus savings.** Ebbinghaus's savings measure is the proportion of learning time saved on relearning, S = (L − L_t)/L, where L is the effort to reach criterion initially and L_t the effort to relearn after delay (Ebbinghaus, 1885/1913) [17]. A modern analysis proves the measure is independent of initial encoding strength, learning time, and relearning times [16]. This is the correct primary learning metric for RAVANA because it separates *stored* from *currently accessible*: a single post-test confounds the two. The post-test must be a retrieval test on a held-out paraphrase rather than a re-reading of the taught text, because testing itself improves later retention (Roediger & Karpicke, 2006, *Psychological Science*, 17(3):249–255) [32]. The unit of learning should be a causal chain, not an isolated fact, consistent with the constructive episodic simulation account in which past elements are flexibly recombined into a novel scenario (Schacter & Addis, 2007, *Phil. Trans. R. Soc. B*, 362(1481):773–786) [33] and with the structured-chain criterion already adopted in `docs/BRAIN_REASONING_DESIGN.md`.

**Transfer tests.** Transfer learning uses knowledge from a source domain to improve learning in a different target domain (Pan & Yang, 2010, *IEEE TKDE*, 22(10):1345–1359) [18]; the framing of learning-to-learn makes transfer and reuse explicit design goals rather than incidental benefits (Thrun & Pratt, 1998, *Learning to Learn*, Kluwer) [19]. For RAVANA: teach subject A, then test a lexically disjoint but structurally analogous subject B [18][19]. Report the transfer delta (B after A minus B in a fresh engine), not just A's post-teach score — otherwise memorization of A is misread as generalization.

**Information-gap closure.** Under [12], the measurable event is a change from a detected gap to a resolved gap. Administer a mechanism question, record the gap state and any web query it triggers, provide the teaching episode, then re-measure with a held-out paraphrase [12][32]. Score the *reduction*, and require the post-state to be verified against the graph or an external evidence span, so that an emitted "now I understand" cannot count as closure [12][26].

---

## 4. RQ3 — Pitfalls of small-n probes and saturated benchmarks

### 4.1 Small n and low statistical power

Low statistical power does not merely miss effects; it reduces the probability that a significant result is true, and inflates effect-size estimates (Button et al., 2013, *Nature Reviews Neuroscience*, 14:365–376) [28][29]. The same arithmetic applies to a 27-item front. Computed here (Wilson 95% interval for a proportion at p = 0.50): n=27 gives ±17.6 points; n=60 gives ±12.3; n=100 gives ±9.6. Approximate power for a two-proportion test of a 20-point difference at α=0.05 is 32% at n=27, 61% at n=60, and 76% at n=85; for a 10-point difference it is only 11% at n=27 and 29% at n=100. **A 27-item front therefore cannot resolve a 10–15 point change that a benchmark report would normally call meaningful.** This is the quantitative reason to raise n to 60 rather than to celebrate small deltas.

### 4.2 Multiple comparisons and researcher degrees of freedom

The false-discovery rate is the expected proportion of rejected hypotheses that are false, and a sequential Bonferroni-type procedure controls it with substantially more power than family-wise correction (Benjamini & Hochberg, 1995, *JRSS-B*, 57(1):289–300) [22]; the simpler Holm step-down procedure controls family-wise error (Holm, 1979, *Scandinavian Journal of Statistics*, 6:65–70) [23]. With seven fronts tested in both web modes, fourteen comparisons are in play; a Bonferroni threshold would be α = 0.05/14 ≈ 0.0036. More dangerous than the arithmetic is undisclosed flexibility: simulations and experiments show that choosing among dependent variables, extending data collection until significance, and reporting subsets make it easy to obtain "significant" evidence for a false hypothesis (Simmons et al., 2011, *Psychological Science*, 22(11):1359–1366) [21]. The in-repo history of adjusting rubrics until a front moves is precisely this failure mode. Fix it by pre-registering probe lists, rubrics, seeds, and stopping rules before the run, and by reporting all arms including failed ones [21].

### 4.3 Held-out sets, reuse, and contamination

Newly collected test sets for ImageNet and CIFAR-10 produced accuracy drops of 11–14% and 3–15% respectively, even though the relative ordering of models was preserved; the authors attribute the drop to difficulty drift rather than adaptive overfitting, and note that scores are sensitive to data-cleaning minutiae (Recht et al., 2019, ICML) [24][25]. Separately, MMLU's public test set and its prompt sensitivity (4–5% across 24 prompt styles, versus ~2% for MMLU-Pro) make it a poor discriminator at the frontier [34]. The design consequence is a **sealed replication set**: freeze it before any tuning, reveal it once per release, never use it for rubric development, and refresh the primary set when it saturates.

### 4.4 Probe and instrument validity

High probe accuracy need not show that the representation carries the property; a probe can memorize the task itself [26]. Control tasks with random outputs, and the resulting *selectivity* (real-task accuracy minus control-task accuracy), expose this (Hewitt & Liang, 2019, EMNLP-IJCNLP, pp. 2733–2743) [26]. Translated to RAVANA: a front score can be produced by a rubric pattern rather than by the target behaviour — which is exactly what happened when `that`/`which` were treated as causal markers. Every rubric therefore needs (i) a control probe set that the behaviour cannot pass by construction, and (ii) paired known-bad / known-good strings with the assertion bad < good. The first run of a new rubric is a draft of the instrument, not a measurement [26].

### 4.5 Goodhart effects and the "disable the web" bonus

When an optimized signal is an imperfect proxy, increasing optimization eventually reduces the true objective (Gao, Schulman & Hilton, 2023, ICML, PMLR 202:10835–10866) [27]. The removed "+20 web verified separately" bonus is a textbook instance: the proxy rewarded the *absence* of the web, so the OFFLINE arm scored 100 and the identical ONLINE behaviour scored 80 (`scripts/measure_fronts.py`, lines 620–625). A single guard is insufficient; the rule must be structural — **no score component may be conditional on a capability being disabled**, and every front is reported in both modes.

### 4.6 Saturation

Saturation is not success; it is the loss of a measuring instrument [1][34]. SuperGLUE was created precisely because GLUE had passed non-expert human performance [1], and MMLU-Pro was created because MMLU had saturated [34]. Adopt an explicit rule: a front scoring ≥95 or ≤5 on both modes is flagged non-informative and its items are refreshed before it is used as a progress signal.

---

## 5. RQ4 — Recommended concrete tiered benchmark: RAVANA-7 v1

### 5.1 Tier 0 — instrument gates (block reporting if any fail)

| Gate | Rule | Basis |
|---|---|---|
| G1 Paired rubric check | For every rubric, score the known-bad string and the known-good string; assert bad < good. The causal criterion must reject *"a device that stores chemical energy"* and accept a provenance-trusted 2-hop chain. | [26], in-repo defect |
| G2 Selectivity | Real-probe pass rate minus control-probe pass rate ≥ 0.30 (proposed threshold; ratify after the first dry run). | [26] |
| G3 Symmetric gap guard | Declining when a trusted chain exists is FAIL; asserting when none exists is FAIL. A decline-everything policy cannot score 100. | in-repo design spec |
| G4 Mode parity | Every front is administered in `RAVANA_OFFLINE=1` and web-on arms; scores are reported as a pair. Any capability-gated bonus is invalid. | [27], in-repo defect |
| G5 Census sanity | Verify container iteration and cold-reload persistence before claiming absence or learning. | in-repo measurement-validity note |
| G6 Saturation | Flag ≥95 or ≤5 as non-informative and refresh items. | [1][34] |

These gates are cheap, deterministic, and must run on every commit that touches a rubric. They are the direct answer to "the instrument once produced a confidently wrong number".

### 5.2 Tier 1 — held-out probe suite: 60 items per front

- **Size.** 60 items per front: **30 primary** (used for iteration) + **30 sealed replication** (frozen before tuning, opened once for the release report). The existing 27 items stay in the repository as provenance and are kept disjoint from both pools.
- **Why 60.** At n=27 the Wilson 95% half-width is ±17.6 points and power against a 20-point change is 32%; at n=60 it is ±12.3 points and 61% power (computed above). Sixty is the smallest pool that makes a moderate change detectable without exceeding a 20 W run budget.
- **Difficulty stratification.** 20 items at each of three levels, following GAIA's level pattern [10][11]: L1 single-step recall/greeting, L2 two-step mechanism or grounded fact, L3 multi-hop or cross-session composition.
- **Front coverage.** The same seven fronts as `scripts/measure_fronts.py`: conversation, grounded factual answers, opinions/stance, thinking/reasoning, online language learning, speaking/surface realization, code generation.
- **Grading.** Structured or exact checks wherever possible; for THINKING, grade the structured causal-chain object (hops, shared intermediate, process predicate, trusted provenance, direction, load-bearing edges), never a prose regex. No LLM judge. Each record stores `{front, item_id, level, query, response, passed, reason, control, sub_scores, mode}`.

### 5.3 Tier 2 — reference tracks (quarterly, not per commit)

| Track | Size | Purpose | Grading |
|---|---|---|---|
| BIG-bench reasoning slice [3] | 50 items (`causal_judgement`, `logical_deduction`) | Independent reasoning floor | Exact match |
| MMLU frozen slice [5][34] | 5 × 57 subjects = 285 | Catastrophic-forgetting check only; never a headline | Exact match |
| Live-web grounded QA [7][12] | 30 items, web **on in both arms** | Gap detection → query → evidence-span answer | Evidence-span match, not judge |
| Local code items [8][9] | 10 functions | Code front | Hidden unit tests via `pytest` |
| HELM wrapper [4][30] | all above | Accuracy, calibration, robustness, energy | Multi-metric report |

The web track is the explicit repair for the removed bonus: web access is a **measured capability with an ON score**, never a condition that awards points when switched off. The gap measure is the Loewenstein quantity [12]; the grading is the WebArena functional-correctness principle [7].

### 5.4 Tier 3 — learning track (per release)

| Track | Design | Reported number |
|---|---|---|
| L1 Acquisition [13] | 30 novel subjects; teach at k = 0, 1, 2, 4, 8; retrieval test at each k | Slope to criterion and AUC |
| L2 Continual retention [14][15][20] | A→B→A over 30 subjects; sleep/consolidation on vs off | Backward transfer and forgetting |
| L3 Ebbinghaus savings [16][17][32] | L and L_t to criterion; post-test is a held-out paraphrase | S = (L − L_t)/L, with 95% CI |
| L4 Transfer [18][19][33] | 30 lexically disjoint A→B structural analogues, fresh-engine control | Transfer delta with CI |
| L5 Information-gap closure [12] | 30 mechanism questions; gap state before/after a web query, verified against graph or evidence span | Gap reduction |

All learning claims require a cold reload (G5). All five tracks run with one cold boot, fixed seed, no `eng.save()`, and a copied scratch data directory, matching the existing harness policy.

### 5.5 Controls and reproducibility

1. **Pre-register** probe ids, rubrics, seeds, thresholds, and stopping rules before tuning; report every arm, including failures [21].
2. **Pin** the commit SHA, the probe-file hashes, Python version, and `PYTHONHASHSEED=0`; use `RAVANA_OFFLINE=1` for offline arms and a recorded web snapshot or URL list for online arms.
3. **One cold boot per run**, no `eng.save()`, in-memory learning only — the existing `measure_fronts.py` policy.
4. **Correct for multiplicity**: Benjamini–Hochberg FDR at q=0.05 across the seven primary front comparisons [22]; use Holm [23] when a single front is claimed prospectively. Report both raw and adjusted values.
5. **Never aggregate away a regression**: publish seven front scores plus the learning-track table; a MEAN is optional and must never replace them.
6. **Report non-improvements** and control failures explicitly, following `docs/HONEST_REPORTING_STANDARD.md`.

### 5.6 Reporting format

Per front, one table row:

```
Front | n | Primary (95% CI) | Δ vs baseline (BH q) | Controls G1-G6 | OFF / ON | Energy (J) | Status
```

Each cell carries N= and a measurement method, per the repo's honest-reporting standard. The learning track gets a second table with curve parameters and savings/transfer CIs. A front is only comparable across releases if the same gate set, pool, and mode pair were used.

### 5.7 Decisions requiring human sign-off

- Ratify the G2 selectivity threshold (0.30 is a proposed starting value, not a literature constant).
- Approve the 60-item pool and author/seal the 30 replication items before the first scored run.
- Decide whether the MMLU and BIG-bench slices are worth their runtime as quarterly checks.
- Choose the energy measurement method (wall-power meter vs `psutil` estimate) for the Energy column.

---

## 6. RQ5 — README section (copy verbatim)

````markdown
## Running the RAVANA-7 benchmark

From the repo root, with the pinned environment:

```bash
# instrument gates + seven fronts, offline arm
RAVANA_OFFLINE=1 PYTHONHASHSEED=0 uv run python scripts/measure_fronts.py

# same probes, web arm — both arms are always reported
PYTHONHASHSEED=0 uv run python scripts/measure_fronts_web.py

# within-session learning (acquisition, retention, savings, transfer)
RAVANA_OFFLINE=1 PYTHONHASHSEED=0 uv run python scripts/measure_learning_curve.py

# quarterly reference tracks (optional, slower)
uv run python scripts/external_benchmark.py --quick
uv run python scripts/evaluate_ravana.py --skip-train --no-curiosity
```

Raw per-item records go to `scratch/front_probes_n30.jsonl` and the summary to
`scratch/front_probes_n30_summary.json`. The learning curves write to
`scratch/learning_curve_audit.json`. Commit SHA, seed, and probe hashes are
recorded with every summary; an unpaired or ungated run is not a result.

The scripts currently ship 27 items per front; the 60-item primary/replication
split is the RAVANA-7 v1 target in `docs/BENCHMARKING_SURVEY.md` §5.2. Until it
is adopted, read the interval at n=27 as roughly ±17.6 points.

**What the numbers mean.** Each front is a 0–100 score over 60 held-out items
(30 primary + 30 sealed replication), reported with a 95% interval and a
web-off / web-on pair. A higher score means more held-out probes passed the
rubric, not that the system "understands" the topic. The learning numbers are
the ones to watch for a system that must learn online: `savings` is
(L − L_t)/L — the fraction of relearning effort saved after teaching — and
`transfer` is the gain on a lexically disjoint analogue in a fresh engine.
Both use held-out paraphrases as the post-test.

**What the numbers do not mean.** A front at 100 is saturated and
non-informative, not solved. A score that improves only in the offline arm is
a measurement defect (a capability was rewarded for being off), not a gain.
Any change of more than ~12 points at n=60 is inside the interval unless the
report shows a corrected p-value. See `docs/BENCHMARKING_SURVEY.md` for the
sources, the gates, and the full tiered protocol.
````

---

## 7. Limitations of this recommendation

The proposed 60-item pools, the 0.30 selectivity threshold, and the quarterly cadence are engineering choices derived from the cited statistics, not values stated by those papers; they need the sign-off in §5.7 and one dry run before being frozen. The full agent suites were judged by their published descriptions and release-time results rather than by executing them here, and no engine was run for this document. Energy figures require instrumentation that does not yet exist in the harness. Finally, several cited works are benchmark papers reporting their own release-time baselines; those baselines are historical context, not current state of the art.

## Sources

[1] https://proceedings.neurips.cc//paper_files/paper/2019/hash/4496bf24afe7fab6f046bf4923da8de6-Abstract.html
[2] https://www.gabormelli.com/RKB/2018_GLUEAMultiTaskBenchmarkandAnaly
[3] https://cicl.stanford.edu/publication/srivastava2022imitation
[4] https://arxiv.org/abs/2211.09110
[5] https://arxiv.org/abs/2009.03300
[6] https://arxiv.org/html/2308.03688v3
[7] https://arxiv.org/html/2307.13854
[8] https://www.swebench.com/original.html
[9] https://arxiv.org/abs/2310.06770
[10] https://ai.meta.com/research/publications/gaia-a-benchmark-for-general-ai-assistants
[11] https://arxiv.org/abs/2311.12983
[12] https://psycnet.apa.org/record/1994-41058-001
[13] https://proceedings.neurips.cc/paper_files/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html
[14] https://research.ece.ncsu.edu/ci4r/wp-content/uploads/sites/40/2025/05/continuallifelonglearning_parisi2019.pdf
[15] https://dblp.uni-trier.de/rec/journals/nn/ParisiKPKW19.html
[16] https://link.springer.com/article/10.3758/s13423-022-02172-3
[17] http://psychclassics.yorku.ca/Ebbinghaus/wozniak.htm
[18] https://www.computer.org/csdl/journal/tk/2010/10/ttk2010101345/13rRUxAAT7Y
[19] http://dl.acm.org/doi/10.5555/296635
[20] https://www.pnas.org/doi/full/10.1073/pnas.1611835114
[21] https://www2.psych.ubc.ca/~schaller/528Readings/SimmonsNelsonSimonsohn2011.pdf
[22] https://academic.oup.com/jrsssb/article/57/1/289/7035855
[23] https://support.sas.com/documentation/cdl/en/statug/63962/HTML/default/statug_multtest_sect028.htm
[24] https://icml.cc/virtual/2019/oral/5167
[25] https://ar5iv.labs.arxiv.org/html/1902.10811
[26] https://arxiv.org/html/1909.03368v1
[27] https://proceedings.mlr.press/v202/gao23h.html
[28] https://pubmed.ncbi.nlm.nih.gov/23571845
[29] https://www.nature.com/articles/nrn3475
[30] https://www.osti.gov/servlets/purl/2584386
[31] https://neurips.cc/virtual/2024/101250
[32] https://journals.sagepub.com/doi/10.1111/j.1467-9280.2006.01693.x
[33] https://doi.org/10.1098%2Frstb.2007.2087
[34] https://papers.neurips.cc/paper_files/paper/2024/file/ad236edc564f3e3156e1b2feafb99a24-Paper-Datasets_and_Benchmarks_Track.pdf
