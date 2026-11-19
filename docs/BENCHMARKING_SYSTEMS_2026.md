# Modern tougher benchmarks for RAVANA — research, critique, and a replacement suite (2026)

**Status:** research + recommendation. This document changes no code, no harness, no gates, and no other doc. It is the 2026 successor to `docs/BENCHMARKING_SURVEY.md` (written at `1b8be3d`) and to the measured RAVANA-7 v1 result at `6a2b5fa`.
**Repo / tree:** `C:/Users/Likhith/Documents/projects/ravana`, clean at `6a2b5fa` (remote `main` identical) at the time of research.
**Method:** literature first (authors, year, venue/DOI, availability), then derivation, then recommendation. Every external claim carries an inline ledger id; the `Sources` block at the end is generated mechanically from the retrieval ledger, not retyped. Claims that could not be verified are marked `[unverified]`.
**Scope rule:** this document only *recommends*. It does not weaken a single existing gate and does not claim any score was produced by the proposed suite — none of it has been implemented or run.

---

## 0. Decision in one page

**Adopt RAVANA-LEARN-5 (v2)**, five RAVANA-native components that inherit the *designs* of modern benchmarks rather than importing their item banks:

| ID | Component | Benchmark design inherited | Learning signal | Control |
|---|---|---|---|---|
| CAR | Conversational Acquisition & Retention | LoCoMo, LongMemEval, Ebbinghaus savings [13][15][29] | acquisition curve k=0,1,2,4,8 + delayed paraphrase | untaught (U), shuffled-teaching (S) |
| WFA | Web Fact Acquisition & Attribution | FreshQA, SealQA, SimpleQA Verified, HALoGEN [17][19][20][26] | gap → read → answer, before/after, with source URL | no-read (C), clean snapshot |
| PDR | Poison & Definition-Disambiguation Resistance | PoisonedRAG, Machine Against the RAG, HaluMem [24][25][23] | error acceptance vs number of poisoned docs | clean snapshot (C) |
| CLI | Continual Learning under Interference | LifelongAgentBench, A→B→A, EWC/sleep ablation [22][31][32] | A→B→A retention + forward transfer | fresh engine, consolidation off |
| RSI | Rule Induction / Sealed Transfer | ARC-AGI-2 (few-shot rule induction, exact match) [7][8] | accuracy vs number of demonstrations | permuted demonstrations (S) |

**Do not adopt as scores:** MMLU-Pro, GPQA, HLE, SWE-bench/Pro, HAL, SimpleQA, BrowseComp in their native form. They are excellent difficulty references and grading templates but are static, LLM-shaped, or too heavy for a 20 W CPU box with no LLM; none measures *acquisition*. Use them as design donors and quarterly floor checks only (§2.7).

**Why the change:** RAVANA is specified to learn through conversation and web reading. The current 7-front suite is a one-shot QA instrument: code generation is saturated at 100 and non-informative, thinking (8) and speaking (7) sit at the floor, opinions (87) and conversation (93) reward surface markers, and grounded (17) conflates "not in the concept store" with "cannot close a gap from the web". A static score cannot distinguish a system that *knows* a fact from one that *learned it in this session*, and that distinction is the whole claim [30][34]. Probe-validity and Goodhart failures are the standing warnings here [35], and rewarding a capability for being absent is the specific failure to avoid [36].

**Non-negotiables retained:** G1–G6 instrument gates from `docs/BENCHMARKING_SURVEY.md` §5.1 remain in force and are byte-identical in intent; no score is reported if any gate fails; no rubric improvement is allowed to come from disabling a capability; every component is reported in both `RAVANA_OFFLINE=1` and IntentForge-on arms. [35][36]

---

## 1. Fit constraints — what a benchmark must do to be usable here

| Constraint on RAVANA | Consequence |
|---|---|
| No LLM, ~20 W CPU | LM-log-likelihood scoring, LLM-as-judge, full agent environments and repository-scale patching cannot be *headline scores*. They are data sources or design patterns only. Benchmarks must be gradeable by exact match, structured-object checks, hidden tests, or evidence-span comparison. |
| Must learn online from conversation and web reading | The unit of evaluation must be a **curve** or a **savings/transfer** measure, not a single post-test. Few-shot evaluation established the principle of reporting performance as a function of the number of examples rather than one number [30]; for RAVANA k is the number of teaching events or web reads, not prompt demonstrations. Continual-learning work requires retention *and* transfer to be reported jointly [32], and protected-weight/sleep mechanisms require an on/off ablation [31]. |
| Seven fronts today; five components proposed | Every component reports its own number with an interval and a control delta. A MEAN never replaces them. Multiplicity across five prospective claims is corrected with Benjamini–Hochberg at q=0.05 (Holm for a single prospective claim). |
| Known measurement defects in-repo | A causal regex once counted `that`/`which` as causal; a now-removed `+20 web verified separately` bonus rewarded the web being *off*, scoring OFFLINE 100 vs ONLINE 80 for identical behaviour. Both are probe-validity and Goodhart failures, not system results [35][36]. |
| Repeated search-snippet poisoning incidents | Any web component must grade provenance and poison acceptance separately from answer correctness, and must include adversarial snippet hits by construction [24][25]. |

**Definition of fit used throughout.** A suite fits RAVANA if it (a) scores without an LLM or large tool stack, (b) can be administered to a cold-booted, seeded engine in one process in both web modes, (c) yields a graded *change* signal with a control, and (d) cannot be gamed by switching a capability off.

---

## 2. Catalogue of modern benchmarks (2024–2026)

### 2.1 Static knowledge and reasoning — difficulty references, not scores

**MMLU-Pro.** Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al., **NeurIPS 2024 Datasets and Benchmarks**; arXiv:2406.01574 [1][2]. 14 domains, >12,000 questions, ten options instead of four; accuracy drops 16–33 points relative to MMLU and prompt sensitivity falls from 4–5% to ~2% [1]. *Difficulty:* high for an LLM, meaningless for RAVANA (four/ten-option multiple choice is not its output format). *Suitability:* frozen slice only as a catastrophic-forgetting check; the 2026 HLE paper notes frontier models now exceed 90% on MMLU, i.e. the parent is saturated too [28]. *Licensing/availability:* code and data public; repository Apache-2.0 [2], paper CC BY 4.0 [1]. *Verdict:* design donor (distractor robustness), never a headline.

**GPQA.** David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman, **First Conference on Language Modeling (CoLM 2024)**; arXiv:2311.12022, DOI 10.48550/arXiv.2311.12022 [3][4]. 448 expert-written multiple-choice questions in biology, physics, chemistry; PhD experts reach 65%, skilled non-experts 34% despite >30 min of unrestricted web access ("Google-proof"), strongest GPT-4 baseline 39% [3]. *Difficulty:* very high. *Suitability:* none as a score — no LLM to reason at graduate level, and multiple choice rewards guessing; useful only to argue that "web access" is not the same as "knowledge". *Licensing/availability:* public dataset with canary string; repository MIT [4]. *Venue note:* the arXiv record is dated 2023 [3]; the authors' canonical citation is CoLM 2024 [4]; the commonly repeated "NeurIPS 2023" tag was not verified here and is not used.

**SWE-bench.** Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan, **ICLR 2024**; arXiv:2310.06770 [5]. 2,294 real GitHub issues across 12 Python repositories, graded by hidden fail-to-pass tests; release-time best was 1.96% [5]. *Successor:* **SWE-bench Pro**, Xiang Deng, Jeff Da, et al., 2025, arXiv:2509.16941 [21] — 1,865 human-verified, enterprise-level tasks over 41 repos; top frontier models scored 23.3% (GPT-5) and 23.1% (Opus 4.1) versus >70% on SWE-bench Verified [21]. *Difficulty:* extreme for an agent, infeasible for RAVANA. *Suitability:* **grading philosophy only** — hidden functional tests decide, so the saturated `code generation=100` front can be rebuilt around held-out specs with `pytest` rather than string matching. *Licensing/availability:* code/data public, MIT [6]. *Verdict:* adopt the hidden-test grading, not the repos.

**ARC-AGI-2.** François Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers, Henry Pinkard, 2025, arXiv:2505.11831 — **preprint, not peer-reviewed** [7]. 1,000 public training and 120 public evaluation grid tasks; exact-match output grids, 2 trials per test input; average human performance 66% on the public evaluation sample [8]. Designed specifically to resist brute-force search and reward compositional, context-dependent rule induction [7]. *Suitability:* the grid format is wrong for a language-only system, but the *design* — few novel demonstrations, exact grading, sealed evaluation set, difficulty normalisation, anti-memorisation — is the closest published analogue to what RAVANA's reasoning front should become. *Licensing/availability:* repository Apache-2.0 (ARC Prize Foundation, 2019) [8]. *Verdict:* design donor for RSI; do not import pixels.

**Humanity's Last Exam (HLE).** Center for AI Safety, Scale AI & HLE Contributors Consortium (led by Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li; senior authors Summer Yue, Alexandr Wang, Dan Hendrycks), **Nature 649, 1139–1146 (2026)**, DOI 10.1038/s41586-025-09962-4; earlier preprint arXiv:2501.14249 [28]. 2,500 expert-written, closed-ended, automatically gradable questions across dozens of subjects, designed so answers cannot be retrieved quickly; created explicitly because models exceed 90% on MMLU [28]. *Difficulty:* frontier-level. *Suitability:* none directly — multimodal, static, no learning signal; valuable only as documentary proof that saturation drove the field to harder instruments. *Licensing/availability:* publicly released at lastexam.ai; exact dataset licence `[unverified]`.

**HAL.** Interpreted here as **Holistic Agent Leaderboard**: Sayash Kapoor, Benedikt Stroebl, Peter Kirgis, Nitya Nadgir, Zachary S. Siegel, et al., **ICLR 2026**, arXiv:2510.11977 [9][10]. A unified, cost-aware harness for reproducible agent evaluation; validated with 21,730 rollouts over 9 models and 9 benchmarks (coding, web navigation, science, customer service), with automated log analysis and reported cost [9]. *Difficulty:* that of its constituent benchmarks. *Suitability:* high as a **reporting-schema and harness-hygiene donor** (standardised harness, per-benchmark rather than aggregate reporting, cost as a first-class field), low as a task source — it presumes an LLM agent scaffold. *Licensing/availability:* harness public (`princeton-pli/hal-harness`); licence `[unverified]`. **Ambiguity flag:** the bare name "HAL" matched no hallucination benchmark in this search. If the requester meant a hallucination benchmark, the nearest peer-reviewed candidates are HALoGEN [26] and HalluLens [27] (below). This is an identifier ambiguity, not a verified claim about intent. `[unverified: identifier intent]`

**Hallucination peers (for §4).** **HALoGEN** — Abhilasha Ravichander, Shrusti Ghela, David Wadden, Yejin Choi, **ACL 2025 (Outstanding Paper)**, DOI 10.18653/v1/2025.acl-long.71, pp. 1402–1425 [26]: 10,923 prompts over nine domains with automatic high-precision verifiers that decompose generations into atomic facts and check each against a knowledge source; up to 86% of generated atomic facts hallucinated for the best models. **HalluLens** — Yejin Bang, Ziwei Ji, Alan Schelten, Anthony Hartshorn, Tara Fowler, Cheng Zhang, Nicola Cancedda, Pascale Fung, **ACL 2025**, DOI 10.18653/v1/2025.acl-long.1176, pp. 24128–24156 [27]: separates intrinsic from extrinsic hallucination and adds three extrinsic tasks with **dynamic test-set generation** to mitigate leakage. A 2026 successor, **HalluHard** (arXiv:2602.01031, preprint, 950 seed questions over four high-stakes domains, multi-turn) [38], targets multi-turn error propagation. *Suitability:* HALoGEN's atomic verifier and HalluLens's dynamic generation are directly reusable in WFA/PDR; the full suites are not.

### 2.2 Web and agentic search

**SimpleQA.** Jason Wei, Karina Nguyen, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, William Fedus, 2024, arXiv:2411.04368 — **OpenAI technical report, not peer-reviewed** [11]. 4,326 short fact-seeking questions with a single indisputable answer, adversarially collected against GPT-4; reports correct / not-attempted / incorrect and an F-score so guessing is not rewarded [11]. *Successor:* **SimpleQA Verified**, Lukas Haas, Gal Yona, Giovanni D'Antonio, Sasha Goldshtein, Dipanjan Das, arXiv:2509.07968 (preprint, v2 March 2026) [20] — 1,000 prompts after removing ~77% of original items for non-unique sources, near-duplicates, publisher choices and conflicting ground truths; adds acceptable numeric ranges and explicitly measures *parametric* knowledge. *Difficulty:* hard for closed-book LLMs. *Suitability:* the correct/not-attempted/incorrect split is exactly the symmetric abstention guard RAVANA needs; the parametric framing makes it a **pre-test**, not a post-test, for a web-learning system. *Licensing/availability:* OpenAI `simple-evals`, MIT [verified from repository LICENSE].

**BrowseComp.** Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, Amelia Glaese, 2025, arXiv:2504.12516 — **OpenAI technical report, not peer-reviewed** [12]. 1,266 short-answer questions requiring persistent navigation of many sites; easy to verify, deliberately hard to find. *Successors:* BrowseComp-ZH, Peilin Zhou et al., arXiv:2504.19314 (preprint, 289 Chinese multi-hop questions) [37]; MM-BrowseComp (2025, multimodal) was noted in search results but not verified against a primary paper here `[unverified]`. *Suitability:* the *research loop* (persistence across many sources, short verifiable answer) is the right shape for WFA; direct items are too hard and need unrestricted browsing, which conflicts with a frozen, reproducible snapshot. *Licensing/availability:* `simple-evals`, MIT.

**FreshQA / FreshLLMs.** Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, Thang Luong, **Findings of ACL 2024**, arXiv:2310.03214 [17]. 600 natural questions spanning never-changing, fast-changing, slow-changing and **false-premise** categories; two-mode human evaluation (RELAXED/STRICT) for correctness and hallucination over >50K judgements; all models struggled most on fast-changing knowledge and false premises [17]. The dataset is updated weekly by the authors [18]. *Successor:* **SealQA**, Thinh Pham, Nguyen Nguyen, Pratibha Zunjare, Weiyuan Chen, Yu-Min Tseng, Tu Vu, **ICLR 2026**, arXiv:2506.01062 [19] — search results that are conflicting, noisy or unhelpful; Seal-0 gives frontier agentic models 17.1% (o3) and 6.3% (o4-mini), and documents that more test-time compute often *plateaus or declines* with noisy evidence [19]. *Suitability:* **highest** of the web benchmarks — freshness, false premises and noisy/conflicting evidence are precisely RAVANA's failure surface. *Licensing/availability:* FreshQA repository Apache-2.0 [18]; SealQA on Hugging Face (`vtllms/sealqa`) [19]; exact dataset licence `[unverified]`.

### 2.3 Conversational memory across turns

**LoCoMo.** Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, Yuwei Fang, **ACL 2024 (Long Papers), pp. 13851–13870**, arXiv:2402.17753 [13]. Ten retained long conversations (the original 2024 v1 release had 50), ~300 turns and 9K tokens each over up to 35 sessions, annotated for QA (with `question`, `answer`, `category`, `evidence` dialogue ids), event summarisation and multimodal generation [14]. Long-context LLMs and RAG still lag human performance [13]. *Suitability:* **high as a format donor** — evidence ids make answer grading objective and let a harness check whether the system used the teaching turn. The full 1,540-QA set is too large for repeated per-commit runs; sample a fixed stratified slice. *Licensing/availability:* code/data public at `snap-research/locomo` [14]; explicit licence `[unverified]`.

**LongMemEval.** Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu, **ICLR 2025**, arXiv:2410.10813 [15]. 500 curated questions over freely extensible chat histories, testing five abilities: information extraction, multi-session reasoning, temporal reasoning, **knowledge updates**, and **abstention**; commercial systems and long-context LLMs lose 30–60% [15]. Standard settings range from ~115K tokens (S) to ~1.5M (M); authors cleaned histories in 2025 to remove answer interference [16]. *Suitability:* **highest** — knowledge-update and abstention items are the published instruments closest to "did it learn the new fact, and does it still refuse when the fact was never given?" Use the ability taxonomy, not the 1.5M-token setting. *Licensing/availability:* code/data public, repository MIT [16].

**HaluMem.** Ding Chen, Simin Niu, Kehang Li, Peng Liu, Xiangping Zheng, Bo Tang, Xinchi Li, Feiyu Xiong, Zhiyu Li, 2025, arXiv:2511.03506 — **preprint** [23]. Operation-level hallucination benchmark for memory systems: memory extraction, memory updating and memory QA, with ~15K memory points and 3.5K questions; shows hallucinations accumulating at extraction/updating and propagating to QA [23]. *Suitability:* directly relevant to RAVANA's write path — it separates "the fact was never learned" from "the fact was learned wrongly and then reused". *Licensing/availability:* public; licence `[unverified]`.

### 2.4 Poisoning and source trust

**PoisonedRAG.** Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia, **34th USENIX Security Symposium (USENIX Security '25), pp. 3827–3844**, ISBN 978-1-939133-52-6 [24]. First knowledge-corruption attack on RAG: inject a few malicious texts so the system emits an attacker-chosen answer to an attacker-chosen question; **90% attack success with five poisoned texts** in a million-text database, and evaluated defences were insufficient [24]. **Machine Against the RAG**, Avital Shafran, Roei Schuster, Vitaly Shmatnikov, USENIX Security '25 [25], shows a **single** blocker document can suppress an answer entirely and argues jamming resistance is a distinct safety property. *Suitability:* these are the canonical adversarial protocols for PDR; RAVANA's repeated snippet-poisoning incidents make this the least optional component of the new suite. *Licensing/availability:* papers open access; attack code `[unverified]`.

### 2.5 Learning and continual learning (design citations)

Few-shot curves over k [30]; continual learning as joint retention-and-transfer with catastrophic forgetting [32]; protected-parameter mechanisms requiring an on/off ablation [31]. Transfer learning from a source to a lexically disjoint target domain [33] and information-gap theory as the measurable event [34] complete the theoretical set. Retention must be tested with a held-out paraphrase rather than re-reading the taught text, because testing itself improves later retention [29]. Probe accuracy must be checked against a control task the behaviour cannot pass, i.e. report selectivity (real minus control), not raw accuracy [35]. Goodhart: optimising an imperfect proxy eventually reduces the true objective [36]. These are the theoretical load-bearing citations for §3–§6.

### 2.6 Verdict table

| Benchmark | Adopt as data | Adopt design / reporting | Frozen slice only | Do not adopt |
|---|---|---|---|---|
| MMLU-Pro [1][2] | | | ✓ forgetting check | |
| GPQA [3][4] | | | | ✓ |
| SWE-bench / Pro [5][21] | | ✓ hidden-test grading | | ✓ (full repos) |
| ARC-AGI-2 [7][8] | | ✓ few-shot rule induction, exact match | | ✓ (grid format) |
| HLE [28] | | ✓ saturation rationale | | ✓ |
| HAL (agent leaderboard) [9][10] | | ✓ harness/cost schema | | ✓ (LLM scaffold) |
| HALoGEN / HalluLens / HalluHard [26][27][38] | | ✓ atomic verifiers, dynamic sets | | ✓ (full suites) |
| SimpleQA / Verified [11][20] | | ✓ correct/abstain/wrong split | | ✓ (as post-test) |
| BrowseComp / -ZH [12][37] | | ✓ persistent search loop | | ✓ (open web) |
| FreshQA / SealQA [17][19] | ✓ frozen dated slice | ✓ freshness + false premises + noise | | ✓ (live weekly set as score) |
| LoCoMo [13][14] | ✓ fixed stratified slice | ✓ evidence-id grading | | ✓ (full set per commit) |
| LongMemEval [15][16] | ✓ fixed slice | ✓ five-ability taxonomy | | ✓ (1.5M setting) |
| HaluMem [23] | ✓ fixed slice | ✓ extraction/update/QA separation | | |
| PoisonedRAG / JaR [24][25] | ✓ attack recipe | ✓ poison-sweep curve | | |
| LifelongAgentBench [22] | | ✓ task-dependency + replay ablation | | ✓ (needs LLM agents) |

### 2.7 Quarterly reference tracks (not per commit)

Keep the survey's Tier-2 idea but freeze the date: one dated FreshQA slice, one LongMemEval-S slice (≤115K tokens), 50 BIG-bench-style reasoning items, and one hidden-test code item set. Report them with HAL's cost field [9] and HELM-style multi-metric reporting (from the earlier survey). Never fold them into the release headline.

---

## 3. Conversational learning: separating *knowing* from *learning it in this conversation*

A one-shot post-test cannot distinguish (i) a fact already in the concept store, (ii) a fact produced by parroting the teaching turn, (iii) a fact generalised from it, and (iv) a lucky string match. The measurement must therefore contain **four states and a control**:

1. **Pre-test (T0).** Ask a held-out paraphrase question *before* any teaching. A correct answer here means *knowing*, not learning. Record it separately and never count it as gain.
2. **Teaching events (k = 1, 2, 4, 8).** Deliver a fixed, scripted set of causal/definitional turns. The curve is measured at each k [30]; report slope-to-criterion and area under the curve, not only the final point.
3. **Post-test (T1).** Ask a **held-out paraphrase**, never the taught sentence. This is required because retrieval practice changes later retention [29]. The answer must be checked against a structured object (e.g. a provenance-trusted causal chain) or an evidence id, not a prose regex [13][35].
4. **Delayed retest (T2), across sessions and after a cold reload.** This is where a memory claim lives; a gain that vanishes on cold reload is not learning [15][31].
5. **Control group U (untaught).** Same probes, same timing, but teaching turns replaced with matched neutral chatter of equal length. The reportable number is **T − U**, not T. A second control S uses shuffled/permuted teaching turns to detect order-insensitive pattern matching.

**Additional discriminators.** Knowledge-update items (teach X, then teach a corrected X′) test whether the system overwrites rather than accumulates [15][23]. False-premise items test whether it declines a premise it never accepted [17]. Transfer items use a lexically disjoint structural analogue in a fresh engine; report the delta over that fresh engine, not the raw score [33]. Every held-out item is frozen by `sha256("RAVANA-LEARN-v2|<component>|<difficulty>|<subject>")` before any teaching run, following the existing sealed-set procedure, and revealed once per release.

**Instrument validity first.** Before any curve is trusted, run paired known-bad/known-good strings through each rubric and assert bad < good; run a control task the target behaviour cannot pass and assert real − control ≥ threshold; then run the curve [35]. The first run of a new rubric is a draft of the instrument, not a measurement.

---

## 4. Acquiring facts from websites: freshness, attribution, poisoning, disambiguation

**Freshness.** A dated fact must be treated as correct relative to a pinned snapshot date, not to the model's weights. FreshQA is explicitly dynamic and commits to scheduled ground-truth updates [17]; freeze a dated slice plus its update log so a run in 2027 can still be scored against the 2026 ground truth. Never score against the live weekly sheet — that would make the instrument move while the system is held constant.

**Source attribution.** Answer correctness and attribution correctness are **two numbers**. HALoGEN's verifier pattern — decompose the answer into atomic units and check each against a high-quality source [26] — maps cleanly onto RAVANA's provenance-stamped edges: require that the asserted fact carries the real `source_url` and evidence span of a hit that was actually read. An answer with the right string but no trusted provenance is a FAIL. IntentForge's `authority`/`score` fields are ranking signals, not provenance, and must not be accepted as evidence.

**Poisoning.** Build the poison sweep from PoisonedRAG: for each target question, inject 0, 1, 3 and 5 adversarial snippets into the snapshot and report the attacker-chosen-answer acceptance rate at each point, alongside the clean-snapshot accuracy [24]. Add a single blocker document per Machine Against the RAG to test suppression separately [25]. A system that refuses *all* web-derived answers in the poisoned arm is not passing: the pass criterion requires clean accuracy to stay within 0.10 of the no-poison baseline, so "disable the web" cannot win [36].

**Disambiguated definitions.** FreshQA's false-premise category [17] and the in-repo poisoning incidents give the concrete design: pair each fact question with (a) a dictionary-style hit defining a different sense of the subject, (b) a navigation/boilerplate hit, and (c) a genuine multi-step process paragraph. Only (c) may be admitted as a causal chain. Grade separately: (i) correct sense selected, (ii) wrong-sense hit rejected before extraction, (iii) zero edges created for the rejected hit. The repo's own `INFORMATION_NEED_CODEMAP.md` documents that the gap loop currently takes the first two hits verbatim, which is exactly the behaviour this component must expose rather than excuse.

**Offline/web-on parity.** The frozen snapshot is checked in as bytes. `RAVANA_OFFLINE=1` reads those bytes from a local fixture (IntentForge correctly fails closed to no results); the web-on arm calls IntentForge with the pinned URL allowlist and records a `sources` manifest. Identical probe ids, rubrics and seeds; both arms reported as a pair. Parity is a diagnostic: a large OFF−ON gap points at an instrument or capability defect and must be investigated, never scored as a bonus for the arm where a capability was absent [36].

---

## 5. Honest critique of the current 7-front suite

Measured offline at `6a2b5fa`, honest n=60 (30 primary + 30 hash-sealed), mean 56: conversation 93, grounded 17, opinions 87, thinking 8, speaking 7, code 100, learning 80. Per `BENCHMARK_60_ITEMS.md` §5, G6 flagged code as non-informative; overall status `PARTIAL (saturated fronts INVALID)`.

| Front | Score | Diagnosis |
|---|---|---|
| **code generation** | **100** | Saturated. A perfect score is the *loss of a measuring instrument*, not success — the same phenomenon that produced SuperGLUE after GLUE and MMLU-Pro after MMLU [1][2]. It grades string emission, not software engineering: SWE-bench's hidden tests resolved 1.96% at release [5], and SWE-bench Pro still holds frontier models near 23% [21]. Replace with held-out specs graded by hidden tests, and measure acquisition, not one-shot output. |
| **language understanding / conversation** | **93** | Near ceiling; little headroom. It rewards turn-appropriateness and greeting behaviour, not the ability to acquire a fact in the turn. At n=60 a few points are inside the interval; treat as a floor check, not a progress signal. |
| **opinions / stance** | **87** | Proxy. High scores are consistent with emitting stance markers on cue, not with forming and revising a stance under counter-evidence. No counter-attitudinal teaching event, no before/after curve, no control for "always state an opinion". Needs an update-and-revision design. |
| **thinking / reasoning** | **8** | Floor, therefore also non-informative. The documented `that`/`which` causal-regex defect shows the probe can be gamed by surface markers; a fail-closed system that correctly declines to assert an untrusted chain cannot emit those markers and is scored 8. Grade the structured chain object (hops, direction, trusted provenance, load-bearing edges) instead of prose [35]. |
| **speaking / surface realisation** | **7** | Floor. Likely measures clause count/connective presence rather than the capability claimed. A system that honestly answers "outside what I know" is structurally unable to pass a multi-clause causal-realisation rubric. Needs a content-controlled construction test with matched known content, plus a separate abstention credit. |
| **grounded factual answers** | **17** | Genuinely hard, but the wrong instrument. A one-shot QA probe cannot distinguish "term absent from concept store → correct abstention" from "gap detected → web read → evidence learned → answer". The 93/17 split is mostly a *coverage* boundary in the concept store, not a learning result. The web capability is never given a score because the removed bonus only rewarded it being off [36]. |
| **language learning (online)** | **80** | Measures immediate production of a novel symbol after teaching, not retention, savings or transfer. No untaught control, no held-out paraphrase, no delayed/cold-reload retest. A memorised teaching string scores identically to a generalised rule. |
| **MEAN 56** | — | Aggregates a 100 and a 7 into one number, hiding both pathologies and inviting aggregate gaming. Publish all seven; never let a MEAN replace them. Multiplicity across seven fronts × two modes is uncorrected in the headline [35]. |

**Cross-cutting defects.** (1) One-shot format throughout: nothing is measured as a change. (2) No control group anywhere, so selectivity cannot be computed [35]. (3) Two fronts are saturated and two are at the floor — four of seven carry little or no information. (4) The web capability is measured only as an absence. (5) Small-n uncertainty: n=60 gives a Wilson 95% half-width around ±12 points at p=0.5, so most inter-release deltas are not resolvable. (6) The sealed split is a single reveal, which is correct, but a sealed set cannot repair a mis-specified instrument — re-sealing a proxy just replicates the proxy.

**What a tougher replacement looks like:** fewer fronts, more states per item, a control per claim, exact/ structural grading, and a sealed pool per curve. That is RAVANA-LEARN-5.

---

## 6. Recommended new suite: RAVANA-LEARN-5 (v2)

### 6.0 Shared protocol (applies to all five components)

- **Items.** 30 per component: 20 primary + 10 sealed, frozen by the hash rule in §3, stratified across three difficulty bands. 150 items total.
- **Curve points.** k ∈ {0,1,2,4,8} plus a delayed retest after a cold reload. ~150 probes per component per arm; ~1,500 probes for the full two-arm release run (the RAVANA-7 v1 run was 420 probes in ~437 s, so budget roughly 25–30 min on the same class of hardware; `--quick` halves k and the item count for iteration only).
- **Arms.** `OFF` = `RAVANA_OFFLINE=1 PYTHONHASHSEED=0` with the frozen snapshot injected from a checked-in fixture. `ON` = IntentForge provider enabled with a pinned URL allowlist and a recorded `sources` manifest. Both arms always reported; no score component is conditional on a capability being disabled [36].
- **Controls.** U (untaught/neutral chatter), S (shuffled or permuted teaching/demonstrations), C (clean snapshot / no read), P (poisoned snapshot), F (fresh engine, no history). Every reported delta is treatment minus its matched control.
- **Gates.** G1 paired rubric check, G2 selectivity (real − control ≥ 0.30, threshold to ratify), G3 symmetric decline/answer guard, G4 mode parity, G5 census + cold-reload sanity, G6 saturation flag (≥95 or ≤5 is non-informative). Inherited unchanged; a failing gate blocks reporting. [35][36]
- **Statistics.** Benjamini–Hochberg at q=0.05 across the five component-level primary comparisons; report raw and adjusted p. Wilson 95% intervals on every proportion. No non-improvement is dropped from the report.
- **Provenance.** Commit SHA, probe hashes, Python version, `PYTHONHASHSEED`, snapshot bytes hash, URL allowlist and control-arm ids recorded with every summary. One cold boot per arm, no `eng.save()`.

### 6.1 CAR — Conversational Acquisition & Retention

*Design from:* LoCoMo evidence-id QA [13], LongMemEval knowledge-update/abstention taxonomy [15] and few-shot curves [30]. Ebbinghaus savings supply the relearning measure [29].
*Procedure:* pre-test at k=0; scripted teaching events at k=1,2,4,8; post-test with held-out paraphrase at each k; delayed retest after interference and cold reload. 30 subjects, half with a knowledge-update episode (X then corrected X′). Grade the structured chain/evidence object, not prose.
*Controls:* U (neutral chatter of equal length), S (teaching turns permuted), F (fresh engine for transfer analogues).
*Before–after curves:* accuracy vs k, savings S=(L−L_t)/L, delayed retention vs k.
*Pass criterion (proposed):* (a) primary post-test at k=8 minus k=0 ≥ 0.40; (b) taught − U ≥ 0.30 at BH q<0.05; (c) delayed paraphrase retest ≥ 0.60; (d) savings ≥ 0.30; (e) update items: corrected fact asserted in ≥ 0.70 and superseded fact asserted in ≤ 0.10; (f) sealed pool reproduces every sign with no >0.15 drop; (g) zero gain on untaught distractor subjects.

### 6.2 WFA — Web Fact Acquisition & Attribution

*Design from:* FreshQA dated slice and false premises [17], SealQA noisy/conflicting evidence [19] and SimpleQA's correct/not-attempted/incorrect split [11]. HALoGEN atomic-source verification [26] and information-gap theory [34] complete the design.
*Procedure:* pre-test (expect abstain) → mechanism question opens a gap → one web read from the frozen snapshot → post-test on a held-out paraphrase → attribution check (real `source_url` + evidence span). 30 items: 10 fast-changing, 10 noisy/conflicting, 10 false-premise/unanswerable.
*Controls:* C (no read; same probes, snapshot withheld), U (no gap-triggering question), and clean-snapshot arm for parity.
*Before–after curves:* gap-detection rate, answer accuracy and attribution accuracy plotted against number of teaching reads (0,1,2,4).
*Pass criterion:* (a) gap-detection ≥ 0.80; (b) on the clean snapshot, evidence-span answer accuracy ≥ 0.70 and correct-source attribution ≥ 0.80; (c) taught − C ≥ 0.30 at BH q<0.05; (d) abstention on unanswerable items ≥ 0.90 and confident-wrong ≤ 0.10; (e) every atomic asserted fact carries trusted provenance (zero unstamped edges); (f) OFF/ON scores reported as a pair with any discrepancy investigated and logged, not scored as a bonus.

### 6.3 PDR — Poison & Definition-Disambiguation Resistance

*Design from:* PoisonedRAG poison sweep and Machine Against the RAG blocker [24][25]. FreshQA false premises [17] and HaluMem extraction/update/QA separation [23] complete the disambiguation design.
*Procedure:* for each of 30 questions, four snapshot variants with 0, 1, 3, 5 poisoned snippets; a single-blocker variant; and a disambiguation variant containing a same-sense dictionary decoy, a navigation blurb and one genuine process paragraph. Grade three separate outputs: accepted answer, edges written, and sense selected.
*Controls:* C (clean snapshot, otherwise identical) and P (poisoned snapshot) with clean accuracy required to stay high.
*Before–after curves:* attacker-chosen-answer acceptance vs number of poisoned docs; wrong-sense acceptance vs decoy count; edges written after a rejected hit (must stay zero).
*Pass criterion:* (a) poison acceptance ≤ 0.10 at 5 snippets (against 0.90 in the reference attack [24]) and ≤ 0.05 at 1 snippet; (b) blocker-induced abstention ≤ 0.10 on answerable items; (c) false-premise rejection ≥ 0.80; (d) correct sense from process prose ≥ 0.70 with decoys present; (e) zero new edges with missing/untrusted provenance and zero edges from rejected hits; (f) clean-snapshot accuracy within 0.10 of the no-poison baseline (no "disable the web" reward) [36]; (g) sealed items within 0.15 of primary.

### 6.4 CLI — Continual Learning under Interference

*Design from:* LifelongAgentBench task dependency and automatic verification [22] and the A→B→A continual-learning protocol [32]. EWC/sleep ablation [31] and transfer to a lexically disjoint analogue [33] complete the design.
*Procedure:* teach A (30 subjects) → teach B, topically close and definitionally conflicting → retest A → test B′, a lexically disjoint structural analogue in a fresh engine. Repeat with consolidation/sleep on and off. Cold reload before every retest.
*Controls:* F (fresh engine, never taught A or B), S (task order reversed), consolidation-off arm.
*Before–after curves:* A accuracy at A1 (post-teach), A2 (post-B), A3 (delayed); forward transfer B′ − F; savings on relearning A.
*Pass criterion:* (a) forgetting bound A2 − A1 ≥ −0.15; (b) delayed A3 ≥ 0.60; (c) forward transfer ≥ 0.20 over F at BH q<0.05; (d) consolidation-on minus consolidation-off ≥ 0.10 on A3; (e) all gains survive cold reload (G5); (f) no trust or promotion gate is relaxed to obtain (a)–(e) [31].

### 6.5 RSI — Rule Induction / Sealed Transfer

*Design from:* ARC-AGI-2 few-shot rule induction, exact-match grading, sealed evaluation set and resistance to brute force, expressed in RAVANA's verbal causal-chain substrate; GPT-3-style k-curve reporting [7][8][30].
*Procedure:* 30 novel rule descriptions given as k ∈ {0,1,2,4} demonstrations; the system must produce the missing causal chain for an unseen case. Exact structural match on (hops, relation types, direction, intermediate node), not prose similarity. Held-out items are lexically disjoint from all demonstrations. Public ARC-AGI-2 human average on its evaluation set is 66% [8] — a useful difficulty reference, not a target for a 20 W no-LLM system.
*Controls:* S (demonstrations permuted or relations randomised), U (zero demonstrations), F (fresh engine on transfer items).
*Before–after curves:* exact-match accuracy vs k; transfer delta on lexically disjoint items vs k.
*Pass criterion:* (a) k=4 minus k=0 ≥ 0.30 on primary; (b) permuted-demonstration control ≤ U + 0.10 (no order-insensitive pattern matching); (c) held-out exact match ≥ 0.50; (d) transfer delta over F ≥ 0.20; (e) sealed pool within 0.15 and all signs reproduced; (f) zero credit for reproducing a demonstration item verbatim.

### 6.6 What replaces what

| Retired front | Replacement | Reason |
|---|---|---|
| code generation (100) | CLI + hidden-test variant of RSI | hidden functional correctness and a learning curve instead of saturated string emission [5][21] |
| grounded factual answers (17) | WFA + PDR | gap→read→learn measured with provenance and poison sweep, not one-shot coverage |
| language learning (80) | CAR + CLI | untaught control, held-out paraphrase, delay and transfer instead of immediate parroting |
| thinking / reasoning (8) | RSI + CAR's chain grading | structured exact match replaces the connective regex |
| speaking / surface realisation (7) | CAR's post-test content + explicit abstention credit | content-controlled realisation instead of an unreachable clause-count proxy |
| opinions / stance (87) | CAR knowledge-update items extended to stance revision | before/after revision under counter-evidence instead of marker presence |
| conversation (93) | retained as a cheap floor check, folded into CAR's neutral-chatter control | keep continuity without treating a ceiling score as progress |

---

## 7. Operational contract and pre-registration

1. Before the first scored run: freeze the 150 item ids, the hash split, rubrics, seeds, snapshot byte hashes, URL allowlists, thresholds above, and the stopping rule in a reviewed commit. Report every arm including failures.
2. Pin commit SHA, Python version and `PYTHONHASHSEED=0`; one cold boot per arm; no `eng.save()`; no mutation of `weights/ravana_weights.{pkl,db}`.
3. Offline arm: `RAVANA_OFFLINE=1 PYTHONHASHSEED=0 ...`; IntentForge must fail closed to no results. Web arm: IntentForge on with the pinned allowlist and a recorded `sources` manifest.
4. Run G1–G6 first; a failed gate blocks all five scores.
5. Per component, publish one row: `component | n | primary (95% CI) | Δ vs matched control (BH q) | sealed | OFF | ON | energy (J) | status`. No MEAN as a substitute.
6. A component scoring ≥95 or ≤5 on both arms is flagged non-informative and its items refreshed — never its rubric relaxed [1][2].

## 8. Decisions requiring human sign-off

- Ratify the proposed numeric thresholds in §6.1–§6.5 (derived from the cited literature's effect sizes and the repo's n, but engineering choices, not literature constants).
- Approve retiring/reframing the four non-informative fronts and keeping conversation as a floor check only.
- Approve the 1,500-probe release budget or choose the `--quick` variant.
- Choose the energy measurement method (wall meter vs `psutil`) for the reporting column.
- Resolve the "HAL" identifier ambiguity in the task brief: this document assumes the Holistic Agent Leaderboard [9][10].

## 9. Limitations and unverified claims

This is a literature-and-design document: no engine was booted, no proposed component was implemented, and none of the pass criteria has been observed. Full agent suites were judged from their papers and repositories, not executed. The 2026-harvest papers (SimpleQA Verified, HalluHard, SWE-bench Pro V2 and similar) are preprints unless a venue is stated; their baselines are historical context, not current state of the art. LoCoMo's explicit licence, HAL harness licence, HLE dataset licence and SealQA dataset licence were not confirmed `[unverified]`. "HAL" intent, MM-BrowseComp's primary paper, and any claim not carrying a citation above are `[unverified]`. The mapping from `authority`/`score` to ranking-only is an in-repo design rule, not an external benchmark claim. Nothing here weakens G1–G6 or any trust gate.

## Sources

[1] https://arxiv.org/abs/2406.01574 — Wang et al. 2024 MMLU-Pro (NeurIPS 2024)
[2] https://github.com/TIGER-AI-Lab/MMLU-Pro — MMLU-Pro code/data repo (Apache-2.0)
[3] https://arxiv.org/abs/2311.12022 — Rein et al. GPQA (arXiv/CoLM 2024)
[4] https://github.com/idavidrein/gpqa — GPQA repo (CoLM 2024 citation, MIT)
[5] https://arxiv.org/abs/2310.06770 — Jimenez et al. 2024 SWE-bench (ICLR 2024)
[6] https://raw.githubusercontent.com/SWE-bench/SWE-bench/main/LICENSE — SWE-bench repo license (MIT)
[7] https://arxiv.org/abs/2505.11831 — Chollet et al. 2025 ARC-AGI-2 (preprint)
[8] https://github.com/arcprize/ARC-AGI-2 — ARC-AGI-2 data repo (Apache-2.0)
[9] https://arxiv.org/abs/2510.11977 — Kapoor et al. 2026 HAL (ICLR 2026)
[10] https://hal.cs.princeton.edu — Holistic Agent Leaderboard site
[11] https://arxiv.org/abs/2411.04368 — Wei et al. 2024 SimpleQA (OpenAI technical report)
[12] https://arxiv.org/abs/2504.12516 — Wei et al. 2025 BrowseComp (OpenAI technical report)
[13] https://arxiv.org/abs/2402.17753 — Maharana et al. 2024 LoCoMo (ACL 2024)
[14] https://github.com/snap-research/locomo — LoCoMo data/code repo
[15] https://arxiv.org/abs/2410.10813 — Wu et al. 2025 LongMemEval (ICLR 2025)
[16] https://github.com/xiaowu0162/LongMemEval — LongMemEval repo (MIT)
[17] https://arxiv.org/abs/2310.03214 — Vu et al. 2024 FreshQA / FreshLLMs (Findings ACL 2024)
[18] https://github.com/freshllms/freshqa — FreshQA repo (Apache-2.0)
[19] https://arxiv.org/abs/2506.01062 — Pham et al. 2026 SealQA (ICLR 2026)
[20] https://arxiv.org/abs/2509.07968 — Haas et al. SimpleQA Verified (preprint)
[21] https://arxiv.org/abs/2509.16941 — Deng et al. 2025 SWE-bench Pro (preprint)
[22] https://arxiv.org/abs/2505.11942 — Zheng et al. 2025 LifelongAgentBench (preprint)
[23] https://arxiv.org/abs/2511.03506 — Chen et al. 2025 HaluMem (preprint)
[24] https://www.usenix.org/conference/usenixsecurity25/presentation/zou-poisonedrag — Zou et al. 2025 PoisonedRAG (USENIX Security 2025)
[25] https://www.usenix.org/system/files/usenixsecurity25-shafran.pdf — Shafran et al. 2025 Machine Against the RAG (USENIX Security 2025)
[26] https://aclanthology.org/2025.acl-long.71 — Ravichander et al. 2025 HALoGEN (ACL 2025, DOI 10.18653/v1/2025.acl-long.71)
[27] https://aclanthology.org/2025.acl-long.1176 — Bang et al. 2025 HalluLens (ACL 2025, DOI 10.18653/v1/2025.acl-long.1176)
[28] https://doi.org/10.1038/s41586-025-09962-4 — Phan et al. 2026 HLE (Nature 649:1139-1146)
[29] https://journals.sagepub.com/doi/10.1111/j.1467-9280.2006.01693.x — Roediger & Karpicke 2006 testing effect
[30] https://proceedings.neurips.cc/paper_files/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html — Brown et al. 2020 GPT-3 few-shot (NeurIPS 33)
[31] https://www.pnas.org/doi/full/10.1073/pnas.1611835114 — Kirkpatrick et al. 2017 EWC (PNAS)
[32] https://dblp.uni-trier.de/rec/journals/nn/ParisiKPKW19.html — Parisi et al. 2019 continual learning (Neural Networks)
[33] https://www.computer.org/csdl/journal/tk/2010/10/ttk2010101345/13rRUxAAT7Y — Pan & Yang 2010 transfer learning (IEEE TKDE)
[34] https://psycnet.apa.org/record/1994-41058-001 — Loewenstein 1994 information-gap theory
[35] https://arxiv.org/html/1909.03368v1 — Hewitt & Liang 2019 probe selectivity (EMNLP-IJCNLP)
[36] https://proceedings.mlr.press/v202/gao23h.html — Gao et al. 2023 Goodhart (ICML)
[37] https://arxiv.org/abs/2504.19314 — Zhou et al. 2025 BrowseComp-ZH (preprint)
[38] https://arxiv.org/abs/2602.01031 — HalluHard 2026 multi-turn hallucination (preprint)
