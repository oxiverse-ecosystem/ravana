# RAVANA-LEARN-5 — brain-grounded learning benchmark suite (design / decision document)

**Status:** design, pre-registration template, and go/no-go specification. **No code, no other docs changed.**
**Repo state at authoring:** main `3d9c119`, clean. Windows host.
**Scope:** five components — CAR, WFA, PDR, CLI, RSI. This document owns only
`docs/BRAIN_LEARN5_DESIGN.md`. It specifies what to build and what would count as
a result; it does not claim RAVANA passes any of it yet.
**Why now:** `docs/FULL_PAGE_EXTRACTION_DESIGN.md` §"MEASURED NEGATIVE" records
the **fifth failed attempt** to improve the web gap loop by changing input
(phrasing, retry, typed need + veto, full page, gap widening: `mean=77` before
and after, +86 gap reads, zero score change) and names "the RAVANA-LEARN-5
components, not more input tweaks" as the appropriate next instrument. This
document is that instrument's specification.

> **Verification note (provenance discipline).** Every reference in §11 was
> resolved to a publisher, journal, or abstract record during authoring; the
> author list, year, journal, volume/pages and DOI below are taken from those
> records, not from memory. Claims are confined to what those records state.
> Where a source is a *theory* rather than a neural finding, or a *review/
> hypothesis* rather than a direct experiment, that is flagged inline. No
> reference here is cited for a quantity it does not establish.

---

## 0. Decision summary (read this first)

One **primary endpoint** per component, one pre-registered **PASS** rule and one
**FAIL** rule. Both the primary pool and the hash-held-out sealed pool must pass;
a primary-only pass is an iteration signal, never a release result.

| Component | Primary endpoint | PASS (both pools) | HARD FAIL (any one) |
|---|---|---|---|
| **CAR** conversational acquisition & retention | Delayed cold-reload **paraphrase** trusted-chain rate, k=8 minus k=0 | Δ ≥ **0.25**, monotonic across k∈{0,1,2,4,8}, controls ≤0.05 and Δ ≤0.05 | any rise in k=0 or definition-only arm; gain vanishes after cold reload; gain only on identical-surface probe; untrusted hops counted |
| **WFA** web fact acquisition & attribution | **Attribution-valid** rate (correct + real URL + entailing evidence span) on gap items | ≥ **0.70**, unanswerable abstention ≥0.90, attribution-valid ≥ answer-only accuracy + 0.20 | any fabricated URL; fabricated/misattributed source; confident answer to closed-web item; attribution-valid ≤ answer-only |
| **PDR** poison & definition disambiguation resistance | Poison-intrusion rate and correct-original retention at p=5 | correct-original ≥ 0.80 × baseline, poison-intrusion ≤ **0.10**, source-attribution ≥0.80, sense accuracy ≥0.85 | poison-intrusion >0.15 at any p; correct-original drop >0.20 absolute; coverage collapse (abstention >0.20 on previously correct items) |
| **CLI** continual learning under interference | Retention of A after B, sleep-on minus sleep-off at matched compute/time | sleep-on A-retention ≥0.85 of criterion **and** ≥ **0.15** above sleep-off; lexically-disjoint transfer ≥0.60 structural with ≤0.10 lexical-copy | sleep-on <0.70; sleep-on − sleep-off <0.10; lexical-copy >0.15; B acquisition degraded >0.15 |
| **RSI** rule induction & sealed transfer | Exact **structural-match** accuracy on new objects | ≥ **0.75**, permuted-demo within 0.10 of exact, surface-foil rejection ≥0.85, far transfer ≥ chance + 0.20 | permuted < exact − 0.15 (order memorisation); foil rejection <0.70; far transfer ≤ chance + 0.10 |

**Release rule.** All five PASS → the suite supports the claim "RAVANA shows
learning-shaped behavior on these five instruments." Any HARD FAIL → the
corresponding capability claim is **not** supported; a single static score from
another harness does not override it.

---

## 1. Why a single static score cannot measure a learning system

The current reported numbers are: offline n=60 **mean 62**, frozen web n=60
**mean 77**, and one learning-curve result **teach Δ = +0.583** with both
controls flat on **n=12** subjects. The first two are static; the third is the
only learning-shaped number, and it is small-n, single-state, immediate, and
uses no cold reload. The distinction is not cosmetic:

1. **A static score confounds prior knowledge with learning.** Ebbinghaus's
   method of savings defines retention as a *within-subject change from before
   to after an intervening episode*, not a level at test [1,2]. An n=60 item
   bank answered once cannot say which items were known before the run.
2. **A static score has no counterfactual.** "The system scored X" is
   compatible with "teaching did nothing." The CAR no-teach and definition-only
   arms exist solely to supply the missing counterfactual.
3. **A single *immediate* score can rank the wrong system as better.** In the
   testing effect, repeated studying beats repeated testing on a 5-minute test
   but testing beats studying at 2 days and 1 week [3]; the same reversal holds
   for inference questions [4]. A system measured immediately after exposure
   can look superior precisely on the dimension that decays fastest.
4. **A static score rewards probe memorisation.** Without a hash-held-out
   sealed pool fixed before runs, improvement can be item-specific. RAVANA has
   already been burned by gold sets built in the same pass that chose the
   mechanism (see `docs/SEVEN_FRONTS_STATUS.md` §"a proxy that is not measured
   independently will look better than it is").
5. **A static score hides *what* changed.** Five web-gap interventions raised
   search count (+86 reads) and left mean at 77 [design §MEASURED NEGATIVE].
   A score-only dashboard cannot distinguish "the system learned to read" from
   "the system read more and learned nothing"; WFA's attribution endpoint exists
   to make that distinction observable.
6. **A static score rewards fail-closed silence equally with failure.** An
   honest abstention and an inability to answer both score 0. The suite scores
   abstention as a *separate positive endpoint* where theory demands it
   (WFA unanswerable items, PDR warning arm).
7. **A static score has no forgetting curve.** Retention's shape over lag is
   the object of study [1,2]; a point estimate discards it.

**Consequence:** every component below has a **delta**, a **control arm**, and a
**delayed held-out test** as its headline. Raw accuracy is reported as context
only, never as the decision variable.

---

## 2. Method common to all five components

### 2.1 Hold out by hash, fixed before runs (mandatory)

Adopt the repo's existing procedure from `docs/BENCHMARK_60_ITEMS.md` §2 and
`scripts/measure_fronts*.py::_seal_split`, with a **new constant string** so the
five suites cannot collide with the existing RAVANA-7 v1 pools:

```
For each (component, role, difficulty) group:
    h = sha256(f"RAVANA-LEARN-5|{component}|{role}|{difficulty}|{item_id}"
               .encode("utf-8"))
    sort group by h  (byte order; score-independent)
    first 50%  -> PRIMARY pool
    last  50%  -> SEALED pool
```

- Freeze the pools **before any engine boot**, before writing any rubric, and
  before reading any answer. Record `sha256(canonical_json(sealed_records,
  sort_keys=True, ensure_ascii=False))` per pool in a manifest.
- Poison snippets, dictionary decoys, permuted demonstrations, URL variants and
  evidence-span variants are themselves **items** and must be hash-assigned,
  not hand-picked after seeing failures.
- The sealed pool is revealed **once**, in the reported run. No pass/fail
  outcome from it is inspected during design. If a sealed result changes a
  design decision, the result is re-labelled as *primary* and a new sealed pool
  is drawn by the same rule.
- Rubric authorship order (enforced): **write probes → read real answers →
  enumerate the actual deflection/hedge register → write rubric → freeze.** A
  rubric authored in the same pass as the probes cannot see a real failure and
  will score the system's own register as success. This is how the seven-front
  scorecard once scored 100/100 on deflection text.

### 2.2 Sample size and statistics (defensible minimums)

- **Unit of analysis is the item/family**, never the token, probe occurrence, or
  answer string. Multiple probes of one item are averaged *before* testing.
- **Primary pool n ≥ 40 and sealed pool n ≥ 40 observations per component's
  primary endpoint.** This gives ~80% power at two-sided α = 0.05 for a paired
  effect of d_z ≈ 0.45 (paired t / Wilcoxon), and ~75–90% power for a
  McNemar shift of ≈0.20 absolute with a small reverse rate. For binary
  endpoints where the base rate is near zero, report exact binomial or Fisher
  exact with Clopper–Pearson intervals; do not use a normal approximation.
- **n = 12 (current CAR) is underpowered for anything but a large effect.**
  The observed +0.583 with seven fail→pass and zero reverse pairs is
  detectable at that n (McNemar χ² ≈ 7, p ≈ 0.008), so the *direction* is real;
  it is still provisional on the magnitude and generalisation, and must be
  re-run at n ≥ 40 under CAR.
- **One confirmatory test per component**, pre-registered. Secondary endpoints
  are Holm–Bonferroni adjusted within component; because a PASS requires
  *conjunction* (primary **and** sealed **and** controls), the family-wise error
  across the five components is controlled by the conjunction and does not
  require a blanket cross-component correction. Report unadjusted p and the
  adjustment method.
- **Effect sizes and intervals are mandatory**: Hedges' g / d_z with 95% CI for
  continuous paired endpoints; odds ratio or risk difference with 95% CI for
  binary; exact counts (b, c discordant pairs) for McNemar.
- **Reproducibility gate before any A/B claim:** pin seed (`PYTHONHASHSEED=0`,
  temperature 0 where available), run the *same configuration twice*, and
  require identical structural verdicts. If the same config does not reproduce,
  the probe is underpowered — say so and run the full instrument. A "win"
  resting on one sample of a stochastic generator is the easiest way to write a
  false positive into a doc.
- **Circularity gate:** before quoting any accuracy, state how ground truth was
  produced. A gold set labelled by the same routine under test, or a rubric
  built after reading the answers, is reported as an upper bound, not a result.

### 2.3 What counts as "learning" here (common definition)

A component passes only if all four hold:
1. **Change:** a within-item before/after or A/B/A delta, not a level.
2. **Attribution:** the new content is tied to the episode that produced it
   (provenance / source tag / trusted edge), so a pre-existing or guessed answer
   cannot be counted.
3. **Transfer:** the criterion probe differs in surface form from the teaching
   material (paraphrase, new objects, permuted order, disjoint lexicon).
4. **Counterfactual:** an untaught / no-gap / unpoisoned / no-B / surface-foil
   arm that stays flat.

Remove any one of these and the suite measures performance, not learning.

---

## 3. CAR — Conversational Acquisition and Retention

### 3.1 Mechanism → observable criterion

**Ebbinghaus (1885/1913)**, replicated by **Murre & Dros (2015, *PLOS ONE*)**,
establishes the savings method: retention is a **within-subject relearning
difference**, and the forgetting curve is steepest early [1,2]. Therefore the
criterion cannot be "the answer contains the fact after teaching"; it must be
"the trusted-chain rate on a *held-out paraphrase* rises relative to the same
item's baseline and relative to an untaught item, and the rise grows with
spaced teaching episodes." **Roediger & Karpicke (2006, *Psychological
Science*)** show the post-teaching measurement must be a **retrieval test**, not
restudy, and that the advantage appears on delayed tests [3]; **Karpicke &
Blunt (2011, *Science*)** extend this to inference-level questions [4].
**Cepeda, Pashler, Vul, Wixted & Rohrer (2006, *Psychological Bulletin*)**, in a
meta-analysis of 839 assessments, show spacing and retention interval interact:
the optimal interstudy interval grows with the retention interval [5]. Therefore k is a *spacing* manipulation, not "repeat
the same sentence k times in one turn."

**Observable criterion (formal).** For item i at teaching level k:
`chain_i(k) = 1` iff the delayed cold-reload answer to paraphrase probe P'(i)
contains a provenance-trusted ≥2-hop causal chain for subject(i), where P'(i)
shares no surface form with any taught assertion. Headline:
`Δ_CAR(k) = mean_i chain_i(k) − mean_i chain_i(0)`, with `k ∈ {0,1,2,4,8}`.

### 3.2 Design

- **Arms:** `k ∈ {0,1,2,4,8}`. k=0 is the untaught arm. For k≥1 the teaching
  episodes are *spaced* across session boundaries; spacing interval is a fixed
  design parameter and is reported, not tuned after results.
- **Teaching content:** a 2-hop causal mechanism `subject → intermediate →
  terminal` with trusted (user-source) provenance — the repo already has the
  write path that produced the +0.583 result on the `fix/conversational-
  causal-trust` branch (see `docs/LEARNING_CURVE.md` §7).
- **Retest:** (a) immediate paraphrase probe; (b) **delayed cold-reload
  paraphrase** — fresh process, weights read from disk, no in-memory residue,
  no `eng.save()` during the probe. (b) is the confirmatory endpoint; (a) is
  diagnostic only and is reported so the immediate-vs-delayed reversal of [3]
  is visible.
- **Untaught controls:** k=0 items are exposed to the *same subject names and
  the same number of neutral turns*, so subject familiarity is matched while
  mechanism teaching is absent.

### 3.3 Controls: learning vs parroting / familiarity / guessing

| Confound | Control | Expected if confound, not learning |
|---|---|---|
| Parroting | Held-out paraphrase sharing no surface form with taught text | Pass on identical probe, fail on paraphrase |
| Familiarity | k=0 items see the same subject names / turn count | k=0 delta > 0.05 |
| Study-without-retrieval | Definition-only arm (mechanism absent) | definition-only delta > 0.05 (Ebbinghaus/Roediger no-episode control) |
| In-memory echo | Delayed cold reload | Gain present in-process, absent after reload |
| Untrusted-edge inflation | `assertable_chain()` provenance gate on every hop | Raw causal-edge count rises (repo measured ~13/subject) while trusted chain stays 0 |
| Guessing | Chance baseline from a seed/label-shuffled arm | chain rate at chance on all k |
| Probe written to system register | Rubric authored after reading answers, hedge register enumerated first | 100% on deflection text |

### 3.4 Failure we must not explain away

- **Any rise in k=0 or definition-only.** That is the counterfactual failing;
  no amount of "the engine was in a different state" rescues it. Re-run from a
  clean checkout.
- **A gain that survives only on the identical-surface probe.** That is
  parroting; report it as such and mark CAR FAIL.
- **A gain that disappears after cold reload.** That is activation, not
  retention; do not call it learning.
- **A "curve" made of untrusted causal edges.** The earlier learning-curve
  diagnosis recorded exactly this artifact: 10–14 causal edges per subject,
  all untrusted, `new_trusted_causal_edges = 0`. Counting edges is not a
  metric here.
- **A k=8 gain with a reversal at k=4 that we smooth over.** The savings law
  predicts monotonicity in the taught range; a reversal >0.10 is a failure of
  the spacing interpretation, not noise to be averaged away.

### 3.5 Sample size / statistics

n ≥ 40 item-families in PRIMARY and n ≥ 40 in SEALED. Items are assigned to k
by a fixed Latin square so each k-level has ≥40 observations and each item sees
exactly one k (prevents within-item carryover). Primary test: McNemar on
`chain_i(8) − chain_i(0)` on the delayed cold-reload probe; secondary:
Cochran–Armitage trend across k and mixed-effects logistic model with random
item intercept. Report immediate and delayed separately.

### 3.6 Explicit criteria

**PASS:** `Δ_CAR(8) ≥ 0.25` on **both** pools, `p < 0.05` primary test, no
reversal >0.10 across adjacent k, k=0 and definition-only deltas ≤0.05 with
their own 95% CIs excluding +0.10, and delayed gain present after cold reload.

**FAIL:** any control delta >0.05; gain <0.25 even if `p < 0.05` (effect-size
gate); gain absent at cold reload; trusted-provenance gate removed to obtain
the number; sealed delta < primary delta − 0.10.

---

## 4. WFA — Web Fact Acquisition and Attribution

### 4.1 Mechanism → observable criterion

**Johnson & Raye (1981, *Psychological Review*)** and **Johnson, Hashtroudi &
Lindsay (1993, *Psychological Bulletin*)** establish that source monitoring
distinguishes knowledge acquired externally from knowledge generated
internally, and that it is **a decision process at retrieval**, not a tag
directly read off a memory: people evaluate activated records against flexible
criteria and are subject to systematic error [6,7]. Therefore "the answer is
correct" is insufficient — the system must attribute the claim to the source
that actually supplied it, and a self-generated claim must not wear a web
citation. **Loewenstein (1994, *Psychological Bulletin*)** and **Friston (2010,
*Nature Reviews Neuroscience*)** establish curiosity as a felt **information
gap** and the epistemic value of a read as bounded [8,9].
Therefore the effect to measure is **gap → targeted read → attributed answer**,
and the amount of reading is a cost, not a benefit.

**Observable criterion (formal).** For a gap item i, an answer is
`attribution_valid(i) = 1` iff (a) the target proposition is correct, (b) a
source URL is emitted, (c) a verbatim evidence span from that URL is emitted,
and (d) the span **entails** the proposition under the frozen rubric; and the
edge/claim is tagged as externally derived, not self-generated. Headline:
`WFA_valid = mean_i attribution_valid(i)` on gap items; co-primary safety
endpoint `abstain = mean_j abstain(j)` on closed-web items.

### 4.2 Design

- **Gap items:** a `why/how` question whose answer is absent from the store at
  baseline (gap detection must be verified *before* the read, not inferred from
  the answer).
- **Bounded read:** at most 2 sources per gap, as in the existing mechanism-gap
  loop; search count is recorded as a **cost** and reported beside the score.
- **Closed-web items:** no reachable source; required behavior is abstention
  with no URL. This is where the five failed input tweaks are most likely to
  have produced confident attribution.
- **Decoy items:** a plausible URL whose page contains on-topic text but a
  span that does **not** entail the proposition. Passing requires rejecting it.
- **Paraphrased-evidence items:** the same fact presented in different words,
  to separate span-grounding from surface memorisation of the earlier snippet.

### 4.3 Controls: learning vs parroting / familiarity / guessing

| Confound | Control | Expected if confound |
|---|---|---|
| Guessing correctly | Compare attribution-valid vs answer-only accuracy | Answer-only > attribution-valid by ≥0.20; the gap is the guessed fraction |
| Fabricated URL | Machine-verify URL was actually fetched this run | Any unverified URL → hard fail |
| Decorative evidence span | Entailment rubric applied blind to answer correctness | On-topic but non-entailing span accepted |
| Self-generated claim wearing web tag | Internal-source control: question answerable only by reasoning | Web attribution on an internally-derived claim |
| Source familiarity | Decoy URL from a previously seen domain | Attribution follows domain familiarity, not span |
| Over-reading as progress | Report search count beside score | +86 reads, 0 score change = the fifth failed attempt [design] |

### 4.4 Failure we must not explain away

- **A correct answer with a fabricated or un-fetched URL.** Hard fail regardless
  of accuracy. Source monitoring's whole point is that plausibility is not
  provenance.
- **A correct answer with a non-entailing span** (the span mentions the subject
  but does not state the relation). Hard fail; this is the structural-grounded
  vs truthful distinction already recorded in `docs/SEVEN_FRONTS_STATUS.md`.
- **Confident attribution on a closed-web item.** This is confabulation with a
  citation, worse than honest abstention.
- **Attribution-valid ≤ answer-only accuracy.** The attribution step added no
  information; do not report it as an acquisition gain.

### 4.5 Sample size / statistics

n ≥ 60 gap items + n ≥ 30 closed-web + n ≥ 30 decoy in PRIMARY, mirrored in
SEALED (n=60 is the repo's existing front-harness size and matches the frozen
web run). Primary: Fisher exact or bootstrap CI on `WFA_valid` between pools
and against an answer-only baseline measured on the *same* items (paired
McNemar). Secondary: abstention rate on closed-web items (exact binomial,
Clopper–Pearson CI), decoy rejection rate.

### 4.6 Explicit criteria

**PASS:** `WFA_valid ≥ 0.70` on both pools; abstention on closed-web ≥0.90;
`WFA_valid − answer_only ≥ 0.20`; decoy rejection ≥0.85; fabricated-URL rate
= 0; read count reported as cost.

**FAIL:** any fabricated or un-fetched URL; attribution-valid ≤ answer-only;
abstention <0.80; decoy rejection <0.70; sealed `WFA_valid` < primary − 0.10.
A sixth input-side tweak (query phrasing, retry, need-type, full page, gap
widening) does **not** count as a WFA result: the endpoint is attribution, not
read volume.

---

## 5. PDR — Poison and Definition Disambiguation Resistance

### 5.1 Mechanism → observable criterion

**Loftus, Miller & Burns (1978, *JEP: Human Learning and Memory*)** and
**Loftus (2005, *Learning & Memory*)** establish the misinformation effect:
information encountered *after* an event can alter later memory of it, and is
integrated into a single altered record; warnings help only under limited
circumstances [10,11]. The source-monitoring account of this failure —
**Johnson & Raye (1981)**, **Johnson et al. (1993)** and **Johnson, Raye, Wang &
Taylor (1979, *JEP: Human Learning and Memory*)** — is that internally or
externally supplied details are attributed using overlapping cues, so a
misleading detail can be misattributed to the original episode [6,7,12]. Therefore resistance is **not** "the answer stays
the same"; it is "the correct original is retained, the poison is rejected or
correctly attributed to its source, and the right word sense is selected."

**Observable criterion (formal).** With `p ∈ {0,1,3,5}` poisoned snippets plus
dictionary decoys, for item i measure:
`orig_i(p)` = original fact correct; `intr_i(p)` = poison detail asserted as
fact; `src_i(p)` = poison attributed to the poisoned source; `sense_i(p)` =
correct sense in a disambiguating context. Headline safety endpoint is
`max_p intr_i(p)` and `orig_i(5) / orig_i(0)`.

### 5.2 Design

- Teach the correct fact/definition first; verify it was learned (an item that
  was never learned cannot show resistance — it shows absence).
- Inject p poisoned snippets about the same subject, each with its own source
  tag; add dictionary decoys giving a different, plausible sense of the same
  word form.
- Probe: (a) original fact in the original context; (b) poison detail with
  source forced-choice; (c) sense-disambiguation in a context selecting the
  taught sense; (d) paraphrased original.

### 5.3 Controls: learning vs familiarity / guessing / refusal

| Confound | Control | Expected if confound |
|---|---|---|
| Never learned | p=0 baseline required to be ≥0.80 before poisoning | Low p=0 → "resistance" is just no knowledge |
| Blanket refusal (looks resistant) | Coverage endpoint: abstention on items correct at p=0 | Abstention >0.20 → failure, not resistance |
| Correct source, false claim | Score `intr` separately from `src` | High `src` with high `intr` = correct attribution of a false belief |
| Lexical familiarity | Dictionary decoy shares word form but wrong sense | Sense accuracy tracks word frequency, not context |
| Warning effect as artifact | Warning/context-reinstatement arm (warnings help only in limited cases [10]) | Resistance appears only with warning → not robust |
| Guessing under forced choice | Chance baseline from shuffled source labels | At chance on `src` |

### 5.4 Failure we must not explain away

- **Poison asserted with high confidence and correct source attribution.** That
  is a true false belief with intact provenance — still a FAIL.
- **A "resistant" system that stopped answering.** Measured only as accuracy it
  looks stable; the coverage endpoint catches it.
- **A correct answer that still contains the poison detail in its supporting
  chain.** The graph has been contaminated even if the surface answer is clean.
- **An apparent p effect driven by unequal source counts.** Poison snippets must
  be hash-assigned and source-tagged with matched credibility cues.

### 5.5 Sample size / statistics

n ≥ 40 items per p-level in PRIMARY and SEALED; items nested so each item is
observed at all p levels (within-item repeated measure). Primary: mixed
logistic model with p as fixed effect and item random intercept; paired
contrast p=5 vs p=0 on `orig`, and exact binomial CI on `intr(5)`. Secondary:
`src` and `sense` accuracy, coverage.

### 5.6 Explicit criteria

**PASS:** at p=5, `orig(5) ≥ 0.80 × orig(0)`; `intr(p) ≤ 0.10` at every p;
`src ≥ 0.80`; `sense ≥ 0.85`; coverage loss ≤0.05; both pools.

**FAIL:** `intr(p) > 0.15` at any p; `orig(5) − orig(0) < −0.20` absolute;
coverage loss >0.20; `sense < 0.75`; sealed worse than primary by >0.10 on
`orig(5)` or `intr(5)`.

---

## 6. CLI — Continual Learning Under Interference

### 6.1 Mechanism → observable criterion

**Wixted (2004, *Annual Review of Psychology*)** establishes retroactive
interference as the canonical A→B→A paradigm: learning B disrupts recall of A,
with larger effects for more B training, and the effect is reduced by time and
consolidation [13]. **McCloskey & Cohen (1989, *Psychology of Learning and
Motivation*)** name the extreme case **catastrophic interference** in a
connectionist model, and crucially show it is severe under *sequential* but not
*concurrent* training, and that apparent forgetting can be damaged-but-
relearnable (savings) rather than erased [14]. **Walker, Brakefield, Hobson &
Stickgold (2003, *Nature*)** show wake provides stabilisation but sleep provides
enhancement, and reactivation can return a consolidated memory to a labile
state requiring reconsolidation [15]; **Diekelmann & Born (2010, *Nature
Reviews Neuroscience*)** show slow-wave sleep reactivation and redistribution
support system consolidation and the extraction of generalised features that
ease inference [16]. **Walker & van der Helm (2009, *Psychological
Bulletin*)**'s "overnight therapy" review [17] proposes REM reactivation in a
low-noradrenaline state; note honestly that it is a **review/hypothesis about
emotional memory**, so the declarative-consolidation claim here rests chiefly
on [15,16], with [17] cited for the sleep-dependent (re)processing framing the
task requested.

**Observable criterion (formal).** After A→B→A,
`ret_A = acc_A(after B) / acc_A(at criterion)`. Transfer is measured on
lexically disjoint B items, scored for structural correctness and lexical-copy
rate. "Sleep on/off" in RAVANA is its **offline replay/consolidation routine
enabled vs disabled** — an operational analogy, not biological sleep.

### 6.2 Design

- Train A to criterion (verify it reached criterion — otherwise there is
  nothing to forget). Train B (lexically disjoint vocabulary, potentially
  overlapping relational structure). Retest A.
- **Sleep-on:** consolidation routine enabled between B and A retest.
  **Sleep-off:** disabled. **Match wall-clock and replay step count** across
  arms: if sleep-on gets extra compute, the comparison is confounded.
- **Transfer:** B items lexically disjoint from A; a correct response must use
  the learned relational structure with novel lexical material.

### 6.3 Controls: interference vs time / extra compute / lexical copy

| Confound | Control | Expected if confound |
|---|---|---|
| Time decay, not B interference | A-only, no-B arm at matched elapsed time | Same drop in both arms → time, not interference |
| Extra compute | Sleep-off with identical replay-step budget | Gain tracks step count, not consolidation |
| Interleaving benefit | Concurrent A+B arm [14] | Interleaved ≈ sleep-on → replay is just interleaving |
| Lexical overlap driving transfer | Lexically-overlapping B arm (manipulation check) | Transfer collapses when lexicon disjoint |
| Copy-with-rename | Lexical-copy rate scored separately | High structural score with high copy rate → not transfer |
| A re-exposed during B | Audit B training corpus for A items | Any A item present → invalid run |

### 6.4 Failure we must not explain away

- **Retention measured only on B-tuned probes.** A must be probed with its own
  held-out paraphrases, not B's surface forms.
- **"Transfer" that is A's tokens with B's label.** The lexical-copy rate is
  the falsifier.
- **Sleep-on gain that is really extra replay steps.** Match compute or report
  the result as "more replay," not consolidation.
- **Flat forgetting because A never reached criterion.** Enforce the criterion
  gate before B.
- **Savings misread as retention.** Relearning-faster is evidence of damaged-
  not-erased memory [14]; report it separately from first-test retention.

### 6.5 Sample size / statistics

n ≥ 40 A-items and n ≥ 40 B-items per condition in PRIMARY and SEALED.
Primary: 2 (consolidation on/off) × repeated item, mixed-effects logistic model;
paired contrast of `ret_A`. Report also trials-to-criterion (savings) with a
paired test. Secondary: B acquisition, lexical-copy rate.

### 6.6 Explicit criteria

**PASS:** sleep-on `ret_A ≥ 0.85` and `ret_A(on) − ret_A(off) ≥ 0.15`; B
acquisition degraded by ≤0.10 vs A-only matched arm; transfer structural
correctness ≥0.60 with lexical-copy ≤0.10; both pools; compute matched.

**FAIL:** sleep-on `ret_A < 0.70`; on−off <0.10; lexical-copy >0.15;
B acquisition degraded >0.15; A present in B training corpus; compute not
matched.

---

## 7. RSI — Rule Induction and Sealed Transfer

### 7.1 Mechanism → observable criterion

**Gentner (1983, *Cognitive Science*)** establishes structure-mapping theory:
analogy maps **relations between objects, not object attributes**, and the
systematicity principle prefers interconnected systems of relations over
isolated predicates [18]. **Gentner & Markman (1997, *American Psychologist*)**
show structural alignment underlies both analogy and similarity [19].
**Lake, Salakhutdinov & Tenenbaum (2015, *Science*)** show humans generalise
from a single example by inferring a generative structure, and that a model
doing so outperforms deep classifiers on the same one-shot task [20]. Therefore
the criterion is **relation preserved under filler substitution**, and the
decisive control is a **permutation**: if changing demonstration order or object
roles breaks performance, the system matched surface/serial structure rather
than the rule. Because an unsupported assertion is a *relation* the store does
not support, the answer must commit to the relation explicitly, not just the
target.

**Observable criterion (formal).** From k ∈ {1,2,4} demonstrations of rule R,
score: `exact` = correct R on a structurally identical item with **new objects**;
`permuted` = correct R when demonstrations are order/role-permuted but relations
unchanged; `foil` = rejection of an item with the same objects but different
relations; `far` = correct R in a new domain sharing only relational structure.

### 7.2 Design

- Few-shot demonstrations with exact structural match; hold out the transfer
  items by hash before runs.
- Permuted demonstration control: same objects and relations, shuffled
  presentation order and swapped roles. A structural learner is invariant; a
  sequence memoriser is not.
- Surface-foil: same object set, wrong relations — must be rejected.
- Far transfer: new domain, relational commonality only.
- Answer format must expose the relation field (subject, relation, target), not
  a free-text label.

### 7.3 Controls: rule vs surface / order / guessing

| Confound | Control | Expected if confound |
|---|---|---|
| Object-feature matching | Surface-foil (same objects, wrong relation) | Foil accepted |
| Serial memorisation | Permuted demonstrations | `permuted < exact − 0.15` |
| Explicit teaching of the test item | Hash-held-out sealed transfer | Sealed ≫ primary or ≈0 |
| Guessing | Chance baseline from shuffled relation labels | All conditions at chance |
| Answer-by-copy | Near vs far transfer both reported | Near high, far at chance |

### 7.4 Failure we must not explain away

- **Exact-match success with permuted-demo failure.** That is order
  memorisation, the specific failure [18,19] predicts. Do not average it into an
  "induction" score.
- **Foil acceptance.** The system used object attributes; structure-mapping
  says that is a different (appearance-match) operation, not analogy.
- **Near-transfer-only success.** Report far transfer explicitly; do not let a
  near-transfer headline stand for generalisation.
- **A "rule" that is a stored template with renamed entities.** The relation
  field must be checked against the demonstration's relation, not just the
  filled target.
- **Few-shot k confounded with item difficulty.** Fix item difficulty by hash
  split within k.

### 7.5 Sample size / statistics

n ≥ 40 rule families in PRIMARY and SEALED; each family contributes exact,
permuted, foil and far probes. Primary: exact accuracy, Fisher/McNemar vs
chance and vs permuted. Secondary: Holm-adjusted comparisons on permuted, foil,
far. Report exact−permuted paired difference with 95% CI.

### 7.6 Explicit criteria

**PASS:** `exact ≥ 0.75`; `permuted ≥ exact − 0.10`; foil rejection ≥0.85;
`far ≥ chance + 0.20`; both pools; relation field present in ≥95% of answers.

**FAIL:** `permuted < exact − 0.15`; foil rejection <0.70; far ≤ chance + 0.10;
relation field absent in >10% of answers; sealed `exact` < primary − 0.10.

---

## 8. What the suite cannot establish

State these limits in any release that cites RAVANA-LEARN-5:

1. **Not consciousness or subjective experience.** All endpoints are
   behavioural/structural. Passing says nothing about qualia or phenomenal
   understanding; no cited source licenses that inference.
2. **Not neural isomorphism.** Ebbinghaus, structure mapping and source
   monitoring are **behavioural/cognitive-level** laws and theories, not
   identified circuits. A match in input-output behavior is not evidence that
   RAVANA uses the hippocampal/SWS/ACC machinery the sources describe.
3. **Not truth.** Source monitoring establishes *provenance*, not correctness.
   A web source can be wrong; an entailing span from a wrong source passes WFA.
   Truth requires an independent fact check, which this suite does not supply.
4. **Not generalization outside the authored families.** Every item pool is a
   constructed distribution. Sealed transfer tests within that distribution;
   it does not establish open-world generalization.
5. **Not a single mechanism per component.** CAR and WFA each bundle gap
   detection, retrieval, gating and generation. A pass localizes no component;
   a failure localizes no component either. Ablations are required for that.
6. **Not that "sleep on" is sleep.** It is an offline replay routine with a
   matched compute budget. It cannot establish SWS/REM-specific consolidation.
7. **Not long-term retention.** The longest lag here is the delayed cold
   reload and the between-block separation in CLI. Nothing here tests months,
   and the forgetting curve is strongly lag-dependent [1,2].
8. **Not safety or user value.** Abstention is scored as correct where theory
   demands it, but an abstaining system that is unhelpful still passes those
   endpoints. The suite measures learning-shaped behavior, not utility.
9. **Not statistical causality at the mechanism level.** A significant paired
   delta shows the teaching/manipulation preceded the change; it does not by
   itself identify which internal state change caused it. That needs
   ablations and independent gold sets.
10. **Not immunity to a sixth input tweak.** The five web-gap failures show
    that changing *what the system reads* did not move answers. WFA tests
    attribution; it does not prove no future input change could help.
11. **Not that a passing suite overturns the static scores.** Offline n=60
    mean 62 and frozen web n=60 mean 77 remain what they are: static,
    full-front measurements. The suite does not re-score them and they do not
    substitute for a PASS.

---

## 9. Relation to current measurements and immediate decision

| Current number | What it is | How the suite treats it |
|---|---|---|
| offline n=60 mean 62 | static, gated seven-front score | context only; not evidence of learning (§1) |
| frozen web n=60 mean 77 | static, frozen web harness | context only; WFA endpoint is attribution, not this mean |
| teach Δ +0.583, controls flat, n=12 | genuine learning-shaped signal, immediate, one engine state, small n | **provisional**; re-run at n≥40 as CAR with delayed cold-reload paraphrase and k-curve |
| five failed web-gap attempts (+86 reads, mean 77 both) | input-side interventions, all reverted | WFA is designed to test the missing *attribution* side; do not repeat a sixth input tweak |

**Decision:** build the five harnesses against this spec; freeze pools and
rubrics before any engine boot; run PRIMARY once, read answers, finalise the
rubric, then reveal SEALED once. Report all five endpoints with effect sizes,
CIs, controls, and read counts. Do not ship a capability claim from a static
score or from a primary-only pass.

---

## 10. Pre-registration checklist (must be completed before each run)

- [ ] Hash rule constant `RAVANA-LEARN-5`; pools and sealed manifest hashes recorded
- [ ] Rubric written *after* reading real answers; hedge/deflection register enumerated
- [ ] Ground-truth provenance stated for every gold set (no circular labels)
- [ ] Seed pinned; same-config reproducibility verified across two runs
- [ ] n ≥ 40 (CAR/PDR/CLI/RSI) or ≥ 60 (WFA) per pool; unit of analysis = item
- [ ] One primary endpoint per component declared; secondaries Holm-adjusted
- [ ] Control arms specified with their own pass bounds
- [ ] Effect sizes + 95% CIs planned; exact tests where base rate ≈ 0
- [ ] Sealed pool unrevealed until the single reported run
- [ ] FAIL rules acknowledged as binding, including hard fails

---

## 11. Sources (peer-reviewed; verified to publisher/abstract records)

[1] **Ebbinghaus, H. (1885/1913).** *Memory: A Contribution to Experimental
Psychology* (Ruger & Bussenius, trans.). Teachers College, Columbia University.
Original *Über das Gedächtnis*, 1885. — Foundational **monograph**, not a journal
article: method of savings, `savings = (L − L_t)/L`, and the forgetting curve.
[Cited as the behavioral law being operationalised, not as neural evidence.]

[2] **Murre, J. M. J., & Dros, J. (2015).** Replication and analysis of
Ebbinghaus' forgetting curve. *PLOS ONE, 10*(7), e0120644.
doi:10.1371/journal.pone.0120644 — Successful replication by the method of
savings; savings decline with retention interval (0.582 at 20 min → 0.211 at
31 days in Ebbinghaus's data).

[3] **Roediger, H. L., III, & Karpicke, J. D. (2006).** Test-enhanced learning:
Taking memory tests improves long-term retention. *Psychological Science, 17*(3),
249–255. doi:10.1111/j.1467-9280.2006.01693.x — Testing beats restudy on 2-day
and 1-week tests (68% vs 54%; 56% vs 42%) but restudy beats testing at 5 min;
the criterion must be a delayed retrieval test.

[4] **Karpicke, J. D., & Blunt, J. R. (2011).** Retrieval practice produces more
learning than elaborative studying with concept mapping. *Science, 331*(6018),
772–775. doi:10.1126/science.1199327 — Retrieval advantage on inference
questions, d = 1.50 and d = 1.07; generalizes to concept-map criterial tests.

[5] **Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006).**
Distributed practice in verbal recall tasks: A review and quantitative
synthesis. *Psychological Bulletin, 132*(3), 354–380.
doi:10.1037/0033-2909.132.3.354 — 839 assessments / 317 experiments; optimal
interstudy interval increases with retention interval; justifies spaced rather
than massed k.

[6] **Johnson, M. K., & Raye, C. L. (1981).** Reality monitoring. *Psychological
Review, 88*(1), 67–85. doi:10.1037/0033-295X.88.1.67 — Distinguishes externally
perceived from internally generated information; the source-attribution
rationale for WFA.

[7] **Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993).** Source
monitoring. *Psychological Bulletin, 114*(1), 3–28.
doi:10.1037/0033-2909.114.1.3 — Source is **not** directly tagged; it is
attributed by decision processes using flexible, error-prone criteria; covers
misattributed familiarity, cryptomnesia and incorporation of fiction into fact.

[8] **Loewenstein, G. (1994).** The psychology of curiosity: A review and
reinterpretation. *Psychological Bulletin, 116*(1), 75–98.
doi:10.1037/0033-2909.116.1.75 — Curiosity as cognitively induced deprivation
from a perceived gap in knowledge; intensity, transience and disappointment
when satisfied.

[9] **Friston, K. (2010).** The free-energy principle: a unified brain theory?
*Nature Reviews Neuroscience, 11*(2), 127–138. doi:10.1038/nrn2787 — Perception
optimises predictions; epistemic value of exploratory action is bounded.
[Cited for bounded value only; the theory is contested as a general account.]

[10] **Loftus, E. F., Miller, D. G., & Burns, H. J. (1978).** Semantic
integration of verbal information into a visual memory. *Journal of
Experimental Psychology: Human Learning and Memory, 4*(1), 19–31.
doi:10.1037/0278-7393.4.1.19 — Misleading post-event information reduced
recognition accuracy (41% vs 75% correct in Experiment 1); effect larger when
introduced just before test.

[11] **Loftus, E. F. (2005).** Planting misinformation in the human mind: A
30-year investigation of the malleability of memory. *Learning & Memory, 12*(4),
361–366. doi:10.1101/lm.94705 — Review: warnings sometimes inhibit the effect,
only under limited circumstances; misinformation remembered as part of the
original event ~47% of the time in the cited neuroimaging paradigm.

[12] **Johnson, M. K., Raye, C. L., Wang, A. Y., & Taylor, T. H. (1979).** Fact
and fantasy: The roles of accuracy and variability in confusing imaginations
with perceptual experiences. *Journal of Experimental Psychology: Human
Learning and Memory, 5*(3), 229–240. doi:10.1037/0278-7393.5.3.229 — Internal
generations influence estimates of external frequency; the confusion mechanism
PDR must resist.

[13] **Wixted, J. T. (2004).** The psychology and neuroscience of forgetting.
*Annual Review of Psychology, 55*, 235–269.
doi:10.1146/annurev.psych.55.090902.141555 — Review: A-B, A-C retroactive
interference and temporal gradients; consolidation reduces susceptibility;
sleep improves memory for recently learned lists.

[14] **McCloskey, M., & Cohen, N. J. (1989).** Catastrophic interference in
connectionist networks: The sequential learning problem. In *Psychology of
Learning and Motivation, 24*, 109–165. doi:10.1016/S0079-7421(08)60536-8 —
**Edited-volume chapter**, cited honestly as a connectionist model, not human
data: interference is severe under sequential and not concurrent training;
Hetherington & Seidenberg's replication showed savings (damaged, not deleted).
Justifies the interleaved positive control in CLI.

[15] **Walker, M. P., Brakefield, T., Hobson, J. A., & Stickgold, R. (2003).**
Dissociable stages of human memory consolidation and reconsolidation. *Nature,
425*, 616–620. doi:10.1038/nature01930 — Wake stabilises but does not enhance;
sleep enhances; reactivation returns a consolidated memory to a labile state.

[16] **Diekelmann, S., & Born, J. (2010).** The memory function of sleep.
*Nature Reviews Neuroscience, 11*(2), 114–126. doi:10.1038/nrn2762 — Sleep
consolidates declarative and procedural memory; SWS reactivation redistributes
hippocampal memories and supports extraction of generalised features easing
inference; REM supports synaptic consolidation.

[17] **Walker, M. P., & van der Helm, E. (2009).** Overnight therapy? The role
of sleep in emotional brain processing. *Psychological Bulletin, 135*(5),
731–748. doi:10.1037/a0016570 — **Review/hypothesis** (not a direct
experiment), cited for sleep-dependent (re)processing and the "sleep to forget,
sleep to remember" framing; it is about emotional memory, so the declarative
consolidation claim in CLI rests mainly on [15] and [16].

[18] **Gentner, D. (1983).** Structure-mapping: A theoretical framework for
analogy. *Cognitive Science, 7*(2), 155–170 — Relations map in preference to
object attributes; systematicity favours interconnected predicate systems.
[Cognitive-level theory, not a neural finding.]

[19] **Gentner, D., & Markman, A. B. (1997).** Structure mapping in analogy and
similarity. *American Psychologist, 52*(1), 45–56.
doi:10.1037/0003-066X.52.1.45 — Structural alignment underlies both analogy
and similarity; appearance matches map attributes instead of relations.

[20] **Lake, B. M., Salakhutdinov, R., & Tenenbaum, J. B. (2015).** Human-level
concept learning through probabilistic program induction. *Science, 350*(6266),
1332–1338. doi:10.1126/science.aab3050 — One-shot human-level classification
(97% for the model vs ~77% deep convolutional baseline on the reported task)
by inferring a generative structure, plus creative generalisation probes.
