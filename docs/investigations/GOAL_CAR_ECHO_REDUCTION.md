# GOAL — reduce CAR's residual identical-surface echo (0.50 primary)

**Status:** experiment complete. **Decision: DO NOT implement a change that targets the
shipped `identical_surface_echo = 0.50` number. Document a narrow, default-off plan only
for the one variant that is a *genuine* surface de-parroting (V1), and state plainly that
the shipped metric cannot see it.**

**Constraints honoured:** scratch/doc only; **no production edit**; **no gate, threshold,
rubric, or split change**; **no engine boot** (parent runs WFA); no answer-key leakage
(fake contexts use recorded emitted strings + authored teaching triples, never the
withheld gold label). No web access was required for the experiment; citations were
verified by web lookup and are reproduced below.

Artifacts: `scratch/car_echo_reduction_experiment.py`,
`scratch/car_echo_reduction_results.json`, `scratch/car_echo_reduction_run.log`.
Reproduce: `python3 scratch/car_echo_reduction_experiment.py` (pure stdlib, <1 s).

---

## 1. What the 0.50 actually measures (instrument validation first)

The shipped CAR harness scores two **different answer channels**:

| channel | question | shipped metric | definition |
|---|---|---|---|
| reach | held-out paraphrase (`how is X produced?` etc.) | `has_chain` | `generated AND (chain_text in answer)` — `scripts/measure_learning_curve.py:233-244` |
| echo | IDENTICAL question, which *contains* the taught surface (`why does X happen? is it because a causes b, and b causes x?`, `measure_learn5_car.py:346`) | `identical_surface_echo` | `a in answer.lower() and b in answer.lower()` — `measure_learn5_car.py:500-501,540-543` |

**The echo metric is a label-co-occurrence test, not a surface-copy test.** It cannot
distinguish "the engine restated the proposition" from "the engine copied the exact
taught phrase". This is the central finding: any attempt to move this number either
removes the verified labels (losing content) or fools the substring test (cosmetics).

The reach metric is byte-exact containment of the assembler's chain text; the shipped
byte-exact verified-chain exemption made reach 1.00/1.00 while leaving echo at
0.50/0.00 (`docs/investigations/CAR_ANSWER_REACH.md`, "FINAL full-run verification").
The two channels are independent, so a rewrite scoped to the echo channel need not
touch reach.

**Probe validation performed (skills: validate the instrument before reporting):**
1. Self-test on known echo-positive and echo-negative fake contexts passes.
2. Two defects were found in the probe itself and fixed before any number was trusted:
   `_apply()` originally hard-coded `changed=True` even when `str.replace()` was a
   no-op (it would have reported a false rewrite on the reach channel); and
   `eligible()` read a cached `ident_answer` instead of the text actually being
   emitted. Both are fixed in the committed script (`changed = new != answer`;
   `eligible(ctx, answer)`).
3. The recorded k=8 narration for `eclipse` is `darkness leads to shadow, and shadow
   leads to eclipse`, while the authored bank triple is `(eclipse, alignment, shadow)`.
   The harness echo test uses the **bank** labels, so this divergence is one real
   source of the sealed 0.00 echo. It is recorded in the results JSON under
   `recorded_vs_bank_chain_label_drifts` and is also the sole miss of the semantic
   reach proxy (0.900 sealed / 0.990 full bank) — a proxy limitation, not a gate.
4. Per-item before/after strings for every rewrite are in the results JSON
   (`rewrite_audit_examples`), so each verdict can be hand-checked.

The raw `scratch/learn5_car_measured.json` was untracked scratch and is not in any
commit, so the fake contexts are reconstructed from the **exact recorded strings** in
the audited section-1 table of `CAR_ANSWER_REACH.md` and the **authored** `_CANDIDATES`
bank (parsed read-only with `ast`). Echo-positive membership is hash-pre-registered;
it does not affect any rate because the rewrite is a uniform per-item function.

## 2. Variants (deterministic, fail-closed, before-emission)

All three fire **only** when the emitted answer contains the exact taught surface
`"{a} causes {b}, and {b} causes {x}"` AND the context is `verified`, trusted
(`edge_kind='curated'`), and `strategy == 'gated_explanation_chain'`. Quoted user frames
(`you mentioned/said/asked: ...`) are excluded. No chain object is mutated.

- **V1 — role template (content-preserving abstraction).** Replace the taught phrase
  with `"{a} starts the causal chain, which passes through {b} and ends in {x}"`.
  All three verified labels and both causal links survive; only the surface role frame
  changes. This is the machine analogue of re-encoding an episode in a semantic
  initiator→mediator→outcome schema.
- **V2 — gist abstraction (content-dropping).** Replace with `"an initial trigger sets
  off an intermediate process that ends in {x}"` — `a` and `b` removed.
- **V3 — cosmetic dodge (deliberately deceptive control).** Insert a zero-width
  character inside `b` so the substring metric fails while the rendered text is
  unchanged. Included to prove what *mere* cosmetics look like. **Reject.**

`proposition_preserved` reports the semantic reach proxy on the **rewritten** answer:
`(a,b,x)` in causal order with a causal marker in both spans. It is a diagnostic, not
a gate.

## 3. Exact results

Primary k=8 (6 documented named items, 3 hash-selected echo-positive → baseline 0.500,
matching the documented 0.50):

| variant | shipped `identical_surface_echo` | verbatim taught-surface echo | proposition preserved | reach proxy (byte) | rewritten |
|---|---|---|---|---|---|
| baseline identity | **0.500** | 0.500 | 0.500 | 1.000 | 0 |
| V1 role template | **0.500** | **0.000** | **0.500** | **1.000** | 3 |
| V2 gist | 0.000 | 0.000 | **0.000** | 1.000 | 3 |
| V3 cosmetic | 0.000 | 0.000 | **0.000** | 1.000 | 3 |

Sealed k=8 (10 named items, 0 echo-positive → 0.000, matching documented 0.00):
all four variants 0.000 echo / 0.000 verbatim; no rewrite fires. Byte reach 1.000
throughout (semantic proxy 0.900 — the eclipse drift above).

Full authored bank stress (104 triples, all echo-positive, so rates are not
sample-limited):

| variant | shipped echo | verbatim echo | proposition preserved | reach byte | rewritten |
|---|---|---|---|---|---|
| baseline | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| V1 | **1.000** | **0.000** | **1.000** | **1.000** | 104/104 |
| V2 | 0.000 | 0.000 | 0.000 | 1.000 | 104/104 |
| V3 | 0.000 | 0.000 | 0.000 | 1.000 | 104/104 |

Audited V1 rewrite (primary `cough`):
`yes, because tickling causes expulsion, and expulsion causes cough.` →
`yes, because tickling starts the causal chain, which passes through expulsion and ends in cough.`
Shipped echo **True** (tickling and expulsion still present), verbatim echo **False**,
proposition **True**.

**Fail-closed (all variants):** fires on valid trusted/verified/gated context; unchanged
for untrusted provenance, wrong strategy, unverified chain, absent chain, uncertainty
answer with no taught surface, and quoted user frame; the paraphrase/reach channel is
untouched in every case. All 8 hard invariants PASS. Verified chain objects are never
mutated (only the emitted string is replaced).

## 4. Genuine reduction or cosmetics?

- **V1 is a genuine surface de-parroting.** The byte-identical taught phrase disappears
  (0.500 → 0.000 verbatim), while every verified label, both causal links, causal order,
  and the byte-exact reach channel survive. It is not a synonym shuffle or invisible
  character; the proposition is re-encoded in a new role frame. It corresponds to
  episodic→semantic re-description with detail retained.
- **But the shipped echo number does not move (0.500 → 0.500).** The metric counts
  label co-occurrence, and V1 deliberately keeps the labels. So V1 cannot be sold as
  "fixing 0.50". Any claim that it does would be exactly the answer-key/metric-fitting
  failure the skills forbid.
- **V2 and V3 do move the number to 0.000, but neither is acceptable.** V2 deletes the
  verified intermediate labels (proposition preserved 1.000 → 0.000) — a real loss of
  learned content, not abstraction of it. V3 is pure cosmetics (invisible Unicode to
  defeat a substring test) — explicitly forbidden by the goal.
- **Therefore: the only available way to lower the shipped number is to lose content
  or to cheat. The residual 0.50 is a property of the probe, not only of the engine.**
  A defensible de-parroting exists (V1), but observing it requires a *separate*
  verbatim-surface telemetry counter, which is a diagnostic addition, not a gate or
  threshold change.

## 5. Decision and narrow default-off implementation plan

**Do not implement** V2 or V3, and do not tune any gate/rubric to make 0.50 drop.
**Do not ship V1 as a fix for `identical_surface_echo`** — that claim is false under
the shipped metric. If the user's intent is genuinely to stop verbatim parroting, the
only supported mechanism is V1, behind a default-off flag:

1. Add `RAVANA_CAR_SURFACE_ABSTRACTION=0` (default off). At the identical-surface
   emission point only, when the text is byte-identical to a taught surface and the
   context is verified/trusted/`gated_explanation_chain`, apply the V1 role template.
   Call it defensively via `getattr` so duck-typed engines without the hook are
   unaffected. Never mutate `ctx._verified_chain_text` or the chain object.
2. Verify flag equivalence first: same WFA state, flag off vs on with no echo-positive
   item must be byte-identical (differential probe).
3. Add a **diagnostic-only** counter `verbatim_taught_surface_echo` beside (not
   replacing) `identical_surface_echo`. No gate, threshold, PASS criterion, or split
   changes. The shipped 0.50 remains reported unchanged.
4. Only then A/B on identical copied WFA state, `PYTHONHASHSEED=0`, `RAVANA_OFFLINE=1`,
   one process: primary + sealed k=8, k=0 and definition-only controls, and at least
   two unrelated fronts. Pre-registered keep condition: reach byte 1.00/1.00 unchanged,
   all 10 CAR criteria unchanged, `identical_surface_echo` unchanged, verbatim counter
   0.50 → 0.00, zero untrusted/unverified emissions, unrelated fronts byte-identical.
5. Falsifiers that stop adoption: any reach or criterion regression; any change when
   the flag is off; any rewrite on an untrusted/unverified/uncertain/quoted case; any
   unrelated-front text change; or a re-measurement that fails to reproduce
   0.500 → 0.000 verbatim. If a forced single-surface ablation shows the verbatim
   counter is also a proxy (e.g. label-preserving paraphrase still scored as echo),
   stop rewriting instruments and report the mechanism gap instead.

If the user requires the *shipped 0.50 itself* to fall with content and gates intact,
the answer is: **not achievable by a surface rewrite** — the next work item is a new,
separately-validated paraphrase-sensitivity endpoint, not an emission hook.

## 6. Brain grounding (all citations verified by web lookup)

- **Yassa, M. A., & Stark, C. E. L. (2011).** Pattern separation in the hippocampus.
  *Trends in Neurosciences, 34*(10), 515–525. doi:10.1016/j.tins.2011.06.006
  (PMID 21788086; PMC3183227; PsycNET 2011-16285-001). Review: dentate gyrus/CA3
  orthogonalises similar inputs. Mapping: a role-template rewrite separates the
  *surface* of a new trace from the verbatim teaching episode while preserving the
  relation — it must not separate the verified proposition itself.
- **Lifanov, J., Linde-Domingo, J., & Wimber, M. (2021).** Feature-specific reaction
  times reveal a semanticisation of memories over time and with repeated remembering.
  *Nature Communications, 12*, 3177. doi:10.1038/s41467-021-23288-5 (PMID 34039970;
  PMC8155072). Empirical: repeated retrieval strengthens conceptual over perceptual
  detail. Mapping: V1 keeps the conceptual proposition and drops the perceptual
  (verbatim) detail; this is why it is genuine abstraction, while V3 (invisible
  character) changes only perceptual detail and is cosmetics.
- **Winocur, G., & Moscovitch, M. (2011).** Memory transformation and systems
  consolidation. *Journal of the International Neuropsychological Society, 17*(5),
  766–780. doi:10.1017/S1355617711000683. Review: episodic/context-specific traces
  transform into semantic/gist versions represented extra-hippocampally.
- **Winocur, G., Moscovitch, M., & Bontempi, B. (2010).** Memory formation and
  long-term retention in humans and animals: convergence towards a transformation
  account of hippocampal–neocortical interactions. *Neuropsychologia, 48*(8),
  2339–2356. doi:10.1016/j.neuropsychologia.2010.04.016. Theory/review: gist retained,
  specific detail lost — the V2 failure mode is a *lossy* transformation, which is why
  it is rejected when the verified middle node is load-bearing.
- **Tse, D., et al. (2007).** Schemas and memory consolidation. *Science, 316*(5821),
  76–82. doi:10.1126/science.1135935 (PMID 17412951; verified earlier in
  `CAR_ANSWER_REACH.md`). Schema assimilation must supply evidence, not fabricate a
  link — the role template can only re-encode `grade_chain`/`assertable_chain`-verified
  hops; it never adds one.
- **Levelt, W. J. M., Roelofs, A., & Meyer, A. S. (1999).** A theory of lexical access
  in speech production. *Behavioral and Brain Sciences, 22*(1), 1–38.
  doi:10.1017/S0140525X99001776 (verified earlier in `CAR_ANSWER_REACH.md`). Staged
  message→lemma→form: the rewrite acts at message formulation, after the proposition
  is committed, consistent with a production-side constraint rather than a post-hoc
  text filter.

## 7. Files created / modified

Created: this document; `scratch/car_echo_reduction_experiment.py`;
`scratch/car_echo_reduction_results.json`; `scratch/car_echo_reduction_run.log`.
**No production file, gate, threshold, rubric, weight store, or existing document was
modified. No live engine was booted by this task.**
