# Unified brain-grounded root-cause map across four failed benchmarks

**Status:** synthesis only. **No engine boot, no `process_turn`, no production edit, no
gate/threshold/item/gold/scoring change.** Every number below is read from a committed
(or explicitly marked uncommitted) investigation document; nothing is re-measured here.
Worktree `C:/Users/Likhith/Documents/projects/ravana_unified_synthesis`, branch
`docs/unified-brain-root-cause-map`, base `3aa4081`.

**Scope:** WFA internal-source monitoring, RSI/TRA unseen-subject relational transfer,
TRA-K typed-channel use, CAR chain acquisition. The question is not "how do we make the
numbers go up" but "which human-brain function is absent, which single mechanism could
cover more than one absence, and what would rule each candidate in or out."

**Evidence discipline inherited from the loaded skills:** a saturated rule table is G6
mechanism evidence, not capability (source evaluator `R=1.00`; H_a 80/80); a metric that
counts co-occurrence is not a surface-copy metric (CAR `identical_surface_echo=0.50`);
a candidate-side lookup that fires identically for an unrelated token contains zero
information about the subject (TRA H_b subject-dependent 0/80); a correct decline is not
a broken gate, and a wiring gap is not an algorithm gap.

---

## 1. Authoritative verdicts (last complete measurements, not re-run)

| Benchmark | Last authoritative result | Authoritative source |
|---|---|---|
| **WFA** | Full bank, unchanged scoring: primary credible/discredited/unknown `0.985/0.923/1.000`, sealed `0.955/0.627/1.000`; anti-tautology PASS; **overall FAIL**. PD/forced-choice gates failed (`R=0`, `pd_unparseable_rate=1.00`). Prior-act trace local A/B improves sealed discredited `0.627 → 0.985` (132 items), but its full-harness endpoint is **UNMEASURED** (two timeouts). | `WFA_PHASE0_RUNTIME_BLOCK.md` §"UPDATE 2026-11-27"; `WFA_PRIOR_ACT_TRACE.md` §4–5 |
| **RSI** | Live rule induction: **no rule induced**. `process_turn` called on demonstrations only, never the query; 1,392/1,440 emissions (96.67%) are uncertainty frames; `0/1,440` contain explicit `relation=`/`target=`/`verdict=`; structural near/far `target_ok=0/16` at every k. Foil rejection `1.00` is the same fail-closed decline, not evidence. Fake `1.00` is an oracle artifact. | `RSI_LIVE_RULE_INDUCTION.md` §1–1.2 |
| **TRA** | Standing unseen-subject transfer **not achieved**. H_a (ostensive subject-type teaching): 80/80 in the type-label space but 0/80 on the original word task (answer supplied). H_b (broad type channel): 7/80, subject-dependent picks 0/80, 12/40 foil leaks. Information audit: query subject's own type set contains taught label `0/160`; gold candidate's `80/80`. | `GOAL_UNSEEN_TYPE_TRANSFER.md` §2–5; `TYPE_LEARNING_BRAIN.md` §0 |
| **TRA-K** | Focused live (40 families, all instrument controls OK, G6 not saturated): transfer `0.625`, near `0.450`, far `0.300`, foil `0.350`, negative `0.100`, retention `0.000`, growth `+0.000` → `OVERALL: FAIL`, genuine. | `TRA_K_FOCUSED_LIVE.md`; `TRA_K_SPEC.md` |
| **CAR** | Sealed k=2 acquisition fails in-process: 7/10 items have `raw_chains=0` from node-id drift after sleep pruning. Across 100 items, 29/30 with a stale id fail vs 58/60 without pass. Cold reload additionally flips `curated → curriculum` provenance and deletes phrase nodes with all incident edges. Residual `identical_surface_echo=0.50` is a label-co-occurrence artifact. | `CAR_K2_ROOT_CAUSE.md` §0, §2; `CAR_ROOT_CAUSE.md` §1.2–1.4; `GOAL_CAR_ECHO_REDUCTION.md` §1, §4 |

---

## 2. Missing human-brain function, per benchmark

### 2.1 WFA — internal source monitoring (not trust, not item memory)

**Missing:** autonoetic memory of *one's own prior endorsement* of this claim, plus an
engine-owned read path to stored provenance. Three precise absences are documented:

1. No engine-owned read path: `_source_monitor_decision(ctx)` returns `None` unless
   `ctx.source_context` is populated, and the only writer is the harness setter
   `set_source_context_for_turn()`. On every un-staged turn the gate is inert
   (`GOAL_AUTONOMOUS_SOURCE_MONITORING.md` §1, `engine.py:8090-8153`).
2. No stored credibility attestation: no real writer stores a credibility token or
   trust class; the evaluator demands a token that must be manufactured from `edge_kind`
   — the G6 hazard (same doc §1, §6).
3. No claim-keyed decision memory: `_last_source_evaluation` is one global slot; when
   `gg032` (same claim text) invalidates it tags `g62`'s shared `source_id`, `g62`'s first
   turn ABSTAINs and the prior act is never formed. Supplying only
   `prior_decision=ENDORSE` flips ABSTAIN → WITHDRAW with provenance byte-identical
   (`WFA_PRIOR_ACT_TRACE.md` §1; `WFA_ISOLATED_CONFIRM.md` §"g05 anomaly", an uncommitted
   parent-worktree artifact at synthesis time).

**Ruled out as the missing function:** the trust evaluator and its rules — they exist and
saturate at 100/100; familiarity-only (V3) endorses 0/72, so item strength cannot stand in
for source (`GOAL_AUTONOMOUS_SOURCE_MONITORING.md` §5.1, §5.3).

**Brain function:** source memory dissociates from item memory; internal source monitoring
distinguishes what *I* thought/said from what is merely known, using qualities of
experience evaluated at retrieval and frontal monitoring systems — not an abstract tag
(Johnson, Hashtroudi & Lindsay 1993; Tulving 1985; Mitchell & Johnson 2009; Yonelinas 2002).

### 2.2 RSI/TRA — fast mapping over subject-specific evidence + a pre-existing schema

**Missing:** information *about the query subject*, and a structured hypothesis space that
makes sparse exemplars diagnostic. The mutual information between available input and gold
type is ~0 by construction: teaching turns assert things about other words, the query is a
bare unseen token in a relation-neutral forced-choice frame, and the graph holds no typed
evidence for that token (`TYPE_LEARNING_BRAIN.md` §0). Both routes that moved an endpoint
did so only after the benchmark supplied the answer (H_a) or typed the candidate options
(H_b); H_b's picks were identical for the real subject and `zeppelin` (0/80
subject-dependent), so it is candidate-side type lookup, not subject typing
(`GOAL_UNSEEN_TYPE_TRANSFER.md` §3–4).

RSI has an additional, distinct deficit: the live backend never issues the query turn, so
near and far receive byte-identical emissions and no near/far difference is *possible*
under that instrument (`RSI_LIVE_RULE_INDUCTION.md` §1, observation 1).

**Ruled out as the missing function:** ranking/threshold/scorer tuning, and a broad
candidate-side type lexicon (measured to fail with 0/80 subject-dependent picks and
12/40 foil leaks). GloVe slot centres are non-separable (pairwise cosine 0.729–0.980,
top-2 margin 0.0198–0.0873 against a frozen 0.10 gate) — a similarity proxy cannot be
tuned into a structural measure (`TYPE_LEARNING_BRAIN.md` §0; `TRA_CATEGORY_CENTRE_EXPERIMENT.md` E1).

**Brain function:** fast mapping obtains a word's type from an ostensive/contrastive event
about *that word* (Carey & Bartlett 1978; Dollaghan 1985; Medina et al. 2011), from
distributional contexts containing that word (Landauer & Dumais 1997; Lany & Saffran 2010),
or from sparse evidence over a *pre-existing structured hypothesis space* / neocortical
schema (Xu & Tenenbaum 2007; Tenenbaum & Griffiths 2001; Tse et al. 2007; Gick & Holyoak 1983).
A child does not induce a type by cross-tabulating assertions about other words (Medina et al. 2011).

### 2.3 TRA-K — schema-assisted durable typed-slot assimilation *and a wired reader*

**Missing:** the write→store→read seam, not the knowledge channel. Three static checks
locate the FAIL (`TRA_K_REPLAY_CONSOLIDATION_DESIGN.md` §0):

1. **Persistence is not the failure**: 468 trusted typed-slot edges
   (`slot_writer='tra_typed_slot'`, `edge_kind='curated'`, `relation_type='is_a'`) survive
   cold reload; some are decayed below the 0.20 floor, some were written unprovenanced —
   a write-time acquisition drop, not a save/load drop.
2. **The recorded delayed probes are not scorable by a typed-slot reader** (domain-word
   options vs source-native Wikidata labels) — an instrument-scope mismatch in that run.
3. **Emission is the separate gap**: recorded responses are the honesty deflection
   register and the typed-slot read primitive is default-OFF and unwired (`hook None`).
   "No storage mechanism can make an unwired reader speak."

The channel itself is clean: 3,200 facts / 400 items, `answer_key_leak=0`,
`overlap_world_model=0`, `provenance_missing=0`
(`TRAK_INTEGRATION.md` §2). So the absence is the mechanism that turns a provenanced
one-trial teaching fact into a retained, *read* subject-typed decision (retention `0.000`,
growth `+0.000`), plus read wiring at emission.

**Ruled out:** "the channel is missing" (it is supplied and uncontaminated), and "the
engine cannot store typed slots" (468 survive reload).

**Brain function:** one-trial assimilation becomes durable only when a neocortical schema
already exists (Tse et al. 2007); fast episodes and slow regularity extraction are
complementary (McClelland, McNaughton & O'Reilly 1995); replay is causal for consolidation
(Wilson & McNaughton 1994; Diba & Buzsáki 2007; Girardeau et al. 2009); retrieval returns a
consolidated trace to a labile state requiring renewed stabilisation (Nader, Schafe &
LeDoux 2000). The design itself honestly flags the boundary: no cited source establishes
replay of typed semantic `is_a` facts in a symbolic graph.

### 2.4 CAR — synaptic tagging / reconsolidation and pattern separation through maintenance

**Missing:** a maintenance process that preserves the episode tag and node identity across
sleep and save/load, plus a surface re-encoding that separates a new trace's surface from
the teaching episode without losing the proposition. Measured drop points:

- **k=2 in-process**: sleep junk-node pruning deletes freshly created concept nodes without
  removing their ids from `_concept_keywords`; `_all_labels` returns the first (dead) id, so
  both taught hops dangle on non-existent nodes; `raw_chains=0` although the edges exist with
  correct `curated` provenance. Mechanism (c) node-id resolution drift, CONFIRMED; ranking
  crowd-out RULED OUT for the seven failures (`gate.raw_chains=0` at every trust level)
  (`CAR_K2_ROOT_CAUSE.md` §0, §2).
- **Cold reload**: `_seed_curriculum_causal` overwrites `curated` conversation edges for
  any (src,tgt) that coincides with a curriculum hop (its guard exempts `web_fact` but
  admits `curated`); `_sanitize_graph` deletes phrase nodes and every incident edge
  (8 nodes in the measured log); load-time value drift re-values edges
  (`CAR_ROOT_CAUSE.md` §1.2–1.4).
- **Echo residual**: `identical_surface_echo` counts label co-occurrence, not verbatim
  copying, so the only content-preserving rewrite (V1 role template) removes verbatim echo
  `0.500 → 0.000` while the shipped number stays `0.500`; V2 drops verified labels and V3
  is an invisible character (`GOAL_CAR_ECHO_REDUCTION.md` §3–4).

**Ruled out:** trust-gate/provenance-class absence (taught hops are already `curated` and
trusted), and "the chain was never written" (it is written, then detached).

**Brain function:** a weak stimulus sets a time-limited tag that captures plasticity
proteins from a strong event (Frey & Morris 1997); reactivation makes a trace labile and
requires restabilisation (Nader et al. 2000); complementary learning systems keep fast
episodes distinct from slow statistical structure (McClelland et al. 1995); pattern
separation orthogonalises similar inputs (Yassa & Stark 2011); repeated retrieval
semanticises memories, strengthening conceptual over perceptual detail (Lifanov et al.
2021); hippocampal–neocortical transformation preserves gist while dropping specific
detail (Winocur & Moscovitch 2011; Winocur, Moscovitch & Bontempi 2010). V2's loss of the
verified middle node is exactly the *lossy* transformation these accounts warn about.

---

## 3. Candidate shared mechanisms — and the evidence that rules them in or out

### M1 (best coverage) — an identity-keyed, provenance-bearing, durable act record, written at the act and read at retrieval by an engine-owned path

Computational core: bind content to (i) its origin, (ii) the system's own prior decision, and
(iii) a stable identity key for the subject/claim; write it at encoding/commit; read it by a
path the engine owns; keep it through maintenance/consolidation. This is the *computational
shape* shared by internal source monitoring, schema-assisted one-trial binding, and
synaptic tagging/reconsolidation. It is **not** a claim that the brain has one module doing
all four jobs.

| Benchmark | Evidence FOR M1 | Evidence AGAINST / boundary |
|---|---|---|
| WFA | Supplying only a claim-keyed `prior_decision=ENDORSE` flips sealed discredited `0.627 → 0.985`; scratch read-side walk gets 24/24 credible endorse, 24/24 discredited/unknown abstain, MUT flips 24/24, TAUT invariant; g05 fails precisely when the taught edge disappears in the first turn (`WFA_PRIOR_ACT_TRACE.md` §4; `GOAL_AUTONOMOUS_SOURCE_MONITORING.md` §5; `WFA_ISOLATED_CONFIRM.md`) | The walk's `R=1.00` is a deterministic rule table over clean tags (G6). No stored credibility attestation exists; the token is manufactured from `edge_kind`. Invalidation cue is experimenter-supplied (extrapolation). Full-harness endpoint for the trace remains unmeasured. |
| CAR | All three cold/reload root causes are loss or overwrite of the tag/identity (`curated→curriculum`, stale ids, phrase-node deletion); V1 preserves the proposition while changing only the surface frame | M1 preserves provenance but does not fix ranking/cue-overload when trusted chains do exist but are crowded out at limit 64 (F3 in `CAR_K2_BRAIN.md` §1.2). |
| TRA-K | 468 typed slots survive reload; replay design targets exactly the write→store→read seam; failure is `retention=0` with an unwired reader | Persistence already works, so M1 alone cannot fix the endpoint unless the read primitive is wired; replay R2 has not been run live. |
| RSI/TRA | H_a works **only** when the subject's type is written as a provenance-stamped trusted edge and read by the existing primitive (80/80, clean wrong-subject/negation rejection); H_b fails because its trace is candidate-keyed, not subject-keyed | H_a's teaching turn supplies the answer → changed benchmark, not transfer; saturated 100/100 is a mechanism check. |

**Verdict:** M1 is the only candidate with a positive, mutation-controlled intervention in
one benchmark (WFA), a natural explanation of CAR's three drop points, and a direct account
of TRA-K's retention-zero/unwired-reader seam and RSI/TRA's subject-vs-candidate
asymmetry. It is **necessary but not sufficient**: without subject-side information (M2)
and without a reader that fires (M3), M1 is a store nobody reads.

### M2 — a pre-existing structured hypothesis space / schema over kinds and relations

- **In:** Tse et al. 2007 (one-trial assimilation needs a schema); Xu & Tenenbaum 2007 and
  Tenenbaum & Griffiths 2001 (generalisation depends on representational structure, not
  exemplar count); Gick & Holyoak 1983 and Holyoak & Thagard 1989 (transfer needs an
  abstract schema / relational fit, not feature overlap). H_b shows that a broad lexicon
  without a hierarchy connecting the *subject* yields 0/80 subject-dependent decisions.
- **Out / limit:** under the original TRA input regime no structure over the available
  signals carries the gold type (mutual information ~0); a schema cannot create missing
  subject-side information. TRA-K supplies an external Wikidata schema (0 answer-key leak)
  and still FAILs with retention 0 → schema alone is insufficient.

### M3 — wiring the reader / committing to the proposition before emission

- **In:** WFA's evaluator is inert without a caller-staged context; TRA-K's typed-slot reader
  is default-OFF with `hook None`; RSI never sends the query to `process_turn`, so near/far
  are byte-identical; CAR's reach was recovered by routing through the assembler, not by
  improving the decoder. Call-count/reachability tracing is the right first test in every
  case. The brain's production-side conflict monitor commits to a proposition before form
  encoding (Levelt, Roelofs & Meyer 1999; Nozari, Dell & Schwartz 2011).
- **Out / limit:** wiring a saturated rule table yields G6 mechanism evidence, not
  capability. "No storage mechanism can make an unwired reader speak" — but an unwired
  reader also cannot reveal a substrate deficit until it is wired.

### M4 — pattern separation / episodic→semantic re-encoding that preserves the proposition

- **In:** CAR V1 removes verbatim echo `0.500 → 0.000` with proposition preserved and byte
  reach 1.00; Yassa & Stark 2011, Lifanov et al. 2021, Winocur & Moscovitch 2011 support
  surface/detail separation with gist retained.
- **Out:** the shipped CAR metric cannot see it, and V2 (content-dropping) / V3 (cosmetic
  Unicode) are rejected — the only way to lower the shipped number is to lose content or
  cheat. This is a diagnostic mechanism, not a fix for any of the four headline FAILs.

### Explicitly ruled out as shared fixes

Familiarity/strength proxies (WFA V3: 0/72 endorsements; TRA GloVe centres: non-separable),
threshold/gate tuning (repeatedly measured flat or harmful; `THETA_LOWER_IS_BETTER.md`),
candidate-side broad lexicons (TRA H_b 0/80 subject-dependent), and cosmetic string edits
(CAR V3). None creates information or restores a lost trace.

---

## 4. Ranked experiment queue (highest information gain first)

Default-OFF, state-controlled, no gate/threshold/gold edits in any arm; citations are the
DOIs verified in the cited investigation docs (ledgers there record Crossref/PubMed checks;
not re-fetched in this synthesis).

| # | Priority | Experiment — decisive question | Design / controls | Pass / fail endpoint | Citation |
|---|---|---|---|---|---|
| **E1** | P0 | **Complete the full-harness WFA prior-act trace A/B.** Does the claim-keyed act record work end-to-end, not just in a 132-item discredited-arm probe? | Same copied store per arm, `RAVANA_PRIOR_ACT_TRACE` unset/`0`/`1`; full bank with anti-tautology; schedule >60 min or profile the harness first. | Sealed discredited ≥ 0.90 with credible/unknown unchanged and anti-tautology PASS; a third timeout → instrument work, not a capability result. | Nader et al. 2000, doi:10.1038/35021052; Johnson et al. 1993, doi:10.1037/0033-2909.114.1.3 |
| **E2** | P0 | **Claim-keyed provenance-retention probe (g05 stratum).** Does the taught edge survive the first `process_turn`? | Fresh engine per condition; mutate only the edge's `source_status`; assert edge survival AND decision flip; include g64 as survival control and `zeppelin`-style orphan as fail-closed control. | g05-class item retains ≥1 provenance edge through turn 1 and emits WITHDRAW after invalidation; edge loss with correct ABSTAIN → maintenance write-path defect located. | Yonelinas 2002, doi:10.1006/jmla.2002.2864; Mitchell & Johnson 2009, doi:10.1037/a0015849 |
| **E3** | P0 | **TRA-K replay consolidation A/B with the reader wired.** Does offline reactivation convert channel teaching into retained, read decisions? | Arms OFF / R1 provenance restamp / R2 = R1 + 3 reactivation cycles / R1_CONF confidence sham / R1_WRONG wrong-tag sham; typed-option space; subject- and channel-independence; G6 + M-A/M-B; focused 20/pool first. | `R2 − OFF ≥ +0.10` on primary delayed near/far, foil/negative drop ≤ 0.05, shams flat, gain survives cold reload; otherwise STOP — no full 400-family run. | Wilson & McNaughton 1994, doi:10.1126/science.8036517; Diba & Buzsáki 2007, doi:10.1038/nn1961; Girardeau et al. 2009, doi:10.1038/nn.2384; Tse et al. 2007, doi:10.1126/science.1135935 |
| **E4** | P1 | **RSI query-turn wiring.** Does the engine ever see the query, and can it emit a structured relation/target? | Fix the instrument so `present()` issues a `process_turn` for the query (demonstrations unchanged); count query turns and explicit `relation=/target=/verdict=` fields before reading any score; keep near/far byte-difference as the reachability check. | Non-zero structured targets and near ≠ far emissions; persistent 0/1,440 fields → production-side commitment gap, not a parser gap. | Gick & Holyoak 1983, doi:10.1016/0010-0285(83)90002-6; Holyoak & Thagard 1989, doi:10.1207/s15516709cog1303_1; Nozari, Dell & Schwartz 2011 (production-side monitor) |
| **E5** | P1 | **TRA subject-side evidence channel.** Does subject-specific evidence beat candidate-side lookup under the original controls? | Independent ontology (external, not `type_sets`/`world_model`) or frozen corpus containing the test token; keep near/far/foil/negative, disjoint options, and the same-subject-vs-`zeppelin` independence control; no threshold change. | Subject-dependent picks > 0 with foil/negative flat and channel overlap with the answer key = 0; 0/80 again → relabel TRA an information-availability test. | Carey & Bartlett 1978 (no DOI; proceedings PDF); Dollaghan 1985, doi:10.1044/jshr.2803.454; Medina et al. 2011, doi:10.1073/pnas.1105040108; Landauer & Dumais 1997, doi:10.1037/0033-295X.104.2.211; Xu & Tenenbaum 2007, doi:10.1037/0033-295X.114.2.245 |
| **E6** | P2 | **CAR tag/identity preservation across sleep and load.** Does keeping the tag alive recover k=2 chains? | A/B on the 29/30 stale-id stratum vs the 58/60 clean stratum; guard `_concept_keywords` resolution and the `curated→curriculum` overwrite; cold reload in both arms; no trust-set change. | Sealed k=2 `raw_chains > 0` and immediate + cold reach improve with cold ≈ immediate; flat → the residual cause is ranking/cue overload, not tag loss. | Frey & Morris 1997 (repo-verified); Nader et al. 2000, doi:10.1038/35021052; McClelland et al. 1995, doi:10.1037/0033-295X.102.3.419; Yassa & Stark 2011, doi:10.1016/j.tins.2011.06.006 |
| **E7** | P2 | **CAR verbatim-surface telemetry (diagnostic only).** Can genuine de-parroting be observed without fitting the shipped metric? | Default-off V1 role template at the identical-surface seam; add a **separate** verbatim counter beside `identical_surface_echo`; flag equivalence; never rewrite on untrusted/unverified/quoted cases. | Verbatim `0.500 → 0.000`, reach 1.00/1.00 and all CAR criteria unchanged, shipped number reported unchanged; any unrelated-front text change stops adoption. | Lifanov et al. 2021, doi:10.1038/s41467-021-23288-5; Winocur & Moscovitch 2011, doi:10.1017/S1355617711000683; Levelt et al. 1999, doi:10.1017/S0140525X99001776 |

Sequencing rule: E1–E3 first because each converts a locally-positive but unverified claim
into a decisive endpoint; E4–E5 next because they test whether the substrate exists at all;
E6–E7 last because they are refinements whose endpoints are already instrument-limited.

---

## 5. Top 3 actionable hypotheses

1. **H1 — Identity-keyed provenance + prior-act record is the common denominator.**
   Implement the claim-keyed prior-act trace and an engine-owned claim→trace read path as
   default-off instruments; extend the same "write the act, preserve the tag, read it at
   retrieval" shape to CAR's curated-hop maintenance and TRA-K's typed-slot seam. Evidence:
   WFA `0.627 → 0.985` with only the prior decision supplied; g05 fails when the edge is
   lost; all three CAR cold/k=2 drop points are tag/identity losses; TRA-K has 468 surviving
   typed slots but `retention=0` and no reader. Falsifier: E1 full-harness timeout/failure,
   E2 edge loss persists, E3 R2 shams not flat.

2. **H2 — Wire the reader and issue the query before building more mechanism.** Several
   fronts have the substrate but no firing path: WFA's `source_context` is harness-staged,
   TRA-K's typed-slot reader is default-OFF (`hook None`), RSI never sends the query turn.
   Count calls at the entry point first; wire the existing primitive; then re-measure.
   Falsifier: with call counts > 0 and a wired reader, endpoints stay flat → substrate
   deficit, not wiring.

3. **H3 — Only subject-side evidence can unlock transfer; the external ontology is a
   legitimate input, the item bank's answer key is not.** TRA/TRA-K must connect the
   *subject* to a type via an ostensive turn, an independently authored hierarchy, or a
   distributional channel over sentences containing the test word. Candidate-side lookup
   is empirically ruled out (0/80 subject-dependent, 12/40 foil leaks). Falsifier: E5 with
   a clean independent channel still gives 0/80 subject-dependent picks → relabel the
   benchmark an information-availability test rather than lowering any gate.

---

## 6. Honest boundaries

- No capability claim is made. WFA's evaluator and H_a are G6-saturated rule-table
  mechanisms; TRA-K replay is an untested design; CAR V1 is invisible to the shipped echo
  metric; RSI's fake arm is an oracle artifact.
- The "shared mechanism" in §3 is a *computational analogy* across source monitoring,
  schema-assisted consolidation and synaptic tagging — not a claim that the brain runs one
  module for all four benchmarks. M2 and M3 are separate necessary conditions; the
  literature does not support collapsing them into M1.
- Citations are reproduced from the verification ledgers of the cited investigation docs
  (which record Crossref/PubMed/DOI checks and, in `TYPE_LEARNING_BRAIN.md` §1.1, corrections
  of previously guessed DOIs). They were not re-fetched in this docs-only task. Where a
  source has no DOI (Carey & Bartlett 1978) or was repo-verified without one (Frey & Morris
  1997), that is stated.
- `WFA_ISOLATED_CONFIRM.md` was uncommitted in the parent worktree at synthesis time; its
  finding is attributed explicitly and is not treated as an authoritative full-bank result.
- No gate, threshold, item, gold, rubric, weight store, or production file was modified;
  no engine was booted.
