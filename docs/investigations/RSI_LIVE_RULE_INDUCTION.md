# Live RSI rule-induction failure: evidence, mechanism, and fixes (2026-11-23)

Status: **static/evidence investigation.** No engine process was run and no
pytest suite was run for this note; all numbers below are computed from the
saved rows in `scratch/rsi_focus_live_rows.json` and
`scratch/rsi_focus_fake_rows.json` (384 rows each) and from direct calls of the
parser. See `RSI_FOCUSED_LIVE.md` for the originating live run (1,440
`process_turn` calls, one engine boot, no backend error).

## 1. Headline answer to the delegated question

**No rule is induced at all. The engine is never asked the query, and it
answers the teaching statements with epistemic-uncertainty chatter. The parser
is not too strict: there is no structured answer to recover, and loosening it
would manufacture a match out of an absence of evidence.**

Three independent, count-based observations force this conclusion:

1. **`eng.process_turn` is called only on demonstrations, never on the query.**
   For every one of the 384 live rows, `len(engine_responses) == k`
   (k in {1,2,4,8}); the distribution is 96 rows per response-count 1/2/4/8,
   exactly matching the k distribution. `RealBackend.present()`
   (`scripts/measure_learn5_rsi.py:505-529`) loops `for triple in
   demo_triples: eng.process_turn(...)` and then passes those k demo-reaction
   strings to `_parse_engine_answer`. The item's query subject/candidates are
   used only *downstream*, as tokens to look for in those strings. Consequence:
   near/far/foil share identical demos at a given item+k and therefore receive
   **byte-identical engine emissions**. A near-vs-far difference cannot be
   produced by this instrument no matter what the engine knows.
2. **The engine declines the demonstrations instead of asserting them.**
   Of 1,440 saved emissions, 1,392 (96.67%) are uncertainty frames:
   `HONESTLY_<X>_OUTSIDE <TAIL>` 480+17, `NOSOLID_<X>_SOFAR <TAIL>`
   431+30, `NOTSURE_<X>_OK <TAIL>` 413+21 (354 distinct strings; the `<X>` is a
   demo entity or the relation word). A further 36 are bare acknowledgments
   (`right./yeah./got it./ok, noted.`), 12 are pseudo-assertions about a demo
   object (`It appears that it comes before a rainbow.` ×8, `It is similar to
   a rainbow.` ×3, one malformed hybrid), and **0/1,440 contain an explicit
   `relation=`/`target=`/`verdict=` field**. There is no empty-string failure:
   the engine always says something, but what it says is a decline.
3. **No target is ever recovered on a positive condition.** Structural near:
   `target_ok = 0/16`, `match = 0/16` at every k (1/2/4/8); structural far:
   `target_ok = 0/16`, `match = 0/16`. 26/384 rows (6.8% of rows — the
   originating note's "26/1440 (1.8%)" divides the per-row counter by the
   per-call total, a labeling slip; the counter increments once per `present`)
   are unparsed and scored `no_answer`. Even if all 26 became perfect matches,
   0.00 -> 0.07 could not close a gap to 1.00.

So the answer is not "unrelated chatter like WFA's mirror route" (the emissions
are on-topic declines, not instruction echoes), not "plausible but wrong
relation/target" (no candidate target is emitted on any positive structural
row), and not "empty string". It is: **the teaching assertion is routed to an
uncertainty emitter, no trace is written, no query turn is issued, and the
grade fail-closes to reject/no_answer.**

### 1.1 Structural arm, near condition, k=1 vs k=8 — fake vs live (8 rows)

| id / pool / relation | k | fake output | live raw emission (first turn) | live output | flags |
|---|---|---|---|---|---|
| rsi_000 primary CAUSE | 1 | CAUSE/mist/match | "hmm, i'm not totally sure about that yet, and that's okay..." | None/None/**reject** | correct=False |
| rsi_001 primary PART_OF | 1 | PART_OF/tempest/match | "hmm, i'm not totally sure about shower yet..." | None/None/**reject** | correct=False |
| rsi_002 sealed PRECEDES | 1 | PRECEDES/typhoon/match | "hmm, i'm not totally sure about breeze yet..." | None/None/**reject** | correct=False |
| rsi_005 sealed RESEMBLES | 1 | RESEMBLES/chill/match | "hmm, i'm not totally sure about resembles yet..." | RESEMBLES/None/**reject** | correct=False |
| rsi_000 primary CAUSE | 8 | CAUSE/mist/match | 8 hedges ("outside what i know" ×3, "not sure" ×3, ...) | None/None/**reject** | correct=False |
| rsi_001 primary PART_OF | 8 | PART_OF/tempest/match | 8 hedges (mentions `shower`, `component`, `belongs`) | None/None/**reject** | correct=False |
| rsi_002 sealed PRECEDES | 8 | PRECEDES/typhoon/match | 8 hedges ("not sure about precedes") | PRECEDES/None/**reject** | correct=False |
| rsi_005 sealed RESEMBLES | 8 | RESEMBLES/chill/match | 8 hedges ("not sure about resembles"/"similar") | RESEMBLES/None/**reject** | correct=False |

The fake rows are pure oracle rows: `FakeBackend.present()` returns the demo
triples, and `run_item()` grades structural via `induce()` + world-model lookup
(`measure_learn5_rsi.py:636-649`), never via the parser. Fake 1.00 therefore
validates the *score key*, not the live observation path — exactly the
"fake oracle artifact" the prior note flagged.

### 1.2 Full structural verdict census (96 rows)

| condition | reject | no_answer | match | correct target |
|---|---|---|---|---|
| near   | 16 | 0 | 0 | 0/16 |
| far    | 12 | 4 | 0 | 0/16 |
| foil   | 15 | 1 | 0 | n/a (reject is correct) |
| negative | 11 | 5 | 0 | n/a (reject is correct) |
| permuted | 27 | 5 | 0 | 0/32 |

Foil rejection (1.00) and improving negative rejection (0.5 -> 1.0) are
therefore **not evidence of rule induction**: they are the same fail-closed
decline that produces near=0, re-scored against a rubric that wants a reject.
`grade()` gives no free pass for `no_answer` (`test_unparsed_response_scores_no_answer`
pins this).

### 1.3 Parser validity: a real, measured false-positive

Before the fix, 46/96 live structural rows carried `output.relation != None`,
but in every case the relation word appeared only inside an uncertainty hedge
("not totally sure about **resembles** yet"; "**resembles** is a bit outside
what i know"). The old "only ever negated" guard checked only `"not <phrase>"`
and missed the hedge frame. Three foil rows even recovered a wrong target
(`shower`) from "not totally sure about shower yet". These are echoed prompt
vocabulary, not induced relations. The fix (section 4) removes them without
changing a single structural verdict.

## 2. Brain mechanism: what the literature supports and what is extrapolation

### Verified citations

- **Tse, D., Langston, R. F., Kakeyama, M., Bethus, I., Spooner, P. A., Wood,
  E. R., Witter, M. P., & Morris, R. G. M. (2007). Schemas and memory
  consolidation. *Science, 316*(5821), 76–82.**
  doi:10.1126/science.1135935. Verified against the publisher abstract
  (science.org): *"systems consolidation can occur extremely quickly if an
  associative 'schema' ... has previously been created ... New traces, trained
  for only one trial, then became assimilated and rapidly hippocampal-independent."*
- **McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there
  are complementary learning systems in the hippocampus and neocortex.
  *Psychological Review, 102*(3), 419–457.**
  doi:10.1037/0033-295X.102.3.419. Verified against the APA PsycNet record.
- **Yassa, M. A., & Stark, C. E. L. (2011). Pattern separation in the
  hippocampus. *Trends in Neurosciences, 34*(10), 515–525.**
  doi:10.1016/j.tins.2011.06.006; PMID 21788086. Verified against PubMed and
  PsycNet records.
- **Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy.
  *Cognitive Science, 7*(2), 155–170.**
  doi:10.1207/s15516709cog0702_3. Verified against the Wiley abstract:
  *"Relations between objects, rather than attributes of objects, are mapped
  ... rules depend only on syntactic properties ... not on the specific content
  of the domains."*
- **Roediger, H. L. III, & Karpicke, J. D. (2006). Test-enhanced learning:
  Taking memory tests improves long-term retention. *Psychological Science,
  17*(3), 249–255.** doi:10.1111/j.1467-9280.2006.01693.x. Verified against
  the APA PsycNet record.

### Support-vs-extrapolation ledger

| Claim used in this investigation | Source | Verdict | Why / limit |
|---|---|---|---|
| A pre-existing schema lets a single trace be assimilated in one trial | Tse et al. 2007 | **SUPPORT** (direct empirical claim, rat paired-associate) | Supports *why* a relation vocabulary should exist before teaching; does not show a naive system can form one from the same traces. |
| Schema congruence alone can create false memories; sources must admit links | Tse et al. 2007 + McClelland et al. 1995 (schema-congruence side effect) | **EXTRAPOLATION** | The false-memory consequence is established in the broader schema literature, not in the verified Tse abstract text quoted above; treat as a design caution, keep schema hypotheses unassertable. |
| Fast episodic encoding must be complemented by slow extraction of regularities; similar episodes must be kept distinct | McClelland et al. 1995; Yassa & Stark 2011 | **SUPPORT** (conceptual/neuroscientific) | Maps to "store each demo as a trusted trace, then abstract the relation type"; the hippocampus/neocortex ↔ engine-buffer/graph mapping is an **ANALOGY**, not a homology claim. |
| Relation transfer, not attribute/lexical overlap, is the operation under test | Gentner 1983 | **SUPPORT** (theoretical framework) | The harness's near/far/permuted/foil design is a direct operationalisation of structure mapping; RAVANA is not claimed to perform human analogy. |
| Inserting an active retrieval (a query) after study improves retention vs restudy alone | Roediger & Karpicke 2006 | **SUPPORT** for the *design principle*, **EXTRAPOLATION** for RAVANA | Human memory-test effect; the effect of an extra `process_turn` call on a statistical language/graph engine is unmeasured. |
| The engine's 96.7% hedges are the functional analogue of failed pattern separation | — | **EXTRAPOLATION (weak)** | The observable is a refusal-to-assert response string; no neural or architectural mechanism was measured. Label it a *behavioural* failure to encode, not a dentate-gyrus analogue. |

## 3. Ranked fixes, predictions, falsifiers

**F1 (P0, engine-side; upstream of articulation).** Make a declarative triple
assertion write a trace before any reply is generated; then induce the shared
relation type from converged traces. Detail in section 5.
- Prediction: on an isolated store with k verified trusted traces, structural
  near/far target accuracy moves off 0.00 and `growth_headroom > 0`, while
  foil/negative stay at their current reject rates.
- Falsifier: census the written edges first; if k trusted traces exist and a
  forced single-token decision surface (echo stripped) still yields ~0
  explicit decisions, the gap is downstream of encoding — stop editing the
  articulation path and work on retrieval/gate routing.

**F2 (P1, harness construct validity; NOT implemented — needs an engine run).**
After teaching, call `eng.process_turn` once on an explicit query prompt and
grade **that** response, not the demo reactions. Right now near and far are
guaranteed identical because the engine never sees the query.
- Prediction: near/far/foil stop being byte-identical at fixed item+k; the
  instrument can finally measure transfer.
- Falsifier: if the query-turn answer still contains no candidate target on a
  store verified to hold the demos, the bottleneck is F1, not the harness.
- Why deferred: it redefines the graded turn, invalidates the focused live
  numbers, and would require changing `test_real_present_grades_engine_not_demo_triples`
  (which pins `process_turn == 1`). Under the no-engine constraint it cannot be
  measured, so shipping it would be an unvalidated instrument change — the
  exact failure mode the debugging skill warns against.

**F3 (P2, localized harness fix; IMPLEMENTED).** Strip uncertainty-hedge and
acknowledgment frames from the *evidence* text used to recover relation/target,
while keeping them for decline/negation detection. See section 4.
- Prediction: spurious relation recovery disappears; no structural verdict
  changes; fake oracle unchanged.
- Falsifier: any structural verdict change or fake-oracle change => revert.
- Measured: 0/96 structural verdict changes; structural near rows with a
  relation 7/16 -> 0/16, permuted 20/32 -> 0/32; fake near stays 1.00.

## 4. Implemented change (localized, gate-preserving)

Files: `scripts/measure_learn5_rsi.py` (`_parse_engine_answer`),
`tests/unit/test_learn5_rsi.py` (3 new tests).

`_parse_engine_answer` now builds `ev_low` by dropping responses that are
uncertainty hedges (`not totally sure`, `not sure about`, `solid grasp on`,
`outside what i know`) or bare acknowledgments (`right.`, `yeah.`, `got it.`,
`ok, noted.`). Relation phrases and candidate targets are searched only in
`ev_low`; negation/verdict detection still uses the full text, so a decline
still scores `reject`/`no_answer`. The explicit `relation=/target=/verdict=`
path is untouched. **No threshold, gate, arm, item bank, engine.py, or
response_gen.py was changed.**

Offline verification (direct calls through `RealBackend.present()` with a stub
engine — no engine boot, no pytest, as required):
- The three new test bodies pass: hedge-echo cannot supply relation or target;
  explicit `match` and a genuine `gale causes rainbow` assertion still parse;
  hedge-only decline still fails closed and the fake path stays byte-identical.
- Re-parsing all 1,440 saved live emissions with the new parser: **0 structural
  row-level verdict changes**; correct target stays 0/16 near and 0/16 far.
- Fake sweep on 4 items still yields near = 1.00 with no `engine_answer` fields.
- Both files compile (`py_compile`). The 3 new tests were written but **not run
  under pytest** (parent runs the suite).

## 5. Exactly what engine-side change is needed if the failure is upstream of articulation

The observation says the failure **is** upstream of articulation: the engine
never stores the demonstration and never receives the query. The required
`engine.py` / response path change is therefore:

1. **Pre-generation declarative-assertion route in `process_turn`.** Detect the
   harness's assertion surface `"<s> <phrase> <o>"` (including the multi-word
   relation phrases in `PHRASE_TO_REL` and the negated `"not <phrase>"` form),
   canonicalise `(s, relation, o)` with the same vocabulary as the harness, and
   write a **trusted** edge with real `source_metadata` provenance
   (conversation-assertion / curriculum tag) before any answer is generated.
   Today these strings are classified as unknown queries and fall through to
   the `_human_like_uncertainty` emitter (the 96.7% hedge class). Do **not**
   loosen any trust gate; stamp provenance at write time.
2. **Schema induction from converged traces.** Aggregate the trusted edges by
   relation type into an *unassertable* `schema_hypothesis`; at query time admit
   the schema only when >=2 independently taught edges converge, so
   schema-congruence alone cannot mint a fact (Tse/McClelland caution). Pattern
   separation requires the k demonstrations stay distinct traces until the
   relation type is abstracted.
3. **A query-time decision surface.** After teaching, answer the query subject
   against candidates with an explicit decision token (or forced single-token
   choice), with quoted/mirrored prompt text stripped before scoring, so the
   grader has an assertion to read. This alone will **not** help: steps 1–2 must
   first create something to decide about.

Verification required before accepting an engine fix: dump the k incident
edges (kind, relation_type, provenance) after teaching one item; confirm k
trusted edges and an assertable chain; only then re-run the focused live
harness with positive (near/far) **and** negative (foil/negative) metrics on
identical state.

## 6. Unverified items / caveats

- n = 2 items per pool (8 structural near/far rows per k); this is a mechanism
  probe, not the 120-item suite verdict.
- No engine was run for this note; F1 and F2 are unmeasured designs, not
  delivered capability. The parser fix (F3) is verified only against saved
  emissions and stub-engine calls; the parent must run pytest.
- The claim that the engine "does not store the demonstrations" is inferred
  from its emissions and from the absence of a query turn. A direct graph
  census after teaching was **not** performed here (would require an engine).
- The 26 unparsed rows are unevenly spread (mostly permuted/negative and
  confound arms; structural near k=1 has 0 unparsed).
- "Relation coverage" (`flags.relation_present`) is computed from the authored
  demo triples, not from the engine emission, so it is 1.00 for every live row
  and cannot be used as evidence of engine-side relation coverage. This is a
  separate instrument limitation, not changed here.
- Citations were verified via publisher/abstract records on 2026-11-23; page
  ranges and DOIs are as listed above. The schema false-memory caution is
  marked extrapolation rather than a verified claim from Tse et al.'s abstract.

## 7. Reproduction

```bash
# recompute the counts from saved rows (no engine)
python3 - <<'PY'
import json
L=json.load(open('C:/Users/Likhith/Documents/projects/ravana/scratch/rsi_focus_live_rows.json'))
F=json.load(open('C:/Users/Likhith/Documents/projects/ravana/scratch/rsi_focus_fake_rows.json'))
print('rows', len(L), 'responses==k', all(len(r['engine_responses'])==r['k'] for r in L))
print('structured tokens', sum('verdict=' in s.lower() for r in L for s in r['engine_responses']))
PY
```
