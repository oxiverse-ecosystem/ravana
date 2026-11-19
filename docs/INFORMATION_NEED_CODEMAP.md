# Where a structured information need and source filter enter the mechanism-gap loop

**Status:** read-only investigation at tree `6a2b5fa`. No code changed, no
harness edited, no engine booted, no `eng.save()`. Every claim below is
`file:line` against the on-disk tree; measurements quoted from other docs are
attributed, not re-run here.

**Scope:** answer five questions about the moment RAVANA detects an
explanatory gap, the fields a search hit carries, where a source filter can be
inserted without weakening the trust gate, what chain-shape object already
exists, and the narrowest change that converts the gap into a structured
information need. Companion docs: `GAP_DRIVEN_WEB_LEARNING.md` (why the loop
exists), `CAUSAL_SUBSTRATE_CODEMAP.md` (where facts die and which gates are
load-bearing), `BRAIN_CAUSAL_SUBSTRATE.md` (G1–G5), `ravana-debugging` skill
(instrument validation, provenance-at-write).

**Bottom line.** The trigger is fine and the write path is already trusted;
what is missing is a *specification* of the missing link. At gap time RAVANA
holds the subject, the raw `why/how` query, the direction, the failed
definition text, the graph neighbourhood and the prior-attempt registry — and
throws almost all of it away. `close_mechanism_gap` reduces the gap to a single
lowercased subject string, issues one direction-agnostic free-text query, takes
the first two hits verbatim, and calls `learn_text` on each. The narrowest fix
is **not** another query wording (that A/B was tried and reverted) but: build a
small `InformationNeed` from data already in scope, widen the *candidate pool*
while keeping the query string byte-identical, filter candidates for PROCESS
PROSE and IntentForge trust **signals** before `learn_text`, and leave every
provenance/trust gate untouched. The positive metric must be an
`enumerate_chains` → `assertable_chain` → `grade_chain` result, not the
`chain_hops` counter, which is an undirected connected-component count.

---

## 1. What concretely exists at the moment `_mechanism_gap` is set

Detection is after generation, inside `_process_turn_impl`:

- `_wants_mechanism` regex over `ctx.raw_input` — `engine.py:11831-11839`.
- `_got_mechanism` regex over the emitted `response` — `engine.py:11840-11846`.
- `_mechanism_gap = bool(_wants_mechanism and not _got_mechanism)` —
  `engine.py:11847`.
- The explanatory-only branch is `if (quality_score < 0.55 or
  _mechanism_gap) and ctx.subject and self.baby_mode` — `engine.py:11849`.
- The call site passes only two strings: `self.close_mechanism_gap(ctx.subject,
  (ctx.raw_input or ""))` — `engine.py:11902-11905`.

Everything below is in scope at `engine.py:11847-11905`.

| Field | Where it lives at gap time | Passed to `close_mechanism_gap`? |
|---|---|---|
| `subject` (grounded, singularised candidate already resolved) | `ctx.subject`, `models.py:66`; built `engine.py:11576-11578` | **Yes** — `engine.py:11904`, lowercased at `web_learning.py:684` |
| raw query text | `ctx.raw_input`, `models.py:69`; `ctx.cleaned_input`, `models.py:103` | **Yes as `query`** — `engine.py:11905`, but *never used to retrieve* (see below) |
| query type (`why` / `how` / `what causes` / `explain`) | derivable from `ctx.raw_input`; the exact regex is already written at `engine.py:11833-11837`, `response_gen.py:8496-8498` | **No** — only re-derived inside generation, not at the read |
| direction (`backward` for why, `forward` for how) | `engine.py:engine_self_query.py:2460-2465` and `response_gen.py:8548-8554` compute it from the same regex | **No** — `close_mechanism_gap` builds a direction-free query at `web_learning.py:715` |
| missing relation | *de facto* `causes` (`_got_mechanism` connective set, `engine.py:11843-11844`; `CAUSAL_RELATIONS`, `explanation.py:38`) | **No** — implicit in the fixed template `_GAP_CAUSAL_QUERY`, `web_learning.py:93` |
| existing graph neighbours / candidate hops | `_concept_keywords[subject]` + `_singular_candidates` → node ids → `graph.get_incoming/get_outgoing` filtered to causal; the working resolution code is `response_gen.py:8529-8554` and `engine_self_query.py:2449-2466` | **No** — the only graph access in the gap loop is the before/after causal-edge snapshot at `web_learning.py:728-729, 768-779` |
| definition text that failed (the non-answer) | `response` in scope at `engine.py:11842-11846`; `self._definitions` dict at `engine.py:955`, loaded `engine.py:13877`, 583 stored / 115 with causal verbs per `CAUSAL_SUBSTRATE_CODEMAP.md:164-168` | **No** |
| prior failed queries | `self._impossible_queries: List[FailedQuery]`, `engine.py:1108`; model `models.py:30-41` (`query`, `subject`, `turn`, `response_quality`, `strategy`, `resolved`); the just-failed item is appended `engine.py:11853-11861` | **No** — the loop has a *separate* dedup set `_gap_read_subjects`, `web_learning.py:696-702` |
| prior gap-read attempts | `_gap_read_subjects` set and `_gap_read_count`, `web_learning.py:696-708` | Internal only |
| quality score / strategy / baby_mode | `quality_score`, `strategy` in scope; used only for curiosity tags, `engine.py:11849-11889` | **No** |
| emit metadata (`authority`-style trust signals) | see §2 | **No** — only `title`/`content` read at `web_learning.py:740` |

### What is available but thrown away

1. **The free-text query is discarded at retrieval.** `query` is accepted at
   `web_learning.py:661` but the search string is built **only** from the
   lowercased subject: `causal_query = _GAP_CAUSAL_QUERY.format(subject=subj)`
   (`web_learning.py:715`). `query` reappears once, as the passive metadata tag
   `gap_query` (`web_learning.py:796`). So the `why` vs `how` distinction, the
   object/slot the question asks about (`ctx.object`, `ctx.relation`,
   `models.py:67-68`), and any reformulated wording never influence which
   sources are read. This is the structural reason query reformulation could
   not help: the retrieval path is already insensitive to it except through the
   fixed template.
2. **Direction is computed twice downstream but never at the read.**
   `engine_self_query.py:2460-2465` and `response_gen.py:8548-8554` both choose
   `backward`/`forward` from `ctx.raw_input`; `close_mechanism_gap` never calls
   either. A `why X` gap and a `how X work` gap issue the *identical* query.
3. **The missing hop is never represented.** `enumerate_chains` can enumerate
   the 1-hop neighbourhood that exists (`explanation.py:156-265`), and the
   resolution code at `response_gen.py:8529-8554` already knows how to find the
   subject's node ids. The gap loop uses neither: it does not ask "which middle
   node would make a 2-hop chain ending at X", it just learns whatever the two
   hits say.

   - Note the *measured* consequence, already documented: `chain_hops` is an
     undirected connected-component count (`web_learning.py:819-857`), not the
     directed provenance-trusted object `enumerate_chains` builds
     (`CAUSAL_SUBSTRATE_CODEMAP.md:103-106`). The reported `sources=2,
     chain_hops=0-1` failure is therefore a claim about that proxy; the
     confirmatory measurement must use the real chain object.
4. **The failed definition text is not reused as a comparator.** It is in
   scope as `response` at `engine.py:11842` and in `self._definitions`
   (`engine.py:955`), yet the loop cannot reject a retrieved hit for being
   another dictionary definition of the same kind.
5. **IntentForge's trust signals are dropped** (§2).
6. **`Fact.source_sentence` is dropped.** OpenIE returns it
   (`openie.py:144-154`), but the loop keeps only `subject/relation/obj/
   confidence` (`web_learning.py:748-754`), so there is no way to audit whether
   the extracted triple came from process prose or from a navigation blurb.

---

## 2. What a search hit carries, and whether it can be filtered before `learn_text`

`SearchEngine.search()` normalises every provider to a list of plain dicts.
The IntentForge branch is the richest (`web/learner.py:586-616`):

- provider order: IntentForge is tried **first** (`web/learner.py:178-182`,
  `:388-398`, `:420-443`), with local-api as the second local provider;
- `RAVANA_OFFLINE=1` fails closed to `[]` before any network call
  (`web/learner.py:145-155`, `:331-333`; the loop's own short-circuit is
  `web_learning.py:690-692`);
- the emitted hit dict (`web/learner.py:607-615`) has exactly:
  `title`, `url`, `content`, `authority` (float or `None`), `score`,
  `is_local` (bool), `sources` (list).

**Field answer:** `title`, `url`, `content` are always present. `snippet` is
**not** a separate downstream field — the normalizer folds it into `content`
as `r.get("content", r.get("snippet", ""))` (`web/learner.py:610`, same for
local_api at `:572-574` and oxiverse at `:580-582`). IntentForge additionally
surfaces `authority`/`score`/`is_local`/`sources`; the docstring at
`web/learner.py:166-174` says these exist specifically for downstream
source-trust ranking. So: yes — title/snippet-content/url are all available,
plus four unused trust signals.

**Can we filter for PROCESS PROSE before `learn_text`? Yes, cleanly.** The
gap loop reads only `title` and `content` into `text` at
`web_learning.py:739-740`, then calls `w2g.openie.extract` at `:745` and
`w2g.learn_text` at `:756`. Any predicate over `hit` / `text` inserted between
`:721` (search) and `:738` (hit loop) decides whether that URL is ever handed
to `learn_text`. A suitable process-prose test can be built entirely from
existing in-tree material:

- causal connectives already used to *detect* a mechanism in an answer:
  `leads to|causes?|because|which makes|so that|in turn|results in|produces`
  (`engine.py:11842-11844`);
- the process/phase-change verb lexicon mapped to `causes` in
  `web/openie.py:87-120` (`evaporate`, `condense`, `scatter`, `forms`,
  `leads`, …);
- multi-step sequence markers (then / after / first … next / which in turn)
  can be checked on `Fact.source_sentence` (`openie.py:151`), so the filter can
  verify the *evidence sentence* for a triple, not just the whole blob;
- `authority`/`score`/`is_local`/`sources` (`learner.py:607-615`) are ranking
  tie-breakers only — they are **not** provenance and must not be treated as a
  trust gate (see §3).

`_clean_snippet` (`web_learning.py:39-63`) is available if the filter wants
markup-free text; note the gap loop currently uses raw `hit['content']`, not
the cleaned form.

Concretely, the measured failure (`sources=2, chain_hops=0-1`) is consistent
with `search(max_results=cap)` at `web_learning.py:721` returning two
dictionary/navigation hits that are consumed unfiltered at `:738-756`. Because
`cap=2` (`web_learning.py:694-695`, constant `:94`), there is currently **no
candidate pool to filter** — the filter must be paired with requesting more
raw hits and selecting two, while leaving the query string unchanged.

---

## 3. Exact placement of a source-quality filter so it cannot weaken the trust gate

**Insertion point:** immediately after `hits = searcher.search(...)` at
`web_learning.py:721` and before `for hit in hits[:cap]` at
`web_learning.py:738`. Rank/filter the hit list there into at most
`_GAP_MAX_SOURCES` accepted candidates; the existing loop then operates on the
accepted list unchanged. Do **not** insert the filter inside the
`w2g.learn_text` call at `:756` (too late — extraction at `:745` and source
bookkeeping at `:743,752` would already have run) and do not add it to the
provenance-stamp block at `:768-806`.

Why this is gate-safe:

- The filter is **pre-admission**: a rejected URL never reaches
  `w2g.openie.extract` (`:745`), `w2g.learn_text` (`:756`), `sources_by_edge`
  (`:735-754`) or `populate_provenance` (`:788-793`). No edge is created for
  it, so there is nothing for a trust gate to admit or reject differently.
- Provenance is still stamped from the **real** `source_url` of each accepted
  hit (`web_learning.py:739, 752, 756, 788-793`) with
  `edge_kind="web_fact"`, `method="mechanism_gap_read"`,
  `retrieval_conf`, `converging_sources` (`:789-793`), `acquired_by`,
  `gap_subject`, `gap_query` (`:794-796`) — exactly as designed in
  `GAP_DRIVEN_WEB_LEARNING.md:68-69,82-88`.
- The gates that must remain byte-identical:
  - emitter `_edge_provenanced` — `response_gen.py:141-172` (checks
    `edge_kind in {web_fact, curated, definition, conversation, curriculum,
    co_occurrence, typed}` or an accepted `source`; fail-closed at `:156,
    171-172`). Note this emitter gate is *looser* than the chain gate — it
    accepts `co_occurrence` at `:165` — so the filter must not be excused by
    it; the real chain gate is the next item.
  - chain provenance `_hop_provenance` — `explanation.py:268-299`
    (`web_fact → "web"` at `:285-286`; `definition/curriculum/curated` at
    `:292-293`; `co_occurrence → ""` at `:294-295`).
  - `provenance_trusted` — `explanation.py:302-316` (accepts only
    `web/definition/conversation/curriculum/curated`).
  - `assertable_chain` — `explanation.py:319-328` (one bad hop sinks the
    whole chain).
  - `grade_chain`'s structural/provenance override —
    `explanation.py:331-366`, especially `:357-362`.
  - node-promotion convergence `_promote_min_sources = 2` —
    `web_to_graph.py:166-183`, configured `engine.py:2064`; do not try to
    recover "lost" facts by lowering it (`CAUSAL_SUBSTRATE_CODEMAP.md:88-91`).
- Explicitly forbidden by the project's own rules: lowering
  `provenance_trusted` / `_promote_min_sources`, accepting
  `schema_hypothesis` as evidence (`BRAIN_CAUSAL_SUBSTRATE.md:317,350-352`,
  `CAUSAL_SUBSTRATE_CODEMAP.md:212-234`), or fabricating a `web_fact`
  provenance stamp for a hit the filter rejected. A filtered-out source should
  yield `status="no_chain"` (`web_learning.py:816`), not a weakened gate.

---

## 4. Reusable "expected answer shape" object

Yes. The repo already has a directed, typed 2-hop shape; no new shape object is
needed, only a small need wrapper around it.

- `Hop` — `learn/explanation.py:51-65`: `from_label`, `relation_type`,
  `to_label`, `weight`, `confidence`, `provenance`,
  `source_metadata` (`:59-61` reads the edge's real dict), with `provenanced`
  at `:63-65`.
- `ExplanationChain` — `explanation.py:68-95`: `subject` (`:70`),
  **`direction: 'backward'|'forward'|'bidirectional'` (`:71`)**, `hops`
  (`:72`), `score`/`runner_up_score` (`:73-74`), and the `labels` property
  (`:76-91`) that returns the ordered node sequence for narration. This is
  precisely the object that answers "I need a 2-hop causal chain ending at X":
  `direction="backward"`, `subject=X`, `len(hops)>=2`.
- `enumerate_chains` — `explanation.py:156-265`: direction-aware traversal at
  `:204-207`, causal-only edges via `_edge_is_causal` at `:209`, simple-path
  and cycle guards at `:215-225`, **`min_steps=2` default at `:160`** and the
  min-steps emission check at `:245-252`. It already accepts a list of seed
  node ids and returns sorted chains with a runner-up margin (`:261-265`).
- Consumers that show the intended call shape:
  `engine_self_query.py:2462-2465` (`min_steps=2, max_steps=4, limit=3`,
  direction from `why`), and `response_gen.py:8551-8557` plus grading at
  `:8555-8559`.
- Grading/assertability: `grade_chain` `explanation.py:331-366` and
  `assertable_chain` `:319-328`.
- The weaker, **non-reusable** shape is `_gap_chain_hops`
  (`web_learning.py:819-857`): undirected connected components, with a
  substring fallback at `:854-857`. It cannot express direction or provenance
  and should not be the need object's target — it is the measurement trap
  called out in `CAUSAL_SUBSTRATE_CODEMAP.md:103-106` and in the
  `ravana-debugging` §1 rule "validate the instrument before reporting".

There is **no** existing `InformationNeed` dataclass (repo-wide search returns
only `schema_hypothesis` references in `BRAIN_CAUSAL_SUBSTRATE.md:317,350-352`
and `CAUSAL_SUBSTRATE_CODEMAP.md:215-216,281-282`). The narrow change therefore
adds one small local dataclass whose *target* is `ExplanationChain`'s shape;
it must not create or assert edges itself (source-memory rule §7a of
`ravana-debugging`: schemas propose, sources admit).

Brain grounding for the need object is already established in the companion
docs and is inherited, not re-argued here: the gap specifies what information
would resolve it (Loewenstein 1994, *Psychological Bulletin* 116(1):75–98 —
quoted `GAP_DRIVEN_WEB_LEARNING.md:22-28`), and the target shape is >=2
recombined causal hops (Schacter & Addis 2007, *Phil Trans R Soc B*
362(1481):773–786 — `GAP_DRIVEN_WEB_LEARNING.md:30-37`).

---

## 5. Narrowest correct change, and how to A/B it

### The change

Keep `_GAP_CAUSAL_QUERY` (`web_learning.py:93`) and the query construction at
`web_learning.py:715` **byte-identical** — query reformulation already failed
its A/B. Change only what happens to the *results*.

1. **Build the need from data already in scope**, without a call-site change.
   Inside `close_mechanism_gap` (after the subject/budget checks at
   `web_learning.py:684-708`, before `:715`), construct a small local
   `InformationNeed` carrying:
   - `subject=subj` (`:684`),
   - `query=query` (`ctx.raw_input`, `engine.py:11905`) retained for metadata,
   - `direction` from the same `why|what causes|what makes` regex used at
     `engine_self_query.py:2460-2461` / `response_gen.py:8548-8550`
     (`backward` else `forward`),
   - `missing_relation="causes"` (the `_got_mechanism` connective set,
     `engine.py:11843-11844`; `CAUSAL_RELATIONS`, `explanation.py:38`),
   - `target_hops=2` (`_GAP_MIN_HOPS`, `web_learning.py:96`),
   - `seed_ids` = `_concept_keywords[subj]` + `_singular_candidates(subj)`
     ids + `_graph_node_id_for_label` fallback, exactly the deduped list at
     `response_gen.py:8529-8546`,
   - `existing_neighbours` / `candidate_hops` = causal in/out edges of those
     seeds (`graph.get_incoming`/`get_outgoing`, `response_gen.py:3244-3253`),
   - `failed_definition` = `self._definitions.get(subj)` (`engine.py:955`),
   - `prior_failed` = matching `self._impossible_queries` entries
     (`engine.py:1108`, `models.py:30-41`) plus `_gap_read_subjects`
     (`web_learning.py:696-702`).
   This is a *description* of the gap; it writes no edge and asserts nothing.
   Optionally pass `ctx` from `engine.py:11904-11905` instead of recomputing —
   the narrower change is to recompute inside, leaving `engine.py` untouched.
2. **Widen only the candidate pool.** At `web_learning.py:721`, request a
   larger pool (e.g. a new `_GAP_MAX_CANDIDATES ≈ 8`) with the **same**
   `causal_query`, then filter/rank down to `cap = _GAP_MAX_SOURCES` before the
   loop at `:738`. Query text unchanged; provenance unchanged.
3. **Filter candidates for process prose** using the signals in §2: require at
   least one causal connective (`engine.py:11843-11844`) or a `causes`-lexicon
   process verb (`openie.py:87-120`) in `hit['content']` (and ideally in the
   matched `Fact.source_sentence`, `openie.py:151`); prefer multi-step sequence
   language; reject bare `X is a Y` definitions and Wikipedia navigation
   blurbs; break ties with IntentForge `authority`/`score`/`is_local`/`sources`
   (`learner.py:607-615`) as ranking only. Record rejected hits in the result
   dict for the A/B (e.g. `result["rejected_sources"]`).
4. **Leave the loop and the gates as-is.** `openie.extract` (`:745`),
   `learn_text` (`:756`), per-URL provenance (`:768-806`), `_gap_chain_hops`
   (`:811`), `_edge_provenanced` (`response_gen.py:141-172`),
   `_hop_provenance`/`provenance_trusted`/`assertable_chain`/`grade_chain`
   (`explanation.py:268-366`) and `_promote_min_sources=2`
   (`web_to_graph.py:182-183`, `engine.py:2064`) are all untouched. A
   filtered-out source simply means fewer `read_sources` (`:760`) and an
   honest `status="no_chain"` (`:816`).

This targets the measured failure exactly: with `sources=2` but `chain_hops=0-1`
and definition/navigation text, the fix is *which two sources are read*, not
how the query is spelled and not what the trust gate accepts.

### A/B protocol (one process, identical state, no `save()`)

Follow `ravana-debugging` §1 and §7: validate the probe, log every row, use a
fixed seed and a held-out split by `sha256(query|difficulty|subject)`, and
adopt only if the negative metrics stay clean. Probes run with
`RAVANA_OFFLINE=1` where possible; web arms need a large timeout and must check
row counts against the expected total before trusting a summary.

- **Arms:** baseline (current `close_mechanism_gap`) vs filtered, in the same
  process on a frozen in-memory graph copy; no `eng.save()` on `weights/`.
- **Positive (must rise):** for each gap subject, after the read run
  `enumerate_chains(seed_ids, direction=need.direction, min_steps=2,
  max_steps=4)` → `assertable_chain` → `grade_chain`, and count subjects with
  `verdict == "assert"` and `assertable_chain is True`. That is the real
  structure; print the engine's `sources`/`chain_hops` too but label
  `chain_hops` a weak proxy (`web_learning.py:819-857`,
  `CAUSAL_SUBSTRATE_CODEMAP.md:103-106`). Sample size >= 20 held-out subjects;
  report primary and sealed pools side by side.
- **Instrument check (do this first):** print each candidate hit's
  `title/url/content` prefix, the filter's accept/reject decision and the
  reason, and the exact `Fact` triples extracted. Hand-check that a rejected
  definition and an accepted process paragraph are classified as intended —
  a moved number that is only a moved probe is not a result.
- **Negative (must stay zero/unchanged):**
  1. zero new edges whose `source_metadata.edge_kind` is missing or outside
     the trusted set (`explanation.py:315-316`, `response_gen.py:164-169`);
  2. zero `co_occurrence` / `schema_hypothesis` / unstamped edges added by
     either arm — prove `assertable_chain` still rejects them
     (`CAUSAL_SUBSTRATE_CODEMAP.md:295-296`);
  3. poison control: a bare dictionary-definition hit must be rejected by the
     filter and produce zero new edges (this is the specific measured failure);
  4. convergence unchanged: a one-URL-only node still writes zero edges
     (`web_to_graph.py:182-183`, `engine.py:2064`);
  5. `RAVANA_OFFLINE=1` still returns `status="offline"` with zero network
     calls (`web_learning.py:690-692`, `learner.py:145-155,331-333`);
  6. no `eng.save()` and no mutation of `weights/ravana_weights.{pkl,db}`;
  7. `chain_hops` rising while the assertable-chain count does not is a FAIL,
     not a win (it would mean the undirected proxy gamed the metric).
- **Interpretation:** adopt only if the positive count rises on held-out items
  *and* all negatives stay clean. If the filtered arm has fewer `sources` but
  the same assertable-chain count, report the honest flat result and the
  located bottleneck (likely `_promote_min_sources=2` or OpenIE's
  one-fact-per-sentence loss, `CAUSAL_SUBSTRATE_CODEMAP.md:127-141`) as the
  next work item — do not lower a gate to make the number move.

---

## Explicitly not the change

- Rewording `_GAP_CAUSAL_QUERY` (`web_learning.py:93`) — already A/B-tested and
  reverted.
- Lowering `_promote_min_sources=2` (`engine.py:2064`), relaxing
  `provenance_trusted`/`assertable_chain`/`grade_chain`/
  `_edge_provenanced`, or exempting gap-written edges from them.
- Accepting `schema_hypothesis` as evidence, relabelling an existing verified
  chain's nodes, or fabricating a `web_fact` stamp for a rejected source
  (`BRAIN_CAUSAL_SUBSTRATE.md:317,319,350-352`;
  `CAUSAL_SUBSTRATE_CODEMAP.md:212-234,281-284`).
- Treating IntentForge `authority`/`score` as provenance — they are ranking
  signals (`learner.py:166-174,586-615`), not statements of the causal
  relation.

## Files consulted (read-only)

`chat/web_learning.py` (`:39-98, 661-857`), `chat/engine.py`
(`:955, 1108, 2064, 11570-11601, 11790-11916`), `chat/models.py`
(`:30-41, 64-105`), `chat/engine_self_query.py` (`:2436-2468`),
`chat/response_gen.py` (`:141-172, 3244-3253, 8447-8586`),
`chat/provenance.py` (`:168-197`), `learn/explanation.py` (`:1-366`),
`web/learner.py` (`:140-200, 323-545, 547-643`), `web/openie.py`
(`:38-255`), `web/web_to_graph.py` (`:126-183, 206-298`),
`docs/GAP_DRIVEN_WEB_LEARNING.md`, `docs/CAUSAL_SUBSTRATE_CODEMAP.md`,
`docs/BRAIN_CAUSAL_SUBSTRATE.md`, `ravana-debugging` skill.
