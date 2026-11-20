# Full-page extraction for causal substrate growth — decision-ready design

**Status:** design only. This document makes no code change. It owns exactly one
file (`docs/FULL_PAGE_EXTRACTION_DESIGN.md`); scripts, fixtures, tests, and all
other docs belong to other agents.

**Decision in one line:** the gap-driven read is bottlenecked at the INPUT, not
at query phrasing, the source filter, or the reasoning gates. Keep the query and
every trust gate byte-identical; replace the search *snippet* fed to OpenIE with
the fetched *page body* for the top candidates, require two distinct URLs per
causal triple, and stamp provenance at the same write point the current loop
already uses. A/B on identical state over >=20 held-out probes; no `eng.save()`.

---

## 0. Measured context this design starts from

These are prior measurements, not claims re-derived here (see the cited docs).

| Attempt | Result | Source |
|---|---|---|
| Query reformulation | no gain; the retrieval path is insensitive to the wording except through a fixed template | `docs/INFORMATION_NEED_CODEMAP.md:65-74` |
| Bounded retry over the same corpus | no gain | `docs/RAIN_DEFINITION_GAP.md:65-83`; `ravana-debugging` §online-learning |
| Typed `InformationNeed` + veto-only source filter, 8 candidates | `iron rust candidates=8 admitted=2 causal_hops=0 chain_hops=0`; `ocean tides` same; code REVERTED | `docs/BRAIN_INFORMATION_NEED.md:449-476` |
| Authored curriculum scaffold (53 trusted edges) | `assertable_chain`/`grade_chain` work; controlled thinking A/B 5 -> 13 | `docs/CURRICULUM_SCAFFOLD.md:22-45`; task context |
| Current gap loop | up to 2 sources, but the text handed to OpenIE is `f"{title}. {content}"` — a snippet | `web_learning.py:740` |

The scaffold is **scaffolding, not learning** (`CURRICULUM_SCAFFOLD.md:1-7, 62-65`);
it proves the downstream reasoning pipeline works when substrate exists. The
absence of substrate is therefore located upstream of `enumerate_chains`.

---

## 1. Brain basis: why a whole page, and why a gist snippet cannot do this

### 1.1 Situation models require relations across the whole text

- **van Dijk, T. A., & Kintsch, W. (1983).** *Strategies of Discourse
  Comprehension.* New York: Academic Press. ISBN 0127120505. Ch. 10, "From the
  text representation to the situation model": comprehension builds a
  *textbase* (a propositional gist of what the text states) **and**, in
  parallel, a *situation model* of the events, participants, and situation the
  text is about; the situation model controls the updating and use of the
  textbase.
- **Kintsch, W. (1998).** *Comprehension: A Paradigm for Cognition.* Cambridge
  University Press. ISBN 9780521629867. Construction-Integration: a shallow
  textbase is integrated with prior knowledge into a situation model;
  "comprehension will be shallow since the reader understands only what is
  explicitly stated" when the situation model is not built.

A search snippet is a maximally compressed textbase fragment. It contains the
head proposition of a page, often a definition or a title, and by construction
drops the connecting sentences. RAVANA cannot build a situation model — a model
of *how one event leads to another* — from a single decontextualized
proposition, because the causal relations live in the sentences the snippet
omitted. This is not a wording problem; the information is absent from the
input. `docs/RAIN_DEFINITION_GAP.md:65-83` is the concrete case: the top hits
for `rain` are an aquarium exhibit and a weather-radar title, while the
definition-shaped sentence sits further down the same result set and the
extractor only reads the top result.

### 1.2 Causal inference in comprehension is a network property

- **Trabasso, T., & van den Broek, P. (1985).** Causal thinking and the
  representation of narrative events. *Journal of Memory and Language, 24*(5),
  612–630. DOI 10.1016/0749-596X(85)90049-X. Causal networks are derived by
  necessity/counterfactual tests over **all pairs of events in the text**;
  membership in a causal chain from opening to closing, and the number of
  causal connections, predict recall, summarization, and judged importance.
- **Trabasso, T., & Sperry, L. L. (1985).** Causal relatedness and importance of
  story events. *Journal of Memory and Language, 24*(5), 595–611.
  DOI 10.1016/0749-596X(85)90048-8. Importance tracks the number of direct
  causal connections and chain membership, not mere mention.

The operational consequence for RAVANA: a causal *hop* is a relation between
two events in the same text, and a *mechanism* is a chain of such hops. A
snippet typically contains one event, or two events in a definitional (`is_a`)
rather than counterfactual (`causes`) relation. Reading more of the source is
what makes the second and third event available for the counterfactual test
the extractor implements as an explicit causal verb. This is exactly the
`_GAP_MIN_HOPS = 2` requirement at `web_learning.py:96` and the `min_steps=2`
chain gate in `explanation.py:121-124`.

### 1.3 Active inference and epistemic value bound the read

- **Friston, K. (2010).** The free-energy principle: a unified brain theory?
  *Nature Reviews Neuroscience, 11*(2), 127–138. DOI 10.1038/nrn2787.
  Perception/action minimize free energy; epistemic value is bounded.
- **Friston, K., Rigoli, F., Ognibene, D., Mathys, C., Fitzgerald, T., &
  Pezzulo, G. (2015).** Active inference and epistemic value. *Cognitive
  Neuroscience, 6*(4), 187–214. DOI 10.1080/17588928.2015.1020053. Expected
  free energy decomposes into extrinsic and **epistemic** (information-gain)
  value; epistemic value is maximized only while information gain remains,
  which licenses a small, bounded sample and a hard stop once the page has
  been read.
- **Loewenstein, G. (1994).** The psychology of curiosity. *Psychological
  Bulletin, 116*(1), 75–98. DOI 10.1037/0033-2909.116.1.75. A gap is a
  deprivation that drives information-seeking selected by its estimated power
  to close *that* gap.

This is why the change stays bounded (candidate cap, byte cap, time cap,
per-session read cap) and fail-open: once the page body is sampled, further
query reformulation has no epistemic value (it re-samples the same
distribution), which is precisely the measured negative result.

### 1.4 Why the existing gates must not move

- **Johnson, M. K., & Raye, C. L. (1981).** Reality monitoring. *Psychological
  Review, 88*(1), 67–85. DOI 10.1037/0033-295X.88.1.67. Retrieved structure must
  be distinguishable from internally generated association; a trust gate that
  admits unprovenanced co-occurrence destroys that distinction.
- **Tse, D., et al. (2007).** Schema acceleration of relational memory
  formation. *Science, 316*, 76–82. And **Griffiths, T. L., & Tenenbaum, J. B.
  (2009).** Theory-based causal induction. *Psychological Review, 116*(4),
  661–716. A schema proposes candidates, but convergence from sources admits
  them; schema-congruence is also a principal false-memory source. This is the
  `schemas propose; sources admit` rule in `ravana-debugging` §7a and
  `CURRICULUM_SCAFFOLD.md:11-18`.

So the design changes what text is read, never what counts as evidence.

---

## 2. Codebase map with file:line evidence

All paths relative to `C:/Users/Likhith/Documents/projects/ravana`.

### 2.1 Search hits -> `close_mechanism_gap`

1. **Entry and bounds** — `ravana/src/ravana/chat/web_learning.py:661-708`.
   `close_mechanism_gap(subject, query="", max_sources=None)`. Constants at
   `:93-98`: `_GAP_CAUSAL_QUERY`, `_GAP_MAX_SOURCES = 2` (`:94`),
   `_GAP_MAX_READS = 4` (`:95`), `_GAP_MIN_HOPS = 2` (`:96`), base/confidence
   scalars (`:97-98`). Offline short-circuit `:690-692`; subject cap clamped to
   `[1, 2]` at `:694-695`; once-per-subject dedup `:696-702`; per-session read
   cap `:703-708`. Fail-open contract stated in the docstring `:671-679`;
   **no `eng.save()`** (`:677-678`).
2. **Graph handle and query** — `:710-721`. `_get_web_to_graph()` at `:710`;
   `causal_query = _GAP_CAUSAL_QUERY.format(subject=subj)` at `:715` (the free
   `query` argument is never used to build the search string — see
   `INFORMATION_NEED_CODEMAP.md:65-74`); `searcher.search(causal_query,
   max_results=cap)` at `:721`.
3. **Before-snapshot** — `:727-729`: `before = {(s,t) ... if relation_type ==
   "causal"}`, so only newly written edges are stamped.
4. **The extraction loop — THIS IS THE INPUT BOTTLENECK** — `:738-759`:
   - `:739` `url = str(hit.get("url") or "").strip()`
   - `:740` `text = f"{hit.get('title', '')}. {hit.get('content', '')}".strip()`
     — **the entire input to extraction is the search snippet.**
   - `:745` `facts = w2g.openie.extract(text)`
   - `:748-754` only `f.relation == "causes"` facts are kept; each normalized
     `(subject, obj)` key accumulates a set of source URLs and a max confidence.
   - `:755-756` `w2g.learn_text(text, source_url=url or subj, context=subj)`.
   - `:746-747, 757-759`: swallowed exceptions.
5. **Provenance stamping** — `:766-806`: `label_by_id` (`:766-767`), newly
   written causal edges only (`:768-774`), confidence/convergence-scaled
   strength (`:775-776`), `_ensure_source_metadata()` for unpickled edges
   (`:779-786`), `populate_provenance(edge, edge_kind="web_fact", source=...,
   source_url=..., method="mechanism_gap_read", retrieval_conf=...,
   converging_sources=len(urls))` (`:788-793`), plus
   `acquired_by`/`gap_subject`/`gap_query` (`:794-796`), and
   `causal_hops += 1` (`:802`).
6. **Closure test** — `:808-816`: `chain_hops = _gap_chain_hops(...)`; status
   `learned` only if `chain_hops >= _GAP_MIN_HOPS`.
   Caveat for the A/B: `_gap_chain_hops` (`:819-857`) is an **undirected
   connected-component hop count**, not the directed, provenance-trusted object
   `enumerate_chains` builds (`explanation.py:156-265`). It is a useful loop
   counter but must NOT be the reported success metric; use `assertable_chain`
   and `grade_chain` (§5).
7. **Call site** — the gap detector in `chat/engine.py` `_process_turn_impl`;
   the wired contract is documented at `docs/GAP_DRIVEN_WEB_LEARNING.md:6-9,
   90-104`.

### 2.2 What a hit carries

- **Canonical shape** — `ravana/src/ravana/web/learner.py:571-582` (local API /
  remote results): each hit is `{"title", "url", "content"}` where `content`
  falls back to `snippet`. `:609-610` same for the IntentForge branch.
- **Extra trust signals already present but unused by the gap loop** —
  `learner.py:589-614`: `authority` (0..1), `score`, `is_local`, `sources`.
  `INFORMATION_NEED_CODEMAP.md:61` confirms the gap loop reads only
  `title`/`content` at `web_learning.py:740`. Availability of these signals is
  why a veto filter was tryable; their presence does not supply the missing
  prose.
- **DuckDuckGo HTML parse** — `learner.py:619-642` produces the same
  `title`/`url`/`content` shape (`:633`, `:641-642`).

### 2.3 Where a page-text fetch goes, and what HTML parsing already exists

- **Existing full-page fetcher** — `ravana/src/ravana/web/learner.py:751-775`
  `_fetch_article_text(url)`:
  - `:753-755` `urllib.request.Request(url, headers=self.search_engine._headers)`,
    `urlopen(req, timeout=8)`, `resp.read().decode(...)` — note the read at `:755`
    is **unbounded by bytes**; a new call site must bound it.
  - `:758-761` `trafilatura.extract(..., favor_precision=True)`, cap `[:5000]`.
  - `:763-769` `BeautifulSoup(html, 'html.parser')`, decompose
    `script/style/nav/header/footer`, prefer `article`/`main`/`body`,
    `get_text(separator=' ', strip=True)`, cap `[:5000]`.
  - `:770-773` raw regex strip fallback, cap `[:3000]`.
  - `:774-775` `except Exception: return None` (fail-open).
- **Optional dependencies** — `learner.py:63-71` `bs4` and `:73-81`
  `trafilatura`, both genuinely optional with `_import_guard.report_missing`.
- **Existing consumer** — `learn_from_web` at `learner.py:701-749`: `:731`
  calls `_fetch_article_text(url)`, `:733` uses `article_text[:5000]`, and
  `:738-741` fetches the top 3 results in a 4-worker `ThreadPoolExecutor`.
  This is the proof the fetcher works in this codebase — it is simply **not
  called** by `close_mechanism_gap`.
- **Insertion point:** inside the `:738-759` loop, replace `:740` with a helper
  that prefers `_fetch_article_text(url)` (bounded) and falls back to the
  current snippet string when the fetch returns `None`. Everything downstream
  (`:745`, `:756`, `:766-806`) is unchanged. This is the narrowest change that
  alters the substrate.

### 2.4 Limits that already exist (reuse, do not raise)

| Limit | Value | Location |
|---|---|---|
| socket/HTTP timeout (search) | `SearchConfig.timeout = 5` | `learner.py:88` |
| local engine timeout | `local_timeout = 10` | `learner.py:105` |
| whole-search wall deadline | `search_deadline = 14.0` | `learner.py:132` |
| transient retries | `transient_retries = 5` | `learner.py:128` |
| article fetch timeout | `urlopen(timeout=8)` | `learner.py:754` |
| article text cap | 5000 chars (3000 raw fallback) | `learner.py:761, 769, 773` |
| gap source cap | `_GAP_MAX_SOURCES = 2` | `web_learning.py:94`, clamp `:694-695` |
| gap per-session read cap | `_GAP_MAX_READS = 4` | `web_learning.py:95`, check `:704-708` |
| gap hop minimum | `_GAP_MIN_HOPS = 2` | `web_learning.py:96` |
| node promotion by distinct source | `_promote_min_sources = 2` | `engine.py:2064`; used `web_to_graph.py:149, 182-183` |
| fact convergence threshold | `CONVERGENCE_FACT_THRESHOLD = 2` | `provenance.py:59`, used `:117-123` |
| junk gate | `_junk_theta = 0.5` | `engine.py:2063`; `web_to_graph.py:148, 157` |

**Known separate blocker (do not conflate):** `docs/WEB_NODE_ADMISSION_BLOCKED.md`
reports the fitted junk scorer degenerates to `0.501` for ordinary nouns, so
`_ensure_node` (`web_to_graph.py:126-165`) rejects them at the junk gate before
the promotion gate. Full-page text supplies more sentences and more distinct
URLs, but if the junk classifier is still collapsed the new nodes will still be
rejected. The full-page change is necessary but may not be sufficient; measure
node admission separately (§5 negative criteria) and hand the junk-scorer defect
back as its own work item rather than weakening the gate here.

---

## 3. Provenance contract — stamp, do not loosen

The new path writes through the **same** `web_to_graph.learn_text` and the
**same** `populate_provenance` call as the current loop. Nothing below changes:

- `ravana/src/ravana/chat/provenance.py:168-197` `populate_provenance` sets
  `edge_kind`, `source`, `source_url`, `method`, `retrieval_conf`, `timestamp`,
  `converging_sources`, and `epistemic_status` (`"fact"` iff `edge_kind ==
  "web_fact"`, `:196`). `web_fact` + a real URL is the verifiable class
  (`provenance.py:85-114`: `provenance_class` -> `VERIFIED_WEB` only when a
  `source_url`/`source`/`retrieval_conf` is present at `:91-93`).
- `ravana/src/ravana/learn/explanation.py:268-299` `_hop_provenance` maps
  `web_fact` -> `"web"` (`:285-286`); **`co_occurrence` -> `""` (`:294-295`)**,
  i.e. unassertable. `:302-316` `provenance_trusted` allows only
  `web|definition|conversation|curriculum|curated`. `:319-328`
  `assertable_chain` fails closed. `:331-366` `grade_chain` explicitly notes
  that a structurally perfect chain of unprovenanced co-occurrence edges still
  grades 1.000 and is false (`:357-362`), so it downgrades to `decline` unless
  `assertable_chain`.
- `ravana/src/ravana/web/web_to_graph.py:285-296` stamps `edge_kind="web_fact"`
  on every OpenIE edge at write time; `learn_text` also dedupes by existing
  edge (`:247-278`), so repeated reads bump confidence rather than fork nodes.
- Two-distinct-source rule is enforced in **three** places and all three stay
  untouched: node promotion by distinct `source_url` md5
  (`web_to_graph.py:171-183`, `engine.py:2064`), fact admission
  (`provenance.py:59, 117-123`), and the gap loop's convergence scaling
  (`web_learning.py:776`).

**Stamping contract for the new path (exact location):** the existing
`web_learning.py:788-796` block. The new path must additionally require that
`len(urls) >= 2` for a triple before it is written/stamped:

1. Build `sources_by_edge` (`:735`, populated `:752`) from **page** extractions
   keyed on normalized `(subject, obj)`, with the real page URL as the set
   member. A page and its search snippet from the same URL count as ONE source.
2. Only for keys with `>= 2` distinct URLs, call `populate_provenance` at
   `:789-793` with `source_url = sorted(urls)[0]` and
   `converging_sources = len(urls)`; add `acquired_by="full_page_extraction"`
   and keep `gap_subject`/`gap_query` (`:794-796`). `edge_kind` remains
   `web_fact`; do **not** set `co_occurrence`, and do **not** invent a trust
   kind.
3. Single-source triples are held in the local `sources_by_edge` buffer only —
   not written as trusted edges, and not labelled `schema_hypothesis` as a
   shortcut. There is no `schema_hypothesis` token in the current source (grep:
   0 hits); it remains an unassertable design-class whose only real enforcement
   is the closed allow-list at `explanation.py:315-316`. If a future schema
   proposer emits candidates, they must never be relabelled as `web_fact`.
4. Keep strengths at `_GAP_BASE_CONF`/`_GAP_CONF_PER_SOURCE` (`:97-98`, `:776`);
   do not raise `confidence`, `weight`, or any trust threshold to make a curve
   move.

**Never weaken:** `assertable_chain`, `grade_chain`, `_hop_provenance`,
`provenance_trusted`, `admit_as_fact`, `_promote_min_sources`, `_junk_theta`,
the offline gate (`learner.py:155`; `web_learning.py:690-692`), or the emit
guard's chain pass-through (`docs/GAP_DRIVEN_WEB_LEARNING.md:71-72`).

---

## 4. Narrowest correct change

**Scope:** one function, `close_mechanism_gap` (`web_learning.py:661-817`), plus
a small private helper for bounded page text. No query, config, gate, or schema
change. Ship behind a feature flag, default OFF until §5 passes, per
`brain-faithful-switch-rule` ("ship the gate disabled when the measurement says
the generative path loses").

Algorithm:

1. **Keep byte-identical:** `_GAP_CAUSAL_QUERY` (`:93`), query construction
   (`:715`), source cap (`:94, 694-695`), read cap (`:95, 703-708`), dedup
   (`:696-702`), offline short-circuit (`:690-692`).
2. **Fetch page body for top candidates.** In the `:738-759` loop, for each
   `url`, call a bounded wrapper around `learner.py:751-775`
   `_fetch_article_text(url)`. Prefer trafilatura; bs4 `article/main/body`;
   regex fallback. Use the returned text when it is longer than the snippet;
   otherwise fall back to the existing `f"{title}. {content}"` (`:740`) so the
   current behaviour is a strict subset of the new one.
3. **Bound size and time.** Before reading the body, wrap `resp.read()` with a
   byte cap (reuse the 5000-char output cap at `learner.py:761/769/773` as the
   post-extraction cap, and add a hard read cap on the socket side, e.g.
   `resp.read(MAX_PAGE_BYTES)` with `MAX_PAGE_BYTES` a small constant). Reuse
   `urlopen(timeout=8)` (`:754`) and the `search_deadline = 14.0` (`:132`)
   ceiling; do not raise any timeout. Cap the number of fetched URLs at
   `_GAP_MAX_SOURCES` (already `:721`).
4. **Run existing OpenIE over paragraphs and sentences.** Do not write a new
   extractor. `w2g.openie.extract(text)` (`:745`) already splits on sentences
   via `_split_sentences` (`openie.py:157-161`) and maps process verbs to
   `causes` (`openie.py:87-120`); the phase-change/process lexicon was added
   precisely because the original 9-verb set extracted zero facts from rain
   prose (`openie.py:73-86`). Feed full page text (optionally split into
   paragraph blocks and concatenated with sentence boundaries preserved) so
   each sentence is an independent extraction unit, as the literature requires
   relations between all event pairs. Keep `max_facts` default (200,
   `openie.py:219`).
5. **Require two distinct URLs per causal triple.** Populate `sources_by_edge`
   from page extractions with the real URL (§3). Write/stamp only keys with
   `>= 2` distinct URLs. This mirrors the node/fact promotion rules already in
   force and prevents a single page's incidental cross-product from becoming a
   trusted assertion.
6. **Dedupe and fail-open.** `learn_text` dedupes by edge (`:247-278`); the
   before-snapshot (`:727-729`) prevents re-stamping old edges. Any exception in
   fetching, parsing, extraction, or stamping is swallowed exactly as today
   (`:746-747, 757-759, 799-801`; `learner.py:774-775`); a failed fetch degrades
   to snippet, never breaks the turn. Offline path stays a no-op (`:690-692`,
   `learner.py:155`).
7. **Persistence unchanged.** No `eng.save()` in this method (`:677-678`); the
   normal engine save path persists pkl + db mirror.

**Non-goals / out of scope:** changing the query, adding a new retriever,
raising thresholds, touching the junk scorer, writing `co_occurrence` edges,
or counting undirected `chain_hops` as success.

---

## 5. Observable criteria and A/B

Instrument must be validated before the number is trusted (`ravana-debugging`
§1: print each probe's input and output; >=20 held-out probes or it is a
hypothesis; fixed hash split; no score without gates).

### Positive metrics (both must move)

- `assertable_chain(enumerate_chains(..., min_steps=2))` from the indexed
  subject node, plus `grade_chain` verdict, on **>=20 held-out probes**. Report
  count and verdict distribution, not just an aggregate.
- `grade_chain` verdict rising (`decline/hedge -> assert`) on the same probes.

### Negative metrics (must stay at zero / unchanged)

- **Zero unprovenanced edges:** every new causal edge has
  `edge.source_metadata["edge_kind"] == "web_fact"` **and** a real
  `source_url` **and** `converging_sources >= 2`.
- **Zero poisoned definitions:** `self._definitions` unchanged; no
  dictionary/navigation chrome written as a definition or as a causal edge
  (`docs/RAIN_DEFINITION_GAP.md:65-83` is the failure to prevent).
- **Offline path unchanged:** with `RAVANA_OFFLINE=1`, `close_mechanism_gap`
  returns `status="offline"` and writes nothing (`web_learning.py:690-692`).
- **No `eng.save` on weights:** assert no call to `eng.save()` in the probe;
  use an isolated `data_dir` copy, as in
  `docs/GAP_DRIVEN_WEB_LEARNING.md:106-130`.
- **Gate integrity:** `assertable_chain` still rejects a synthetic
  `co_occurrence` chain and a `schema_hypothesis` chain
  (`explanation.py:294-295, 319-328`).

### A/B protocol (identical state, one process)

- Same engine, same frozen weights, same probe list, same `RAVANA_OFFLINE`
  setting, same `max_results` and budgets.
- **Arm A (control):** flag off; current snippet path at `web_learning.py:740`.
- **Arm B (treatment):** flag on; full-page path of §4.
- Assign primary/sealed pools by `sha256(front|difficulty|subject)` **before**
  either run; record manifest hashes. Report primary-only and full-set scores
  side by side; the held-out score is the baseline.
- Run both arms in ONE process (cold boot ~200 s; never concurrently with other
  agents' engine probes). Reset `self._gap_read_subjects` and
  `self._gap_read_count` between arms, or use distinct engine instances from
  the same pristine snapshot, so the per-subject dedup (`:696-702`) does not
  suppress arm B.
- Print per-probe: subject, query, fetched URL(s), extracted triples, per-hop
  `edge_kind`, `source_url`, `converging_sources`, `assertable_chain` verdict,
  `grade_chain` verdict. Report the paired delta on probes where both arms
  produced a chain, with denominators.
- Decision rule: adopt only if positive metrics rise AND all negative metrics
  hold; otherwise record the honest negative and revert (as done in
  `docs/BRAIN_INFORMATION_NEED.md:466-468`). Do not re-tune the probe to make
  the number move.

---

## 6. Interaction with frozen snapshots / offline testability

Another agent is building a web harness with fixture-backed search. The page
fetch must be capturable on the same footing, or the A/B in §5 cannot be
reproduced.

- **Capture page bytes once, checked in.** For each probe URL, store the raw
  HTTP response body (not the extracted text) plus the status code and the
  final URL after redirects, in a fixture directory keyed by a stable hash of
  the URL. Keeping raw bytes preserves the trafilatura -> bs4 -> regex fallback
  chain (`learner.py:758-773`) as the thing under test.
- **Add one page-fetch seam**, not a new backend: a single function (the bounded
  wrapper of §4.2) that, when a fixture directory is configured, returns the
  fixture bytes through the *same* trafilatura/bs4 path; otherwise calls
  `_fetch_article_text(url)`. The gap loop calls this seam so offline replay
  and live runs exercise identical code. Do not branch on `RAVANA_OFFLINE`
  inside the parser — the offline contract at `web_learning.py:690-692` must
  remain an early no-op.
- **Record the search snapshot too:** store the `title`/`url`/`content` hit list
  (and `authority`/`score`/`is_local` if kept) so `searcher.search` at
  `web_learning.py:721` is replayable. Search fixture and page fixture must
  agree on URLs; a missing page fixture should degrade to the snippet fallback
  (`:740`) and be logged as such, not silently pass.
- **Determinism:** the A/B must replay byte-identically from identical state.
  No RNG in candidate ordering; sort URLs (`:788` already does `sorted(urls)[0]`).
  Stamp `timestamp` only through `populate_provenance` (`provenance.py:193`);
  if the harness compares provenance dicts, mask that field.
- **Coverage check before reporting:** count probe rows against the expected
  total and check the summary file's mtime; `EXIT=124` or a process that dies
  without an `EXIT` line means incomplete (`ravana-debugging` §2). A timed-out
  run leaves a stale summary that looks valid.
- **Do not save weights:** fixtures + no `eng.save()` make the run idempotent.

---

## 7. Decision summary

- **Change:** replace the snippet at `web_learning.py:740` with a bounded full
  page body from the existing `learner.py:751-775` fetcher; keep OpenIE
  (`openie.py:219-255`) and all provenance/trust code unchanged; require two
  distinct URLs per causal triple before stamping at `web_learning.py:788-796`.
- **Why:** situation models and causal-network comprehension need relations
  across the whole text; a snippet is a textbase fragment with no second
  event, which is exactly the measured `causal_hops=0` failure across three
  input-side attempts.
- **Proof it works:** `assertable_chain` + `grade_chain` rising on >=20
  held-out probes, zero unprovenanced or poisoned edges, offline path and
  `eng.save()` untouched.
- **Honest caveat:** `_gap_chain_hops` is undirected and the junk classifier may
  still be collapsed (`WEB_NODE_ADMISSION_BLOCKED.md`); neither is fixed by
  this change, and neither is a reason to weaken a gate.

## Peer-reviewed references
1. van Dijk, T. A., & Kintsch, W. (1983). *Strategies of Discourse Comprehension.* Academic Press.
2. Kintsch, W. (1998). *Comprehension: A Paradigm for Cognition.* Cambridge University Press.
3. Trabasso, T., & van den Broek, P. (1985). Causal thinking and the representation of narrative events. *Journal of Memory and Language, 24*(5), 612–630.
4. Trabasso, T., & Sperry, L. L. (1985). Causal relatedness and importance of story events. *Journal of Memory and Language, 24*(5), 595–611.
5. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience, 11*(2), 127–138.
6. Friston, K., Rigoli, F., Ognibene, D., Mathys, C., Fitzgerald, T., & Pezzulo, G. (2015). Active inference and epistemic value. *Cognitive Neuroscience, 6*(4), 187–214.
7. Loewenstein, G. (1994). The psychology of curiosity: a review and reinterpretation. *Psychological Bulletin, 116*(1), 75–98.
8. Johnson, M. K., & Raye, C. L. (1981). Reality monitoring. *Psychological Review, 88*(1), 67–85.
9. Tse, D., et al. (2007). Schema-dependent gene expression and memory formation. *Science, 316*, 76–82.
10. Griffiths, T. L., & Tenenbaum, J. B. (2009). Theory-based causal induction. *Psychological Review, 116*(4), 661–716.
11. Schacter, D. L., & Addis, D. R. (2007). The cognitive neuroscience of constructive memory. *Philosophical Transactions of the Royal Society B, 362*(1481), 773–786.

---

## MEASURED NEGATIVE: full-page flag failed its A/B (REVERTED)

Implemented behind `RAVANA_FULL_PAGE_GAP=1`: bounded `_fetch_article_text`
on each admitted URL, existing OpenIE over the longer text, two distinct URLs
required per causal triple before stamping, `acquired_by=full_page_extraction`.
Default-OFF; existing 10 gap tests passed.

A/B with IntentForge on, identical state, separate processes (3 subjects):

```
FLAG=0  iron rust   hops=0  chain=0
FLAG=0  ocean tides hops=2  chain=1
FLAG=0  rain        hops=0  chain=0

FLAG=1  iron rust   hops=0  chain=0
FLAG=1  ocean tides hops=0  chain=0
FLAG=1  rain        hops=0  chain=0
```

The whole-page path produced **no** causal hops where the snippet path did.
The four assertable chains counted in both arms came from the authored
curriculum scaffold, not from web learning. The code was **REVERTED**; default
snippet behaviour is restored.

**Conclusion:** fetching more text does not help when the retriever cannot
find pages that state two extractable causal relations for the subject. This
is the fourth failed input-side attempt (query phrasing, bounded retry, typed
need plus source veto, full page). The remaining hypothesis is that the
search provider's result set itself lacks the needed pages; test that directly
by measuring what fraction of returned documents contain two distinct
causal triples before building any further extraction machinery.

## MEASURED: the frozen result set, not the extractor, is the limit

Ran the existing OpenIE extractor over all 1,087 frozen search fixtures
(`tests/fixtures/web_snapshots/web_fixtures.json`), counting distinct causal
triples per query:

```
all queries:  n=1087   >=1 triple = 386 (35.5%)   >=2 triples = 159 (14.6%)
gap queries:  n=4      >=1 triple = 2             >=2 triples = 1
```

So the corpus does contain usable causal prose, and OpenIE can extract it.
But the specific gap-read query template appears only **4 times** in 434
harness probes, and only once yields two triples. The dominant failure is
upstream: `close_mechanism_gap` is rarely invoked because the mechanism-gap
detector does not open a gap for most definition-style answers.

**Consequence:** further extraction work will not help until gap detection
opens a gap for `why/how` questions answered only by a definition. That is the
next work item; it is a detection problem, not an acquisition or extraction
problem.

## MEASURED NEGATIVE: widening gap detection did not help

Hypothesis from the previous section: definitions with one causal connective
were falsely counted as mechanisms, so `close_mechanism_gap` rarely fired.
Changed `_got_mechanism` to require >=2 causal clauses. Instrumented call
sites: 6 of 14 why/how probes now opened a gap (subjects resolved correctly;
the old single-clause test closed the rest).

A/B on the frozen web harness, identical fixtures and seed:

```
before: search=1666  mean=77
after:  search=1752  mean=77   (+86 gap reads, zero score change)
```

All seven fronts unchanged. The code was **REVERTED**; no capability gain,
only extra reads. This is the fifth failed attempt to improve web-driven
learning by changing what the system reads or when it decides to read
(query phrasing, retry, typed need + veto, full page, gap widening).

**Conclusion:** the web gap loop can fire and can read, but the relations it
retrieves still do not reach the emitter as trusted chains in a way that
changes answers. The authored curriculum scaffold remains the only measured
substrate gain (thinking 5 -> 13). Stop tuning this loop without a new
information source; the RAVANA-LEARN-5 components, not more input tweaks,
are the appropriate next instrument.
