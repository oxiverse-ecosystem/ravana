# Web encoding gate — end-to-end map, default-OFF experiment, brain basis

**Status:** static investigation + design (no engine runs, no pytest, no source changes).
**Locked decision:** D7 in `docs/investigations/HANDOFF_LEARN5_FIXES.md` — "read-schedule
tweaks exhausted; the only untried axis is the ENCODING GATE ... Ship default-OFF, A/B on
frozen state, revert on no gain."
**Scope:** one new deliverable doc only. Everything below was read from the checked-out tree
(branch `docs/learn5-final-verdict`) and from publisher/Crossref records for the citations.

## 0. Corrections to the task brief (verified, not cosmetic)

| Brief says | Verified record | Consequence |
|---|---|---|
| "Kang et al 2009 **Cognition**" | Kang, Hsu, Krajbich, Loewenstein, McClure, Wang & Camerer (2009), *The wick in the candle of learning: Epistemic curiosity activates reward circuitry and enhances memory*, **Psychological Science** 20(8):963–973, doi:10.1111/j.1467-9280.2009.02402.x (SAGE + APA PsycNET record 2009-12141-008). | Use *Psychological Science*. No 2009 Kang curiosity paper in *Cognition* was found. |
| "Bjork et al 1994" (curiosity/information-gap) | The verifiable 1994 Bjork item is **Bjork, R. A. (1994)**, *Memory and metamemory considerations in the training of human beings*, in Metcalfe & Shimamura (eds.), *Metacognition: Knowing about Knowing*, MIT Press, pp. 185–205 — **single-authored book chapter, no DOI**. The information-gap paper is **Loewenstein (1994)**, *Psychological Bulletin* 116(1):75–98, doi:10.1037/0033-2909.116.1.75. | Cite Loewenstein for information-gap; cite Bjork 1994 for desirable difficulties / metamemory misjudgement. Do not attribute information-gap theory to Bjork et al. 1994. |
| "schema accommodation vs assimilation (Tse et al 2007)" | Tse et al. 2007 is explicitly about **assimilation** into a pre-existing schema (one-trial, rapidly hippocampal-independent). Accommodation (schema change) is not demonstrated by it. | Label the `revise` route an engineering extrapolation; Tse supports `fast_merge`, not `revise`. |
| Working memory "7±2" | Miller 1956 proposed ~7 chunks; Cowan 2001 argues ~4 once rehearsal/LTM support is controlled. The cap is on **chunks**, not raw items. | Encode one fact per read; do not cite "7" as a fixed capacity number. |

---

## 1. The current web-learning path, end to end

### 1.1 What triggers a fetch

There are **two distinct read paths plus a background learner**; they use different triggers
and only one of them is touched by the new flag.

1. **FOK / "LPFC pause" synchronous read** — `ravana/src/ravana/chat/engine.py:11468–11619`.
   In `baby_mode`, count strong (>0.2) associations for the subject among the top 12
   (`strong_assocs`); a subject is "known" if it has a definition, a web-source entry, an
   event schema, or is in the protected/seeded concept set. Low FOK (`strong_assocs < 2` and
   not phrase-known, or multi-word and not phrase-known) sets `_fok_pre_queued`, appends the
   topic to `_pending_learning_queue`, and — if `baby_mode and not _fok_pause_done` — calls
   `learn_from_web(_fok_query, max_results=2)`. The query is the raw user input only when
   `_is_informational_query` accepts it; otherwise it is
   `"{subject} definition meaning explained"`. After the read it re-activates, re-spreads,
   re-filters and re-checks FOK. **This is the read gated by the new default-OFF curiosity
   flag** at `:11534–11535` (`_curiosity_gate_allows(subject)`).
2. **Mechanism-gap read** — `engine.py:11982–12006`. Fires only when `_mechanism_gap` is true
   (a why/how question whose answer contains no mechanism) after a weak response was
   detected; calls `close_mechanism_gap(ctx.subject, ctx.raw_input)` with a swallowed
   exception. `close_mechanism_gap` lives in `chat/web_learning.py:661–817`: offline short
   circuit, once-per-subject dedup (`_gap_read_subjects`), per-session cap `_GAP_MAX_READS=4`,
   at most `_GAP_MAX_SOURCES=2` hits, query
   `"{subject} causes mechanism why process explained"` (`_GAP_CAUSAL_QUERY`, :93). **This
   path is still driven by the self-rated gap and is NOT gated by the curiosity flag.**
3. **Background / generic learning** — `learn_from_web` (:438–657) and `_learn_from_text`
   (:918–1318), also reachable from the background learning thread and corpus harvest.
   Offline short-circuits at :453. Search failures fall back to a local-only engine
   (:467–514) and then to GloVe-only learning from the query text (:516–530).

### 1.2 Search, deep read, and extraction

`learn_from_web` searches (`search_engine.search`, max_results=3 by default), keeps the top
3 snippets unconditionally (`f"{title}. {snippet}"`), runs `_extract_definitions` on each
snippet/title, then ranks by `_definition_coherence_score` (subject↔snippet GloVe cosine) and
deep-reads only results above `_deep_read_relevance_gate` (0.12), capped at
`_deep_read_max=1`; deep reads may be offloaded to the background thread. `_fetch_article_text`
(:873–914) strips `script/style/nav/header/footer`, prefers `<article>/<main>/<body>`, and
truncates to 5,000 chars.

### 1.3 Filtering and writing extracted facts

- **Co-occurrence channel** (`_learn_from_text`): tokenise, drop stop/garbage/inappropriate
  words, keep top 30 by frequency plus topic words; reject `junk_score >= _junk_theta` (0.5);
  non-junk tokens enter `_provisional_nodes` and need `_promote_min_sources=2` distinct source
  ids to become graph nodes — **except** the text's own topic word, which fast-maps on first
  sight (Tse-style slot supplied by the text, :1012–1015). Edges: co-occurrence `semantic`
  edges weight 0.3–0.7 (:1133–1137), plus top-8 vector-similarity edges at sim >0.45
  (:1164–1192). Those edges are **not** causal and carry no `web_fact` provenance.
- **Definition channel** (`_extract_definitions`, with `_definition_acceptable`,
  `_definition_looks_clean`, `_definition_quality`, `_strip_code_fragments`): requires a
  copula/defining verb (`_DEFINITION_PREDICATE`), rejects disambiguation pages, UI chrome
  ("add to word list", "collocation"), code/GTM residue, Markdown/LaTeX/citation artefacts,
  and low alphabetic ratio; GloVe coherence >0.15 for non-pattern matches (OOV/pattern-matched
  subjects fall back to cleanliness).
- **Typed-fact channel** (C-lite, `web_learning.py:1294–1316`): `WebToGraph.learn_text` writes
  OpenIE-derived `web_fact` edges; the same facts feed `triplet_op`. Suppressed on error.
- **Mechanism-gap channel** (`close_mechanism_gap`): keeps only OpenIE facts with
  `relation == "causes"`, accumulates distinct URLs per `(subject, object)`, calls
  `populate_provenance(edge_kind="web_fact", source=first_url, method="mechanism_gap_read",
  retrieval_conf=conf, converging_sources=n)` on **newly written causal edges only**, sets
  `acquired_by="mechanism_gap"`, `gap_subject`, `gap_query`, and confidence
  `min(1, conf + 0.15*(n-1))`. `_gap_chain_hops` counts hops in the largest **undirected**
  connected component of the new edges; status `"learned"` only if `chain_hops >= 2`.
  This is *not* `assertable_chain`.
- **Schema-consistency gate** (`learn/intake.py`, `SchemaConsistencyGate`): a pure function of
  (fact, graph, provenance, salience). `consistency` = 0.30 subject known + 0.20 object known
  + 0.30 subject already has an edge of this relation type + 0.10 endpoints already linked.
  `elaboration` = 0.55·(degree/8) + 0.30·(shared-neighbours/3) + 0.25 bridging (bounded by
  Cowan-style saturation). `entrenchment` = 0.45·incumbent support + 0.35·source trust +
  0.20·corroboration. `detect_contradiction` is restricted to functional relations —
  `causal`/`physical_cause` are explicitly excluded (CLI write-path fix, test in
  `tests/unit/test_cli_write_path.py`). `route`: contradiction → `revise` or
  `episodic_buffer` (a single uncorroborated `web` source has base reliability 0.5 < 0.7 and
  cannot overturn an incumbent); `consistency >= 0.55` → `fast_merge`; otherwise
  `novel_write` (written weakly, not buffered). `KnowledgeIntake._write_edge` writes a typed
  edge with `source_metadata` + `edge_kind` from source type (`"web" → "web_fact"`) at write
  time, and **upgrades a pre-existing `semantic` edge to `causal`** when a trusted causal fact
  arrives (never downgrades). Accepted facts also get a retrieval-practice due-turn.

### 1.4 Why all five read-side tweaks produced zero hops

| # | Tweak (reverted) | Axis tuned | Measured outcome |
|---|---|---|---|
| 1 | query phrasing | when/what to read | 0 causal hops |
| 2 | bounded retry | when to read | 0 causal hops |
| 3 | typed InformationNeed + source veto | what to read | 0 causal hops |
| 4 | full-page extraction | how to read | failed A/B, reverted |
| 5 | gap widening | when to read | 86 extra reads, 0 score change |

All five moved the **read schedule**; none changed the **signal that authorises encoding**, the
**unit encoded**, or the **schema relation** into which it is written. Two measured input facts
explain the flat results (`docs/investigations/WEB_AND_SATURATION.md` §1):

- **Corpus poverty:** only **14.6% of 1,087 frozen queries** contain ≥2 extractable causal
  triples. 85.4% cannot supply a connected 2-hop causal chain regardless of scheduling.
- **Read-path rarity:** the gap-read template fired **4 times in 434 probes**. A better
  extractor has almost nothing to operate on.

The only measured learning gain in this area came from authored curriculum provenance (53
edges), not from the web loop. Two instrument caveats matter before any new A/B: the frozen
arm's mean-77 was measured on the degraded 209-concept cold-start graph and is stale (repaired
graph ≈2 min/probe), and `chain_hops` (undirected component size) can rise while the
fail-closed `assertable_chain` metric stays flat — `assertable_chain`
(`learn/explanation.py:343–352`) requires **every** hop provenance-trusted, and
`grade_chain` downgrades an "assert" verdict to "decline" when trust fails (:381–386).

**Verdict:** the bottleneck is (a) input relation density for the 2-hop endpoint and (b) how
rarely a gap authorises a read — not phrasing, retry, veto, page length, or detection width.
Per skill §7a, do not build a sixth read-schedule tweak.

---

## 2. Default-OFF encoding-gate experiment

### 2.1 Signal: external re-ask / correction, not intrinsic self-rating

Use an **external information-gap signal**: the listener's next-turn behaviour. Concretely,
the already-wired predicates in `engine.py:8146–8198`:

- `_track_curiosity_signal(user_input)` records content words (len ≥4, minus a stop set) when
  (a) the same content words recur within 10 turns (**re-ask** — previous answer did not close
  the gap) or (b) an explicit correction regex fires (`no, actually...`, `you're wrong`,
  `that's not...`) and marks the **previous** utterance's topic.
- `_curiosity_need(subject)` = any signal for a substring of `subject` within 10 turns.
- `_curiosity_gate_allows(subject)` = flag on → require `_curiosity_need(subject)` and a
  **3-read-per-session** budget; flag off → return `True` (prior behaviour preserved).
- Flag: `RAVANA_CURIOSITY_GATE` read at `engine.py:2034`, default `"0"`. `_process_turn_impl`
  calls `_track_curiosity_signal` at the top of every turn (:8203).

**Do not use** an intrinsic FOK score, low association count, or a quality score computed from
RAVANA's own response as the encoding signal: those are self-ratings, and reinforcing from
them is a self-confirming loop (skill §7). The FOK pre-check may still *schedule* background
learning; it must not *authorise encoding* in the ON arm.

Honest limit: a re-ask/correction is an **observable proxy** for an unresolved gap, not a
measurement of epistemic curiosity. Kang/Gruber show curiosity gates encoding; they do **not**
show that a re-ask is curiosity. The translation is an extrapolation (§4).

**Wiring note:** the flag currently gates only path (1) (the FOK synchronous read). Path (2),
`close_mechanism_gap`, is still self-rated. For the experiment, either leave path (2)
byte-identical in both arms and document it as a confine, or gate it with the same predicate
and register that as a *second* pre-registered arm — never change it silently mid-A/B.

### 2.2 What to encode: one provenance-stamped fact, schema-anchored

- **Unit = one fact** `(subject, relation, object)` from a source span, not a whole page and
  not a bag of multi-triple co-occurrences. A gated read admits at most **one** candidate fact
  (choose the highest-confidence OpenIE fact with `relation == "causes"` or a definition-type
  relation whose subject matches the signaled subject); the rest of the page stays unwritten.
  Rationale: single facts/definitions are abundant in the corpus; connected triples are 14.6%;
  Miller/Cowan say the effective buffer is a few chunks, so one chunk per authorised read.
- **Provenance at write time:** route the single fact through `KnowledgeIntake.intake()` with a
  real `Provenance(source=URL, source_type="web", evidence_span=...)`. Assert `edge_kind`,
  URL, retrieval confidence, acquiring mechanism and evidence span in `source_metadata` **at
  creation**; never lower a trust threshold to admit a fact (skill §6). Do not count raw
  co-occurrence edges as learned facts.
- **Schema anchor:** prefer facts whose subject and object are already graph nodes, whose
  subject already has an outgoing edge of that relation type, or whose object is an existing
  neighbour of the subject — i.e. `SchemaConsistencyGate.consistency >= 0.55` → `fast_merge`.
  A fact with `consistency < 0.55` takes `novel_write` at low confidence; a contradiction
  on a functional relation takes `episodic_buffer`/`revise` and is **not** scored as an
  acquired fact. Schema *proposes* candidates; actual sources *admit* them (skill §7a).
- **Relation typing:** a causal fact must land as `relation_type="causal"` (upgrade a
  pre-existing `semantic` edge, never downgrade); otherwise `assertable_chain`/`grade_chain`
  cannot see the hop. Causal relations are many-to-many — do not let the second effect of one
  cause be buffered as a contradiction.
- **Underpowered corpus case:** if the signaled subject has no single extractable fact in the
  fixture, write nothing and record `no_fact` — do not fabricate a definition from the query.

### 2.3 A/B arms on frozen fixtures

Use the frozen-snapshot recipe (`ravana-debugging/references/frozen-snapshots.md`) and
`scripts/measure_fronts_web_frozen.py` pattern; keep `RAVANA_OFFLINE=0` so web-learning code
still executes, but patch the fixture provider and `urllib.request.urlopen` at the fetch
boundary to **raise and count**. No live fetch, no save attempts (assert both counters zero).

| Arm | Flag | Input state | Manipulation |
|---|---|---|---|
| **OFF (control)** | `RAVANA_CURIOSITY_GATE=0` | exact frozen fixture corpus + frozen graph snapshot | prior path, byte-identical to pre-change code |
| **ON (treatment)** | `RAVANA_CURIOSITY_GATE=1` | same corpus, `copy.deepcopy` of the same post-maintenance graph snapshot | external-signal gate + one-fact schema-anchored write |
| **ON-wrong-tag (negative control)** | `1` | same snapshot | same reads, but stamp facts with a wrong/incomplete provenance tag |
| **no-signal (floor)** | `1`, no re-ask/correction injected | same snapshot | gate must admit zero reads; proves the gate is not "answer everything" |

Execution discipline:

1. **Flag equivalence first:** with the flag OFF, run a short script and assert the output is
   byte-identical to pre-change (the `test_curiosity_gate.py` contract already checks
   `_curiosity_gate_allows` returns `True` and consumes no budget when OFF). If OFF differs,
   stop — the default is broken.
2. **One process, identical state, ON computed before OFF.** Cache maintenance once, deep-copy
   the `__getstate__()` snapshot (graph `__getstate__` returns live dicts by reference —
   deep-copy or arm 2 silently repairs arm 1's hops). Assert equal work counters (facts
   scanned, intake calls, maintenance iterations) at matched compute; report wall-clock ratio
   and reject >10% overrun as a confound.
3. **Fixtures:** re-capture after any change that alters which queries are issued; assert
   `fixture_miss == 0`, `network_attempts == 0`, `save_attempts == 0`, `offline='0'`,
   `frozen=true`, probe-row count == expected total, summary mtime fresh. Use `--resume` keyed
   by `commit:weights-mtime`; `EXIT=124` or a missing EXIT line means no score may be quoted.
4. **Held-out split:** assign primary/sealed pools by `sha256(front|difficulty|subject)` before
   runs; report primary-only and full-set side by side. Treat <20 held-out probes as a
   hypothesis, not a result. Log every probe row (input signals, gate decision, route,
   written fact, chain verdict) so a perfect aggregate can be audited.
5. **No engine runs in this deliverable.** The harness is specified, not executed.

### 2.4 Metric that proves a gain

**Primary (positive): `assertable_chain` on learned items, plus emitted reach.**
For each probe subject, enumerate chains from the indexed node and count chains where
`assertable_chain(chain)` is true **and** `grade_chain(...).verdict == "assert"`; count only
chains containing at least one hop written in the ON arm. Separately, check emitted answers
on those same subjects (chain-level gain is not answer-level gain). Report
`Δ = ON − OFF` on primary and sealed pools. `chain_hops`, `sources`, `causal_hops` and
`_gap_chain_hops` are **diagnostic counters only** (undirected component proxy; a rise there
with a flat `assertable_chain` is a failure, not a win).

**Secondary positive:** count of newly written provenance-stamped single facts with
`edge_kind="web_fact"`, URL and evidence span, that survive a cold reload (the SQLite mirror
must carry provenance, per the durability fixes).

**Negative metrics (must stay flat):**
- untrusted / `edge_kind="unknown"` edge count and known-bad probe rejection rate;
- wrong-tag control: `assertable_chain` must **not** rise — this proves the metric is reading
  provenance, not just graph size;
- functional-relation contradiction rate and episodic-buffer overflow (no causal fact
  mis-routed by slot exclusivity);
- false-assert rate on decoy / near-miss subjects (schema congruence is also a false-memory
  source — Johnson et al.; skill §7a);
- compute parity: intake calls, extractor calls per admitted read, wall-clock ratio.

**Adopt only if** the positive metric moves on the primary **and** sealed pools, the wrong-tag
control stays flat, all negatives stay flat and parity holds. Otherwise revert the flag and
record the negative (skill §7a/§7b). "Both arms retain all items" = no headroom = INVALID,
not a win.

### 2.5 Falsifiers (any one kills the mechanism)

1. The gate fires (re-ask/correction observed) but the extractor yields **zero** usable single
   facts for those subjects — corpus is a hard ceiling; stop tuning the loop.
2. Facts are extracted but `SchemaConsistencyGate.route` returns `episodic_buffer`/`revise`
   for all of them, or `_write_edge` returns `False` (missing endpoints) → write-path drop,
   not a gate success.
3. Written facts have `relation_type="semantic"` or incomplete `source_metadata` →
   `assertable_chain` cannot see them; the gain is measurement or provenance failure.
4. Only `chain_hops`/`sources` rise while `assertable_chain` and emitted reach stay flat →
   the undirected proxy is being mistaken for learning.
5. The wrong-tag control also gains → the metric is not provenance-sensitive.
6. Any negative metric moves, or compute parity fails → confound; revert.
7. Sealed-pool score drops below primary by more than the overfitting tolerance → the
   primary number was fitted.
8. ON arm changes OFF-arm output with the flag off → default-OFF contract violated; revert.

### 2.6 Pre-registered order of operations

`flag-equivalence check → corpus triple census over frozen fixtures → no-signal floor arm →
OFF vs ON primary+sealed A/B (one process, shared deep-copied state) → wrong-tag control →
cold-reload provenance check → decision (keep default-OFF + enable, or full revert + record
negative) → re-measure stale mean-77 frozen arm separately`. No rubric or gate threshold may
be edited to rescue a failed arm.

---

## 3. Brain research — verified citations, then explicit extrapolation

All journal/DOI records below were checked against publisher, APA PsycNET, PubMed, Crossref or
PMC records. Book chapters without a DOI are marked as such.

### 3.1 Curiosity / confidence-based information gap

| Source (verified) | What the paper supports |
|---|---|
| **Loewenstein, G. (1994).** The psychology of curiosity: A review and reinterpretation. *Psychological Bulletin, 116*(1), 75–98. doi:10.1037/0033-2909.116.1.75 | Curiosity is "a form of cognitively induced deprivation that arises from the perception of a gap in knowledge or understanding." Specific gap, not diffuse ignorance. |
| **Kang, M. J., Hsu, M., Krajbich, I. M., Loewenstein, G., McClure, S. M., Wang, J. T.-y., & Camerer, C. F. (2009).** The wick in the candle of learning: Epistemic curiosity activates reward circuitry and enhances memory. *Psychological Science, 20*(8), 963–973. doi:10.1111/j.1467-9280.2009.02402.x | Curiosity ratings correlated with caudate (anticipated-reward) activity; participants spent tokens/wait time to learn answers; higher curiosity predicted better recall of surprising answers 1–2 weeks later. **Journal is Psychological Science, not Cognition.** |
| **Gruber, M. J., Gelman, B. D., & Ranganath, C. (2014).** States of curiosity modulate hippocampus-dependent learning via the dopaminergic circuit. *Neuron, 84*(2), 486–496. doi:10.1016/j.neuron.2014.08.060 | High-curiosity states improved memory for target **and incidental** material; nucleus accumbens/midbrain activity and midbrain–hippocampus connectivity predicted the benefit. Encoding is gated by motivational state, not exposure. |
| **Bjork, R. A. (1994).** Memory and metamemory considerations in the training of human beings. In J. Metcalfe & A. P. Shimamura (Eds.), *Metacognition: Knowing about knowing* (pp. 185–205). MIT Press. (book chapter; no DOI) | Learners misread their own progress; conditions that slow acquisition can improve retention and transfer ("desirable difficulties"). **Not an information-gap paper and not "et al."** Used here to justify not trusting a self-rated gap, not to define curiosity. |

### 3.2 Retrieval practice / test-potentiated encoding

| Source (verified) | What the paper supports |
|---|---|
| **Karpicke, J. D., & Blunt, J. R. (2011).** Retrieval practice produces more learning than elaborative studying with concept mapping. *Science, 331*(6018), 772–775. doi:10.1126/science.1199327 | Free-recall practice beat concept mapping on a 1-week test, including inference questions (84% of 120 students performed better after retrieval). Learning comes from reconstructing, not elaborative re-study. |
| **Kornell, N., Hays, M. J., & Bjork, R. A. (2009).** Unsuccessful retrieval attempts enhance subsequent learning. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 35*(4), 989–998. doi:10.1037/a0015729 | A failed attempt followed by feedback beats equal-time reading (Exp. 1: 41% vs 31%, d=0.58); even wrong guesses were recalled better than blank items. A **failed** retrieval is a useful encoding opportunity. |

This supports treating a failed answer + re-ask as a *high-value encoding moment*, and supports
the retrieval-practice review queue in `KnowledgeIntake` — but only when followed by correct
feedback. A re-ask without an admitted correct fact must write nothing.

### 3.3 Finiteness of working memory and chunking

| Source (verified) | What the paper supports |
|---|---|
| **Miller, G. A. (1956).** The magical number seven, plus or minus two: Some limits on our capacity for processing information. *Psychological Review, 63*(2), 81–97. doi:10.1037/h0043158 | Severe limits on information received/processed/remembered; recoding into **chunks** can stretch the bottleneck. |
| **Cowan, N. (2001).** The magical number 4 in short-term memory: A reconsideration of mental storage capacity. *Behavioral and Brain Sciences, 24*(1), 87–114. doi:10.1017/S0140525X01003922 | With rehearsal and LTM support controlled, focus-of-attention capacity is ~4 chunks, not 7. |

Supports **one fact per authorised read** and the `elaboration` saturation term at degree/8 in
`intake.py`. It does not specify a numeric RAVANA read budget; the 3-read cap is an engineering
analogue, not a derived constant.

### 3.4 Schema: assimilation vs accommodation

| Source (verified) | What the paper supports |
|---|---|
| **Tse, D., Langston, R. F., Kakeyama, M., Bethus, I., Spooner, P. A., Wood, E. R., Witter, M. P., & Morris, R. G. M. (2007).** Schemas and memory consolidation. *Science, 316*(5821), 76–82. doi:10.1126/science.1135935 | With a prior neocortical schema, new one-trial associations became rapidly hippocampal-independent; schemas played a causal role in lasting associative memory. **Assimilation into existing structure.** |
| **Tse, D., Takeuchi, T., Kakeyama, M., Kajii, Y., Okuno, H., Tohyama, C., Bito, H., & Morris, R. G. M. (2011).** Schema-dependent gene activation and memory encoding in neocortex. *Science, 333*(6044), 891–895. doi:10.1126/science.1205274 | New paired associates assimilated into a schema drive immediate Zif268/Arc upregulation in prelimbic mPFC; disrupting mPFC prevents new learning and recall. Assimilation is cortically encoded at the time of hippocampal learning. |

Supports `fast_merge` for schema-consistent facts and anchoring new facts to existing
neighbours. **Accommodation (changing the schema) is not established by either paper**; the
`revise` route is a plausible engineering extension, not a brain-grounded claim here.

### 3.5 Source monitoring

| Source (verified) | What the paper supports |
|---|---|
| **Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993).** Source monitoring. *Psychological Bulletin, 114*(1), 3–28. doi:10.1037/0033-2909.114.1.3 | Memories are not stored with an origin tag; source is inferred by monitoring processes. Misattributed familiarity, cryptomnesia, and incorporation of fiction into fact are documented failure modes. |

Supports provenance stamping at write time **and** the negative-control requirement: provenance
is a claim to be checked, not a guarantee of truth. A schema-congruent but unsupported fact
must be refused — congruence is a source of false memories as well as fast learning.

---

## 4. Supported by literature vs extrapolation — explicit ledger

| Claim in the design | Status | Basis |
|---|---|---|
| Curiosity/information-gap gates what gets encoded | **Supported** | Loewenstein 1994; Kang et al. 2009; Gruber et al. 2014 |
| A failed retrieval + feedback is a good moment to encode | **Supported (with feedback)** | Kornell et al. 2009; Karpicke & Blunt 2011 |
| Encoding capacity is limited; chunking helps | **Supported** | Miller 1956; Cowan 2001 |
| A fact fitting an existing schema consolidates fast | **Supported (assimilation)** | Tse et al. 2007, 2011 |
| Origin must be monitored; congruence can create false memories | **Supported** | Johnson et al. 1993 |
| A **re-ask/correction** is the RAVANA analogue of epistemic curiosity | **Extrapolation** | Observable proxy; no cited paper equates the two |
| Re-ask → "one fact per read" | **Extrapolation** | WM capacity motivates the unit but does not dictate it |
| `consistency >= 0.55` → `fast_merge`; `revise` for contradictions | **Extrapolation** | Tse supports assimilation; thresholds are engineering choices |
| 3-reads/session budget; 10-turn signal decay; 0.55/0.15 thresholds | **Engineering constants** | Not derived from any citation; must be A/B tested, not defended as brain-grounded |
| `assertable_chain` as the learning endpoint | **Instrument choice** | Follows directly from the trust-gate design, not from brain evidence |
| Deleting/changing a schema (`revise`) when a source disagrees | **Extrapolation** | Accommodation is Piagetian terminology; Tse et al. do not demonstrate it |

---

## 5. What this doc does not claim

- No web-loop gain is claimed. The encoding gate is a falsifiable hypothesis with a
  default-OFF flag; the A/B is specified, not run.
- No source file, test, or existing doc was modified. No engine, harness, or pytest run was
  performed.
- The 14.6% / 4-of-434 figures are inherited measurements from
  `docs/investigations/WEB_AND_SATURATION.md`, not re-measured here.
- The mean-77 frozen-web number is stale (degraded cold-start graph) and is not reused.
- Bibliographic verification is from publisher/Crossref/PMC records retrieved during this
  investigation; all 1994/2009/2011/2014 entries above were confirmed. Books/chapters without
  DOIs are marked as such.

## 6. Files consulted

- `docs/investigations/HANDOFF_LEARN5_FIXES.md` (D7, phase 3), `WEB_AND_SATURATION.md`
- `ravana/src/ravana/chat/web_learning.py` (fetch, `close_mechanism_gap`, `_learn_from_text`)
- `ravana/src/ravana/chat/engine.py` (`:2034` flag, `:8146–8198` gate predicates,
  `:11468–11619` FOK read, `:11982–12006` mechanism-gap call)
- `ravana/src/ravana/learn/intake.py` (`SchemaConsistencyGate`, `KnowledgeIntake`)
- `ravana/src/ravana/learn/explanation.py` (`assertable_chain`, `grade_chain`)
- `ravana/src/ravana/chat/engine_memory.py` (episodic/fact recall paths, inspected only)
- `tests/unit/test_curiosity_gate.py`, `tests/unit/test_cli_write_path.py`
- `ravana-debugging` skill and `references/{online-learning,frozen-snapshots}.md`
