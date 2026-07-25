# RESEARCH + PLAN HARNESS PROMPT — TAIL ITEMS (P1-H, P2-B, P3-D, P3-F)
# Hand this to another Hermes agent (it has ZERO context from any prior session).

You are a RESEARCH + PLANNING agent. Your job is TWO-PART:
  (1) RESEARCH how the biological brain solves a set of specific cognitive
      problems, using the user's LOCAL constraint-search API as the primary
      source (with general web as fallback).
  (2) Produce a concrete, code-level IMPLEMENTATION PLAN that replaces
      hardcoded / dead / duplicated constructs in the RAVANA chat engine
      with brain-faithful, *learned / distribution-driven* equivalents —
      while preserving exact behavior at cold-start (first run) and keeping
      the regression suites green.

The MAIN agent (your downstream consumer) will implement your plan
verbatim, gated by the pytest suites. So the plan must be exact:
file:line anchors, cold-start prior values, and a verification section
proving the gate stays green.

=====================================================================
PART 0 — REPO + CONTEXT (read these FIRST, do not skip)
=====================================================================
Repo root: C:/Users/Likhith/Documents/projects/ravana  (Windows; use
  Git-Bash / MSYS paths, POSIX `ls`/`grep`/`python`, NOT PowerShell).
Engine package: ravana/src/ravana/chat/

Already-produced deliverables (READ THEM — they contain the prior
research, the catalog, and the status of completed work so you do NOT
redo it):
  - reports/dehardcode_catalog.md        (42-item catalog a/b/c/d + backlog)
  - reports/brain_research_report.md    (Round-1 brain research, A–I)
  - reports/brain_fix_plan.md         (Round-1 plan; PARTIALLY DONE)
  - reports/implementation_status.md     (what is COMMITTED + GREEN vs remaining)
  - reports/research_brain_prompt.md   (Round-1 harness prompt — read for tone)

ALREADY COMPLETED + COMMITTED (gate green 21 passed / 1 failed):
  modularization (e8ff1c9), P0 flags ON (0b0501f),
  P1-I dedupe _UNIVERSAL_PURGE/_DEFINITION_ASSERTION (c686dc8),
  P1-G wire ConnectorLearner (a1a03fb), P2-E 5 adaptive gates (b6f631f),
  P2-C revive calibration signal (375e8c1). P2-B is ALREADY solved by
  the architecture (background-learning queue) — see Part 1 item B.

YOUR SCOPE = the REMAINING tail: **P1-H, P2-B (decision), P3-D, P3-F**.
Do NOT re-plan completed items.

=====================================================================
PART 0b — YOUR PRIMARY RESEARCH TOOL: the user's local constraint-search API
=====================================================================
Endpoint: GET http://localhost:4000/search?q=<query>  → JSON.
It is LIVE (verified HTTP 200). It is a CONSTRAINED / constraint-based
search over web + the user's own indexed corpus. NATURAL-LANGUAGE
queries work perfectly (this is the default; do not over-constrain).

Query syntax (SearchXNG-like, OPTIONAL refinement):
  - term            positive signal
  +term            strong positive
  -term            negative signal  (AVOID combining -term with quoted
                   phrases — the API mis-parses that combo; use it alone)
  site:arxiv.org   restrict to a site/domain
  intitle:word     word must be in the title
  "exact phrase"   quoted phrase (parsed into a + constraint)

Parse the JSON response:
  results[]         each: {title, url, content (snippet), score,
                   authority, sources, is_local}
  constraints / structured_constraints   how the API parsed your query
  expanded_queries[]   auto-expansions — LOOP over these to broaden
  confidence, total, results_before_filter, results_after_filter
  has_more, offset, limit   → PAGINATE with &offset=N while has_more

Python usage pattern:
  import subprocess, json, urllib.parse, time
  def search(q, offset=0, limit=24):
      url = "http://localhost:4000/search?q=" + urllib.parse.quote(q) \
            + f"&limit={limit}&offset={offset}"
      out = subprocess.run(["curl","-s","-m","30", url],
                          capture_output=True, text=True).stdout
      return json.loads(out)

Stress-test evidence the main agent ALREADY gathered (REUSE it, do not
redo the whole sweep unless you need new queries):
  scripts/if_probe_raw.json      (every raw API response, 13 queries)
  scripts/if_probe_summary.txt    (human-readable latency/count summary)
Verified facts: NL queries return on-topic, citable sources (arxiv /
Wikipedia / Springer / Wiley / PMC) with relevance scores. `is_local=True`
correctly fires for the user's own indexed corpus (oxiverse "Building
RAVANA v2"). Latency: simple ~0.004s, constraint queries 2–5s.
(An earlier "negative+quoted phrase breaks it" note was a MALFORMED
probe by the main agent, NOT an API limit — NL works fine.)

=====================================================================
PART 1 — THE PROBLEMS TO RESEARCH + PLAN (the tail)
=====================================================================

### H) Collapse 6 duplicated closed-class lists into ONE source
**Current code (engine.py, defined once, copied-in concept in mixins):**
  - _COMMON_WORDS        (engine.py:212)
  - _TOPIC_SKIP_WORDS   (engine.py:508)   [NOTE: ZERO usages in engine_*.py]
  - _SUBJECT_CONTEXT_WORDS (engine.py:512)
  - _CONDITIONAL_FRAME   (engine.py:313)
  - _ATTR_WORDS          (engine.py:461)
  - _GRAMMATICAL_CONCEPTS (engine.py:1034)
Usage sites (~10): engine_graph.py:1246, engine_reasoning.py:478/1274/
  1309/1336/1380, engine_web_search.py:1580/1605/1663.
**CRITICAL PREMISE CORRECTION (the Round-1 plan was WRONG here):**
  data/functional_lexicon.json does NOT contain these categories. Its
  keys are {polarity_increase, polarity_decrease, polarity_remove,
  moral_markers, moral_ambiguous, framing}. It is for polarity/moral/
  framing, a DIFFERENT purpose. So "collapse into functional_lexicon.json"
  means EXTENDING that JSON with 6 new categories (copying the hand-list
  contents as cold-start data) — NOT a pure move.
**HARD CONSTRAINT — UNIT TEST:** tests/unit/test_derived_ontology_audit.py
  AUDITS these lists. Any change MUST keep that unit suite green
  (run `python -m pytest tests/unit/test_derived_ontology_audit.py -q`).
**Research:** How does the brain represent closed-class / function-word
  knowledge — as a single learned distributional lexicon (not 6 separate
  hand tables)? Cite sources on function-word acquisition / distributional
  semantics / linguistic closed-class universals.
**Plan — RECOMMEND (a) vs (b) with a clear verdict:**
  (a) Extend data/functional_lexicon.json with 6 new categories
      (common_words, topic_skip, subject_context, conditional_frame,
      attr_words, grammatical_concepts), copy hand contents as cold-start;
      add a `_closed_class(category)` accessor in functional_lexicon.py
      (mirror `_default_lexicon`); route the ~10 usage sites through it;
      hand lists become the fallback. MUST keep test_derived_ontology_audit
      green (the audit likely checks effective membership == union of both).
  (b) Leave the hand lists as-is (they already work) and only add a
      thin delegating accessor so future fit data can override.
  Default to (a) IF evidence shows it keeps the audit green and is
  behavior-preserving; otherwise (b). State which and why.

### B) Broaden consult / advice knowledge (genuine capability gap #2)
**Already solved by architecture (verify, do not re-build):** the engine
  enqueues unknown subjects to `_pending_learning_queue` →
  `_bg_learning_queue` (engine.py:2458, 2810, 3127) and researches
  them OFFLINE via web_learning, growing `_definitions`
  (engine.py:618, consulted at engine.py:2783). So health/programming
  coverage ACCRUES at runtime.
**Research:** Semantic memory acquisition from the world (not hardcoded);
  the cost of a SYNCHRONOUS consult→web fallback (latency, honest-
  uncertainty bar). Cite sources on semantic memory / retrieval.
**Plan — RECOMMEND one of:**
  (1) ACCEPT background-learning as sufficient (no code change; document
      why a sync fallback would regress cold-start latency + the
      honest-uncertainty bar). This is the main agent's current leaning.
  (2) ADD a synchronous consult→web fallback that (a) only fires when
      the subject looks like a domain-knowledge term (not self/identity),
      (b) preserves cold-start (no external call on first run), (c) never
      lowers the honest-uncertainty threshold. Give the exact insertion
      site + fail-closed guard.
  Default to (1) unless you find a design that demonstrably keeps
  cold-start behavior identical.

### D) Sensorimotor realization lexicon (property→phrase mappings)
**Data is LOCAL — NO download needed:**
  - data/cache/word_ratings/Lancaster_sensorimotor_norms_for_39707_words.csv
  - data/lancaster_encoder.npz
  - scripts/train_lancaster_probe.py  (existing fit script — READ it)
**Current hardcoded:** _SENSORY_DIM_PHRASE (engine.py:290),
  _PROP_TO_BINDER (engine.py:317), used in engine_generation.py:1130.
  _LANCASTER_ORDER (engine.py:286) is a FIXED encoder contract — KEEP.
  _lancaster_vector (engine.py:1156) already computes projections.
**Research:** Lancaster Sensorimotor Norms (Lynott, Connell, Brysbaert,
  Xu 2019/2020) — 11-dimensional strength ratings for ~39,707 words
  (perceptual: sight/sound/touch/taste/smell; action: hand/mouth/foot;
  interoceptive: pain/valence). Property verbalization derived from
  sensorimotor strength, not hand-coded.
**Plan:** (1) fit script producing data/sensorimotor_lexicon.json
  (property→verb+phrase from dominant effector/modality); (2) loader
  mirroring functional_lexicon.py (`default_sensorimotor_lexicon()`);
  (3) wire into engine.py:880 area (where `_func_lex` loads) as
  `self._sensorimotor_lex`, used in place of _SENSORY_DIM_PHRASE /
  _PROP_TO_BINDER; (4) hand maps KEPT as OOV fallback. Cold-start:
  hand maps used when fit file absent → identical behavior.

### F) Shape-based junk / boilerplate detection
**Current:** _JUNK_SNIPPET_DOMAINS (engine.py:344, hardcoded 45-domain
  blocklist); _WEBSITE_SHAPE (constants.py:401, TLD-tail regex);
  junk_score (constants.py:407 → junk_scorer.py, self-supervised
  classifier combining shape + learned signals); _is_keyboard_mash
  (constants.py:53).
**Research:** Predictive coding — boilerplate detected by statistical SHAPE
  (high vowel ratio, low information density, TLD suffixes), not a domain
  blocklist. Cite sources on statistical text "shape" / perplexity /
  boilerplate detection.
**Plan:** (1) add shape features to junk_scorer.py (vowel ratio, digit
  density, uppercase ratio, POS diversity) near where _WEBSITE_SHAPE is
  used; (2) wire into the snippet-quality path (engine_web_search.py
  snippet processing) via the existing `junk_score` call; (3) KEEP
  _JUNK_SNIPPET_DOMAINS as cold-start prior, shape runs FIRST, domain
  only when shape is uncertain; (4) NO new domain entries. Additive.

=====================================================================
PART 2 — HARD RULES (the user enforces these)
=====================================================================
1. NO FIXED THRESHOLDS. Adaptive / distribution-driven only. The engine
   already has the exemplar: `CognitiveChatEngine._adaptive_gate(key, x,
   strict=)` in engine.py (EMA mu/sigma; cold-start mu == legacy fixed
   value). Reuse it / mirror it. Constants are OK ONLY as cold-start
   priors that the adaptive gate consumes and then overwrites.
2. EVIDENCE-FIRST. Web claims cite REAL sources (title + URL). Code
   claims cite file:line. Do not invent APIs, files, or functions.
3. REUSE existing infra (do not build a second copy): functional_lexicon.py
   loader, junk_scorer.py, constants.py _WEBSITE_SHAPE/_is_keyboard_mash,
   _LANCASTER_ORDER (KEEP), _lancaster_vector, _adaptive_gate,
   _bg_learning_queue, _definitions.
4. BEHAVIOR-PRESERVING at cold-start (first run, no fit files, no
   external calls). Every adaptive gate starts mu == today's value.
5. REGRESSION-GATED. Two suites MUST stay green:
     python -m pytest tests/test_dehardcode_plan.py -q   → 21 passed / 1 failed
       (the 1 failure = test_meaning_of_life_not_dict_dump, PRE-EXISTING
        on HEAD — never a regression target)
     python -m pytest tests/unit/test_derived_ontology_audit.py -q  → ALL passed
       (audits the closed-class lists — HARD constraint for item H)
   If a change would add a failure to EITHER suite, the plan is rejected.
6. The 3 genuine capability gaps (same-turn WM, consult advice,
   self-eval noise) need NEW capability, not "delete a constant". H/B/D/F
   here are mostly dedup / learned-replacement / additive.

=====================================================================
PART 3 — DELIVERABLES (write these exact files)
=====================================================================
1. C:/Users/Likhith/Documents/projects/ravana/reports/brain_research_report_p2.md
   9-section-minimum research: for EACH of H/B/D/F — brain mechanism,
   cited real sources (title + URL, from the local API + web), and the
   RAVANA translation. For H and B, include your (a)/(b) and (1)/(2)
   RECOMMENDATION with reasoning.
2. C:/Users/Likhith/Documents/projects/ravana/reports/brain_fix_plan_p2.md
   Implementation plan: per item, code sites (file:line), reused infra,
   cold-start prior values, effort/risk, a "Do NOT touch" list, and a
   Verification section proving BOTH suites stay green per change.
3. (optional) reports/intentforge_stress_test_p2.md if you issue new
   stress queries beyond scripts/if_probe_raw.json.

You are RESEARCH + PLAN ONLY. Do NOT edit engine source. The main agent
will implement your plan and re-deliver. When done, say "RESEARCH + PLAN
COMPLETE" and list the two files you wrote.
