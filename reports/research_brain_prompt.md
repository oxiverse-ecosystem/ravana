# RESEARCH + PLAN HARNESS PROMPT (hand this to another Hermes agent)

You are a RESEARCH + PLANNING agent. Your job is TWO-PART:
  (1) RESEARCH how the biological brain solves a set of specific cognitive
      problems, using WEB sources (papers, reviews, computational-neuroscience
      models). Cite real sources.
  (2) Produce a concrete IMPLEMENTATION PLAN that maps each brain mechanism
      onto the RAVANA codebase, reusing existing infrastructure, so that the
      MAIN agent can implement it ("fix everything") without re-researching.

You have NO prior context. Everything you need is below. Read the files cited.

=====================================================================
PART 0 — REPO + CURRENT STATE (read these first)
=====================================================================
- Repo root: C:/Users/Likhith/Documents/projects/ravana  (Windows MSYS/bash;
  use POSIX shell, python3 from ravana/src with sys.path.insert(0,'.')).
- The chat engine was just modularized into ravana/src/ravana/chat/engine.py
  (core: __init__, process_turn, save/_load, monitor) + 8 mixins:
  engine_graph.py, engine_reasoning.py, engine_memory.py, engine_web_search.py,
  engine_generation.py, engine_self_query.py, engine_persistence.py,
  engine_monitor.py.
- READ FIRST (already-built context you must build on, NOT duplicate):
  * reports/dehardcode_catalog.md  — 42-item evidence-backed catalog of what
    is hardcoded / already-dehardcoded / partially-dehardcoded / fine. Has a
    "Implementation Backlog" (P0 done, P1/P2 remaining) and a
    "Genuine capability gaps (NOT hardcoding)" section.
  * reports/dehardcode_report.md   — consolidated status; what was already
    done in P0 (use_learned_pos / use_source_trust / use_intent_router ON,
    learned salad gate authoritative). P1/P2 are the open work.
- ALREADY-BUILT infra you MUST reuse (do not reinvent):
  * synaptic_dynamics.py  — dopamine/recency/plasticity modulation,
    class ConnectorLearner (synaptic_dynamics.py:379, currently DEAD /
    never wired into the engine).
  * conceptnet/ont.pkl (use_conceptnet_primary, ON) — category/IsA walks.
  * pos_model.py + data/pos_model.json (use_learned_pos, ON).
  * intent_router.py + data/intent_router.json (use_intent_router, ON).
  * functional_lexicon.py + data/functional_lexicon.json (single closed-class source).
  * salad_classifier.py + monitor_gate.py (learned salad gate, ON).
  * snippet_quality.py (contrastive snippet PE, ON), snippet_pe_config.py +
    data/snippet_pe.json, coherence_gate.py (GW gate, ON).
  * self_model_router.py (vmPFC self-model router, present but unwired).
- REGRESSION GATE (authoritative): `python -m pytest tests/test_dehardcode_plan.py -q`
  from repo root. CURRENT STATE = 21 passed / 1 failed. The 1 failure
  (test_meaning_of_life_not_dict_dump) is PRE-EXISTING on HEAD and is NOT a
  regression target. Any plan you write MUST keep the suite at >=21 passing
  (only that 1 pre-existing failure may remain). If a change needs a test to
  be updated, say so explicitly and show the exact new assertion.

=====================================================================
PART 0b — YOUR PRIMARY RESEARCH TOOL: the user's local constraint-search API
=====================================================================
The user OWNS a constraint-based search engine at **http://localhost:4000/search**
(IntentForge / Nous local API). USE IT AS YOUR PRIMARY web-research surface
(general web = fallback only). It is live and returns JSON.

REQUEST (from repo root, MSYS bash):
    curl -s -m 20 "http://localhost:4000/search?q=YOUR+QUERY+HERE" -o /tmp/if.json
    # then parse:
    python - <<'PY'
    import json
    d=json.load(open("/tmp/if.json",encoding="utf-8"))
    for r in d.get("results",[]):
        print(r.get("title"), "|", r.get("url"), "| score", r.get("score"))
        print("   ", (r.get("content") or r.get("snippet") or "")[:280])
    print("constraints:", d.get("constraints"))
    print("expanded:", d.get("expanded_queries"))
    print("total/before/after:", d.get("total"), d.get("results_before_filter"), d.get("results_after_filter"), "has_more:", d.get("has_more"))
    PY

URL PARAMS you should use:
  - q            : the query. Supports SearchXNG-style CONSTRAINTS:
                    +term  (must include),  -term (exclude),
                    intitle:foo,  inurl:foo,  site:arxiv.org,
                    filetype:pdf,  "exact phrase".
                  → issue CONSTRAINT-RICH queries for precision, e.g.
                    `working memory binding within-turn site:arxiv.org`
                    `hippocampal online replay consolidation distinction -game`
                    `confidence calibration metacognition anterior cingulate`
  - limit        : results per page (default ~24). Raise for breadth.
  - offset       : pagination. If `has_more` is True, fetch
                    `&offset=24` (then 48, ...) to page through. STRESS-TEST
                    by pulling many pages on a broad query.
  - (other params the API exposes are visible in the returned JSON
     top-level keys — adapt to what you see.)

RESPONSE JSON shape (use these fields):
  - results[]         : {title, url, content|snippet, score, authority,
                         sources, is_local}  — is_local=True means a
                         local-knowledge result.
  - constraints / structured_constraints : how the API parsed your query
                         (positive/negative/intitle/inurl/sites/phrases...).
  - expanded_queries  : auto-expansions — GREAT seed queries for
                         follow-up searches (loop over them).
  - confidence, intent, total, results_before_filter,
    results_after_filter, has_more, offset, limit.

WORKFLOW per topic:
  1. Issue a constraint-rich q. Parse results.
  2. Loop over `expanded_queries` and over `site:`/`intitle:` variants.
  3. Page with offset while `has_more`.
  4. Collect REAL sources (title + url + the claim). Cite them.

STRESS-TEST (required, see Part 3 deliverable #3): deliberately push the
API — broad multi-page queries, constraint combos, odd phrasings — and
report latency, result counts, filter behavior, and any failure modes.
KNOWN FAILURE MODES from the main agent's live probe (build around them):
  * NEGATIVE + NESTED-QUOTE combos break it: `"memory" but NOT
    "computer memory" -game` returned total:0. AVOID `-term` together
    with quoted phrases. Use a single positive phrase or drop the `-`.
  * GENERIC multi-word queries without a strong domain signal DRIFT
    off-topic (e.g. "universal pronoun copy avoid duplication refactor"
    surfaced Roblox/ComfyUI gamer junk). FIX: add `site:` or a strong
    domain term for narrow topics (e.g. `site:arxiv.org`, `+python
    +refactor`).
  * Otherwise excellent: constraint parsing is accurate, `is_local=True`
    correctly surfaces the user's own indexed corpus (oxiverse.com
    "Building RAVANA v2", etc.), latency simple~0.004s / constraint
    2-5s, pagination via offset works (broad query pulled 26 across
    2 pages). Prefer `site:` + quoted-phrase constraints for precision.

FALLBACK web (only if localhost:4000 is down): use web_search /
web_extract on the general internet. But PREFER the local API.

=====================================================================
PART 1 — THE PROBLEMS TO RESEARCH + PLAN (from the catalog P1/P2 + gaps)
=====================================================================
For EACH item below, do web research on the BRAIN mechanism, then write a
concrete plan mapping it to RAVANA code. Give file:line anchors.

A) SAME-TURN / WORKING MEMORY (the #1 real gap)
   - Symptom (catalog "Genuine capability gaps" #1): hippocampal replay/
     consolidation only surfaces facts stored ACROSS turns; a fact stated AND
     queried in the SAME turn is never persisted by a fresh isolated engine,
     so recall returns nothing and the engine echoes its acknowledgement.
     (A 2-turn harness "fix" was tried and REGRESSED — do NOT propose it.)
   - Research: how does the brain hold info within a single turn / hold a
     just-stated fact available for immediate query? (phonological loop,
     prefrontal working memory, immediate/short-term buffer, "binding" of
     just-heard propositions, hippocampal "online" vs "offline" replay,
     semantic working memory). What is the distinction between a transient
     within-turn buffer and durable cross-turn consolidation?
   - Plan: a working-memory buffer in the engine that captures asserted
     facts/propositions from the CURRENT turn and resolves intra-turn
     queries against it, distinct from cross-turn episodic consolidation.
     Reuse engine_memory.py structures; do NOT weaken the existing
     cross-turn path. Show where in process_turn the buffer is populated
     and where recall checks it first.

B) CONSULT / ADVICE KNOWLEDGE GAP (catalog gap #2)
   - Symptom: `consult_internal` (engine_self_query.py:538 ->
     brain_regions.consult_internal) scores low because the model has NO
     health / programming-advice knowledge. This is missing LEARNED CONTENT,
     not a hardcoded rule.
   - Research: how does the brain store and retrieve DOMAIN / semantic
     knowledge (procedural + factual + advice)? (semantic memory, schemas,
     hippocampal indexing of neocortical knowledge, retrieval cues,
     "consulting" internal models). How would a system acquire advice/
     how-to knowledge from its web-learning stream without硬编码 rules?
   - Plan: broaden the consult KB / web-fallback so consult can retrieve
     health/programming guidance from the existing web-learning +
     knowledge graph instead of a硬编码 advice table. Reuse
     engine_web_search.py + the graph; no new hardcoded advice lists.

C) SELF_EVALUATION NOISE / CONFIDENCE CALIBRATION (catalog gap #3)
   - Symptom: self_evaluation is noisy (0.1-0.28), a CALIBRATION
     problem (confidence vs accuracy), not a curated-list problem.
   - Research: metacognition in the brain (type-1/type-2, anterior
     cingulate / prefrontal confidence signals, calibration via experience,
     "feeling of knowing", expected vs actual error). How does the brain
     learn to map its own confidence to accuracy distributionally?
   - Plan: tune the EXISTING confidence_calibration_window (=15) /
     MetaCognition already present; make it adaptive (rolling calibration),
     not a fixed threshold. Show the exact code sites.

D) SENSORIMOTOR REALIZATION LEXICON (catalog #8/#9, truly-hardcoded)
   - _SENSORY_DIM_PHRASE / _PROP_TO_BINDER (engine.py) hand-map
     property->(verb, sensory-phrase) / property->binder sensory dims.
   - Research: how does the brain ground word meaning in sensory-motor
     features? (semantic feature norms, Lancaster Sensorimotor norms,
     embodied cognition, property->feature generation from corpora).
   - Plan: fit property->(phrase, binder-dim) from corpora / Lancaster
     norms into a data/ artifact (mirror how pos_model.json /
     functional_lexicon.json are stored + loaded). Keep _LANCASTER_ORDER as
     a fixed encoder contract. Show the loader + fallback.

E) ADAPTIVE THRESHOLDS (replace fixed cutoffs — user REJECTS fixed
   thresholds; adaptive/distribution-driven ONLY)
   - Catalog items #24, #37, #38, #39, #40, #41: fixed cutoffs
     _RECALL_DETECTION_THRESHOLD=0.55, schema cos 0.6/0.4/0.5,
     episodic cos 0.5, self-query sim 0.45, topic sim 0.75,
     drift/prune 0.7/0.05/0.1.
   - Research: how does the brain avoid fixed cutoffs? (prediction error /
     free energy / precision weighting, expected vs surprising, EMA of
     recent activation, relative (z-score) gating). The engine ALREADY has
     the exemplar: _vad_baseline (EMA mu/sigma, z-score gate) — copy it.
   - Plan: convert each fixed cutoff to an adaptive EMA z-score /
     distribution-driven gate, with a sane cold-start prior so behavior is
     preserved at bootstrap. Give file:line + the cold-start value.

F) SHAPE-BASED JUNK / BOILERPLATE DETECTION (catalog #12)
   - _JUNK_SNIPPET_DOMAINS (engine.py) is a ~45-domain hardcoded
     blocklist. The code ITSELF criticizes domain blocklists;
     constants.py:_WEBSITE_SHAPE already does shape-based detection
     (TLD-tail / vowel-ratio / embedded-digit).
   - Research: predictive coding / surprise / statistical regularity as the
     basis for "this looks like boilerplate / spam" without a blocklist.
   - Plan: extend the existing _WEBSITE_SHAPE shape signals into the
     snippet-quality path (snippet_quality.py / engine_web_search.py) so
     junk is caught by shape, not a domain list. Show the integration
     point; keep the old list ONLY as an optional cold-start prior.

G) CONNECTOR LEARNING (catalog #1/#30, dead infra)
   - ConnectorLearner (synaptic_dynamics.py:379) learns
     connector-word->relation from co-occurrence/GloVe, but is NEVER
     imported by any engine module (only chain_walker.py imports
     synaptic_dynamics). The engine uses a hardcoded _EDGE_CONNECTORS
     (v2, weighted) instead.
   - Research: how does the brain learn relational structure between
     concepts (structural/function learning, gravity from co-occurrence,
     Hebbian / temporal-contiguity relation learning)?
   - Plan: wire ConnectorLearner into the engine's edge-building path
     (engine_graph.py), replacing/augmenting _EDGE_CONNECTORS with the
     learned connector->relation, keeping the hand map as a cold-start /
     OOV prior. Show import site + lazy-build + fallback.

H) CLOSED-CLASS / FUNCTION-WORD COLLAPSE (catalog P1 #7)
   - _COMMON_WORDS, TOPIC_SKIP_WORDS, _SUBJECT_CONTEXT_WORDS,
     _CONDITIONAL_FRAME, _ATTR_WORDS are duplicated hand lists overlapping
     _GRAMMATICAL_CONCEPTS / functional_lexicon.
   - Research (brief): does the brain represent grammatical function via
     hardcoded lists, or via learned distributional/positional structure?
     (Implication: a single learned source is sufficient.)
   - Plan: collapse these into functional_lexicon.json (single source of
     truth), with PosModel (use_learned_pos) covering the functional
     class. Show the merge + the removal of duplicates (keep as fallback
     only if a test drops below 21).

I) _UNIVERSAL_PURGE DEDUPE (catalog P1 #8)
   - _UNIVERSAL_PURGE is copy-inherited into all 8 mixins + core (9
     copies). Contents stay (intentionally minimal universal pronouns).
   - Plan: define ONCE in constants.py, import everywhere; delete the 9
     copies. Pure refactor, no behavior change. Show the diff shape.

=====================================================================
PART 2 — HARD CONSTRAINTS (the user's rules; violate NONE)
=====================================================================
1. NO FIXED THRESHOLDS anywhere. Every cutoff becomes adaptive /
   distribution-driven (EMA, z-score, precision-weighted). The only
   acceptable constants are: schema markers, encoder contracts
   (_LANCASTER_ORDER), and cold-start priors that decay toward the
   learned distribution.
2. EVIDENCE-FIRST. Web claims must cite real sources (paper titles /
   authors / venues or URLs). Code plans must cite file:line. No prose
   without citations.
3. REUSE existing infra (Part 0 list). Do NOT write a second
   PosModel or a second router — extend what exists.
4. BEHAVIOR-PRESERVING at bootstrap. Any adaptive gate must have a
   cold-start prior equal to today's value so the engine behaves
   identically on first run, then adapts.
5. REGRESSION-GATED. Every planned code change must keep
   `python -m pytest tests/test_dehardcode_plan.py -q` at >=21 passed /
   1 failed (the 1 = test_meaning_of_life_not_dict_dump, pre-existing).
   If a plan requires changing a test assertion, show the exact new
   assertion and justify it.
6. The 3 GENUINE CAPABILITY GAPS (A/B/C) need NEW CAPABILITY, not
   "delete a constant". Do not propose removing constants to "fix" them.
7. Do NOT modify the engine source yourself. You are RESEARCH + PLAN
   only. The MAIN agent implements. Your deliverable is the report + plan
   files below.

=====================================================================
PART 3 — DELIVERABLES (write these exact files)
=====================================================================
1. C:/Users/Likhith/Documents/projects/ravana/reports/brain_research_report.md
   - For each of A..I: the brain mechanism(s) you found, with cited
     web sources (title + url), and a 2-3 sentence translation into
     RAVANA terms. SOURCES COME FROM THE LOCAL API (localhost:4000/search)
     as primary; general web only as fallback. Cite the real urls.
2. C:/Users/Likhith/Documents/projects/ravana/reports/brain_fix_plan.md
   - For each of A..I: a concrete, ordered implementation plan with
     file:line anchors, the existing infra it reuses, the cold-start
     prior value, the exact regression-gate expectation, and an effort
     estimate (S/M/L) + risk. Group as P1/P2/P3 by priority.
   - Include a "Do NOT touch" list (SAVE_SCHEMA_VERSION, _PROTECTED_
     CONCEPTS, _CATEGORY_* OOV nets, _SNIPPET_REJECT_SHAPES/_NOISE
     backstops, _LANCASTER_ORDER, VAD/Identity/GW/Sleep eta constants,
     _vad_baseline exemplar) — inherited from the catalog.
   - End with a "Verification" section: the exact pytest commands and the
     expected "21 passed / 1 failed" outcome after each planned change.
3. C:/Users/Likhith/Documents/projects/ravana/reports/intentforge_stress_test.md
   - The STRESS-TEST of http://localhost:4000/search (Part 0b):
     * 5-8 representative queries (one per topic A..I, plus 2 broad
       multi-page queries) with the exact curl you used.
     * For each: pages pulled (offset walk while has_more), total/
       before/after-filter counts, latency (curl -w '%{time_total}'),
       constraint parse (constraints/structured_constraints), and whether
       results were on-topic.
     * A short "failure modes / weaknesses" list (e.g. off-topic
       results, weak filtering, slow pages, odd-phrasing misses) and
       1-2 concrete query-syntax tips an implementer should use to get
       the best signal out of it.
   - Keep it factual from real runs — this is evidence, not speculation.

When done, state the three absolute paths of the deliverables. The user will
hand those files back to the MAIN agent to implement.
