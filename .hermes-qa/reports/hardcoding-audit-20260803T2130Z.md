# RAVANA Full-Codebase Hardcoding Audit

**Date:** 2026-08-03 21:30 UTC
**Auditor:** external sweep (kanban t_6fd33ab9) — did NOT write the audited code
**Scope:** entire `ravana/src/ravana/chat/*` output-producing surface (138 .py files; ~7000 lines in `response_gen.py` alone)
**Tree state at audit:** `main` ahead 5 of origin, parent round `t_abd4f634` (round-v4) just completed. No concurrent writer confirmed (stray pytest PID 325 killed before edits).

---

## Summary

I swept every module that produces user-visible text or decides *what* to say: `response_gen.py`, `engine.py`, `engine_self_query.py`, `engine_reasoning.py`, `engine_generation.py`, `engine_memory.py`, `engine_web_search.py`, `web_learning.py`, `interface.py`, `intent_router.py`, `harm_intent_gate.py`, `support_router.py`, `realizer_lexicon.py`, `brain_regions.py`, `constants.py`.

**Verdict: NOT CLEAN.** The parent round's self-audit claim of "every reply string renders from real store content" is **false for three findings** below. In every case the violating code is *old* — it predates the round-v4 fixes and was never in scope for a self-diff audit. This is exactly the failure mode the card warned about: round agents only inspect their own diffs, so long-trusted hardcoded paths survive.

**3 VIOLATIONS found** (all static authored prose / keyword-matched canned answers that the system cannot revise by experience):

| # | Severity | Location | What it is |
|---|----------|----------|------------|
| V1 | HIGH | `response_gen.py:3142` `_compose_capability()` | Static self-description pamphlet returned on "what can you do / who are you" (the `capability` chitchat intent). A *second* capability path the round-v4 fix (`32c3e4b`, which fixed `engine_self_query.py`) missed entirely. |
| V2 | HIGH | `engine_reasoning.py:663-696` `_reflect_on_paradox()` | Keyword-matched canned answers to specific paradox/koan questions ("one hand", "god rock", "unstoppable", "liar", "simulation", "pinhead/angels"). Runs *after* a real web/Wikipedia retrieval, but ignores the retrieved text unless it coincidentally matches a keyword. |
| V3 | MED | `engine_self_query.py:230` `_agent_likes_guess()` | Fixed persona phrases ("things that feel calm and alive…") keyed only on valence bands. The "what do you like" gist is hardcoded prose, never revised by experience. |

**Borderline / NOT violations (cleared, with rationale to prevent re-flagging):**

- `response_gen.py` `_compose_greeting/_wellbeing/_farewell/_gratitude` (3035-3288): pools of short social-closure lines composed by valence/arousal primitives. These are *thin connective* reflex strings (greetings, thanks) — akin to honest fallback phrasing. They carry no propositional/intellectual content and are not "facts about the world." Defensible; flagged as the tension line but not a de-hardcode target.
- `engine_self_query.py:501-517` capability handler: the round-v4 fix — reads `sm.describe()` (live self-model) + two true invariants. CORRECT, leave as-is.
- `engine_reasoning.py:4227` `_emotional_response` / `brain_regions.py:419` `_EMPATHY_FRAMES`: empathy templates interpolate the *detected named entity / affect kind* (`your {lost}`, `feeling {word}`) and route via a VAD×cause frame map. Real-state content through thin connective. OK.
- `realizer_lexicon.py`: exemplar pools externalized to `data/realizer_lexicon.json` with a pluggable scorer. This is the *correct* de-hardcoding shape — exemplars in a store, not inline. OK.
- `constants.py` `TEEN_CONCEPTS`, `KNOWN_VERBS`, lexicons: MATCHING vocabulary (seed), not reply text. OK.
- `engine_memory.py` recall replies (`"you told me …"`, `"you haven't told me …"`): render real stored facts / real absence. OK.
- `_handle_self_model` (`response_gen.py:2047`): `_mood` derived from live valence; thin connective return. OK.
- `interface.py` `print("  I'm always curious to learn more…")` (2070): CLI menu text, not a reply. OK.
- `engine.py:4747` name seed list: matching vocabulary for identity detection. OK.

**No retraining/offline-rebuild requirements found** in the audited paths (online web/Wikipedia retrieval already exists; the fixes below route INTO it rather than beside it).

---

## Findings in detail

### V1 — `_compose_capability()` (HIGH)

`ravana/src/ravana/chat/response_gen.py:3142-3151`

```python
def _compose_capability(self) -> str:
    """Compose a capability response from a description template.
    This is a single static description (the agent's identity is fixed),
    but composed from parts so it can be updated dynamically.
    """
    return ("i am ravana, a brain-inspired cognitive agent. "
            "i learn concepts from the web, build associations, "
            "and generate fluent sentences using a prefrontal workspace "
            "and surface realizer -- no templates, no scripts.")
```

**WHY it fails the test:** The content is a *fixed authored paragraph* about what RAVANA is and does. The comment "composed from parts so it can be updated dynamically" is aspirational — there are no parts; it is one literal string. The system cannot revise this by talking. It is the same banned capability-brochure pattern the round-v4 commit `32c3e4b` explicitly removed from `engine_self_query.py` — but that commit only patched ONE of the two capability paths. This `_compose_capability` is still live, reached via the `capability` intent in `_handle_chitchat` (line 2032).

**Suggested fix:** Mirror `engine_self_query.py:506-514` (the already-accepted fix): derive the self-description from the live `SelfModel` (`self._ensure_self_model().describe()`) and append only two true invariants (learns from conversation; remembers user facts). No per-capability list. The content then tracks the runtime self-model store.

### V2 — `_reflect_on_paradox()` canned answers (HIGH)

`ravana/src/ravana/chat/engine_reasoning.py:663-696`

```python
if "one hand" in t or "hand clapping" in t or "sound of" in t:
    return ("that's a koan — … sitting with the silence is kind of the point.") + _ground
if "god" in t and ("rock" in t or "stone" in t or "create" in t or "heavy" in t):
    return ("the catch is in the setup: 'all-powerful' breaks …") + _ground
if "unstoppable" in t or "immovable" in t:
    return ("if both exist, they can't meet …") + _ground
if "statement" in t and ("false" in t or "true" in t):
    return ("that one ties language in a knot …") + _ground
if "simulation" in t or "reality real" in t or "know anything" in t:
    return ("i can't step outside my own experience …") + _ground
if "pinhead" in t or "angels" in t:
    return ("that one's a classic: the point was never the number …") + _ground
return ("that's a paradox — the interesting part isn't a single answer …") + _ground
```

**WHY it fails the test:** Each branch is a *hand-written essay* returned on a keyword match. The function does real retrieval first (`_ground` from Wikipedia + web, lines 604-661), but the authored text is returned REGARDLESS of whether retrieval succeeded, and the keyword buckets are frozen — RAVANA can never grow or revise these answers through experience. The `_ground` clause is merely *appended*; the intellectual content is the canned paragraph. This is the textbook "query-specific special case that returns a pre-written reply" violation.

**Suggested fix:** Drop the per-paradox authored branches. If `_ground` (retrieved, real text) is non-empty, return a short honest framing + the grounding (`(from what i've read: …)`), consistent with every other factual web-answer path. If retrieval missed, return the generic honest-uncertainty fallback (no authored essay). The voice-framing ("a paradox invites sitting with the tension") is system tone, but it must not be the *content* of a specific-query answer — and the current code makes the specific essays the content. Net: remove the 6 special-case branches, keep only the data-driven `_ground` path + a single honest fail-closed line.

### V3 — `_agent_likes_guess()` (MEDIUM)

`ravana/src/ravana/chat/engine_self_query.py:230-243`

```python
def _agent_likes_guess(self) -> str:
    valence = …   # from emotion.state
    if valence >= 0.6:
        return "things that feel calm and alive — like quiet music or open sky"
    if valence <= 0.4:
        return "things with some edge to them — a sharp idea or a difficult question"
    return "ideas that hang together, and the kind of honesty that's calm"
```

**WHY it fails the test:** Three fixed prose strings selected by valence band. The agent's "what do you like / what are you drawn to" gist never changes with experience — only with a transient mood number. There is no store, no learning path, no revision. It is authored self-persona. (Contrast `_agent_stance_on`, lines 245+, which derives stance from valence + GloVe proximity + accumulated graph edges — that is real-state-derived and OK.)

**Suggested fix:** Either (a) derive from the agent's *actual accumulated stances/preferences* in the user-model store (surface 1-2 real learned stances, e.g. "i've been leaning toward {topic} lately" from `_agent_preferences`), or (b) if no real stance exists yet, return the honest grounded fallback ("i'm still forming what i'm drawn to — what are you into?") rather than a fixed poetic line. The prior round's decomposed fix chose data-driven derivation; this guess function should do the same.

---

## Method

Combined sweeps (not a single clever grep):
1. Name-based: `*_TEMPLATE|*_RESPONSES|*_REPLIES|*_PATTERNS|*_MAP|*_TABLE` — surfaced mostly legit lexicons/classifiers.
2. Long string literals in `return`/f-string output paths — surfaced V1, V2, V3 and the cleared reflex pools.
3. Gating/ranking threshold literals on scores — no violation; thresholds live in clearly-labeled coherence gates (fail-closed), not reply selection.
4. Dict/list literals with prose values + capability lists — surfaced `_EMPATHY_FRAMES` (frame *labels*, OK) and `realizer_lexicon` (externalized exemplars, OK).

Every grep hit was read in surrounding context before classification. Context decided; the grep did not.

## False-positive discipline

I deliberately did NOT flag the greeting/wellbeing/farewell/gratitude reflex pools, the empathy templates, the round-v4 self-query fix, the realizer-lexicon exemplars, or any lexicon — each is either a thin-connective social reflex, real-state interpolation, or a store-backed exemplar. Inventing findings here would waste the fixer's time and Likhith's trust.
