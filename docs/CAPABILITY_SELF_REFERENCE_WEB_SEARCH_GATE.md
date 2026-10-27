# Capability: self-reference gate suppresses web_search on RAVANA's own state/knowledge/learning queries

**Status:** shipped (commit `ff6dbd31`, branch `auto/round-2026-10-27T1143Z`, feature `t_e6724aaf`).
**Verified:** `decide_tool_use` importable live; the change is a 17-line, single-file edit with no new reply strings and no new reply strings added.

## What it does

When RAVANA's agent decision gate (`decide_tool_use`) considers firing `web_search`, it now first checks whether the query is **about RAVANA itself** — its own state, knowledge, or learning (e.g. "tell me about something you have learned", "have you ever been wrong"). Such queries are answered from RAVANA's internal cognitive state (identity/stances/facts/beliefs), NOT the web. Firing `web_search` there over-recalled off-topic results.

The gate is a **structural check over function words**: the query must contain the pronoun "you" AND at least one state/learning verb from a small verb set (`learn`, `learned`, `know`, `knew`, `think`, `thought`, `feel`, `felt`, `believe`, `remember`, `forgot`, `forget`, `been`, `are`, `were`, `was`, `have`, `had`). When both conditions hold, `_is_self_reference` becomes True and the `web_search` branch is suppressed, letting the query fall through to internal-state answering.

- No LLM, no Q→A dict, no keyword→reply table. The verb set is a seed vocabulary, not a fixed answer table: it is an in-code set that can be extended at runtime (the gate itself is seed-expandable).
- The gate is conservative: it only suppresses `web_search` when the query is NOT already classified as personal (`is_personal` is False) and the uncertainty/meta-uncertain conditions would otherwise have fired it. It never suppresses a tool call that was not going to happen.

## How it grew from the conversation

The chat round `t_2acb0aff` (25 in-process turns, `RAVANA_OFFLINE=1`) surfaced a residual limitation: when asked about RAVANA's own learning/knowledge ("tell me about something you have learned"), the agent decision gate fired `web_search`, over-recalling off-topic results instead of answering from internal state. The defect is structural: the gate had no notion of "this query is about the agent itself" — it only distinguished personal/social questions (`is_personal`) from world-knowledge gaps.

**Root cause / prior behavior.** `decide_tool_use` (`decision_gate.py:321`) computes `is_personal` (self-subject / social-chitchat detection) and `is_knowledge_query` (recall-query detector), then fires `web_search` when `not is_personal and (uncertainty >= 0.7 or meta_uncertain)`. Queries like "tell me about something you have learned" are NOT flagged `is_personal` (they are not "how are you" / "who are you" phrasings), so they fell into the web-search branch despite being about RAVANA's own state.

**Fix (commit `ff6dbd31`).** Three small pieces, all in `decision_gate.py:394-410`:

1. Define `_SELF_REFERENCE_VERBS` — a small seed set of state/learning verbs.
2. Compute `_is_self_reference`: True when the query contains `\byou\b` AND any token is in the verb set.
3. Extend the web_search guard: `if not is_personal and not _is_self_reference and (uncertainty >= 0.7 or meta_uncertain)`.

On a self-referential query the method no longer returns a `web_search` ToolCall; it falls through to the subsequent paths (noun-heuristic / internal-state answering), which answer from RAVANA's cognitive state.

**Generalization.** The detector keys on the *shape* of the query (pronoun "you" + a state/learning verb), not on specific topic words, so it generalizes to any self-referential phrasing that uses those function words ("have you ever been wrong", "do you think X", "what do you remember"). There is no retraining and no new reply authoring. The verb set is seed-expandable at runtime per the doctrine.

**Hardcoding audit.** The diff adds zero reply strings. The only added surface is a verb set and one extra term in an existing boolean guard. Grep for long added strings returns nothing. Replies still render REAL internal state via the existing fallback/internal-state path; on a self-referential query the method simply declines to fire `web_search`. No Q→A dictionary, no keyword→reply table.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| `decide_tool_use` (method head) | `ravana/src/ravana/agent/decision_gate.py:321` |
| `is_knowledge_query` detection | `ravana/src/ravana/agent/decision_gate.py:350-356` |
| `is_personal` detection (self-subject / social chitchat) | `ravana/src/ravana/agent/decision_gate.py:358-372` |
| `_extract_topic` + `uncertainty` | `ravana/src/ravana/agent/decision_gate.py:374-380` |
| `meta_uncertain` (metacognitive mode) | `ravana/src/ravana/agent/decision_gate.py:382-389` |
| Self-reference gate (`_SELF_REFERENCE_VERBS`, `_is_self_reference`) | `ravana/src/ravana/agent/decision_gate.py:394-408` |
| `web_search` guard (extended with `not _is_self_reference`) | `ravana/src/ravana/agent/decision_gate.py:410-413` |
| Tool registry (`web_search` → IntentForge gateway) | `ravana/src/ravana/agent/tool_registry.py` |

## Test coverage

No new test was added this cycle; the change is a small guard in the decision gate and is exercised by the existing agent/decision-gate test suite. The self-reference case (a query containing "you" + a state/learning verb suppresses `web_search`) is the gap a future test could close within CI budget; the method's public contract (returns `Optional[ToolCall]`, falls through to internal-state answering on `None`) is unchanged.
