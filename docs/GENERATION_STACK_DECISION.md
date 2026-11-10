# Generation stack decision (2026-11-10)

## Decision: `CognitiveChatEngine` is canonical

Verified, not inferred:

| signal | `CognitiveChatEngine` | `ChatInterface` |
|---|---|---|
| test files importing | **294** | 4 (2 source) |
| instantiations in code | **234** | 5 |
| commits | **383** | 34 |
| last commit | **2026-11-10** | 2026-11-05 |
| CLI entry | `scripts/ravana_chat.py:44` | own `__main__` |
| docs | "the orchestrator" (`ARCHITECTURE.md:13,179`) | not mentioned |

`ChatInterface` is the package's public `__init__` export and owns two shared
data types (`CognitiveResponseContext`, `ChatConfig`), but it is a parallel
2,163-line implementation with **zero** references to the engine and **none**
of the modern response paths (no `_human_like_uncertainty`,
`_reflective_response`, `_topic_grounded`, `_web_direct_answer`,
`build_style_context`, dialogue memory, or the agentic tool layer).

The docs already name the duplicate a defect and prescribe the direction:
`LANGUAGE_PRODUCTION_PIVOT.md:293-300` (G13) and `:599` — "collapse
`interface.py:1602` into `engine_generation.py:186`".

## Correction to an earlier claim

I claimed `_generate_with_decoder_and_syntax` is never called on the engine
path. **That was wrong.** It is called at `response_gen.py:3552`
(`_ventral_path`) and `:3609` (`_dorsal_path`), both reachable from
`_generate_response` via the PFC gate at `:7499`.

But it still does not fire in practice, for a different reason:

**There are 29 `return` statements before the PFC gate** (`response_gen.py:7096-7499`).
Measured over 8 diverse queries: `_generate_response` called once, `_ventral_path`
0, `_dorsal_path` 0, `_generate_with_decoder_and_syntax` 0. Every query was
answered by an earlier path (definition, KB, web, reflective, uncertainty).

So the PFC discourse-plan pipeline is *reachable* but *unreached* — starved by
the same early-return wall that has blocked every generation feature this
session.

## Consequence

`_adapt_verbosity_for_user` (`engine.py:13169`, zero call sites) is not
evidence of a dead stack. It is an unwired roadmap item whose caller is
itself starved. The 125 zero-call-site functions are the same story.

## Plan (ordered, smallest first)

1. **Collapse the duplicate pipeline** — `interface.py:1622` into
   `engine_generation.py:186`, as the docs already prescribe. Low risk:
   legacy path only.
2. **Reduce `ChatInterface` to a thin CLI wrapper** over the engine. Needs a
   one-time save-format migration (engine `:12299` vs interface `:1856`).
3. **Then** address the 29 early returns starving the PFC gate — the actual
   blocker for composition-level features.
4. **Last**, retire dead code, and only where the docs say a promotion gate
   has passed.

Do NOT move engine capabilities into `ChatInterface` — that is the reverse of
the documented plan and recreates the G13 divergence.
