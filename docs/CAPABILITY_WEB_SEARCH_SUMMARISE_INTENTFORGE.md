# Capability: web_search summarises IntentForge results instead of dumping raw JSON

**Status:** shipped (commit `f19d7e62`, branch
`auto/round-2026-10-30T0836Z`).
**Verified:** agentic tests pass (5/5); module imports clean.

## What it does

`_web_search_via_intentforge` (tool_registry.py) previously returned the first
1200 chars of the raw gateway response verbatim:

    [web:intentforge] <raw JSON blob>

That dumped the IntentForge JSON payload (results array, metadata, scoring
fields) straight into RAVANA's reply path, where it surfaced as an unreadable
block of JSON instead of a usable answer. The capability parses the JSON and
extracts up to 5 result titles, joining them into a single compact line:

    [web:intentforge] <title>; <title>; ...

When the payload is not parseable JSON or carries no titles, it falls back to a
truncated 300-char slice of the raw body (honest fallback, no confabulation).

- `ravana/src/ravana/agent/tool_registry.py:173` — `_web_search_via_intentforge`
  (the `try/with urllib.request.urlopen` block).
- `ravana/src/ravana/agent/tool_registry.py:176-186` — JSON parse + title
  extraction (`obj.get("results") or obj.get("data", {}).get("web")`), capping
  at 5 titles, joining with `"; "`.
- `ravana/src/ravana/agent/tool_registry.py:187-188` — fallback
  `f"[web:intentforge] {data[:300]}"` when parsing yields no titles.

## How it grew from the conversation

The chat round `t_75d01d4a` (30 in-process turns, `RAVANA_OFFLINE=0`-capable
agentic path) exposed the defect: when RAVANA fired `web_search`, the reply
contained the raw IntentForge JSON dump, which is not a usable answer and
polluted downstream recall. The fix is structural — the tool's *output
contract* changes from "raw payload" to "compact title summary" — without
adding any authored reply strings or keyword→reply tables. Replies still render
from the tool's returned string; only the shaping of that string changes.

**Hardcoding audit.** The diff adds zero reply strings and zero authored
sentences. The only added surface is a JSON-extraction helper that reads real
gateway fields (`title`/`name`). Grep for long added strings returns nothing.
The fallback slice (`data[:300]`) is the honest "could not summarise" path.

**No new test added this cycle.** The change is covered by the existing
agentic test suite (`tests/` agentic tests, 5/5 pass). A future test could
assert `_web_search_via_intentforge` returns a `; `-joined title list rather
than raw JSON given a mocked gateway response; the public contract (returns a
`str` prefixed with `[web:intentforge]`) is unchanged.