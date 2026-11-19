# Frozen web snapshots for the n=60 web arm

`web_fixtures.json` is the checked-in snapshot bundle used by
`scripts/measure_fronts_web_frozen.py` to make the web-enabled n=60 arm
measurable without live network I/O.

Schema (JSON, sorted keys, 2-space indent):

```jsonc
{
  "schema": "ravana-web-fixtures-v1",
  "captured_at": "...Z",          // UTC capture date
  "provider": "intentforge(localhost:4000) -> local_api -> duckduckgo -> oxiverse",
  "queries":       { "<normalized query>": {"query", "hits":[{title,url,content,...}], ...} },
  "kb_describes":  { "<normalized concept>": {"concept", "extract"} },
  "tool_web_search": { "<normalized query>": {"query", "response"} }
}
```

`queries` is the required query -> list-of-hits map. The other two sections
freeze the agentic `web_search` tool and the secondary Wikipedia `kb_describe`
lookup; both are live retrieval paths reached by the same probes, and freezing
them is what lets the harness execute web-enabled code paths at zero network
cost. Counts in the final run: **1087** queries, **61** KB extracts, **207**
tool responses.

## Exactly what was run

All commands from the repo root with `uv run python` (project venv, numpy/bs4
available). `RAVANA_OFFLINE` is set to `0` by the harness itself.

1. **First live capture pass** — one live call per distinct query, in-memory
   cache, incremental JSON writes:

   ```
   uv run python scripts/capture_web_fixtures.py
   ```

   Result: `rc=0`, 434 probe rows, 985 query fixtures + 58 KB extracts,
   `elapsed_s=2794.5`. The harness's committed data files were never written:
   the capture script redirects them to `scratch/front_probes_web_capture*`.

2. **Second live capture pass** — the first pass did not stub the agentic
   `web_search` tool (`ravana/agent/tool_registry.py`), so those calls still
   went live and the frozen replay attempted them. The capture script was
   extended to cache that tool too; re-running it resumed from the existing
   JSON (no duplicate live calls) and wrote the tool section:

   ```
   uv run python scripts/capture_web_fixtures.py
   ```

   Result: `rc=0`, 434 rows, 1049 queries + 61 KB + 207 tool responses,
   `elapsed_s=1085.4`.

3. **Missing-query top-ups.** The live harness has a 12 s per-answer search
   deadline; slow live calls truncate variant loops, so a fast frozen replay
   can request variants the live pass never reached. A diagnostic replay
   dumped the missing normalized keys to JSON, and the provider was queried
   once per missing query:

   ```
   uv run python scripts/capture_web_fixtures.py --top-up <missing.json>
   ```

   Two rounds (32 + 6 keys) converged to zero missing fixtures. Final
   fixture count: **1087** query entries.

4. **Frozen run to completion** (direct invocation, no wrapper):

   ```
   uv run python scripts/measure_fronts_web_frozen.py
   ```

   Result: `rc=0`, **434** probe rows, `elapsed_s=151.3` (≈2.5 min, versus
   ≈47 min for the live capture pass), zero network attempts, zero saves,
   zero missing fixtures. Summary written to
   `scratch/front_probes_web_frozen_summary.json`.

## Stale-summary trap

`scratch/front_probes_web_summary.json` is the old n=27 result (756 bytes,
mtime 2026-11-19 07:24). It was left untouched. The frozen run writes a new
file (`front_probes_web_frozen_summary.json`, 1846 bytes, mtime
2026-11-20 00:54) with `probe_rows: 434` and `n_per_front` = 60 for all seven
fronts. The harness asserts the row count before writing and prints both the
row count and `elapsed_s`; a timed-out or partially-dead run cannot leave a
smaller-n summary that looks like a result.

## Measured scores (frozen run, full 60 per front)

| Front | Primary (30) | Sealed (30) | Full 60 |
|---|---:|---:|---:|
| language understanding / conversation | 87 | 100 | 93 |
| grounded factual answers | 77 | 67 | 72 |
| opinions / stance | 90 | 90 | 90 |
| thinking / reasoning | 13 | 30 | 22 |
| speaking / surface realization | 83 | 80 | 82 |
| code generation | 100 | 100 | 100 |
| language learning (online) | 80 | 80 | 80 |
| **MEAN** | | | **77** |

Gates: G1 PASS, G2 PASS, G3 PASS, GC PASS, G4 PASS, G5 PASS, G6 FLAG
(code generation = 100 is saturated and INVALID as a progress signal).
No threshold or rubric was changed; the scoring region is byte-identical to
`scripts/measure_fronts.py` (G4 verified). Scores match the live capture run
exactly at the full-60 level, which is the replay-fidelity check.

Honest reading: grounded (72) and thinking (22) are the weak fronts; the
sealed thinking score (30) exceeds primary (13), so there is no evidence of
overfitting there — the low full score is a real capability gap. No tuning
was applied.

## Guarantees in the frozen harness

* `SearchEngine.search` is replaced by fixture lookup; a missing key raises
  `MissingFixtureError` — no live fallback.
* `urllib.request.urlopen`, `SearchEngine._call_api`, the `web_search` tool and
  `CognitiveChatEngine.kb_describe` are stubs/tripwires; any live call raises
  and is counted. The run refuses to write a summary unless the counters are
  zero.
* `CognitiveChatEngine.save` is a tripwire; the run refuses to write a
  summary unless zero saves were attempted. No weight file is modified.
* `RAVANA_OFFLINE=0` throughout, so the web-enabled answer/learning code paths
  still execute — only their retrieval inputs are frozen.
