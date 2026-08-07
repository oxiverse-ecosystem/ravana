# Independent Audit Report — Docs Rounds t_df2ce0c1 (RAVANA) + t_e593bde5 (IntentForge)

Auditor: t_981b56e0 (independent reviewer — wrote none of the audited work)
Date: 2026-08-05
Scope: real committed diffs of both rounds, re-verified by executing commands / live API probes.

---

## Per-check verdicts

### CHECK 1 — Is every documented claim REAL?

**RAVANA (round t_df2ce0c1)** — OK
- `docs/_generated/suite-timings.md` (ac25e75): timings claimed were re-derived from the worker's own measured output; the worker explicitly marked CI-minute non-equivalence as not comparable (honest).
- `docs/TESTING.md` (04af788): every claim cites a real path. Verified independently:
  - `ci.yml` timeout-minutes = `10/12/12/10/20` (grep lines 37/63/94/119/149) — proves the old "15 min" claim was correctly corrected.
  - `conftest.py:52` sets `RAVANA_OFFLINE=1` via `os.environ.setdefault` — real.
  - `pyproject.toml:75` registers the `slow` marker; `tests/unit/test_generation.py:232` carries bare `@pytest.mark.slow`. Confirmed: `@pytest.mark.slow` ALONE excludes NOTHING (no skip/modifyitems hook) — matches the doc.
  - `.github/scripts/check_lfs_assets.py` exists; `scripts/check_lfs_assets.py` does NOT — confirms the doc's own path correction.
- `DEVELOPMENT.md` / `GETTING_STARTED.md` corrections trace to the same ci.yml evidence. OK.

**IntentForge (round t_e593bde5)** — 1 VIOLATION (see below)
- Transcript `docs/_generated/api-transcript.md` is genuine: 37 real queries captured from `localhost:4000`, bodies are real (verified by re-issuing live queries and matching top-level keys).
- Error codes `empty_query` / `invalid_query` (transcript lines 5803/5835/6195/6227/6259/6292) match docs. OK.
- `confidence` is a real float (transcript samples 0.31, 0.60, 0.90 — NOT fixed 0.75). OK.
- TTL = 5 min: `services/gateway/src/main.rs:5949` → `Duration::from_secs(300)`; live re-measure cold 6.3s → hot 0.004s. OK.
- Operators `site:`/`filetype:`/`intitle:`/`inurl:`/`after:` trace to transcript blocks 32–36. Unverified ones (`price:`/`lang:`/`intext:`/`related:`, 200+upstream_unavailable) are HONESTLY marked Unverified in-doc. OK.
- **VIOLATION**: `API_REFERENCE.md:230` claims `/search/fast` has **no top-level `source` field**. FALSE. Live API returns top-level keys `['count','results','source']`, and the worker's OWN transcript (block 13, line 3197/3200) shows `"source":"local"` top-level (`topkeys=['count','results','source']`). The doc contradicts its own cited evidence. (FIX card: t_fdab0124.)

### CHECK 2 — New tests: honest, deterministic, in budget?

**RAVANA** — OK
- `tests/unit/test_personal_fact_store.py` (bb5340d): 20 tests. Ran independently:
  `RAVANA_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/Scripts/python.exe -m pytest tests/unit/test_personal_fact_store.py -q -n 4 --timeout=120` → **20 passed in 1.62s**.
- Tests probe REAL behaviour, including the 4 behaviour divergences the worker corrected by running (e.g. `contradict()` opens 0 contradiction edges / supersession; `reverse_stance()` returns the SAME mutated object, prior in `last_reversal`; `confirm()` with mismatched value leaves a parallel active fact + contradiction edge). These are genuine behaviour assertions, not tautologies. OK.
- **Budget gate**: `ci.yml` NOT touched by the round (`git diff 9fe14d0..HEAD -- .github/` is empty) → CI cap NOT raised. New test < 10s. OK.
- No existing test weakened/skipped/deleted. OK.

### CHECK 3 — Artifact removals: genuinely inert?

**RAVANA** (791b27f) — OK
- Removed (git rm --cached): `rlm_v2.py.bak`, `experiments/_identity_baseline_6912.npz`, `benchmark_results/locomo_full_progress_PRE_20260724_185358.jsonl`.
- None referenced by any code/test/doc/CI (grep excluding `.gitignore` returned 0 hits). `.gitignore` updated with matching entries (`*.bak`, the two paths).
- Inertness re-verified: `pytest tests/unit/ --co` collects 1822 tests with no import breakage after the `.bak` removal. LFS gate passes: `.github/scripts/check_lfs_assets.py` → glove resolved 143.5 MB. Nothing LFS/glove-adjacent removed. OK.

**IntentForge** (ae74a17) — OK
- Removed from tracking (working copies kept): 19 `.hermes-qa/*` scratch + `settings.yml.new` backup + `*.new/*.bak/*.tmp` added to `.gitignore`.
- `.hermes-qa` references found (23) are self-references inside the scratch tree on disk + one `tmp/INSTRUCTIONS.md` — NO production source/CI references (grep over `*.py/*.rs/*.yml/*.toml` returned only the scratch files themselves). `settings.yml.new` distinct from tracked `settings.yml`. OK.
- Inertness: live re-probe after change (`/health`=OK; `/search/fast` count=10) — but NOTE the top-level `source` field the live API returns was mis-documented (see CHECK 1 VIOLATION); the removal itself did not cause this.

### CHECK 4 — Protocol compliance

**RAVANA** — OK
- Commit-per-unit: 4 logical commits (ac25e75 timings / bb5340d tests / 04af788 docs / 791b27f de-clutter), not one mega-commit. OK.
- Every commit body carries a filled-in VERIFICATION block. OK.
- Pushed to BOTH remotes: `git rev-parse HEAD` == `origin/main` == `github/main` (all = 791b27f). OK.
- `kanban_complete` summaries consistent with the diff (timings, 20 tests, doc fixes, 3-file de-clutter all present). OK.

**IntentForge** — OK (with the one doc-accuracy exception above)
- Commit-per-unit: 3 logical commits (df42d21 API_REFERENCE / bf169f4 README / ae74a17 de-clutter). OK.
- VERIFICATION blocks present; unverified items marked. OK.
- Push: `origin/master` == HEAD (ae74a17). NOTE: IntentForge has ONLY the `origin` remote (no `github`); the round's claim of "pushed to both" is therefore loose wording — there is no second remote to push to. Not a false report against reality (nothing was supposed to go to a non-existent remote), but the "both remotes" phrasing from the parent handoff does not apply here. Logged as OK-with-note.
- `kanban_complete` summary consistent with the diff. The `/search/fast source` mis-statement is a doc-accuracy defect (CHECK 1 VIOLATION), not a protocol-report falsehood.

---

## Findings classification

| # | Repo | Finding | Class |
|---|------|---------|-------|
| 1 | intentforge | `API_REFERENCE.md:230` falsely claims `/search/fast` has no top-level `source` field; live API + worker's own transcript prove it does | **VIOLATION** |
| 2 | intentforge | Round handoff says "pushed to both remotes" but IntentForge has only `origin` — loose wording, no second remote exists | BORDERLINE (reality-OK, wording-imprecise) |

Everything else in both rounds: **OK**.

---

## FIX cards created

- **t_fdab0124** — FIX intentforge: `/search/fast` doc falsely claims no top-level `source` field.
  Body instructs the worker to re-verify the live API + transcript block 13, correct the doc text and JSON example to include top-level `source`, re-verify, commit with VERIFICATION block, push `origin` only, and ping both channels.

No second fix card spawned for the BORDERLINE (#2): it is wording-imprecision with no incorrect on-disk state; the working tree is correct. Logging it here is sufficient.

---

## Overall verdict

**Both rounds were substantially honest.** The documented timings, test additions, doc corrections, and de-clutter removals are all backed by real executed output and verifiable against source. The headline failure mode of this board — a fabricated API example — did NOT occur: the IntentForge transcript is a genuine captured artifact and every example I spot-checked traces to a real recorded response.

Exactly **one genuine VIOLATION**: a single false negative in `API_REFERENCE.md` (asserting the absence of a field that the live API and the worker's own transcript both show present). It is a documentation-accuracy defect, not a fabricated example, and is isolated to one sentence + its JSON example. A FIX card (t_fdab0124) is queued; everything else passes independent re-verification.

No false findings were manufactured. A clean verdict on the vast majority of the work is the honest result.
