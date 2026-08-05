# Testing RAVANA

How the test suite is organized, how CI runs it, and the rules that keep the
local and CI runs green. Every command and number below was exercised during
the 2026-08 round; the measured wall-clock table lives in
`docs/_generated/suite-timings.md`.

## Five CI suites

`.github/workflows/ci.yml` defines five independent jobs (plus an aggregate
gate). They run on `ubuntu-latest` (4 vCPU) except the soak, which runs on
`windows-latest`. The CI uses `-n auto` with `--dist worksteal`; the
**`unit-tests`** job is additionally sharded four ways with
`--splits 4 --group N` so each shard is a fraction of the single-machine time.

| suite | invocation (key flags) | CI `timeout-minutes` | per-test `--timeout` |
|---|---|---|---|
| ci-critical | `tests/ci/ -k "not soak" --ci -n auto --timeout=120` | 10 | 120 |
| unit (sharded ×4) | `tests/unit/ --splits 4 --group N -n auto --timeout=180` | 12 | 180 |
| integration | `tests/integration/ -n auto --timeout=240` | 12 | 240 |
| misc | `tests/ -n auto --ignore=unit --ignore=integration --ignore=ci --timeout=120` | 10 | 120 |
| av-soak (Windows) | `tests/ci/test_av_soak.py -p no:cacheprovider --timeout=180 --timeout-method=thread` with `RAVANA_AV_SOAK_ROUNDS=6` | 20 | 180 |

## The `-n 4` rule (local runs)

CI runs in parallel, but your **local** machine is not the CI runner. The
repeated failure mode is `-n auto` on a dev box: `pytest-xdist` spins up one
worker *per CPU*, and every RAVANA worker builds its own `CognitiveChatEngine`
(~27 s init, plus GloVe memory). On a 16-core box that is 16 simultaneous
engine builds that OOM into a wall of bogus
`worker 'gwN' crashed ... Traceback (most recent call last):` mass failures.

> **Rule: on a local box, use `-n 4` (never `-n auto`).** Four workers is the
> safe ceiling that matches the CI runner's vCPU count without overwhelming
> memory. Do NOT co-run `tests/ci/test_av_soak.py` with the unit suite — the
> soak is the thread-race stress test and must run alone.

## The `RAVANA_OFFLINE=1` rule

`tests/conftest.py` pins `RAVANA_OFFLINE=1` via `os.environ.setdefault`, so a
local run already defaults to offline. Export it explicitly anyway so you get
the same behaviour if you invoke a script (not a test) directly:

```bash
export RAVANA_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

Why this matters: engine boot falls back to auto-downloading `glove.6B.zip`
(~822 MB) whenever the projected-vector cache (`data/ravana_glove_cache.npz`,
git-lfs) is missing. In CI a cache miss turns into a multi-minute stall or a
hard job timeout; offline degrades instantly to "no GloVe vectors" instead.

**LFS gate:** every CI job runs `.github/scripts/check_lfs_assets.py` first. It
fails fast (exit 1) if the glove cache is still a git-lfs *pointer file*
instead of the real payload. Run it locally after a fresh clone to catch a
missed `git lfs pull`:

```bash
python .github/scripts/check_lfs_assets.py
```

## The slow/av-soak rules (from the brief, verified)

- `@pytest.mark.slow` **alone excludes nothing.** There is no auto-skip hook on
  the marker anywhere in the repo. Slow tests must add **explicit skip logic**
  (e.g. `@pytest.mark.skipif(not os.environ.get("RAVANA_RUN_SLOW"), ...)`) or
  they run in CI and blow the job timeout. See `tests/unit/test_generation.py`
  (`test_instruction_compression`) for a slow test that currently has no guard
  — it runs unconditionally.
- Nothing new goes into the av-soak job. The soak is the thread-race stress
  test (6 engine builds + hundreds of turns on Windows); keep it scoped.
- A test that is genuinely valuable but slow gets `@pytest.mark.slow` **and** an
  explicit skip.

## Markers

Registered in `pyproject.toml` (`[tool.pytest.ini_options] markers`):

- `ci` — runs in the fast ci-critical job. Select with `--ci` (defined in
  `tests/ci/conftest.py` via `pytest_addoption`).
- `integration` — cross-module integration tests.
- `slow` — long-running tests (see rule above: does NOT auto-skip).

## Local reproduction (recommended)

```bash
# 1) LFS gate
python .github/scripts/check_lfs_assets.py

# 2) Fast critical slice (excludes the soak)
python -m pytest tests/ci/ -k "not soak" -q -n 4 --timeout=120

# 3) Unit (broadest suite)
python -m pytest tests/unit/ -q -n 4 --timeout=180

# 4) Integration (slower, builds real engines)
python -m pytest tests/integration/ -q -n 4 --timeout=240

# 5) Misc root-level
python -m pytest tests/ -q -n 4 --ignore=tests/unit --ignore=tests/integration --ignore=tests/ci --timeout=120

# 6) AV soak (Windows-style stress; local single box is fine, run alone)
RAVANA_AV_SOAK_ROUNDS=6 python -m pytest tests/ci/test_av_soak.py -q --timeout=180 --timeout-method=thread
```

## Determinism rules for new tests

- No network, no wall-clock races, no ordering dependence.
- Run your new tests twice and under `-n 4` to prove stability.
- New tests must fit inside the existing CI job caps: after adding them the
  suite's wall time stays under **70% of its `timeout-minutes`** and no single
  test exceeds **10 s**. If you cannot fit, shrink or delete the test — you may
  NOT raise `timeout-minutes`, raise `--timeout`, or edit `ci.yml` to make room.
- Record each new test's measured seconds in the commit body.

## Coverage

```bash
# core cognition paths, error handling, boundaries
python -m pytest tests/unit/ --cov=ravana --cov-report=term-missing
```

If `pytest-cov` is missing: `uv pip install --python .venv/Scripts/python.exe pytest-cov`.
