# Development

## Repository layout

```
ravana/            package source  →  ravana/src/ravana/        (chat engine)
ravana_ml/         package source  →  ravana_ml/src/ravana_ml/  (ML substrate)
ravana-v2/         package source  →  ravana-v2/src/ravana_grace/ (GRACE governor)
scripts/           runnable entry points (chat, train, learn, benchmarks)
experiments/       research harnesses imported by benchmarks
tests/             pytest suite (ci / unit / integration / eval)
docs/              this documentation
data/              runtime artifacts (gitignored): corpus, weights, caches
checkpoints/       training snapshots (gitignored)
output/            run output (gitignored)
benchmark_results/ benchmark output (gitignored)
```

## Path shims (important)

There is no installed package at import time in normal dev use. Every entry
point and `tests/conftest.py` prepends the three `src` dirs to `sys.path`:

```python
for p in ["ravana_ml/src", "ravana/src", "ravana-v2/src", "."]:
    sys.path.insert(0, p)
```

`scripts/ravana_chat.py` adds them in **reverse priority** so the modular
`ravana` package shadows any stale root `ravana/` dir. Always run scripts from
the repo root.

## Environment

- Python 3.10+ (verified on 3.14).
- Core deps: `numpy`, `scipy`. Everything else is optional (see
  `requirements.txt` / `pyproject.toml` `[project.optional-dependencies]`).
- Install editable (also what CI does):

  ```bash
  pip install -e .[full,dev]
  ```

## Running the tests

```bash
# Fast CI-critical slice (used by .github/workflows/ci.yml)
python -m pytest tests/ci/ -v --ci

# Module-level unit tests
python -m pytest tests/unit/ -q

# Cross-module integration tests
python -m pytest tests/integration/ -q

# Everything
python -m pytest tests/ --tb=short
```

The `ci` mark is registered in `pyproject.toml`. `tests/ci/test_av_soak.py`
contains slow soak rounds — expect the critical job to take ~15 minutes.

## Running the system

```bash
python scripts/ravana_chat.py            # interactive chatbot
python scripts/train.py --mode test      # quick training diagnostic
python scripts/ravana_learn.py           # autonomous background learning
```

## Code conventions

- **Three packages, one system.** Changes to chat behavior usually touch
  `ravana/src/ravana/chat/`; cognitive regulation lives in
  `ravana-v2/src/ravana_grace/core/`; the ML substrate in `ravana_ml`.
- **Fail-closed grounding.** New retrieval/generation paths must abstain when
  coherence is below the distributed floor — never emit ungrounded text.
- **No hardcoded thresholds where a distribution exists.** Prefer
  data-derived gates (see `ravana/chat/brain_regions.py`).
- **Brain-repair prepasses run before `process_turn` routing.** Keep them
  ordered and tightly gated (e.g. the empathy cause-fallback only fires on
  state-disclosure syntax, not on recall/question/request/humor frames).
- **Tests before merge.** Add or update a `tests/unit` or `tests/integration`
  case for behavior changes; keep `tests/ci` green.

## Common pitfalls (from prior fixes)

- **Connectivity/anchor gates** must snapshot stable nodes at init
  (`self._stable_node_ids = set(graph.nodes)` right after bootstrap). Same-turn
  web-learning adds edges to freshly-looked-up concepts and would otherwise
  defeat a live edge check.
- **Humor coherence** ORs the learned salad classifier with the rule-based
  `_is_word_salad` (in `ravana/chat/constants.py`, *not* in
  `salad_classifier.py`) and fails closed.
- **LingGen promotion** is gated on free-form generation quality, not
  verbatim CE — a tiny GRU cannot beat the KB from scratch, and that is not the
  success criterion.
