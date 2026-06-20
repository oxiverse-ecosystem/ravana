# RAVANA PyPI Publishing Guide

## Package Strategy: Three Independent Packages (Recommended)

Because PyPI package names must use underscores/hyphens and the directory `ravana-v2` has a hyphen, the cleanest approach is **three separate packages**:

| Package | PyPI Name | Directory | Version | Purpose |
|---------|-----------|-----------|---------|---------|
| **ML Framework** | `ravana-ml` | `ravana_ml/` | 0.3.2 | Tensors, modules, ConceptGraph, RLMv1/RLMv2 |
| **GRACE Core** | `ravana-grace` | `ravana-v2/` | 0.2.2 | 27-phase cognitive architecture |
| **Modular Chat** | `ravana-chat` | `ravana/` | 0.3.2 | Decoder-first chat, web learning, CLI |

(Optional) Unified: `ravana-cognitive` (all three together)

---

## PyPI Names Status
| Name | Status |
|------|--------|
| `ravana` | ❌ Taken (MIDI package) |
| `ravana-ml` | ✅ Published v0.3.2 |
| `ravana-grace` | ✅ Published v0.2.2 |
| `ravana-chat` | ✅ Published v0.1.1 |
| `ravana-cognitive` | ✅ Available |

---

## Build & Publish Commands

### Option A: Individual Packages (Recommended)
```bash
# Install tools
python -m pip install build twine --quiet

# 1. Build ravana-ml
cd ravana_ml
python -m build --wheel --sdist
cd ..

# 2. Build ravana-grace
cd ravana-v2
python -m build --wheel --sdist
cd ..

# 3. Build ravana-chat
cd ravana
python -m build --wheel --sdist
cd ..

# 4. Publish to TestPyPI first
export TESTPYPI_TOKEN='your-testpypi-token'
python -m twine upload --repository-url https://test.pypi.org/legacy/ -u "__token__" -p "$TESTPYPI_TOKEN" dist/*

# 5. Verify on https://test.pypi.org, then publish to PyPI
export PYPI_TOKEN='your-pypi-token'
python -m twine upload --repository-url https://upload.pypi.org/legacy/ -u "__token__" -p "$PYPI_TOKEN" dist/*
```

### Option B: Unified Package
```bash
# From repo root
python -m build --wheel --sdist
python -m twine upload --repository-url https://test.pypi.org/legacy/ -u "__token__" -p "$TESTPYPI_TOKEN" dist/*
```

---

## Secure Token Handling
**Never commit tokens to git.** Use environment variables or `.pypirc`:

```bash
# ~/.pypirc
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...

[testpypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...
```

Then:
```bash
python -m twine upload dist/*               # uses .pypirc
python -m twine upload --repository testpypi dist/*
```

---

## Verification After Publish
```bash
# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ravana-ml

# Test import
python -c "import ravana_ml; print(ravana_ml.__version__)"
python -c "from ravana_ml.nn import RLMv2; print('RLMv2 OK')"

# Test install from PyPI (after prod publish)
pip install ravana-ml
pip install ravana-grace
pip install ravana-chat
```

---

## Files Created
| File | Purpose |
|------|---------|
| `pyproject.toml` (root) | Unified package config |
| `ravana_ml/pyproject.toml` | ML framework config |
| `ravana-v2/pyproject.toml` | GRACE core config |
| `ravana/pyproject.toml` | Modular chat config |
| `ravana-v2/__init__.py` | Root init for ravana_v2 package |
| `build_and_publish.sh` | Convenience script |

---

## Quick Test Before Publishing
```bash
# Test local editable installs
pip install -e ravana_ml
pip install -e ravana-v2
pip install -e ravana

# Run tests
python -m pytest tests/ ravana-v2/tests/ -v --tb=short

# Test imports
python -c "
from ravana_ml.nn import RLMv2
from ravana_ml.graph import ConceptGraph
from ravana_ml.tensor import StateTensor
print('ravana_ml OK')

from ravana_v2 import cognitive
print('ravana_v2 OK')

from ravana.chat.interface import ChatInterface
print('ravana.chat OK')
"
```

---

## Versioning
Update version in each `pyproject.toml` before release:
- `ravana_ml/pyproject.toml` → `version = "0.1.0"`
- `ravana-v2/pyproject.toml` → `version = "0.1.0"`
- `ravana/pyproject.toml` → `version = "0.1.0"`

Use semantic versioning: `MAJOR.MINOR.PATCH`
- MAJOR: Breaking API changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes