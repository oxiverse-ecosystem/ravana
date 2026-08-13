#!/usr/bin/env bash
# Faithful reproduction of ci-critical + integration + misc CI jobs (sequential).
set -u
cd /c/Users/Likhith/Documents/Projects/ravana
export RAVANA_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
PY=.venv-real/Scripts/python.exe
mkdir -p data output checkpoints
OUT=/c/Users/Likhith/AppData/Local/hermes/kanban/boards/oxiverse-qa/workspaces/t_8ceb604a/audit_results
mkdir -p "$OUT"

echo "===== CI-CRITICAL =====" | tee -a "$OUT/ci_critical.log"
$PY -m pytest tests/ci/ -q --tb=short -x --ci -k "not soak" -n auto --dist worksteal --timeout=120 >> "$OUT/ci_critical.log" 2>&1
echo "CI-CRITICAL EXIT=$?" | tee -a "$OUT/ci_critical.log"

echo "===== INTEGRATION =====" | tee -a "$OUT/integration.log"
$PY -m pytest tests/integration/ -q --tb=short -n auto --dist worksteal --timeout=240 >> "$OUT/integration.log" 2>&1
echo "INTEGRATION EXIT=$?" | tee -a "$OUT/integration.log"

echo "===== MISC (root-level, ignoring unit/integration/ci) =====" | tee -a "$OUT/misc.log"
$PY -m pytest tests/ -q --tb=short --ignore=tests/unit --ignore=tests/integration --ignore=tests/ci -n auto --dist worksteal --timeout=120 >> "$OUT/misc.log" 2>&1
echo "MISC EXIT=$?" | tee -a "$OUT/misc.log"

echo "ALL OTHER SUITES DONE" | tee -a "$OUT/other.log"
