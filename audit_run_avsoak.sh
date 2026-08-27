#!/usr/bin/env bash
# Faithful reproduction of the av-soak CI job (Windows-only thread-race soak).
set -u
cd /c/Users/Likhith/Documents/Projects/ravana
export RAVANA_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export RAVANA_AV_SOAK_ROUNDS=6
PY=.venv-real/Scripts/python.exe
mkdir -p data output checkpoints
OUT=/c/Users/Likhith/AppData/Local/hermes/kanban/boards/oxiverse-qa/workspaces/t_8ceb604a/audit_results
mkdir -p "$OUT"
echo "===== AV-SOAK (RAVANA_AV_SOAK_ROUNDS=6) =====" | tee -a "$OUT/av_soak.log"
$PY -m pytest tests/ci/test_av_soak.py -q --tb=short -p no:cacheprovider --timeout=180 --timeout-method=thread --durations=10 >> "$OUT/av_soak.log" 2>&1
echo "AV-SOAK EXIT=$?" | tee -a "$OUT/av_soak.log"
