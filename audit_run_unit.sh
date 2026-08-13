#!/usr/bin/env bash
# Faithful reproduction of the `unit-tests` CI job (4 shards, sequential).
set -u
cd /c/Users/Likhith/Documents/Projects/ravana
export RAVANA_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
PY=.venv-real/Scripts/python.exe
mkdir -p data output checkpoints
OUT=/c/Users/Likhith/AppData/Local/hermes/kanban/boards/oxiverse-qa/workspaces/t_8ceb604a/audit_results
mkdir -p "$OUT"
for SHARD in 1 2 3 4; do
  echo "===== UNIT SHARD $SHARD/4 =====" | tee -a "$OUT/unit_shard${SHARD}.log"
  $PY -m pytest tests/unit/ -q --tb=short \
    --splits 4 --group $SHARD \
    --splitting-algorithm least_duration \
    -n auto --dist worksteal --timeout=180 \
    >> "$OUT/unit_shard${SHARD}.log" 2>&1
  echo "SHARD $SHARD EXIT=$?" | tee -a "$OUT/unit_shard${SHARD}.log"
done
echo "ALL UNIT SHARDS DONE" | tee -a "$OUT/unit.log"
