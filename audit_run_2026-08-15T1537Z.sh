#!/usr/bin/env bash
# Independent CI audit runner for round 2026-08-15T1537Z
# Reproduces .github/workflows/ci.yml suites locally (Windows/git-bash).
# Runs sequentially (heavy suites in parallel OOM per memory note).
set -u
cd "C:/Users/Likhith/Documents/Projects/ravana" || exit 1

export RAVANA_OFFLINE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export RAVANA_AV_SOAK_ROUNDS=6

PY=.venv-real/Scripts/python.exe
LOGDIR="audit_logs_2026-08-15T1537Z"
mkdir -p "$LOGDIR" data output checkpoints

summary() { echo "===== $1 =====" >> "$LOGDIR/SUMMARY.txt"; }

# 0. LFS asset check
echo "### check_lfs_assets.py" | tee -a "$LOGDIR/SUMMARY.txt"
$PY .github/scripts/check_lfs_assets.py >> "$LOGDIR/lfs.txt" 2>&1
echo "LFS_EXIT=$?" >> "$LOGDIR/SUMMARY.txt"

# 1. ci-critical (drop -x to enumerate ALL failures for the audit)
echo "### ci-critical" | tee -a "$LOGDIR/SUMMARY.txt"
$PY -m pytest tests/ci/ -q --tb=line -k "not soak" -n auto --dist worksteal --timeout=120 > "$LOGDIR/ci-critical.txt" 2>&1
echo "CI_CRITICAL_EXIT=$?" >> "$LOGDIR/SUMMARY.txt"

# 2. unit shards (4, sequential)
for SHARD in 1 2 3 4; do
  echo "### unit shard $SHARD" | tee -a "$LOGDIR/SUMMARY.txt"
  $PY -m pytest tests/unit/ -q --tb=line --splits 4 --group $SHARD --splitting-algorithm least_duration -n auto --dist worksteal --timeout=180 > "$LOGDIR/unit-shard-$SHARD.txt" 2>&1
  echo "UNIT_SHARD_${SHARD}_EXIT=$?" >> "$LOGDIR/SUMMARY.txt"
done

# 3. integration
echo "### integration" | tee -a "$LOGDIR/SUMMARY.txt"
$PY -m pytest tests/integration/ -q --tb=line -n auto --dist worksteal --timeout=240 > "$LOGDIR/integration.txt" 2>&1
echo "INTEGRATION_EXIT=$?" >> "$LOGDIR/SUMMARY.txt"

# 4. misc (root-level, ignore unit/integration/ci)
echo "### misc" | tee -a "$LOGDIR/SUMMARY.txt"
$PY -m pytest tests/ -q --tb=line --ignore=tests/unit --ignore=tests/integration --ignore=tests/ci -n auto --dist worksteal --timeout=120 > "$LOGDIR/misc.txt" 2>&1
echo "MISC_EXIT=$?" >> "$LOGDIR/SUMMARY.txt"

# 5. av-soak (Windows-only, thread-race stress)
echo "### av-soak" | tee -a "$LOGDIR/SUMMARY.txt"
$PY -m pytest tests/ci/test_av_soak.py -q --tb=line -p no:cacheprovider --timeout=180 --timeout-method=thread > "$LOGDIR/av-soak.txt" 2>&1
echo "AV_SOAK_EXIT=$?" >> "$LOGDIR/SUMMARY.txt"

echo "ALL DONE" | tee -a "$LOGDIR/SUMMARY.txt"
