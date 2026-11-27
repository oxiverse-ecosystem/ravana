#!/usr/bin/env bash
# Pre-registered TRA-K far-repair A/B run loop (one process per arm, 1800s cap).
set -u
cd C:/Users/Likhith/Documents/projects/ravana_trak_far_ab || exit 9
export PYTHONHASHSEED=0
export RAVANA_OFFLINE=1
export TRAK_CHANNEL_JSON=C:/Users/Likhith/Documents/projects/ravana/scratch/trak_channel_from_substrate.json
PY=C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe
BASE="$PWD/scratch/m123_ab_data"
OUT="$PWD/scratch/m123_ab_out"
LOG="$PWD/scratch/m123_ab_logs"
mkdir -p "$BASE" "$OUT" "$LOG"
echo "AB_START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
for SPEC in "OFF:off1" "OFF2:off2" "M1+M2+M3:m123" "REPAIR:repair" "REPAIR_SELFLOOP:repair_self" "SHAM:sham"; do
  A="${SPEC%%:*}"; T="${SPEC##*:}"
  echo "ARM_START $A $T $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
  timeout 1800 "$PY" scratch/m123_combined_runner.py --stage trak --arm "$A" \
    --tag "$T" --base "$BASE" --out "$OUT/trak_${T}.json" --max-per-pool 2 \
    > "$LOG/trak_${T}.log" 2>&1
  RC=$?
  echo "ARM_END $A $T rc=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
done
echo "AB_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG/driver.log"
