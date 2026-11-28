#!/usr/bin/env bash
# TRA-K taught-focus REPLICATION at 20/pool.  Fresh isolated stores and output;
# gates/thresholds/scoring are unchanged from PR #573.  3600s cap per arm.
set -u
WT=C:/Users/Likhith/Documents/projects/ravana_trak_arb_repl20
cd "$WT" || exit 9
PY=C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe
export PYTHONHASHSEED=0
export RAVANA_OFFLINE=1
export TRAK_CHANNEL_JSON="$WT/scratch/trak_channel_from_substrate.json"
BASE="$WT/scratch/m123_ab_data_repl20"
OUT="$WT/scratch/m123_ab_out_repl20"
mkdir -p "$BASE" "$OUT"
STAMP="$OUT/run_markers.txt"
echo "START $(date -u +%FT%TZ)" >> "$STAMP"
for SPEC in "OFF:off1" "OFF2:off2" "M1+M2+M3:m123" "ARBITRATION:arb" "SHAM:sham"; do
  A="${SPEC%%:*}"; T="${SPEC##*:}"
  echo "ARM_START $A $T $(date -u +%FT%TZ)" >> "$STAMP"
  timeout 3600 "$PY" scratch/m123_combined_runner.py --stage trak --arm "$A" \
    --tag "$T" --base "$BASE" --out "$OUT/trak_${T}.json" --max-per-pool 20 \
    --taught-focus >> "$OUT/run.log" 2>&1
  RC=$?
  echo "EXIT $A $RC $(date -u +%FT%TZ)" >> "$STAMP"
done
echo "ALLDONE $(date -u +%FT%TZ)" >> "$STAMP"
