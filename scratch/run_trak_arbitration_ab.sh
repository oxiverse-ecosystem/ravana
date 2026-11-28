#!/usr/bin/env bash
set -u
cd C:/Users/Likhith/Documents/projects/ravana_trak_live_arb || exit 9
PY=C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe
export PYTHONHASHSEED=0
export RAVANA_OFFLINE=1
export TRAK_CHANNEL_JSON="C:/Users/Likhith/Documents/projects/ravana_trak_live_arb/scratch/trak_channel_from_substrate.json"
BASE="C:/Users/Likhith/Documents/projects/ravana_trak_live_arb/scratch/m123_ab_data"
OUT="C:/Users/Likhith/Documents/projects/ravana_trak_live_arb/scratch/m123_ab_out"
mkdir -p "$OUT"
echo "START $(date -u +%FT%TZ)" >> "$OUT/ab.log"
for SPEC in "OFF:off1" "M1+M2+M3:m123" "ARBITRATION:arb" "SHAM:sham" "OFF2:off2"; do
  A="${SPEC%%:*}"; T="${SPEC##*:}"
  echo "ARM_START $A $T $(date -u +%FT%TZ)" >> "$OUT/ab.log"
  timeout 1800 "$PY" scratch/m123_combined_runner.py --stage trak --arm "$A" \
    --tag "$T" --base "$BASE" --out "$OUT/trak_${T}.json" --max-per-pool 2 \
    >> "$OUT/ab.log" 2>&1
  echo "EXIT $A $? $(date -u +%FT%TZ)" >> "$OUT/ab.log"
done
echo "ALLDONE $(date -u +%FT%TZ)" >> "$OUT/ab.log"
