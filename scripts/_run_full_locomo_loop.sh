#!/usr/bin/env bash
# Crash-resilient outer loop for the full LoCoMo eval.
# The driver self-resumes from benchmark_results/locomo_full_progress.jsonl;
# this loop restarts the driver until ALL cases are recorded (exit 0).
set -u
# Windows-native absolute path (forward slashes) so python.exe resolves it.
PROJ="C:/Users/Likhith/Documents/projects/ravana"
PROGRESS="$PROJ/benchmark_results/locomo_full_progress.jsonl"
LOG="$PROJ/benchmark_results/locomo_full_freshengine.log"
export PYTHONUNBUFFERED=1
export RAVANA_SILENT=1
PY="$PROJ/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY=python

: > "$LOG"
ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT+1))
  echo "===== ATTEMPT $ATTEMPT ($(date)) =====" >> "$LOG"
  "$PY" "$PROJ/scripts/_fulleval_locomo.py" >> "$LOG" 2>&1
  RC=$?
  echo "attempt $ATTEMPT exit=$RC" >> "$LOG"
  if [ "$RC" -eq 0 ]; then
    echo "LOOP_DONE rc=$RC" >> "$LOG"
    break
  fi
  if [ ! -s "$PROGRESS" ] && [ "$ATTEMPT" -ge 5 ]; then
    echo "LOOP_BAIL: no progress after $ATTEMPT attempts" >> "$LOG"
    break
  fi
  sleep 2
done
echo "DONE_AT=$(date)" >> "$LOG"
