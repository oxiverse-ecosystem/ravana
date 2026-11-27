#!/usr/bin/env bash
# Full WFA with both flags on a GloVe-free store, with guaranteed restore.
set -u
cd "C:/Users/Likhith/Documents/projects/ravana_wfa_glove_memo" || exit 9

restore() {
  if [ -e weights/ravana_glove_cache.npz.hidden ] && [ ! -e weights/ravana_glove_cache.npz ]; then
    mv weights/ravana_glove_cache.npz.hidden weights/ravana_glove_cache.npz; fi
  if [ -e data/ravana_glove_cache.npz.hidden ] && [ ! -e data/ravana_glove_cache.npz ]; then
    mv data/ravana_glove_cache.npz.hidden data/ravana_glove_cache.npz; fi
  if [ -e data/glove.hidden ] && [ ! -e data/glove ]; then
    mv data/glove.hidden data/glove; fi
}
trap restore EXIT
restore

if [ -e weights/ravana_glove_cache.npz ]; then
  mv weights/ravana_glove_cache.npz weights/ravana_glove_cache.npz.hidden || exit 8; fi
if [ -e data/ravana_glove_cache.npz ]; then
  mv data/ravana_glove_cache.npz data/ravana_glove_cache.npz.hidden || exit 8; fi
if [ -e data/glove ]; then
  mv data/glove data/glove.hidden || exit 8; fi
rm -f scratch/learn5_wfa_summary.json scratch/learn5_wfa_probes.jsonl

LOG=scratch/wfa_full_both_flags_noglove.log
printf "RUN_START %s\n" "$(date)" > "$LOG"
PYTHONPATH="C:/Users/Likhith/Documents/projects/ravana_wfa_glove_memo/ravana_ml/src;C:/Users/Likhith/Documents/projects/ravana_wfa_glove_memo/ravana/src;C:/Users/Likhith/Documents/projects/ravana_wfa_glove_memo/ravana-v2/src;C:/Users/Likhith/Documents/projects/ravana_wfa_glove_memo/." \
RAVANA_OFFLINE=1 PYTHONHASHSEED=0 RAVANA_PRIOR_ACT_TRACE=1 RAVANA_GLOVE_BOOT_MEMO=1 \
timeout 3600 "C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe" \
  -c "import sys; sys.path.insert(0,'scripts'); import measure_learn5_wfa as w; sys.exit(w.main([]))" \
  >> "$LOG" 2>&1
echo "WFA_NOGLOVE_EXIT=$? $(date)" >> "$LOG"
