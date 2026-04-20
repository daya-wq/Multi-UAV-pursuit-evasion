#!/bin/bash
# Auto-monitor for v7 training
# Waits 3 hours, then analyzes metrics and decides next action
# Usage: nohup bash scripts/auto_monitor_v7.sh > analysis/auto_monitor_v7.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

ANALYSIS_SCRIPT="scripts/auto_analyze_and_iterate.py"
V7_LOG="analysis/rl_warmstart_v7.log"
REPORT_DIR="analysis/auto_monitor_reports"
WAIT_HOURS=3

mkdir -p "$REPORT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-monitor started. Waiting ${WAIT_HOURS}h for v7 training..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] V7 log: $V7_LOG"

# Wait for the specified time
sleep $((WAIT_HOURS * 3600))

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Wait complete. Running analysis..."

# Activate conda and run the analysis
source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim

python3 "$ANALYSIS_SCRIPT" \
  --log-file "$V7_LOG" \
  --report-dir "$REPORT_DIR" \
  --checkpoint-dir "checkpoints" \
  --scripts-dir "scripts" \
  --config-file "cfg/task/HideAndSeek.yaml" \
  --auto-restart

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-monitor cycle complete."
