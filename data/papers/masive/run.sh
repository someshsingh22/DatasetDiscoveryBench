#!/usr/bin/env bash
# Scores a MASIVE-format dataset: run.sh SUBMISSION.csv OUT_DIR -> OUT_DIR/scores.json
# Fine-tunes $MODEL (default google/mt5-large) on the submission, then on each benchmark, then scores each benchmark.
# SUBMISSION=none skips the first stage (the paper's mT5 baseline). Uses every visible GPU (1, 2, 4 or 8).
set -euo pipefail
OUT=$(mkdir -p "$2" && realpath "$2") MODEL=${MODEL:-google/mt5-large}
[[ $1 == none ]] || SUB=$(realpath "$1")
cd "$(dirname "$0")"
TRAIN=(uv run torchrun --standalone --nproc-per-node=gpu train.py)
if [[ $1 != none ]]; then
  uv run python validate.py "$SUB"
  "${TRAIN[@]}" "$MODEL" "$SUB" "$OUT/source"
  MODEL=$OUT/source
fi
for task in emoevent_en goemo goemo_full; do "${TRAIN[@]}" "$MODEL" "$task" "$OUT/$task"; done
uv run python eval.py "$OUT"
