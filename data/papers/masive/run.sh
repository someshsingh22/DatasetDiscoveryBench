#!/usr/bin/env bash
# Scores a MASIVE-format dataset: run.sh SUBMISSION.csv OUT_DIR -> OUT_DIR/scores.json
# Fine-tunes $MODEL (default google/mt5-large) on the submission (every visible GPU: 1, 2, 4 or 8), then scores it
# zero-shot on the emotion benchmarks and on the MASIVE test set (eval.py). SUBMISSION=none scores $MODEL as is.
set -euo pipefail
OUT=$(mkdir -p "$2" && realpath "$2") MODEL=${MODEL:-google/mt5-large}
[[ $1 == none ]] || SUB=$(realpath "$1")
cd "$(dirname "$0")"
if [[ $1 != none ]]; then
  uv run python validate.py "$SUB"
  uv run torchrun --standalone --nproc-per-node=gpu train.py "$MODEL" "$SUB" "$OUT/source"
  MODEL=$OUT/source
fi
uv run python eval.py "$MODEL" "$OUT"
