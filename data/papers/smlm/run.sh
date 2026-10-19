#!/usr/bin/env bash
# Scores a discussion-thread dataset: run.sh SUBMISSION.csv OUT_DIR -> OUT_DIR/scores.json
# Pre-trains $MODEL (default allenai/longformer-base-4096) with selective MLM on the submission's threads, then
# fine-tunes it 5 times per CMV Modes task and scores each task. SUBMISSION=none skips the first stage (the paper's
# base Longformer). Uses every visible GPU (1, 2, 4 or 8).
set -euo pipefail
OUT=$(mkdir -p "$2" && realpath "$2") MODEL=${MODEL:-allenai/longformer-base-4096}
[[ $1 == none ]] || SUB=$(realpath "$1")
cd "$(dirname "$0")"
TRAIN=(uv run torchrun --standalone --nproc-per-node=gpu train.py)
if [[ $1 != none ]]; then
  uv run python validate.py "$SUB"
  "${TRAIN[@]}" "$MODEL" "$SUB" "$OUT/source"
  MODEL=$OUT/source
fi
"${TRAIN[@]}" "$MODEL" cmv_modes "$OUT/cmv_modes"
uv run python eval.py "$OUT"
