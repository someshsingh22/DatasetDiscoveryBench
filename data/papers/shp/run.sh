#!/usr/bin/env bash
# Scores a preference-pair dataset: run.sh SUBMISSION.csv OUT_DIR [--epochs K | --steps N] -> OUT_DIR/scores.json
# Fine-tunes $MODEL (default google/flan-t5-large) on the submission in the SteamSHP format, and v_info.py's null model on
# empty inputs (every visible GPU: 1, 2, 4 or 8); eval.py scores both on HH-RLHF and on the submission's held-out posts.
# SUBMISSION=none scores $MODEL as is. --epochs/--steps (optimizer steps at batch 16) replace both trainings' 1 epoch.
set -euo pipefail
OUT=$(mkdir -p "$2" && realpath "$2") MODEL=${MODEL:-google/flan-t5-large} SUB=none NULL=none
[[ $1 == none ]] || SUB=$(realpath "$1")
cd "$(dirname "$0")"
if [[ $1 != none ]]; then
  uv run python validate.py "$SUB"
  uv run torchrun --standalone --nproc-per-node=gpu train.py "$MODEL" "$SUB" "$OUT/model" model "${@:3}"
  uv run torchrun --standalone --nproc-per-node=gpu train.py "$MODEL" "$SUB" "$OUT/null" null "${@:3}"
  MODEL=$OUT/model NULL=$OUT/null
fi
uv run python eval.py "$MODEL" "$NULL" "$SUB" "$OUT"
