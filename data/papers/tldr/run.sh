#!/usr/bin/env bash
# Scores a Reddit TL;DR-format dataset: run.sh SUBMISSION.csv OUT_DIR -> OUT_DIR/scores.json
# Fine-tunes $MODEL (default EleutherAI/pythia-1b-deduped) on the submission with the re-implementation's SFT recipe
# (train.py; every visible GPU: 1, 2, 4 or 8, effective batch 128), then has it summarize the CNN/DM test articles,
# without further training, and scores the summaries with ROUGE (eval.py). SUBMISSION=none scores $MODEL as is.
set -euo pipefail
OUT=$(mkdir -p "$2" && realpath "$2") MODEL=${MODEL:-EleutherAI/pythia-1b-deduped}
[[ $1 == none ]] || SUB=$(realpath "$1")
cd "$(dirname "$0")"
if [[ $1 != none ]]; then
  uv run python validate.py "$SUB"
  read -r N PORT < <(uv run python -c \
    'import socket, torch; s = socket.socket(); s.bind(("", 0)); print(torch.cuda.device_count(), s.getsockname()[1])')
  uv run accelerate launch --config_file deepspeed.yaml --num_processes "$N" --main_process_port "$PORT" train.py \
    --base_model="$MODEL" --query_dataset="$SUB" --output_dir="$OUT/model" --gradient_accumulation_steps=$((128 / N)) \
    --deepspeed
  MODEL=$OUT/model
fi
uv run python eval.py "$MODEL" "$OUT"
