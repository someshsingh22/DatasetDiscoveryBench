#!/usr/bin/env bash
# Scores a Reddit TL;DR-format dataset: run.sh SUBMISSION.csv OUT_DIR [--epochs K | --steps N] -> OUT_DIR/scores.json
# Fine-tunes $MODEL (default EleutherAI/pythia-1b-deduped) on the submission with the re-implementation's SFT recipe
# (train.py preprocesses, src/summarize_from_feedback_details/sft.py trains on every visible GPU: 1, 2, 4 or 8,
# effective batch 128; default 1 epoch, or K epochs, or N optimizer steps on 128N shuffled rows), then has it summarize
# the CNN/DM test articles, without further training, and scores the summaries with ROUGE (eval.py).
# SUBMISSION=none scores $MODEL as is.
set -euo pipefail
OUT=$(mkdir -p "$2" && realpath "$2") MODEL=${MODEL:-EleutherAI/pythia-1b-deduped} EPOCHS=1 EPISODES=
[[ $1 == none ]] || SUB=$(realpath "$1")
case ${3:-} in
  --epochs) EPOCHS=$4 ;;
  --steps) EPISODES=$((128 * $4)) ;;
  "") ;;
  *) echo "usage: run.sh SUBMISSION.csv OUT_DIR [--epochs K | --steps N]" >&2 && exit 2 ;;
esac
cd "$(dirname "$0")"
if [[ $1 != none ]]; then
  uv run python validate.py "$SUB"
  uv run python train.py "$SUB" "$MODEL" "$OUT/data"
  read -r N PORT < <(uv run python -c \
    'import socket, torch; s = socket.socket(); s.bind(("", 0)); print(torch.cuda.device_count(), s.getsockname()[1])')
  uv run accelerate launch --config_file src/summarize_from_feedback_details/deepspeed.yaml --num_processes "$N" \
    --main_process_port "$PORT" src/summarize_from_feedback_details/sft.py --base_model="$MODEL" \
    --query_dataset="$OUT/data" --output_dir="$OUT/model" --gradient_accumulation_steps=$((128 / N)) \
    --num_train_epochs="$EPOCHS" ${EPISODES:+--total_episodes="$EPISODES"} --deepspeed
  MODEL=$OUT/model
fi
uv run python eval.py "$MODEL" "$OUT"
