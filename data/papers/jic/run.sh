#!/usr/bin/env bash
# Scores a JIC-format dialogue dataset: run.sh SUBMISSION.json OUT_DIR [--epochs K | --steps N] -> OUT_DIR/scores.json
# QLoRA-fine-tunes $MODEL (default mistralai/Mistral-7B-Instruct-v0.3) on the submission with the authors' code and
# recipe (src/Beyond-Discrete-Personas/src/training/mistral/finetune.py, every visible GPU: 1, 2, 4 or 8), then scores
# it 0-shot on the Big Five persona tasks (eval.py). SUBMISSION=none scores $MODEL as is.
# --epochs K or --steps N (optimizer steps, effective batch 4 x 128) replace the recipe's 3 epochs; early stopping kept.
set -euo pipefail
OUT=$(mkdir -p "$2" && realpath "$2") MODEL=${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}
[[ $1 == none ]] || SUB=$(realpath "$1")
case ${3:-} in --epochs) export EPOCHS=$4 ;; --steps) export MAX_STEPS=$4 ;; esac
cd "$(dirname "$0")"
if [[ $1 != none ]]; then
  uv run python validate.py "$SUB"
  uv run torchrun --standalone --nproc-per-node=gpu src/Beyond-Discrete-Personas/src/training/mistral/finetune.py \
    --model_name "$MODEL" --input_file "$SUB" --cache_dir "${HF_HOME:-$HOME/.cache/huggingface}/hub" \
    --hf_token "${HF_TOKEN:-}" --output_dir "$OUT/source"
fi
uv run python eval.py "$MODEL" "$OUT" ${SUB:+"$OUT/source"}
