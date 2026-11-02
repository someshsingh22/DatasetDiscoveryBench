#!/usr/bin/env bash
# Scores a discussion-thread dataset: run.sh SUBMISSION.csv OUT_DIR -> OUT_DIR/scores.json
# Runs the authors' code (src/arg_mining): selective-MLM pre-training of Longformer on the submission's threads (3 GPUs),
# then 5 seeded fine-tuning runs per task, one per GPU at a time: argument component identification on Dr Inventor and
# CMV Modes, relation types on CMV Modes. SUBMISSION=none fine-tunes the base Longformer (the paper's baseline).
# Uses the GPUs in CUDA_VISIBLE_DEVICES, default all (pre-training needs 3).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}
ENV=$(dirname "$(realpath "$0")")
export OUT=$(mkdir -p "$2/logs" && realpath "$2") TOK=allenai/longformer-base-4096 MODEL=allenai/longformer-base-4096
export PYTHONPATH=$ENV/src PYTHONHASHSEED=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1  # same thread order on every rank; TF32
SRC=$ENV/src/arg_mining/exp_scripts
cd "$ENV"  # uv finds the project from the working directory
# Cache the base Longformer and seqeval once; transformers 4.20.1 cannot follow the hub's relative redirects.
uv run python -c "import requests, datasets, transformers as t; from urllib.parse import urljoin; head = requests.head
def fixed(url, **kw): r = head(url, **kw); r.headers['Location'] = urljoin(url, r.headers.get('Location', '')); return r
requests.head = fixed; datasets.load_metric('seqeval'); t.LongformerTokenizer.from_pretrained('$TOK')
t.LongformerForMaskedLM.from_pretrained('$TOK')"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # every run below reads that cache
if [[ $1 != none ]]; then
  SUB=$(realpath "$1")
  uv run python validate.py "$SUB"
  uv run python train.py "$SUB" "$OUT"
  cd "$SRC"  # pre_train.py reads ../Discourse_Markers.txt
  uv run torchrun --standalone --nproc_per_node=3 smlm_pretrain/pre_train.py "$OUT/corpus" "$OUT/smlm" 2>&1 |
    tee "$OUT/logs/pretrain.log"
  TOK=$OUT/smlm/tokenizer MODEL=$OUT/smlm/model
fi
cd "$ENV/external_eval_data"  # the scripts read ./Discourse_Markers.txt and the benchmarks from here
for task in "rtp_80 relation_type/prompt_5_cons_ft_runs_CmvModes.py 80" "rtp_50 relation_type/prompt_5_cons_ft_runs_CmvModes.py 50" \
  "drinv_aci_80 segmentation/5_cons_ft_runs_DrInventor.py 80" "cmv_aci_80 segmentation/5_cons_ft_runs_CmvModes.py 80"; do
  for run in 0 1 2 3 4; do echo "$task $run"; done
done | xargs -L1 -P "$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")" --process-slot-var=SLOT bash -c \
  'CUDA_VISIBLE_DEVICES=$(cut -d, -f$((SLOT + 1)) <<< "$CUDA_VISIBLE_DEVICES") uv run python "$0/$2" "$TOK" "$MODEL" "$3" "$4" \
  > "$OUT/logs/$1_$4.log" 2>&1' "$SRC"
uv run python ../eval.py "$OUT"
