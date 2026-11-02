# DatasetDiscoveryBench

Environments that score a discovered dataset with the external evaluation of the paper that introduced the original
dataset. An agent reads `task.md` and submits one dataset; the environment fixes the paper's method and
hyperparameters (only the data, including its size, varies) and reports scores comparable to the original dataset's.

## Layout

One directory per paper, as in [PaperBench](https://github.com/openai/frontier-evals/tree/main/project/paperbench):

```
data/papers/<id>/
  config.yaml          paper, code and original-dataset links
  task.md              what the agent sees: research question, rubric, submission format
  run.sh               run.sh SUBMISSION OUT -> OUT/scores.json {"primary": {headline metrics}, "metrics": {...}}
  validate.py          submission format + benchmark-contamination check (the original dataset passes)
  train.py, eval.py    the paper's recipe and evaluation
  external_eval_data/  the evaluation benchmarks
  tests/               mock inputs, 50 sampled rows of the original, and a smoke run of run.sh
  reference.json       paper numbers and run.sh scores for the original dataset and the no-dataset baseline
  pyproject.toml       the environment (uv)
```

## Usage

On one node with 8×A100 80GB:

```bash
cd data/papers/masive
uv run python -m unittest discover -s tests -t .
bash run.sh submission.csv out/   # out/scores.json
```

## Adding a paper

Give an agent [`META_PROMPT.md`](META_PROMPT.md) with the paper and code URLs; it builds the environment, pushes it and
validates it on AIP.

## Papers

| id | paper | evaluation | primary |
|---|---|---|---|
| `masive` | [Deas et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1139/) | mT5-large fine-tuned on the submission, scored zero-shot on EmoEvent-En, GoEmotions-7 and GoEmotions-28 (the paper's per-benchmark fine-tune leaks the gold label) and on the MASIVE English test set | zero-shot mean macro-F1; MASIVE test Acc@1 |
| `smlm` | [Dutta et al., ACL 2022](https://aclanthology.org/2022.acl-long.536/) | Longformer-base pre-trained with selective MLM (masked discourse markers) on the submission's threads, then fine-tuned on CMV Modes (argument component identification, relation type prediction; ACI omits the authors' re-tokenization artifact, a lone-space token before 42% of gold segment boundaries that alone lifts F1 from ~0.25 to ~0.56) | F1 of each: ACI micro, RTP 80/20 and 50/50 weighted |
| `shp` | [Ethayarajh et al., ICML 2022](https://proceedings.mlr.press/v162/ethayarajh22a.html) | FLAN-T5-large fine-tuned on the submission as a SteamSHP preference model ("Which response is better?" -> A/B), scored with the paper's `v_info.py` on the HH-RLHF helpful-base test pairs in both orders (so H_V(Y) = 1 bit); the submission's own held-out V-information (trained null model) is reported too | HH-RLHF accuracy; HH-RLHF V-usable information (bits) |
