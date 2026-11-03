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
  src/<repo>/          the authors' training and evaluation code
  train.py, eval.py    thin wrappers that run it on the submission
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
| `smlm` | [Dutta et al., ACL 2022](https://aclanthology.org/2022.acl-long.536/) | The authors' code: Longformer-base pre-trained with selective MLM (masked discourse markers) on the submission's threads, then fine-tuned once per task (`RUNS=5` as in the paper): argument component identification on Dr Inventor and CMV Modes (80/20), relation types on CMV Modes (80/20, 50/50) | F1 of each task: ACI seqeval micro, RTP class-weighted |
| `jic` | [Pal et al., COLING 2025](https://aclanthology.org/2025.coling-main.470/) | Mistral-7B-Instruct-v0.3 QLoRA-fine-tuned on the submission's dialogues (authors' recipe), scored 0-shot on lm-evaluation-harness's Big Five persona tasks (Anthropic model-written evals) | accuracy of each trait: openness, conscientiousness, extraversion, agreeableness, neuroticism |
| `shp` | [Ethayarajh et al., ICML 2022](https://proceedings.mlr.press/v162/ethayarajh22a.html) | FLAN-T5-large fine-tuned on the submission as a SteamSHP preference model ("Which response is better?" -> A/B), scored with the paper's `v_info.py` on the HH-RLHF helpful-base test pairs in both orders (so H_V(Y) = 1 bit); the submission's own held-out V-information (trained null model) is reported too | HH-RLHF accuracy; HH-RLHF V-usable information (bits) |
| `tldr` | [Stiennon et al., NeurIPS 2020](https://arxiv.org/abs/2009.01325) | Pythia-1B-deduped fine-tuned on the submission with the SFT recipe of the paper's re-implementation (Huang et al., COLM 2024; the paper's code is inference-only), then summarizing the 11,490 CNN/DailyMail test articles without further training (512-token `{article}\n\nTL;DR:` queries, greedy) | ROUGE-1, ROUGE-2, ROUGE-L F1 |
