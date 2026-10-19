# Meta prompt: build a DatasetDiscoveryBench environment from a paper

Give an agent this file plus `PAPER_URL` and `CODE_URL` (optionally a hand-written `task.md`). It must deliver, with no
further input, a new `data/papers/<id>/` environment pushed to `main` of
`git@github.com:someshsingh22/DatasetDiscoveryBench.git` and validated end to end on AIP (Pluto) 8×A100-80GB nodes.
`data/papers/masive/` is the worked example; read it first and copy its shape. **Maintain this file**: when a run
teaches you something (a pitfall, a better default, a faster check), add it to *Lessons* before you finish.

## 1. What an environment is

An agent reads only `task.md` and submits one dataset file. The environment scores it with the *external evaluation*
of the paper that introduced the original dataset, so the score is comparable with the original dataset's.

- `bash run.sh SUBMISSION OUT` writes `OUT/scores.json` = `{"primary": float, "metrics": {...}}`. Higher is better.
  If the paper's evaluation compares "with vs without the dataset", `run.sh none OUT` is the without-dataset baseline.
- The environment fixes the **method and hyperparameters** (the paper's); only the data varies, including its size.
- Not the environment's job: provenance checks, reading the agent's workspace, policing synthetic data or dataset size.
  The source corpus and its time window are controlled elsewhere.
- `validate.py` rejects a submission that is malformed or contains the evaluation's test texts (contamination).
  The authors' original dataset must pass it.
- Hardware: one node, 8×A100-80GB, CUDA 12.8 image. `run.sh` uses every visible GPU (1, 2, 4 or 8) and keeps the paper's
  effective batch size: per-device batch = paper batch // `WORLD_SIZE`.
- Tooling: `uv` (Python 3.12, `pyproject.toml` + `uv.lock`); no Docker for now.

Layout (as in [PaperBench](https://github.com/openai/frontier-evals/tree/main/project/paperbench), one dir per paper):

```
data/papers/<id>/
  config.yaml          id, title, paper, code, original-dataset download
  task.md              agent-facing: research question, rubric, submission format (+ example)
  run.sh               validate -> train on submission (skipped for `none`) -> downstream training -> eval
  validate.py          format asserts + n-gram contamination check against every external test set
  train.py, eval.py    the paper's recipe and evaluation (shared constants live in train.py)
  external_eval_data/  the evaluation benchmarks (small CSVs, as distributed by the authors)
  tests/               __init__.py, sample.csv (50 original rows), test_<id>.py
  reference.json       paper numbers + AIP scores (original, baseline, author recipe, seed spread)
  pyproject.toml, uv.lock
```

**Minimalism is a hard requirement**: not one line more than necessary. argv instead of argparse; no config systems,
classes, logging frameworks, retries, or defensive checks the data does not need. Match `masive/`: `from __future__ import
annotations`, Google-style docstrings, one-line module docstring with usage, ruff (line-length 120, single-line imports).

## 2. Procedure

### 2.1 Eligibility
Proceed only if (a) the paper proposes a dataset derived from Reddit, (b) it evaluates the dataset externally (e.g. train
on it, test on an independent benchmark), and (c) the evaluation code is available or very simple. Otherwise stop and
report why. Pick a short lowercase `<id>`.

### 2.2 Research (notes go in your session files, never in the repo)
1. Paper: dataset construction and format, size, the external evaluation (benchmarks, model, stages, hyperparameters,
   metrics) and the results table to reproduce (with and without the dataset).
2. Code: clone it. Find the evaluation scripts, their *default* arguments (they are the hyperparameters actually used),
   the dataset download link and the benchmark files.
3. Diff paper text vs code and list every discrepancy. Decision rule: the environment implements the **paper-described
   method with the code's hyperparameters**; clear bugs are fixed in the environment (document each in a code comment
   only if non-obvious). The authors' exact code path is reproduced as a *patch* applied to a copy of the env during
   validation (§2.6), never as env code.
4. Local work is limited to light file prep: downloading the dataset to sample 50 rows (`random_state=0`, all columns),
   copying benchmark files, `uv lock`, ruff, `bash -n`. **Never run env code, training or tests locally.**

### 2.3 Write the environment
Copy `masive/` file by file and adapt:
- `train.py`: `TASKS` (benchmark -> files, label names), `load_task`, and the training entry point
  (`train.py INIT DATA OUT`, DATA = submission path or benchmark name). `MAX_STEPS` env var for smoke tests only.
  Use `Seq2SeqTrainer`/`Trainer` + `trainer.save_model()` (DDP-safe), `save_strategy="no"`, bf16.
- `eval.py`: reproduce the paper's metric exactly as the authors compute it (e.g. macro averaging over *all* label
  names with `zero_division=0`). Writes `OUT/scores.json`; `primary` = the mean of the headline metric over benchmarks.
- `validate.py`: required columns non-empty, format invariants, and contamination: a test text with ≥8 words is
  leaked if all its word 8-grams occur in the submission.
- `run.sh`: `set -euo pipefail`; absolute paths; `uv run torchrun --standalone --nproc-per-node=gpu`.
- `tests/test_<id>.py` (stdlib unittest, run with `uv run python -m unittest discover -s tests -t .`):
  our mock valid/invalid rows, `sample.csv` passes, and a GPU smoke run of `run.sh` (tiny model, `MAX_STEPS=2`)
  producing a well-formed `scores.json`.
- `task.md`: research question; rubric = expectations about the dataset inferred from the paper (source, signal,
  diversity, language, scale of the original); exact submission format with an example. Do not describe the benchmarks.
- `pyproject.toml`: `requires-python = "==3.12.*"`, only needed deps, `torch>=2.4,<2.11`, `transformers<5`.
- Root `README.md`: add a row to the Papers table.

### 2.4 Static checks (local)
`uvx ruff check . && uvx ruff format --check .`, `bash -n run.sh`, `uv lock`. Inspect the lock for major-version jumps
(pandas, datasets, torch CUDA wheels) and statically confirm the APIs you call exist in the locked versions.
Dry-apply every validation patch to a temp copy and `ast.parse` the result.

### 2.5 Push
Commit with the trailer `Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>` and
`git push` over SSH. (Repo creation, first time only:
`GH_TOKEN=$(env -u GH_TOKEN gh auth token -u someshsingh22) gh repo create someshsingh22/DatasetDiscoveryBench --private`.)
Jobs `git clone` main at start, so every fix is a push plus a relaunch.

### 2.6 Validate on AIP (aip MCP)
Use the `aip` MCP tools; if they are not loaded in your session, spawn a coordinated child session in `~/Github/pluto`
(where they are) and give it these instructions verbatim. Follow `~/Github/pluto/AGENTS.md`.
- Job scripts: copy `~/Github/pluto/jobs/ddb-masive/` (`init.sh`, `main.sh`, `README.md`) to `jobs/ddb-<id>/` and
  adapt. Only INIT and MAIN reach the pod: embed patches as heredocs.
- Job spec: project `mdsr-training`, image
  `docker-matrix-experiments-snapshot.ff.adobe.io/training-base-01-31-25:12.8.0-runtime-ubuntu22.04-CUDNN-v9.7.0.66-NCCL-v2.25.1-1`,
  1 pod × 8 `NVIDIA_A100_80GB`, `useSingleNodeGroup`, `disablePodRestart`, `enableLogCollection`, an env var selecting the
  experiment. Set the exec spec without `scripts`, upload INIT+MAIN together via `compute_update_job_scripts`, start.
- In MAIN: copy `/sensei-fs/users/someshs/{.ssh,.aws}` to `$HOME` (dirs 700, files 600), unset inherited AWS vars,
  `source secrets.sh` (HF_TOKEN), `HF_HOME`/`UV_CACHE_DIR` on `/mnt/localssd`, clone, `uv sync` once per env copy before
  concurrent runs, print the commit, driver and `torch.cuda.is_available()`, tee every run's log, and export only
  `scores.json`, logs and trainer metrics to `s3://mdsr-gcp-usw2/somesh/ddb/<id>/<run>-<stamp>/` (never weights).
  Never print credentials; never work around auth errors.
- Max 3 active jobs. Machines start slowly: launch all three at once, put the longest runs first, and split each node
  into two 4-GPU experiments (`CUDA_VISIBLE_DEVICES=0,1,2,3` / `4,5,6,7`). Default matrix:

  | job | GPUs 0-3 | GPUs 4-7 |
  |---|---|---|
  | M1 | unittests (fail fast, all 8 GPUs), then env on the original dataset | author recipe on the original dataset |
  | M2 | env baseline (`none`) | author-recipe baseline |
  | M3 | env baseline, `seed=1` (noise estimate) | env on a 10% sample of the original (`random_state=0`) |

  Relaunch freed machines with whatever is still missing. If a precision/stability question is open, split a node
  between the variants instead.
- Monitoring: `get_run_logs` returns a zip URL; download it and grep the `DDB_` markers and Trainer `'loss'` lines.
  Within ~5 minutes of training, check the first loss lines: `grad_norm: nan`, `loss: 0.0` or a learning rate that
  never decays means a broken run. Fix, push, relaunch immediately. Check every 20-30 min; submission is not success:
  confirm terminal state, outputs in S3, and that pods were released.

### 2.7 Acceptance (all must hold; otherwise iterate or report precisely why not)
1. Unittests pass on AIP; the original dataset passes `validate.py`.
2. The author recipe roughly reproduces the paper's table (same ordering; within ~5 points or explained).
3. Env original vs env baseline difference exceeds the seed spread; if not, say the environment does not
   discriminate datasets and propose the smallest change that would (do not silently change the method).
4. `reference.json` records the paper numbers, every valid AIP run (scores, commit, GPUs, wall time per stage) and the
   seed spread; the README row and this file's *Lessons* are updated; everything is pushed.
5. Final report: a table of paper vs author recipe vs env (original, baseline, 10%, seed 1), the discrepancies found in
   the authors' code, runtimes, and open questions.

## Lessons

- mT5 overflows in fp16 (`grad_norm: nan`, loss 0.0, every step skipped); train T5-family models in bf16. HF's
  `logging_nan_inf_filter` hides it (reported loss 0.0), so `train.py` asserts `0 < training_loss < inf`.
- torch ≥ 2.11 on PyPI ships CUDA 13 wheels; pin `torch<2.11` (2.10.0+cu128 verified on AIP p4de, driver 580).
- Locks resolve to new majors (pandas 3, datasets 5, accelerate 1.15 with transformers 4.57): fine for `masive/`.
- Authors' code can contradict the paper. MASIVE: the downstream *training* input contained the gold label
  (`text I feel <extra_id_0> angry`) while test inputs did not, and candidate scoring counted pad tokens. Such bugs can
  explain the paper's absolute numbers, hence the author-recipe validation run.
- Saving from every DDP rank races; `trainer.save_model()` saves once and includes the tokenizer.
- The MASIVE unittests (smoke run with mt5-small) take ~7.5 min on 8×A100; `uv sync` + model download ~1 min.
- A test-set contamination check by word 8-grams costs seconds and the authors' dataset passes it.
