# Meta prompt: build a DatasetDiscoveryBench environment from a paper

Give an agent this file plus `PAPER_URL` and `CODE_URL` (optionally a hand-written `task.md`). It must deliver, with no
further input, a new `data/papers/<id>/` environment pushed to `main` of
`git@github.com:someshsingh22/DatasetDiscoveryBench.git` and validated end to end on AIP (Pluto) 8×A100-80GB nodes.
The existing `data/papers/*/` are worked examples; read one first and copy its shape. **Maintain this file**: when a run
teaches you something general (a pitfall, a better default, a faster check), add it to *Lessons* before you finish;
keep this file paper-agnostic. Other agents build other papers concurrently: touch only your `data/papers/<id>/`, your
README row and *Lessons*.

## 1. What an environment is

An agent reads only `task.md` and submits one dataset file. The environment scores it with the *external evaluation*
of the paper that introduced the original dataset, so the score is comparable with the original dataset's.

- `bash run.sh SUBMISSION OUT` writes `OUT/scores.json` = `{"primary": {name: float}, "metrics": {...}}`: `primary`
  holds the paper's headline metrics, each compared separately (never combined), higher is better.
  If the paper's evaluation compares "with vs without the dataset", `run.sh none OUT` is the without-dataset baseline.
- The environment runs the **authors' own code and hyperparameters**, minimally changed (§2.2); only the data varies,
  including its size.
- Not the environment's job: provenance checks, reading the agent's workspace, policing synthetic data or dataset size.
  The source corpus and its time window are controlled elsewhere.
- `validate.py` rejects a submission that is malformed or contains the evaluation's test texts (contamination).
  The authors' original dataset must pass it.
- Hardware: one node, 8×A100-80GB, CUDA 12.8 image. `run.sh` uses every visible GPU (1, 2, 4 or 8) and keeps the paper's
  effective batch size: per-device batch = paper batch // `WORLD_SIZE`.
- Tooling: `uv` (`pyproject.toml` + `uv.lock`); no Docker for now.

Layout (as in [PaperBench](https://github.com/openai/frontier-evals/tree/main/project/paperbench), one dir per paper):

```
data/papers/<id>/
  config.yaml          id, title, paper, code (URL + pinned commit), original-dataset download
  task.md              agent-facing: research question, rubric, submission format (+ example)
  run.sh               validate -> train on submission (skipped for `none`) -> downstream training, if any -> eval
  validate.py          format asserts + n-gram contamination check against every external test set
  src/<repo>/          the authors' files that training and evaluation need, from the pinned commit (§2.2)
  train.py, eval.py    thin wrappers: read the submission / write scores.json, call the authors' functions in src/
  external_eval_data/  the evaluation benchmarks (small CSVs, as distributed by the authors)
  tests/               __init__.py, sample.csv (50 original rows), test_<id>.py
  reference.json       paper numbers + AIP scores (original, baseline, 10%, seed spread)
  pyproject.toml, uv.lock
```

**Minimalism is a hard requirement** for our code (wrappers, `run.sh`, `validate.py`, tests, `# DDB:` edits): not one
line more than necessary. argv instead of argparse; no config systems, classes, logging frameworks, retries, or defensive
checks the data does not need. Match the existing environments: `from __future__ import annotations`, Google-style
docstrings, one-line module docstring with usage, ruff (line-length 120, single-line imports). Never restyle copied
code; it must stay diffable against the original.

## 2. Procedure

### 2.1 Eligibility
Proceed only if (a) the paper proposes a dataset derived from Reddit, (b) it evaluates the dataset externally (e.g. train
on it, test on an independent benchmark), and (c) the evaluation code is available or very simple. Otherwise stop and
report why. Pick a short lowercase `<id>`.

### 2.2 Research (notes go in your session files, never in the repo)
1. Paper: dataset construction and format, size, the external evaluation (benchmarks, model, stages, hyperparameters,
   metrics) and the results table to reproduce (with and without the dataset).
2. Code: clone it and pin a commit. Find the evaluation scripts, their *default* arguments (they are the hyperparameters
   actually used), the dataset download link and the benchmark files.
3. Diff paper text vs code and list every discrepancy; the code wins, since it produced the paper's numbers. Reuse the
   authors' code, changed only to (a) read the submission and write `scores.json`, (b) run efficiently on 1-8 GPUs
   (e.g. DDP keeping the effective batch, evaluating only scored epochs), or (c) fix a critical bug, only after an AIP
   run shows it causally and the user approves. Mark each change with a short `# DDB: reason` comment.
4. Local work is limited to light file prep: downloading the dataset to sample 50 rows (`random_state=0`, all columns),
   copying benchmark files, `uv lock`, ruff, `bash -n`. **Never run env code, training or tests locally, not even on
   CPU**; every execution happens on AIP.

### 2.3 Write the environment
Copy the shape of the closest existing environment:
- `src/<repo>/`: copy only the authors' files that training and evaluation use, byte-identical except §2.2 edits.
- `train.py`, `eval.py`: import and call the authors' functions with their defaults (re-host a function only if its
  module cannot be imported, e.g. a script with top-level code, and say so in the docstring); add only submission I/O,
  a `MAX_STEPS` env var for smoke tests and an assert of a finite non-zero training loss. `eval.py` writes
  `OUT/scores.json`; `primary` = each headline metric (e.g. its mean over benchmarks).
- `validate.py`: required columns non-empty, format invariants, and contamination: a test text with ≥8 words is
  leaked if all its word 8-grams occur in the submission.
- `run.sh`: `set -euo pipefail`; absolute paths; `uv run torchrun --standalone --nproc-per-node=gpu`.
- `tests/test_<id>.py` (stdlib unittest, run with `uv run python -m unittest discover -s tests -t .`):
  our mock valid/invalid rows, `sample.csv` passes, and a GPU smoke run of `run.sh` (tiny model, `MAX_STEPS=2`)
  producing a well-formed `scores.json`.
- `task.md`: research question; rubric = expectations about the dataset inferred from the paper (source, signal,
  diversity, language, scale of the original); exact submission format with an example. Do not describe the benchmarks.
- `pyproject.toml`: only needed deps, at the authors' versions (their requirements, else the last release before their
  final commit); change one only if it cannot run on the image. Upgrades silently change results, e.g. fast tokenizers
  keep the spaces around added tokens that slow ones strip.
- Root `README.md`: add a row to the Papers table.

### 2.4 Static checks (local)
`uvx ruff check` and `uvx ruff format --check` on our own files, `python -m py_compile` on copied code, `bash -n run.sh`,
`uv lock`; confirm the lock holds the intended versions.

### 2.5 Push
Work in your own clone (`git clone git@github.com:someshsingh22/DatasetDiscoveryBench.git $TMPDIR/ddb-<id>`), never in
another agent's checkout. Commit with the trailer `Co-authored-by: Copilot App <223556219+Copilot@users.noreply.github.com>`,
then `git pull --rebase && git push` over SSH. (Repo creation, first time only:
`GH_TOKEN=$(env -u GH_TOKEN gh auth token -u someshsingh22) gh repo create someshsingh22/DatasetDiscoveryBench --private`.)
Jobs `git clone` main at start, so every fix is a push plus a relaunch.

### 2.6 Validate on AIP (aip MCP)
Use the `aip` MCP tools; if they are not loaded in your session, spawn a coordinated child session in `~/Github/pluto`
(where they are) and give it these instructions verbatim. Follow `~/Github/pluto/AGENTS.md`.
- Job scripts: copy an existing `~/Github/pluto/jobs/ddb-*/` (`init.sh`, `main.sh`, `README.md`) to `jobs/ddb-<id>/`
  and adapt; commit it in the pluto repo when done. Only INIT and MAIN reach the pod.
- Job spec: project `mdsr-training`, image
  `docker-matrix-experiments-snapshot.ff.adobe.io/training-base-01-31-25:12.8.0-runtime-ubuntu22.04-CUDNN-v9.7.0.66-NCCL-v2.25.1-1`,
  1 pod × 8 `NVIDIA_A100_80GB`, `useSingleNodeGroup`, `disablePodRestart`, `enableLogCollection`, an env var selecting the
  experiment. Set the exec spec without `scripts`, upload INIT+MAIN together via `compute_update_job_scripts`, start.
- In MAIN: copy `/sensei-fs/users/someshs/{.ssh,.aws}` to `$HOME` (dirs 700, files 600), unset inherited AWS vars,
  `source secrets.sh` (HF_TOKEN), `HF_HOME`/`UV_CACHE_DIR` on `/mnt/localssd`, clone, `uv sync` once per env copy before
  concurrent runs, print the commit, driver and `torch.cuda.is_available()`, tee every run's log, and export only
  `scores.json`, logs and trainer metrics to `s3://mdsr-gcp-usw2/somesh/ddb/<id>/<run>-<stamp>/` (never weights).
  Never print credentials; never work around auth errors.
- Max 3 active jobs across **all** of the user's agents: before each launch count the user's active jobs
  (`query_entities`, owner = `get_current_user`); if 3 are active, wait and re-check every ~20 min. Never stop, restart
  or edit a job you did not create. Machines start slowly: launch as many as the cap allows at once, put the longest
  runs first, and split each node into two 4-GPU experiments (`CUDA_VISIBLE_DEVICES=0,1,2,3` / `4,5,6,7`). Default
  matrix:

  | job | GPUs 0-3 | GPUs 4-7 |
  |---|---|---|
  | M1 | unittests (fail fast, all 8 GPUs), then env on the original dataset | env on the original, `seed=1` |
  | M2 | env baseline (`none`) | env baseline, `seed=1` |
  | M3 | env on a 10% sample of the original (`random_state=0`) | authors' unmodified code, only after a §2.2(c) fix |

  Relaunch freed machines with whatever is still missing. If a precision/stability question is open, split a node
  between the variants instead.
- Monitoring: `get_run_logs` returns a zip URL; download it and grep the `DDB_` markers and Trainer `'loss'` lines.
  Within ~5 minutes of training, check the first loss lines: `grad_norm: nan`, `loss: 0.0` or a learning rate that
  never decays means a broken run: fix, push, relaunch immediately (if the authors' unmodified code does the same, ask
  the user, §2.2(c)). Check every 20-30 min; submission is not success:
  confirm terminal state, outputs in S3, and that pods were released.

### 2.7 Acceptance (all must hold; otherwise iterate or report precisely why not)
1. Unittests pass on AIP; the original dataset passes `validate.py`.
2. The env on the original dataset and the baseline reproduce the paper's table (same ordering; within ~5 points or
   the paper's own spread). A gap is ours until shown otherwise; after a §2.2(c) fix, the authors' unmodified code
   must reproduce it instead.
3. Env original vs env baseline difference exceeds the seed spread; if not, say the environment does not
   discriminate datasets and propose the smallest change that would (do not silently change the method).
4. `reference.json` records the paper numbers, every valid AIP run (scores, commit, GPUs, wall time per stage) and the
   seed spread; the README row and this file's *Lessons* are updated; everything is pushed.
5. Final report: a table of paper vs env (original, baseline, 10%, seed 1), every `# DDB:` change and why, paper vs
   code discrepancies, runtimes, and open questions.

## Lessons

- T5-family models (T5 v1.1, mT5) overflow in fp16 (`grad_norm: nan`, every step skipped); use bf16. HF's
  `logging_nan_inf_filter` hides it (reported loss 0.0), and tiny smoke runs pass anyway, hence the loss assertion.
- Pin torch to a version verified on AIP: 2.10.0+cu128 works on p4de (driver 580); ≥ 2.11 pulls CUDA 13 wheels.
- Rewriting the authors' code lost both smlm results: our encoding dropped a boundary cue theirs has (ACI 0.25 vs 0.56)
  and our RTP runs collapsed to a constant predictor (macro recall ≈ 1/K). Hence reuse. To locate a gap, diff our
  encoded inputs against theirs token by token and check per-epoch predictions. A real author bug shows in a run:
  MASIVE puts each gold label in its own *training* input but not test inputs, so training loss is ≈ 0 within steps.
- Authors' train/test splits can share posts; drop those test rows so the original passes the contamination check.
- Saving from every DDP rank races; `trainer.save_model()` saves once and includes the tokenizer.
- Typical costs on 4×A100 bf16: a smoke unittest run ~7.5 min (8 GPUs); `uv sync` + model download ~1 min; a ~1B-param
  model trains ~2 it/s at per-device batch 2, so a 94k-row, 3-epoch stage takes ~4.6h. Measure it/s in the first
  minutes and extrapolate; the full-data original run is the critical path, so start it first.
- Word 8-gram contamination checks cost seconds and do not flag genuine original datasets.
- aip MCP tokens expire ("Re-authentication required") and a live session keeps its stale connection even after the
  user re-authenticates; ask the user to re-auth, then make aip calls from a fresh session in `~/Github/pluto`. Never
  switch to another interface. Tokens can drop again within the hour, but running jobs are unaffected, so launch every
  pending job right after re-authenticating.
- Unlabeled corpora can contain the benchmark itself (sMLM: 30 Winning Arguments conversations are CMV Modes
  threads). Match by thread ID as well as by n-grams, drop whole conversations, and record how many were dropped.
- CSV text with `\r` breaks rows on reload unless written with `quoting=csv.QUOTE_ALL`.
- When the metric averages the last k epochs, evaluate only those epochs. Evaluating every epoch can double runtime.
- For threaded or tree data, take the 10% sample by conversation (root), not by row; a row sample orphans most replies.
