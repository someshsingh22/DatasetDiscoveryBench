"""Scores the CMV Modes fine-tuning runs of train.py (paper Tables 1 and 3) and writes OUT/scores.json.

A task's F1 is the mean over its 5 runs (random thread splits) of the run's mean test F1 over its last 5 epochs:
micro-F1 of the BIO component tags for aci_80, class-weighted F1 of the 5 relation types for rtp_80 and rtp_50.
primary = mean F1 over the tasks.

Usage: eval.py OUT   (reads the OUT/cmv_modes/<task>_<run>.json files written by train.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from train import TASKS

if __name__ == "__main__":
    out = Path(sys.argv[1])
    metrics = {}
    for task in TASKS:
        runs = [json.loads(p.read_text())["scores"] for p in sorted(out.glob(f"cmv_modes/{task}_*.json"))]
        runs = [sum(scores) / len(scores) for scores in runs]
        metrics[task] = {"f1": sum(runs) / len(runs), "runs": runs}
    scores = {"primary": sum(m["f1"] for m in metrics.values()) / len(metrics), "metrics": metrics}
    (out / "scores.json").write_text(json.dumps(scores, indent=2))
