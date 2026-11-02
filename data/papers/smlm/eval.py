"""Scores the authors' fine-tuning logs (written by run.sh) and writes OUT/scores.json.

A task's F1 is the mean over its runs (seeded random splits; the paper: 5) of the run's mean test F1 over its last 5
epochs, as in the paper: seqeval micro-F1 of the component tags (Dr Inventor and CMV Modes ACI, 80/20 split) and
class-weighted F1 of the relation types (CMV Modes RTP, 80/20 and 50/50 splits). primary = {<task>_f1}; higher is
better.

Usage: eval.py OUT   (reads OUT/logs/<task>_<run>.log)
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

TASKS = ["drinv_aci_80", "cmv_aci_80", "rtp_80", "rtp_50"]

if __name__ == "__main__":
    out = Path(sys.argv[1])
    metrics = {}
    for task in TASKS:
        runs = []
        for log in sorted(out.glob(f"logs/{task}_*.log")):  # each epoch's test metrics: a dict printed after 4 tabs
            epochs = [ast.literal_eval(s) for s in log.read_text().splitlines() if s.startswith("\t\t\t\t ")][-5:]
            f1s = [e["overall_f1"] if "overall_f1" in e else e["weighted_avg"]["f1"] for e in epochs]
            runs.append(sum(f1s) / len(f1s))
        metrics[task] = {"f1": sum(runs) / len(runs), "runs": runs}
    scores = {"primary": {f"{task}_f1": m["f1"] for task, m in metrics.items()}, "metrics": metrics}
    (out / "scores.json").write_text(json.dumps(scores, indent=2))
