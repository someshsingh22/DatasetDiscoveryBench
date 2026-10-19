"""Scores the fine-tuned models on the external benchmarks (paper Table 5) and writes OUT/scores.json.

Each single-label test text gets " I feel <extra_id_0>" appended; the benchmark's labels are ranked by their likelihood
under the model and the most likely is the prediction. primary = mean macro-F1 over the benchmarks.

Usage: eval.py OUT   (reads the OUT/<benchmark> models written by train.py)
"""

from __future__ import annotations

import json
import sys

import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from train import TASKS
from train import load_task


@torch.no_grad()
def evaluate(model_dir: str, task: str) -> dict[str, float]:
    """Returns macro P/R/F1 and accuracy of the most likely label on `task`'s single-label test rows."""
    labels = TASKS[task][1]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).cuda().eval()
    targets = tok([f"<extra_id_0> {label}" for label in labels], padding=True, return_tensors="pt").input_ids.cuda()
    targets[targets == tok.pad_token_id] = -100
    test = load_task(task, "test")
    test = test[~test.label.str.contains(",")]
    preds = []
    for text in test.clip_input:
        logits = model(**tok([text] * len(labels), return_tensors="pt").to("cuda"), labels=targets).logits
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction="none").sum(-1)
        preds.append(labels[nll.argmin()])
    gold = test.t5_trg.str.removeprefix("<extra_id_0> ")
    p, r, f, _ = precision_recall_fscore_support(gold, preds, labels=labels, average="macro", zero_division=0)
    return {
        "macro_f1": f,
        "macro_precision": p,
        "macro_recall": r,
        "accuracy": accuracy_score(gold, preds),
        "n": len(test),
    }


if __name__ == "__main__":
    out = sys.argv[1]
    metrics = {task: evaluate(f"{out}/{task}", task) for task in TASKS}
    scores = {"primary": sum(m["macro_f1"] for m in metrics.values()) / len(metrics), "metrics": metrics}
    with open(f"{out}/scores.json", "w") as f:
        json.dump(scores, f, indent=2)
