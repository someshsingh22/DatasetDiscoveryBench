"""Scores an mT5 checkpoint with the paper's evaluations and writes OUT/scores.json.

- zero_shot (Table 5 benchmarks, without the paper's per-benchmark fine-tune, whose training inputs contain the gold
  label): each single-label test text gets " I feel <extra_id_0>" appended; the benchmark's labels are ranked by their
  likelihood under the model and the most likely is the prediction. Macro P/R/F1 over all labels.
- masive_test (Table 4): five beams per post of the MASIVE English test set (rows that duplicate MASIVE train posts
  removed); acc@k = share of the post's masked terms found at the same slot in the top-k beams (authors' metrics.py).

primary = {zero_shot_macro_f1: mean over the benchmarks, masive_acc_1}; compare each separately, higher is better.

Usage: eval.py MODEL OUT
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

DATA = Path(__file__).parent / "external_eval_data"
MASIVE_TEST = DATA / "masive_en_test.csv"
TASKS = {  # benchmark: (file prefix, adjective labels indexed by the CSV's integer labels), from the authors' code
    "emoevent_en": ("emo_event_en", ["angry", "disgusted", "scared", "happy", "sad", "surprised", "nothing"]),
    "goemo": ("goemo_ekman", ["angry", "disgusted", "afraid", "happy", "sad", "surprised", "nothing"]),
    "goemo_full": ("goemo_full", [
        "admiration", "amused", "angry", "annoyed", "approving", "caring", "confused", "curious", "desire",
        "disappointed", "disapproval", "disgusted", "embarassed", "excited", "afraid", "grateful", "grief", "happy",
        "love", "nervous", "optimistic", "proud", "realized", "relieved", "remorseful", "sad", "surprised", "nothing",
    ]),
}  # fmt: skip
SLOT = re.compile(r"<extra_id_\d+>|</s>|<pad>")


def load_benchmark(task: str) -> pd.DataFrame:
    """Returns `task`'s test rows (text, label) as in the authors' CSVs."""
    return pd.read_csv(DATA / f"{TASKS[task][0]}_test.csv", dtype=str, keep_default_na=False)


@torch.no_grad()
def zero_shot(model, tok, task: str) -> dict[str, float]:
    """Returns macro P/R/F1 and accuracy of the most likely label on `task`'s single-label test rows."""
    labels = TASKS[task][1]
    targets = tok([f"<extra_id_0> {label}" for label in labels], padding=True, return_tensors="pt").input_ids.cuda()
    targets[targets == tok.pad_token_id] = -100
    test = load_benchmark(task)
    test = test[~test.label.str.contains(",")]
    preds = []
    for text in test.text + " I feel <extra_id_0>":
        logits = model(**tok([text] * len(labels), return_tensors="pt").to("cuda"), labels=targets).logits
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction="none").sum(-1)
        preds.append(labels[nll.argmin()])
    gold = test.label.astype(int).map(labels.__getitem__)
    p, r, f, _ = precision_recall_fscore_support(gold, preds, labels=labels, average="macro", zero_division=0)
    return {
        "macro_f1": f,
        "macro_precision": p,
        "macro_recall": r,
        "accuracy": accuracy_score(gold, preds),
        "n": len(test),
    }


def slots(text: str) -> list[str]:
    return [s.strip() for s in SLOT.split(text) if s.strip()]


@torch.no_grad()
def masive_test(model, tok, batch_size: int = 4) -> dict[str, float]:
    """Returns mean top-1/3/5 slot accuracy of five-beam generations on the MASIVE test set (authors' batch size)."""
    test = pd.read_csv(MASIVE_TEST, dtype=str)
    acc = {1: [], 3: [], 5: []}
    for i in range(0, len(test), batch_size):
        batch = test.iloc[i : i + batch_size]
        enc = tok(list(batch.clip_input), truncation=True, max_length=512, padding=True, return_tensors="pt")
        gold = tok(list(batch.t5_trg), truncation=True, max_length=512).input_ids
        beams = model.generate(
            **enc.to("cuda"), num_beams=5, num_return_sequences=5, max_new_tokens=2 * max(map(len, gold))
        )
        beams = [slots(text) for text in tok.batch_decode(beams)]
        for j, terms in enumerate(slots(text) for text in tok.batch_decode(gold)):
            for k in acc:
                top = beams[5 * j : 5 * j + k]
                acc[k].append(sum(t in [b[n] for b in top if len(b) > n] for n, t in enumerate(terms)) / len(terms))
    return {f"acc_{k}": sum(v) / len(v) for k, v in acc.items()} | {"n": len(test)}


if __name__ == "__main__":
    model_dir, out = sys.argv[1:]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).cuda().eval()
    metrics = {
        "zero_shot": {task: zero_shot(model, tok, task) for task in TASKS},
        "masive_test": masive_test(model, tok),
    }
    primary = {
        "zero_shot_macro_f1": sum(m["macro_f1"] for m in metrics["zero_shot"].values()) / len(TASKS),
        "masive_acc_1": metrics["masive_test"]["acc_1"],
    }
    Path(out, "scores.json").write_text(json.dumps({"primary": primary, "metrics": metrics}, indent=2))
