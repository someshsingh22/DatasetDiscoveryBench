"""Fine-tunes an mT5 checkpoint with the MASIVE recipe (Deas et al., EMNLP 2024; the authors' code defaults).

Usage: train.py INIT DATA OUT
  DATA is a submission CSV (clip_input -> t5_trg, effective batch 8) or a benchmark name from TASKS
  (text + " I feel <extra_id_0>" -> "<extra_id_0> label", effective batch 16). Both: 3 epochs, Adafactor,
  lr 1e-4 linear, weight decay 0.01, fp16. MAX_STEPS shortens training (smoke tests only).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

DATA = Path(__file__).parent / "external_eval_data"
TASKS = {  # benchmark: (file prefix, adjective labels indexed by the CSV's integer labels), from the authors' code
    "emoevent_en": ("emo_event_en", ["angry", "disgusted", "scared", "happy", "sad", "surprised", "nothing"]),
    "goemo": ("goemo_ekman", ["angry", "disgusted", "afraid", "happy", "sad", "surprised", "nothing"]),
    "goemo_full": ("goemo_full", [
        "admiration", "amused", "angry", "annoyed", "approving", "caring", "confused", "curious", "desire",
        "disappointed", "disapproval", "disgusted", "embarassed", "excited", "afraid", "grateful", "grief", "happy",
        "love", "nervous", "optimistic", "proud", "realized", "relieved", "remorseful", "sad", "surprised", "nothing",
    ]),
}  # fmt: skip


def load_task(task: str, split: str) -> pd.DataFrame:
    """Loads a benchmark split in the submission format, targeting each row's first label."""
    prefix, labels = TASKS[task]
    df = pd.read_csv(DATA / f"{prefix}_{split}.csv", dtype=str, keep_default_na=False)
    df["clip_input"] = df.text + " I feel <extra_id_0>"
    df["t5_trg"] = "<extra_id_0> " + df.label.str.split(",").str[0].astype(int).map(labels.__getitem__)
    return df


if __name__ == "__main__":
    init, data, out = sys.argv[1:]
    df = load_task(data, "train") if data in TASKS else pd.read_csv(data, usecols=["clip_input", "t5_trg"], dtype=str)
    tok = AutoTokenizer.from_pretrained(init)
    ds = Dataset.from_pandas(df[["clip_input", "t5_trg"]]).map(
        lambda b: tok(b["clip_input"], text_target=b["t5_trg"], truncation=True, max_length=512),
        batched=True,
        remove_columns=["clip_input", "t5_trg"],
    )
    args = Seq2SeqTrainingArguments(
        out,
        num_train_epochs=3,
        max_steps=int(os.environ.get("MAX_STEPS", -1)),
        per_device_train_batch_size=(16 if data in TASKS else 8) // int(os.environ.get("WORLD_SIZE", 1)),
        optim="adafactor",
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        fp16=True,
        save_strategy="no",
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(init)
    trainer = Seq2SeqTrainer(
        model, args, data_collator=DataCollatorForSeq2Seq(tok, model), train_dataset=ds, processing_class=tok
    )
    trainer.train()
    trainer.save_model()
