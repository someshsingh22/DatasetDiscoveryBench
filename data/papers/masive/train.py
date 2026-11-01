"""Fine-tunes an mT5 checkpoint on a submission with the MASIVE recipe (Deas et al., EMNLP 2024; masive_train.py).

Usage: train.py INIT SUBMISSION.csv OUT
  clip_input -> t5_trg, effective batch 8, 3 epochs, Adafactor, lr 1e-4 linear, weight decay 0.01, bf16.
  MAX_STEPS shortens training (smoke tests only).
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

if __name__ == "__main__":
    init, data, out = sys.argv[1:]
    df = pd.read_csv(data, usecols=["clip_input", "t5_trg"], dtype=str)
    tok = AutoTokenizer.from_pretrained(init)
    ds = Dataset.from_pandas(df).map(
        lambda b: tok(b["clip_input"], text_target=b["t5_trg"], truncation=True, max_length=512),
        batched=True,
        remove_columns=["clip_input", "t5_trg"],
    )
    args = Seq2SeqTrainingArguments(
        out,
        num_train_epochs=3,
        max_steps=int(os.environ.get("MAX_STEPS", -1)),
        per_device_train_batch_size=8 // int(os.environ.get("WORLD_SIZE", 1)),
        optim="adafactor",
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        bf16=True,  # the authors used fp16, which overflows in mT5 under current transformers (every step skipped)
        save_strategy="no",
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(init)
    trainer = Seq2SeqTrainer(
        model, args, data_collator=DataCollatorForSeq2Seq(tok, model), train_dataset=ds, processing_class=tok
    )
    assert 0 < trainer.train().training_loss < float("inf"), "training diverged (NaN/inf loss or no optimizer steps)"
    trainer.save_model()
