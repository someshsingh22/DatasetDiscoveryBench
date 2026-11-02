"""Fine-tunes FLAN-T5 to pick the preferred response, as SteamSHP was trained (stanfordnlp/SteamSHP-flan-t5-large).

Usage: train.py INIT SUBMISSION.csv OUT model|null [--epochs K | --steps N]
  The submission's pairs, minus the held-out posts (heldout), in the SteamSHP format -> "A" or "B". Hyperparameters from
  SteamSHP-Large's training_args.bin: 1 epoch, AdamW lr 5e-5 linear, no warmup or weight decay, batch 2 per device (16 on
  the 8 GPUs assumed), seed 42; bf16 instead of their fp32. `null` trains v_info.py's null model: every input is " ".
  --epochs K or --steps N (optimizer steps at batch 16) replaces the 1 epoch.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

TEMPLATE = "POST: {}\n\n RESPONSE A: {}\n\n RESPONSE B: {}\n\n Which response is better? RESPONSE"


def heldout(df: pd.DataFrame) -> pd.Series:
    """Marks the rows of 5% of the posts (random_state=0), held out to estimate the submission's V-information."""
    return df.post_id.isin(df.post_id.drop_duplicates().sample(frac=0.05, random_state=0))


def steamshp(tok, history, a, b) -> list[str | None]:
    """SteamSHP inputs (fields without newlines), crammed under 500 tokens by cutting the post's end (SteamSHP card);
    None where the two responses alone do not fit."""
    history, a, b = ([" ".join(t.split()) for t in col] for col in (history, a, b))
    budget = [500 - len(ids) for ids in tok([TEMPLATE.format("", x, y) for x, y in zip(a, b)]).input_ids]
    posts = tok(history, add_special_tokens=False).input_ids
    return [
        TEMPLATE.format(h if len(p) <= n else tok.decode(p[:n]), x, y) if n >= 0 else None
        for h, p, n, x, y in zip(history, posts, budget, a, b)
    ]


def seq2seq_classifier(model, tokenizer, labels=("A", "B")):
    """Stands in for v_info.py's pipeline('text-classification'): P(label) from the first decoder step, softmax over
    `labels`."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(model).cuda().eval()
    ids = [tokenizer(label).input_ids[0] for label in labels]

    @torch.no_grad()
    def classify(texts):
        enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        start = torch.full((len(texts), 1), model.config.decoder_start_token_id, device="cuda")
        probs = model(**enc, decoder_input_ids=start).logits[:, 0, ids].float().softmax(-1).tolist()
        return [[{"label": label, "score": p} for label, p in zip(labels, row)] for row in probs]

    return classify


if __name__ == "__main__":
    init, data, out, kind = sys.argv[1:5]
    length = dict(zip(sys.argv[5::2], sys.argv[6::2]))
    df = pd.read_csv(data, dtype=str, keep_default_na=False)
    df = df[~heldout(df)]
    tok = AutoTokenizer.from_pretrained(init)
    df = df.assign(x=steamshp(tok, df.history, df.human_ref_A, df.human_ref_B), y=df.labels.map({"1": "A", "0": "B"}))
    df = df.dropna(subset=["x"])
    print("DDB_TRAIN pairs", len(df), sys.argv[4:], flush=True)
    ds = Dataset.from_dict({"x": [" "] * len(df) if kind == "null" else list(df.x), "y": list(df.y)})
    ds = ds.map(lambda b: tok(b["x"], text_target=b["y"]), batched=True, remove_columns=["x", "y"])
    args = Seq2SeqTrainingArguments(
        out,
        num_train_epochs=float(length.get("--epochs", 1)),
        max_steps=int(length.get("--steps", -1)),
        per_device_train_batch_size=16 // int(os.environ.get("WORLD_SIZE", 1)),
        bf16=True,
        logging_steps=100,
        save_strategy="no",
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(init)
    trainer = Seq2SeqTrainer(
        model, args, data_collator=DataCollatorForSeq2Seq(tok, model), train_dataset=ds, processing_class=tok
    )
    assert 0 < trainer.train().training_loss < float("inf"), "training diverged (NaN/inf loss or no optimizer steps)"
    trainer.save_model()
