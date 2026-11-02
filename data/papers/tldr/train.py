"""Preprocesses a submission for the re-implementation's SFT, as tldr_dataset.py does the authors' TL;DR data.

Writes OUT, a datasets.Dataset for src/summarize_from_feedback_details/sft.py --query_dataset=OUT. process_query_data
is re-hosted verbatim from vwxyzjn/summarize_from_feedback_details@ee77a9a tldr_dataset.py, whose __main__ block
defines it and cannot be imported; the TL;DR query parameters are its Args().tldr_params.

Usage: train.py SUBMISSION.csv MODEL OUT
"""

from __future__ import annotations

import copy
import multiprocessing
import sys

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from src.summarize_from_feedback_details.tldr_dataset import Args
from src.summarize_from_feedback_details.tldr_dataset import process_query

COLUMNS = ["subreddit", "title", "post", "summary"]

if __name__ == "__main__":
    args = Args(base_model=sys.argv[2])
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    args.tldr_params.pad_token = [tokenizer.pad_token_id]
    sft_ds = Dataset.from_pandas(pd.read_csv(sys.argv[1], usecols=COLUMNS, dtype=str, keep_default_na=False))

    # fmt: off
    def process_query_data(x):
        # the `x['summary']` in `vwxyzjn/summarize_from_feedback_tldr_3_filtered`
        # DOES NOT HAVE a leading space so we are adding the leading space and
        # `<|endoftext|>` token
        reference_response = f" {x['summary']}<|endoftext|>"
        y = {
            **process_query(x, encoder=tokenizer, hparams=args.tldr_params),
            "reference_response": reference_response,
            "reference_response_token": tokenizer.encode(
                reference_response,
                padding="max_length",
                max_length=args.tldr_params.max_sft_response_length,
                truncation=True,
            ),
            "reference_response_token_len": len(tokenizer.encode(reference_response)),
        }
        y["query_reference_response"] = y["query"].strip() + y["reference_response"]
        # if padding is space, then we can just concatenate the tokens
        if args.tldr_params.padding == "empty_space":
            y["query_reference_response_token"] = y["query_token"] + y["reference_response_token"]
        else:
            y["query_reference_response_token"] = tokenizer.encode(
                y["query_reference_response"],
                padding="max_length",
                max_length=args.tldr_params.max_sft_query_response_length,
                truncation=True,
            )
        y["query_reference_response_token_response_label"] = copy.deepcopy(y["query_reference_response_token"])
        unpadded_query_token = [token for token in y["query_token"] if token != tokenizer.pad_token_id]
        y["query_reference_response_token_response_label"][:len(unpadded_query_token)] = [tokenizer.pad_token_id for _ in range(len(unpadded_query_token))]
        y["query_reference_response_token_len"] = len(tokenizer.encode(y["query_reference_response"]))
        return y
    # fmt: on

    sft_ds = sft_ds.map(
        process_query_data, load_from_cache_file=False, num_proc=1 if args.debug else multiprocessing.cpu_count()
    )
    sft_ds.save_to_disk(sys.argv[3])
