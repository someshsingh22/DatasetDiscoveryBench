"""Checks validate.py on our mock inputs and 50 sampled TL;DR rows, train.py's preprocessing against the authors'
preprocessed dataset, and smoke-runs run.sh end to end (needs a GPU).

Run from the environment directory: uv run python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from train import TLDR
from train import Args
from train import process_query_data
from validate import test_ngrams
from validate import validate

ENV = Path(__file__).resolve().parents[1]
SAMPLE = ENV / "tests" / "sample.csv"
OK = {
    "subreddit": "relationships",
    "title": "My [25F] best friend [26F] forgot my birthday again",
    "post": "Third year in a row.\n\nShould I say something?",
    "summary": "best friend forgot my birthday for the third time, do I bring it up?",
}


def validate_rows(rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".csv") as f:
        pd.DataFrame(rows).to_csv(f.name, index=False)
        validate(f.name)


class TestTldr(unittest.TestCase):
    def test_valid_inputs_pass(self):
        validate(str(SAMPLE))
        validate_rows([OK])

    def test_invalid_inputs_fail(self):
        article, reference = list(test_ngrams())[:2]
        for rows in (
            [{k: v for k, v in OK.items() if k != "summary"}],
            [OK, {**OK, "post": ""}],
            [OK, {**OK, "summary": reference}],
            [OK, {**OK, "post": f"Saw this on CNN:\n\n{article}"}],
        ):
            with self.subTest(rows=rows), self.assertRaises((AssertionError, ValueError)):
                validate_rows(rows)

    def test_preprocessing_matches_authors(self):
        tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped", padding_side="right")
        tok.add_special_tokens({"pad_token": "[PAD]"})
        TLDR.pad_token = [tok.pad_token_id]
        cols = ["id", "query_reference_response_token"]
        theirs = load_dataset(Args.query_dataset, split="train").select_columns(cols).to_pandas().set_index("id")
        for row in pd.read_csv(SAMPLE, dtype=str, keep_default_na=False).to_dict("records"):
            ours = process_query_data(row, tok)["query_reference_response_token"]
            self.assertEqual(ours, list(theirs.query_reference_response_token[row["id"]]))

    def test_run_sh(self):
        with tempfile.TemporaryDirectory() as out:
            env = {**os.environ, "MODEL": "EleutherAI/pythia-70m-deduped"}
            subprocess.run(["bash", ENV / "run.sh", SAMPLE, out], env=env, check=True)
            scores = json.loads(Path(out, "scores.json").read_text())
        self.assertEqual(set(scores["primary"]), {"rouge1", "rouge2", "rougeL"})
        self.assertTrue(all(0 <= v <= 1 for v in scores["primary"].values()))
        self.assertEqual(scores["metrics"]["n"], 11490)
