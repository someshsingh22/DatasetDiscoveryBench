"""Checks validate.py on our mock inputs and 50 sampled TL;DR rows, train.py's preprocessing against the authors'
preprocessed dataset, and smoke-runs run.sh --steps 2 end to end (needs a GPU).

Run from the environment directory: uv run python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from datasets import load_from_disk

from src.summarize_from_feedback_details.sft import Args
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
        cols = ["id", "query_reference_response_token"]
        theirs = load_dataset(Args.query_dataset, split="train").select_columns(cols).to_pandas().set_index("id")
        with tempfile.TemporaryDirectory() as out:
            subprocess.run([sys.executable, ENV / "train.py", SAMPLE, "EleutherAI/pythia-1b-deduped", out], check=True)
            ours = load_from_disk(out)["query_reference_response_token"]
        ids = pd.read_csv(SAMPLE, dtype=str, keep_default_na=False).id
        self.assertEqual(ours, [list(theirs.query_reference_response_token[i]) for i in ids])

    def test_run_sh(self):
        with tempfile.TemporaryDirectory() as out:
            env = {**os.environ, "MODEL": "EleutherAI/pythia-70m-deduped"}
            subprocess.run(["bash", ENV / "run.sh", SAMPLE, out, "--steps", "2"], env=env, check=True)
            scores = json.loads(Path(out, "scores.json").read_text())
        self.assertEqual(set(scores["primary"]), {"rouge1", "rouge2", "rougeL"})
        self.assertTrue(all(0 <= v <= 1 for v in scores["primary"].values()))
        self.assertEqual(scores["metrics"]["n"], 11490)
