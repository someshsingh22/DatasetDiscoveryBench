"""Checks validate.py on our mock inputs and 50 sampled SHP rows, and smoke-runs run.sh end to end (needs a GPU).

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

from eval import load_hh
from validate import validate

ENV = Path(__file__).resolve().parents[1]
SAMPLE = ENV / "tests" / "sample.csv"
OK = {"history": "How do I keep basil fresh?", "human_ref_A": "Stand it in water.", "human_ref_B": "Freeze it whole."}
POSTS = [{**OK, "post_id": f"p{i}", "labels": str(i % 2)} for i in range(20)]


def validate_rows(rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".csv") as f:
        pd.DataFrame(rows).to_csv(f.name, index=False)
        validate(f.name)


class TestShp(unittest.TestCase):
    def test_valid_inputs_pass(self):
        validate(str(SAMPLE))
        validate_rows(POSTS)

    def test_invalid_inputs_fail(self):
        leaked = next(text for text in load_hh().human_ref_A if len(text.split()) >= 8)
        for rows in (
            POSTS[:19],
            [{k: v for k, v in row.items() if k != "post_id"} for row in POSTS],
            [{**row, "labels": "A" if row["labels"] == "1" else "B"} for row in POSTS],
            [{**row, "labels": "1"} for row in POSTS],
            [*POSTS, {**POSTS[0], "human_ref_B": f"Someone wrote: {leaked}"}],
        ):
            with self.subTest(rows=rows[-1]), self.assertRaises((AssertionError, ValueError)):
                validate_rows(rows)

    def test_run_sh(self):
        with tempfile.TemporaryDirectory() as out:
            env = {**os.environ, "MODEL": "google/flan-t5-small", "MAX_STEPS": "2"}
            subprocess.run(["bash", ENV / "run.sh", SAMPLE, out], env=env, check=True)
            scores = json.loads(Path(out, "scores.json").read_text())
        self.assertEqual(set(scores["primary"]), {"hh_rlhf_acc", "hh_rlhf_v_info"})
        self.assertEqual(set(scores["metrics"]), {"hh_rlhf", "heldout"})
        self.assertEqual(scores["metrics"]["hh_rlhf"]["n"] + scores["metrics"]["hh_rlhf"]["skipped"], 2 * 2343)
        self.assertTrue(0 <= scores["primary"]["hh_rlhf_acc"] <= 1)
