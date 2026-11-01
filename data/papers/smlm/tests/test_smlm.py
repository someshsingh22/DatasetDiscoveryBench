"""Checks validate.py on our mock inputs and 50 sampled Winning Arguments rows, and smoke-runs run.sh (needs a GPU).

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

from train import DATA
from train import TASKS
from train import load_xml
from validate import validate

ENV = Path(__file__).resolve().parents[1]
SAMPLE = ENV / "tests" / "sample.csv"
OP = {"id": "t3_a", "reply_to": "", "user": "alice", "title": "CMV: Cats beat dogs", "text": "Cats are tidier."}
OK = [OP, {"id": "t1_b", "reply_to": "t3_a", "user": "bob", "title": "", "text": "&gt; tidier\n\nBut dogs are loyal."}]


def validate_rows(rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".csv") as f:
        pd.DataFrame(rows).to_csv(f.name, index=False)
        validate(f.name)


class TestSmlm(unittest.TestCase):
    def test_valid_inputs_pass(self):
        validate(str(SAMPLE))
        validate_rows(OK)

    def test_invalid_inputs_fail(self):
        replies = load_xml(sorted(DATA.glob("*/*.xml"))[0]).find_all("reply")
        reply = next(r.get_text() for r in replies if len(r.get_text().split()) >= 8)
        for rows in (
            [{k: v for k, v in OP.items() if k != "user"}],
            [OP, {**OK[1], "id": "t3_a"}],
            [OP, {**OK[1], "user": ""}],
            [OP, {**OK[1], "text": f"Quoting the other thread: {reply}"}],
        ):
            with self.subTest(rows=rows), self.assertRaises((AssertionError, ValueError)):
                validate_rows(rows)

    def test_run_sh(self):
        with tempfile.TemporaryDirectory() as out:
            subprocess.run(["bash", ENV / "run.sh", SAMPLE, out], env={**os.environ, "MAX_STEPS": "2"}, check=True)
            scores = json.loads(Path(out, "scores.json").read_text())
        self.assertEqual(set(scores["metrics"]), set(TASKS))
        self.assertEqual(set(scores["primary"]), {f"{task}_f1" for task in TASKS})
        self.assertTrue(all(0 <= v <= 1 for v in scores["primary"].values()))
