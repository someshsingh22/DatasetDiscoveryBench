"""Checks validate.py on our mock inputs and 50 sampled MASIVE rows, and smoke-runs run.sh end to end (needs a GPU).

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

from train import load_task
from validate import validate

ENV = Path(__file__).resolve().parents[1]
SAMPLE = ENV / "tests" / "sample.csv"
OK = {"clip_input": "Long week at work. I feel <extra_id_0> but proud.", "t5_trg": "<extra_id_0> drained"}


def validate_rows(rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".csv") as f:
        pd.DataFrame(rows).to_csv(f.name, index=False)
        validate(f.name)


class TestMasive(unittest.TestCase):
    def test_valid_inputs_pass(self):
        validate(str(SAMPLE))
        validate_rows([OK, {"clip_input": "<extra_id_0> and <extra_id_1>", "t5_trg": "<extra_id_0> a <extra_id_1> b"}])

    def test_invalid_inputs_fail(self):
        test = load_task("emoevent_en", "test").text
        leaked = test[test.str.split().str.len() >= 8].iloc[0]
        for rows in (
            [{"clip_input": OK["clip_input"]}],
            [{**OK, "clip_input": "I feel drained."}],
            [{**OK, "t5_trg": "drained"}],
            [{**OK, "t5_trg": ""}],
            [OK, {**OK, "clip_input": f"Saw this today: {leaked} I feel <extra_id_0>"}],
        ):
            with self.subTest(rows=rows), self.assertRaises((AssertionError, ValueError)):
                validate_rows(rows)

    def test_run_sh(self):
        with tempfile.TemporaryDirectory() as out:
            env = {**os.environ, "MODEL": "google/mt5-small", "MAX_STEPS": "2"}
            subprocess.run(["bash", ENV / "run.sh", SAMPLE, out], env=env, check=True)
            scores = json.loads(Path(out, "scores.json").read_text())
        self.assertEqual(set(scores["metrics"]), {"emoevent_en", "goemo", "goemo_full"})
        self.assertTrue(0 <= scores["primary"] <= 1)
