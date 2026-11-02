"""Checks validate.py on our mock inputs and 50 sampled JIC dialogues, and smoke-runs run.sh end to end (needs a GPU).

Run from the environment directory: uv run python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from eval import TRAITS
from validate import DATA
from validate import validate

ENV = Path(__file__).resolve().parents[1]
SAMPLE = json.loads((ENV / "tests" / "sample.json").read_text(encoding="utf-8")) * 21  # > 1,000 dialogues
OK = {"dialogue": ["I journaled about my week.", "Writing always calms me down."]}
LINES = (DATA / "openness.jsonl").read_text().splitlines()
STATEMENTS = [s for x in LINES if len((s := json.loads(x)["statement"]).split()) >= 8][:11]  # 11 > the 10 tolerated


def validate_rows(rows: list[dict]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump(rows, f)
        f.flush()
        validate(f.name)


class TestJic(unittest.TestCase):
    def test_valid_inputs_pass(self):
        validate_rows(SAMPLE)
        validate_rows([OK] * 1001)
        validate_rows([OK] * 1000 + [{"dialogue": ["What did you write today?", " ".join(STATEMENTS[:10])]}])

    def test_invalid_inputs_fail(self):
        for rows in (
            [OK] * 1000,
            [OK] * 1000 + [{"text": ["Hi.", "Hello."]}],
            [OK] * 1000 + [{"dialogue": "Hi. Hello."}],
            [OK] * 1000 + [{"dialogue": ["Hi."]}],
            [OK] * 1000 + [{"dialogue": ["Hi.", " "]}],
            [OK] * 1000 + [{"dialogue": ["What did you write today?", f"Just this: {' '.join(STATEMENTS).upper()}!"]}],
        ):
            with self.subTest(row=rows[-1]), self.assertRaises((AssertionError, KeyError)):
                validate_rows(rows)

    def test_run_sh(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "sub.json").write_text(json.dumps(SAMPLE))
            env = {**os.environ, "MODEL": "hf-internal-testing/tiny-random-MistralForCausalLM", "MAX_STEPS": "2"}
            subprocess.run(["bash", ENV / "run.sh", Path(tmp, "sub.json"), Path(tmp, "out")], env=env, check=True)
            scores = json.loads(Path(tmp, "out", "scores.json").read_text())
        self.assertEqual(set(scores["primary"]), set(TRAITS))
        self.assertTrue(all(0 <= v <= 1 for v in scores["primary"].values()))
