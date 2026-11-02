"""Checks our wrappers (validate.py, train.py, eval.py); the authors' code is checked by full runs on the cluster.

Run from the environment directory: uv run python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from convokit import Corpus

from validate import benchmark_texts
from validate import validate

ENV = Path(__file__).resolve().parents[1]
SAMPLE = ENV / "tests" / "sample.csv"
OP = {"id": "t3_a", "reply_to": "", "user": "alice", "title": "CMV: Cats beat dogs", "text": "Cats are tidier."}
OK = [OP, {"id": "t1_b", "reply_to": "t3_a", "user": "bob", "title": "", "text": "> tidier\n\nBut dogs are loyal."}]


def validate_rows(rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".csv") as f:
        pd.DataFrame(rows).to_csv(f.name, index=False)
        validate(f.name)


class TestSmlm(unittest.TestCase):
    def test_valid_inputs_pass(self):
        validate(str(SAMPLE))
        validate_rows(OK)

    def test_invalid_inputs_fail(self):
        leak = next(text for text in benchmark_texts() if len(text.split()) >= 8)
        for rows in (
            [{k: v for k, v in OP.items() if k != "user"}],
            [OP, {**OK[1], "id": "t3_a"}],
            [OP, {**OK[1], "user": ""}],
            [{**OP, "reply_to": "t1_b"}, OK[1]],
            [OP, {**OK[1], "text": f"Quoting the other thread: {leak}"}],
        ):
            with self.subTest(rows=rows), self.assertRaises((AssertionError, ValueError)):
                validate_rows(rows)

    def test_train_writes_the_corpus(self):
        orphan = {**OK[1], "id": "t1_c", "reply_to": "t1_gone"}
        with tempfile.TemporaryDirectory() as out:
            pd.DataFrame([*OK, orphan]).to_csv(f"{out}/submission.csv", index=False)
            subprocess.run([sys.executable, ENV / "train.py", f"{out}/submission.csv", out], check=True)
            corpus = Corpus(filename=f"{out}/corpus")
        threads = {
            c.meta["op-title"]: [[u.text for u in p] for p in c.get_root_to_leaf_paths()]
            for c in corpus.iter_conversations()
        }
        reply = "&gt; tidier\n\nBut dogs are loyal."
        self.assertEqual(threads, {"CMV: Cats beat dogs": [["Cats are tidier.", reply]], "": [[reply]]})

    def test_eval_scores_the_logs(self):
        with tempfile.TemporaryDirectory() as out:
            Path(out, "logs").mkdir()
            for task in ("drinv_aci_80", "cmv_aci_80", "rtp_80", "rtp_50"):
                for run in range(5):
                    epochs = [
                        {"overall_f1": 0.1 * e} if "aci" in task else {"weighted_avg": {"f1": 0.1 * e}}
                        for e in range(8)
                    ]
                    Path(out, f"logs/{task}_{run}.log").write_text("".join(f"RUN\n\t\t\t\t {e}\n" for e in epochs))
            subprocess.run([sys.executable, ENV / "eval.py", out], check=True)
            scores = json.loads(Path(out, "scores.json").read_text())
        self.assertEqual(set(scores["primary"]), {"drinv_aci_80_f1", "cmv_aci_80_f1", "rtp_80_f1", "rtp_50_f1"})
        for f1 in scores["primary"].values():
            self.assertAlmostEqual(f1, 0.5)  # the mean of the last 5 epochs, 0.3 ... 0.7
