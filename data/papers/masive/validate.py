"""Rejects a submission that is not in MASIVE's format or that contains an evaluation test text.

Usage: validate.py SUBMISSION.csv
"""

from __future__ import annotations

import re
import sys

import pandas as pd

from eval import MASIVE_TEST
from eval import TASKS
from eval import load_benchmark

MASK = re.compile(r"<extra_id_\d+>")


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = MASK.sub(" ", text).lower().split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def validate(path: str) -> None:
    """Raises AssertionError or ValueError if `path` is not a valid submission."""
    df = pd.read_csv(path, usecols=["clip_input", "t5_trg"], dtype=str)
    assert len(df) and df.notna().all().all(), "clip_input and t5_trg must be non-empty"
    assert df.clip_input.str.contains("<extra_id_0>", regex=False).all(), "every clip_input needs an <extra_id_0> mask"
    assert df.t5_trg.str.startswith("<extra_id_0>").all(), "every t5_trg must start with <extra_id_0>"
    # A test text leaks if 80% of its 8-grams (masks ignored) occur in the submission: copies of a post differ in their
    # masks and clipping, and a lower bar flags quotes. Texts under 8 words are too generic to flag.
    tests = [load_benchmark(task).text for task in TASKS] + [pd.read_csv(MASIVE_TEST, dtype=str).clip_input]
    tests = {text: g for text in pd.concat(tests) if (g := ngrams(text))}
    pool = set().union(*tests.values())
    seen = {g for text in df.clip_input for g in ngrams(text) if g in pool}
    leaked = [text for text, g in tests.items() if 5 * len(g & seen) >= 4 * len(g)]
    assert not leaked, f"{len(leaked)} evaluation test texts appear in the submission, e.g. {leaked[0]!r}"


if __name__ == "__main__":
    validate(sys.argv[1])
