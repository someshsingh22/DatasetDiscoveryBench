"""Rejects a submission that is not in MASIVE's format or that contains a benchmark test text.

Usage: validate.py SUBMISSION.csv
"""

from __future__ import annotations

import sys

import pandas as pd

from train import DATA
from train import TASKS


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = text.lower().split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def validate(path: str) -> None:
    """Raises AssertionError or ValueError if `path` is not a valid submission."""
    df = pd.read_csv(path, usecols=["clip_input", "t5_trg"], dtype=str)
    assert len(df) and df.notna().all().all(), "clip_input and t5_trg must be non-empty"
    assert df.clip_input.str.contains("<extra_id_0>", regex=False).all(), "every clip_input needs an <extra_id_0> mask"
    assert df.t5_trg.str.startswith("<extra_id_0>").all(), "every t5_trg must start with <extra_id_0>"
    # A test text leaks if all its 8-grams occur in the submission; shorter texts are too generic to flag.
    tests = [
        pd.read_csv(DATA / f"{prefix}_test.csv", dtype=str, keep_default_na=False).text for prefix, _ in TASKS.values()
    ]
    tests = {text: g for text in pd.concat(tests) if (g := ngrams(text))}
    pool = set().union(*tests.values())
    seen = {g for text in df.clip_input for g in ngrams(text) if g in pool}
    leaked = [text for text, g in tests.items() if g <= seen]
    assert not leaked, f"{len(leaked)} benchmark test texts appear in the submission, e.g. {leaked[0]!r}"


if __name__ == "__main__":
    validate(sys.argv[1])
