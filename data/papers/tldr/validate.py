"""Rejects a submission that is not in the filtered Reddit TL;DR format or that contains a CNN/DM test text.

Usage: validate.py SUBMISSION.csv
"""

from __future__ import annotations

import functools
import sys

import pandas as pd

from cnndm import cnndm_generator
from train import COLUMNS


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = text.lower().split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


@functools.cache
def test_ngrams() -> dict[str, set[tuple[str, ...]]]:
    """Returns the 8-grams of each CNN/DM test article and reference summary; shorter texts are too generic to flag."""
    texts = (text for ex in cnndm_generator("test") for text in (ex["article"], ex["reference"]))
    return {text: g for text in texts if (g := ngrams(text))}


def validate(path: str) -> None:
    """Raises AssertionError or ValueError if `path` is not a valid submission."""
    df = pd.read_csv(path, usecols=COLUMNS, dtype=str, keep_default_na=False)
    assert len(df) and (df != "").all().all(), f"{', '.join(COLUMNS)} must be non-empty"
    # A test text leaks if all its 8-grams occur in the submission.
    tests = test_ngrams()
    pool = set().union(*tests.values())
    seen = {g for text in pd.concat([df.title, df.post, df.summary]) for g in ngrams(text) if g in pool}
    leaked = [text for text, g in tests.items() if g <= seen]
    assert not leaked, f"{len(leaked)} CNN/DM test texts appear in the submission, e.g. {leaked[0][:300]!r}"


if __name__ == "__main__":
    validate(sys.argv[1])
