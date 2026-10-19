"""Rejects a submission that is not a table of discussion posts or that contains a CMV Modes text.

Usage: validate.py SUBMISSION.csv
"""

from __future__ import annotations

import html
import sys

import pandas as pd

from train import COLUMNS
from train import DATA
from train import load_xml


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = html.unescape(text).lower().split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def validate(path: str) -> None:
    """Raises AssertionError or ValueError if `path` is not a valid submission."""
    df = pd.read_csv(path, usecols=COLUMNS, dtype=str, keep_default_na=False)
    assert len(df) and (df.id != "").all() and (df.user != "").all(), "id and user must be non-empty"
    assert df.id.is_unique, "ids must be unique"
    # A test text (title or post) leaks if all its 8-grams occur in the submission; shorter texts are too generic.
    tests = [node.get_text() for p in DATA.glob("*/*.xml") for node in load_xml(p).find_all(["title", "OP", "reply"])]
    tests = {text: g for text in tests if (g := ngrams(text))}
    pool = set().union(*tests.values())
    seen = {g for text in pd.concat([df.title, df.text]) for g in ngrams(text) if g in pool}
    leaked = [text for text, g in tests.items() if g <= seen]
    assert not leaked, f"{len(leaked)} CMV Modes test texts appear in the submission, e.g. {leaked[0]!r}"


if __name__ == "__main__":
    validate(sys.argv[1])
