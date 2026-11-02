"""Rejects a submission that is not in SHP's format or that contains an HH-RLHF test text.

Usage: validate.py SUBMISSION.csv
"""

from __future__ import annotations

import sys

import pandas as pd

from eval import load_hh

COLUMNS = ["post_id", "history", "human_ref_A", "human_ref_B", "labels"]


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = text.lower().split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def validate(path: str) -> None:
    """Raises AssertionError or ValueError if `path` is not a valid submission."""
    df = pd.read_csv(path, usecols=COLUMNS, dtype=str, keep_default_na=False)
    assert df.post_id.nunique() >= 20, "at least 20 posts are needed (5% are held out)"
    assert df.labels.isin(["0", "1"]).all(), "labels must be 1 (A preferred) or 0 (B preferred)"
    assert abs((df.labels == "1").mean() - 0.5) <= 2 / len(df) ** 0.5, "the A/B order must be random (labels ~50/50)"
    # A test text (dialogue or last turn) leaks if all its 8-grams occur in the submission; shorter texts are too generic.
    hh = load_hh()
    tests = {text: g for text in pd.concat([hh.history, hh.human_ref_A]).unique() if (g := ngrams(text))}
    pool = set().union(*tests.values())
    texts = pd.concat([df.history, df.human_ref_A, df.human_ref_B]).unique()
    seen = {g for text in texts for g in ngrams(text) if g in pool}
    leaked = [text for text, g in tests.items() if g <= seen]
    assert not leaked, f"{len(leaked)} HH-RLHF test texts appear in the submission, e.g. {leaked[0]!r}"


if __name__ == "__main__":
    validate(sys.argv[1])
