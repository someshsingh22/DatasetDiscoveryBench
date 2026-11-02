"""Rejects a submission that is not a forest of discussion posts or that contains an external test text.

Usage: validate.py SUBMISSION.csv
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

DATA = Path(__file__).parent / "external_eval_data"


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = html.unescape(text).lower().split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def benchmark_texts() -> list[str]:
    """CMV Modes titles and posts (bare "&"s escaped first, or the XML parser drops what follows) and Dr Inventor papers."""
    xmls = [re.sub(r"&(?!#?\w+;)", "&amp;", p.read_text()) for p in DATA.glob("change-my-view-modes/v2.0/*/*.xml")]
    posts = [node.get_text() for x in xmls for node in BeautifulSoup(x, "xml").find_all(["title", "OP", "reply"])]
    return posts + [p.read_text() for p in DATA.glob("compiled_corpus/*.txt")]


def validate(path: str) -> None:
    """Raises AssertionError or ValueError if `path` is not a valid submission."""
    df = pd.read_csv(path, usecols=["id", "reply_to", "user", "title", "text"], dtype=str, keep_default_na=False)
    assert len(df) and (df.id != "").all() and (df.user != "").all(), "id and user must be non-empty"
    assert df.id.is_unique, "ids must be unique"
    parent, done = dict(zip(df.id, df.reply_to)), set()
    for i in df.id:
        chain = set()
        while i in parent and i not in done:
            assert i not in chain, f"reply_to chains must end at a tree's first post, not loop back to {i}"
            chain.add(i)
            i = parent[i]
        done |= chain
    # A test text leaks if all its 8-grams occur in the submission; shorter texts are too generic.
    tests = {text: g for text in benchmark_texts() if (g := ngrams(text))}
    pool = set().union(*tests.values())
    seen = {g for text in pd.concat([df.title, df.text]) for g in ngrams(text) if g in pool}
    leaked = [text for text, g in tests.items() if g <= seen]
    assert not leaked, f"{len(leaked)} external test texts appear in the submission, e.g. {leaked[0][:200]!r}"


if __name__ == "__main__":
    validate(sys.argv[1])
