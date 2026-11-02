"""Rejects a submission that is not in JIC's format or that contains a persona benchmark statement.

Usage: validate.py SUBMISSION.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).parent / "external_eval_data"


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def validate(path: str) -> None:
    """Raises AssertionError or KeyError if `path` is not a valid submission."""
    dialogues = [row["dialogue"] for row in json.loads(Path(path).read_text(encoding="utf-8"))]
    assert len(dialogues) > 1000, "need more than 1,000 dialogues (training holds 1,000 out for validation)"
    assert all(
        isinstance(d, list) and len(d) >= 2 and all(isinstance(u, str) and u.strip() for u in d) for d in dialogues
    ), "every dialogue must be a list of at least 2 non-empty utterances"
    # A statement leaks if all its word 8-grams occur in the submission; statements under 8 words are too generic.
    tests = {
        s: g
        for f in sorted(DATA.glob("*.jsonl"))
        for s in (json.loads(x)["statement"] for x in f.read_text(encoding="utf-8").splitlines())
        if (g := ngrams(s))
    }
    pool = set().union(*tests.values())
    seen = {g for d in dialogues for g in ngrams(" ".join(d)) if g in pool}
    leaked = [s for s, g in tests.items() if g <= seen]
    assert not leaked, f"{len(leaked)} persona benchmark statements appear in the submission, e.g. {leaked[0]!r}"


if __name__ == "__main__":
    validate(sys.argv[1])
