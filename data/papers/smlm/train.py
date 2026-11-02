"""Writes a submission as the ConvoKit corpus that the authors' sMLM pre-training (src/.../pre_train.py) reads.

Usage: train.py SUBMISSION.csv OUT   (writes OUT/corpus; a post whose reply_to is empty or unknown starts a tree)
"""

from __future__ import annotations

import html
import sys

import pandas as pd
from convokit import Corpus
from convokit import Speaker
from convokit import Utterance


def reddit(text: str) -> str:
    """Escapes &, <, > as Reddit stores them, which the authors' regexes expect (e.g. "&gt;" opens a quote)."""
    return html.escape(html.unescape(text), quote=False)


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False).set_index("id")
    parent = dict(zip(df.index, df.reply_to.where(df.reply_to.isin(df.index), "")))
    root = {}
    for i in df.index:
        j = i
        while parent[j]:
            j = parent[j]
        root[i] = j
    speakers = {user: Speaker(id=user) for user in df.user.unique()}
    corpus = Corpus(
        utterances=[
            Utterance(
                id=i, speaker=speakers[r.user], conversation_id=root[i], reply_to=parent[i] or None, text=reddit(r.text)
            )
            for i, r in df.iterrows()
        ]
    )
    for conversation in corpus.iter_conversations():
        conversation.meta["op-title"] = reddit(df.title[conversation.id])
    corpus.dump("corpus", base_path=sys.argv[2])
