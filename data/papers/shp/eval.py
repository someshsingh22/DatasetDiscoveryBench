"""Scores a FLAN-T5 preference model and its null model with the authors' v_info.py and writes OUT/scores.json.

Rows are SteamSHP inputs (train.steamshp; pairs that do not fit 500 tokens are skipped, as in SteamSHP's evaluation)
with gold label "A" or "B". v_info.py gives each row PVI = log2 P_model(gold | x) - log2 P_null(gold | " ") (bits) and
whether the model's more likely label is gold.
- hh_rlhf: the HH-RLHF helpful-base test pairs (SteamSHP's "anthropic (helpfulness)" test set), each in both orders
  (chosen as A, then as B); acc = mean correctness, v_info = mean PVI, i.e. the information about HH-RLHF's labels that
  the submission makes usable.
- heldout: the same on the submission's held-out posts (train.heldout), i.e. the submission's own V-usable information
  as in the paper; not primary, since the submission sets its own labels.

primary = {hh_rlhf_acc, hh_rlhf_v_info}; compare each separately, higher is better.

Usage: eval.py MODEL NULL_MODEL SUBMISSION.csv|none OUT
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from train import heldout
from train import steamshp
from v_info import v_info

HH = Path(__file__).parent / "external_eval_data" / "hh_rlhf_helpful_base_test.jsonl.gz"
TURN = "\n\nAssistant:"


def load_hh() -> pd.DataFrame:
    """HH-RLHF test pairs in both orders as submission rows: the dialogue before the last assistant turn and the two last
    turns. Pairs whose dialogues differ or whose last turns are identical are dropped (11 of 2,354)."""
    pairs = [
        (c[:i], c[i + len(TURN) :], r[j + len(TURN) :])
        for c, r in pd.read_json(HH, lines=True).itertuples(index=False)
        for i, j in [(c.rfind(TURN), r.rfind(TURN))]
        if c[:i] == r[:j] and c[i:] != r[j:]
    ]
    h, c, r = map(list, zip(*pairs))
    return pd.DataFrame(
        {"history": h * 2, "human_ref_A": c + r, "human_ref_B": r + c, "labels": ["1"] * len(c) + ["0"] * len(c)}
    )


def score(model: str, null: str, df: pd.DataFrame, out: str, name: str) -> dict:
    """Runs v_info.py on the rows of `df` that fit and returns their accuracy, V-information and counts."""
    tok = AutoTokenizer.from_pretrained(model)
    x = pd.DataFrame({"sentence1": steamshp(tok, df.history, df.human_ref_A, df.human_ref_B)})
    x = x.assign(label=list(df.labels.map({"1": "A", "0": "B"}))).dropna()
    x.to_csv(f"{out}/{name}.csv", index=False)
    x.assign(sentence1=" ").to_csv(f"{out}/{name}_null.csv", index=False)
    pvi = v_info(f"{out}/{name}.csv", model, f"{out}/{name}_null.csv", null, model, out_fn=f"{out}/{name}_pvi.csv")
    return {"acc": pvi.correct_yx.mean(), "v_info": pvi.PVI.mean(), "n": len(pvi), "skipped": len(df) - len(pvi)}


if __name__ == "__main__":
    model, null, data, out = sys.argv[1:]
    metrics = {"hh_rlhf": score(model, null, load_hh(), out, "hh_rlhf")}
    if data != "none":
        df = pd.read_csv(data, dtype=str, keep_default_na=False)
        metrics["heldout"] = score(model, null, df[heldout(df)], out, "heldout")
    primary = {"hh_rlhf_acc": metrics["hh_rlhf"]["acc"], "hh_rlhf_v_info": metrics["hh_rlhf"]["v_info"]}
    Path(out, "scores.json").write_text(json.dumps({"primary": primary, "metrics": metrics}, indent=2))
