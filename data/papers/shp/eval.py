"""Scores a FLAN-T5 preference model with the authors' v_info.py and writes OUT/scores.json.

Rows are SteamSHP inputs (train.steamshp; pairs that do not fit 500 tokens are skipped, as in SteamSHP's evaluation)
with gold label "A" or "B"; v_entropy gives each row -log2 P_model(gold | x) (bits) and whether the more likely label is
gold.
- hh_rlhf: the HH-RLHF helpful-base test pairs (SteamSHP's "anthropic (helpfulness)" test set), each in both orders
  (chosen as A, then as B). acc = mean correctness; v_info = H_V(Y) - H_V(Y | X) with H_V(Y) = 1 bit, exact since the
  two orders balance the labels: the information about HH-RLHF's labels that the submission makes usable.
- heldout: v_info.py on the submission's held-out posts (train.heldout), H_V(Y) from the null model: the submission's own
  V-usable information, as in the paper; not primary, since the submission sets its own labels.

primary = {hh_rlhf_acc, hh_rlhf_v_info}; compare each separately, higher is better.

Usage: eval.py MODEL NULL_MODEL|none SUBMISSION.csv|none OUT
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from train import heldout
from train import steamshp
from v_info import v_entropy
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


def inputs(model: str, df: pd.DataFrame, path: str) -> int:
    """Writes the SteamSHP inputs of `df` that fit, with labels "A"/"B", to PATH.csv and their null twins (" ") to
    PATH_null.csv; returns how many fit."""
    tok = AutoTokenizer.from_pretrained(model)
    x = pd.DataFrame({"sentence1": steamshp(tok, df.history, df.human_ref_A, df.human_ref_B)})
    x = x.assign(label=list(df.labels.map({"1": "A", "0": "B"}))).dropna()
    x.to_csv(f"{path}.csv", index=False)
    x.assign(sentence1=" ").to_csv(f"{path}_null.csv", index=False)
    return len(x)


if __name__ == "__main__":
    model, null, data, out = sys.argv[1:]
    hh = load_hh()
    n = inputs(model, hh, f"{out}/hh_rlhf")
    h, correct, _ = v_entropy(f"{out}/hh_rlhf.csv", model, model)
    metrics = {"hh_rlhf": {"acc": sum(correct) / n, "v_info": 1 - sum(h) / n, "n": n, "skipped": len(hh) - n}}
    if data != "none":
        df = pd.read_csv(data, dtype=str, keep_default_na=False)
        held = df[heldout(df)]
        n = inputs(model, held, f"{out}/heldout")
        pvi = v_info(f"{out}/heldout.csv", model, f"{out}/heldout_null.csv", null, model, f"{out}/heldout_pvi.csv")
        metrics["heldout"] = {"acc": pvi.correct_yx.mean(), "v_info": pvi.PVI.mean(), "n": n, "skipped": len(held) - n}
    primary = {"hh_rlhf_acc": metrics["hh_rlhf"]["acc"], "hh_rlhf_v_info": metrics["hh_rlhf"]["v_info"]}
    Path(out, "scores.json").write_text(json.dumps({"primary": primary, "metrics": metrics}, indent=2))
