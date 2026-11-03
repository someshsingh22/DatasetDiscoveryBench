"""Scores a TL;DR summarizer on CNN/DM news articles, without further training, and writes OUT/scores.json.

The transfer of Stiennon et al. (NeurIPS 2020, Fig. 14b): the 11,490 CNN/DM test articles and references of OpenAI's
cnndm.py; each query is "{article}\\n\\nTL;DR:", its article cut at the last newline until it fits 512 tokens (OpenAI's
tldr_zero_shot_cnndm_task, run by tldr_dataset.py's process_query) and left-padded with masked [PAD] tokens as in the
re-implementation's sft.py evaluation; greedy (T=0) summaries of at most 53 tokens (sft.py's response_length) stop at
<|endoftext|> and are cut at their first newline. The cut is ours: OpenAI's transfer task stops only at <|endoftext|>,
but its zero-shot tasks stop at "\\n"; without it, a model that never learned to stop fills all 53 tokens, which ROUGE
F1 rewards. primary = {rouge1, rouge2, rougeL}: mean F1 against the references (rouge_score, stemmed); compare each
separately, higher is better. metrics adds rouge_mean (the paper's "ROUGE": the mean of the three) and the mean summary
length in tokens; OUT/summaries.json holds the summaries.

Usage: eval.py MODEL OUT   (MODEL: an sft.py output dir or a Hugging Face model id)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from src.summarize_from_feedback.datasets.cnndm import cnndm_generator
from src.summarize_from_feedback_details.tldr_dataset import TaskQueryHParams
from src.summarize_from_feedback_details.tldr_dataset import process_query

QUERY = TaskQueryHParams(
    length=512, format_str="{article}\n\nTL;DR:", truncate_field="article", truncate_text="\n", pad_side="left"
)
RESPONSE_LENGTH = 53
ROUGE = ["rouge1", "rouge2", "rougeL"]


@torch.no_grad()
def summarize(model, tok, articles: list[str], batch_size: int = 128) -> list[str]:
    """Returns the greedy TL;DR summary of each article, up to its first newline."""
    QUERY.pad_token = [tok.pad_token_id]
    queries = torch.tensor([process_query({"article": a}, encoder=tok, hparams=QUERY)["query_token"] for a in articles])
    summaries = []
    for batch in queries.split(batch_size):
        mask = batch != tok.pad_token_id
        out = model.generate(
            input_ids=batch.masked_fill(~mask, 0).cuda(),
            attention_mask=mask.cuda(),
            max_new_tokens=RESPONSE_LENGTH,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        texts = tok.batch_decode(out[:, QUERY.length :], skip_special_tokens=True)
        summaries += [s.strip().split("\n")[0] for s in texts]
    return summaries


if __name__ == "__main__":
    model_dir, out = sys.argv[1:]
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.add_special_tokens({"pad_token": "[PAD]"})  # as sft.py, whose checkpoints already have it
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).cuda().eval()
    test = list(cnndm_generator("test"))
    summaries = summarize(model, tok, [ex["article"] for ex in test])
    scorer = rouge_scorer.RougeScorer(ROUGE, use_stemmer=True)
    scores = [scorer.score(ex["reference"], summary) for ex, summary in zip(test, summaries)]
    primary = {k: sum(s[k].fmeasure for s in scores) / len(scores) for k in ROUGE}
    metrics = {
        **primary,
        "rouge_mean": sum(primary.values()) / len(ROUGE),
        "summary_tokens": sum(len(tok.encode(s)) for s in summaries) / len(summaries),
        "n": len(test),
    }
    Path(out, "scores.json").write_text(json.dumps({"primary": primary, "metrics": metrics}, indent=2))
    Path(out, "summaries.json").write_text(json.dumps(dict(zip([ex["id"] for ex in test], summaries)), indent=0))
