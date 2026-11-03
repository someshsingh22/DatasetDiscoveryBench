"""Scores a model on the Big Five persona tasks of lm-evaluation-harness and writes OUT/scores.json.

The paper's evaluation (Pal et al., COLING 2025, Table 4; the authors released no evaluation code): lm-eval 0.4.3
defaults (0-shot, batch size 1, the model's dtype, no chat template) on persona_openness, persona_conscientiousness,
persona_extraversion, persona_agreeableness and persona_neuroticism (1,000 statements each, from Anthropic's
model-written evals). Accuracy = share of statements answered as a person with the trait would.
primary = each trait's accuracy; compare each separately, higher is better.

Usage: eval.py MODEL OUT [ADAPTER]  (ADAPTER: a PEFT adapter applied to MODEL)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import datasets
import lm_eval

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

if __name__ == "__main__":
    model, out, *adapter = sys.argv[1:]
    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True  # persona is a loading script (lm_eval --trust_remote_code)
    args = f"pretrained={model}" + "".join(f",peft={a}" for a in adapter)
    results = lm_eval.simple_evaluate(model="hf", model_args=args, tasks=[f"persona_{t}" for t in TRAITS])["results"]
    metrics = {t: results[f"persona_{t}"] for t in TRAITS}
    primary = {t: m["acc,none"] for t, m in metrics.items()}
    Path(out, "scores.json").write_text(json.dumps({"primary": primary, "metrics": metrics}, indent=2))
    print(json.dumps(primary))
