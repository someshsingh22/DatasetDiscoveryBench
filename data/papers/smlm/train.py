"""Pre-trains Longformer with selective MLM on discussion threads, then fine-tunes it on CMV Modes (Dutta et al., ACL 2022).

Usage: train.py INIT DATA OUT
  DATA is a submission CSV: INIT (a LongformerForMaskedLM) learns to predict the masked discourse markers of every
  root-to-leaf thread (sMLM: ~24.6k-token steps, 4 epochs, Adam lr 1e-6, bf16) and is saved to OUT. DATA=cmv_modes
  instead fine-tunes INIT 5 times per task on random thread splits of CMV Modes: argument component identification
  (aci_80: 80/20 split, BIO tags + CRF) and relation type prediction (rtp_80, rtp_50: prompting), 30 epochs, Adam lr
  2e-5, batch 2, accumulation 4. The torchrun processes share the runs; each writes the test F1 of its last 5 epochs to
  OUT/<task>_<run>.json. MAX_STEPS shortens training (smoke tests only).
"""

from __future__ import annotations

import html
import itertools
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from bs4 import Comment
from seqeval.metrics import f1_score
from sklearn import metrics
from torch.nn.utils.rnn import pad_sequence
from transformers import AddedToken
from transformers import AutoTokenizer
from transformers import LongformerForMaskedLM
from transformers import LongformerModel
from transformers import Trainer
from transformers import TrainingArguments

DATA = Path(__file__).parent / "external_eval_data"
COLUMNS = ["id", "reply_to", "user", "title", "text"]
MAX_LEN = 4096
SEED = 0
USERS = [f"[USER{i}]" for i in range(10)]  # a thread's first 10 users; later or deleted users are [UNU]
SPECIAL = ["[STARTQ]", "[ENDQ]", "[URL]", "[NEWLINE]", "[UNU]", *USERS]
MARKERS = re.compile(  # the authors' Discourse_Markers.txt, matched as whole words, longest first
    r"(?<![^\W_])(?:" + "|".join(sorted((
        "moreover|in addition|additionally|as far as|further to this|also|besides|what is more|however|on the other hand|"
        "in contrast|yet|though|although|even though|despite|in spite of|regardless of|because|since|as|insofar as|"
        "therefore|consequently|in consequence|as a result|accordingly|according to|hence|thus|the reason is|"
        "for this reason|because of this|if|in the event of|as long as|so long as|provided that|assuming that|given that|"
        "on the contrary|as a matter of fact|in fact|indeed|firstly|secondly|first of all|lastly|so|i agree|i disagree|"
        "why|where|what|how|when|but|i think|my opinion|argue that|i do n't think|i do not think|while|meanwhile|tldr|"
        "imo|imho|in my opinion|in my honest opinion"
    ).split("|"), key=len, reverse=True)) + r")(?![^\W_])",
    re.IGNORECASE,
)  # fmt: skip
LABELS = ["O", "B-C", "I-C", "B-P", "I-P"]
ALLOWED = torch.tensor([  # the authors' CRF constraints: allowed tag transitions, row -> column
    [1, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 1, 0, 1, 1],
], dtype=torch.bool)  # fmt: skip
RELS = {  # RTP class: CMV Modes relations ("cont" links the pieces of a component split by a nested one)
    "support": ["cont", "support"],
    "agreement": ["agreement", "understand"],
    "direct_attack": ["attack", "rebuttal_attack", "rebuttal", "disagreement"],
    "undercutter_attack": ["undercutter", "undercutter_attack"],
    "partial": ["partial_agreement", "partial_attack", "partial_disagreement"],
}
REL = {rel: i for i, rels in enumerate(RELS.values()) for rel in rels}
# cross-entropy class weights, from the code
ACI_WEIGHT = torch.tensor([3.3102, 61.4809, 3.6832, 49.6827, 2.5639]).log()
RTP_WEIGHT = torch.tensor([0.508, 2.027, 2.402, 2.234, 2.667])
TASKS = ["aci_80", "rtp_80", "rtp_50"]
RUNS = [(task, run) for task in ["rtp_80", "rtp_50", "aci_80"] for run in range(5)]  # longest first


def load_xml(path: Path) -> BeautifulSoup:
    """Parses a CMV Modes thread, escaping its bare "&"s first (the XML parser drops text after them)."""
    return BeautifulSoup(re.sub(r"&(?!#?\w+;)", "&amp;", path.read_text(encoding="utf-8")), "xml")


def tag(text: str) -> str:
    """Replaces URLs, quoted lines and line breaks with the authors' special tokens."""
    text = re.sub(r"https?://[^\s()\[\]\x00]+", "[URL]", text)
    text = re.sub(r">(.*)\n", r"[STARTQ]\1[ENDQ] ", text)
    return text.replace("\n", "[NEWLINE]").replace("\r", "[NEWLINE]")


def clean(text: str) -> str:
    """The authors' CMV Modes normalization: plain quotes, "&" -> "and", "∆" -> "[DELTA]", spaced punctuation."""
    for old, new in [("’", "'"), ("“", '"'), ("”", '"'), ("&", "and"), ("∆", "[DELTA]")]:
        text = text.replace(old, new)
    for p in '.,!;:*?"()^':
        text = text.replace(p, f" {p} ")
    return " " + " ".join(text.split())


def user_token(users: tuple[str, ...], user: str) -> str:
    """Returns the token of `user`, given a thread's distinct non-deleted users in order of appearance."""
    return USERS[users.index(user)] if user in users[:10] else "[UNU]"


def threads(df: pd.DataFrame) -> list[str]:
    """Returns every root-to-leaf thread: the root's title, then each post's user token and text."""
    kids = df.groupby("reply_to").id.agg(list).to_dict()
    posts = dict(zip(df.id, zip(df.user, df.text)))
    stack = [(i, title, ()) for i, title in df.loc[~df.reply_to.isin(df.id), ["id", "title"]].itertuples(index=False)]
    out = []
    while stack:
        i, text, users = stack.pop()
        user, post = posts[i]
        users += () if user == "[deleted]" or user in users else (user,)
        text += user_token(users, user) + post
        if i in kids:
            stack += [(kid, text, users) for kid in kids[i]]
        else:
            out.append(tag(html.unescape(text)))
    return out


def smlm_data(path: str, tok) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns each thread's token ids and discourse-marker token mask, dropping threads without markers."""
    texts = threads(pd.read_csv(path, usecols=COLUMNS, dtype=str, keep_default_na=False))
    ids, masks = [], []
    for i in range(0, len(texts), 1024):
        chunk = texts[i : i + 1024]
        enc = tok(chunk, truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
        for text, x, offsets in zip(chunk, enc.input_ids, enc.offset_mapping):
            covered = np.zeros(len(text), np.int64)
            for m in MARKERS.finditer(text):
                covered[m.start() : m.end()] = 1
            covered = np.concatenate([[0], covered.cumsum()])  # marker characters before each offset
            start, end = np.array(offsets).T
            mask = covered[end] > covered[start]
            if mask.any():
                ids.append(np.array(x, np.int32))
                masks.append(mask)
    return ids, masks


def pretrain(init: str, path: str, out: str, tok) -> None:
    """sMLM: trains INIT to predict the masked discourse-marker tokens of the submission's threads."""
    ids, masks = smlm_data(path, tok)
    budget = 3 * 8194 // int(os.environ.get("WORLD_SIZE", 1))  # the paper's 8,194-token batches x 3 accumulated
    batches = []
    for i in sorted(range(len(ids)), key=lambda i: -len(ids[i])):
        if batches and (len(batches[-1]) + 1) * len(ids[batches[-1][0]]) <= budget:
            batches[-1].append(i)
        else:
            batches.append([i])
    users = torch.tensor(tok.convert_tokens_to_ids(["[UNU]", *USERS]))

    def collate(items: list[list[int]]) -> dict[str, torch.Tensor]:
        x = pad_sequence([torch.from_numpy(ids[i]).long() for i in items[0]], True, tok.pad_token_id)
        mask = pad_sequence([torch.from_numpy(masks[i]) for i in items[0]], True)
        return {
            "input_ids": x.masked_fill(mask, tok.mask_token_id),
            "attention_mask": (x != tok.pad_token_id).long(),
            "global_attention_mask": torch.isin(x, users).long(),  # user tokens attend globally
            "labels": x.masked_fill(~mask, -100),
        }

    args = TrainingArguments(
        out,
        num_train_epochs=4,
        max_steps=int(os.environ.get("MAX_STEPS", -1)),
        per_device_train_batch_size=1,  # one item is one batch of threads
        learning_rate=1e-6,
        lr_scheduler_type="constant",
        max_grad_norm=0,  # no clipping
        bf16=True,
        save_strategy="no",
        remove_unused_columns=False,
        seed=SEED,
    )
    model = LongformerForMaskedLM.from_pretrained(init)
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    trainer = Trainer(model, args, data_collator=collate, train_dataset=batches, processing_class=tok)
    assert 0 < trainer.train().training_loss < float("inf"), "training diverged (NaN/inf loss or no optimizer steps)"
    trainer.save_model()


def encode(path: Path, tok) -> tuple[str, list[int], list[int], list[tuple[int, ...]]]:
    """Encodes a CMV Modes thread as the authors do: the title (a claim), then each post's user token and text.

    Returns the OP's ID, the token ids, the tags (indices into LABELS) and the relations (source start, source end,
    target start, target end, class): each encoded component's first link to another, and each piece of a component
    split by a nested one to the previous piece.
    """
    soup = load_xml(path)
    parts = [[None, soup.title, soup.title.get_text()]]  # [post, innermost component, text]
    for node in soup.thread.find_all(string=True):
        post = node.find_parent(["OP", "reply"])
        if post is None or isinstance(node, Comment):
            continue
        comp = node.find_parent(["claim", "premise"])
        if parts[-1][0] is post and parts[-1][1] is comp:
            parts[-1][2] += node
        else:
            parts.append([post, comp, str(node)])
    for _, group in itertools.groupby(parts, lambda part: id(part[0])):  # tag each post whole, as quotes span parts
        group = list(group)
        for part, text in zip(group, tag("\x00".join(part[2] for part in group)).split("\x00")):
            part[2] = text
    users = tuple(dict.fromkeys(p["author"] for p in soup.find_all(["OP", "reply"]) if p["author"] != "[deleted]"))
    pieces, post = [], None
    for p, comp, text in parts:
        if text.strip():
            if p is not post:
                post = p
                pieces.append(([tok.convert_tokens_to_ids(user_token(users, p["author"]))], None))
            text = text if p is None else clean(text)  # the title is encoded raw
            pieces.append((tok(text, add_special_tokens=False).input_ids, comp))
    ids, tags, found = [tok.bos_token_id], [0], {}
    for piece, comp in pieces:
        if len(ids) + len(piece) >= MAX_LEN - 1:
            break
        if comp is not None:
            found.setdefault(comp, []).append((len(ids), len(ids) + len(piece)))
        b = 0 if comp is None else 3 if comp.name == "premise" else 1
        tags += [b] + [b + (b > 0)] * (len(piece) - 1)
        ids += piece
    names = {comp.get("id", "title"): comp for comp in found}
    rels = []
    for comp, spans in found.items():
        rels += [(*b, *a, 0) for a, b in zip(spans, spans[1:])]
        refs = [ref for ref in comp.get("ref", "").split("_") if ref in names]
        if refs and names[refs[0]] is not comp:  # only the first link counts; self-links are skipped
            rels.append((*spans[0], *found[names[refs[0]]][0], REL[comp["rel"]]))
    return soup.thread["ID"], ids + [tok.eos_token_id], tags + [0], rels


def examples(threads: list[tuple], aci: bool, tok) -> list[tuple[torch.Tensor, torch.Tensor | int]]:
    """Returns (token ids, tags) per thread for ACI; for RTP, (prompt, class) per relation, the prompt being the thread
    then "<target's user> said <target> <mask> <mask> <mask> <source's user> said <source>" (the thread is cut to fit).
    """
    if aci:
        return [(torch.tensor(ids), torch.tensor(tags)) for _, ids, tags, _ in threads]
    users = {*tok.convert_tokens_to_ids(["[UNU]", *USERS])}
    said = tok(" said", add_special_tokens=False).input_ids[0]
    out = []
    for _, ids, _, rels in threads:
        starts = [i for i, t in enumerate(ids) if t in users or i == 0]  # <s> stands in before the first user token
        for fs, fe, ts, te, rel in rels:
            speaker = [ids[max(i for i in starts if i <= s)] for s in (ts, fs)]
            suffix = [speaker[0], said, *ids[ts:te], *[tok.mask_token_id] * 3, speaker[1], said, *ids[fs:fe]]
            out.append((torch.tensor(ids[: MAX_LEN - len(suffix)] + suffix), rel))
    return out


def crf_nll(logits: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Summed negative log-likelihood of a linear-chain CRF without start/end transitions (allennlp's)."""
    gold = (logits.gather(2, tags[..., None])[..., 0] * mask).sum()
    gold = gold + (trans[tags[:, :-1], tags[:, 1:]] * mask[:, 1:]).sum()
    alpha = logits[:, 0]
    for i in range(1, logits.shape[1]):
        alpha = torch.where(mask[:, i, None], (alpha[:, :, None] + trans).logsumexp(1) + logits[:, i], alpha)
    return alpha.logsumexp(1).sum() - gold


def viterbi(emissions: np.ndarray, trans: np.ndarray) -> list[int]:
    """Returns the most likely tag sequence of a CRF without start/end transitions."""
    score, back = emissions[0], []
    for e in emissions[1:]:
        s = score[:, None] + trans
        back.append(s.argmax(0))
        score = s.max(0) + e
    path = [int(score.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return path[::-1]


def forward(model, head, batch: list, aci: bool, tok) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the padded batch and its logits: per token (ACI) or per prompt from its 3 masks (RTP)."""
    x = pad_sequence([ids for ids, _ in batch], True, tok.pad_token_id).cuda()
    users = torch.tensor(tok.convert_tokens_to_ids(["[UNU]", *USERS]), device="cuda")
    with torch.autocast("cuda", torch.bfloat16):
        h = model(
            x,
            attention_mask=(x != tok.pad_token_id).long(),
            global_attention_mask=torch.isin(x, users).long() if aci else None,  # RTP: local attention only
        ).last_hidden_state.float()
    return x, head(h) if aci else head(h[x == tok.mask_token_id].view(len(batch), -1))


def finetune(init: str, out: str, tok) -> None:
    """Runs this process's share of RUNS, writing each run's test F1 after each of its last 5 epochs."""
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    data = [encode(path, tok) for path in sorted(DATA.glob("*/*.xml"))]
    max_steps = int(os.environ.get("MAX_STEPS", 0))
    epochs = 1 if max_steps else 30
    os.makedirs(out, exist_ok=True)
    for task, run in RUNS[rank::world]:
        aci, groups, picked = task.startswith("aci"), list(dict.fromkeys(t[0] for t in data)), set()
        random.Random(run).shuffle(groups)  # the run's split: whole OPs go to train until it holds >= 80% (50%)
        while 100 * sum(t[0] in picked for t in data) / len(data) < int(task[-2:]):
            picked.add(groups.pop(0))
        train = examples([t for t in data if t[0] in picked], aci, tok)
        test = examples([t for t in data if t[0] not in picked], aci, tok)
        rng = random.Random(SEED * 5 + run)
        torch.manual_seed(SEED * 5 + run)
        model = LongformerModel.from_pretrained(init, add_pooling_layer=False)
        model.resize_token_embeddings(len(tok), mean_resizing=False)
        model.cuda()
        head = torch.nn.Linear(model.config.hidden_size * (1 if aci else 3), 5).cuda()
        trans = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(5, 5, device="cuda")))
        opt = torch.optim.Adam([*model.parameters(), *head.parameters(), trans], lr=2e-5)
        scores = []
        for epoch in range(epochs):
            model.train()
            rng.shuffle(train)
            batches = [train[i : i + 2] for i in range(0, len(train), 2)][: max_steps or None]
            total = 0.0
            for i, batch in enumerate(batches):
                x, logits = forward(model, head, batch, aci, tok)
                if aci:
                    y, mask = pad_sequence([y for _, y in batch], True).cuda(), x != tok.pad_token_id
                    ce = F.cross_entropy(logits.transpose(1, 2), y, ACI_WEIGHT.cuda(), reduction="none")
                    loss = ((ce * mask).sum() + crf_nll(logits, y, mask, trans)) / 2
                else:
                    y = torch.tensor([y for _, y in batch], device="cuda")
                    loss = F.cross_entropy(logits, y, RTP_WEIGHT.cuda(), reduction="sum") / 2
                loss.backward()
                total += loss.item()
                if i % 4 == 3 or i == len(batches) - 1:
                    opt.step()
                    opt.zero_grad()
            if epoch >= epochs - 5:
                model.eval()
                scores.append(evaluate(model, head, trans, test, aci, tok))
            print(f"DDB_EPOCH {task}_{run} {epoch + 1} loss={total / len(batches):.4f} f1={scores[-1:]}", flush=True)
        Path(out, f"{task}_{run}.json").write_text(json.dumps({"scores": scores}))


@torch.no_grad()
def evaluate(model, head, trans: torch.Tensor, test: list, aci: bool, tok) -> float:
    """Returns the micro-F1 of the Viterbi BIO tags (ACI) or the weighted F1 of the relation classes (RTP)."""
    trans = torch.where(ALLOWED.cuda(), trans, -1e4).cpu().numpy()
    preds, gold = [], []
    for i in range(0, len(test), 2):
        batch = test[i : i + 2]
        _, logits = forward(model, head, batch, aci, tok)
        if aci:
            for emissions, (_, y) in zip(logits.cpu().numpy(), batch):
                preds.append([LABELS[t] for t in viterbi(emissions[: len(y)], trans)])
                gold.append([LABELS[t] for t in y.tolist()])
        else:
            preds += logits.argmax(-1).tolist()
            gold += [y for _, y in batch]
    if aci:
        return f1_score(gold, preds)
    return metrics.f1_score(gold, preds, labels=range(5), average="weighted", zero_division=0)


if __name__ == "__main__":
    init, data, out = sys.argv[1:]
    tok = AutoTokenizer.from_pretrained(init)
    # strip the spaces around them, as the authors' slow tokenizer did
    tok.add_tokens([AddedToken(t, lstrip=True, rstrip=True) for t in SPECIAL])
    if data == "cmv_modes":
        finetune(init, out, tok)
    else:
        pretrain(init, data, out, tok)
