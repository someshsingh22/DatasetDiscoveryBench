"""Fine-tunes a causal LM on a submission with the TL;DR SFT recipe of the re-implementation of Stiennon et al. (2020).

Source: vwxyzjn/summarize_from_feedback_details@ee77a9a summarize_from_feedback_details/sft.py, plus the SFT
preprocessing of its tldr_dataset.py. Unchanged except `# DDB:` lines: the submission CSV replaces the authors'
preprocessed TL;DR dataset, and the TL;DR ROUGE evaluation (--run_eval) is removed (eval.py scores CNN/DM instead).

Usage: accelerate launch --config_file deepspeed.yaml --num_processes N train.py --base_model=MODEL \
         --query_dataset=SUBMISSION.csv --output_dir=OUT --gradient_accumulation_steps=128/N --deepspeed
  Loss on query and summary tokens, 1 epoch, AdamW lr 3e-6 eps 1e-5, cosine, no warmup, batch 128, bf16 ZeRO-2, seed 1.
"""

import collections
import copy  # DDB: tldr_dataset.py helpers
import multiprocessing  # DDB: tldr_dataset.py helpers
import os
import random
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Dict, Literal, Optional  # DDB: Dict for tldr_dataset.py helpers

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import tyro
from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast
from datasets import Dataset  # DDB: the submission CSV instead of load_dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    get_scheduler,
)
from huggingface_hub import HfApi

api = HfApi()
COLUMNS = ["subreddit", "title", "post", "summary"]  # DDB: the submission's columns


# DDB: from tldr_dataset.py: its TL;DR query parameters (Args.tldr_params) and SFT preprocessing (process_query_data)
@dataclass
class TaskQueryHParams:
    length: Optional[int] = None
    format_str: Optional[str] = None
    truncate_field: Optional[str] = None
    truncate_text: Optional[str] = None
    padding: Optional[Literal["empty_space", "pad_token"]] = None
    pad_token: Optional[str] = None
    pad_side: Optional[str] = None
    max_sft_response_length: Optional[int] = None
    max_sft_query_response_length: Optional[int] = None
    max_rm_response_length: Optional[int] = None
    max_rm_query_response_length: Optional[int] = None


TLDR = TaskQueryHParams(
    length=512,
    format_str="SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:",
    truncate_field="post",
    truncate_text="\n",
    padding="pad_token",
    pad_side="left",
    max_sft_response_length=53,
    max_sft_query_response_length=562,
    max_rm_response_length=169,
    max_rm_query_response_length=638,
)


def _ensure_length(toks, l, pad_sequence=None, pad_side=None, truncate_side=None):
    assert pad_side in (None, "left", "right")
    assert truncate_side in (None, "left", "right")
    if len(toks) < l:
        assert pad_sequence is not None
        pad_amt = l - len(toks)
        assert len(pad_sequence) >= pad_amt, f"{len(pad_sequence)} < {pad_amt}"
        if pad_side is None:
            assert len(toks) == l, f"Needed to pad! {len(toks)} < {l}"
            return toks
        elif pad_side == "left":
            return pad_sequence[-pad_amt:] + toks
        else:
            assert pad_side == "right"
            return toks + pad_sequence[:pad_amt]
    if truncate_side is None:
        assert len(toks) == l, f"Needed to truncate! {len(toks)} > {l}"
        return toks
    elif truncate_side == "left":
        return toks[-l:]
    else:
        assert truncate_side == "right"
        return toks[:l]


def _get_query_padding_for_task(encoder, hparams: TaskQueryHParams):
    return hparams.pad_token * hparams.length


def process_query(query_info: Dict[str, str], *, encoder, hparams: TaskQueryHParams, pad_sequence=None):
    if pad_sequence is None:
        pad_sequence = _get_query_padding_for_task(encoder, hparams)
    if isinstance(query_info, str):
        query_info = dict(query=query_info)
    else:
        # copy to avoid mutating input
        query_info = dict(**query_info)

    format_str = hparams.format_str or "{query}"
    query_tokens = encoder.encode(format_str.format(**query_info))
    truncate_field = hparams.truncate_field or "query"

    if truncate_field not in query_info:
        raise ValueError(f"Could not truncate field {truncate_field}, found fields: {query_info.keys()}!")
    while len(query_tokens) > hparams.length:
        if not len(query_info[truncate_field]):
            raise ValueError("Could not truncate enough!")

        i = -1  # default to just remove one character
        if hparams.truncate_text:
            try:
                i = query_info[truncate_field].rindex(hparams.truncate_text)
            except ValueError:
                pass
        query_info[truncate_field] = query_info[truncate_field][:i]
        query_tokens = encoder.encode(format_str.format(**query_info))

    query_token = _ensure_length(query_tokens, hparams.length, pad_side=hparams.pad_side, pad_sequence=pad_sequence)
    query = encoder.decode(query_token, skip_special_tokens=True).lstrip()
    return dict(
        query_token=query_token,
        query=query,
    )


def process_query_data(x, tokenizer):  # DDB: was a closure over tldr_dataset.py's args.tldr_params and tokenizer
    # the `x['summary']` in `vwxyzjn/summarize_from_feedback_tldr_3_filtered`
    # DOES NOT HAVE a leading space so we are adding the leading space and
    # `<|endoftext|>` token
    reference_response = f" {x['summary']}<|endoftext|>"
    y = {
        **process_query(x, encoder=tokenizer, hparams=TLDR),
        "reference_response": reference_response,
        "reference_response_token": tokenizer.encode(
            reference_response,
            padding="max_length",
            max_length=TLDR.max_sft_response_length,
            truncation=True,
        ),
        "reference_response_token_len": len(tokenizer.encode(reference_response)),
    }
    y["query_reference_response"] = y["query"].strip() + y["reference_response"]
    # if padding is space, then we can just concatenate the tokens
    if TLDR.padding == "empty_space":
        y["query_reference_response_token"] = y["query_token"] + y["reference_response_token"]
    else:
        y["query_reference_response_token"] = tokenizer.encode(
            y["query_reference_response"],
            padding="max_length",
            max_length=TLDR.max_sft_query_response_length,
            truncation=True,
        )
    y["query_reference_response_token_response_label"] = copy.deepcopy(y["query_reference_response_token"])
    unpadded_query_token = [token for token in y["query_token"] if token != tokenizer.pad_token_id]
    y["query_reference_response_token_response_label"][:len(unpadded_query_token)] = [tokenizer.pad_token_id for _ in range(len(unpadded_query_token))]
    y["query_reference_response_token_len"] = len(tokenizer.encode(y["query_reference_response"]))
    return y


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    load_from_cache_file: bool = False
    """Whether to load data from the local cache file in `dataset.map`"""
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 220
    """How often to print sample output"""
    run_eval: bool = False
    """Whether to run evaluation"""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer"""
    lr: float = 3e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    # various batch sizes
    world_size: Optional[int] = None
    """The number of processes (GPUs) to use"""
    num_train_epochs: int = 1
    """Number of epochs to train"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    gradient_accumulation_steps: int = 16
    """The number of gradient accumulation steps"""
    local_micro_batch_size: Optional[int] = 1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = None
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    local_eval_batch_size: int = 4
    """per rank eval batch size"""

    # other args
    base_model: str = "EleutherAI/pythia-160m"
    """the name of the pretrained model to use"""
    query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    """the query dataset"""
    response_length: int = 53
    """the length of the response"""
    truncate_token: Literal["eos"] = "eos"
    """the truncate token"""
    truncate_token_id: Optional[int] = None
    """the truncation token id"""
    temperature: float = 0.01
    """the sampling temperature"""

    # wandb and HF tracking configs
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tldr_summarize"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    push_to_hub: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: Optional[str] = None
    """the user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: Optional[str] = None
    """the id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: Optional[str] = None
    """the revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: Optional[str] = None
    """the url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: str = "models/sft_model"
    """Where to save the model"""


def parse_args() -> tuple[Args, Accelerator]:
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.local_micro_batch_size * args.gradient_accumulation_steps
    args.micro_batch_size = int(args.local_micro_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
    time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
    args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
    if args.push_to_hub:
        if args.hf_repo_id is None: # auto-generate one
            args.hf_repo_id = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        if args.hf_entity is None:  # find the current user
            args.hf_entity = api.whoami()["name"]
        if "/" not in args.hf_repo_id: # prepend the current user
            args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
    return args, accelerator


# taken from https://github.com/vwxyzjn/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/utils.py#L99
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


def forward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )


if __name__ == "__main__":
    args, accelerator = parse_args()
    local_seed = args.seed + accelerator.process_index * 100003  # Prime

    # load dataset
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # DDB: the submission CSV (--query_dataset), preprocessed as tldr_dataset.py does (tokenizer moved up for it)
    TLDR.pad_token = [tokenizer.pad_token_id]
    dataset = Dataset.from_pandas(pd.read_csv(args.query_dataset, usecols=COLUMNS, dtype=str, keep_default_na=False))
    dataset = dataset.map(process_query_data, fn_kwargs={"tokenizer": tokenizer}, num_proc=multiprocessing.cpu_count())
    dataset = dataset.with_format("torch", columns=["query_reference_response_token"])
    dataloader = DataLoader(dataset, batch_size=args.local_micro_batch_size, shuffle=True)
    args.total_episodes = len(dataset)
    args.num_updates = args.total_episodes // args.batch_size
    if args.truncate_token == "eos":
        args.truncate_token_id = tokenizer.eos_token_id

    console = Console(force_terminal=True)
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    if accelerator.is_main_process:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=asdict(args),
                name=args.run_name,
                save_code=True,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"runs/{args.run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.backends.cudnn.deterministic = True
    model_config = AutoConfig.from_pretrained(args.base_model)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=model_config,
        trust_remote_code=True,
    )
    disable_dropout(model)
    model.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    model.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_updates * args.num_train_epochs,
    )

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    torch.manual_seed(local_seed)  # reset the local seed again

    accelerator.print("===training model===")
    loss_stats = torch.zeros(args.gradient_accumulation_steps, device=device)
    model.train()
    gradient_accumulation_idx = 0
    global_step = 0
    update = 0
    for epoch in range(args.num_train_epochs):
        accelerator.print(f"epoch: {epoch}")
        for data in dataloader:
            update += 1
            global_step += args.micro_batch_size
            # reference_responses = data["reference_response_token"].to(device, non_blocking=True)
            # queries = data["query_token"].to(device, non_blocking=True)
            query_responses = data["query_reference_response_token"]
            with accelerator.accumulate(model):
                output = forward(model, query_responses, tokenizer)
                # mask out gradient effects on response padding tokens
                labels = query_responses.masked_fill(query_responses == tokenizer.pad_token_id, -1)
                lm_logits = output.logits
                # hand-rolled transformer loss: Shift so that tokens < n predict n
                # but unlike `transformers` we mask the padding tokens via `ignore_index=-1`
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-1)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            loss_stats[gradient_accumulation_idx] = loss
            gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.gradient_accumulation_steps
            if update > 1 and (update - 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()
                writer.add_scalar("train/sft/loss", accelerator.gather(loss_stats).mean().item(), update)
                writer.add_scalar("train/sft/lr", scheduler.get_last_lr()[0], update)
                accelerator.print(f"{loss.item()=}, {scheduler.get_last_lr()=}, {optimizer.param_groups[0]['lr']=}, {update=}")

    assert 0 < loss.item() < float("inf"), "training diverged (NaN/inf loss)"  # DDB: fail on a diverged run

    # save model
    if args.output_dir and args.num_train_epochs > 0:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                tokenizer.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision)
        unwrapped: PreTrainedModel = accelerator.unwrap_model(model)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
                safe_serialization=False,
            )
            if args.push_to_hub:
                unwrapped.push_to_hub(repo_id=args.hf_repo_id, revision=args.hf_repo_revision, safe_serialization=False)
                accelerator.print(f"🔥 pushed to {args.hf_repo_url}")
