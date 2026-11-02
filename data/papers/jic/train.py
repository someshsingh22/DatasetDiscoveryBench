"""QLoRA fine-tuning on JIC-format dialogues: the authors' src/training/mistral/finetune.py (Pal et al., COLING 2025;
Sayantan-world/Beyond-Discrete-Personas @ e099650), changed only where marked `# DDB:`.

Usage: torchrun train.py --model_name M --input_file SUBMISSION.json --cache_dir DIR --hf_token TOKEN --output_dir OUT
  LoRA r 64 / alpha 16 on q,k,v,o of the 4-bit model; packed 2048-token sequences, effective batch 4 x 128, lr 2e-5
  cosine, 3 epochs, eval on 1,000 held-out dialogues every 100 steps (best checkpoint kept). Saves the adapter to OUT.
  MAX_STEPS shortens training (smoke tests only). Removed: `import wandb`, the unused prompt_id (it needed author and
  journal ids) and a tokenizer save from every DDP rank (save_model saves it once).
"""
import os
import json
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from multiprocessing import cpu_count
from transformers import BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import TrainingArguments, EarlyStoppingCallback
import argparse
from pathlib import Path
from datasets import load_dataset
import random

class MistralFineTuner:
    def __init__(self, model_name, input_file, cache_dir, hf_token, output_dir, seed=42):
        self.model_name = model_name
        self.input_file = input_file
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.output_dir = output_dir
        self.seed = seed
        set_seed(self.seed)

        # Set environment variables
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
        os.environ["WANDB_API_KEY"] = ""
        os.environ["WANDB_PROJECT"] = ""
        os.environ["WANDB_WATCH"] = ""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
        self.tokenizer.padding_side = "right"
        self.model = self.load_model()

    def load_model(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}  # DDB: one GPU per torchrun process
        model_kwargs = dict(
            torch_dtype="auto",
            use_cache=False,
            device_map=device_map,
            quantization_config=quantization_config,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir,
            token=self.hf_token,
            **model_kwargs
        )
        return model

    def json_to_hf_dataset(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)

        records = []
        for entry in data:
            messages = []
            for i, utterance in enumerate(entry['dialogue']):
                role = 'user' if i % 2 == 0 else 'assistant'
                messages.append({"content": utterance, "role": role})
            records.append({"messages": messages})  # DDB: only `dialogue` is read (prompt_id was dropped unused)

        train_data, test_data = train_test_split(records, test_size=1000, random_state=42)
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
        test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))
        dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})

        return dataset_dict

    def load_spc_from_hf(self, input_file):
        raw_datasets = load_dataset(self.input_file, cache_dir=self.cache_dir)

        def prepare_dataset(dataset_split):
            data = dataset_split["Best Generated Conversation"]
            personas = dataset_split["user 2 personas"]
            records = []
            for dialogue, persona in zip(data, personas):
                dialogue = dialogue.replace("User 1: ", '')
                dialogue = dialogue.replace("User 2: ", '')
                dialogue = dialogue.split("\n")
                if len(dialogue) == 0:
                    continue
                messages = []
                for i, utterance in enumerate(dialogue):
                    role = 'user' if i % 2 == 0 else 'assistant'
                    messages.append({"content": utterance, "role": role})
                if messages[0]["role"] != "system":
                    messages.insert(0, {"role": "system", "content": f"You are a helpful assistant. Respond concisely in English, using 20 words or fewer. Ground your reply to the assistant persona when needed, and use the first person. \nPersona: {persona}"})
                records.append({"messages": messages})
            return records

        train_data = prepare_dataset(raw_datasets["train"])
        test_data = prepare_dataset(raw_datasets["validation"])    
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
        test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))
        dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})
        return dataset_dict

    def load_pc_from_hf(self, input_file):
        raw_datasets = load_dataset(input_file, cache_dir=self.cache_dir)

        def prepare_dataset(dataset_split):
            data = dataset_split["utterances"]
            records = []
            personas = dataset_split["personality"]
            for dialogue, persona in zip(data, personas):
                dialogue = dialogue[-1]["history"]
                if len(dialogue) == 0:
                    continue
                messages = []
                for i, utterance in enumerate(dialogue):
                    role = 'user' if i % 2 == 0 else 'assistant'
                    messages.append({"content": utterance, "role": role})
                messages = messages[:-1]
                context = '\n'.join(persona)
                if messages[0]["role"] != "system":
                    messages.insert(0, {"role": "system", "content": f"You are a helpful assistant. Respond concisely in English, using 20 words or fewer. Ground your reply to the persona when needed, and use the first person. \nPersona: {context}"})
                records.append({"messages": messages})
            return records

        # Prepare the entire dataset
        all_data = prepare_dataset(raw_datasets["train"])

        # Shuffle the data and split into train and validation sets
        random.seed(42)
        random.shuffle(all_data)
        validation_size = 1000
        validation_data = all_data[:validation_size]
        train_data = all_data[validation_size:]

        # Create Huggingface datasets
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
        validation_dataset = Dataset.from_pandas(pd.DataFrame(validation_data))

        # Create a DatasetDict with train and validation splits
        dataset_dict = DatasetDict({'train': train_dataset, 'test': validation_dataset})
        return dataset_dict

    def load_bst_from_hf(self, input_file):
        raw_datasets = load_dataset(input_file)

        def prepare_dialogue(example):
            persona = example['personas']
            persona = "\n".join(persona)
            conversation = example['previous_utterance']
            for human, bot in zip(example['free_messages'], example['guided_messages']):
                conversation.append(human)
                conversation.append(bot)
            messages = []
            for i, utterance in enumerate(conversation):
                role = 'user' if i % 2 == 0 else 'assistant'
                messages.append({"content": utterance, "role": role})
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": f"You are a helpful assistant. Respond concisely in English, using 20 words or fewer. Ground your reply to the assistant persona when needed, and use the first person. \nPersona: {persona}"})
            return messages

        def prepare_dataset(dataset_split):
            records = []
            for item in dataset_split:
                messages = prepare_dialogue(item)
                records.append({"messages": messages})
            return records

        train_data = prepare_dataset(raw_datasets["train"])
        test_data = prepare_dataset(raw_datasets["validation"])    
        train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
        test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))
        dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})
        return dataset_dict

    def prepare_data(self):
        dataset = self.json_to_hf_dataset(self.input_file) 
        # dataset = self.load_spc_from_hf(self.input_file)
        # dataset = self.load_pc_from_hf(self.input_file)
        # dataset = self.load_bst_from_hf(self.input_file)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.tokenizer.model_max_length > 100_000:
            self.tokenizer.model_max_length = 2048

        DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '' }}\n{% endif %}\n{% endfor %}"
        self.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

        def apply_chat_template(example, tokenizer):
            messages = example["messages"]
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": "Reply concisely within 20 words "})
            example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
            return example

        column_names = list(dataset["train"].features)
        dataset = dataset.map(
            apply_chat_template,
            # num_proc=cpu_count(),
            fn_kwargs={"tokenizer": self.tokenizer},
            remove_columns=column_names,
            desc="Applying chat template",
        )

        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        return train_dataset, eval_dataset

    def train(self):
        train_dataset, eval_dataset = self.prepare_data()
        
        peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        sft_config = SFTConfig(
            bf16=True,
            do_eval=True,
            eval_strategy="steps", 
            gradient_accumulation_steps=128 // int(os.environ.get("WORLD_SIZE", 1)),  # DDB: effective batch 4 x 128 on any GPU count
            ddp_find_unused_parameters=False,  # DDB: every LoRA weight gets a gradient
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=2.0e-05,
            log_level="info",
            logging_steps=1,
            logging_strategy="steps",
            lr_scheduler_type="cosine",
            max_steps=int(os.environ.get("MAX_STEPS", -1)),  # DDB: smoke tests
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            per_device_eval_batch_size=4,
            per_device_train_batch_size=4,
            report_to="none",  # DDB: no wandb
            save_strategy="steps",
            save_steps=100, 
            eval_steps=100, 
            save_total_limit=2,
            load_best_model_at_end=True,
            seed=self.seed,
            dataset_text_field="text",
            packing=True,
            max_seq_length=self.tokenizer.model_max_length
        )

        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
        )

        train_result = trainer.train()
        assert 0 < train_result.training_loss < float("inf"), "training diverged (NaN/inf loss or no steps)"  # DDB
        trainer.save_model(self.output_dir)

        # Save training arguments to JSON file
        with open(os.path.join(self.output_dir, 'sft_config.json'), 'w') as f:
            json.dump(sft_config.to_dict(), f, default=list)
        with open(os.path.join(self.output_dir, 'peft_config.json'), 'w') as f:
            json.dump(peft_config.to_dict(), f, default=list)

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Fine-tune Mistral 7B model.")
        parser.add_argument('--model_name', type=str, required=True, help="Model name or path.")
        parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file.")
        parser.add_argument('--cache_dir', type=str, required=True, help="Path to the cache directory.")
        parser.add_argument('--hf_token', type=str, required=True, help="Hugging Face token.")
        parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the fine-tuned model.")
        parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
        return parser.parse_args()

    @classmethod
    def from_arguments(cls):
        args = cls.parse_arguments()
        return cls(args.model_name, args.input_file, args.cache_dir, args.hf_token, args.output_dir, args.seed)

    def run(self):
        self.train()

if __name__ == '__main__':
    fine_tuner = MistralFineTuner.from_arguments()
    fine_tuner.run()
