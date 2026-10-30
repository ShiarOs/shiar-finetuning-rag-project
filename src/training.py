from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from src.data_validation import ROOT, read_training, validate


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def render_example(context: str, question: str, answer: str) -> str:
    return (
        "You are a Northstar support assistant. Use only the evidence. "
        "If the evidence is insufficient, say so. Cite policy claims "
        "using source IDs in square brackets.\n\n"
        f"Evidence:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}"
    )


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[dict[str, object]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ):
        self.items = []
        for record in records:
            text = render_example(
                str(record["context"]),
                str(record["question"]),
                str(record["answer"]),
            )
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            # Train only on the answer: mask the prompt tokens (instruction, evidence, question).
            prompt = render_example(str(record["context"]), str(record["question"]), "")
            prompt_length = len(tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"])
            labels[:prompt_length] = -100
            if not (labels != -100).any():
                raise ValueError(
                    f"Record {record.get('id')!r}: no answer tokens left after truncation; "
                    "raise MAX_SEQUENCE_LENGTH"
                )
            self.items.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def main() -> None:
    load_dotenv(ROOT / ".env")
    validate()
    records = read_training(ROOT / "data" / "training" / "instruction_examples.jsonl")
    if len(records) < 2:
        raise ValueError("Need at least two training examples")

    seed = int(os.getenv("RANDOM_SEED", "42"))
    seed_everything(seed)
    model_name = os.getenv(
        "BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"
    )
    revision = os.getenv("BASE_MODEL_REVISION", "").strip()
    model_args = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **model_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(os.getenv("MAX_SEQUENCE_LENGTH", "256"))
    dataset = TrainingDataset(records, tokenizer, max_length)
    # The checkpoint defaults to bfloat16, which is extremely slow on CPUs without bf16 support.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, **model_args
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    target_modules = ["q_proj", "v_proj"]
    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = set(target_modules) - module_names
    if missing:
        raise ValueError(
            f"LoRA target modules not present in this model: {sorted(missing)}"
        )

    adapter_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, adapter_config)
    model.print_trainable_parameters()

    output = Path(os.getenv("ADAPTER_PATH", "artifacts/support-lora-adapter"))
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output.parent / "training-runs"),
        num_train_epochs=int(os.getenv("TRAIN_EPOCHS", "8")),
        learning_rate=float(os.getenv("LEARNING_RATE", "0.0002")),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"Saved adapter to {output}")


if __name__ == "__main__":
    main()
