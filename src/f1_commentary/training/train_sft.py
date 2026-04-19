"""Phase 8: SFT fine-tuning of Qwen2.5-7B with LoRA via Unsloth.

Usage:
    python -m f1_commentary.training.train_sft [OPTIONS]

Requires a separate training venv with unsloth installed:
    pip install unsloth transformers peft bitsandbytes datasets trl accelerate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import unsloth  # must be imported before transformers/trl/peft  # noqa: F401
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# ── Defaults ────────────────────────────────────────────────────────

DEFAULT_MODEL = "unsloth/Qwen2.5-7B-bnb-4bit"
DEFAULT_TRAIN_DATA = "data/datasets/sft_train_split.jsonl"
DEFAULT_VAL_DATA = "data/datasets/sft_val_split.jsonl"
DEFAULT_OUTPUT_DIR = "data/artifacts/qwen25-7b-f1-lora"
DEFAULT_MAX_SEQ_LEN = 2048

# LoRA defaults
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training defaults
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 8
DEFAULT_LR = 2e-4
DEFAULT_WARMUP_STEPS = 10
DEFAULT_LOGGING_STEPS = 5
DEFAULT_SAVE_STEPS = 50
DEFAULT_SAVE_TOTAL_LIMIT = 3
DEFAULT_SEED = 42


def format_example(example: dict) -> dict:
    """Format an instruction/input/output example into the Alpaca-style prompt."""
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return {"text": text}


def load_jsonl(path: str | Path):
    """Load a JSONL file into a HuggingFace Dataset."""
    return load_dataset("json", data_files=str(path), split="train")


def train(
    model_name: str = DEFAULT_MODEL,
    train_data: str = DEFAULT_TRAIN_DATA,
    val_data: str | None = DEFAULT_VAL_DATA,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_seq_length: int = DEFAULT_MAX_SEQ_LEN,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    lr: float = DEFAULT_LR,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Run SFT training and return the path to the saved adapter."""
    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load and format datasets
    print(f"Loading training data: {train_data}")
    train_dataset = load_jsonl(train_data).map(format_example)
    print(f"  {len(train_dataset)} training examples")

    eval_dataset = None
    if val_data and Path(val_data).exists():
        print(f"Loading validation data: {val_data}")
        eval_dataset = load_jsonl(val_data).map(format_example)
        print(f"  {len(eval_dataset)} validation examples")

    # Training arguments
    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=DEFAULT_LOGGING_STEPS,
        save_steps=DEFAULT_SAVE_STEPS,
        save_total_limit=DEFAULT_SAVE_TOTAL_LIMIT,
        seed=seed,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=DEFAULT_SAVE_STEPS if eval_dataset else None,
    )

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    trainer.train()

    # Save final adapter
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Adapter saved to {final_dir}")

    # Save training config for reproducibility
    config = {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": lr,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "train_examples": len(train_dataset),
        "val_examples": len(eval_dataset) if eval_dataset else 0,
    }
    config_path = Path(output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to {config_path}")

    return final_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-7B for F1 commentary")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model name")
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_DATA, help="Training JSONL path")
    parser.add_argument("--val-data", default=DEFAULT_VAL_DATA, help="Validation JSONL path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model,
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )
