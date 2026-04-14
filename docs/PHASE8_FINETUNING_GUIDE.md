# Phase 8: Fine-Tuning Guide (Qwen2.5-7B with Unsloth)

Run this on the **4060 laptop** (8GB VRAM, 48GB RAM).

## Setup

```bash
# Create separate venv for training
python -m venv .venv-train
source .venv-train/bin/activate

# Install unsloth (follow latest install instructions for your CUDA version)
pip install unsloth
pip install transformers peft bitsandbytes datasets trl accelerate
```

## Dataset Format

Training examples live in `data/datasets/sft_train.jsonl`. Each line:

```json
{
  "instruction": "You are an expert Formula 1 commentator. Generate professional race commentary based on the following race situation.",
  "input": "LAP 42/56 | Monza 2024 Race\nOBSERVED: HAM P4, gap to NOR P3: 0.8s, closing 0.3s/lap. HAM on MEDIUM (age 15). NOR on HARD (age 22). DRS available.\nINFERRED: HAM will be within DRS in ~2 laps. NOR tire degradation accelerating.\nCONTEXT: Hamilton overtook Sainz 5 laps ago. Norris has not pitted since lap 20.",
  "output": "Hamilton is absolutely reeling in Norris now, the gap down to eight tenths and shrinking every lap. Those mediums are clearly working well for Lewis, and you have to wonder about Lando's hard tires — twenty-two laps old and the pace is starting to fall away. DRS is live, and if Hamilton can get within that one-second window in the next lap or two, we could see a move into the Variante della Roggia."
}
```

### Building the dataset

Use the dataset builder (Phase 7 — or manually create examples):

```bash
# If Phase 7 pipeline exists:
python -m f1_commentary.cli build-sft-dataset --year 2024 --gp Monza --session R

# Otherwise, manually create 50-200 examples in the format above
# Sources for "output" text: FastF1 team radio + your own written commentary
```

**Minimum viable dataset:** 50 high-quality examples. Aim for 100-200.

## Training Script

Create `src/f1_commentary/training/train_sft.py`:

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import torch

# === Model ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-bnb-4bit",
    max_seq_length=2048,
    dtype=None,  # auto-detect
    load_in_4bit=True,
)

# === LoRA ===
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # saves VRAM
)

# === Dataset ===
dataset = load_dataset("json", data_files="data/datasets/sft_train.jsonl", split="train")

# Format into chat template
def format_example(example):
    text = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    return {"text": text}

dataset = dataset.map(format_example)

# === Training ===
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="data/artifacts/qwen25-7b-f1-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        seed=42,
    ),
)

trainer.train()

# === Save ===
model.save_pretrained("data/artifacts/qwen25-7b-f1-lora/final")
tokenizer.save_pretrained("data/artifacts/qwen25-7b-f1-lora/final")
print("Adapter saved to data/artifacts/qwen25-7b-f1-lora/final")
```

## Expected VRAM usage

| Component | VRAM |
|-----------|------|
| Qwen2.5-7B 4-bit | ~4.5 GB |
| LoRA adapters | ~0.3 GB |
| Optimizer states | ~1.5 GB |
| Activations (grad ckpt) | ~1.2 GB |
| **Total** | **~7.5 GB** |

Tight on 8GB 4060 but should fit with gradient checkpointing enabled.

**If OOM:** reduce `per_device_train_batch_size` to 1, increase `gradient_accumulation_steps` to 8.

## Inference sanity check

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "data/artifacts/qwen25-7b-f1-lora/final",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

prompt = """### Instruction:
You are an expert Formula 1 commentator. Generate professional race commentary.

### Input:
LAP 52/56 | Monza 2024 Race
OBSERVED: VER P1, HAM P2, gap 1.2s. HAM MEDIUM (age 8), VER HARD (age 20). DRS available next lap.
INFERRED: Hamilton closing 0.4s/lap. Overtake likely within 3 laps.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Checklist

- [ ] Create/collect 50+ training examples in `data/datasets/sft_train.jsonl`
- [ ] Install unsloth + deps on 4060 laptop
- [ ] Run training (~30-60 min for 3 epochs on 50 examples)
- [ ] Verify adapter saves to `data/artifacts/qwen25-7b-f1-lora/final`
- [ ] Run inference sanity check
- [ ] Copy adapter back to M1 for integration with commentary generator
