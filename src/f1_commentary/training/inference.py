"""Phase 8: Inference sanity check for the fine-tuned LoRA adapter.

Usage:
    python -m f1_commentary.training.inference [--adapter-dir PATH] [--prompt TEXT]
"""

from __future__ import annotations

import argparse

from unsloth import FastLanguageModel

DEFAULT_ADAPTER_DIR = "data/artifacts/qwen25-7b-f1-lora/final"
DEFAULT_MAX_SEQ_LEN = 2048

DEFAULT_PROMPT = """\
### Instruction:
You are an expert Formula 1 commentator. Generate professional race commentary.

### Input:
LAP 52/56 | Monza 2024 Race
OBSERVED: VER P1, HAM P2, gap 1.2s. HAM MEDIUM (age 8), VER HARD (age 20). DRS available next lap.
INFERRED: Hamilton closing 0.4s/lap. Overtake likely within 3 laps.

### Response:
"""


def run_inference(
    adapter_dir: str = DEFAULT_ADAPTER_DIR,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """Load the fine-tuned model and generate commentary."""
    print(f"Loading adapter from: {adapter_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        adapter_dir,
        max_seq_length=DEFAULT_MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the response portion
    response_marker = "### Response:\n"
    if response_marker in result:
        result = result.split(response_marker, 1)[1].strip()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned F1 commentary model")
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR, help="Path to saved adapter")
    parser.add_argument("--prompt", default=None, help="Custom prompt (overrides default)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    prompt = args.prompt or DEFAULT_PROMPT
    result = run_inference(
        adapter_dir=args.adapter_dir,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print("\n" + "=" * 60)
    print("GENERATED COMMENTARY:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
