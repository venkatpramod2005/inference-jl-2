#!/usr/bin/env python3
"""Run inference with the trained Qwen2.5 LoRA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from unsloth import FastLanguageModel


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER_DIR = PROJECT_DIR / "models" / "lora_adapters"
TEST_PROMPTS = ["What is AI?", "Explain neural networks.", "Tell me a joke."]


def format_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate responses from the fine-tuned Qwen2.5 adapter.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter_dir", default=str(DEFAULT_ADAPTER_DIR))
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()


def ensure_adapter_exists(adapter_dir: Path) -> None:
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"LoRA adapter not found at {adapter_dir}. Run train.py first.")


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model: Any, tokenizer: Any, instruction: str, args: argparse.Namespace) -> str:
    prompt = format_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()


def main() -> None:
    args = parse_args()
    ensure_adapter_exists(Path(args.adapter_dir))
    if not torch.cuda.is_available():
        print("WARNING: CUDA is unavailable. Qwen2.5-7B inference will be slow or may not fit in memory.")

    model, tokenizer = load_model(args)
    prompts = [args.prompt] if args.prompt else TEST_PROMPTS
    for prompt in prompts:
        print("\n" + "=" * 80)
        print(f"Prompt: {prompt}")
        print("-" * 80)
        print(generate(model, tokenizer, prompt, args))


if __name__ == "__main__":
    main()
