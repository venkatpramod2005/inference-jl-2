#!/usr/bin/env python3
"""Compare base Qwen2.5 responses with the fine-tuned LoRA adapter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from unsloth import FastLanguageModel
from peft import PeftModel

PROJECT_DIR = Path("/home/user/inference-jl-2")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER_DIR = PROJECT_DIR / "models" / "lora_adapters"

TRAIN_DIR = PROJECT_DIR / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from dataset_utils import format_prompt_for_generation  # noqa: E402


TEST_PROMPTS = [
    "What is machine learning?",
    "Explain the water cycle.",
    "Write a short poem about the ocean.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run base vs fine-tuned inference comparison.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter_dir", default=str(DEFAULT_ADAPTER_DIR))
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()


def print_vram(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"[VRAM:{label}] CUDA unavailable")
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[VRAM:{label}] allocated={allocated:.2f}GB reserved={reserved:.2f}GB")


def ensure_adapter_exists(path: str) -> None:
    adapter_path = Path(path)
    if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Fine-tuned adapter not found at {adapter_path}. Run python train/finetune.py first."
        )


def load_base_model(model_name: str, max_seq_length: int) -> tuple[Any, Any]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)
    print_vram("after base load")
    return model, tokenizer


def load_tuned_model(model_name: str, adapter_dir: str, max_seq_length: int) -> tuple[Any, Any]:
    model, tokenizer = load_base_model(model_name, max_seq_length)
    model = PeftModel.from_pretrained(model, adapter_dir)
    FastLanguageModel.for_inference(model)
    print_vram("after adapter load")
    return model, tokenizer


def generate(model: Any, tokenizer: Any, prompt: str, args: argparse.Namespace) -> str:
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
    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable. This comparison is intended for the JarvisLabs L4 GPU.")

    ensure_adapter_exists(args.adapter_dir)

    prompts = [format_prompt_for_generation(prompt) for prompt in TEST_PROMPTS]

    base_model, base_tokenizer = load_base_model(args.model_name, args.max_seq_length)
    base_outputs = [generate(base_model, base_tokenizer, prompt, args) for prompt in prompts]
    del base_model
    torch.cuda.empty_cache()

    tuned_model, tuned_tokenizer = load_tuned_model(args.model_name, args.adapter_dir, args.max_seq_length)
    tuned_outputs = [generate(tuned_model, tuned_tokenizer, prompt, args) for prompt in prompts]

    for idx, (raw_prompt, base, tuned) in enumerate(zip(TEST_PROMPTS, base_outputs, tuned_outputs), start=1):
        print("\n" + "=" * 88)
        print(f"Prompt {idx}: {raw_prompt}")
        print("-" * 88)
        print("Base model response:")
        print(base)
        print("-" * 88)
        print("Fine-tuned response:")
        print(tuned)


if __name__ == "__main__":
    main()
