#!/usr/bin/env python3
"""Fine-tune Qwen2.5-7B-Instruct on Databricks Dolly 15K with Unsloth QLoRA."""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

from datasets import load_dataset


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "checkpoints"
DEFAULT_ADAPTER_DIR = PROJECT_DIR / "models" / "lora_adapters"


def format_dolly_record(record: dict[str, str]) -> str:
    instruction = (record.get("instruction") or "").strip()
    context = (record.get("context") or "").strip()
    response = (record.get("response") or "").strip()

    if context:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Context:\n"
            f"{context}\n\n"
            "### Response:\n"
            f"{response}"
        )

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{response}"
    )


def load_formatted_dataset(max_samples: int | None = None):
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset.map(
        lambda row: {"text": format_dolly_record(row)},
        remove_columns=dataset.column_names,
        desc="Formatting Dolly 15K",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsloth QLoRA training for Qwen2.5-7B-Instruct.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=60, help="Use -1 to train by epochs.")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--adapter_dir", default=str(DEFAULT_ADAPTER_DIR))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def print_gpu_status(label: str) -> None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(f"[GPU:{label}] {result.stdout.strip()}")
    except Exception as exc:
        print(f"[GPU:{label}] nvidia-smi unavailable: {exc}")


def print_vram(torch_module: Any, label: str) -> None:
    if not torch_module.cuda.is_available():
        return
    allocated = torch_module.cuda.memory_allocated() / 1024**3
    reserved = torch_module.cuda.memory_reserved() / 1024**3
    peak = torch_module.cuda.max_memory_allocated() / 1024**3
    print(f"[VRAM:{label}] allocated={allocated:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB")
    print_gpu_status(label)


def build_trainer(args: argparse.Namespace, model: Any, tokenizer: Any, train_dataset: Any) -> Any:
    from transformers import TrainingArguments

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        SFTConfig = None
        from trl import SFTTrainer

    import torch

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    common_args = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        optim="adamw_8bit",
        seed=args.seed,
        report_to="none",
        save_total_limit=2,
    )
    if args.max_steps and args.max_steps > 0:
        common_args["max_steps"] = args.max_steps
    else:
        common_args["num_train_epochs"] = args.num_train_epochs

    if SFTConfig is not None:
        sft_fields = {field.name for field in fields(SFTConfig)}
        sft_kwargs = {**common_args, "dataset_text_field": "text", "packing": False}
        if "max_seq_length" in sft_fields:
            sft_kwargs["max_seq_length"] = args.max_seq_length
        elif "max_length" in sft_fields:
            sft_kwargs["max_length"] = args.max_seq_length
        sft_args = SFTConfig(**sft_kwargs)
        try:
            return SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train_dataset, args=sft_args)
        except TypeError:
            return SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_dataset, args=sft_args)

    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=TrainingArguments(**common_args),
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from unsloth import FastLanguageModel
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use a JarvisLabs GPU instance before training.")

    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    print_gpu_status("initial")

    train_dataset = load_formatted_dataset(args.max_samples)
    print(f"Training examples: {len(train_dataset):,}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print_vram(torch, "after 4-bit model load")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    print_vram(torch, "after LoRA setup")

    trainer = build_trainer(args, model, tokenizer, train_dataset)
    trainer.train()

    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved fine-tuned LoRA adapter to {adapter_dir}")

    gc.collect()
    torch.cuda.empty_cache()
    print_vram(torch, "final")


if __name__ == "__main__":
    main()
