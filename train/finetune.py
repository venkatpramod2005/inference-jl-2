#!/usr/bin/env python3
"""Fine-tune Qwen2.5 7B on Dolly 15K with Unsloth QLoRA."""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

PROJECT_DIR = Path("/home/user/inference-jl-2")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-7B"
DEFAULT_CHECKPOINT_DIR = PROJECT_DIR / "models" / "checkpoints"
DEFAULT_LORA_DIR = PROJECT_DIR / "models" / "lora_adapters"
DEFAULT_MERGED_DIR = PROJECT_DIR / "models" / "merged_model"
DEFAULT_GGUF_DIR = PROJECT_DIR / "models" / "gguf"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_utils import load_formatted_dolly_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5 7B QLoRA fine-tuning on Dolly 15K.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--fallback_model_name", default=FALLBACK_MODEL)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=60, help="Quick test default. Set -1 with --num_train_epochs for full epochs.")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--lora_dir", default=str(DEFAULT_LORA_DIR))
    parser.add_argument("--merged_dir", default=str(DEFAULT_MERGED_DIR))
    parser.add_argument("--gguf_dir", default=str(DEFAULT_GGUF_DIR))
    parser.add_argument("--merge_model", action="store_true", help="Also save a merged 16-bit model.")
    parser.add_argument("--export_gguf", action="store_true", help="Export GGUF if llama.cpp support is available.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_num_proc", type=int, default=2)
    return parser.parse_args()


def require_cuda(torch_module: Any) -> None:
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is not available. Verify the JarvisLabs L4 instance with nvidia-smi before training.")
    print_gpu_status("initial")


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


def ensure_tokenizer_padding(tokenizer: Any) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def load_model_and_tokenizer(args: argparse.Namespace, FastLanguageModel: Any, torch_module: Any) -> tuple[Any, Any, str]:
    common_kwargs = dict(
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    for model_name in (args.model_name, args.fallback_model_name):
        try:
            print(f"Loading {model_name} in 4-bit with Unsloth...")
            model, tokenizer = FastLanguageModel.from_pretrained(model_name=model_name, **common_kwargs)
            ensure_tokenizer_padding(tokenizer)
            print_vram(torch_module, "after model load")
            return model, tokenizer, model_name
        except Exception as exc:
            print(f"Failed to load {model_name}: {exc}")
            if model_name == args.fallback_model_name:
                raise
            print(f"Falling back to {args.fallback_model_name}...")
    raise RuntimeError("No model could be loaded.")


def add_lora_adapters(args: argparse.Namespace, FastLanguageModel: Any, model: Any) -> Any:
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
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")
    return model


class GpuLoggingCallback:
    def __init__(self, torch_module: Any, logging_steps: int):
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def __init__(self, outer: "GpuLoggingCallback"):
                self.outer = outer
                self.last_time = time.time()

            def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
                if state.global_step % self.outer.logging_steps != 0:
                    return
                elapsed = max(time.time() - self.last_time, 1e-6)
                self.last_time = time.time()
                loss = logs.get("loss", "n/a") if logs else "n/a"
                print(f"[train] step={state.global_step} loss={loss} elapsed_since_last_log={elapsed:.1f}s")
                print_vram(self.outer.torch, f"step {state.global_step}")

        self.torch = torch_module
        self.logging_steps = logging_steps
        self.callback = _Callback(self)


def build_sft_trainer(args: argparse.Namespace, model: Any, tokenizer: Any, train_dataset: Any) -> Any:
    from transformers import TrainingArguments
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        SFTConfig = None
        from trl import SFTTrainer

    use_bf16 = False
    try:
        import torch

        use_bf16 = torch.cuda.is_bf16_supported()
    except Exception:
        pass

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
        if "dataset_num_proc" in sft_fields:
            sft_kwargs["dataset_num_proc"] = args.dataset_num_proc
        sft_args = SFTConfig(**sft_kwargs)
        try:
            return SFTTrainer(model=model, processing_class=tokenizer, train_dataset=train_dataset, args=sft_args)
        except TypeError:
            return SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_dataset, args=sft_args)

    training_args = TrainingArguments(**common_args)
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        dataset_num_proc=args.dataset_num_proc,
        args=training_args,
    )


def save_artifacts(args: argparse.Namespace, model: Any, tokenizer: Any) -> None:
    Path(args.lora_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.lora_dir)
    tokenizer.save_pretrained(args.lora_dir)
    print(f"Saved LoRA adapters and tokenizer to {args.lora_dir}")

    if args.merge_model:
        Path(args.merged_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained_merged(args.merged_dir, tokenizer, save_method="merged_16bit")
        print(f"Saved merged model to {args.merged_dir}")

    if args.export_gguf:
        try:
            Path(args.gguf_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained_gguf(args.gguf_dir, tokenizer, quantization_method="q4_k_m")
            print(f"Saved GGUF export to {args.gguf_dir}")
        except Exception as exc:
            print(f"GGUF export skipped: llama.cpp export support is unavailable or failed: {exc}")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from unsloth import FastLanguageModel
    import torch

    require_cuda(torch)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    train_dataset = load_formatted_dolly_dataset(max_samples=args.max_samples)
    print(f"Training examples: {len(train_dataset):,}")

    model, tokenizer, loaded_name = load_model_and_tokenizer(args, FastLanguageModel, torch)
    print(f"Loaded model: {loaded_name}")

    model = add_lora_adapters(args, FastLanguageModel, model)
    print_vram(torch, "after LoRA setup")

    trainer = build_sft_trainer(args, model, tokenizer, train_dataset)
    trainer.add_callback(GpuLoggingCallback(torch, args.logging_steps).callback)

    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM. Try --batch_size 1, lower --max_seq_length, or reduce --max_steps for smoke tests.")
        torch.cuda.empty_cache()
        raise

    print("Recent trainer log history:")
    print(trainer.state.log_history[-5:])

    save_artifacts(args, model, tokenizer)
    gc.collect()
    torch.cuda.empty_cache()
    print_vram(torch, "final")


if __name__ == "__main__":
    main()
