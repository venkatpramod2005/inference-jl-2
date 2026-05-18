#!/usr/bin/env python3
"""Dataset helpers for Dolly 15K instruction tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

from datasets import Dataset, load_dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SUBSET_DIR = PROJECT_DIR / "train" / "dolly_500_sample"


def format_dolly_record(record: Mapping[str, str]) -> str:
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


def format_prompt_for_generation(instruction: str, context: str | None = None) -> str:
    instruction = (instruction or "").strip()
    context = (context or "").strip()

    if context:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Context:\n"
            f"{context}\n\n"
            "### Response:\n"
        )

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def load_formatted_dolly_dataset(max_samples: int | None = None) -> Dataset:
    train_ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    if max_samples is not None:
        train_ds = train_ds.select(range(min(max_samples, len(train_ds))))

    return train_ds.map(
        lambda example: {"text": format_dolly_record(example)},
        remove_columns=train_ds.column_names,
        desc="Formatting Dolly instruction examples",
    )


def preview_samples(dataset: Dataset | None = None, n: int = 3) -> None:
    dataset = dataset if dataset is not None else load_formatted_dolly_dataset()
    for idx, sample in enumerate(dataset.select(range(min(n, len(dataset))))):
        print(f"\n--- Sample {idx + 1} ---")
        print(sample["text"])


def save_quick_test_subset(
    output_dir: str | Path = DEFAULT_SUBSET_DIR,
    sample_count: int = 500,
) -> Dataset:
    output_path = Path(output_dir)
    dataset = load_formatted_dolly_dataset(max_samples=sample_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    print(f"Saved {len(dataset)} formatted samples to {output_path}")
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview and save formatted Dolly 15K samples.")
    parser.add_argument("--preview", type=int, default=3, help="Number of samples to print.")
    parser.add_argument("--subset-size", type=int, default=500, help="Quick-test subset size.")
    parser.add_argument("--subset-dir", default=str(DEFAULT_SUBSET_DIR), help="Output directory for quick-test subset.")
    parser.add_argument("--no-save", action="store_true", help="Preview only; do not save the quick-test subset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_formatted_dolly_dataset()
    preview_samples(dataset, args.preview)
    if not args.no_save:
        save_quick_test_subset(args.subset_dir, args.subset_size)


if __name__ == "__main__":
    main()
