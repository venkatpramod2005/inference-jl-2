"""Dataset formatting helpers shared by training and notebooks."""

from __future__ import annotations

from datasets import load_dataset


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


def format_prompt_for_generation(instruction: str, context: str | None = None) -> str:
    instruction = (instruction or "").strip()
    context = (context or "").strip()
    if context:
        return f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def load_formatted_dolly_dataset(max_samples: int | None = None):
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset.map(
        lambda row: {"text": format_dolly_record(row)},
        remove_columns=dataset.column_names,
        desc="Formatting Dolly 15K",
    )

