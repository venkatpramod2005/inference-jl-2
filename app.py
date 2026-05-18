#!/usr/bin/env python3
"""Streamlit chatbot UI for the fine-tuned Qwen2.5 LoRA adapter."""

from __future__ import annotations

import html
import subprocess
from pathlib import Path
from typing import Any

import streamlit as st
import torch


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = PROJECT_DIR / "models" / "lora_adapters"


def format_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"


def gpu_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.stdout.strip():
            name, used, total = [part.strip() for part in result.stdout.strip().split(",")]
            return f"{name}: {used} MB / {total} MB"
    except Exception:
        pass
    return torch.cuda.get_device_name(0)


def adapter_ready() -> bool:
    return (ADAPTER_DIR / "adapter_config.json").exists()


@st.cache_resource(show_spinner="Loading fine-tuned Qwen2.5 adapter...")
def load_model() -> tuple[Any, Any]:
    if not adapter_ready():
        raise FileNotFoundError(f"Adapter files not found at {ADAPTER_DIR}. Run `python train.py` first.")

    from peft import PeftModel
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=DEFAULT_MODEL,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    FastLanguageModel.for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_response(model: Any, tokenizer: Any, instruction: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    prompt = format_prompt(instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()


st.set_page_config(page_title="Qwen2.5 Dolly Chatbot", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 1040px; padding-top: 1.5rem; }
    .chat-row { display: flex; margin: 0.65rem 0; }
    .chat-row.user { justify-content: flex-end; }
    .bubble {
        max-width: 78%;
        border: 1px solid #d8dee9;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        line-height: 1.45;
    }
    .user .bubble { background: #e8f1ff; border-color: #b8cef2; }
    .assistant .bubble { background: #f7f7f4; border-color: #ddddda; }
    .status { color: #555; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Qwen2.5-7B Chatbot Fine-Tuned with Unsloth")
st.caption("QLoRA adapter trained on Databricks Dolly 15K. Designed for a JarvisLabs L4 GPU.")

with st.sidebar:
    st.subheader("Generation")
    max_new_tokens = st.slider("Max new tokens", 32, 1024, 256, step=32)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, step=0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
    st.subheader("Runtime")
    st.markdown(f'<p class="status">{html.escape(gpu_summary())}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="status">Adapter: {html.escape(str(ADAPTER_DIR))}</p>', unsafe_allow_html=True)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    model, tokenizer = load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()

for message in st.session_state.messages:
    role = message["role"]
    content = html.escape(message["content"])
    st.markdown(f'<div class="chat-row {role}"><div class="bubble">{content}</div></div>', unsafe_allow_html=True)

user_prompt = st.chat_input("Ask the fine-tuned chatbot...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.spinner("Generating response..."):
        try:
            response = generate_response(model, tokenizer, user_prompt, max_new_tokens, temperature, top_p)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            response = "CUDA ran out of memory. Lower max tokens or restart the app."
        except Exception as exc:
            response = f"Generation failed: {exc}"
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
