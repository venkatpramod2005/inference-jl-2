#!/usr/bin/env python3
"""Streamlit chatbot for the fine-tuned Qwen2.5 LoRA adapter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import torch

PROJECT_DIR = Path("/home/user/inference-jl-2")
TRAIN_DIR = PROJECT_DIR / "train"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = PROJECT_DIR / "models" / "lora_adapters"

if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from dataset_utils import format_prompt_for_generation  # noqa: E402


st.set_page_config(page_title="Qwen 2.5 Fine-tuned Chatbot", page_icon=":speech_balloon:", layout="centered")

st.markdown(
    """
    <style>
    .chat-row {
        display: flex;
        margin: 0.45rem 0;
    }
    .chat-row.user {
        justify-content: flex-end;
    }
    .chat-row.assistant {
        justify-content: flex-start;
    }
    .chat-bubble {
        max-width: 78%;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        line-height: 1.45;
        border: 1px solid rgba(49, 51, 63, 0.18);
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }
    .chat-row.user .chat-bubble {
        background: #e7f0ff;
    }
    .chat-row.assistant .chat-bubble {
        background: #f7f7f8;
    }
    .subtitle {
        color: #5f6368;
        margin-top: -0.55rem;
        margin-bottom: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def gpu_info() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or torch.cuda.get_device_name(0)
    except Exception:
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        used = torch.cuda.memory_allocated() / 1024**3
        return f"{torch.cuda.get_device_name(0)}, {used:.2f}GB used / {total:.2f}GB total"


def adapter_ready() -> bool:
    return ADAPTER_DIR.exists() and (ADAPTER_DIR / "adapter_config.json").exists()


@st.cache_resource(show_spinner="Loading fine-tuned model...")
def load_model() -> tuple[Any, Any, str]:
    if not adapter_ready():
        raise FileNotFoundError(
            f"Model files not found at {ADAPTER_DIR}. Run `python train/finetune.py` first."
    )

    if torch.cuda.is_available():
        from unsloth import FastLanguageModel
        from peft import PeftModel

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
        return model, tokenizer, "cuda"

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, "cpu"


def generate_response(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    formatted = format_prompt_for_generation(prompt)
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
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
    return decoded[len(formatted) :].strip() if decoded.startswith(formatted) else decoded.strip()


with st.sidebar:
    st.header("Model")
    st.write(f"Name: `{DEFAULT_MODEL}`")
    st.write(f"Adapter: `{ADAPTER_DIR}`")
    st.write(f"GPU: `{gpu_info()}`")
    if not torch.cuda.is_available():
        st.warning("CUDA unavailable. CPU inference is supported as a fallback but will be slow.")

    st.header("Generation")
    max_new_tokens = st.slider("Max new tokens", 64, 512, 256, step=32)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.title("Qwen 2.5 Fine-tuned Chatbot")
st.markdown('<div class="subtitle">Powered by Unsloth + QLoRA on Dolly 15K</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    model, tokenizer, device_label = load_model()
    st.sidebar.success(f"Loaded on {device_label}")
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info("Train first: `cd /home/user/inference-jl-2 && python train/finetune.py --max_steps 60`")
    st.stop()
except Exception as exc:
    st.error(f"Model initialization failed: {exc}")
    st.stop()

for message in st.session_state.messages:
    role = message["role"]
    st.markdown(
        f'<div class="chat-row {role}"><div class="chat-bubble">{message["content"]}</div></div>',
        unsafe_allow_html=True,
    )

with st.form("chat-form", clear_on_submit=True):
    user_prompt = st.text_input("Ask me anything...", label_visibility="collapsed", placeholder="Ask me anything...")
    send = st.form_submit_button("Send", use_container_width=True)

if send and user_prompt.strip():
    prompt = user_prompt.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Generating response..."):
        try:
            response = generate_response(model, tokenizer, prompt, max_new_tokens, temperature, top_p)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            response = "CUDA ran out of memory. Try lowering max new tokens or restart the app."
        except Exception as exc:
            response = f"Generation failed: {exc}"
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
