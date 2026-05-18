# inference-jl-2

End-to-end Qwen 2.5 7B QLoRA fine-tuning on Databricks Dolly 15K for a JarvisLabs NVIDIA L4 instance. The project includes dataset formatting, Unsloth training, LoRA inference comparison, and a Streamlit chatbot.

## Prerequisites

- JarvisLabs NVIDIA L4 GPU instance
- CUDA visible through `nvidia-smi`
- Python 3.10+
- Git and GitHub access
- All persistent files under `/home/user/inference-jl-2`

The L4 has about 23 GB VRAM. Keep checkpoints, adapters, and datasets under `/home/user/inference-jl-2` so they survive pause/resume.

## Setup

```bash
mkdir -p /home/user
cd /home/user
git clone <your-github-repo-url> inference-jl-2
cd /home/user/inference-jl-2
bash setup.sh
source .venv/bin/activate
```

`setup.sh` verifies CUDA with `nvidia-smi`, creates a local Python environment, installs the Python dependencies, and prints GPU memory information. If Python 3.10+ is not already available, it installs Miniconda under `/home/user/miniconda3` and creates the project environment at `/home/user/inference-jl-2/.venv`.

## Project Structure

```text
inference-jl-2/
├── README.md
├── requirements.txt
├── setup.sh
├── train/
│   ├── finetune.py
│   └── dataset_utils.py
├── inference/
│   └── infer.py
├── chatbot/
│   └── app.py
└── models/
    └── .gitkeep
```

## Dataset

Preview formatted Dolly samples and save a 500-sample quick-test subset:

```bash
cd /home/user/inference-jl-2
source .venv/bin/activate
python train/dataset_utils.py
```

The formatter uses:

```text
### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}
```

When `context` is empty, the Context section is omitted.

## Training

Smoke test:

```bash
cd /home/user/inference-jl-2
source .venv/bin/activate
python train/finetune.py --max_samples 16 --max_steps 1
```

Quick QLoRA run:

```bash
python train/finetune.py --max_steps 60
```

For a fuller run, disable step-based training and use epochs:

```bash
python train/finetune.py --max_steps -1 --num_train_epochs 1
```

Adapters and tokenizer are saved to:

```text
/home/user/inference-jl-2/models/lora_adapters
```

Optional merged model and GGUF export:

```bash
python train/finetune.py --max_steps 60 --merge_model
python train/finetune.py --max_steps 60 --export_gguf
```

GGUF export only runs if the required llama.cpp export support is available.

## Inference

After training:

```bash
cd /home/user/inference-jl-2
source .venv/bin/activate
python inference/infer.py
```

The script prints base-model and fine-tuned responses side by side for:

- `What is machine learning?`
- `Explain the water cycle.`
- `Write a short poem about the ocean.`

## Streamlit Chatbot

Launch the chatbot:

```bash
cd /home/user/inference-jl-2
source .venv/bin/activate
streamlit run /home/user/inference-jl-2/chatbot/app.py --server.port 8501 --server.address 0.0.0.0
```

The app provides:

- Chat history in the session
- Right-aligned user bubbles and left-aligned assistant bubbles
- Sidebar controls for max new tokens, temperature, and top-p
- GPU/model information
- Clear error if LoRA adapter files are missing
- CPU fallback warning when CUDA is unavailable

## JarvisLabs Access And Port Forwarding

If your JarvisLabs instance exposes custom HTTP ports, expose port `8501` and open the generated endpoint URL.

For SSH forwarding from your workstation:

```bash
ssh -o StrictHostKeyChecking=no -L 8501:127.0.0.1:8501 root@217.18.55.196
```

Then open:

```text
http://127.0.0.1:8501
```

If using the provided JarvisLabs notebook endpoint, confirm the platform is routing port `8501` or start Streamlit on an exposed port.

## Model Architecture

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Fallback base model: `Qwen/Qwen2.5-7B`
- Fine-tuning method: QLoRA
- Quantization: 4-bit via Unsloth
- LoRA rank: 16
- LoRA alpha: 16
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Sequence length: 2048
- Optimizer: `adamw_8bit`

## Expected VRAM Usage

| Stage | Expected VRAM on L4 |
|---|---:|
| Idle after CUDA check | 0-1 GB |
| 4-bit model loaded | 5-8 GB |
| LoRA adapters attached | 6-9 GB |
| Training, batch 2, grad accum 4, seq 2048 | 14-22 GB |
| Inference with LoRA | 6-10 GB |
| Merged 16-bit save | May exceed 23 GB; use only if needed |

## Troubleshooting

### `nvidia-smi` not found

Use a GPU JarvisLabs template or restart on an NVIDIA L4 instance.

### Python is older than 3.10

Run `bash setup.sh`. It will install Miniconda under `/home/user/miniconda3` and create a Python 3.10 environment at `/home/user/inference-jl-2/.venv`.

### CUDA out of memory

Try:

```bash
python train/finetune.py --batch_size 1 --gradient_accumulation_steps 8 --max_seq_length 1024
```

### Adapter not found in Streamlit or inference

Run training first:

```bash
python train/finetune.py --max_steps 60
```

### Hugging Face download errors

Check network access and retry. If a gated or private model is used later, authenticate with:

```bash
huggingface-cli login
```

### Streamlit is not reachable

Start with `--server.address 0.0.0.0`, expose or forward port `8501`, and verify the process is running:

```bash
ps -ef | grep streamlit
```
