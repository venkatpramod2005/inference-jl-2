# Qwen2.5-7B Chatbot Fine-Tuning using Unsloth

This project fine-tunes `Qwen/Qwen2.5-7B-Instruct` on the Databricks Dolly 15K instruction dataset with Unsloth, 4-bit quantization, and LoRA/QLoRA. It includes both flat assignment entrypoints (`train.py`, `inference.py`, `app.py`) and module folders for deeper experiments.

## Project Objective

Build an instruction-following chatbot by loading Qwen2.5-7B in 4-bit precision, training lightweight LoRA adapters on Dolly 15K, and serving the trained adapter through a Streamlit web chat interface on JarvisLabs.

## Technologies Used

- Python
- Unsloth
- QLoRA and LoRA
- Hugging Face Transformers, Datasets, PEFT, TRL
- BitsAndBytes
- Streamlit
- JarvisLabs L4 GPU

## Model Used

[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

## Dataset Used

[Databricks Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

Rows are formatted as:

```text
### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}
```

The context block is omitted when the row has no context.

## Project Structure

```text
inference-jl-2/
├── train.py
├── inference.py
├── app.py
├── requirements.txt
├── setup.sh
├── README.md
├── DEPLOYMENT_GUIDE.md
├── models/
├── outputs/
├── screenshots/
├── scripts/
│   ├── start_streamlit.sh
│   └── stop_streamlit.sh
├── train/
│   ├── dataset_utils.py
│   └── finetune.py
├── inference/
│   └── infer.py
└── chatbot/
    └── app.py
```

## Setup on JarvisLabs

Create or open a JarvisLabs L4 GPU instance, then clone this repository:

```bash
cd /home
git clone https://github.com/venkatpramod2005/inference-jl-2.git
cd inference-jl-2
bash setup.sh
source .venv/bin/activate
```

If your instance persists files under `/root`, clone there instead. The main scripts resolve paths relative to the repository root.

## Training Process

Smoke test:

```bash
python train.py --max_samples 16 --max_steps 1
```

Quick QLoRA run:

```bash
python train.py --max_steps 60
```

Fuller one-epoch run:

```bash
python train.py --max_steps -1 --num_train_epochs 1
```

Optimization techniques:

- 4-bit quantization
- LoRA rank `16`
- QLoRA training on quantized base weights
- Unsloth memory-efficient loading
- Gradient checkpointing
- Mixed precision based on GPU support
- `adamw_8bit` optimizer

Outputs:

```text
models/lora_adapters/
outputs/checkpoints/
```

## Inference

After training:

```bash
python inference.py
```

Custom prompt:

```bash
python inference.py --prompt "Explain neural networks in simple words."
```

Default test prompts:

- What is AI?
- Explain neural networks.
- Tell me a joke.

## Streamlit Chatbot UI

Run locally or on JarvisLabs:

```bash
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
```

Background helper:

```bash
bash scripts/start_streamlit.sh
tail -f outputs/streamlit.log
```

Stop:

```bash
bash scripts/stop_streamlit.sh
```

JarvisLabs endpoint:

[https://ac5f144115441.notebooksn.jarvislabs.net](https://ac5f144115441.notebooksn.jarvislabs.net)

If the endpoint does not load, confirm that JarvisLabs exposes port `7860` and that Streamlit is bound to `0.0.0.0`.

## Screenshots

Add final screenshots to `screenshots/`:

- Training logs showing adapter save
- Streamlit chatbot UI
- Example chatbot response

## Troubleshooting

CUDA unavailable:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

CUDA out of memory:

```bash
python train.py --batch_size 1 --gradient_accumulation_steps 8 --max_seq_length 1024
```

Missing adapter:

```bash
python train.py --max_steps 60
```

Hugging Face download issue:

```bash
huggingface-cli login
```

## Future Improvements

- Train for more epochs and compare validation prompts
- Add RAG integration
- Add durable multi-turn memory
- Improve the UI with prompt presets and export
- Export a merged model or GGUF for alternate serving stacks

