# Qwen2.5-7B Chatbot Fine-Tuning using Unsloth

This project fine-tunes `Qwen/Qwen2.5-7B-Instruct` on the Databricks Dolly 15K instruction dataset with Unsloth, 4-bit quantization, and LoRA/QLoRA. It also includes a Streamlit chatbot UI for deployment on a JarvisLabs L4 GPU endpoint.

## Project Objective

Build an instruction-following chatbot by loading Qwen2.5-7B in 4-bit precision, fine-tuning lightweight LoRA adapters on Dolly 15K, and serving the trained adapter through a web chat interface.

## Technologies Used

- Python
- Unsloth
- QLoRA and LoRA
- Hugging Face Transformers, Datasets, PEFT, TRL
- BitsAndBytes 4-bit quantization
- Streamlit
- JarvisLabs L4 GPU

## Project Structure

```text
inference-jl-2/
├── train.py
├── inference.py
├── app.py
├── requirements.txt
├── setup.sh
├── README.md
├── models/
├── outputs/
├── screenshots/
└── scripts/
    ├── start_streamlit.sh
    └── stop_streamlit.sh
```

## Model Used

[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

## Dataset Used

[Databricks Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

The training script converts each row into this instruction format:

```text
### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}
```

The context block is omitted when the dataset row has no context.

## Setup on JarvisLabs

Create or open a JarvisLabs L4 GPU instance, then clone this repository under a persistent home directory:

```bash
cd /home
git clone https://github.com/venkatpramod2005/inference-jl-2.git
cd inference-jl-2
bash setup.sh
source .venv/bin/activate
```

If your instance uses `/root` as the persistent home, clone there instead. The code resolves paths relative to the repository, so it does not require a hardcoded `/home/user` directory.

## Training Process

Run a smoke test first:

```bash
python train.py --max_samples 16 --max_steps 1
```

Run a quick QLoRA training pass:

```bash
python train.py --max_steps 60
```

Run a fuller one-epoch training pass:

```bash
python train.py --max_steps -1 --num_train_epochs 1
```

Training uses:

- 4-bit model loading with Unsloth
- LoRA rank `16`
- LoRA alpha `16`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Gradient checkpointing with Unsloth
- Mixed precision based on GPU support
- `adamw_8bit` optimizer

The trained adapter is saved to:

```text
models/lora_adapters/
```

Checkpoints are saved under:

```text
outputs/checkpoints/
```

## Inference

After training, test the adapter:

```bash
python inference.py
```

Or send a custom prompt:

```bash
python inference.py --prompt "Explain neural networks in simple words."
```

Default test prompts:

- What is AI?
- Explain neural networks.
- Tell me a joke.

## Streamlit Deployment

Start the chatbot on JarvisLabs:

```bash
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
```

Or use the helper script:

```bash
bash scripts/start_streamlit.sh
tail -f outputs/streamlit.log
```

Stop the background server:

```bash
bash scripts/stop_streamlit.sh
```

Expected deployment endpoint:

[https://ac5f144115441.notebooksn.jarvislabs.net](https://ac5f144115441.notebooksn.jarvislabs.net)

If the endpoint does not load, confirm that JarvisLabs is exposing port `7860` for this instance and that Streamlit is bound to `0.0.0.0`.

## Screenshots

Add final screenshots to `screenshots/`:

- Training logs showing successful LoRA adapter save
- Streamlit chatbot page
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

Missing adapter in the app:

```bash
python train.py --max_steps 60
```

Hugging Face download issue:

```bash
huggingface-cli login
```

## Future Improvements

- Train for more epochs and compare validation prompts
- Add RAG over project documents
- Add durable multi-turn conversation memory
- Improve the UI with prompt presets and response export
- Export a merged model or GGUF artifact for alternate serving stacks

