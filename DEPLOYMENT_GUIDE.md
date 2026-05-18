# JarvisLabs Deployment Guide

## 1. Install

```bash
cd /home
git clone https://github.com/venkatpramod2005/inference-jl-2.git
cd inference-jl-2
bash setup.sh
source .venv/bin/activate
```

## 2. Train

```bash
python train.py --max_steps 60
```

For a quick validation only:

```bash
python train.py --max_samples 16 --max_steps 1
```

## 3. Test Inference

```bash
python inference.py
```

## 4. Start Streamlit

```bash
streamlit run app.py --server.port 6006 --server.address 0.0.0.0
```

Or run in the background:

```bash
bash scripts/start_streamlit.sh
tail -f outputs/streamlit.log
```

Use port `7860` only when the JarvisLabs instance was created or resumed with that custom HTTP port exposed:

```bash
STREAMLIT_PORT=7860 bash scripts/start_streamlit.sh
```

## 5. Open Endpoint

Open:

```text
https://ac5f144115441.notebooksn.jarvislabs.net
```

## 6. Verify

- The app loads without adapter errors.
- The sidebar shows CUDA/GPU information.
- A prompt generates a model response.
- `nvidia-smi` shows GPU memory usage while the app is running.
