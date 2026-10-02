---
title: N-ATLaS
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.29.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# N-ATLaS serving app

Serves [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) on a GPU for the AgriVoice backend. It follows the model card's usage: official weights in float16, the chat template with today's `date_string`, `repetition_penalty=1.12`, greedy decoding, up to 1000 new tokens.

| File | Purpose |
|---|---|
| `app.py` | Gradio app exposing `/generate` (plus a chat page for manual testing) |
| `build_kaggle_notebook.py` | Builds the Kaggle notebook from `app.py` (run after editing it) |
| `kaggle/agrivoice_natlas.ipynb` | Runs `app.py` on Kaggle's free 2 × T4 GPUs and registers its public link with the backend |
| `requirements.txt`, this README's header | Hugging Face Space configuration |

## API

`/generate` takes chat messages and returns the reply text:

```python
from gradio_client import Client

client = Client("https://xxxx.gradio.live")   # or a Space id, "owner/name"
reply = client.predict(
    [{"role": "user", "content": "How do I stop armyworms on my maize?"}],
    512,                                      # max_new_tokens (≤ 1000)
    api_name="/generate",
)
```

## Running it

- **Kaggle (used in production):** see [docs/deployment.md](../docs/deployment.md). The AgriVoice admin dashboard starts, stops and schedules the notebook.
- **Hugging Face Space:** create a Gradio Space from this folder, select **ZeroGPU** hardware and add an `HF_TOKEN` secret from an account that accepted the N-ATLaS licence. ZeroGPU time is charged to the caller's daily quota, so the backend calls it with its own token.
- **Locally, for testing without a GPU:** `MODEL_ID=HuggingFaceTB/SmolLM2-135M-Instruct DEVICE=cpu python app.py` runs the same app with a tiny stand-in model.

`DEVICE` selects placement: `cuda` (default; ZeroGPU), `auto` (spread across several GPUs, e.g. Kaggle's two T4s) or `cpu`.
