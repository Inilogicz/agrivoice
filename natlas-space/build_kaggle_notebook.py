"""
Build the Kaggle notebook from app.py.

The notebook runs the exact same N-ATLaS app as the Hugging Face Space, on
Kaggle's free 2 × T4 GPUs, exposes it through a public *.gradio.live link,
and registers that link with the AgriVoice backend.

Two ways to run it:
  - By hand on kaggle.com: uses Kaggle Secrets, registers with the admin token.
  - From the AgriVoice admin dashboard: the backend fills in RUN_CONFIG with a
    one-run key and pushes the notebook through the Kaggle API; the notebook
    checks in every minute and stops when the dashboard says so.

Writes kaggle/agrivoice_natlas.ipynb and the backend's copy of the template.
Run after editing app.py:  python build_kaggle_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "kaggle" / "agrivoice_natlas.ipynb"
BACKEND_TEMPLATE = HERE.parent / "backend" / "app" / "resources" / "natlas_kaggle.ipynb"

# The backend replaces this exact line with the run's settings
RUN_CONFIG_LINE = "RUN_CONFIG = {}"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


SETUP = """
# AgriVoice: N-ATLaS on Kaggle

Runs the official [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) on Kaggle's free GPUs and connects it to the AgriVoice backend.

## One-time setup
1. **Settings → Accelerator → GPU T4 x2** (N-ATLaS needs both GPUs).
2. **Settings → Internet → On** (requires a phone-verified Kaggle account).
3. **Add-ons → Secrets**, add and attach to this notebook:
   - `HF_TOKEN`: Hugging Face token of an account that accepted the N-ATLaS licence
   - `AGRIVOICE_URL`: your backend, e.g. `https://your-username-agrivoice.hf.space` *(optional)*
   - `AGRIVOICE_ADMIN_TOKEN`: the backend's `ADMIN_TOKEN` *(optional)*

   Secrets only work for runs started from the Kaggle website. For runs started
   through the Kaggle API, put the same values in a **private** dataset named
   `agrivoice-secrets`, one file per secret (`hf_token`, `agrivoice_url`,
   `agrivoice_admin_token`), and attach it to this notebook.

## Each session
- **Easiest:** press **Start** in the AgriVoice admin dashboard (N-ATLaS tab). It runs this notebook for you and **Stop** shuts it down. Needs the private `agrivoice-secrets` dataset.
- **By hand:** *Save Version → Save & Run All (Commit)*. Set `RUN_HOURS` in the last cell first (default 2; ~9 on review days).

The last cell prints the public N-ATLaS link and registers it with the backend. Without backend settings, paste the link into the backend's `/admin` page.
"""

CONFIG = f"""
# Filled in by the AgriVoice backend when it starts this notebook from /admin.
# Leave it empty for runs started by hand (Kaggle Secrets are used instead).
{RUN_CONFIG_LINE}
"""

INSTALL = """
!pip install -q "gradio==6.29.0" spaces "transformers==5.17.0" accelerate
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""

# Text-to-speech (YarnGPT2) is switched on when the private "agrivoice-tts-assets"
# dataset (WavTokenizer checkpoint + config) is attached; N-ATLaS works without it.
TTS_SETUP = """
import glob, os, subprocess

def _find(name):
    hits = glob.glob(f"/kaggle/input/**/agrivoice-tts-assets/**/{name}", recursive=True) or \\
           glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None

ckpt = _find("wavtokenizer_large_speech_320_24k.ckpt")
config = _find("wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
if ckpt and config:
    # outetts without its llama-cpp dependency (only its audio decoder is used)
    subprocess.run("pip install -q --no-deps outetts==0.3.3 && pip install -q uroman inflect einops soundfile",
                   shell=True, check=True)
    if not os.path.isdir("/kaggle/working/yarngpt"):
        subprocess.run("git clone -q --depth 1 https://github.com/saheedniyi02/yarngpt.git /kaggle/working/yarngpt",
                       shell=True, check=True)
    os.environ.update(TTS_ENABLED="1", YARNGPT_DIR="/kaggle/working",
                      WAVTOKENIZER_CKPT=ckpt, WAVTOKENIZER_CONFIG=config)
    print("Text-to-speech: on")
else:
    print("Text-to-speech: off (attach the agrivoice-tts-assets dataset to enable it)")
"""

SECRETS = """
import os
from kaggle_secrets import UserSecretsClient

import glob

_secrets = UserSecretsClient()

def _secret(name):
    # 1) Kaggle Secrets: only reachable when the run is started from the website
    try:
        value = _secrets.get_secret(name)
        if value:
            return value
    except Exception:
        pass
    # 2) Private dataset "agrivoice-secrets" holding one file per secret,
    #    used for runs started through the Kaggle API
    for path in glob.glob(f"/kaggle/input/**/agrivoice-secrets/{name.lower()}", recursive=True):
        value = open(path).read().strip()
        if value:
            return value
    return None

os.environ["HF_TOKEN"] = _secret("HF_TOKEN") or ""
os.environ["DEVICE"] = "auto"  # spread N-ATLaS across both T4 GPUs
AGRIVOICE_URL = (RUN_CONFIG.get("backend_url") or _secret("AGRIVOICE_URL") or "").rstrip("/")
AGRIVOICE_ADMIN_TOKEN = None if RUN_CONFIG else _secret("AGRIVOICE_ADMIN_TOKEN")
assert os.environ["HF_TOKEN"], "Add the HF_TOKEN secret, or the private agrivoice-secrets dataset."
print("Started from:", "AgriVoice dashboard" if RUN_CONFIG else "Kaggle website",
      "| Backend:", AGRIVOICE_URL or "none")
"""

LAUNCH = """
import time
import requests

# How long to keep N-ATLaS online (runs started by hand). Kaggle gives 30 GPU
# hours a week and stops a session after ~9-12 hours anyway.
RUN_HOURS = float(RUN_CONFIG.get("run_hours", 2))
RUN_KEY = RUN_CONFIG.get("run_key")

import app  # downloads N-ATLaS (~16 GB) on first run and loads it onto the GPUs

_, local_url, share_url = app.demo.launch(share=True, prevent_thread_lock=True)
print("\\nN-ATLaS public link:", share_url)


def now():
    return time.strftime("%H:%M:%S")


def notify(event):
    \"\"\"Dashboard-started runs: tell the backend; returns its JSON reply or None.\"\"\"
    try:
        r = requests.post(
            f"{AGRIVOICE_URL}/api/v1/natlas/{event}",
            json={"run_key": RUN_KEY, "endpoint": share_url},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(now(), f"Could not reach AgriVoice ({event}):", exc)
        return None


def register_by_hand():
    if not (AGRIVOICE_URL and AGRIVOICE_ADMIN_TOKEN):
        print("Paste this link into your backend's /admin page.")
        return
    try:
        r = requests.put(
            f"{AGRIVOICE_URL}/api/v1/admin/llm-endpoint",
            json={"endpoint": share_url},
            headers={"Authorization": f"Bearer {AGRIVOICE_ADMIN_TOKEN}"},
            timeout=30,
        )
        r.raise_for_status()
        print(now(), "Registered with AgriVoice:", r.json()["endpoint"])
    except Exception as exc:
        print(now(), "Could not register with AgriVoice:", exc)


stop_at = time.time() + RUN_HOURS * 3600
reason = f"reached {RUN_HOURS:g} h"
try:
    if RUN_KEY:
        notify("register") and print(now(), "Registered with AgriVoice")
        # Check in every minute; the dashboard's Stop button answers "stop"
        while time.time() < stop_at:
            time.sleep(60)
            reply = notify("heartbeat")
            if reply and reply.get("action") == "stop":
                reason = "stopped from the dashboard"
                break
    else:
        register_by_hand()
        # Keep the session alive (and re-register in case the backend restarted)
        while time.time() < stop_at:
            time.sleep(min(600, max(1, stop_at - time.time())))
            register_by_hand()
finally:
    if RUN_KEY:
        notify("stopped")
    app.demo.close()
    print(now(), f"N-ATLaS is now offline ({reason}).")
"""


def main() -> None:
    app_source = (HERE / "app.py").read_text(encoding="utf-8")
    tts_source = (HERE / "tts.py").read_text(encoding="utf-8")
    notebook = {
        "cells": [
            md(SETUP),
            code(CONFIG),
            code(INSTALL),
            code(SECRETS),
            code(TTS_SETUP),
            code("%%writefile tts.py\n" + tts_source),
            code("%%writefile app.py\n" + app_source),
            code(LAUNCH),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    text = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    for path in (OUT, BACKEND_TEMPLATE):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
