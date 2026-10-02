# Deployment

Production runs in two parts (see [architecture](architecture.md)):

1. **Backend** on an always-on CPU server, with Docker Compose and Caddy (automatic HTTPS).
2. **N-ATLaS** on a GPU: a Kaggle notebook (free, 30 GPU h/week) started from the admin dashboard, or a Hugging Face Space.

## Prerequisites

- A Hugging Face account that has accepted the licences of the NCAIR1 models (N-ATLaS and the four ASR models), and a read token.
- A Linux server with ≥ 12 GB RAM and ~25 GB of disk, ports 80 and 443 open. We use an Oracle Cloud *Always Free* Ampere instance (2 OCPU, 12 GB, Ubuntu 24.04).
- A domain name pointing at the server, or a wildcard DNS service such as `<ip-with-dashes>.sslip.io`.
- For N-ATLaS on Kaggle: a phone-verified Kaggle account.

## 1. Backend server

```bash
# On the server
git clone <this repository> ~/agrivoice
sudo bash ~/agrivoice/backend/deploy/oracle/setup.sh   # Docker, 4 GB swap, firewall ports 80/443
```

On Oracle Cloud, also allow TCP 80 and 443 in the subnet's security list.

Create `backend/deploy/oracle/.env` (keep it private, `chmod 600`):

```bash
SITE_ADDRESS=api.example.com           # Caddy gets a certificate for it
HF_TOKEN=hf_...
ADMIN_TOKEN=...                        # openssl rand -hex 24
PUBLIC_BASE_URL=https://api.example.com
# For starting N-ATLaS from the dashboard (step 3):
KAGGLE_API_TOKEN=...
KAGGLE_NOTEBOOK_ID=your-kaggle-username/agrivoice-natlas
KAGGLE_SECRETS_DATASET=your-kaggle-username/agrivoice-secrets
KAGGLE_TTS_DATASET=your-kaggle-username/agrivoice-tts-assets   # spoken answers
```

Then start it:

```bash
cd ~/agrivoice/backend/deploy/oracle
sudo docker compose up -d --build
```

The first start downloads the speech and detection models (~7.5 GB) and takes a few minutes; later starts take ~1 minute. Check `https://<SITE_ADDRESS>/api/v1/health/ready`.

The compose file runs the backend with SQLite on a Docker volume, `LLM_PROVIDER=remote`, and restarts it automatically. Other settings (allowed frontend origins, answer length, detection threshold…) are changed live from `/admin` → Settings.

**Updating:** `git pull`, then `sudo docker compose up -d --build api`.

## 2. N-ATLaS on Kaggle

The notebook is [`natlas-space/kaggle/agrivoice_natlas.ipynb`](../natlas-space/kaggle/agrivoice_natlas.ipynb), generated from `natlas-space/app.py` by `build_kaggle_notebook.py`.

1. On Kaggle: **Create → New Notebook → File → Import Notebook**, upload it, and name it `agrivoice-natlas`.
2. Settings: **Accelerator: GPU T4 ×2**, **Internet: on**.
3. Create a **private** dataset named `agrivoice-secrets` containing one file, `hf_token`, holding your Hugging Face token. Runs started through the Kaggle API can't read Kaggle Secrets, so they read the token from here.
4. Create a Kaggle API token (Settings → API) and put it in the server's `KAGGLE_API_TOKEN`.
5. *(Spoken answers)* Create a **private** dataset named `agrivoice-tts-assets` with YarnGPT's decoder files: `wavtokenizer_large_speech_320_24k.ckpt` (YarnGPT's [Google Drive link](https://drive.google.com/file/d/1-ASeEkrn4HY49yZWHTASgfGFNXdVnLTt)) and `wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml` ([Hugging Face](https://huggingface.co/novateur/WavTokenizer-medium-speech-75token)). Set `KAGGLE_TTS_DATASET` on the server. When the dataset is attached, the notebook switches speech on; without it, N-ATLaS runs as before.

## 3. Operating N-ATLaS from the dashboard

Open `https://<SITE_ADDRESS>/admin` and sign in with `ADMIN_TOKEN`. On the **N-ATLaS** tab:

- **Start** (choose hours): N-ATLaS is answering within ~4–6 minutes.
- **Stop**: the notebook shuts down within ~1 minute and stops using GPU hours.
- **Schedule**: days, hours, optional date range and time zone; starts and stops automatically, and won't restart a run stopped by hand within the same window.
- **GPU hours** used in the last 7 days, against a configurable weekly limit.

The setup checklist on that tab shows anything missing from step 1–2.

**Without the dashboard:** run the notebook on kaggle.com (*Save Version → Save & Run All*) with Kaggle Secrets `HF_TOKEN`, `AGRIVOICE_URL` and `AGRIVOICE_ADMIN_TOKEN` attached; it registers itself the same way.

## Alternatives for N-ATLaS

- **Hugging Face Space.** `natlas-space/` is a ready Gradio Space (ZeroGPU-compatible). Set the backend's endpoint to the Space id (`owner/name`) in `/admin`. Hugging Face currently requires a PRO subscription to host Gradio Spaces.
- **Same machine.** With a GPU of ≥ 20 GB, set `LLM_PROVIDER=local` and N-ATLaS loads inside the backend. The provided Dockerfile is CPU-only; use a CUDA build of PyTorch.

## Local development

See [backend/README.md](../backend/README.md). With `DEV_USE_MOCK_MODELS=true` everything runs without models, GPU or tokens.
