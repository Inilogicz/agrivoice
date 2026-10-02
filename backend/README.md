# AgriVoice backend

FastAPI service behind AgriVoice: language detection, NCAIR speech-to-text,
N-ATLaS answers and translations, conversation storage, and the admin
dashboard. See the [project README](../README.md) for the overview and
[docs/](../docs) for the architecture, API and deployment guides.

## Run locally

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/) and `ffmpeg`.

```bash
cp .env.example .env          # then edit it
uv sync
uv run uvicorn app.main:app --reload
```

The API is at <http://localhost:8000>, interactive docs at <http://localhost:8000/docs>.

### Without models (fastest)

```bash
DEV_USE_MOCK_MODELS=true DATABASE_URL=sqlite+aiosqlite:///./dev.db \
  uv run uvicorn app.main:app --reload
```

Every component is replaced by a labelled mock (`[MOCK] …`), so no downloads,
GPU or Hugging Face token are needed.

### With the real models

1. On Hugging Face, accept the licences of the gated NCAIR1 models
   ([N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS),
   [Yoruba-ASR](https://huggingface.co/NCAIR1/Yoruba-ASR),
   [Hausa-ASR](https://huggingface.co/NCAIR1/Hausa-ASR),
   [Igbo-ASR](https://huggingface.co/NCAIR1/Igbo-ASR),
   [NigerianAccentedEnglish](https://huggingface.co/NCAIR1/NigerianAccentedEnglish))
   and put a token in `HF_TOKEN`.
2. Choose where N-ATLaS runs with `LLM_PROVIDER`:
   - `remote` – N-ATLaS on a GPU elsewhere (see [natlas-space](../natlas-space)); recommended
   - `local` – in this process; needs a GPU with ~20 GB memory
   - `mock` – real speech models, placeholder answers

Models download on first start (≈7.5 GB without N-ATLaS).

### Database

SQLite (`sqlite+aiosqlite:///…`) creates its tables on startup. For PostgreSQL,
run `uv run alembic upgrade head` once before starting.

## Tests

```bash
DEV_USE_MOCK_MODELS=true uv run pytest
```

Tests use mock models and an in-memory SQLite database.

## Layout

```
app/
  api/routes/     HTTP endpoints (voice, chat, translate, conversations,
                  feedback, languages, health, admin, natlas callbacks)
  services/       pipeline orchestration, prompts, runtime settings,
                  N-ATLaS endpoint registry and Kaggle runner
  models/         model adapters: NCAIR ASR, N-ATLaS (local/remote),
                  MMS language detector
  repositories/   database access
  db/             SQLAlchemy models and session
  schemas/        request/response models
  resources/      Kaggle notebook template for N-ATLaS
alembic/          PostgreSQL migrations
deploy/oracle/    production deployment (Docker Compose + Caddy)
tests/            unit and integration tests
```
