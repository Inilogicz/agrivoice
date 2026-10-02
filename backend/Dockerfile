# AgriVoice backend (CPU). Runs the NCAIR speech models and language detection;
# N-ATLaS runs remotely on a GPU (see natlas-space/) and is managed from /admin.
# Used by deploy/oracle/docker-compose.yml.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "uv>=0.8,<1"

WORKDIR /app

# CPU-only PyTorch (the default Linux wheels bundle several GB of CUDA libraries)
RUN uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/cpu \
    torch torchaudio

# Everything else from the lockfile, minus torch and its CUDA-only dependencies
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-hashes --no-emit-project \
      | grep -viE '^(torch|torchaudio|triton|nvidia-|cuda-)' > /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt

COPY app ./app

# Hugging Face runs Spaces as uid 1000
# (/home/user/.cache is pre-created so a mounted volume inherits user ownership)
RUN useradd -m -u 1000 user && mkdir -p /data /tmp/agrivoice_audio /home/user/.cache \
    && chown -R user:user /app /data /tmp/agrivoice_audio /home/user/.cache
USER user

ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=development \
    DATABASE_URL=sqlite+aiosqlite:////data/agrivoice.db \
    SPEECH_CACHE_DIR=/data/speech \
    TTS_ENABLED=true \
    MODEL_CACHE_DIR=/home/user/.cache/agrivoice-models \
    DEVICE=cpu \
    DEV_USE_MOCK_MODELS=false \
    LLM_PROVIDER=remote \
    LLM_MAX_NEW_TOKENS=350 \
    CORS_ORIGINS=http://localhost:3000,http://localhost:5173

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
