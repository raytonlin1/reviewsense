# ReviewSense API + web UI in a container.
#   docker build -t reviewsense .
#   docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts" -v "$HOME/.cache/huggingface:/home/app/.cache/huggingface" \
#              -e REVIEWSENSE_SENTIMENT_MODEL=/app/artifacts/distilbert-20k reviewsense
# then open http://localhost:8000/ui (web page) or http://localhost:8000/docs (API).
# Models are not baked into the image: trained models (artifacts/) and downloaded ones (Hugging Face cache) are
# mounted, so the image stays small and a new model doesn't need a new image. Not built on the author's Mac
# (no Docker installed there): build and run it once before relying on it.
FROM python:3.14-slim

# Faster, quieter Python in containers: no .pyc files, logs straight to the console.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: this layer is rebuilt only when requirements.txt changes, not on every code change.
# The CPU-only PyTorch index avoids several GB of NVIDIA libraries the server doesn't use.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY reviewsense ./reviewsense
COPY experiments ./experiments
COPY data ./data

# Don't run as root: if the service is ever compromised, the attacker has a user with no privileges.
RUN useradd --create-home app && mkdir -p /app/artifacts && chown -R app /app
USER app

EXPOSE 8000
# The platform restarts the container when this fails (python is used because the slim image has no curl).
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
# One worker: each worker loads every model (several GB with the chatbot). Scale with more containers instead.
CMD ["uvicorn", "reviewsense.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
