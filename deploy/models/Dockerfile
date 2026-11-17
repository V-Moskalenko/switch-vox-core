# Один небольшой CPU-образ загружает модели, не включая их в Docker layers.
FROM ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uv
# Общая с приложением база поддерживает wheel orjson без сборки Rust-зависимостей.
FROM python:3.15.1-slim-trixie@sha256:533b0b2f9b5d46fd097091cf826f27902f1d6c3bca714b6dc74551356aa97a51

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/models/.cache/huggingface HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30
COPY --from=uv /uv /usr/local/bin/uv
COPY requirements.lock /tmp/requirements.lock
RUN uv pip sync --system --require-hashes --no-cache /tmp/requirements.lock
WORKDIR /app
COPY download_models.py /app/download_models.py
# Root создаёт файлы моделей; Compose ограничивает capabilities и запись каталогами /models и /tmp.
CMD ["python", "/app/download_models.py"]
