# Инструмент установки и Python runtime закреплены digest для воспроизводимой сборки.
FROM ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uv
# Debian trixie совместим с Linux wheels orjson для Python 3.15 (manylinux_2_39).
FROM python:3.15.1-slim-trixie@sha256:533b0b2f9b5d46fd097091cf826f27902f1d6c3bca714b6dc74551356aa97a51

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
# Сначала устанавливаются только production-зависимости из lock; код не сбрасывает этот слой.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache && useradd --uid 10001 --create-home app
ENV PATH=/app/.venv/bin:$PATH PYTHONPATH=/app/sources
# В образ попадают приложение и клиент; локальные модели и секреты остаются на хосте.
COPY --chown=10001:10001 sources/ /app/sources/
COPY --chown=10001:10001 static/ /app/static/
USER 10001:10001
CMD ["python", "sources/main.py"]
