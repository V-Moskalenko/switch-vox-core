# uv 0.12.5; digest является единственным идентификатором образа.
FROM ghcr.io/astral-sh/uv@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uv
# CUDA 12.1.1 / cuDNN 8 / Ubuntu 22.04.
FROM nvidia/cuda@sha256:f4d8e1264366940438f0353da6f289c7bef069d993d111f8106086ccd18c4a30

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 LANG=C.UTF-8
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-dev git build-essential \
        ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
RUN uv venv --python /usr/bin/python3 /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.lock /tmp/requirements.lock
COPY build-constraints.txt /tmp/build-constraints.txt
RUN uv pip sync --python /opt/venv/bin/python /tmp/requirements.lock \
    --build-constraint /tmp/build-constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match \
    && uv cache clean
COPY runtime-extras.lock /tmp/runtime-extras.lock
RUN uv pip install --python /opt/venv/bin/python -r /tmp/runtime-extras.lock -c /tmp/requirements.lock && uv cache clean

ARG COSYVOICE_REVISION=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
RUN git init /opt/CosyVoice \
    && cd /opt/CosyVoice \
    && git remote add origin https://github.com/QwenAudio/CosyVoice.git \
    && git fetch --depth 1 origin ${COSYVOICE_REVISION} \
    && git checkout FETCH_HEAD \
    && git submodule update --init --depth 1
COPY patch_upstream.py /tmp/patch_upstream.py
RUN python /tmp/patch_upstream.py
ENV PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS:/app
WORKDIR /app
COPY backend.py streaming.py schemas.py server.py /app/
RUN python -c "import backend, server; \
import cosyvoice.flow.flow_matching, cosyvoice.flow.DiT.dit, cosyvoice.hifigan.generator, matcha.utils.audio; \
print('CosyVoice graph imports OK')" \
    && groupadd --gid 10001 vox \
    && useradd --uid 10001 --gid vox --create-home --home-dir /home/vox vox \
    && mkdir -p /home/vox/.cache \
    && chown -R vox:vox /home/vox
ENV HOME=/home/vox XDG_CACHE_HOME=/home/vox/.cache
USER vox:vox
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
