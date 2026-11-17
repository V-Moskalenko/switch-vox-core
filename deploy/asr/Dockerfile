# Базовый vLLM runtime закреплён digest образа vllm/vllm-openai v0.32.0.
FROM vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
COPY requirements.txt /tmp/asr-requirements.txt
RUN python3 -m pip install --no-cache-dir --no-deps -r /tmp/asr-requirements.txt
ENV HF_HUB_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/vox \
    XDG_CACHE_HOME=/home/vox/.cache \
    HF_HOME=/home/vox/.cache/huggingface
RUN groupadd --gid 10001 vox \
    && useradd --uid 10001 --gid 10001 --create-home vox \
    && install -d -o 10001 -g 10001 /home/vox/.cache
WORKDIR /app
COPY --chown=0:0 server.py entrypoint.py /app/
USER 10001:10001
ENTRYPOINT ["python3", "/app/entrypoint.py"]
