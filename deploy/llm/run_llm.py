"""Запускает закреплённую Qwen3-4B AWQ с ограниченным контекстом и VRAM."""

import os

if __name__ == "__main__":
    args = [
        "vllm",
        "serve",
        "/models/llm",
        "--served-model-name",
        "qwen3-4b",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "float16",
        "--gpu-memory-utilization",
        "0.30",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "4096",
        "--enforce-eager",
    ]
    # vLLM заменяет launcher, получает сигналы контейнера напрямую и ведёт собственный журнал.
    os.execvp(args[0], args)
