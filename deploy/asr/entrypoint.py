"""Один ASR контейнер: прямой vLLM backend и конкретный streaming gateway."""

import subprocess
import time

import httpx
import uvicorn


def main() -> None:
    """Запускает один экземпляр весов и освобождает его при остановке runtime."""
    # Модель живёт в одном дочернем процессе; шлюз обращается к ней через loopback.
    backend = subprocess.Popen(
        [
            "vllm",
            "serve",
            "/models/asr",
            "--served-model-name",
            "qwen3-asr",
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
            "--dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            "0.20",
            "--max-model-len",
            "2048",
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "2048",
            "--enforce-eager",
        ]
    )
    try:
        # Не открываем WebSocket-шлюз до подтверждения готовности весов.
        deadline = time.monotonic() + 180
        with httpx.Client(trust_env=False) as client:
            while time.monotonic() < deadline:
                if backend.poll() is not None:
                    raise RuntimeError("vLLM ASR остановился до готовности")
                try:
                    if client.get("http://127.0.0.1:8001/health", timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError("Истёк срок загрузки ASR")
        # Намеренная граница контейнера: gateway доступен API только в выделенной internal Compose-сети без ports; vLLM выше остаётся на loopback.
        uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1, ws_max_size=8192, access_log=False)
    finally:
        # Остановка контейнера не должна оставлять дочерний процесс с занятой VRAM.
        backend.terminate()
        try:
            backend.wait(timeout=15)
        except subprocess.TimeoutExpired:
            backend.kill()
            backend.wait(timeout=5)


if __name__ == "__main__":
    main()
