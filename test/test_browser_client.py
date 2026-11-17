"""Проверяет раздачу клиента и настоящий HTTP Upgrade транспортного сервера."""

import base64
import hashlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from api.fastapi_app import app
from settings import PROJECT_ROOT


def test_browser_page_and_worklet_assets() -> None:
    """Проверяет страницу и независимые модули, загружаемые AudioWorklet."""
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert 'lang="ru"' in page.text
        for asset in ["app.js", "audio-worklet.js", "audio-core.mjs", "style.css"]:
            response = client.get(f"/static/{asset}")
            assert response.status_code == 200
        assert "javascript" in client.get("/static/audio-core.mjs").headers["content-type"]


def test_real_uvicorn_supports_browser_websocket_upgrade(tmp_path: Path) -> None:
    """Проверяет WS за пределами TestClient, который не замечает отсутствие драйвера Uvicorn.

    :param tmp_path: Каталог диагностического вывода собственного дочернего сервера.
    """
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = dict(os.environ, API__PORT=str(port), API__ALLOWED_ORIGINS=f'["http://127.0.0.1:{port}"]')
    log_path = tmp_path / "uvicorn.log"
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "--app-dir", str(PROJECT_ROOT / "sources"), "api.fastapi_app:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            connection = None
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError(f"Uvicorn завершился до проверки: {log_path.read_text(encoding='utf-8')}")
                try:
                    connection = socket.create_connection(("127.0.0.1", port), timeout=1)
                    break
                except OSError:
                    time.sleep(0.05)
            assert connection is not None, "Uvicorn не запустился за 10 секунд"
            with connection:
                nonce = base64.b64encode(b"browser-test-key").decode()
                request = (
                    f"GET /api/v1/session HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nOrigin: http://127.0.0.1:{port}\r\n"
                    f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {nonce}\r\nSec-WebSocket-Version: 13\r\n\r\n"
                )
                connection.sendall(request.encode("ascii"))
                response = b""
                while b"\r\n\r\n" not in response and len(response) < 8192:
                    chunk = connection.recv(4096)
                    if not chunk:
                        break
                    response += chunk
                assert response.startswith(b"HTTP/1.1 101"), response.decode(errors="replace")
                expected = base64.b64encode(hashlib.sha1((nonce + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode(), usedforsecurity=False).digest())
                headers = dict(line.split(b":", maxsplit=1) for line in response.split(b"\r\n")[1:] if b":" in line)
                assert next(value.strip() for name, value in headers.items() if name.lower() == b"sec-websocket-accept") == expected
                connection.sendall(b"\x88\x80\x00\x00\x00\x00")
        finally:
            process.terminate()
            process.wait(timeout=10)
