"""Проверяет фоновые записи ASR и отсутствие содержимого ответа в диагностике."""

import asyncio
import importlib.util
import io
import threading
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from loguru import logger
from starlette.types import Message
from starlette.websockets import WebSocket


class ThreadRecordingStream(io.StringIO):
    """Сохраняет идентификаторы потоков, действительно выполнявших запись журнала."""

    def __init__(self) -> None:
        """Создаёт текстовый журнал и список потоков-писателей."""
        super().__init__()
        self.writers: list[int] = []

    def write(self, value: str) -> int:
        """Запоминает поток до записи сообщения в память.

        :param value: Готовая строка журнала.
        :return: Число записанных символов.
        """
        self.writers.append(threading.get_ident())
        return super().write(value)


def load_runtime() -> ModuleType:
    """Загружает шлюз без запуска vLLM и открытия модельных файлов.

    :return: Отдельный экземпляр модуля ASR.
    """
    path = Path(__file__).resolve().parents[1] / "server.py"
    spec = importlib.util.spec_from_file_location("asr_logging_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lifespan_flushes_background_logs_to_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Записи startup/shutdown доходят в stderr из рабочего потока до завершения lifespan.

    :param monkeypatch: Изолированная подмена потока stderr.
    """
    runtime = load_runtime()
    stream = ThreadRecordingStream()
    monkeypatch.setattr("sys.stderr", stream)
    caller = threading.get_ident()

    async def scenario() -> None:
        """Выполняет настоящий lifecycle без сетевых запросов к модели."""
        async with runtime.lifespan(runtime.app):
            await asyncio.sleep(0)

    asyncio.run(scenario())
    assert stream.getvalue().count("ASR") >= 2
    assert stream.writers and all(writer != caller for writer in stream.writers)


def test_decode_logs_failure_type_without_backend_contents() -> None:
    """HTTP-ошибка отражается в журнале без URL и тела ответа backend."""
    runtime = load_runtime()
    messages: list[str] = []
    sink = logger.add(messages.append, format="{level} {message}", level="DEBUG")

    async def scenario() -> None:
        """Вызывает реальное декодирование через локальный ошибочный HTTP transport."""
        incoming: asyncio.Queue[Message] = asyncio.Queue()
        incoming.put_nowait({"type": "websocket.connect"})

        async def send(message: Message) -> None:
            """Принимает отправку события ASGI без сетевого соединения.

            :param message: Событие отправки JSON либо закрытия WebSocket.
            """

        socket = WebSocket({"type": "websocket", "headers": []}, incoming.get, send)
        await socket.accept()
        transport = httpx.MockTransport(lambda request: httpx.Response(500, content=b"PRIVATE_RESPONSE"))
        async with httpx.AsyncClient(transport=transport, base_url="http://private-host-secret", trust_env=False) as client:
            connection = runtime.ASRConnection(socket, client)
            connection.buffer.extend(b"\0" * 32000)
            connection.final = True
            connection.changed.set()
            await connection.decode()

    try:
        asyncio.run(scenario())
        output = "".join(messages)
        assert "HTTPStatusError" in output
        assert "PRIVATE_RESPONSE" not in output and "private-host-secret" not in output
    finally:
        logger.remove(sink)
