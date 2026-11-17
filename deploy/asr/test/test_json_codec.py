"""Проверяет строгий JSON и текстовые WebSocket-кадры внутреннего ASR API."""

import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from starlette.types import Message
from starlette.websockets import WebSocket


def load_runtime() -> ModuleType:
    """Загружает шлюз ASR без запуска модели и HTTP-сервера.

    :return: Модуль с обработчиком команд и декодером ответов backend.
    """
    source = Path(__file__).resolve().parents[1] / "server.py"
    spec = importlib.util.spec_from_file_location("asr_json_codec_under_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_backend_rejects_non_json_number(constant: str) -> None:
    """Нестандартное число делает весь ответ ASR некорректным.

    :param constant: Константа, отсутствующая в JSON-стандарте.
    """
    runtime = load_runtime()
    body = ('{"text":"Привет","score":' + constant + "}").encode()
    with pytest.raises(runtime.ASRResponseError):
        runtime.transcription_text(httpx.Response(200, content=body))


def test_websocket_rejects_non_json_number_and_keeps_text_frames() -> None:
    """Шлюз сохраняет текстовый транспорт и закрывает команду с NaN кодом 1008."""
    runtime = load_runtime()

    async def scenario() -> None:
        """Проверяет кодек на настоящем ASGI WebSocket без сетевого сервера."""
        incoming: asyncio.Queue[Message] = asyncio.Queue()
        incoming.put_nowait({"type": "websocket.connect"})
        incoming.put_nowait({"type": "websocket.receive", "text": '{"type":"session.update","score":NaN}'})
        outgoing: list[Message] = []

        async def send(message: Message) -> None:
            """Сохраняет кадры, отправляемые настоящим WebSocket.

            :param message: Текстовый кадр или событие закрытия ASGI.
            """
            outgoing.append(message)

        socket = WebSocket({"type": "websocket", "headers": []}, incoming.get, send)
        await socket.accept()
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500)), trust_env=False) as client:
            connection = runtime.ASRConnection(socket, client)
            await connection.send({"type": "transcription.done", "text": "Привет"})
            frame = outgoing[-1]
            assert frame["type"] == "websocket.send"
            assert isinstance(frame["text"], str)
            assert "Привет" in frame["text"]
            assert await connection.receive_command() is None
        assert outgoing[-1] == {"type": "websocket.close", "code": 1008, "reason": ""}

    asyncio.run(scenario())
