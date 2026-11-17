"""Регрессии concrete ASR runtime: неверный backend не блокирует единственный слот."""

import asyncio
import base64
import importlib.util
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import httpx
import orjson
import pytest
from fastapi import FastAPI
from starlette.types import Message
from starlette.websockets import WebSocket


def load_asr_runtime() -> tuple[FastAPI, Callable[[WebSocket], Awaitable[None]]]:
    """Загружает контейнерный gateway без model imports и изменения PYTHONPATH.

    :return: Собственный экземпляр ASR-приложения и его реальный WS-handler.
    """
    source = Path(__file__).resolve().parents[1] / "deploy" / "asr" / "server.py"
    spec = importlib.util.spec_from_file_location("asr_runtime_under_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(FastAPI, module.app), cast(Callable[[WebSocket], Awaitable[None]], module.recognize)


async def run_asr_connection(handler: Callable[[WebSocket], Awaitable[None]], backend_called: asyncio.Event, *, final: bool) -> list[dict[str, object]]:
    """Подаёт секунду настоящего PCM и закрывает WS после ответа либо отказа backend.

    :param handler: Реальный обработчик внутреннего ASR-протокола.
    :param backend_called: Сигнал фактического POST к backend.
    :param final: Завершить запись явной командой flush либо проверить prefix-распознавание.
    :return: JSON-события, реально отправленные ASGI WebSocket.
    """
    incoming: asyncio.Queue[Message] = asyncio.Queue()
    outgoing: list[dict[str, object]] = []
    incoming.put_nowait({"type": "websocket.connect"})
    commands: list[dict[str, object]] = [{"type": "session.update", "model": "qwen3-asr"}, {"type": "input_audio_buffer.commit"}]
    commands.extend({"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\0" * 3200).decode()} for _ in range(10))
    if final:
        commands.append({"type": "input_audio_buffer.commit", "final": True})
    for command in commands:
        incoming.put_nowait({"type": "websocket.receive", "text": orjson.dumps(command).decode("utf-8")})

    async def send(message: Message) -> None:
        """Сохраняет события без подмены состояния самого WebSocket.

        :param message: Отправляемое сервером ASGI-событие.
        """
        if message["type"] == "websocket.send":
            outgoing.append(orjson.loads(message["text"]))
        if message["type"] == "websocket.close":
            incoming.put_nowait({"type": "websocket.disconnect", "code": message["code"]})

    async def disconnect() -> None:
        """Закрывает клиент после завершения backend coroutine, а не до её запуска."""
        await backend_called.wait()
        await asyncio.sleep(0.01)
        incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})

    later = asyncio.create_task(disconnect())
    try:
        await asyncio.wait_for(handler(WebSocket({"type": "websocket", "headers": []}, incoming.get, send)), timeout=1)
        return outgoing
    finally:
        later.cancel()
        await asyncio.gather(later, return_exceptions=True)


@pytest.mark.parametrize("body,final", [(b"not-json", False), (b"{}", True), (b'{"text":null}', True), (b'{"text":42}', True), (b"[]", True)])
def test_invalid_backend_response_releases_slot_and_allows_next_turn(body: bytes, final: bool) -> None:
    """Неверный HTTP200 превращается в безопасный отказ без навечно занятого ASR.

    :param body: Некорректный JSON, отсутствующий либо нестроковый transcript.
    :param final: Проверяемая стадия распознавания: prefix или final flush.
    """

    async def scenario() -> None:
        """Проверяет две последовательные реальные ASGI-сессии одного runtime."""
        application, handler = load_asr_runtime()
        called = asyncio.Event()
        current = body

        def respond(request: httpx.Request) -> httpx.Response:
            """Имитирует только HTTP backend, сохраняя реальный WS gateway.

            :param request: Сформированный gateway запрос распознавания.
            :return: Управляемый фактический HTTP200.
            """
            assert request.url.path == "/v1/audio/transcriptions"
            called.set()
            return httpx.Response(200, content=current)

        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://asr-test") as client:
                application.state.client = client
                events = await run_asr_connection(handler, called, final=final)
                assert application.state.active is False
                errors = [event for event in events if event["type"] == "error"]
                assert len(errors) == 1
                assert errors[0]["code"] == "asr_invalid_response"
                assert not [event for event in events if event["type"] == "transcription.done"]
                current = orjson.dumps({"text": "Проверка следующей реплики"})
                called.clear()
                following = await run_asr_connection(handler, called, final=True)
                assert following[0]["type"] == "session.created"
                assert following[-1] == {"type": "transcription.done", "text": "Проверка следующей реплики"}
                assert application.state.active is False

    asyncio.run(scenario())


def test_unexpected_decoder_exception_still_releases_slot() -> None:
    """Даже неожиданная ошибка decoder не пропускает reset в cleanup."""

    async def scenario() -> None:
        """Сохраняет диагностику неожиданной ошибки и проверяет finally слота."""
        application, handler = load_asr_runtime()
        called = asyncio.Event()

        def respond(request: httpx.Request) -> httpx.Response:
            """Вызывает внутренний отказ, не маскируемый как штатный HTTPError.

            :param request: Сформированный запрос распознавания.
            :return: Ответ не формируется: моделируется внутренний отказ транспорта.
            :raises RuntimeError: Диагностический отказ fake transport.
            """
            called.set()
            raise RuntimeError("decoder-test-failure")

        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://asr-test") as client:
                application.state.client = client
                with pytest.raises(RuntimeError, match="decoder-test-failure"):
                    await run_asr_connection(handler, called, final=True)
                assert application.state.active is False

    asyncio.run(scenario())
