"""Проверяет услышанную историю и отсутствие наложения отменённых модельных задач."""

import asyncio
from typing import Any
from uuid import uuid4

import orjson
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from api.fastapi_app import app
from services.session_service import GenerationContext

START = {"version": 1, "type": "session.start", "language": "ru", "input_sample_rate": 16000, "output_sample_rate": 24000, "channels": 1, "sample_width_bytes": 2}
ORIGIN = {"origin": "http://127.0.0.1:8000"}


def begin_turn(socket: WebSocketTestSession) -> dict[str, Any]:
    """Записывает короткую PCM-реплику по настоящему браузерному протоколу.

    :param socket: Подключённый и согласованный WebSocket.
    :return: Идентификаторы новой реплики.
    """
    socket.send_text(orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8"))
    started = orjson.loads(socket.receive_text())
    if started["type"] == "response.cancelled":
        started = orjson.loads(socket.receive_text())
    assert started["type"] == "input.started"
    socket.send_bytes(b"\0" * 640)
    socket.send_text(orjson.dumps({"version": 1, "type": "input.commit", "generation_id": started["generation_id"], "turn_id": started["turn_id"]}).decode("utf-8"))
    assert orjson.loads(socket.receive_text())["type"] == "response.started"
    return dict(started)


def finish_turn(socket: WebSocketTestSession) -> list[dict[str, Any]]:
    """Читает конечный тестовый ответ, сохраняя реальные audio headers.

    :param socket: Настоящий WebSocket приложения.
    :return: Заголовки аудиопар для последующего playback ACK.
    """
    audio: list[dict[str, Any]] = []
    for _ in range(30):
        event = orjson.loads(socket.receive_text())
        assert event["type"] != "error", event
        if event["type"] == "response.done":
            return audio
        if event["type"] == "response.audio":
            assert len(socket.receive_bytes()) == event["byte_length"]
            audio.append(event)
    pytest.fail("Тестовый ответ не завершился")


@pytest.mark.parametrize("explicit_cancel", [False, True])
def test_next_turn_contains_only_fully_heard_phrases(explicit_cancel: bool) -> None:
    """Следующая реплика получает вопрос и только полностью услышанный префикс ответа.

    :param explicit_cancel: Архивировать через stop либо непосредственно следующую реплику.
    """
    snapshots: list[list[tuple[str, str]]] = []

    async def generate(context: GenerationContext, pcm: bytes) -> None:
        """Публикует две фразы через реального владельца сессии.

        :param context: Контекст новой генерации со снимком истории.
        :param pcm: Подтверждённая PCM-реплика.
        """
        snapshots.append([(message.role, message.content) for message in context.history])
        await context.emit_transcript("Первый вопрос" if len(snapshots) == 1 else "Второй вопрос", final=True)
        for text in ["Услышанная фраза.", "Оборванная фраза."]:
            segment = uuid4()
            await context.emit_text(segment, text)
            await context.emit_audio(segment, b"\x01\0" * 480)
            await context.finish_segment(segment)

    with TestClient(app) as client:
        app.state.session_service.generation_handler = generate
        with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
            socket.send_text(orjson.dumps(START).decode("utf-8"))
            orjson.loads(socket.receive_text())
            started = begin_turn(socket)
            audio = finish_turn(socket)
            for event, played in zip(audio, [480, 240], strict=True):
                progress = {"version": 1, "type": "playback.progress", "generation_id": event["generation_id"], "segment_id": event["segment_id"], "played_samples": played}
                socket.send_text(orjson.dumps(progress).decode("utf-8"))
                socket.send_text(orjson.dumps(progress).decode("utf-8"))
            if explicit_cancel:
                socket.send_text(orjson.dumps({"version": 1, "type": "response.cancel", "generation_id": started["generation_id"]}).decode("utf-8"))
                assert orjson.loads(socket.receive_text())["type"] == "response.cancelled"
            begin_turn(socket)
            finish_turn(socket)
            assert snapshots == [[], [("user", "Первый вопрос"), ("assistant", "Услышанная фраза.")]]
        with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
            socket.send_text(orjson.dumps(START).decode("utf-8"))
            orjson.loads(socket.receive_text())
            begin_turn(socket)
            finish_turn(socket)
            assert snapshots[-1] == []


def test_next_handler_waits_for_previous_runtime_cleanup() -> None:
    """Быстрый stop/new turn не запускает модель, пока старая задача закрывает upstream."""
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    release_cleanup = asyncio.Event()
    snapshots: list[bool] = []

    async def generate(context: GenerationContext, pcm: bytes) -> None:
        """Первый вызов имитирует runtime с кооперативной, но не мгновенной отменой.

        :param context: Контекст конкретной генерации.
        :param pcm: Входная реплика.
        """
        if not entered.is_set():
            entered.set()
            await context.emit_transcript("Первый вопрос", final=True)
            try:
                await asyncio.Future()
            finally:
                await release_cleanup.wait()
                cleaned.set()
        else:
            snapshots.append(cleaned.is_set())
            await context.emit_transcript("Новый вопрос", final=True)

    with TestClient(app) as client:
        app.state.session_service.generation_handler = generate
        with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
            socket.send_text(orjson.dumps(START).decode("utf-8"))
            orjson.loads(socket.receive_text())
            started = begin_turn(socket)
            assert orjson.loads(socket.receive_text())["type"] == "transcript.final"
            socket.send_text(orjson.dumps({"version": 1, "type": "response.cancel", "generation_id": started["generation_id"]}).decode("utf-8"))
            assert orjson.loads(socket.receive_text())["type"] == "response.cancelled"
            begin_turn(socket)
            try:
                assert client.portal is not None
                client.portal.call(asyncio.sleep, 0)
                assert snapshots == [], "Новый handler запущен до освобождения старого runtime"
            finally:
                assert client.portal is not None
                client.portal.call(release_cleanup.set)
            finish_turn(socket)
            assert snapshots == [True]


def test_model_cleanup_quarantine_survives_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reconnect после cleanup timeout не допускает наложения запросов GPU.

    :param monkeypatch: Короткий deadline кооперативной остановки тестовой модели.
    """
    release_cleanup = asyncio.Event()
    entered = asyncio.Event()
    started_calls: list[int] = []

    async def generate(context: GenerationContext, pcm: bytes) -> None:
        """Удерживает старый upstream после отключения первого браузера.

        :param context: Контекст текущего WS-подключения.
        :param pcm: Короткая PCM-реплика.
        """
        started_calls.append(1)
        if not entered.is_set():
            entered.set()
            await context.emit_transcript("Старый вопрос", final=True)
            try:
                await asyncio.Future()
            finally:
                await release_cleanup.wait()

    with TestClient(app) as client:
        service = app.state.session_service
        monkeypatch.setattr(service.settings.api, "close_timeout_seconds", 0.05)
        service.generation_handler = generate
        try:
            with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
                socket.send_text(orjson.dumps(START).decode("utf-8"))
                orjson.loads(socket.receive_text())
                begin_turn(socket)
                assert orjson.loads(socket.receive_text())["type"] == "transcript.final"
            assert client.portal is not None
            client.portal.call(asyncio.sleep, 0.1)
            with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
                socket.send_text(orjson.dumps(START).decode("utf-8"))
                assert orjson.loads(socket.receive_text())["type"] == "session.ready"
                begin_turn(socket)
                event = orjson.loads(socket.receive_text())
                assert event.get("code") == "model_busy", event
                assert started_calls == [1]
        finally:
            assert client.portal is not None
            client.portal.call(release_cleanup.set)
