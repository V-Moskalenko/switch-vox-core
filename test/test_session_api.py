"""Проверяет настоящий WS-протокол, отмену и единственный слот с fake streams."""

import asyncio
from collections.abc import Iterator
from uuid import UUID, uuid4

import orjson
import pytest
from fastapi.testclient import TestClient
from starlette.types import Message
from starlette.websockets import WebSocket, WebSocketDisconnect

from api.fastapi_app import app
from services.session_service import GenerationContext, SessionService
from settings import Settings

START = {"version": 1, "type": "session.start", "language": "ru", "input_sample_rate": 16000, "output_sample_rate": 24000, "channels": 1, "sample_width_bytes": 2}
ORIGIN = {"origin": "http://127.0.0.1:8000"}


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Запускает отдельный lifespan для каждого сценария.

    :yield: Клиент реального приложения без загруженных моделей.
    """
    with TestClient(app) as instance:
        app.state.session_service.generation_handler = None
        yield instance


def test_handshake_input_validation_cancel_and_reconnect(client: TestClient) -> None:
    """Проверяет переходы PTT, неизвестные команды и освобождение слота.

    :param client: Запущенный экземпляр приложения.
    """
    for _ in range(2):
        with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
            socket.send_text(orjson.dumps(START).decode("utf-8"))
            ready = orjson.loads(socket.receive_text())
            assert ready["type"] == "session.ready"
            UUID(ready["session_id"])
            socket.send_text(orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8"))
            started = orjson.loads(socket.receive_text())
            assert started["type"] == "input.started"
            socket.send_bytes(b"\0")
            assert orjson.loads(socket.receive_text())["code"] == "invalid_audio_frame"
            socket.send_text(orjson.dumps({"version": 1, "type": "mode.change", "mode": "e2e"}).decode("utf-8"))
            assert orjson.loads(socket.receive_text())["code"] == "invalid_command"
            socket.send_text(orjson.dumps({"version": 1, "type": "response.cancel", "generation_id": started["generation_id"]}).decode("utf-8"))
            assert orjson.loads(socket.receive_text())["type"] == "response.cancelled"
            socket.send_text(orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8"))
            assert orjson.loads(socket.receive_text())["type"] == "input.started"


def test_origin_and_second_session_rejected(client: TestClient) -> None:
    """Не допускает посторонний сайт и второй WS до вызова моделей.

    :param client: Запущенный экземпляр приложения.
    """
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/v1/session", headers={"origin": "https://example.org"}):
            pytest.fail("Посторонний Origin принят")
    with client.websocket_connect("/api/v1/session", headers=ORIGIN) as first:
        first.send_text(orjson.dumps(START).decode("utf-8"))
        assert orjson.loads(first.receive_text())["type"] == "session.ready"
        with client.websocket_connect("/api/v1/session", headers=ORIGIN) as second:
            assert orjson.loads(second.receive_text())["code"] == "session_busy"
            with pytest.raises(WebSocketDisconnect):
                orjson.loads(second.receive_text())


def test_audio_pairing_and_playback_progress(client: TestClient) -> None:
    """Проверяет порядок и атомарность JSON/binary на настоящем WebSocket.

    :param client: Запущенный экземпляр приложения.
    """
    segment_id = uuid4()
    audio = b"\x01\0" * 480

    async def generate(context: GenerationContext, pcm: bytes) -> None:
        """Выдаёт две части одной тестовой фразы.

        :param context: Канал событий конкретной генерации.
        :param pcm: Подтверждённая входная реплика.
        """
        assert len(pcm) == 640
        await context.emit_text(segment_id, "Проверка")
        await context.emit_audio(segment_id, audio)
        await context.emit_audio(segment_id, audio)
        await context.finish_segment(segment_id)

    app.state.session_service.generation_handler = generate
    with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
        socket.send_text(orjson.dumps(START).decode("utf-8"))
        orjson.loads(socket.receive_text())
        socket.send_text(orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8"))
        started = orjson.loads(socket.receive_text())
        socket.send_bytes(b"\0" * 640)
        socket.send_text(orjson.dumps({"version": 1, "type": "input.commit", "turn_id": started["turn_id"], "generation_id": started["generation_id"]}).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["type"] == "response.started"
        assert orjson.loads(socket.receive_text())["text"] == "Проверка"
        for sequence in range(2):
            header = orjson.loads(socket.receive_text())
            assert header["type"] == "response.audio"
            assert header["sequence"] == sequence
            assert header["byte_length"] == len(audio)
            assert header["generation_id"] == started["generation_id"]
            assert header["segment_id"] == str(segment_id)
            assert socket.receive_bytes() == audio
        ended = orjson.loads(socket.receive_text())
        assert ended["type"] == "response.segment.end"
        assert ended["total_samples"] == 960
        assert orjson.loads(socket.receive_text())["type"] == "response.done"
        socket.send_text(orjson.dumps({"version": 1, "type": "playback.progress", "generation_id": started["generation_id"], "segment_id": str(segment_id), "played_samples": 961}).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["code"] == "invalid_playback_progress"
        progress = {"version": 1, "type": "playback.progress", "generation_id": started["generation_id"], "segment_id": str(segment_id), "played_samples": 480}
        socket.send_text(orjson.dumps(progress).decode("utf-8"))
        socket.send_text(orjson.dumps(progress | {"played_samples": 479}).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["code"] == "invalid_playback_progress"
        socket.send_text(orjson.dumps(progress).decode("utf-8"))
        socket.send_text(orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["type"] == "response.cancelled"
        assert orjson.loads(socket.receive_text())["type"] == "input.started"


def test_prefill_cancel_and_late_generation_discard(client: TestClient) -> None:
    """Отмена не ждёт первого токена, старая генерация не портит новую.

    :param client: Запущенный экземпляр приложения.
    """

    async def generate(context: GenerationContext, pcm: bytes) -> None:
        """Имитирует долгий prefill и запоздалое событие после отмены.

        :param context: Канал событий конкретной генерации.
        :param pcm: Входная тестовая реплика.
        """
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            assert not await context.emit_text(uuid4(), "Устаревший ответ")

    app.state.session_service.generation_handler = generate
    with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
        socket.send_text(orjson.dumps(START).decode("utf-8"))
        orjson.loads(socket.receive_text())
        socket.send_text(orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8"))
        started = orjson.loads(socket.receive_text())
        socket.send_bytes(b"\0" * 640)
        socket.send_text(orjson.dumps({"version": 1, "type": "input.commit", "turn_id": started["turn_id"], "generation_id": started["generation_id"]}).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["type"] == "response.started"
        socket.send_text(orjson.dumps({"version": 1, "type": "response.cancel", "generation_id": started["generation_id"]}).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["type"] == "response.cancelled"
        socket.send_text(orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8"))
        following = orjson.loads(socket.receive_text())
        assert following["type"] == "input.started"
        assert following["generation_id"] != started["generation_id"]


def test_unconfigured_model_and_mismatched_ids(client: TestClient) -> None:
    """Без реального контура commit не подменяется тестовым ответом.

    :param client: Запущенный экземпляр приложения.
    """
    with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
        socket.send_text(orjson.dumps(START).decode("utf-8"))
        orjson.loads(socket.receive_text())
        socket.send_text(orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8"))
        started = orjson.loads(socket.receive_text())
        socket.send_bytes(b"\0" * 640)
        command = {"version": 1, "type": "input.commit", "turn_id": str(uuid4()), "generation_id": started["generation_id"]}
        socket.send_text(orjson.dumps(command).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["code"] == "stale_generation"
        command["turn_id"] = started["turn_id"]
        socket.send_text(orjson.dumps(command).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["code"] == "model_unavailable"


def test_invalid_handshake_sizes_and_stale_cancel(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Не принимает неверный формат и не отменяет текущую запись чужим ID.

    :param client: Запущенное приложение.
    :param monkeypatch: Временный лимит размера управляющей команды.
    """
    monkeypatch.setattr(app.state.settings.api, "max_json_bytes", 256)
    with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
        socket.send_text(orjson.dumps(START | {"input_sample_rate": 48000}).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["code"] == "invalid_command"
        socket.send_text(" " * 257)
        assert orjson.loads(socket.receive_text())["code"] == "command_too_large"
        socket.send_text(orjson.dumps(START).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["type"] == "session.ready"
        socket.send_text(orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8"))
        started = orjson.loads(socket.receive_text())
        socket.send_bytes(b"\0" * 3202)
        assert orjson.loads(socket.receive_text())["code"] == "invalid_audio_frame"
        socket.send_text(orjson.dumps({"version": 1, "type": "response.cancel", "generation_id": str(uuid4())}).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["code"] == "stale_generation"
        socket.send_bytes(b"\0" * 640)
        socket.send_text(orjson.dumps({"version": 1, "type": "input.commit", "turn_id": started["turn_id"], "generation_id": started["generation_id"]}).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["code"] == "model_unavailable"


def test_output_budget_timeout_keeps_session_responsive(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Не копит аудио, если клиент перестал подтверждать фактический playback.

    :param client: Запущенное приложение.
    :param monkeypatch: Ускоренные лимиты воспроизводимого сценария переполнения.
    """
    monkeypatch.setattr(app.state.settings.audio, "max_output_buffer_seconds", 0.02)
    monkeypatch.setattr(app.state.settings.inference, "idle_timeout_seconds", 0.05)

    async def generate(context: GenerationContext, pcm: bytes) -> None:
        """Заполняет разрешённые 20 мс и пытается опередить неподвижный playback.

        :param context: Канал одной генерации.
        :param pcm: Входная тестовая реплика.
        """
        segment = uuid4()
        await context.emit_text(segment, "Проверка переполнения")
        await context.emit_audio(segment, b"\0" * 960)
        await context.emit_audio(segment, b"\0" * 960)

    app.state.session_service.generation_handler = generate
    with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
        socket.send_text(orjson.dumps(START).decode("utf-8"))
        orjson.loads(socket.receive_text())
        socket.send_text(orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8"))
        started = orjson.loads(socket.receive_text())
        socket.send_bytes(b"\0" * 640)
        socket.send_text(orjson.dumps({"version": 1, "type": "input.commit", "turn_id": started["turn_id"], "generation_id": started["generation_id"]}).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["type"] == "response.started"
        assert orjson.loads(socket.receive_text())["type"] == "response.text"
        assert orjson.loads(socket.receive_text())["type"] == "response.audio"
        assert len(socket.receive_bytes()) == 960
        assert orjson.loads(socket.receive_text())["code"] == "output_buffer_overflow"
        socket.send_text(orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["type"] == "input.started"


def test_handshake_deadline_closes_and_releases_slot() -> None:
    """Молчаливый клиент не удерживает process-local слот после deadline."""

    async def scenario() -> None:
        """Исполняет настоящий ASGI WebSocket с управляемым транспортом."""
        incoming: asyncio.Queue[Message] = asyncio.Queue()
        incoming.put_nowait({"type": "websocket.connect"})
        sent: list[Message] = []

        async def send(message: Message) -> None:
            """Сохраняет реальные ASGI сообщения сервера.

            :param message: Событие сервера.
            """
            sent.append(message)

        settings = Settings(_env_file=None)
        settings.api.handshake_timeout_seconds = 0.01
        service = SessionService(settings)
        socket = WebSocket({"type": "websocket", "headers": []}, incoming.get, send)
        await asyncio.wait_for(service.serve(socket), timeout=0.5)
        assert any(message["type"] == "websocket.close" for message in sent)
        errors = [orjson.loads(message["text"])["code"] for message in sent if message["type"] == "websocket.send"]
        assert errors == ["session_timeout"]
        assert service.active_session is None

    asyncio.run(scenario())


def test_slow_sender_queue_limit_and_task_cleanup() -> None:
    """Медленный transport ограничен deadline, очередь конечна, задача модели отменена."""

    async def scenario() -> None:
        """Соединяет настоящий ASGI WebSocket с блокирующим send без сети."""
        incoming: asyncio.Queue[Message] = asyncio.Queue()
        incoming.put_nowait({"type": "websocket.connect"})
        incoming.put_nowait({"type": "websocket.receive", "text": orjson.dumps(START).decode("utf-8")})
        model_stopped = asyncio.Event()
        observed: list[int] = []

        async def generate(context: GenerationContext, pcm: bytes) -> None:
            """Пополняет очередь быстрее заблокированного клиента.

            :param context: Канал одной генерации.
            :param pcm: Входной тестовый PCM.
            """
            try:
                for index in range(100):
                    await context.emit_text(uuid4(), f"Фраза {index}")
                    observed.append(index)
            finally:
                model_stopped.set()

        async def send(message: Message) -> None:
            """Заполняет вход PTT и блокирует transport с началом ответа.

            :param message: Реальное ASGI-событие отправки сервера.
            """
            if message["type"] != "websocket.send":
                return
            event = orjson.loads(message["text"])
            if event["type"] == "session.ready":
                incoming.put_nowait({"type": "websocket.receive", "text": orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8")})
            elif event["type"] == "input.started":
                incoming.put_nowait({"type": "websocket.receive", "bytes": b"\0" * 640})
                command = {"version": 1, "type": "input.commit", "turn_id": event["turn_id"], "generation_id": event["generation_id"]}
                incoming.put_nowait({"type": "websocket.receive", "text": orjson.dumps(command).decode("utf-8")})
            elif event["type"] == "response.started":
                await asyncio.sleep(60)

        settings = Settings(_env_file=None)
        settings.api.outbound_queue_size = 4
        settings.api.send_timeout_seconds = 0.05
        service = SessionService(settings, generate)
        socket = WebSocket({"type": "websocket", "headers": []}, incoming.get, send)
        await asyncio.wait_for(service.serve(socket), timeout=1)
        assert 0 < len(observed) <= settings.api.outbound_queue_size
        assert model_stopped.is_set()
        assert service.active_session is None
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and task.get_name().startswith("session-")]

    asyncio.run(scenario())


def test_response_phrase_count_is_bounded(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Даже быстрый клиент не позволяет бесконечно накапливать текстовые фразы.

    :param client: Запущенное приложение.
    :param monkeypatch: Малый лимит для воспроизводимой проверки.
    """
    monkeypatch.setattr(app.state.settings, "api", app.state.settings.api.model_copy(update={"max_response_segments": 3}))

    async def generate(context: GenerationContext, pcm: bytes) -> None:
        """Публикует больше фраз, чем разрешено одной генерации.

        :param context: Канал одной генерации.
        :param pcm: Входной тестовый PCM.
        """
        for _ in range(4):
            await context.emit_text(uuid4(), "Фраза")

    app.state.session_service.generation_handler = generate
    with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
        socket.send_text(orjson.dumps(START).decode("utf-8"))
        orjson.loads(socket.receive_text())
        socket.send_text(orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8"))
        started = orjson.loads(socket.receive_text())
        socket.send_bytes(b"\0" * 640)
        socket.send_text(orjson.dumps({"version": 1, "type": "input.commit", "turn_id": started["turn_id"], "generation_id": started["generation_id"]}).decode("utf-8"))
        assert orjson.loads(socket.receive_text())["type"] == "response.started"
        for _ in range(4):
            event = orjson.loads(socket.receive_text())
            if event["type"] == "error":
                assert event["code"] == "response_too_long"
                break
            assert event["type"] == "response.text"
        else:
            pytest.fail("Число фраз не ограничено")
