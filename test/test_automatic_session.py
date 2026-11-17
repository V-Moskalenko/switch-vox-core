"""Проверяет непрерывный микрофон, ASR до паузы и перебивание через реальный WS."""

import asyncio
from collections import deque
from typing import Any, cast
from uuid import uuid4

import orjson
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket

import services.session_service as session_module
from api.fastapi_app import app
from api.schemas.session_schemas import PlaybackProgress
from services.session_service import GenerationContext, SessionService, SpeechSegment, _Session
from services.turn_detection import TurnEvent
from settings import Settings, TurnSettings

START = {"version": 1, "type": "session.start", "input_mode": "auto", "language": "ru", "input_sample_rate": 16000, "output_sample_rate": 24000, "channels": 1, "sample_width_bytes": 2}
ORIGIN = {"origin": "http://127.0.0.1:8000"}
PCM = b"\x01\0" * 320


def set_detector(monkeypatch: pytest.MonkeyPatch, script: list[list[TurnEvent]]) -> None:
    """Заменяет только VAD-решения, сохраняя реальный owner-loop и модельную очередь.

    :param monkeypatch: Временная замена CPU-детектора.
    :param script: События VAD для последовательных входных PCM-кадров.
    """
    events = deque(script)

    class ControlledDetector:
        """Детерминированный источник границ без подмены сессии и ASR-потока."""

        def __init__(self, settings: TurnSettings, max_turn_seconds: float) -> None:
            """Принимает тот же проверенный конфиг, что настоящий CPU VAD.

            :param settings: Пороги VAD.
            :param max_turn_seconds: Максимальная длительность реплики.
            """

        def feed(self, pcm: bytes, *, interrupting: bool = False) -> list[TurnEvent]:
            """Выдаёт заранее заданную границу для очередного реального WS-кадра.

            :param pcm: Действительный входной PCM.
            :param interrupting: Сейчас звучит/генерируется ответ ассистента.
            :return: События одного входного кадра.
            """
            return events.popleft() if events else []

        def abort(self) -> None:
            """Обозначает отбрасывание реплики, не меняя сценарий следующей речи."""

    monkeypatch.setattr(session_module, "TurnDetector", ControlledDetector, raising=False)


def test_automatic_input_streams_before_endpoint_and_commits_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Гипотеза появляется до паузы; только конец VAD разрешает финальный ответ.

    :param monkeypatch: Контролируемые границы речи.
    """
    set_detector(monkeypatch, [[TurnEvent("start", PCM)], [TurnEvent("audio", PCM)], [TurnEvent("end", reason="silence")]])
    received = bytearray()

    async def generate(context: GenerationContext, pcm: bytes) -> None:
        """Читает живой источник реплики и выдаёт final только после его EOF.

        :param context: Реальный контекст owner-loop с аудиоисточником.
        :param pcm: Буфер PTT; для auto он пуст, аудио приходит отдельным потоком.
        """
        assert pcm == b"" and context.input_audio is not None
        async for chunk in context.input_audio:
            received.extend(chunk)
            if len(received) == len(PCM):
                await context.emit_transcript("Гипотеза", final=False)
        await context.emit_transcript("Полная реплика", final=True)

    with TestClient(app) as client:
        app.state.session_service.generation_handler = generate
        with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
            socket.send_text(orjson.dumps(START).decode("utf-8"))
            ready = orjson.loads(socket.receive_text())
            assert ready["type"] == "session.ready", ready
            assert ready["input_mode"] == "auto"
            socket.send_bytes(PCM)
            assert orjson.loads(socket.receive_text())["type"] == "input.started"
            assert orjson.loads(socket.receive_text())["type"] == "transcript.hypothesis"
            socket.send_bytes(PCM)
            socket.send_bytes(PCM)
            assert orjson.loads(socket.receive_text())["type"] == "response.started"
            assert orjson.loads(socket.receive_text())["text"] == "Полная реплика"
            assert orjson.loads(socket.receive_text())["type"] == "response.done"
            assert received == PCM * 2
            socket.send_bytes(PCM)
            socket.send_text(orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8"))
            assert orjson.loads(socket.receive_text())["code"] == "input_mode_conflict"


def test_automatic_barge_in_discards_unheard_tail_and_late_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Новая речь отменяет старую модель, отбрасывает хвост и терпит поздний старый ACK.

    :param monkeypatch: Подтверждённая речь и конечные паузы двух реплик.
    """
    set_detector(monkeypatch, [[TurnEvent("start", PCM)], [TurnEvent("end", reason="silence")], [TurnEvent("start", PCM)], [TurnEvent("end", reason="silence")]])
    segment = uuid4()
    snapshots: list[list[tuple[str, str]]] = []
    stopped = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def generate(context: GenerationContext, pcm: bytes) -> None:
        """Первая генерация остаётся активной до настоящей отмены задачи.

        :param context: Контекст текущей реплики.
        :param pcm: Буфер PTT, не используемый в auto.
        """
        snapshots.append([(message.role, message.content) for message in context.history])
        assert context.input_audio is not None
        async for chunk in context.input_audio:
            assert chunk
        await context.emit_transcript("Первый вопрос" if len(snapshots) == 1 else "Второй вопрос", final=True)
        if len(snapshots) == 1:
            await context.emit_text(segment, "Оборванная фраза")
            await context.emit_audio(segment, b"\x01\0" * 960)
            await context.finish_segment(segment)
            try:
                await asyncio.Future()
            finally:
                await release_cleanup.wait()
                stopped.set()
        else:
            assert stopped.is_set()

    with TestClient(app) as client:
        app.state.session_service.generation_handler = generate
        with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
            socket.send_text(orjson.dumps(START).decode("utf-8"))
            assert orjson.loads(socket.receive_text())["type"] == "session.ready"
            socket.send_bytes(PCM)
            first: dict[str, Any] = orjson.loads(socket.receive_text())
            assert first["type"] == "input.started"
            socket.send_bytes(PCM)
            assert orjson.loads(socket.receive_text())["type"] == "response.started"
            assert orjson.loads(socket.receive_text())["type"] == "transcript.final"
            assert orjson.loads(socket.receive_text())["type"] == "response.text"
            assert orjson.loads(socket.receive_text())["type"] == "response.audio"
            socket.receive_bytes()
            assert orjson.loads(socket.receive_text())["type"] == "response.segment.end"
            progress = {"version": 1, "type": "playback.progress", "generation_id": first["generation_id"], "segment_id": str(segment), "played_samples": 480}
            socket.send_text(orjson.dumps(progress).decode("utf-8"))
            socket.send_bytes(PCM)
            assert orjson.loads(socket.receive_text())["type"] == "response.cancelled"
            assert orjson.loads(socket.receive_text())["type"] == "input.started"
            try:
                assert client.portal is not None
                client.portal.call(asyncio.sleep, 0)
                assert len(snapshots) == 1
            finally:
                assert client.portal is not None
                client.portal.call(release_cleanup.set)
            socket.send_text(orjson.dumps(progress | {"played_samples": 960}).decode("utf-8"))
            socket.send_text(orjson.dumps(progress | {"generation_id": str(uuid4())}).decode("utf-8"))
            assert orjson.loads(socket.receive_text())["code"] == "stale_generation"
            socket.send_bytes(PCM)
            assert orjson.loads(socket.receive_text())["type"] == "response.started"
            assert orjson.loads(socket.receive_text())["type"] == "transcript.final"
            assert orjson.loads(socket.receive_text())["type"] == "response.done"
            assert snapshots == [[], [("user", "Первый вопрос")]]


def test_stalled_automatic_asr_never_blocks_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Заполненная очередь живого ASR отменяется, а owner продолжает принимать команды.

    :param monkeypatch: VAD без конечной паузы.
    """
    set_detector(monkeypatch, [[TurnEvent("start", PCM)]] + [[TurnEvent("audio", PCM)]] * 70)

    async def generate(context: GenerationContext, pcm: bytes) -> None:
        """Не читает вход, воспроизводя зависший upstream ASR.

        :param context: Канал текущей генерации.
        :param pcm: Пустой PTT-буфер автоматического режима.
        """
        await asyncio.Future()

    with TestClient(app) as client:
        app.state.session_service.generation_handler = generate
        with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
            socket.send_text(orjson.dumps(START).decode("utf-8"))
            orjson.loads(socket.receive_text())
            socket.send_bytes(PCM)
            assert orjson.loads(socket.receive_text())["type"] == "input.started"
            for _ in range(70):
                socket.send_bytes(PCM)
            assert orjson.loads(socket.receive_text())["code"] == "input_queue_overflow"
            socket.send_text(orjson.dumps({"version": 1, "type": "input.start"}).decode("utf-8"))
            assert orjson.loads(socket.receive_text())["code"] == "input_mode_conflict"


def test_model_cannot_finalize_before_vad_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Даже ошибочный backend final не подменяет решение VAD об окончании реплики.

    :param monkeypatch: VAD, который ещё считает реплику открытой.
    """
    set_detector(monkeypatch, [[TurnEvent("start", PCM)]])

    async def generate(context: GenerationContext, pcm: bytes) -> None:
        """Имитирует преждевременный ASR final.

        :param context: Живой контекст текущего речевого входа.
        :param pcm: Неиспользуемый PTT-буфер.
        """
        await context.emit_transcript("Преждевременный final", final=True)

    with TestClient(app) as client:
        app.state.session_service.generation_handler = generate
        with client.websocket_connect("/api/v1/session", headers=ORIGIN) as socket:
            socket.send_text(orjson.dumps(START).decode("utf-8"))
            orjson.loads(socket.receive_text())
            socket.send_bytes(PCM)
            assert orjson.loads(socket.receive_text())["type"] == "input.started"
            assert orjson.loads(socket.receive_text())["code"] == "invalid_model_event"


def test_retired_ack_accepts_inflight_pair_without_changing_history() -> None:
    """Поздний ACK уже начатой пары не ломает новый turn и не дописывает услышанную историю."""
    session = _Session(SessionService(Settings(_env_file=None)), cast(WebSocket, None))
    generation, segment = uuid4(), uuid4()
    session.input_mode = "auto"
    session.generation_id = generation
    session.user_text = "Вопрос"
    session.segments[segment] = SpeechSegment(text="Не полностью услышано", prepared_samples=960, sent_samples=480, played_samples=480, complete=True)
    session._cancel_generation(notify=False)
    session._playback_progress(PlaybackProgress(version=1, type="playback.progress", generation_id=generation, segment_id=segment, played_samples=960))
    assert [(message.role, message.content) for turn in session.history for message in turn] == [("user", "Вопрос")]
