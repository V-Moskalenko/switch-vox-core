"""Проверяет события жизненного цикла и отсутствие содержимого диалога в журналах."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, NoReturn

import httpx
import orjson
import pytest
from loguru import logger
from starlette.types import Message
from starlette.websockets import WebSocket

from clients import asr_client
from clients.asr_client import ASRClient
from clients.llm_client import LLMClient
from clients.tts_client import TTSClient
from services.session_service import GenerationContext, SessionError, SessionService
from settings import Settings

if TYPE_CHECKING:
    from loguru import Message as LogMessage

PRIVATE_TEXT = "Личный вопрос, который нельзя записывать в журнал"
PRIVATE_AUDIO = b"private-pcm-content\0"
START = {"version": 1, "type": "session.start", "language": "ru", "input_sample_rate": 16000, "output_sample_rate": 24000, "channels": 1, "sample_width_bytes": 2}


@pytest.fixture
def records() -> Iterator[list[dict[str, Any]]]:
    """Собирает реальные записи Loguru без изменения конфигурации приложения.

    :yield: Записи журнала вместе с уровнями и полями корреляции.
    """
    captured: list[dict[str, Any]] = []

    def collect(message: LogMessage) -> None:
        """Сохраняет запись, которую получил настоящий обработчик Loguru.

        :param message: Отформатированное сообщение и его структурированные поля.
        """
        captured.append(dict(message.record))

    handler = logger.add(collect, level="DEBUG", diagnose=False)
    try:
        yield captured
    finally:
        logger.remove(handler)


@pytest.mark.parametrize("outcome", ["done", "cancelled", "expected_error", "unexpected_error"])
def test_session_events_are_correlated_without_dialogue_payload(records: list[dict[str, Any]], outcome: str) -> None:
    """Связывает подключение, генерацию и закрытие, сохраняя приватность диалога.

    :param records: Реальные записи журнала.
    :param outcome: Успешный ответ, отмена либо ожидаемый/неожиданный отказ модели.
    """

    async def scenario() -> None:
        """Исполняет протокол с управляемым ASGI-транспортом без подмены владельца сессии."""
        incoming: asyncio.Queue[Message] = asyncio.Queue()
        incoming.put_nowait({"type": "websocket.connect"})
        incoming.put_nowait({"type": "websocket.receive", "text": orjson.dumps(START).decode()})
        incoming.put_nowait({"type": "websocket.receive", "text": '{"version":1,"type":"input.start"}'})
        entered = asyncio.Event()

        async def generate(context: GenerationContext, pcm: bytes) -> None:
            """Публикует приватные данные только по протоколу и исполняет выбранный исход.

            :param context: Настоящий канал результатов текущей генерации.
            :param pcm: Приватное входное аудио из транспортного сценария.
            """
            assert pcm == PRIVATE_AUDIO
            entered.set()
            await context.emit_transcript(PRIVATE_TEXT, final=True)
            if outcome == "expected_error":
                raise SessionError("model_unavailable", PRIVATE_TEXT)
            if outcome == "unexpected_error":
                raise RuntimeError(PRIVATE_TEXT)
            if outcome == "cancelled":
                await asyncio.Event().wait()

        async def send(message: Message) -> None:
            """Продолжает пользовательский сценарий по наблюдаемым ответам сервера.

            :param message: Реальное исходящее ASGI-событие.
            """
            if message["type"] != "websocket.send" or "text" not in message:
                return
            event = orjson.loads(message["text"])
            if event["type"] == "input.started":
                incoming.put_nowait({"type": "websocket.receive", "bytes": PRIVATE_AUDIO})
                command = {"version": 1, "type": "input.commit", "turn_id": event["turn_id"], "generation_id": event["generation_id"]}
                incoming.put_nowait({"type": "websocket.receive", "text": orjson.dumps(command).decode()})
            elif event["type"] == "response.started" and outcome == "cancelled":
                await entered.wait()
                command = {"version": 1, "type": "response.cancel", "generation_id": event["generation_id"]}
                incoming.put_nowait({"type": "websocket.receive", "text": orjson.dumps(command).decode()})
            elif event["type"] in {"response.done", "response.cancelled", "error"}:
                incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})

        service = SessionService(Settings(_env_file=None), generate)
        await asyncio.wait_for(service.serve(WebSocket({"type": "websocket"}, incoming.get, send)), timeout=2)
        assert entered.is_set()

    asyncio.run(scenario())
    events = {record["extra"].get("event"): record for record in records}
    for name in ("session_opened", "session_configured", "generation_started", "session_closed"):
        assert name in events
        assert events[name]["extra"]["session_id"] == events["session_opened"]["extra"]["session_id"]
    expected = {"done": "generation_completed", "cancelled": "generation_cancelled", "expected_error": "session_failed", "unexpected_error": "generation_failed"}
    assert expected[outcome] in events
    if outcome == "unexpected_error":
        assert events["generation_failed"]["level"].name == "ERROR"
        assert events["generation_failed"]["extra"]["error_type"] == "RuntimeError"
    if outcome == "expected_error":
        assert events["session_failed"]["level"].name == "WARNING"
        assert events["session_failed"]["extra"]["code"] == "model_unavailable"
    assert events["generation_started"]["extra"]["generation_id"]
    assert events["generation_started"]["extra"]["turn_id"]
    assert PRIVATE_TEXT not in repr(records)
    assert PRIVATE_AUDIO.decode() not in repr(records)
    assert all(record["exception"] is None for record in records)


@pytest.mark.parametrize("stage", ["asr", "llm", "tts"])
def test_model_transport_failures_log_stage_without_error_payload(records: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    """Логирует границы отказавшего транспорта без URL, аудио и сообщения исключения.

    :param records: Реальные записи журнала.
    :param monkeypatch: Подмена только сетевого подключения ASR.
    :param stage: Проверяемый клиент модели.
    """

    def reject_asr(*args: object, **kwargs: object) -> NoReturn:
        """Воспроизводит отказ соединения без настоящего сокета.

        :param args: Позиционные параметры соединения.
        :param kwargs: Параметры сетевых ограничений клиента.
        :raises OSError: Контролируемый сетевой отказ с приватным сообщением.
        """
        raise OSError(PRIVATE_TEXT)

    def reject_http(request: httpx.Request) -> NoReturn:
        """Возвращает транспортный отказ с приватным сообщением и объектом запроса.

        :param request: Исходящий запрос с текстом либо приватным query-параметром.
        :raises httpx.ConnectError: Контролируемый сетевой отказ.
        """
        raise httpx.ConnectError(PRIVATE_TEXT, request=request)

    async def scenario() -> None:
        """Запускает настоящий клиент и наблюдает его публичную ошибку."""
        settings = Settings(_env_file=None)
        async with httpx.AsyncClient(transport=httpx.MockTransport(reject_http)) as client:
            stream: AsyncIterator[object]
            if stage == "asr":
                monkeypatch.setattr(asr_client, "connect", reject_asr)
                stream = ASRClient(settings).transcribe(PRIVATE_AUDIO)
            elif stage == "llm":
                model = LLMClient(settings, client)
                model.base_url += "?private-query=do-not-log"
                stream = model.stream_response([{"role": "user", "content": PRIVATE_TEXT}])
            else:
                stream = TTSClient(settings, client).synthesize(PRIVATE_TEXT)
            with pytest.raises(SessionError):
                await anext(stream)

    asyncio.run(scenario())
    events = {record["extra"].get("event"): record for record in records}
    for suffix in ("started", "failed", "closed"):
        assert f"{stage}_{suffix}" in events
    assert events[f"{stage}_failed"]["level"].name == "WARNING"
    assert PRIVATE_TEXT not in repr(records)
    assert "do-not-log" not in repr(records)
    assert PRIVATE_AUDIO.decode() not in repr(records)
    assert all(record["exception"] is None for record in records)
