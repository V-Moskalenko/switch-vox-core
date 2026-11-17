"""Проверяет реальные протоколы клиентов каскада, поток PCM и закрытие upstream."""

import asyncio
import base64
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import cast
from uuid import UUID

import httpx
import orjson
import pytest
from websockets.asyncio.server import ServerConnection, serve

from clients.asr_client import ASRClient
from clients.llm_client import LLMClient
from clients.tts_client import TTSClient
from services.cascade_service import CascadeService
from services.session_service import DialogueMessage, GenerationContext, SessionError
from settings import Settings


class ByteStream(httpx.AsyncByteStream):
    """Выдаёт управляемые части настоящего HTTP-body и фиксирует закрытие."""

    def __init__(self, chunks: list[bytes], *, gate: asyncio.Event | None = None) -> None:
        """Сохраняет body без объединения границ транспортных chunks.

        :param chunks: Части одного HTTP-ответа.
        :param gate: Необязательное ожидание перед завершением body.
        """
        self.chunks = chunks
        self.gate = gate
        self.closed = False
        self.started = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Отдаёт байты и позволяет отменить ещё открытый upstream.

        :yield: Следующая транспортная порция body.
        """
        self.started.set()
        for chunk in self.chunks:
            yield chunk
        if self.gate is not None:
            await self.gate.wait()

    async def aclose(self) -> None:
        """Фиксирует фактическое закрытие HTTP-потока."""
        self.closed = True


@dataclass
class OutputContext:
    """Наблюдает публичные события обработчика без подмены модельных clients."""

    history: tuple[DialogueMessage, ...] = ()
    input_audio: AsyncIterator[bytes] | None = None
    events: list[tuple[str, object]] = field(default_factory=list)
    first_audio: asyncio.Event = field(default_factory=asyncio.Event)
    accept: bool = True
    reject_kind: str = ""

    async def emit_transcript(self, text: str, *, final: bool) -> bool:
        """Сохраняет ASR-событие.

        :param text: Гипотеза или окончательный transcript.
        :param final: Признак окончательного transcript.
        :return: Разрешена ли ещё публикация.
        """
        self.events.append(("final" if final else "hypothesis", text))
        return self.accept and self.reject_kind != "transcript"

    async def emit_text(self, segment_id: UUID, text: str) -> bool:
        """Сохраняет фразу перед синтезом.

        :param segment_id: Публичный идентификатор фразы.
        :param text: Озвучиваемая фраза.
        :return: Разрешена ли ещё публикация.
        """
        self.events.append(("text", text))
        return self.accept and self.reject_kind != "text"

    async def emit_audio(self, segment_id: UUID, pcm: bytes) -> bool:
        """Сохраняет PCM и уведомляет наблюдателя первого звука.

        :param segment_id: Публичный идентификатор фразы.
        :param pcm: PCM16 mono 24 кГц.
        :return: Разрешена ли ещё публикация.
        """
        self.events.append(("audio", pcm))
        self.first_audio.set()
        return self.accept and self.reject_kind != "audio"

    async def finish_segment(self, segment_id: UUID) -> bool:
        """Фиксирует успешный EOF аудио фразы.

        :param segment_id: Идентификатор завершённой фразы.
        :return: Разрешена ли ещё публикация.
        """
        self.events.append(("end", segment_id))
        return self.accept and self.reject_kind != "end"


def sse(content: str = "", **delta: object) -> bytes:
    """Кодирует настоящий SSE event vLLM.

    :param content: Обычный текст ответа.
    :param delta: Остальные поля token delta.
    :return: Байты одной data-строки и разделителя событий.
    """
    return b"data: " + orjson.dumps({"choices": [{"delta": {"content": content, **delta}}]}) + b"\n\n"


def test_asr_uses_pcm_protocol_and_replaces_hypotheses() -> None:
    """Использует настоящий WebSocket и проверяет входной PCM, flush и final."""

    async def scenario() -> None:
        """Соединяет клиент с локальным concrete ASR-протоколом."""
        received = bytearray()

        async def recognize(socket: ServerConnection) -> None:
            """Проверяет контракт без GPU inference.

            :param socket: Настоящее внутреннее ASR-соединение.
            """
            await socket.send(orjson.dumps({"type": "session.created"}).decode("utf-8"))
            update = await socket.recv()
            assert isinstance(update, str)
            assert orjson.loads(update) == {"type": "session.update", "model": "qwen3-asr"}
            commit = await socket.recv()
            assert isinstance(commit, str)
            assert orjson.loads(commit) == {"type": "input_audio_buffer.commit"}
            while True:
                raw = await socket.recv()
                assert isinstance(raw, str)
                event = orjson.loads(raw)
                if event.get("final") is True:
                    break
                chunk = base64.b64decode(event["audio"], validate=True)
                assert len(chunk) <= 3200 and len(chunk) % 2 == 0
                received.extend(chunk)
            for kind, text in [("transcription.hypothesis", "Привет"), ("transcription.hypothesis", "Привет, мир"), ("transcription.done", "Привет, мир.")]:
                await socket.send(orjson.dumps({"type": kind, "text": text}).decode("utf-8"))
            await socket.wait_closed()

        async with serve(recognize, "127.0.0.1", 0) as server:
            settings = Settings(_env_file=None)
            settings.inference.asr_base_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"  # type: ignore[assignment]
            pcm = b"\x01\0" * 4000
            events = [event async for event in ASRClient(settings).transcribe(pcm)]
            assert events == [("Привет", False), ("Привет, мир", False), ("Привет, мир.", True)]
            assert bytes(received) == pcm

    asyncio.run(scenario())


@pytest.mark.parametrize("role", ["llm", "tts"])
def test_cancel_before_first_result_closes_http_stream(role: str) -> None:
    """Отмена не ждёт первой token delta либо первого PCM от runtime.

    :param role: Модель, удерживающая body в стадии prefill.
    """

    async def scenario() -> None:
        """Отменяет реальное ожидание чтения HTTP body без таймерной гонки."""
        stream = ByteStream([], gate=asyncio.Event())

        def respond(request: httpx.Request) -> httpx.Response:
            """Публикует headers сразу, удерживая первый body chunk.

            :param request: Запрос health либо model inference.
            :return: Открытый stream без первого результата.
            """
            if request.url.path == "/health":
                return httpx.Response(200, json={"ready": True, "busy": False})
            headers = {"Content-Type": "text/event-stream"} if role == "llm" else {"Content-Type": "audio/pcm", "X-Sample-Rate": "24000", "X-Channels": "1", "X-Sample-Width-Bytes": "2"}
            return httpx.Response(200, headers=headers, stream=stream)

        settings = Settings(_env_file=None)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            task: asyncio.Task[str] | asyncio.Task[bytes]
            if role == "llm":
                tokens = LLMClient(settings, client).stream_response([{"role": "user", "content": "Привет"}])
                task = asyncio.create_task(anext(tokens))
            else:
                audio = TTSClient(settings, client).synthesize("Привет")
                task = asyncio.create_task(anext(audio))
            await asyncio.wait_for(stream.started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert stream.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("body", [b"data: {bad}\n\n", sse("Нет EOF"), b"data: []\n\ndata: [DONE]\n\n"])
def test_llm_rejects_invalid_or_unfinished_sse(body: bytes) -> None:
    """Не объявляет повреждённый ответ или преждевременный EOF успешным.

    :param body: Некорректный поток vLLM.
    """

    async def scenario() -> None:
        """Пропускает реальный HTTP stream parser через контролируемый transport."""
        stream = ByteStream([body])

        def respond(request: httpx.Request) -> httpx.Response:
            """Возвращает повреждённый body выбранного сценария.

            :param request: Реальный запрос клиента.
            :return: SSE-ответ с проверяемым EOF.
            """
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(SessionError):
                _ = [text async for text in LLMClient(Settings(_env_file=None), client).stream_response([{"role": "user", "content": "Привет"}])]
        assert stream.closed

    asyncio.run(scenario())


def test_llm_uses_exact_token_budget_and_whole_turn_pruning() -> None:
    """Удаляет старую целую пару и сохраняет одинаковый chat template при подсчёте."""

    async def scenario() -> None:
        """Заставляет первый реальный tokenize request превысить бюджет."""
        requests: list[dict[str, object]] = []

        def respond(request: httpx.Request) -> httpx.Response:
            """Проверяет payload и возвращает точный счётчик токенов.

            :param request: Запрос /tokenize к корню vLLM.
            :return: Проверяемый счётчик и реальное context window.
            """
            assert request.url.path == "/tokenize"
            assert request.extensions["timeout"]["connect"] == 5
            payload = orjson.loads(request.content)
            assert payload["add_generation_prompt"] is True
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            requests.append(payload)
            return httpx.Response(200, json={"count": 4000 if len(requests) == 1 else 100, "max_model_len": 4096})

        history = (DialogueMessage("user", "Старый вопрос"), DialogueMessage("assistant", "Услышанный ответ"), DialogueMessage("user", "Новый вопрос"), DialogueMessage("assistant", "Ещё ответ"))
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            messages = await LLMClient(Settings(_env_file=None), client).prepare_messages(history, "Текущая реплика")
        assert len(requests) == 2
        assert [message["content"] for message in messages[1:]] == ["Новый вопрос", "Ещё ответ", "Текущая реплика"]

    asyncio.run(scenario())


def test_llm_rejects_oversized_current_turn_without_generation() -> None:
    """Не обрезает текущую реплику, если она одна превышает точный token budget."""

    async def scenario() -> None:
        """Проверяет отказ по реальному ответу tokenizer endpoint."""

        def respond(request: httpx.Request) -> httpx.Response:
            """Разрешает только проверку токенов, не генерацию.

            :param request: Фактический HTTP-запрос.
            :return: Счётчик, не оставляющий запаса для ответа.
            """
            assert request.url.path == "/tokenize"
            return httpx.Response(200, json={"count": 4000, "max_model_len": 4096})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(SessionError) as error:
                await LLMClient(Settings(_env_file=None), client).prepare_messages((), "Длинная реплика")
            assert error.value.code == "context_too_long"

    asyncio.run(scenario())


def test_llm_ignores_reasoning_and_handles_split_utf8_sse() -> None:
    """Служебное reasoning не озвучивается даже при разделённых байтах UTF-8."""

    async def scenario() -> None:
        """Разбивает реальный SSE stream на отдельные транспортные байты."""
        body = b': keepalive\n\ndata: {"choices": []}\n\n' + sse(reasoning_content="Скрытое рассуждение") + sse("Привет.") + b"data: [DONE]\n\n"

        def respond(request: httpx.Request) -> httpx.Response:
            """Выдаёт SSE c выделенным reasoning field.

            :param request: Запрос chat completions.
            :return: Однобайтовые chunks, включая разорванный UTF-8.
            """
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=ByteStream([body[index : index + 1] for index in range(len(body))]))

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = [text async for text in LLMClient(Settings(_env_file=None), client).stream_response([{"role": "user", "content": "Привет"}])]
            assert "".join(result) == "Привет."

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body,content_type,code",
    [
        (b"data: [DONE]\n\n", "text/event-stream", "llm_empty_response"),
        (sse("Текст"), "application/json", "llm_invalid_response"),
        (b"event: bad\n\n", "text/event-stream", "llm_invalid_response"),
        (b"data: " + b"x" * 65537 + b"\n\n", "text/event-stream", "llm_invalid_response"),
        (b'data: {"choices": [{}]}\n\n', "text/event-stream", "llm_invalid_response"),
        (sse(content=cast(str, 123)), "text/event-stream", "llm_invalid_response"),
        (sse("<think>Скрыто"), "text/event-stream", "llm_invalid_response"),
        (sse("x" * 26000), "text/event-stream", "response_too_long"),
    ],
)
def test_llm_rejects_invalid_public_stream_contract(body: bytes, content_type: str, code: str) -> None:
    """Неверный runtime-контракт не доходит до синтеза и сохраняет безопасный код.

    :param body: Проверяемый SSE payload либо повреждённое событие.
    :param content_type: Заявленный тип HTTP body.
    :param code: Ожидаемый публичный код отказа.
    """

    async def scenario() -> None:
        """Читает HTTP body настоящим LLM-клиентом."""
        stream = ByteStream([body])

        def respond(request: httpx.Request) -> httpx.Response:
            """Возвращает конкретное нарушение протокола.

            :param request: Запрос chat completions.
            :return: Ответ с контролируемым типом и содержимым.
            """
            return httpx.Response(200, headers={"Content-Type": content_type}, stream=stream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(SessionError) as error:
                _ = [text async for text in LLMClient(Settings(_env_file=None), client).stream_response([])]
            assert error.value.code == code
        assert stream.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("payload", [{"count": True, "max_model_len": 4096}, {"count": -1, "max_model_len": 4096}, [], "invalid-json", "http-error"])
def test_llm_tokenizer_failures_are_explicit(payload: object) -> None:
    """Не заменяет неизвестный token budget приблизительной оценкой текста.

    :param payload: Некорректный ответ либо отказ tokenizer endpoint.
    """

    async def scenario() -> None:
        """Проверяет безопасный отказ перед любым chat completion."""

        def respond(request: httpx.Request) -> httpx.Response:
            """Моделирует только transport/протокольный отказ tokenizer.

            :param request: Фактический запрос /tokenize.
            :return: Ошибка либо неверный JSON.
            """
            assert request.url.path == "/tokenize"
            if payload == "http-error":
                return httpx.Response(503)
            if payload == "invalid-json":
                return httpx.Response(200, text="broken")
            return httpx.Response(200, json=payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(SessionError):
                await LLMClient(Settings(_env_file=None), client).prepare_messages((), "Реплика")

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["handshake", "error", "unknown", "json", "shape"])
def test_asr_protocol_errors_close_connection(failure: str) -> None:
    """Повреждённый ASR stream явно отклоняется и не удерживает соединение.

    :param failure: Этап и форма нарушения upstream-контракта.
    """

    async def scenario() -> None:
        """Запускает настоящий WS endpoint без GPU модели."""
        closed = asyncio.Event()

        async def recognize(socket: ServerConnection) -> None:
            """Выдаёт выбранный некорректный ответ и наблюдает закрытие.

            :param socket: Реальное model-client соединение.
            """
            await socket.send(orjson.dumps({"type": "wrong" if failure == "handshake" else "session.created"}).decode("utf-8"))
            if failure != "handshake":
                while not orjson.loads(await socket.recv()).get("final"):
                    pass
                event: object = {"type": "error"} if failure == "error" else {"type": "unknown", "text": "bad"}
                if failure == "shape":
                    event = []
                await socket.send("invalid-json" if failure == "json" else orjson.dumps(event).decode("utf-8"))
            await socket.wait_closed()
            closed.set()

        async with serve(recognize, "127.0.0.1", 0) as server:
            settings = Settings(_env_file=None)
            settings.inference.asr_base_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"  # type: ignore[assignment]
            with pytest.raises(SessionError):
                _ = [event async for event in ASRClient(settings).transcribe(b"\0\0")]
            await asyncio.wait_for(closed.wait(), timeout=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["not_ready", "invalid_json", "health_status", "busy", "format", "synthesis_status"])
def test_tts_waits_for_idle_and_rejects_failed_contracts(failure: str) -> None:
    """Отказ готовности, формата или синтеза не выдаётся за успешный PCM EOF.

    :param failure: Наблюдаемый отказ runtime или занятый worker.
    """

    async def scenario() -> None:
        """Использует короткий deadline занятого runtime без CUDA."""
        settings = Settings(_env_file=None)
        settings.inference.first_response_timeout_seconds = 0.01

        def respond(request: httpx.Request) -> httpx.Response:
            """Возвращает контролируемый отказ health либо синтеза.

            :param request: Настоящий HTTP-запрос TTS-клиента.
            :return: Выбранный ошибочный ответ runtime.
            """
            if request.url.path == "/health":
                if failure == "health_status":
                    return httpx.Response(503)
                if failure == "invalid_json":
                    return httpx.Response(200, text="broken")
                return httpx.Response(200, json={"ready": failure != "not_ready", "busy": failure == "busy"})
            if failure == "synthesis_status":
                return httpx.Response(503)
            return httpx.Response(200, headers={"Content-Type": "audio/wav"}, content=b"fake WAV")

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(SessionError):
                _ = [chunk async for chunk in TTSClient(settings, client).synthesize("Привет")]

    asyncio.run(scenario())


@pytest.mark.parametrize("chunks,valid", [([b"\x01", b"\0\x02", b"\0"], True), ([b"\x01"], False), ([], False)])
def test_tts_handles_arbitrary_http_chunks_and_rejects_truncated_pcm(chunks: list[bytes], valid: bool) -> None:
    """HTTP chunk boundary не изменяет отсчёты PCM и не маскирует нечётный EOF.

    :param chunks: Части реального HTTP body.
    :param valid: Должен ли stream содержать полноценное аудио.
    """

    async def scenario() -> None:
        """Проверяет stream без подмены итератора самого TTS-клиента."""
        stream = ByteStream(chunks)

        def respond(request: httpx.Request) -> httpx.Response:
            """Возвращает health либо PCM response.

            :param request: Реальный запрос к TTS runtime.
            :return: Ответ протокола закреплённого TTS.
            """
            if request.url.path == "/health":
                return httpx.Response(200, json={"ready": True, "busy": False})
            return httpx.Response(200, headers={"Content-Type": "audio/pcm", "X-Sample-Rate": "24000", "X-Channels": "1", "X-Sample-Width-Bytes": "2"}, stream=stream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            tts = TTSClient(Settings(_env_file=None), client)
            if valid:
                result = [chunk async for chunk in tts.synthesize("Привет")]
                assert b"".join(result) == b"\x01\0\x02\0"
                assert all(len(chunk) % 2 == 0 for chunk in result)
            else:
                with pytest.raises(SessionError):
                    _ = [chunk async for chunk in tts.synthesize("Привет")]
        assert stream.closed

    asyncio.run(scenario())


def test_cascade_delivers_audio_before_llm_eof_and_cancel_closes_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Настоящие HTTP clients не ждут полного текста и закрываются по отмене.

    :param monkeypatch: Заменяет только отдельно проверенный ASR transport.
    """

    async def scenario() -> None:
        """Удерживает LLM открытым, пока каскад уже отдаёт PCM первой фразы."""
        gate = asyncio.Event()
        llm_stream = ByteStream([sse("Первая фраза. ")], gate=gate)
        tts_stream = ByteStream([b"\x01\0" * 480], gate=gate)
        payloads: list[dict[str, object]] = []

        async def transcribe(self: ASRClient, pcm: bytes) -> AsyncIterator[tuple[str, bool]]:
            """Возвращает final как из отдельно проверенного ASR gateway.

            :param pcm: Входная реплика пользователя.
            :yield: Окончательный transcript.
            """
            yield "Тестовый вопрос", True

        def respond(request: httpx.Request) -> httpx.Response:
            """Маршрутизирует реальные HTTP requests без замены clients.

            :param request: Запрос LLM/TTS.
            :return: Контролируемый поток выбранной модели.
            """
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 30, "max_model_len": 4096})
            if request.url.path.endswith("/chat/completions"):
                payloads.append(orjson.loads(request.content))
                return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=llm_stream)
            if request.url.path == "/health":
                return httpx.Response(200, json={"ready": True, "busy": False})
            assert orjson.loads(request.content) == {"text": "Первая фраза."}
            return httpx.Response(200, headers={"Content-Type": "audio/pcm", "X-Sample-Rate": "24000", "X-Channels": "1", "X-Sample-Width-Bytes": "2"}, stream=tts_stream)

        monkeypatch.setattr(ASRClient, "transcribe", transcribe)
        context = OutputContext(history=(DialogueMessage("user", "Предыдущий вопрос"), DialogueMessage("assistant", "Услышанная фраза")))
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            task = asyncio.create_task(CascadeService(Settings(_env_file=None), client).generate(cast(GenerationContext, context), b"\0\0"))
            try:
                await asyncio.wait_for(context.first_audio.wait(), timeout=1)
                assert not task.done()
                assert not llm_stream.closed and not tts_stream.closed
                assert context.events[:2] == [("final", "Тестовый вопрос"), ("text", "Первая фраза.")]
                assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
                assert payloads[0]["messages"][1] == {"role": "user", "content": "Предыдущий вопрос"}  # type: ignore[index]
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert llm_stream.closed and tts_stream.closed

    asyncio.run(scenario())


def test_cascade_successful_eof_flushes_every_phrase_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Разбитый на части SSE и хвост без точки полностью озвучиваются по одному разу.

    :param monkeypatch: Заменяет только отдельно проверенный ASR transport.
    """

    async def scenario() -> None:
        """Пропускает успешный ответ через настоящий SSE/PCM pipeline."""
        body = sse("Первая фраза. ") + sse(reasoning_content="Не озвучивать") + sse("Вторая фраза без точки") + b"data: [DONE]\n\n"
        spoken: list[str] = []
        streams: list[ByteStream] = []

        async def transcribe(self: ASRClient, pcm: bytes) -> AsyncIterator[tuple[str, bool]]:
            """Выдаёт final закреплённого ASR-контракта.

            :param pcm: Входной PCM реплики.
            :yield: Окончательная транскрипция.
            """
            yield "Вопрос", True

        def respond(request: httpx.Request) -> httpx.Response:
            """Возвращает SSE по три байта и нечётные PCM transport chunks.

            :param request: Реальный запрос model client.
            :return: Успешный body выбранного runtime.
            """
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 10, "max_model_len": 4096})
            if request.url.path.endswith("/chat/completions"):
                stream = ByteStream([body[index : index + 3] for index in range(0, len(body), 3)])
                streams.append(stream)
                return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)
            if request.url.path == "/health":
                return httpx.Response(200, json={"ready": True, "busy": False})
            spoken.append(orjson.loads(request.content)["text"])
            stream = ByteStream([b"\x01", b"\0\x02", b"\0"])
            streams.append(stream)
            return httpx.Response(200, headers={"Content-Type": "audio/pcm", "X-Sample-Rate": "24000", "X-Channels": "1", "X-Sample-Width-Bytes": "2"}, stream=stream)

        monkeypatch.setattr(ASRClient, "transcribe", transcribe)
        context = OutputContext()
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            await CascadeService(Settings(_env_file=None), client).generate(cast(GenerationContext, context), b"\0\0")
        assert spoken == ["Первая фраза.", "Вторая фраза без точки"]
        assert [payload for kind, payload in context.events if kind == "text"] == spoken
        assert len([kind for kind, _ in context.events if kind == "end"]) == 2
        assert b"".join(cast(bytes, payload) for kind, payload in context.events if kind == "audio") == b"\x01\0\x02\0" * 2
        assert all(stream.closed for stream in streams)

    asyncio.run(scenario())


@pytest.mark.parametrize("role", ["llm", "tts"])
def test_upstream_idle_deadline_excludes_downstream_backpressure(role: str) -> None:
    """Медленное чтение потребителем не выдаётся за бездействие upstream.

    :param role: Поток текста либо PCM с отдельным idle deadline.
    """

    async def scenario() -> None:
        """Приостанавливает потребителя дольше idle timeout между готовыми chunks."""
        settings = Settings(_env_file=None)
        settings.inference.idle_timeout_seconds = 0.01

        def respond(request: httpx.Request) -> httpx.Response:
            """Отдаёт две уже готовые части без задержки на стороне runtime.

            :param request: Реальный HTTP-запрос model client.
            :return: Полный успешный текст или PCM.
            """
            if request.url.path == "/health":
                return httpx.Response(200, json={"ready": True, "busy": False})
            if role == "llm":
                return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=ByteStream([sse("Привет. "), sse("Пока.") + b"data: [DONE]\n\n"]))
            return httpx.Response(200, headers={"Content-Type": "audio/pcm", "X-Sample-Rate": "24000", "X-Channels": "1", "X-Sample-Width-Bytes": "2"}, stream=ByteStream([b"\x01\0", b"\x02\0"]))

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            if role == "llm":
                text = LLMClient(settings, client).stream_response([{"role": "user", "content": "Привет"}])
                assert await anext(text) == "Привет. "
                await asyncio.sleep(0.03)
                assert [chunk async for chunk in text] == ["Пока."]
            else:
                audio = TTSClient(settings, client).synthesize("Привет")
                assert await anext(audio) == b"\x01\0"
                await asyncio.sleep(0.03)
                assert [chunk async for chunk in audio] == [b"\x02\0"]

    asyncio.run(scenario())
