"""WebSocket-шлюз ASR: распознавание накопленного PCM через HTTP API vLLM."""

import asyncio
import base64
import io
import os
import sys
import time
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

import httpx
import orjson
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from loguru import logger


class ASRResponseError(ValueError):
    """Успешный HTTP-ответ ASR не содержит корректного строкового текста."""


def transcription_text(response: httpx.Response) -> str:
    """Проверяет конкретный JSON-контракт backend, не публикуя его ошибочный payload.

    :param response: Успешный HTTP-ответ стандартного vLLM ASR endpoint.
    :return: Строковый transcript, включая допустимую пустую строку.
    :raises ASRResponseError: Ответ не JSON-объект либо text отсутствует/не является строкой.
    """
    try:
        payload = orjson.loads(response.content)
    except ValueError as exc:
        raise ASRResponseError("ASR backend вернул неверный JSON") from exc
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str):
        raise ASRResponseError("ASR backend не вернул строковый transcript")
    return text


def pcm_wav(pcm: bytes) -> bytes:
    """Упаковывает реальные PCM16 LE mono 16 kHz в WAV без изменения отсчётов.

    :param pcm: Накопленная с начала реплики последовательность отсчётов.
    :return: WAV для стандартного endpoint закреплённого vLLM.
    """
    target = io.BytesIO()
    with wave.open(target, "wb") as recording:
        recording.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        recording.writeframes(pcm)
    return target.getvalue()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Открывает HTTP-клиент одного backend, не загружая второй экземпляр модели.

    :param application: Экземпляр ASR gateway данного контейнера.
    :yield: Работа runtime с одним владельцем голосового входа.
    """
    # Фоновая очередь отделяет запись в stderr от обработки аудио в event loop.
    await logger.complete()
    logger.remove()
    sink = logger.add(
        sys.stderr, level=os.getenv("RUNTIME_LOG_LEVEL", "INFO"), enqueue=True, diagnose=False, backtrace=False, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message} | {extra}"
    )
    try:
        application.state.active = False
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8001", timeout=60, trust_env=False) as client:
            application.state.client = client
            logger.info("ASR-шлюз готов к приёму реплик")
            yield
    except Exception as exc:
        logger.error("Сбой жизненного цикла ASR-шлюза: type={}", type(exc).__name__)
        raise
    finally:
        logger.info("ASR-шлюз остановлен")
        await logger.complete()
        logger.remove(sink)


app = FastAPI(title="Switch Vox Qwen3 ASR runtime", lifespan=lifespan)


@app.get("/health")
async def health(response: Response) -> Response:
    """Проверяет реальную готовность единственного vLLM ASR backend.

    :param response: HTTP-ответ для кода недоступности.
    :return: JSON-ответ с готовностью backend и HTTP 200 либо 503.
    """
    try:
        backend = await app.state.client.get("/health", timeout=2)
        ready = backend.status_code == 200
    except httpx.HTTPError:
        ready = False
    response.status_code = 200 if ready else 503
    return Response(content=orjson.dumps({"ready": ready}), media_type="application/json", status_code=response.status_code)


class ASRConnection:
    """Владеет PCM, состоянием протокола и задачей распознавания одной реплики."""

    def __init__(self, socket: WebSocket, client: httpx.AsyncClient) -> None:
        """Создаёт состояние соединения без второго экземпляра ASR-модели.

        :param socket: Внутреннее WebSocket-соединение с API.
        :param client: Общий HTTP-клиент единственного vLLM backend.
        """
        self.socket = socket
        self.client = client
        self.buffer = bytearray()
        self.changed = asyncio.Event()
        self.final = False
        self.configured = False
        self.generation: asyncio.Task[None] | None = None
        self.sender = asyncio.Lock()

    async def send(self, event: dict[str, object]) -> None:
        """Передаёт JSON-события последовательно под блокировкой отправки.

        :param event: Событие конкретного внутреннего ASR-протокола.
        """
        async with self.sender:
            await self.socket.send_text(orjson.dumps(event).decode())

    async def transcribe(self, pcm: bytes) -> str:
        """Отправляет текущий полный префикс стандартному endpoint ASR.

        :param pcm: Снимок PCM с начала реплики.
        :return: Проверенная полная гипотеза либо final.
        """
        backend = await self.client.post(
            "/v1/audio/transcriptions", files={"file": ("speech.wav", pcm_wav(pcm), "audio/wav")}, data={"model": "qwen3-asr", "language": "ru", "response_format": "json", "temperature": "0"}
        )
        backend.raise_for_status()
        return transcription_text(backend)

    async def decode(self) -> None:
        """Отправляет полные гипотезы по мере накопления PCM и итоговый текст после final."""
        processed = 0
        passes = 0
        try:
            while True:
                await self.changed.wait()
                self.changed.clear()
                if not self.final and len(self.buffer) - processed < 32000:
                    continue
                snapshot = bytes(self.buffer)
                if not snapshot:
                    await self.send({"type": "transcription.done", "text": ""})
                    return
                text = await self.transcribe(snapshot)
                passes += 1
                processed = len(snapshot)
                if self.final and processed == len(self.buffer):
                    await self.send({"type": "transcription.done", "text": text})
                    logger.debug("Распознавание ASR завершено: audio_bytes={}, passes={}", processed, passes)
                    return
                await self.send({"type": "transcription.hypothesis", "text": text})
                if self.final or len(self.buffer) - processed >= 32000:
                    self.changed.set()
        except (httpx.HTTPError, ASRResponseError) as exc:
            logger.error("ASR backend не завершил распознавание: type={}, audio_bytes={}, passes={}", type(exc).__name__, len(self.buffer), passes)
            code = "asr_invalid_response" if isinstance(exc, ASRResponseError) else "asr_unavailable"
            await self.send({"type": "error", "code": code, "error": "ASR backend не завершил корректное распознавание"})
            await self.socket.close(code=1011)

    async def receive_command(self) -> dict[str, object] | None:
        """Читает JSON-команду и закрывает соединение при неверном формате/размере.

        :return: Объект команды либо None после закрытия транспорта.
        """
        raw = await self.socket.receive_text()
        if len(raw.encode("utf-8")) > 8192:
            logger.warning("Отклонена команда ASR: превышен предел размера JSON")
            await self.socket.close(code=1009)
            return None
        try:
            command = orjson.loads(raw)
        except orjson.JSONDecodeError:
            command = None
        if not isinstance(command, dict):
            logger.warning("Отклонена команда ASR: неверный JSON-объект")
            await self.socket.close(code=1008)
            return None
        return dict(command)

    async def commit(self, *, final: bool) -> bool:
        """Начинает декодирование либо просит полный final после последнего PCM.

        :param final: Явно завершить подачу аудио данной реплики.
        :return: Допустима ли команда в текущем состоянии.
        """
        if final:
            if self.generation is None:
                logger.warning("Отклонён final ASR до запуска распознавания")
                await self.socket.close(code=1008)
                return False
            self.final = True
            self.changed.set()
        elif self.generation is None:
            self.generation = asyncio.create_task(self.decode())
        return True

    async def append(self, audio: object) -> bool:
        """Проверяет base64/PCM до добавления ограниченного кадра в текущую реплику.

        :param audio: Base64-поле внутренней команды API приложения.
        :return: Принят ли кадр; False означает закрытый транспорт.
        """
        try:
            if not isinstance(audio, (str, bytes)):
                raise TypeError("Нужно base64-поле audio")
            pcm = base64.b64decode(audio, validate=True)
        except (ValueError, TypeError):
            logger.warning("Отклонён кадр ASR: неверное base64-поле")
            await self.socket.close(code=1008)
            return False
        if not pcm or len(pcm) % 2 or len(pcm) > 3200 or len(self.buffer) + len(pcm) > 960000:
            logger.warning("Отклонён кадр ASR: frame_bytes={}, buffered_bytes={}", len(pcm), len(self.buffer))
            await self.socket.close(code=1009)
            return False
        self.buffer.extend(pcm)
        self.changed.set()
        return True

    async def handle_command(self, command: dict[str, object]) -> bool:
        """Применяет одну допустимую команду к состоянию этой реплики.

        :param command: Декодированный JSON-объект без транспортной обёртки.
        :return: Продолжать ли чтение следующей команды.
        """
        kind = command.get("type")
        if kind == "session.update" and command.get("model") == "qwen3-asr":
            self.configured = True
            return True
        if self.configured:
            if kind == "input_audio_buffer.commit":
                return await self.commit(final=command.get("final") is True)
            if kind == "input_audio_buffer.append" and self.generation is not None and not self.final:
                return await self.append(command.get("audio", ""))
        await self.socket.close(code=1008)
        return False

    async def run(self) -> None:
        """Обслуживает одну WS-реплику и дожидается отмены своего HTTP decoder."""
        started = time.perf_counter()
        logger.debug("Начата ASR-реплика")
        try:
            await self.socket.accept()
            await self.send({"type": "session.created"})
            while (command := await self.receive_command()) is not None:
                if not await self.handle_command(command):
                    return
        except WebSocketDisconnect:
            pass
        finally:
            # Сначала отменяем HTTP-декодер, затем освобождаем слот следующей реплики.
            if self.generation is not None:
                self.generation.cancel()
                with suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await self.generation
            logger.debug("ASR-реплика закрыта: audio_bytes={}, final={}, elapsed_ms={:.1f}", len(self.buffer), self.final, (time.perf_counter() - started) * 1000)


@app.websocket("/v1/realtime")
async def recognize(socket: WebSocket) -> None:
    """Занимает единственный ASR-слот до await и освобождает при любом исходе.

    :param socket: Внутреннее WS-соединение API с единственным ASR runtime.
    """
    if app.state.active:
        logger.warning("ASR отклонил параллельную реплику: слот занят")
        await socket.close(code=1013)
        return
    app.state.active = True
    try:
        await ASRConnection(socket, app.state.client).run()
    finally:
        app.state.active = False
