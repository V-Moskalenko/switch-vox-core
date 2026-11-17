"""HTTP-сервис CosyVoice3 с потоковой передачей PCM и отменой синтеза."""

import asyncio
import faulthandler
import os
import queue
import signal
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse, StreamingResponse
from loguru import logger
from schemas import HealthResponse, SynthesisRequest
from starlette.types import Receive, Scope, Send
from streaming import AudioStream


class PCMResponse(StreamingResponse):
    """Вызывает очистку синтеза при завершении ответа, включая разрыв до первого аудиокадра."""

    def __init__(self, content: AsyncIterator[bytes], cleanup: Callable[[], Awaitable[None]]) -> None:
        """Привязывает очистку worker к ASGI response, а не только генератору.

        :param content: Последовательность реальных PCM chunks.
        :param cleanup: Остановка worker и освобождение единственного slot.
        """
        self._cleanup = cleanup
        super().__init__(content, media_type="audio/pcm", headers={"X-Sample-Rate": "24000", "X-Channels": "1", "X-Sample-Width-Bytes": "2"})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Закрывает worker при любом завершении ASGI-send.

        :param scope: Параметры HTTP-соединения.
        :param receive: Получение событий disconnect.
        :param send: Отправка headers и PCM.
        """
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._cleanup()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Загружает модель и признаки выбранного голоса в отдельном рабочем потоке.

    :param application: Единственный экземпляр HTTP runtime.
    :yield: Управление серверу после реальной загрузки модели.
    """
    # Очередь журнала освобождает модельный worker и event loop от записи в stderr.
    await logger.complete()
    logger.remove()
    sink = logger.add(
        sys.stderr, level=os.getenv("RUNTIME_LOG_LEVEL", "INFO"), enqueue=True, diagnose=False, backtrace=False, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message} | {extra}"
    )
    started = time.perf_counter()
    try:
        # Тяжёлые зависимости и загрузка весов принадлежат только модельному контейнеру.
        from backend import REFERENCE_PREFIX, REFERENCE_TEXT, CosyVoiceEngine

        reference_wav = os.getenv("TTS_REFERENCE_WAV", "").strip()
        reference_text = os.getenv("TTS_REFERENCE_TEXT", "").strip()
        if bool(reference_wav) != bool(reference_text):
            raise ValueError("Нужно одновременно задать WAV и transcript референса TTS")
        reference = Path(reference_wav) if reference_wav else Path("/opt/CosyVoice/asset/zero_shot_prompt.wav")
        prompt = REFERENCE_PREFIX + reference_text if reference_text else REFERENCE_TEXT
        logger.info("Загрузка TTS-модели и подготовка образца голоса")
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1)
        application.state.engine = await asyncio.to_thread(CosyVoiceEngine, model_dir=Path(os.getenv("TTS_MODEL_DIR", "/models/tts")), reference=reference, reference_text=prompt)
        application.state.busy = False
        logger.info("TTS готов к синтезу: load_ms={:.1f}", (time.perf_counter() - started) * 1000)
        yield
    except Exception as exc:
        logger.error("Сбой жизненного цикла TTS: type={}", type(exc).__name__)
        raise
    finally:
        logger.info("TTS остановлен")
        await logger.complete()
        logger.remove(sink)


app = FastAPI(title="CosyVoice3 internal runtime", lifespan=lifespan, default_response_class=ORJSONResponse)


async def _pcm_chunks(stream: AudioStream, request: Request) -> AsyncIterator[bytes]:
    """Отдаёт PCM с отдельными сроками первого аудио и простоя между частями.

    :param stream: Ограниченная очередь реального synthesis worker.
    :param request: HTTP-соединение для обнаружения disconnect.
    :yield: Готовый PCM без промежуточного WAV-файла.
    :raises TimeoutError: Синтез не выдал первый либо очередной аудиокадр вовремя.
    """
    started = time.perf_counter()
    deadline = asyncio.get_running_loop().time() + 60
    pcm_bytes = 0
    completed = False
    try:
        while not await request.is_disconnected():
            try:
                chunk = stream.read()
            except queue.Empty:
                if asyncio.get_running_loop().time() > deadline:
                    raise TimeoutError("TTS не выдал аудио в отведённое время") from None
                await asyncio.sleep(0.01)
                continue
            if chunk is None:
                completed = True
                return
            if pcm_bytes == 0:
                logger.debug("TTS выдал первый PCM: elapsed_ms={:.1f}", (time.perf_counter() - started) * 1000)
            pcm_bytes += len(chunk)
            deadline = asyncio.get_running_loop().time() + 15
            yield chunk
    except Exception as exc:
        logger.error("Сбой потока TTS: type={}, pcm_bytes={}", type(exc).__name__, pcm_bytes)
        raise
    finally:
        stream.cancel.set()
        logger.debug("Поток TTS закрыт: completed={}, pcm_bytes={}, elapsed_ms={:.1f}", completed, pcm_bytes, (time.perf_counter() - started) * 1000)


@app.get("/health", summary="Готовность резидентного CosyVoice3", response_model=HealthResponse, response_description="Модель загружена", responses={503: {"description": "Модель не готова"}})
async def health() -> HealthResponse:
    """Возвращает готовность после загрузки реальных weights и reference.

    :return: Состояние резидентной модели и текущего синтеза.
    :raises HTTPException: HTTP 503, если модель ещё не загружена.
    """
    if not hasattr(app.state, "engine"):
        raise HTTPException(status_code=503, detail="TTS-модель не готова")
    progress = getattr(app.state.engine.model, "progress", {})
    return HealthResponse(ready=True, busy=app.state.busy, progress=progress)


@app.post(
    "/synthesize",
    summary="Потоковый PCM16 LE mono 24 кГц",
    response_class=StreamingResponse,
    response_description="Поток аудиокадров PCM16 LE mono 24000 Гц",
    responses={
        200: {"content": {"audio/pcm": {"schema": {"type": "string", "format": "binary"}}}},
        409: {"description": "Другой запрос TTS ещё выполняется"},
        503: {"description": "Резидентная TTS-модель ещё не готова"},
    },
)
async def synthesize(body: SynthesisRequest, request: Request) -> StreamingResponse:
    """Синтезирует одну фразу, отказывая второму запросу до model calls.

    :param body: Текст короткой фразы.
    :param request: HTTP-соединение для обнаружения disconnect.
    :return: HTTP-ответ с потоком PCM и освобождением слота синтеза при закрытии.
    :raises HTTPException: HTTP 503 до загрузки модели либо HTTP 409 при занятом worker.
    """
    if not hasattr(app.state, "engine"):
        raise HTTPException(status_code=503, detail="TTS-модель не готова")
    if app.state.busy:
        logger.warning("TTS отклонил параллельный синтез: worker занят")
        raise HTTPException(status_code=409, detail="TTS уже выполняет другой запрос")
    app.state.busy = True
    logger.debug("Начат синтез TTS: text_characters={}", len(body.text))
    stream = AudioStream(lambda cancel: app.state.engine.synthesize(body.text, cancel))
    stream.start()

    async def cleanup() -> None:
        """Завершает worker под shield даже при disconnect до запуска generator."""
        with anyio.CancelScope(shield=True):
            try:
                await asyncio.to_thread(stream.close, 3)
            except TimeoutError:
                logger.error("TTS-worker не остановился: завершаем runtime без оставшихся CUDA-задач")
                await logger.complete()
                os._exit(70)
            finally:
                app.state.busy = False
                logger.debug("Освобождён слот синтеза TTS")

    return PCMResponse(_pcm_chunks(stream, request), cleanup)
