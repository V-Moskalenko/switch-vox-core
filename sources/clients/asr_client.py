"""Клиент закреплённого ASR gateway: одна PCM-реплика, гипотезы и полный final."""

import asyncio
import base64
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from urllib.parse import urlsplit, urlunsplit

import orjson
from loguru import logger
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from services.session_service import SessionError
from settings import Settings


def transcription_event(raw: str | bytes) -> tuple[str, bool]:
    """Проверяет одно событие конкретного ASR gateway.

    :param raw: Полученный JSON text/binary WebSocket message.
    :return: Полная гипотеза либо final и признак завершения.
    :raises SessionError: Отказ runtime либо неверная структура события транскрипции.
    :raises ValueError: Сообщение не является корректным UTF-8 JSON.
    """
    event = orjson.loads(raw)
    if not isinstance(event, dict):
        raise SessionError("asr_invalid_response", "ASR вернул неверное событие")
    kind, text = event.get("type"), event.get("text")
    if kind == "error":
        raise SessionError("asr_unavailable", "ASR не завершил распознавание")
    if kind not in {"transcription.hypothesis", "transcription.done"} or not isinstance(text, str) or len(text) > 16384:
        raise SessionError("asr_invalid_response", "ASR вернул неверную транскрипцию")
    return text, kind == "transcription.done"


class ASRClient:
    """Открывает отдельный WebSocket для полной PTT-реплики или потока речи из VAD."""

    def __init__(self, settings: Settings) -> None:
        """Сохраняет адрес и deadlines без сетевого соединения.

        :param settings: Настройки формата PCM и конкретного ASR runtime.
        """
        self.settings = settings
        base = urlsplit(str(settings.inference.asr_base_url))
        self.url = urlunsplit(("wss" if base.scheme == "https" else "ws", base.netloc, base.path.rstrip("/") + "/v1/realtime", "", ""))

    async def _upload(self, socket: ClientConnection, chunks: AsyncIterator[bytes], input_done: asyncio.Event) -> None:
        """Передаёт ограниченные PCM-кадры и явный flush без ожидания гипотез.

        :param socket: Открытое соединение конкретной ASR-реплики.
        :param chunks: Живые PCM chunks; каждый может содержать несколько транспортных кадров.
        :param input_done: Сигнал настоящего EOF источника до отправки final commit.
        """
        try:
            async for pcm in chunks:
                if len(pcm) % 2:
                    raise SessionError("invalid_audio_frame", "Поток микрофона вернул незавершённый PCM16 отсчёт")
                for offset in range(0, len(pcm), 3200):
                    audio = base64.b64encode(pcm[offset : offset + 3200]).decode("ascii")
                    await asyncio.wait_for(socket.send(orjson.dumps({"type": "input_audio_buffer.append", "audio": audio}), text=True), timeout=self.settings.inference.idle_timeout_seconds)
            input_done.set()
            # Final разрешён только после EOF источника; flush не ждёт завершения распознавания.
            await asyncio.wait_for(socket.send(orjson.dumps({"type": "input_audio_buffer.commit", "final": True}), text=True), timeout=self.settings.inference.idle_timeout_seconds)
            logger.bind(event="asr_input_committed", stage="asr").debug("Входной поток ASR завершён и подтверждён")
        finally:
            if isinstance(chunks, AsyncGenerator):
                await chunks.aclose()

    async def _receive(self, socket: ClientConnection, sender: asyncio.Task[None], response_timeout_seconds: float) -> str | bytes:
        """Ждёт результат и одновременно наблюдает ошибку живого входа.

        Во время активного ввода deadline реплики контролирует owner сессии. ASR
        first/idle deadline начинает отсчёт после EOF либо окончания downstream pause.

        :param socket: Открытое ASR-соединение.
        :param sender: Единственная задача подачи PCM и final commit.
        :param response_timeout_seconds: Deadline результата после завершения ввода, секунды.
        :return: Следующее настоящее WS-сообщение.
        """
        receiver = asyncio.create_task(socket.recv(), name="asr-receive")
        try:
            if not sender.done():
                completed, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
                if sender in completed:
                    await sender
                if receiver in completed:
                    return receiver.result()
            else:
                await sender
            return await asyncio.wait_for(receiver, timeout=response_timeout_seconds)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)

    async def transcribe(self, pcm: bytes) -> AsyncGenerator[tuple[str, bool]]:
        """Отправляет PCM и закрывает upstream при отмене независимо от первого текста.

        :param pcm: Полная PTT-реплика PCM16 LE mono 16 кГц.
        :yield: Заменяемая гипотеза либо окончательный transcript и признак final.
        :raises SessionError: Отказ, неверный ответ или истёкший deadline ASR.
        """

        async def source() -> AsyncGenerator[bytes]:
            """Представляет завершённую PTT-реплику тем же протоколом живого ввода.

            :yield: Вся подтверждённая PCM-реплика без копирования.
            """
            yield pcm

        async with aclosing(self.transcribe_stream(source())) as events:
            async for event in events:
                yield event

    async def transcribe_stream(self, chunks: AsyncIterator[bytes]) -> AsyncGenerator[tuple[str, bool]]:
        """Принимает живой PCM и выдаёт гипотезы до окончания пользовательской реплики.

        :param chunks: Ограниченный владельцем сессии источник PCM16 LE mono 16 кГц.
        :yield: Заменяемые ASR-гипотезы и единственный final после настоящего EOF.
        :raises SessionError: Ошибка входа, неверный ответ, отказ или timeout ASR.
        """
        inference = self.settings.inference
        started = asyncio.get_running_loop().time()
        log = logger.bind(stage="asr")
        log.bind(event="asr_started").debug("Открывается поток распознавания речи")
        try:
            async with connect(self.url, proxy=None, open_timeout=inference.connect_timeout_seconds, close_timeout=self.settings.api.close_timeout_seconds, max_size=65536, max_queue=4) as socket:
                created = orjson.loads(await asyncio.wait_for(socket.recv(), timeout=inference.first_response_timeout_seconds))
                if not isinstance(created, dict) or created.get("type") != "session.created":
                    raise SessionError("asr_invalid_response", "ASR не подтвердил создание соединения")
                await socket.send(orjson.dumps({"type": "session.update", "model": "qwen3-asr"}), text=True)
                await socket.send(orjson.dumps({"type": "input_audio_buffer.commit"}), text=True)
                log.bind(event="asr_connected").debug("ASR подтвердил соединение и параметры распознавания")

                input_done = asyncio.Event()
                sender = asyncio.create_task(self._upload(socket, chunks, input_done), name="asr-upload")
                try:
                    timeout = inference.first_response_timeout_seconds
                    while True:
                        text, final = transcription_event(await self._receive(socket, sender, timeout))
                        if final:
                            if not input_done.is_set():
                                raise SessionError("asr_invalid_response", "ASR завершил реплику до окончания входного аудио")
                            await sender
                            log.bind(event="asr_completed").debug("Получен окончательный результат ASR")
                        yield text, final
                        if final:
                            return
                        timeout = inference.idle_timeout_seconds
                finally:
                    # Закрытие потребителем и отмена обязательно останавливают отправку микрофона.
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
        except (OSError, WebSocketException) as exc:
            log.bind(event="asr_failed", error_type=type(exc).__name__).warning("Соединение ASR не завершило распознавание")
            raise SessionError("asr_unavailable", "ASR недоступен или не ответил вовремя") from exc
        except (ValueError, TypeError) as exc:
            log.bind(event="asr_failed", error_type=type(exc).__name__).warning("ASR вернул некорректный JSON")
            raise SessionError("asr_invalid_response", "ASR вернул неверный JSON") from exc
        except SessionError as exc:
            log.bind(event="asr_failed", code=exc.code).warning("ASR не выполнил контракт распознавания")
            raise
        except asyncio.CancelledError:
            log.bind(event="asr_cancelled").debug("Поток ASR отменён")
            raise
        finally:
            log.bind(event="asr_closed", elapsed_ms=round((asyncio.get_running_loop().time() - started) * 1000, 2)).debug("Поток ASR закрыт")
