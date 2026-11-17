"""Клиент резидентного CosyVoice3: ожидание idle и точные PCM16 кадры из HTTP."""

import asyncio
from collections.abc import AsyncGenerator

import httpx
import orjson
from loguru import logger

from services.session_service import SessionError
from settings import Settings


class TTSClient:
    """Последовательно синтезирует короткие фразы без накопления целого WAV."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        """Сохраняет общий transport и адрес конкретного TTS.

        :param settings: Параметры PCM, endpoint и deadlines.
        :param client: HTTP-клиент, принадлежащий lifespan API.
        """
        self.settings = settings
        self.client = client
        self.base_url = str(settings.inference.tts_base_url).rstrip("/")

    async def _wait_idle(self) -> None:
        """Дожидается окончания предыдущего worker без повторного запуска синтеза.

        :raises SessionError: TTS не готов либо предыдущий worker не освободился.
        """
        deadline = asyncio.get_running_loop().time() + self.settings.inference.first_response_timeout_seconds
        waiting = False
        while True:
            response = await self.client.get(self.base_url + "/health", timeout=self.settings.inference.connect_timeout_seconds)
            response.raise_for_status()
            status = orjson.loads(response.content)
            if not isinstance(status, dict) or status.get("ready") is not True or type(status.get("busy")) is not bool:
                raise SessionError("tts_unavailable", "TTS не подтвердил готовность синтеза")
            if not status["busy"]:
                if waiting:
                    logger.bind(event="tts_idle", stage="tts").debug("Предыдущий синтез TTS освободил модель")
                return
            if not waiting:
                logger.bind(event="tts_waiting_idle", stage="tts").debug("Ожидание освобождения TTS после предыдущего синтеза")
                waiting = True
            if asyncio.get_running_loop().time() >= deadline:
                raise SessionError("tts_unavailable", "Предыдущий синтез TTS не завершился вовремя")
            await asyncio.sleep(0.02)

    async def synthesize(self, text: str) -> AsyncGenerator[bytes]:
        """Преобразует произвольные HTTP chunks в непустые чётные PCM16 кадры.

        :param text: Одна plain-text фраза не длиннее 500 символов.
        :yield: PCM16 LE mono 24 кГц, не более audio.max_frame_ms.
        :raises SessionError: Отказ/timeout TTS, неверный формат либо оборванный PCM.
        """
        inference = self.settings.inference
        timeout = httpx.Timeout(inference.first_response_timeout_seconds, connect=inference.connect_timeout_seconds)
        frame_bytes = 2 * self.settings.audio.output_sample_rate * min(self.settings.audio.frame_ms, self.settings.audio.max_frame_ms) // 1000
        started = asyncio.get_running_loop().time()
        log = logger.bind(stage="tts")
        log.bind(event="tts_started").debug("Запрошен поток синтеза речи")
        try:
            # Отмена HTTP не гарантирует мгновенной остановки CUDA; новый синтез ждёт idle модели.
            await self._wait_idle()
            async with self.client.stream("POST", self.base_url + "/synthesize", content=orjson.dumps({"text": text}), headers={"Content-Type": "application/json"}, timeout=timeout) as response:
                response.raise_for_status()
                expected = {"x-sample-rate": "24000", "x-channels": "1", "x-sample-width-bytes": "2"}
                if response.headers.get("content-type", "").split(";", maxsplit=1)[0].strip() != "audio/pcm" or any(response.headers.get(name) != value for name, value in expected.items()):
                    raise SessionError("tts_invalid_response", "TTS вернул неподдерживаемый формат аудио")
                pending = bytearray()
                samples = 0
                chunks = response.aiter_bytes()
                deadline = asyncio.get_running_loop().time() + inference.first_response_timeout_seconds
                while True:
                    try:
                        chunk = await asyncio.wait_for(anext(chunks), timeout=max(0, deadline - asyncio.get_running_loop().time()))
                    except StopAsyncIteration:
                        break
                    pending.extend(chunk)
                    # Границы HTTP chunks произвольны; PCM передаётся только целыми 16-битными отсчётами.
                    while len(pending) >= 2:
                        size = min(frame_bytes, len(pending) // 2 * 2)
                        frame = bytes(pending[:size])
                        del pending[:size]
                        samples += size // 2
                        yield frame
                        deadline = asyncio.get_running_loop().time() + inference.idle_timeout_seconds
                if pending or samples == 0:
                    raise SessionError("tts_invalid_response", "TTS вернул пустой или оборванный PCM")
                log.bind(event="tts_completed", samples=samples).debug("Поток синтеза речи завершён")
        except (httpx.HTTPError, TimeoutError) as exc:
            log.bind(event="tts_failed", error_type=type(exc).__name__).warning("TTS недоступен или превысил время ожидания")
            raise SessionError("tts_unavailable", "TTS недоступен или не выдал аудио вовремя") from exc
        except ValueError as exc:
            log.bind(event="tts_failed", error_type=type(exc).__name__).warning("TTS вернул некорректный JSON готовности")
            raise SessionError("tts_invalid_response", "TTS вернул неверный ответ готовности") from exc
        except SessionError as exc:
            log.bind(event="tts_failed", code=exc.code).warning("TTS не выполнил контракт синтеза речи")
            raise
        except asyncio.CancelledError:
            log.bind(event="tts_cancelled").debug("Поток синтеза речи отменён")
            raise
        finally:
            log.bind(event="tts_closed", elapsed_ms=round((asyncio.get_running_loop().time() - started) * 1000, 2)).debug("Поток TTS закрыт")
