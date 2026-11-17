"""Каскад ASR → поток LLM → ограниченная очередь фраз → поток TTS."""

import asyncio
from contextlib import aclosing
from uuid import uuid4

import httpx
from loguru import logger

from clients.asr_client import ASRClient
from clients.llm_client import LLMClient
from clients.tts_client import TTSClient
from services.session_service import GenerationContext, SessionError
from settings import Settings


class _GenerationStopped(Exception):
    """Владелец сессии больше не принимает события данной генерации."""


def phrase_end(text: str, maximum: int) -> int:
    """Находит границу короткой фразы без потери пробелов и незавершённого хвоста.

    :param text: Накопленный content потока LLM.
    :param maximum: Жёсткий предел символов одной TTS-фразы.
    :return: Позиция среза либо ноль, когда нужно продолжение текста.
    """
    for index, character in enumerate(text[:maximum]):
        if character in "!?\n":
            return index + 1
        # Следующая delta может продолжить число, дату, домен или многоточие.
        if character == "." and index + 1 < len(text) and text[index + 1].isspace():
            return index + 1
    if len(text) < maximum:
        return 0
    whitespace = text.rfind(" ", 0, maximum + 1)
    return maximum if whitespace <= 0 else whitespace


class CascadeService:
    """Распознаёт реплику из буфера или живого потока и озвучивает ответ LLM."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        """Создаёт сетевые клиенты; сами модели остаются в отдельных сервисах.

        :param settings: Адреса моделей, тайм-ауты и ограничения очередей.
        :param client: Общий HTTP-клиент, открытый на время работы приложения.
        """
        self.settings = settings
        self.asr = ASRClient(settings)
        self.llm = LLMClient(settings, client)
        self.tts = TTSClient(settings, client)

    async def generate(self, context: GenerationContext, pcm: bytes) -> None:
        """Передаёт итоговую транскрипцию в LLM и озвучивает ответ по мере генерации.

        :param context: Канал результатов со снимком истории и необязательным живым входом.
        :param pcm: Буфер PTT в PCM16 LE mono 16 кГц; при context.input_audio не используется.
        :raises SessionError: Конкретный отказ ASR/LLM/TTS либо ограничение ответа.
        """
        transcript = ""
        # Автоматическая реплика поступает в ASR до своей конечной паузы; PTT уже содержит полный буфер.
        source = self.asr.transcribe_stream(context.input_audio) if context.input_audio is not None else self.asr.transcribe(pcm)
        async with aclosing(source) as events:
            async for text, final in events:
                if not await context.emit_transcript(text, final=final):
                    return
                if final:
                    transcript = text.strip()
        if not transcript:
            logger.bind(event="cascade_empty_transcript").debug("Каскад завершён без запуска LLM: итоговая транскрипция пуста")
            return
        messages = await self.llm.prepare_messages(context.history, transcript)
        phrases: asyncio.Queue[str | None] = asyncio.Queue(maxsize=self.settings.inference.phrase_queue_size)
        failure: SessionError | None = None
        try:
            # TaskGroup отменяет соседнюю стадию при отказе; ограниченная очередь не копит весь ответ.
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(self._produce(phrases, messages), name="cascade-text")
                tasks.create_task(self._consume(phrases, context), name="cascade-audio")
        except* _GenerationStopped:
            logger.bind(event="cascade_stopped").debug("Владелец остановил публикацию ответа; связанные модельные потоки закрыты")
        except* SessionError as errors:
            error = errors.exceptions[0]
            if isinstance(error, SessionError):
                failure = error
        if failure is not None:
            raise failure

    async def _produce(self, phrases: asyncio.Queue[str | None], messages: list[dict[str, str]]) -> None:
        """Разбивает текст LLM на короткие фразы и ждёт освобождения очереди TTS.

        :param phrases: Ограниченная очередь последовательного синтеза речи.
        :param messages: Сообщения LLM, проверенные на соответствие размеру контекста.
        """
        pending = ""
        maximum = self.settings.inference.max_phrase_characters
        async with aclosing(self.llm.stream_response(messages)) as tokens:
            async for delta in tokens:
                pending += delta
                while (end := phrase_end(pending, maximum)) > 0:
                    phrase, pending = pending[:end].strip(), pending[end:]
                    if phrase:
                        await phrases.put(phrase)
        if pending.strip():
            # Остаток без конечной пунктуации озвучивается один раз после штатного завершения SSE.
            await phrases.put(pending.strip())
        await phrases.put(None)

    async def _consume(self, phrases: asyncio.Queue[str | None], context: GenerationContext) -> None:
        """Последовательно озвучивает фразы и закрывает поток TTS при любом выходе.

        :param phrases: Ограниченный поток фраз и финальный маркер None.
        :param context: Канал публикации только текущей генерации.
        """
        while (phrase := await phrases.get()) is not None:
            segment_id = uuid4()
            if not await context.emit_text(segment_id, phrase):
                raise _GenerationStopped
            # ID связывает начало и закрытие синтеза фразы; её содержимое остаётся только в протоколе.
            with logger.contextualize(segment_id=str(segment_id)):
                async with aclosing(self.tts.synthesize(phrase)) as audio:
                    async for chunk in audio:
                        if not await context.emit_audio(segment_id, chunk):
                            raise _GenerationStopped
            if not await context.finish_segment(segment_id):
                raise _GenerationStopped
