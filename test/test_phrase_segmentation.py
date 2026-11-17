"""Проверяет TTS-фразы из настоящего SSE независимо от границ текстовых delta."""

import asyncio
from collections.abc import AsyncIterator

import httpx
import orjson
import pytest

from services.cascade_service import CascadeService
from settings import Settings


class PhraseStream(httpx.AsyncByteStream):
    """Передаёт текстовые delta и при необходимости удерживает штатный SSE EOF."""

    def __init__(self, deltas: list[str], *, release: asyncio.Event | None = None) -> None:
        """Сохраняет входные delta без подмены LLM parser.

        :param deltas: Последовательные части ответа модели.
        :param release: Разрешение отправить [DONE] после последней delta.
        """
        self.deltas = deltas
        self.release = release
        self.waiting = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Кодирует content как SSE и отделяет конец текста от конца HTTP body.

        :yield: Настоящее событие chat completion.
        """
        for delta in self.deltas:
            payload = {"choices": [{"delta": {"content": delta}}]}
            yield b"data: " + orjson.dumps(payload) + b"\n\n"
        self.waiting.set()
        if self.release is not None:
            await self.release.wait()
        yield b"data: [DONE]\n\n"


async def produce_phrases(deltas: list[str], maximum: int = 200) -> list[str]:
    """Собирает готовые фразы реального producer вместе с проверкой EOF-маркера.

    :param deltas: Части текста в отдельных SSE-событиях.
    :param maximum: Ограничение длины одного запроса к TTS.
    :return: Фразы в исходном порядке без завершающего маркера.
    """
    stream = PhraseStream(deltas)

    def respond(request: httpx.Request) -> httpx.Response:
        """Предоставляет только HTTP transport настоящему LLM-клиенту.

        :param request: Запрос chat completion.
        :return: Поток с контролируемыми границами content.
        """
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)

    settings = Settings(_env_file=None)
    settings.inference.max_phrase_characters = maximum
    phrases: asyncio.Queue[str | None] = asyncio.Queue()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        await CascadeService(settings, client)._produce(phrases, [])
    result: list[str] = []
    while not phrases.empty():
        phrase = phrases.get_nowait()
        if phrase is None:
            assert phrases.empty()
            return result
        result.append(phrase)
    pytest.fail("Producer не отправил завершающий маркер")


NUMBER_TEXT = "Число 3.14, дата 15.11.2026, версия v1.2.3, сайт example.ru. Готово! Хвост"


@pytest.mark.parametrize("split", range(len(NUMBER_TEXT) + 1))
def test_numeric_and_domain_tokens_survive_every_delta_split(split: int) -> None:
    """Любая граница между двумя текстовыми частями сохраняет числа и целые предложения.

    :param split: Позиция произвольной границы между двумя content delta.
    """
    phrases = asyncio.run(produce_phrases([NUMBER_TEXT[:split], NUMBER_TEXT[split:]]))
    assert phrases == ["Число 3.14, дата 15.11.2026, версия v1.2.3, сайт example.ru.", "Готово!", "Хвост"]


def test_character_deltas_preserve_tokens_and_flush_tail_once() -> None:
    """Даже точка отдельной delta не обрывает число, дату, версию или домен."""
    phrases = asyncio.run(produce_phrases(list(NUMBER_TEXT)))
    assert phrases == ["Число 3.14, дата 15.11.2026, версия v1.2.3, сайт example.ru.", "Готово!", "Хвост"]


@pytest.mark.parametrize("text,expected", [("Подождите... Продолжение.", ["Подождите...", "Продолжение."]), ("Число 3.14", ["Число 3.14"])])
def test_punctuation_at_delta_end_waits_for_continuation(text: str, expected: list[str]) -> None:
    """Многоточие и незавершённый хвост остаются целыми до подтверждения границы.

    :param text: Текст с неоднозначной точкой в конце отдельных delta.
    :param expected: Целые фразы после штатного завершения модели.
    """
    assert asyncio.run(produce_phrases(list(text))) == expected


def test_ordinary_sentence_reaches_queue_before_llm_eof() -> None:
    """Пробел подтверждает короткую фразу, пока [DONE] ещё удерживается."""

    async def scenario() -> None:
        """Наблюдает готовую очередь до разрешения окончания upstream."""
        release = asyncio.Event()
        stream = PhraseStream(["Первая фраза.", " ", "Хвост"], release=release)

        def respond(request: httpx.Request) -> httpx.Response:
            """Удерживает только окончание SSE, оставляя producer работающим.

            :param request: Запрос к LLM runtime.
            :return: Поток с явным разрешением [DONE].
            """
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)

        phrases: asyncio.Queue[str | None] = asyncio.Queue(maxsize=2)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            task = asyncio.create_task(CascadeService(Settings(_env_file=None), client)._produce(phrases, []))
            try:
                await asyncio.wait_for(stream.waiting.wait(), timeout=1)
                assert not task.done()
                assert phrases.get_nowait() == "Первая фраза."
                assert phrases.empty()
            finally:
                release.set()
                await asyncio.wait_for(task, timeout=1)
            assert phrases.get_nowait() == "Хвост"
            assert phrases.get_nowait() is None
            assert phrases.empty()

    asyncio.run(scenario())


@pytest.mark.parametrize("maximum", [200, 500])
@pytest.mark.parametrize("text", ["слово " * 250 + "хвост", "я" * 1257])
def test_hard_phrase_limit_preserves_all_text(maximum: int, text: str) -> None:
    """Длинный ответ ограничен даже без пунктуации и пробелов, без потери букв.

    :param maximum: Поддерживаемый предел длины одного TTS-запроса.
    :param text: Длинный текст с пробелами либо одно длинное слово.
    """
    phrases = asyncio.run(produce_phrases([text], maximum=maximum))
    assert all(0 < len(phrase) <= maximum for phrase in phrases)
    assert len(phrases) > 1
    assert (" ".join(phrases) if " " in text else "".join(phrases)) == text
