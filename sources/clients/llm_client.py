"""Прямой vLLM-клиент: точный token budget и поток только озвучиваемого content."""

import asyncio
from collections.abc import AsyncGenerator

import httpx
import orjson
from loguru import logger

from services.session_service import DialogueMessage, SessionError
from settings import Settings

SYSTEM_PROMPT = "Отвечай кратко на русском языке обычным текстом. Не используй Markdown, служебную разметку и рассуждения."


def content_event(line: str) -> tuple[str, bool]:
    """Проверяет одно SSE-событие и извлекает только plain-text content.

    :param line: Строка data либо допустимый keepalive comment.
    :return: Текстовая часть и признак штатного [DONE].
    :raises SessionError: Неверный JSON-контракт, слишком большое событие или markup.
    :raises ValueError: Содержимое data не является корректным JSON.
    """
    if len(line) > 65536:
        raise SessionError("llm_invalid_response", "LLM вернула слишком большое событие")
    if not line or line.startswith(":"):
        return "", False
    if not line.startswith("data: "):
        raise SessionError("llm_invalid_response", "LLM вернула неверное SSE-событие")
    if line == "data: [DONE]":
        return "", True
    event = orjson.loads(line[6:])
    choices = event.get("choices") if isinstance(event, dict) else None
    if not isinstance(choices, list):
        raise SessionError("llm_invalid_response", "LLM вернула неверный набор вариантов ответа")
    if not choices:
        return "", False
    choice = choices[0]
    delta = choice.get("delta") if isinstance(choice, dict) else None
    if not isinstance(delta, dict):
        raise SessionError("llm_invalid_response", "LLM вернула неверную текстовую часть")
    text = delta.get("content", "")
    if text is None:
        return "", False
    if not isinstance(text, str):
        raise SessionError("llm_invalid_response", "LLM вернула неверный текст ответа")
    # Останавливаем ответ до передачи разметки TTS, включая разорванные tags.
    if any(marker in text for marker in ("<", ">", "`")):
        raise SessionError("llm_invalid_response", "LLM вернула служебную разметку вместо обычного текста")
    return text, False


class LLMClient:
    """Использует tokenizer и chat completions одного закреплённого Qwen runtime."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        """Сохраняет общий HTTP transport и параметры конкретной модели.

        :param settings: Адрес vLLM и ограничения ответа.
        :param client: HTTP-клиент lifespan с явным закрытием response streams.
        """
        self.settings = settings
        self.client = client
        self.base_url = str(settings.inference.llm_base_url).rstrip("/")

    async def prepare_messages(self, history: tuple[DialogueMessage, ...], user_text: str) -> list[dict[str, str]]:
        """Удаляет старые целые turns по реальному tokenizer, сохраняя текущий вопрос.

        :param history: Неизменяемая история подтверждённых и услышанных сообщений.
        :param user_text: Окончательная непустая транскрипция новой реплики.
        :return: Сообщения с запасом для max_response_tokens в реальном context window.
        :raises SessionError: Текущая реплика слишком длинна или tokenizer недоступен.
        """
        previous = [{"role": message.role, "content": message.content} for message in history]
        while True:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *previous, {"role": "user", "content": user_text}]
            count, maximum = await self._tokenize(messages)
            if count + self.settings.inference.max_response_tokens <= maximum:
                logger.bind(event="llm_context_ready", stage="llm", prompt_tokens=count, context_tokens=maximum, history_messages=len(previous)).debug("Контекст LLM укладывается в доступное окно")
                return messages
            if not previous:
                raise SessionError("context_too_long", "Реплика слишком длинна для контекста модели")
            # Удаляем целые прошлые turns, сохраняя system prompt и текущий вопрос без обрезки.
            previous.pop(0)
            while previous and previous[0]["role"] != "user":
                previous.pop(0)
            logger.bind(event="llm_history_pruned", stage="llm", history_messages=len(previous)).debug("Сокращена история для контекстного окна LLM")

    async def _tokenize(self, messages: list[dict[str, str]]) -> tuple[int, int]:
        """Проверяет размер готового chat prompt тем же tokenizer, что и генерация.

        :param messages: Полная кандидатная история вместе с system/current user.
        :return: Точное число токенов prompt и окно контекста runtime.
        :raises SessionError: Отказ tokenizer или неверный счётчик.
        """
        log = logger.bind(stage="llm", operation="tokenize")
        log.bind(event="llm_tokenize_started").debug("Запрошен точный размер контекста LLM")
        try:
            response = await self.client.post(
                self.base_url.removesuffix("/v1") + "/tokenize",
                content=orjson.dumps({"model": "qwen3-4b", "messages": messages, "add_generation_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}),
                headers={"Content-Type": "application/json"},
                timeout=httpx.Timeout(self.settings.inference.first_response_timeout_seconds, connect=self.settings.inference.connect_timeout_seconds),
            )
            response.raise_for_status()
            result = orjson.loads(response.content)
            count, maximum = (result.get("count"), result.get("max_model_len")) if isinstance(result, dict) else (None, None)
            if type(count) is not int or type(maximum) is not int or count < 0 or maximum <= 0:
                raise SessionError("llm_invalid_response", "LLM не вернула корректный размер контекста")
            return count, maximum
        except httpx.HTTPError as exc:
            log.bind(event="llm_tokenize_failed", error_type=type(exc).__name__).warning("Не удалось получить размер контекста LLM")
            raise SessionError("llm_unavailable", "Не удалось проверить контекст LLM") from exc
        except ValueError as exc:
            log.bind(event="llm_tokenize_failed", error_type=type(exc).__name__).warning("Tokenizer LLM вернул некорректный JSON")
            raise SessionError("llm_invalid_response", "LLM вернула неверный ответ tokenizer") from exc
        except SessionError as exc:
            log.bind(event="llm_tokenize_failed", code=exc.code).warning("Tokenizer LLM нарушил контракт подсчёта")
            raise

    async def stream_response(self, messages: list[dict[str, str]]) -> AsyncGenerator[str]:
        """Читает content SSE с отдельными deadlines первого текста и продолжения.

        :param messages: Сообщения, уже проверенные тем же tokenizer и chat template.
        :yield: Части обычного текста, без reasoning и tool payload.
        :raises SessionError: Неверный SSE, преждевременный EOF, markup или timeout.
        """
        inference = self.settings.inference
        payload = {"model": "qwen3-4b", "messages": messages, "stream": True, "max_tokens": inference.max_response_tokens, "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False}}
        timeout = httpx.Timeout(inference.first_response_timeout_seconds, connect=inference.connect_timeout_seconds)
        generated = 0
        started = asyncio.get_running_loop().time()
        log = logger.bind(stage="llm")
        log.bind(event="llm_started").debug("Открывается поток текстового ответа LLM")
        deadline = asyncio.get_running_loop().time() + inference.first_response_timeout_seconds
        try:
            async with self.client.stream("POST", self.base_url + "/chat/completions", content=orjson.dumps(payload), headers={"Content-Type": "application/json"}, timeout=timeout) as response:
                response.raise_for_status()
                if response.headers.get("content-type", "").split(";", maxsplit=1)[0].strip() != "text/event-stream":
                    raise SessionError("llm_invalid_response", "LLM вернула неверный формат потока")
                lines = response.aiter_lines()
                while True:
                    try:
                        line = await asyncio.wait_for(anext(lines), timeout=max(0, deadline - asyncio.get_running_loop().time()))
                    except StopAsyncIteration:
                        raise SessionError("llm_invalid_response", "LLM оборвала поток до завершения ответа") from None
                    text, done = content_event(line)
                    if done:
                        break
                    if text:
                        generated += len(text)
                        if generated > self.settings.api.max_response_segments * inference.max_phrase_characters:
                            raise SessionError("response_too_long", "Превышена допустимая длина ответа")
                        yield text
                        # Медленное чтение потребителем не входит в ожидание следующей части upstream.
                        deadline = asyncio.get_running_loop().time() + inference.idle_timeout_seconds
                if generated == 0:
                    raise SessionError("llm_empty_response", "LLM не вернула текста ответа")
                log.bind(event="llm_completed", generated_characters=generated).debug("Текстовый ответ LLM завершён")
        except (httpx.HTTPError, TimeoutError) as exc:
            log.bind(event="llm_failed", error_type=type(exc).__name__).warning("Поток LLM недоступен или превысил время ожидания")
            raise SessionError("llm_unavailable", "LLM недоступна или не ответила вовремя") from exc
        except (ValueError, TypeError) as exc:
            log.bind(event="llm_failed", error_type=type(exc).__name__).warning("Поток LLM содержит некорректный JSON")
            raise SessionError("llm_invalid_response", "LLM вернула неверный JSON потока") from exc
        except SessionError as exc:
            log.bind(event="llm_failed", code=exc.code).warning("LLM не выполнила контракт текстового ответа")
            raise
        except asyncio.CancelledError:
            log.bind(event="llm_cancelled").debug("Поток ответа LLM отменён")
            raise
        finally:
            log.bind(event="llm_closed", elapsed_ms=round((asyncio.get_running_loop().time() - started) * 1000, 2)).debug("Поток ответа LLM закрыт")
