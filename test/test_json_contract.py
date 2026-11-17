"""Проверяет UTF-8 JSON, текстовые WebSocket-кадры и отказ от невалидного JSON."""

import asyncio

import httpx
import orjson
import pytest
from fastapi.testclient import TestClient

from api.fastapi_app import app
from clients.asr_client import transcription_event
from clients.llm_client import LLMClient, content_event
from clients.tts_client import TTSClient
from settings import Settings


@pytest.mark.parametrize("invalid_number", ["NaN", "Infinity", "-Infinity"])
def test_model_events_reject_nonstandard_json_numbers(invalid_number: str) -> None:
    """Не принимает нестандартные числа даже в неиспользуемых полях ответа.

    :param invalid_number: Числовой литерал, запрещённый стандартом JSON.
    """
    with pytest.raises(ValueError):
        transcription_event('{"type":"transcription.done","text":"Привет","extra":' + invalid_number + "}")
    with pytest.raises(ValueError):
        content_event('data: {"choices":[{"delta":{"content":"Привет"}}],"extra":' + invalid_number + "}")


def test_model_events_preserve_russian_utf8_and_reject_invalid_bytes() -> None:
    """Сохраняет русский текст из UTF-8 и отклоняет повреждённые сообщения."""
    assert transcription_event('{"type":"transcription.done","text":"Привет, мир!"}'.encode()) == ("Привет, мир!", True)
    assert content_event('data: {"choices":[{"delta":{"content":"Привет, мир!"}}]}') == ("Привет, мир!", False)
    for malformed in (b'{"type":', b'{"text":"\xff"}'):
        with pytest.raises(ValueError):
            transcription_event(malformed)


def test_browser_errors_remain_text_frames_after_invalid_json() -> None:
    """Неверная JSON-команда возвращает текстовый JSON-кадр с русским объяснением."""
    with TestClient(app) as client, client.websocket_connect("/api/v1/session", headers={"origin": "http://127.0.0.1:8000"}) as socket:
        socket.send_text('{"type":')
        response = socket.receive()
        assert response["type"] == "websocket.send"
        assert "bytes" not in response
        assert "Неверная команда" in response["text"]
        socket.send_text('{"version":1,"type":"session.start","language":"ru","input_sample_rate":16000,"output_sample_rate":24000,"channels":1,"sample_width_bytes":2}')
        assert '"session.ready"' in socket.receive_text()


@pytest.mark.parametrize("role", ["tokenize", "llm", "tts"])
def test_model_http_requests_preserve_utf8_json_content_type(role: str) -> None:
    """Передаёт русский текст как UTF-8 JSON с корректным HTTP Content-Type.

    :param role: Подсчёт токенов, генерация текста либо синтез речи.
    """

    async def scenario() -> None:
        """Проверяет реальные запросы клиентов через управляемый HTTP-транспорт."""
        requested: list[str] = []

        def respond(request: httpx.Request) -> httpx.Response:
            """Проверяет JSON запроса и возвращает ответ нужного модельного протокола.

            :param request: Настоящий запрос к tokenizer, LLM либо TTS.
            :return: Валидный JSON, SSE или PCM для выбранного клиента.
            """
            if request.url.path == "/health":
                return httpx.Response(200, content=b'{"ready":true,"busy":false}', headers={"Content-Type": "application/json"})
            requested.append(request.url.path)
            assert request.headers["content-type"] == "application/json"
            assert "Русский текст".encode() in request.content
            payload = orjson.loads(request.content)
            if role == "tts":
                assert payload == {"text": "Русский текст"}
                headers = {"Content-Type": "audio/pcm", "X-Sample-Rate": "24000", "X-Channels": "1", "X-Sample-Width-Bytes": "2"}
                return httpx.Response(200, content=b"\x01\0", headers=headers)
            assert payload["messages"] == [{"role": "user", "content": "Русский текст"}]
            if role == "tokenize":
                return httpx.Response(200, content=b'{"count":10,"max_model_len":4096}', headers={"Content-Type": "application/json"})
            content = b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, content=content, headers={"Content-Type": "text/event-stream"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            settings = Settings(_env_file=None)
            messages = [{"role": "user", "content": "Русский текст"}]
            if role == "tokenize":
                assert await LLMClient(settings, client)._tokenize(messages) == (10, 4096)
                assert requested == ["/tokenize"]
            elif role == "llm":
                assert [part async for part in LLMClient(settings, client).stream_response(messages)] == ["OK"]
                assert requested == ["/v1/chat/completions"]
            else:
                assert [part async for part in TTSClient(settings, client).synthesize("Русский текст")] == [b"\x01\0"]
                assert requested == ["/synthesize"]

    asyncio.run(scenario())
