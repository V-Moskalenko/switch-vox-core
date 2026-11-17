"""Проверяет различие liveness/readiness и опубликованный HTTP-контракт."""

import asyncio

import httpx
import orjson
from fastapi.testclient import TestClient

from api.dependencies import get_http_client
from api.fastapi_app import app
from services.session_service import GenerationContext


def test_live_ready_and_openapi() -> None:
    """Не объявляет модельный контур готовым по одному startup приложения."""
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert orjson.loads(response.content)["status"] == "not_ready"
        schema = orjson.loads(client.get("/openapi.json").content)
        operation = schema["paths"]["/health/ready"]["get"]
        assert "503" in operation["responses"]
        assert operation["summary"]
        for model in schema["components"]["schemas"].values():
            if model.get("title") in {"LivenessResponse", "ReadinessResponse", "ComponentReadiness"}:
                assert model["description"]
                for field in model["properties"].values():
                    assert field["description"]


def test_readiness_checks_actual_endpoints_and_pipeline() -> None:
    """Успех требует доступных моделей и подключённого обработчика, а не URL.

    Проверка использует реальный httpx request path с тестовым транспортом.
    """
    requested: list[str] = []
    tts_ready = True

    def respond(request: httpx.Request) -> httpx.Response:
        """Возвращает выбранный статус фактического запрошенного endpoint.

        :param request: HTTP-запрос readiness к отдельной модельной роли.
        :return: Ответ конкретного проверяемого runtime.
        """
        requested.append(str(request.url))
        return httpx.Response(200, json={"ready": tts_ready} if request.url.host == "tts" else {})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))

    def get_test_client() -> httpx.AsyncClient:
        """Предоставляет управляемый сетевой transport только в этом тесте.

        :return: HTTP-клиент с сохранением реальных путей запросов.
        """
        return http_client

    async def generate(context: GenerationContext, pcm: bytes) -> None:
        """Обозначает подключённый обработчик, не вызываемый readiness.

        :param context: Канал генерации.
        :param pcm: Входная реплика.
        """
        raise AssertionError("Readiness не должен вызывать модельную генерацию")

    app.dependency_overrides[get_http_client] = get_test_client
    try:
        with TestClient(app) as client:
            app.state.session_service.generation_handler = None
            assert client.get("/health/ready").status_code == 503
            app.state.session_service.generation_handler = generate
            ready = client.get("/health/ready")
            assert ready.status_code == 200
            assert orjson.loads(ready.content)["status"] == "ready"
            assert set(requested) == {"http://asr:8000/health", "http://llm:8000/health", "http://tts:8000/health"}
            tts_ready = False
            assert client.get("/health/ready").status_code == 503
    finally:
        app.dependency_overrides.pop(get_http_client)
        asyncio.run(http_client.aclose())
