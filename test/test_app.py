"""Проверяет запуск API и подключение каскада без загрузки моделей."""

import orjson
from fastapi.testclient import TestClient

from api.fastapi_app import app
from settings import config


def test_app_startup_and_openapi() -> None:
    """Проверяет штатный startup и публикацию схемы на Python проекта."""
    with TestClient(app) as client:
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert orjson.loads(response.content)["info"]["title"] == "Switch Vox Core"
        assert app.state.settings.api.workers == 1


def test_only_validated_cascade_has_generation_handler() -> None:
    """Startup подключает каскад; неподтверждённый E2E не получает обработчик."""
    with TestClient(app):
        assert app.state.session_service.generation_handler is not None
    original = config.inference.mode
    try:
        config.inference.mode = "e2e"
        with TestClient(app) as client:
            assert app.state.session_service.generation_handler is None
            assert client.get("/health/ready").status_code == 503
    finally:
        config.inference.mode = original
