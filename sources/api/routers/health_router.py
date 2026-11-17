"""Liveness процесса и readiness реальных endpoint выбранного голосового режима."""

import asyncio
from typing import Literal

import httpx
import orjson
from fastapi import APIRouter, Response

from api.dependencies import HttpClientDependency, SessionDependency, SettingsDependency
from api.schemas.health_schemas import ComponentReadiness, LivenessResponse, ReadinessResponse

router = APIRouter(prefix="/health", tags=["Состояние сервиса"])


async def _check_endpoint(client: httpx.AsyncClient, name: Literal["asr", "llm", "tts"], url: str) -> ComponentReadiness:
    """Проверяет фактический ответ endpoint, не подменяя готовность конфигурацией URL.

    :param client: Общий ограниченный по времени HTTP-клиент.
    :param name: Проверяемая модельная роль.
    :param url: Полный URL health-check, выбранный по реальному API модели.
    :return: Результат текущего запроса без технических подробностей в публичном ответе.
    """
    try:
        response = await client.get(url)
        if response.status_code != 200:
            return ComponentReadiness(name=name, ready=False, reason="endpoint_not_ready")
        if name == "tts":
            try:
                status = orjson.loads(response.content)
            except ValueError:
                return ComponentReadiness(name=name, ready=False, reason="invalid_health_response")
            if not isinstance(status, dict) or status.get("ready") is not True:
                return ComponentReadiness(name=name, ready=False, reason="endpoint_not_ready")
        return ComponentReadiness(name=name, ready=True)
    except httpx.HTTPError:
        return ComponentReadiness(name=name, ready=False, reason="endpoint_unavailable")


@router.get(
    "/live",
    summary="Проверить доступность процесса",
    description="Проверяет только API-процесс и не обращается к моделям.",
    response_description="Процесс API принимает HTTP-запросы",
    response_model=LivenessResponse,
    status_code=200,
    responses={200: {"description": "Процесс работает"}},
)
async def health_live() -> LivenessResponse:
    """Возвращает liveness без сетевых запросов к модельному контуру.

    :return: Доступность текущего API-процесса.
    """
    return LivenessResponse()


@router.get(
    "/ready",
    summary="Проверить готовность голосового режима",
    description="Проверяет подключение обработчика и health реальных моделей выбранного режима; до готовности возвращает 503.",
    response_description="Готовность режима и проверенных компонент",
    response_model=ReadinessResponse,
    status_code=200,
    responses={200: {"description": "Выбранный голосовой контур готов"}, 503: {"model": ReadinessResponse, "description": "Обработчик или модели ещё не готовы"}},
)
async def health_ready(response: Response, settings: SettingsDependency, service: SessionDependency, client: HttpClientDependency) -> ReadinessResponse:
    """Проверяет только выбранный режим и не загружает моделей.

    :param response: Ответ, которому устанавливается 503 до готовности.
    :param settings: Адреса и зафиксированный режим процесса.
    :param service: Владелец сессии с реальным обработчиком генерации.
    :param client: HTTP-клиент lifespan с ограниченными ожиданиями.
    :return: Проверенное состояние выбранного модельного контура.
    """
    components = [ComponentReadiness(name="pipeline", ready=service.generation_handler is not None, reason=None if service.generation_handler is not None else "pipeline_not_configured")]
    if settings.inference.mode == "e2e":
        components.append(ComponentReadiness(name="e2e", ready=False, reason="backend_not_validated"))
    else:
        asr_health = f"{str(settings.inference.asr_base_url).rstrip('/')}/health"
        llm_base = str(settings.inference.llm_base_url).rstrip("/").removesuffix("/v1")
        llm_health = f"{llm_base}/health"
        tts_health = f"{str(settings.inference.tts_base_url).rstrip('/')}/health"
        components.extend(await asyncio.gather(_check_endpoint(client, "asr", asr_health), _check_endpoint(client, "llm", llm_health), _check_endpoint(client, "tts", tts_health)))
    ready = all(component.ready for component in components)
    response.status_code = 200 if ready else 503
    return ReadinessResponse(status="ready" if ready else "not_ready", mode=settings.inference.mode, components=components)
