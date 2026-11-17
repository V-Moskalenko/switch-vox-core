"""Типизированные getters process-local ресурсов для HTTP и WebSocket через HTTPConnection."""

from typing import Annotated, cast

import httpx
from fastapi import Depends
from starlette.requests import HTTPConnection

from services.session_service import SessionService
from settings import Settings


def get_settings(connection: HTTPConnection) -> Settings:
    """Возвращает проверенные настройки процесса.

    :param connection: HTTP-запрос либо WebSocket текущего приложения.
    :return: Настройки, установленные в lifespan.
    """
    return cast(Settings, connection.app.state.settings)


def get_session_service(connection: HTTPConnection) -> SessionService:
    """Возвращает единственного владельца голосового слота.

    :param connection: HTTP-запрос либо WebSocket текущего приложения.
    :return: Сервис сессии данного процесса.
    """
    return cast(SessionService, connection.app.state.session_service)


def get_http_client(connection: HTTPConnection) -> httpx.AsyncClient:
    """Возвращает общий HTTP-клиент генерации и проверки готовности моделей.

    :param connection: HTTP-запрос либо WebSocket текущего приложения.
    :return: Общий клиент без proxy из внешнего окружения.
    """
    return cast(httpx.AsyncClient, connection.app.state.http_client)


type SettingsDependency = Annotated[Settings, Depends(get_settings)]
type SessionDependency = Annotated[SessionService, Depends(get_session_service)]
type HttpClientDependency = Annotated[httpx.AsyncClient, Depends(get_http_client)]
