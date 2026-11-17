"""WebSocket-маршрут: проверка Origin и передача сессии process-local владельцу."""

from fastapi import APIRouter, WebSocket
from loguru import logger

from api.dependencies import SessionDependency, SettingsDependency

router = APIRouter(prefix="/api/v1", tags=["Голосовая сессия"])


@router.websocket("/session", name="Голосовая сессия PCM")
async def voice_session(websocket: WebSocket, settings: SettingsDependency, service: SessionDependency) -> None:
    """Согласует один локальный браузерный диалог по отдельному JSON/PCM-протоколу.

    :param websocket: Подключение браузера с обязательным разрешённым Origin.
    :param settings: Настройки локального API и протокольных ограничений.
    :param service: Process-local владелец единственного слота сессии.
    """
    allowed_origins = settings.api.allowed_origins or [f"http://localhost:{settings.api.port}", f"http://127.0.0.1:{settings.api.port}"]
    if websocket.headers.get("origin") not in allowed_origins:
        # Проверка предшествует занятию единственного слота; заголовок не переносится в журнал.
        logger.bind(event="session_rejected", code="origin_not_allowed").warning("Отклонено WebSocket-подключение с недопустимым Origin")
        await websocket.close(code=1008)
        return
    await service.serve(websocket)
