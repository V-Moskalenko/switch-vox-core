"""Раздаёт страницу голосового клиента с ручным и автоматическим режимами ввода."""

from fastapi import APIRouter
from fastapi.responses import FileResponse

from settings import PROJECT_ROOT

router = APIRouter(tags=["Браузерный клиент"])


@router.get("/", summary="Открыть голосовой клиент", response_description="HTML-страница локального голосового диалога", response_class=FileResponse, include_in_schema=False)
async def browser_client() -> FileResponse:
    """Возвращает страницу записи микрофона, выбора режима ввода и воспроизведения ответов.

    :return: Локальная HTML-страница, использующая WebSocket API этого же сервера.
    """
    return FileResponse(path=PROJECT_ROOT / "static" / "index.html", media_type="text/html", headers={"Cache-Control": "no-store"})
