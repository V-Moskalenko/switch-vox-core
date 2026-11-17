"""Композиция локального API: ресурсы сессии, модельный каскад и браузерный клиент."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from loguru import logger
from starlette.staticfiles import StaticFiles

from api.routers.client_router import router as client_router
from api.routers.health_router import router as health_router
from api.routers.session_router import router as session_router
from services.cascade_service import CascadeService
from services.session_service import SessionService
from settings import PROJECT_ROOT, config, setup_logging


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Настраивает ресурсы одного API-процесса без объявления моделей готовыми.

    :param application: Единственный экземпляр FastAPI данного процесса.
    :yield: Управление серверу после настройки процесса.
    """
    setup_logging(config.loging_settings)
    application.state.settings = config
    async with httpx.AsyncClient(timeout=config.inference.connect_timeout_seconds, trust_env=False) as client:
        application.state.http_client = client
        cascade = CascadeService(config, client) if config.inference.mode == "cascade" else None
        application.state.session_service = SessionService(config, cascade.generate if cascade is not None else None)
        logger.info("API запущен; режим {}. Готовность моделей проверяется через /health/ready", config.inference.mode)
        try:
            yield
        finally:
            await application.state.session_service.close()
            logger.info("API остановлен")
            # Дожидаемся записи очереди и coroutine sinks до закрытия event loop.
            await logger.complete()


app = FastAPI(title="Switch Vox Core", description="Локальный русскоязычный голосовой ассистент с каскадом ASR, LLM и TTS и браузерным голосовым клиентом.", version="0.1.0", lifespan=lifespan)
app.include_router(health_router)
app.include_router(session_router)
app.include_router(client_router)
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")
