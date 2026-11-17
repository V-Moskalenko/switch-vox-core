"""HTTP-контракт доступности процесса и проверяемого модельного контура."""

from typing import Literal

from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    """Доступность одного API-процесса без обещания работоспособности моделей."""

    status: Literal["alive"] = Field(default="alive", description="Процесс принимает HTTP-запросы")
    service: Literal["switch-vox-core"] = Field(default="switch-vox-core", description="Имя проверяемого приложения")


class ComponentReadiness(BaseModel):
    """Результат проверки обработчика генерации или одного модельного сервиса."""

    name: Literal["pipeline", "asr", "llm", "tts", "e2e"] = Field(description="Проверяемая часть выбранного режима")
    ready: bool = Field(description="Подтверждена ли готовность этой части сейчас")
    reason: str | None = Field(default=None, description="Краткая причина неготовности; null при успешной проверке")


class ReadinessResponse(BaseModel):
    """Готовность реального режима: HTTP 503 до подключения и доступности моделей."""

    status: Literal["ready", "not_ready"] = Field(description="Готовность выбранного голосового контура")
    mode: Literal["cascade", "e2e"] = Field(description="Режим, фиксированный при запуске приложения")
    components: list[ComponentReadiness] = Field(description="Результаты фактических проверок частей выбранного режима")
