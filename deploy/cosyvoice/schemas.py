"""HTTP-контракт синтеза речи и состояния модели CosyVoice3."""

from pydantic import BaseModel, ConfigDict, Field


class SynthesisRequest(BaseModel):
    """Текст одной фразы для озвучивания голосом, выбранным при запуске сервиса."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str = Field(min_length=1, max_length=500, description="Текст фразы для озвучивания; максимум 500 символов, голос выбирается при запуске сервиса")


class HealthResponse(BaseModel):
    """Реальная готовность модели и факт активного синтеза."""

    ready: bool = Field(description="Веса модели и признаки выбранного образца голоса загружены")
    busy: bool = Field(description="Один синтез или его отмена ещё выполняются")
    sample_rate: int = Field(default=24000, description="Частота выходного PCM, Гц")
    channels: int = Field(default=1, description="Число каналов PCM: моно")
    sample_width_bytes: int = Field(default=2, description="Размер знакового PCM16LE отсчёта, байт")
    progress: dict[str, int] = Field(
        default_factory=dict, description="Счётчики и времена текущего или последнего синтеза без текста и аудио; *_us — мкс от начала генерации, frontend_us — длительность подготовки входа"
    )
