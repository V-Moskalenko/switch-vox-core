"""Настройки API и аудиоконтракта; импорт не запускает модели и журналирование."""

import sys
from pathlib import Path
from typing import Literal, Self

from loguru import logger
from pydantic import AnyHttpUrl, BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class ApiSettings(BaseModel):
    """Параметры локального API с единственным владельцем голосовой сессии."""

    host: str = Field(default="127.0.0.1", min_length=1, description="Адрес привязки API; по умолчанию доступен только локально")
    port: int = Field(default=8000, ge=1, le=65535, description="TCP-порт HTTP и WebSocket сервера")
    workers: int = Field(default=1, ge=1, le=1, description="Единственный Uvicorn worker: состояние сессии хранится в памяти процесса")
    max_sessions: int = Field(default=1, ge=1, le=1, description="Единственная активная голосовая сессия в процессе API")
    allowed_origins: list[str] = Field(default_factory=list, description="Разрешённые WS Origin; пустой список допускает localhost и 127.0.0.1 с api.port")
    max_json_bytes: int = Field(default=16384, ge=256, le=1048576, description="Максимальный размер одной управляющей JSON-команды UTF-8, байт")
    outbound_queue_size: int = Field(default=128, ge=4, le=4096, description="Максимальное количество событий в обычной очереди отправки")
    max_response_segments: int = Field(default=128, ge=1, le=4096, description="Максимальное число текстовых фраз одного ответа, каждая озвучивается отдельно")
    send_timeout_seconds: float = Field(default=5, gt=0, allow_inf_nan=False, description="Общий срок отправки события или пары JSON/PCM и подтверждения владельцу сессии, с")
    handshake_timeout_seconds: float = Field(default=10, gt=0, allow_inf_nan=False, description="Ожидание первой session.start после подключения, с")
    session_idle_timeout_seconds: float = Field(default=300, gt=0, allow_inf_nan=False, description="Предельное ожидание следующего входящего WS-сообщения, с")
    close_timeout_seconds: float = Field(default=2, gt=0, allow_inf_nan=False, description="Предельное ожидание кооперативной остановки задач сессии, с")


class AudioSettings(BaseModel):
    """Фиксированный PCM16 LE mono и ограниченные аудиобуферы сессии."""

    input_sample_rate: int = Field(default=16000, ge=16000, le=16000, description="Частота входного PCM после реального ресемплирования браузером, Гц")
    output_sample_rate: int = Field(default=24000, ge=24000, le=24000, description="Частота PCM, получаемого от TTS и воспроизводимого браузером, Гц")
    channels: int = Field(default=1, ge=1, le=1, description="Один аудиоканал: протокол принимает только моно")
    sample_width_bytes: int = Field(default=2, ge=2, le=2, description="Размер знакового PCM16 little-endian отсчёта, байт")
    frame_ms: int = Field(default=20, ge=1, le=100, description="Обычная длительность входного PCM-кадра, мс")
    max_frame_ms: int = Field(default=100, ge=1, le=100, description="Максимальная длительность входного PCM-кадра, не более 100 мс")
    max_turn_seconds: float = Field(default=30, ge=0.032, allow_inf_nan=False, description="Максимальная длительность одной входной реплики, с; не короче одного окна VAD 32 мс")
    max_output_buffer_seconds: float = Field(default=2, gt=0, allow_inf_nan=False, description="Максимальная длительность принятого от TTS аудио без подтверждения воспроизведения, с")

    @model_validator(mode="after")
    def validate_frame_limits(self) -> Self:
        """Проверяет, что обычный кадр помещается в разрешённый максимум.

        :return: Настройки согласованного размера аудиокадров.
        :raises ValueError: Обычный кадр превышает максимальный размер.
        """
        if self.frame_ms > self.max_frame_ms:
            raise ValueError("Длительность кадра не должна превышать audio.max_frame_ms")
        return self


class TurnSettings(BaseModel):
    """Пороги CPU VAD для непрерывного микрофона, окончания реплики и перебивания."""

    speech_threshold: float = Field(default=0.6, gt=0, le=1, allow_inf_nan=False, description="Минимальная вероятность речи Silero для речевого окна длительностью 32 мс")
    min_speech_ms: int = Field(default=96, ge=32, le=1000, description="Длительность подтверждения начала речи; округляется вверх до окон по 32 мс")
    barge_in_speech_ms: int = Field(default=128, ge=32, le=1000, description="Подтверждение речи для остановки ответа ассистента, мс; округляется вверх до окон VAD")
    end_silence_ms: int = Field(default=640, ge=32, le=3000, description="Непрерывная пауза, завершающая пользовательскую реплику, мс")
    pre_roll_ms: int = Field(default=320, ge=32, le=1000, description="Сохраняемое начало входной речи до подтверждения VAD, мс; покрывает оба порога подтверждения")

    @model_validator(mode="after")
    def validate_pre_roll(self) -> Self:
        """Проверяет, что pre-roll покрывает подтверждение начала речи и перебивания.

        :return: Настройки с достаточной длительностью сохранённого начала речи.
        :raises ValueError: Pre-roll короче одного из порогов подтверждения.
        """
        if self.pre_roll_ms < max(self.min_speech_ms, self.barge_in_speech_ms):
            raise ValueError("turn.pre_roll_ms должен покрывать подтверждение начала речи и перебивания")
        return self


class InferenceSettings(BaseModel):
    """Режим голосового диалога, адреса модельных сервисов и ограничения запросов."""

    mode: Literal["cascade", "e2e"] = Field(default="cascade", description="Режим при запуске: cascade использует ASR, LLM и TTS; обработчик e2e не реализован")
    asr_base_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("http://asr:8000"), description="Базовый HTTP(S)-адрес сервиса распознавания речи; /health проверяет готовность, WebSocket /v1/realtime принимает аудио"
    )
    llm_base_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("http://llm:8000/v1"), description="Базовый HTTP(S)-адрес vLLM с /v1 для генерации текста через /chat/completions; /health и /tokenize вызываются без /v1"
    )
    tts_base_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("http://tts:8000"), description="Базовый HTTP(S)-адрес сервиса озвучивания текста CosyVoice3; /synthesize возвращает поток PCM, /health проверяет готовность"
    )
    connect_timeout_seconds: float = Field(default=5, gt=0, allow_inf_nan=False, description="Ожидание подключения к модельному сервису; для проверок готовности — тайм-аут HTTP-запроса, с")
    first_response_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False, description="Предельное ожидание первого результата модели или освобождения TTS, с")
    idle_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False, description="Предельное ожидание продолжения ответа модели или отправки аудиокадра в ASR, с")
    max_response_tokens: int = Field(default=512, ge=1, le=2048, description="Максимум токенов ответа; сохраняется запас в контексте LLM размером 4096 токенов")
    max_history_turns: int = Field(default=10, ge=1, le=32, description="Максимум прошлых пользовательских реплик с подтверждёнными услышанными ответами в памяти сессии")
    max_phrase_characters: int = Field(default=200, ge=20, le=500, description="Предельная длина одной фразы TTS, символов; runtime принимает не более 500")
    phrase_queue_size: int = Field(default=2, ge=1, le=16, description="Число готовых текстовых фраз между потоками LLM и TTS до backpressure")


class LogingSettings(BaseModel):
    """Уровни фоновой записи в консоль и файл без диагностики локальных значений."""

    console_level: LogLevel = Field(default="INFO", description="Минимальный уровень записей в консоли")
    file_level: LogLevel = Field(default="INFO", description="Минимальный уровень записей в файле, если файловый журнал включён")
    file_path: Path | None = Field(default=None, description="Путь журнала; относительный считается от корня проекта, None отключает файл")
    rotation: str = Field(default="10 MB", min_length=1, description="Порог ротации файлов Loguru, например 10 MB")
    retention: str = Field(default="10 days", min_length=1, description="Срок хранения ротированных файлов Loguru, например 10 days")


class Settings(BaseSettings):
    """Конфигурация из корневого .env с приоритетом переменных окружения."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore", env_nested_delimiter="__")

    api: ApiSettings = Field(default_factory=ApiSettings, description="Настройки HTTP/WebSocket API и ограничений локальной сессии")
    audio: AudioSettings = Field(default_factory=AudioSettings, description="Формат PCM и ограничения аудиобуферов")
    turn: TurnSettings = Field(default_factory=TurnSettings, description="Пороги автоматического определения начала, конца речи и перебивания")
    inference: InferenceSettings = Field(default_factory=InferenceSettings, description="Выбранный режим, модельные адреса и предельные ожидания")
    loging_settings: LogingSettings = Field(default_factory=LogingSettings, description="Уровни консоли и параметры необязательного файлового журнала")


def setup_logging(settings: LogingSettings) -> None:
    """Настраивает фоновую запись Loguru через очередь, заменяя прежние обработчики.

    :param settings: Уровни, путь и политика хранения журнала.
    """
    logger.remove()
    log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra} - {message}"
    # Очередь переносит запись в отдельный поток, чтобы sink не задерживал обработку аудио.
    logger.add(sys.stderr, level=settings.console_level, format=log_format, enqueue=True, diagnose=False, backtrace=False)
    if settings.file_path is not None:
        # Loguru интерпретирует фигурные скобки в пути как шаблон, но настройка задаёт буквальный путь.
        log_file = str(PROJECT_ROOT / settings.file_path).replace("{", "{{").replace("}", "}}")
        logger.add(log_file, level=settings.file_level, format=log_format, encoding="utf-8", rotation=settings.rotation, retention=settings.retention, enqueue=True, diagnose=False, backtrace=False)


config = Settings()
