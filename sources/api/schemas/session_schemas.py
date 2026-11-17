"""Единственная версия JSON-протокола; PCM передаётся отдельными binary кадрами."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class ClientEvent(BaseModel):
    """Основа строгой команды браузера без неизвестных параметров."""

    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1] = Field(description="Единственная поддерживаемая версия протокола, 1")


class SessionStart(ClientEvent):
    """Согласует фиксированные русский язык и аудиоформат сессии."""

    type: Literal["session.start"] = Field(description="Первая команда после открытия WebSocket")
    language: Literal["ru"] = Field(description="Язык голосового диалога, русский")
    input_mode: Literal["ptt", "auto"] = Field(default="ptt", description="PTT с ручным commit либо непрерывный микрофон с CPU VAD; фиксируется при подключении")
    input_sample_rate: Literal[16000] = Field(description="Фактическая частота входного PCM, Гц")
    output_sample_rate: Literal[24000] = Field(description="Фактическая частота воспроизводимого PCM, Гц")
    channels: Literal[1] = Field(description="Количество аудиоканалов, только моно")
    sample_width_bytes: Literal[2] = Field(description="Размер PCM16 LE отсчёта, байт")


class InputStart(ClientEvent):
    """Открывает новую PTT-реплику с идентификаторами, назначаемыми сервером."""

    type: Literal["input.start"] = Field(description="Начало записи очередной пользовательской реплики")


class InputCommit(ClientEvent):
    """Завершает только указанную активную входную реплику."""

    type: Literal["input.commit"] = Field(description="Подтверждение окончания текущей PTT-реплики")
    turn_id: UUID = Field(description="Идентификатор реплики из input.started")
    generation_id: UUID = Field(description="Идентификатор ответа из того же input.started")


class ResponseCancel(ClientEvent):
    """Отменяет запись или ответ указанной генерации."""

    type: Literal["response.cancel"] = Field(description="Немедленная остановка записи и ответа данной генерации")
    generation_id: UUID = Field(description="Отменяемая генерация из input.started")


class PlaybackProgress(ClientEvent):
    """Подтверждает фактически воспроизведённые отсчёты одной TTS-фразы."""

    type: Literal["playback.progress"] = Field(description="Подтверждение воспроизведения конкретной фразы")
    generation_id: UUID = Field(description="Генерация, к которой относится воспроизведение")
    segment_id: UUID = Field(description="Идентификатор TTS-фразы из response.text/audio")
    played_samples: int = Field(ge=0, description="Суммарно воспроизведённые отсчёты данной фразы, моно при 24 кГц")


type ClientCommand = Annotated[SessionStart | InputStart | InputCommit | ResponseCancel | PlaybackProgress, Field(discriminator="type")]
client_command_adapter = TypeAdapter(ClientCommand)


class ServerEvent(BaseModel):
    """Основа серверного события опубликованной версии протокола."""

    version: Literal[1] = Field(default=1, description="Версия протокола, 1")


class SessionReady(ServerEvent):
    """Подтверждает WS-сессию и согласованный формат, а не готовность моделей."""

    type: Literal["session.ready"] = Field(default="session.ready", description="Управляющий канал готов принимать команды")
    session_id: UUID = Field(description="Уникальный идентификатор данного подключения; reconnect создаёт новый")
    mode: Literal["cascade", "e2e"] = Field(description="Режим приложения, фиксированный при его запуске")
    input_mode: Literal["ptt", "auto"] = Field(default="ptt", description="Подтверждённый режим входа: ручной PTT либо автоматические границы речи")
    language: Literal["ru"] = Field(default="ru", description="Язык диалога, русский")
    input_sample_rate: Literal[16000] = Field(default=16000, description="Частота входного PCM, Гц")
    output_sample_rate: Literal[24000] = Field(default=24000, description="Частота выходного PCM, Гц")
    channels: Literal[1] = Field(default=1, description="Один аудиоканал")
    sample_width_bytes: Literal[2] = Field(default=2, description="Размер PCM16 little-endian отсчёта, байт")


class InputStarted(ServerEvent):
    """Сообщает серверные идентификаторы открытой пользовательской реплики и ответа."""

    type: Literal["input.started"] = Field(default="input.started", description="Реплика открыта: после input.start в PTT или подтверждения речи VAD в auto")
    turn_id: UUID = Field(description="Идентификатор открытой входной реплики")
    generation_id: UUID = Field(description="Идентификатор ответа, связанного с репликой")


class ResponseStarted(ServerEvent):
    """Сообщает о запуске ответа после ручного commit или подтверждения конца речи VAD."""

    type: Literal["response.started"] = Field(default="response.started", description="Запущена генерация ответа")
    turn_id: UUID = Field(description="Подтверждённая пользовательская реплика")
    generation_id: UUID = Field(description="Идентификатор запущенного ответа")


class ResponseText(ServerEvent):
    """Текст одной TTS-фразы, сохраняемый как услышанный только после playback."""

    type: Literal["response.text"] = Field(default="response.text", description="Текст целой очередной фразы; не token delta")
    generation_id: UUID = Field(description="Генерация данного ответа")
    segment_id: UUID = Field(description="Уникальный идентификатор TTS-фразы")
    text: str = Field(min_length=1, max_length=8192, description="Текст фразы для показа и последующего синтеза")


class ResponseAudio(ServerEvent):
    """Заголовок непосредственно следующего binary PCM-кадра без промежуточных событий."""

    type: Literal["response.audio"] = Field(default="response.audio", description="Заголовок непосредственно следующего binary сообщения")
    generation_id: UUID = Field(description="Генерация данного звука")
    segment_id: UUID = Field(description="Идентификатор TTS-фразы")
    sequence: int = Field(ge=0, description="Последовательный номер кадра внутри фразы, начиная с нуля")
    byte_length: int = Field(gt=0, description="Точный размер следующего binary сообщения, байт")
    sample_rate: Literal[24000] = Field(default=24000, description="Частота PCM16 LE mono, Гц")


class SegmentEnd(ServerEvent):
    """Фиксирует окончательный размер синтезированной фразы."""

    type: Literal["response.segment.end"] = Field(default="response.segment.end", description="Звук данной фразы полностью сформирован")
    generation_id: UUID = Field(description="Генерация фразы")
    segment_id: UUID = Field(description="Идентификатор завершённой TTS-фразы")
    total_samples: int = Field(ge=0, description="Итоговое число отправляемых моноотсчётов данной фразы")


class ResponseFinished(ServerEvent):
    """Отмечает завершение генерации либо подтверждённую отмену."""

    type: Literal["response.done", "response.cancelled"] = Field(description="Генерация завершена либо отменена; done не означает окончание playback")
    generation_id: UUID = Field(description="Идентификатор завершённого или отменённого ответа")


class TranscriptEvent(ServerEvent):
    """Отделяет заменяемую ASR-гипотезу от окончательного текста реплики."""

    type: Literal["transcript.hypothesis", "transcript.final"] = Field(description="Гипотеза ASR либо окончательная транскрипция")
    turn_id: UUID = Field(description="Пользовательская реплика, к которой относится текст")
    generation_id: UUID = Field(description="Связанный ответ для отсечения устаревших событий")
    text: str = Field(max_length=16384, description="Текст ASR; пустая финальная строка не запускает ответ")


class ErrorEvent(ServerEvent):
    """Публичная ошибка без аудио, текстов пользователя и внутреннего traceback."""

    type: Literal["error"] = Field(default="error", description="Отказ команды или текущей генерации")
    code: str = Field(min_length=1, max_length=64, description="Стабильный машинный код ошибки", examples=["invalid_command", "model_unavailable"])
    message: str = Field(min_length=1, description="Русское объяснение отказа для пользователя")
