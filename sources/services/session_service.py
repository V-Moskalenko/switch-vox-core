"""Одна WebSocket-сессия на процесс: состояние, ограниченные очереди и отмена генераций."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Final, Literal
from uuid import UUID, uuid4

import orjson
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from api.schemas.session_schemas import (
    ErrorEvent,
    InputCommit,
    InputStart,
    InputStarted,
    PlaybackProgress,
    ResponseAudio,
    ResponseCancel,
    ResponseFinished,
    ResponseStarted,
    ResponseText,
    SegmentEnd,
    ServerEvent,
    SessionReady,
    SessionStart,
    TranscriptEvent,
    client_command_adapter,
)
from services.turn_detection import TurnDetector
from settings import Settings

TRANSCRIPT_HYPOTHESIS: Final = "transcript.hypothesis"
TRANSCRIPT_FINAL: Final = "transcript.final"
type TranscriptKind = Literal["transcript.hypothesis", "transcript.final"]
type ResultKind = Literal["text", "audio", "segment_end", "done", "error"] | TranscriptKind


class SessionError(Exception):
    """Ожидаемый отказ протокола или текущей генерации с публичным машинным кодом."""

    def __init__(self, code: str, message: str) -> None:
        """Сохраняет безопасное русское объяснение без диагностических payload.

        :param code: Стабильный код ошибки протокола.
        :param message: Сообщение, разрешённое для передачи браузеру.
        """
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class DialogueMessage:
    """Неизменяемое сообщение подтверждённой пользовательской или услышанной речи."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(slots=True)
class SpeechSegment:
    """Счётчики одной фразы: подготовленные, отправленные и услышанные отсчёты."""

    text: str = ""
    prepared_samples: int = 0
    sent_samples: int = 0
    played_samples: int = 0
    sequence: int = 0
    complete: bool = False


@dataclass(slots=True)
class Outbound:
    """Неделимая пара заголовка и необязательного binary payload."""

    event: ServerEvent
    generation_id: UUID | None = None
    pcm: bytes | None = None
    close_code: int | None = None


@dataclass(slots=True)
class IncomingFrame:
    """Один уже ограниченный транспортом кадр, передаваемый владельцу состояния."""

    text: str | None = None
    pcm: bytes | None = None
    disconnected: bool = False
    error: SessionError | None = None


@dataclass(slots=True)
class ModelResult:
    """Результат модельной задачи; состояние меняется только после обработки владельцем."""

    generation_id: UUID
    kind: ResultKind
    accepted: asyncio.Future[bool]
    segment_id: UUID | None = None
    text: str = ""
    pcm: bytes = b""
    error: SessionError | None = None


@dataclass(slots=True)
class AudioSent:
    """Подтверждение отправки пары sender-задачей владельцу сессии."""

    generation_id: UUID
    segment_id: UUID
    samples: int


class GenerationContext:
    """Канал одной генерации: модельная задача публикует результаты без изменения состояния."""

    def __init__(self, session: _Session, generation_id: UUID, turn_id: UUID, input_audio: AsyncIterator[bytes] | None = None) -> None:
        """Фиксирует IDs на весь срок жизни обработчика.

        :param session: Сессия, владеющая очередями и состоянием.
        :param generation_id: Неизменяемая генерация данного вызова.
        :param turn_id: Подтверждённая пользовательская реплика.
        :param input_audio: Живой вход автоматической реплики; None для буферного PTT.
        """
        self._session = session
        self.generation_id = generation_id
        self.turn_id = turn_id
        self.history = tuple(message for turn in session.history for message in turn)
        self.input_audio = input_audio

    async def _submit(self, kind: ResultKind, *, segment_id: UUID | None = None, text: str = "", pcm: bytes = b"", error: SessionError | None = None) -> bool:
        """Передаёт результат единственному владельцу и ждёт его решения.

        :param kind: Вид публикуемого результата.
        :param segment_id: Идентификатор TTS-фразы, если она есть.
        :param text: Подготовленная фраза или транскрипция.
        :param pcm: Короткий PCM16 LE mono кадр.
        :param error: Ожидаемый безопасный отказ модели.
        :return: False, если генерация уже отменена либо сессия закрыта.
        """
        if not self._session.is_current(self.generation_id):
            return False
        accepted: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        await self._session.inbox.put(ModelResult(self.generation_id, kind, accepted, segment_id, text, pcm, error))
        return await accepted

    async def emit_text(self, segment_id: UUID, text: str) -> bool:
        """Публикует целую фразу перед её аудио.

        :param segment_id: Новый уникальный идентификатор фразы.
        :param text: Текст, который будет синтезирован.
        :return: Принято ли событие текущей генерацией.
        """
        return await self._submit("text", segment_id=segment_id, text=text)

    async def emit_audio(self, segment_id: UUID, pcm: bytes) -> bool:
        """Ограничивает подготовленное аудио и публикует короткий кадр.

        :param segment_id: Идентификатор ранее опубликованной фразы.
        :param pcm: PCM16 LE mono 24 кГц, не более audio.max_frame_ms.
        :return: False при отмене; True после постановки пары в очередь.
        :raises SessionError: Неверный PCM или клиент не освобождает bounded buffer.
        """
        max_bytes = self._session.settings.audio.output_sample_rate * self._session.settings.audio.max_frame_ms // 1000 * 2
        if not pcm or len(pcm) % 2 or len(pcm) > max_bytes:
            raise SessionError("invalid_output_audio", "Модель вернула неверный PCM-кадр")
        maximum = int(self._session.settings.audio.max_output_buffer_seconds * self._session.settings.audio.output_sample_rate)
        samples = len(pcm) // 2
        if samples > maximum:
            raise SessionError("output_buffer_overflow", "Кадр модели превышает допустимый аудиобуфер")
        # Бюджет освобождает playback ACK, а не отправка PCM в сетевой сокет.
        while self._session.is_current(self.generation_id) and self._session.buffered_samples + samples > maximum:
            self._session.capacity_changed.clear()
            try:
                await asyncio.wait_for(self._session.capacity_changed.wait(), timeout=self._session.settings.inference.idle_timeout_seconds)
            except TimeoutError as exc:
                raise SessionError("output_buffer_overflow", "Клиент не освобождает буфер воспроизведения") from exc
        return await self._submit("audio", segment_id=segment_id, pcm=pcm)

    async def finish_segment(self, segment_id: UUID) -> bool:
        """Фиксирует окончательный размер синтезированной фразы.

        :param segment_id: Идентификатор завершённой TTS-фразы.
        :return: Принято ли событие текущей генерацией.
        """
        return await self._submit("segment_end", segment_id=segment_id)

    async def emit_transcript(self, text: str, *, final: bool) -> bool:
        """Передаёт гипотезу ASR либо финальный текст данной реплики.

        :param text: Текст распознавания без записи в эксплуатационный журнал.
        :param final: True для окончательного результата ASR.
        :return: Принято ли событие текущей генерацией.
        """
        return await self._submit(TRANSCRIPT_FINAL if final else TRANSCRIPT_HYPOTHESIS, text=text)


type GenerationHandler = Callable[[GenerationContext, bytes], Awaitable[None]]


class SessionService:
    """Управляет единственным слотом WebSocket-сессии и общим набором модельных задач."""

    def __init__(self, settings: Settings, generation_handler: GenerationHandler | None = None) -> None:
        """Хранит настройки и необязательный реальный обработчик реплики.

        :param settings: Проверенные ограничения приложения.
        :param generation_handler: Асинхронный обработчик реплики; None запрещает запуск генерации.
        """
        self.settings = settings
        self.generation_handler = generation_handler
        self.active_session: _Session | None = None
        self.model_tasks: set[asyncio.Task[None]] = set()

    async def serve(self, websocket: WebSocket) -> None:
        """Занимает слот до первого await и освобождает после завершения сессии.

        :param websocket: Проверенное по Origin соединение браузера.
        """
        if self.active_session is not None:
            logger.bind(event="session_rejected", code="session_busy").warning("Отклонено подключение: слот голосовой сессии занят")
            await websocket.accept()
            await websocket.send_text(orjson.dumps(ErrorEvent(code="session_busy", message="Другая голосовая сессия уже подключена").model_dump(mode="json")).decode("utf-8"))
            await websocket.close(code=1013)
            return
        session = _Session(self, websocket)
        self.active_session = session
        # Дочерние reader/owner/sender и модельные задачи наследуют только безопасный ID сессии.
        with logger.contextualize(session_id=str(session.session_id)):
            try:
                await websocket.accept()
                logger.bind(event="session_opened").info("Голосовая сессия подключена")
                await session.run()
            finally:
                if self.active_session is session:
                    self.active_session = None
                logger.bind(event="session_closed").info("Голосовая сессия завершена, слот освобождён")

    async def close(self) -> None:
        """Просит владельца активной сессии завершиться при остановке приложения."""
        if self.active_session is not None:
            log = logger.bind(session_id=str(self.active_session.session_id))
            log.bind(event="session_shutdown_requested").debug("Запрошено закрытие активной сессии при остановке API")
            await self.active_session.inbox.put(IncomingFrame(disconnected=True))
            try:
                await asyncio.wait_for(self.active_session.finished.wait(), timeout=self.settings.api.close_timeout_seconds + 1)
            except TimeoutError:
                log.bind(event="session_shutdown_timeout").warning("Сессия не подтвердила завершение за установленный срок")


class _Session:
    """Состояние одного подключения; только owner-loop меняет IDs, PCM и счётчики."""

    def __init__(self, service: SessionService, websocket: WebSocket) -> None:
        """Создаёт ограниченные очереди без запуска фоновых задач.

        :param service: Владелец process-local слота.
        :param websocket: Соединение текущего браузера.
        """
        self.service = service
        self.settings = service.settings
        self.websocket = websocket
        self.session_id = uuid4()
        self.inbox: asyncio.Queue[IncomingFrame | ModelResult | AudioSent] = asyncio.Queue(maxsize=self.settings.api.outbound_queue_size)
        self.outbox: asyncio.Queue[Outbound] = asyncio.Queue(maxsize=self.settings.api.outbound_queue_size)
        self.control: asyncio.Queue[Outbound] = asyncio.Queue(maxsize=16)
        self.wake_sender = asyncio.Event()
        self.capacity_changed = asyncio.Event()
        self.finished = asyncio.Event()
        self.started = False
        self.closed = False
        self.recording = False
        self.generating = False
        self.input_pcm = bytearray()
        self.generation_id: UUID | None = None
        self.turn_id: UUID | None = None
        self.segments: dict[UUID, SpeechSegment] = {}
        self.buffered_samples = 0
        self.generation_task: asyncio.Task[None] | None = None
        # Незавершённый upstream блокирует новый inference и после reconnect.
        self.model_tasks = service.model_tasks
        self.history: list[tuple[DialogueMessage, ...]] = []
        self.user_text = ""
        self.input_mode: Literal["ptt", "auto"] = "ptt"
        self.turn_detector: TurnDetector | None = None
        self.input_queue: asyncio.Queue[bytes | None] | None = None
        self.retired_audio: dict[UUID, dict[UUID, int]] = {}

    def is_current(self, generation_id: UUID) -> bool:
        """Проверяет актуальность без изменения принадлежащего owner-loop состояния.

        :param generation_id: Неизменяемый ID публикующего обработчика.
        :return: Разрешена ли публикация результата сейчас.
        """
        return not self.closed and self.generation_id == generation_id

    async def run(self) -> None:
        """Запускает приём, обработку и отправку; при выходе отменяет задачи сессии."""
        reader = asyncio.create_task(self._reader(), name="session-reader")
        sender = asyncio.create_task(self._sender(), name="session-sender")
        owner = asyncio.create_task(self._owner(), name="session-owner")
        try:
            done, _ = await asyncio.wait({owner, sender}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except (WebSocketDisconnect, OSError) as exc:
            if isinstance(exc, TimeoutError):
                logger.bind(event="session_send_timeout").warning("Истекло время отправки события голосовой сессии")
            else:
                logger.bind(event="session_transport_closed", error_type=type(exc).__name__).debug("Транспорт голосовой сессии закрыт")
        finally:
            # Сначала запрещаем новые результаты, затем отменяем все связанные задачи с конечным ожиданием.
            self.closed = True
            self.capacity_changed.set()
            tasks = {reader, sender, owner} | self.model_tasks
            for task in tasks:
                if not task.cancelling():
                    task.cancel()
            _, pending = await asyncio.wait(tasks, timeout=self.settings.api.close_timeout_seconds)
            if pending:
                logger.bind(event="session_cleanup_timeout", pending_tasks=len(pending)).error("Задачи сессии не остановились за установленный срок")
            for task in tasks - pending:
                if not task.cancelled():
                    task.exception()
            self.input_pcm.clear()
            self.segments.clear()
            self.history.clear()
            self.finished.set()
            if self.websocket.application_state != WebSocketState.DISCONNECTED:
                with suppress(RuntimeError, OSError):
                    await self.websocket.close()

    async def _reader(self) -> None:
        """Передаёт транспортные сообщения владельцу, ограничивая время бездействия."""
        try:
            while True:
                timeout = self.settings.api.session_idle_timeout_seconds if self.started else self.settings.api.handshake_timeout_seconds
                message = await asyncio.wait_for(self.websocket.receive(), timeout=timeout)
                if message["type"] == "websocket.disconnect":
                    await self.inbox.put(IncomingFrame(disconnected=True))
                    return
                text = message.get("text")
                pcm = message.get("bytes")
                await self.inbox.put(IncomingFrame(text=text if isinstance(text, str) else None, pcm=pcm if isinstance(pcm, bytes) else None))
        except TimeoutError:
            logger.bind(event="session_input_timeout", configured=self.started).debug("Истекло время ожидания входящего события сессии")
            await self.inbox.put(IncomingFrame(error=SessionError("session_timeout", "Истекло время ожидания команды сессии")))
        except WebSocketDisconnect:
            await self.inbox.put(IncomingFrame(disconnected=True))

    def _enqueue(self, event: ServerEvent, *, generation_id: UUID | None = None, pcm: bytes | None = None, priority: bool = False, close_code: int | None = None) -> None:
        """Ставит неделимую пару в ограниченную очередь без блокировки owner-loop.

        :param event: Валидированное серверное событие.
        :param generation_id: ID для финальной фильтрации запоздалых результатов.
        :param pcm: Необязательный binary payload после заголовка.
        :param priority: Управляющие события обходят очередь аудио.
        :param close_code: Завершить WebSocket после отправки данного события.
        :raises SessionError: Исчерпан конечный лимит очереди.
        """
        queue = self.control if priority else self.outbox
        try:
            queue.put_nowait(Outbound(event, generation_id, pcm, close_code))
        except asyncio.QueueFull as exc:
            raise SessionError("output_queue_overflow", "Клиент не успевает принимать события ответа") from exc
        self.wake_sender.set()

    async def _sender(self) -> None:
        """Единолично отправляет пары; между header и binary нет управляющих событий."""
        while True:
            if self.control.empty() and self.outbox.empty():
                self.wake_sender.clear()
                await self.wake_sender.wait()
            item = self.control.get_nowait() if not self.control.empty() else self.outbox.get_nowait()
            if item.generation_id is not None and not self.is_current(item.generation_id):
                continue
            if await self._send_item(item):
                return

    async def _send_item(self, item: Outbound) -> bool:
        """Отправляет атомарную пару под общим deadline и подтверждает её владельцу.

        :param item: Проверенная на актуальность пара JSON и PCM.
        :return: True после финального закрытия соединения.
        """
        async with asyncio.timeout(self.settings.api.send_timeout_seconds):
            # Заголовок и его PCM отправляет одна задача: управляющее событие не разорвёт пару.
            await self.websocket.send_text(orjson.dumps(item.event.model_dump(mode="json")).decode("utf-8"))
            if item.pcm is not None:
                await self.websocket.send_bytes(item.pcm)
                if isinstance(item.event, ResponseAudio):
                    await self.inbox.put(AudioSent(item.event.generation_id, item.event.segment_id, len(item.pcm) // 2))
            if item.close_code is not None:
                await self.websocket.close(code=item.close_code)
                return True
        return False

    async def _owner(self) -> None:
        """Последовательно применяет команды браузера и результаты модельных задач."""
        while True:
            message = await self.inbox.get()
            if isinstance(message, AudioSent):
                if self.is_current(message.generation_id) and message.segment_id in self.segments:
                    self.segments[message.segment_id].sent_samples += message.samples
                continue
            if isinstance(message, IncomingFrame) and message.disconnected:
                return
            try:
                self._apply_message(message)
            except (SessionError, ValidationError) as exc:
                self._fail_message(message, exc)

    def _apply_message(self, message: IncomingFrame | ModelResult) -> None:
        """Применяет один вход и подтверждает принятый результат модельной задаче.

        :param message: Ограниченный кадр клиента либо результат модели.
        :raises SessionError: Ошибка протокола, входного PCM или модели.
        """
        if isinstance(message, ModelResult):
            accepted = self._handle_result(message)
            if not message.accepted.done():
                message.accepted.set_result(accepted)
        elif message.error is not None:
            raise message.error
        elif message.text is not None:
            self._handle_command(message.text)
        elif message.pcm is not None:
            self._handle_audio(message.pcm)

    def _fail_message(self, message: IncomingFrame | ModelResult, error: SessionError | ValidationError) -> None:
        """Отменяет неверный ответ и публикует только безопасную причину отказа.

        :param message: Источник ошибки и future подтверждения, если он есть.
        :param error: Ожидаемый отказ протокола либо валидации серверной схемы.
        """
        failure = error if isinstance(error, SessionError) else SessionError("invalid_model_event", "Модель вернула неверное событие ответа")
        logger.bind(event="session_failed", code=failure.code, generation_id=str(self.generation_id) if self.generation_id else None).warning("Отклонено событие голосовой сессии")
        if isinstance(message, ModelResult):
            if self.turn_detector is not None:
                self.turn_detector.abort()
            self._cancel_generation(notify=False)
            if not message.accepted.done():
                message.accepted.set_exception(failure)
        self._enqueue(ErrorEvent(code=failure.code, message=str(failure)), priority=True, close_code=1008 if failure.code == "session_timeout" else None)

    def _handle_command(self, text: str) -> None:
        """Валидирует JSON и применяет допустимый переход состояния.

        :param text: Единственная JSON-команда UTF-8.
        :raises SessionError: Неверная команда, формат или устаревшие IDs.
        """
        if len(text.encode("utf-8")) > self.settings.api.max_json_bytes:
            raise SessionError("command_too_large", "Команда превышает допустимый размер JSON")
        try:
            command = client_command_adapter.validate_json(text)
        except ValidationError as exc:
            raise SessionError("invalid_command", "Неверная команда или параметры аудиопротокола") from exc
        if isinstance(command, SessionStart):
            self._start_session(command)
            return
        if not self.started:
            raise SessionError("handshake_required", "Сначала отправьте session.start с форматом PCM")
        if isinstance(command, InputStart):
            if self.input_mode == "auto":
                raise SessionError("input_mode_conflict", "В автоматическом режиме начало речи определяет VAD")
            self._start_input()
        elif isinstance(command, InputCommit):
            if self.input_mode == "auto":
                raise SessionError("input_mode_conflict", "В автоматическом режиме конец речи определяет VAD")
            self._commit_input(command)
        elif isinstance(command, ResponseCancel):
            self._cancel_command(command)
        elif isinstance(command, PlaybackProgress):
            self._playback_progress(command)

    def _cancel_command(self, command: ResponseCancel) -> None:
        """Отменяет только указанное поколение и подавляет хвост незаконченной речи.

        :param command: Ручная остановка текущего ответа или входной реплики.
        :raises SessionError: ID относится к прежней либо неизвестной генерации.
        """
        if command.generation_id != self.generation_id:
            raise SessionError("stale_generation", "Отменяемая генерация уже не активна")
        if self.recording and self.turn_detector is not None:
            self.turn_detector.abort()
        self._cancel_generation(notify=True)

    def _start_session(self, command: SessionStart) -> None:
        """Согласует один режим микрофона и создаёт CPU VAD только для auto.

        :param command: Первая команда с форматом PCM и режимом микрофона.
        :raises SessionError: Сессия уже согласована, live-переключение запрещено.
        """
        if self.started:
            raise SessionError("session_already_started", "Параметры открытой сессии менять нельзя")
        self.started = True
        self.input_mode = command.input_mode
        if self.input_mode == "auto":
            self.turn_detector = TurnDetector(self.settings.turn, self.settings.audio.max_turn_seconds)
        logger.bind(event="session_configured", input_mode=self.input_mode).info("Согласованы параметры голосовой сессии")
        self._enqueue(SessionReady(session_id=self.session_id, mode=self.settings.inference.mode, input_mode=self.input_mode), priority=True)

    def _start_input(self) -> None:
        """Открывает запись после архивирования услышанного и отмены старого ответа.

        :raises SessionError: Клиент уже записывает реплику.
        """
        if self.recording:
            raise SessionError("input_already_started", "Входная реплика уже записывается")
        self._cancel_generation(notify=self.generation_id is not None)
        self.generation_id, self.turn_id = uuid4(), uuid4()
        self.recording = True
        logger.bind(event="turn_started", turn_id=str(self.turn_id), generation_id=str(self.generation_id), input_mode=self.input_mode).debug("Начата входная реплика")
        self._enqueue(InputStarted(turn_id=self.turn_id, generation_id=self.generation_id), priority=True)

    def _commit_input(self, command: InputCommit) -> None:
        """Фиксирует PCM и запускает независимую от owner-loop модельную задачу.

        :param command: Команда завершения текущей реплики.
        :raises SessionError: Реплика пуста, IDs устарели или обработчик не подключён.
        """
        if command.generation_id != self.generation_id or command.turn_id != self.turn_id:
            raise SessionError("stale_generation", "Команда относится к другой реплике")
        if not self.recording or not self.input_pcm:
            raise SessionError("empty_input", "Нет записанной аудиореплики для подтверждения")
        self.recording = False
        pcm = bytes(self.input_pcm)
        self.input_pcm.clear()
        self._launch_generation(pcm, notify=True)

    def _launch_generation(self, pcm: bytes, *, notify: bool, input_audio: AsyncIterator[bytes] | None = None) -> None:
        """Запускает ASR/каскад вне владельца; в auto уведомление откладывается до конца речи.

        :param pcm: Подтверждённый буфер PTT либо пустой байтовый буфер для живого входа.
        :param notify: Сразу публиковать response.started для подтверждённой реплики.
        :param input_audio: Ограниченный живой источник PCM автоматической реплики.
        :raises SessionError: Модельный обработчик не подключён.
        """
        handler = self.service.generation_handler
        if handler is None or self.generation_id is None or self.turn_id is None:
            raise SessionError("model_unavailable", "Модельный контур ещё не подключён")
        context = GenerationContext(self, self.generation_id, self.turn_id, input_audio)
        self.generating = True
        logger.bind(event="generation_started", turn_id=str(self.turn_id), generation_id=str(self.generation_id), streaming_input=input_audio is not None).debug("Запущена обработка реплики")
        if notify:
            self._enqueue(ResponseStarted(turn_id=self.turn_id, generation_id=self.generation_id), generation_id=self.generation_id)
        self.generation_task = asyncio.create_task(self._generate(handler, context, pcm), name="session-generation")
        self.model_tasks.add(self.generation_task)
        self.generation_task.add_done_callback(self.model_tasks.discard)

    def _handle_audio(self, pcm: bytes) -> None:
        """Ограничивает PCM по формату, размеру кадра и длительности реплики.

        :param pcm: Входной PCM16 LE mono при 16 кГц.
        :raises SessionError: Нет записи, неверный кадр или превышен лимит реплики.
        """
        if not self.started:
            raise SessionError("input_not_started", "Перед PCM-кадрами согласуйте session.start")
        maximum = self.settings.audio.input_sample_rate * self.settings.audio.max_frame_ms // 1000 * 2
        if not pcm or len(pcm) % 2 or len(pcm) > maximum:
            raise SessionError("invalid_audio_frame", "Нужен непустой PCM16 LE mono кадр не длиннее audio.max_frame_ms")
        if self.turn_detector is not None:
            self._automatic_audio(pcm)
            return
        if not self.recording:
            raise SessionError("input_not_started", "Перед PCM-кадрами отправьте input.start")
        turn_maximum = int(self.settings.audio.max_turn_seconds * self.settings.audio.input_sample_rate * 2)
        if len(self.input_pcm) + len(pcm) > turn_maximum:
            self._cancel_generation(notify=False)
            raise SessionError("input_too_long", "Превышена допустимая длительность реплики")
        self.input_pcm.extend(pcm)

    def _automatic_audio(self, pcm: bytes) -> None:
        """Обрабатывает границы VAD, не блокируя owner ожиданием медленного ASR.

        :param pcm: Проверенный входной кадр непрерывного микрофона.
        :raises SessionError: ASR не успевает принимать ограниченный живой поток.
        """
        assert self.turn_detector is not None
        try:
            for event in self.turn_detector.feed(pcm, interrupting=self.generating or self.buffered_samples > 0):
                if event.kind == "start":
                    # Новая подтверждённая речь инвалидирует прежний ответ до подачи pre-roll в ASR.
                    self._start_input()
                    self.input_queue = asyncio.Queue(maxsize=64)
                    self.input_queue.put_nowait(event.pcm)
                    self._launch_generation(b"", notify=False, input_audio=self._input_audio(self.input_queue))
                elif event.kind == "audio" and self.input_queue is not None:
                    self.input_queue.put_nowait(event.pcm)
                elif event.kind == "end" and self.input_queue is not None:
                    self.recording = False
                    self.input_queue.put_nowait(None)
                    logger.bind(event="turn_ended", turn_id=str(self.turn_id), generation_id=str(self.generation_id), reason=event.reason).debug("VAD подтвердил конец входной реплики")
                    if self.turn_id is not None and self.generation_id is not None:
                        self._enqueue(ResponseStarted(turn_id=self.turn_id, generation_id=self.generation_id), generation_id=self.generation_id)
                    self.input_queue = None
        except (asyncio.QueueFull, SessionError) as exc:
            self.turn_detector.abort()
            self._cancel_generation(notify=False)
            if isinstance(exc, SessionError):
                raise
            raise SessionError("input_queue_overflow", "ASR не успевает принимать микрофон; текущая реплика остановлена") from exc

    async def _input_audio(self, queue: asyncio.Queue[bytes | None]) -> AsyncIterator[bytes]:
        """Передаёт одной ASR-задаче PCM до единственного подтверждённого EOF от VAD.

        :param queue: Неизменяемая очередь конкретной реплики, не нового поколения.
        :yield: PCM из pre-roll и следующих окон VAD.
        """
        while (pcm := await queue.get()) is not None:
            yield pcm

    def _playback_progress(self, command: PlaybackProgress) -> None:
        """Принимает только монотонный playback в пределах реально отправленного PCM.

        :param command: Подтверждение клиента для конкретной фразы.
        :raises SessionError: Неизвестная фраза, старый ID или неверный счётчик.
        """
        if command.generation_id != self.generation_id:
            retired = self.retired_audio.get(command.generation_id) if self.input_mode == "auto" else None
            if retired is not None and command.segment_id in retired and command.played_samples <= retired[command.segment_id]:
                return
            raise SessionError("stale_generation", "Подтверждение воспроизведения относится к старому ответу")
        segment = self.segments.get(command.segment_id)
        if segment is None or not segment.played_samples <= command.played_samples <= segment.sent_samples:
            raise SessionError("invalid_playback_progress", "Неверное число воспроизведённых отсчётов фразы")
        self.buffered_samples -= command.played_samples - segment.played_samples
        segment.played_samples = command.played_samples
        self.capacity_changed.set()

    def _cancel_generation(self, *, notify: bool) -> None:
        """Сохраняет услышанное, инвалидирует ID и отменяет генерацию без ожидания модели.

        :param notify: Отправлять ли приоритетное подтверждение отмены.
        """
        previous = self.generation_id
        if previous is not None:
            logger.bind(event="generation_reset", generation_id=str(previous), generating=self.generating, recording=self.recording).debug("Инвалидирована текущая генерация и очищены её буферы")
        if previous is not None and self.input_mode == "auto":
            # Начатая пара может закончить send после инвалидирования ID; история уже заморожена.
            self.retired_audio[previous] = {key: segment.prepared_samples for key, segment in self.segments.items()}
            if len(self.retired_audio) > 4:
                del self.retired_audio[next(iter(self.retired_audio))]
        self._archive_turn()
        self.generation_id = None
        self.turn_id = None
        self.recording = False
        self.generating = False
        self.input_pcm.clear()
        self.input_queue = None
        self.segments.clear()
        self.buffered_samples = 0
        self.capacity_changed.set()
        if self.generation_task is not None:
            if not self.generation_task.cancelling():
                self.generation_task.cancel()
            self.generation_task = None
        while not self.outbox.empty():
            self.outbox.get_nowait()
        if notify and previous is not None:
            self._enqueue(ResponseFinished(type="response.cancelled", generation_id=previous), priority=True)

    def _archive_turn(self) -> None:
        """Сохраняет подтверждённый вопрос и непрерывный префикс полностью услышанных фраз."""
        if not self.user_text:
            return
        heard = []
        # В историю входит только непрерывный префикс целых фраз с полным playback ACK.
        for segment in self.segments.values():
            if not segment.complete or not segment.prepared_samples or segment.played_samples != segment.prepared_samples:
                break
            heard.append(segment.text)
        turn = [DialogueMessage(role="user", content=self.user_text)]
        if heard:
            turn.append(DialogueMessage(role="assistant", content=" ".join(heard)))
        self.history.append(tuple(turn))
        self.history = self.history[-self.settings.inference.max_history_turns :]
        self.user_text = ""

    async def _generate(self, handler: GenerationHandler, context: GenerationContext, pcm: bytes) -> None:
        """Изолирует ошибки обработчика; отмена кооперативно закрывает upstream streams.

        :param handler: Конкретный каскадный обработчик данной реплики.
        :param context: Канал неизменяемых IDs.
        :param pcm: Буфер PTT; в автоматическом режиме аудио поступает через context.input_audio.
        """
        with logger.contextualize(turn_id=str(context.turn_id), generation_id=str(context.generation_id)):
            try:
                # Новый запрос к GPU не обгоняет очистку отменённого запроса, даже после reconnect.
                previous = self.model_tasks - {asyncio.current_task()}
                if previous:
                    logger.bind(event="generation_waiting_cleanup", previous_tasks=len(previous)).debug("Ожидание остановки прежних модельных задач")
                    _, pending = await asyncio.wait(previous, timeout=self.settings.api.close_timeout_seconds)
                    if pending:
                        raise SessionError("model_busy", "Предыдущий модельный запрос ещё не завершил остановку")
                if not self.is_current(context.generation_id):
                    return
                await handler(context, pcm)
                await context._submit("done")
            except asyncio.CancelledError:
                logger.bind(event="generation_cancelled").debug("Модельная задача отменена")
                raise
            except SessionError as exc:
                await context._submit("error", error=exc)
            except Exception as exc:
                # Сообщение исключения может содержать текст запроса или URL; сохраняем только тип.
                logger.bind(event="generation_failed", error_type=type(exc).__name__).error("Непредвиденная ошибка модельного обработчика")
                await context._submit("error", error=SessionError("model_error", "Модельный контур завершился с ошибкой"))
            finally:
                logger.bind(event="generation_closed").debug("Модельный обработчик завершил работу")

    def _handle_result(self, result: ModelResult) -> bool:
        """Проверяет generation ID перед каждым изменением состояния и публикацией.

        :param result: Результат конкурентной модельной задачи.
        :return: False для запоздавшей генерации.
        :raises SessionError: Модель нарушила порядок фраз или формат протокола.
        """
        if not self.is_current(result.generation_id) or not self.generating:
            return False
        generation_id = result.generation_id
        if result.kind == "error":
            raise result.error or SessionError("model_error", "Модельный контур завершился с ошибкой")
        if self.input_mode == "auto" and self.recording and result.kind != TRANSCRIPT_HYPOTHESIS:
            raise SessionError("invalid_model_event", "До подтверждённого конца речи допустима только ASR-гипотеза")
        if result.kind == "done":
            self.generating = False
            logger.bind(event="generation_completed", generation_id=str(generation_id), turn_id=str(self.turn_id)).debug("Генерация ответа завершена; воспроизведение может продолжаться")
            self._enqueue(ResponseFinished(type="response.done", generation_id=generation_id), generation_id=generation_id)
            return True
        if result.kind in {TRANSCRIPT_HYPOTHESIS, TRANSCRIPT_FINAL}:
            self._publish_transcript(result)
            return True
        self._handle_segment_result(result)
        return True

    def _publish_transcript(self, result: ModelResult) -> None:
        """Публикует заменяемую гипотезу либо сохраняет подтверждённый вопрос.

        :param result: Актуальная ASR-гипотеза или final после конца входной речи.
        """
        if self.turn_id is None:
            return
        if result.kind == TRANSCRIPT_FINAL:
            self.user_text = result.text.strip()
        event_type: TranscriptKind = TRANSCRIPT_FINAL if result.kind == TRANSCRIPT_FINAL else TRANSCRIPT_HYPOTHESIS
        self._enqueue(TranscriptEvent(type=event_type, turn_id=self.turn_id, generation_id=result.generation_id, text=result.text), generation_id=result.generation_id)

    def _handle_segment_result(self, result: ModelResult) -> None:
        """Применяет текст или PCM к текущей TTS-фразе, сохраняя порядок событий.

        :param result: Актуальный результат после проверки generation ID владельцем.
        :raises SessionError: Неизвестная/повторная фраза либо неправильный порядок PCM.
        """
        generation_id = result.generation_id
        segment_id = result.segment_id
        if segment_id is None:
            raise SessionError("invalid_model_event", "Результат модели не содержит идентификатор фразы")
        if result.kind == "text":
            if len(self.segments) >= self.settings.api.max_response_segments:
                raise SessionError("response_too_long", "Превышено допустимое число фраз одного ответа")
            if segment_id in self.segments:
                raise SessionError("invalid_model_event", "Идентификатор фразы уже использован")
            event = ResponseText(generation_id=generation_id, segment_id=segment_id, text=result.text)
            self.segments[segment_id] = SpeechSegment(text=result.text)
            self._enqueue(event, generation_id=generation_id)
            return
        segment = self.segments.get(segment_id)
        if segment is None or segment.complete:
            raise SessionError("invalid_model_event", "Аудио должно относиться к открытой текстовой фразе")
        if result.kind == "audio":
            samples = len(result.pcm) // 2
            self._enqueue(ResponseAudio(generation_id=generation_id, segment_id=segment_id, sequence=segment.sequence, byte_length=len(result.pcm)), generation_id=generation_id, pcm=result.pcm)
            segment.sequence += 1
            segment.prepared_samples += samples
            self.buffered_samples += samples
        elif result.kind == "segment_end":
            segment.complete = True
            self._enqueue(SegmentEnd(generation_id=generation_id, segment_id=segment_id, total_samples=segment.prepared_samples), generation_id=generation_id)
