"""Оконный CPU VAD и ограниченные границы реплики без ASR, транспорта и генерации."""

from collections import deque
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Final, Literal

from pysilero_vad import SileroVoiceActivityDetector

from settings import TurnSettings

WINDOW_MS: Final = 32
WINDOW_BYTES: Final = 1024


@dataclass(frozen=True, slots=True)
class TurnEvent:
    """Начало с pre-roll, очередное PCM-окно либо подтверждённый конец реплики."""

    kind: Literal["start", "audio", "end"]
    pcm: bytes = b""
    reason: Literal["silence", "limit"] | None = None


class TurnDetector:
    """Один recurrent VAD-контекст на сессию; хранит только pre-roll и неполное окно."""

    def __init__(self, settings: TurnSettings, max_turn_seconds: float) -> None:
        """Фиксирует пороги и загружает Silero VAD для распознавания речи на CPU.

        :param settings: Порог речи, подтверждение начала и длительность конечной паузы.
        :param max_turn_seconds: Предел экспортированного PCM вместе с pre-roll, с.
        :raises ValueError: Предел не конечен либо не вмещает одно 32-мс окно.
        """
        if not isfinite(max_turn_seconds) or max_turn_seconds < WINDOW_MS / 1000:
            raise ValueError("Предел реплики должен быть конечным и составлять не менее 32 мс")
        self._settings = settings
        self._vad = SileroVoiceActivityDetector()
        self._start_windows = ceil(settings.min_speech_ms / WINDOW_MS)
        self._barge_in_windows = ceil(settings.barge_in_speech_ms / WINDOW_MS)
        self._end_windows = ceil(settings.end_silence_ms / WINDOW_MS)
        self._max_windows = max(1, int(max_turn_seconds * 1000 / WINDOW_MS))
        self._pre_roll: deque[bytes] = deque(maxlen=max(ceil(settings.pre_roll_ms / WINDOW_MS), self._start_windows, self._barge_in_windows))
        self._pending = bytearray()
        self.reset()

    def reset(self) -> None:
        """Очищает запись и recurrent state для нового независимого PCM-потока."""
        self._vad.reset()
        self._pre_roll.clear()
        self._pending.clear()
        self._speech_windows = 0
        self._silence_windows = 0
        self._active_windows = 0
        self._suppressed = False
        self._skip_byte = False

    def abort(self) -> None:
        """Отбрасывает текущую реплику и ждёт паузы, сохраняя границу PCM16-отсчёта."""
        # При нечётном transport-срезе следующий байт завершает отброшенный PCM16-отсчёт.
        skip_byte = self._skip_byte or bool(len(self._pending) % 2)
        self.reset()
        self._suppressed = True
        self._skip_byte = skip_byte

    def feed(self, pcm: bytes, *, interrupting: bool = False) -> list[TurnEvent]:
        """Принимает непрерывный PCM16 LE mono 16 кГц с произвольными byte-срезами.

        :param pcm: Очередная часть непрерывной записи; остаток меньше окна сохраняется.
        :param interrupting: Требовать более длинное подтверждение речи во время ответа.
        :return: Упорядоченные события; каждый PCM-байт экспортируется не более раза.
        """
        events: list[TurnEvent] = []
        offset = 0
        if self._skip_byte and pcm:
            offset = 1
            self._skip_byte = False
        while offset < len(pcm):
            end = min(len(pcm), offset + WINDOW_BYTES - len(self._pending))
            self._pending.extend(pcm[offset:end])
            offset = end
            if len(self._pending) == WINDOW_BYTES:
                window = bytes(self._pending)
                self._pending.clear()
                events.extend(self._process_window(window, interrupting=interrupting))
        return events

    def _process_window(self, pcm: bytes, *, interrupting: bool) -> list[TurnEvent]:
        """Классифицирует целое окно и меняет состояние только на его границе.

        :param pcm: Ровно 512 отсчётов PCM16 LE mono.
        :param interrupting: Выбор порога подтверждения начала новой речи.
        :return: События обработки окна; начало содержит накопленный pre-roll.
        """
        speaking = self._vad.process_chunk(pcm) >= self._settings.speech_threshold
        self._silence_windows = 0 if speaking else self._silence_windows + 1
        if self._suppressed:
            # После отмены или лимита хвост той же речи не становится новой репликой до паузы.
            if self._silence_windows >= self._end_windows:
                self.reset()
            return []

        if self._active_windows:
            self._active_windows += 1
            events = [TurnEvent("audio", pcm)]
        else:
            self._pre_roll.append(pcm)
            self._speech_windows = self._speech_windows + 1 if speaking else 0
            required_windows = self._barge_in_windows if interrupting else self._start_windows
            if self._speech_windows < required_windows:
                return []
            # Pre-roll включает начало подтверждаемой речи, но не увеличивает жёсткий лимит реплики.
            start_pcm = b"".join(self._pre_roll)[-self._max_windows * WINDOW_BYTES :]
            self._active_windows = len(start_pcm) // WINDOW_BYTES
            self._pre_roll.clear()
            events = [TurnEvent("start", start_pcm)]

        if self._active_windows >= self._max_windows:
            self.reset()
            self._suppressed = True
            events.append(TurnEvent("end", reason="limit"))
        elif self._silence_windows >= self._end_windows:
            self.reset()
            events.append(TurnEvent("end", reason="silence"))
        return events
