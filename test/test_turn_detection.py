"""Проверяет границы реплик, сохранность PCM и реальный CPU VAD отдельно от ASR."""

import random
from array import array

import pytest
from pysilero_vad import SileroVoiceActivityDetector

from services.turn_detection import TurnDetector, TurnEvent
from settings import TurnSettings

WINDOW_BYTES = 1024
SILENCE = bytes(WINDOW_BYTES)
SPEECH = b"\x01\0" * 512


@pytest.fixture
def controlled_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    """Подменяет вероятность речи для проверки границ реплик без ASR и вычислений VAD.

    :param monkeypatch: Восстановление настоящего вычисления после сценария.
    """

    def probability(detector: SileroVoiceActivityDetector, pcm: bytes) -> float:
        """Отличает помеченную речь от тишины без зависимости теста от модели.

        :param detector: Настоящий CPU detector с собственным контекстом.
        :param pcm: Одно окно PCM16 с тестовой меткой.
        :return: Вероятность речи для детерминированного сценария.
        """
        assert len(pcm) == WINDOW_BYTES
        assert pcm[1::2] == bytes(512)
        return 1.0 if pcm[0] else 0.0

    monkeypatch.setattr(SileroVoiceActivityDetector, "process_chunk", probability)


def audio_bytes(events: list[TurnEvent]) -> bytes:
    """Собирает экспортированный PCM из событий без повторного start.

    :param events: Последовательность начала, аудио и конца реплики.
    :return: Аудио, которое получает ASR.
    """
    return b"".join(event.pcm for event in events)


@pytest.mark.usefixtures("controlled_vad")
def test_transport_boundaries_preserve_each_window_and_internal_pause() -> None:
    """Нечётные transport-срезы и пауза короче порога не режут одну реплику."""
    detector = TurnDetector(TurnSettings(min_speech_ms=64, end_silence_ms=96, pre_roll_ms=128), max_turn_seconds=30)
    before = SILENCE * 6
    assert detector.feed(before) == []
    pcm = SPEECH * 3 + SILENCE * 2 + SPEECH * 2 + SILENCE * 3
    events: list[TurnEvent] = []
    for offset in range(0, len(pcm), 137):
        events.extend(detector.feed(pcm[offset : offset + 137]))
    assert [event.kind for event in events].count("start") == 1
    assert [event.kind for event in events].count("end") == 1
    assert events[0].kind == "start"
    assert events[-1].kind == "end"
    assert events[-1].reason == "silence"
    assert audio_bytes(events) == SILENCE * 2 + pcm
    assert detector.feed(SILENCE * 8) == []


@pytest.mark.usefixtures("controlled_vad")
def test_short_spikes_do_not_start_and_barge_in_requires_more_speech() -> None:
    """Импульсы разделяются тишиной; перебивание требует более долгого подтверждения речи."""
    detector = TurnDetector(TurnSettings(min_speech_ms=64, barge_in_speech_ms=128, pre_roll_ms=128), max_turn_seconds=30)
    assert detector.feed((SPEECH + SILENCE) * 20) == []
    assert detector.feed(SPEECH * 3, interrupting=True) == []
    events = detector.feed(SPEECH, interrupting=True)
    assert [(event.kind, len(event.pcm)) for event in events] == [("start", WINDOW_BYTES * 4)]
    assert audio_bytes(events) == SPEECH * 4


@pytest.mark.usefixtures("controlled_vad")
def test_limit_does_not_split_continuous_speech_into_multiple_answers() -> None:
    """Лимит завершает одну реплику и блокирует продолжение до настоящей паузы."""
    detector = TurnDetector(TurnSettings(min_speech_ms=32, barge_in_speech_ms=32, end_silence_ms=64, pre_roll_ms=32), max_turn_seconds=0.16)
    events = detector.feed(SPEECH * 50)
    assert [event.kind for event in events] == ["start", "audio", "audio", "audio", "audio", "end"]
    assert events[-1].reason == "limit"
    assert audio_bytes(events) == SPEECH * 5
    assert detector.feed(SILENCE + SPEECH * 4) == []
    assert detector.feed(SILENCE * 2) == []
    assert detector.feed(SPEECH)[0].kind == "start"


@pytest.mark.usefixtures("controlled_vad")
@pytest.mark.parametrize("partial_size", [100, 101])
def test_abort_discards_active_audio_and_partial_window_until_pause(partial_size: int) -> None:
    """Отмена не превращает хвост того же слова в новый пользовательский запрос.

    :param partial_size: Размер неполного PCM-окна до отмены, байт.
    """
    detector = TurnDetector(TurnSettings(min_speech_ms=32, barge_in_speech_ms=32, end_silence_ms=64, pre_roll_ms=32), max_turn_seconds=30)
    assert detector.feed(SPEECH + SPEECH[:partial_size])[0].kind == "start"
    detector.abort()
    assert detector.feed(bytes(partial_size % 2) + SPEECH * 5) == []
    assert detector.feed(SILENCE * 2) == []
    events = detector.feed(SPEECH)
    assert [(event.kind, event.pcm) for event in events] == [("start", SPEECH)]


@pytest.mark.usefixtures("controlled_vad")
def test_reset_drops_pre_roll_residual_and_suppression() -> None:
    """Новая сессия начинает framing и VAD-контекст без остатков прошлой записи."""
    detector = TurnDetector(TurnSettings(min_speech_ms=32, barge_in_speech_ms=32, pre_roll_ms=32), max_turn_seconds=30)
    detector.feed(SILENCE[:300])
    detector.abort()
    detector.reset()
    assert detector.feed(SPEECH) == [TurnEvent("start", SPEECH)]


@pytest.mark.usefixtures("controlled_vad")
def test_two_ordinary_turns_rearm_without_replaying_previous_audio() -> None:
    """Подтверждённая пауза позволяет следующую реплику с независимым pre-roll."""
    detector = TurnDetector(TurnSettings(min_speech_ms=32, barge_in_speech_ms=32, end_silence_ms=64, pre_roll_ms=32), max_turn_seconds=30)
    first = detector.feed(SPEECH * 2 + SILENCE * 2)
    second = detector.feed(SPEECH * 3 + SILENCE * 2)
    assert [event.kind for event in first] == ["start", "audio", "audio", "audio", "end"]
    assert [event.kind for event in second] == ["start", "audio", "audio", "audio", "audio", "end"]
    assert audio_bytes(first) == SPEECH * 2 + SILENCE * 2
    assert audio_bytes(second) == SPEECH * 3 + SILENCE * 2


def test_real_cpu_vad_does_not_start_on_silence_or_quiet_noise() -> None:
    """Настоящая модель на CPU отклоняет тишину и тихий воспроизводимый шум."""
    detector = TurnDetector(TurnSettings(), max_turn_seconds=30)
    assert detector.feed(SILENCE * 64) == []
    rng = random.Random(42)
    noise = array("h", (rng.randint(-300, 300) for _ in range(512 * 64))).tobytes()
    assert detector.feed(noise) == []


@pytest.mark.usefixtures("controlled_vad")
def test_maximum_duration_includes_and_crops_pre_roll() -> None:
    """Длинный pre-roll не позволяет экспортировать больше жёсткого лимита PCM."""
    detector = TurnDetector(TurnSettings(min_speech_ms=32, barge_in_speech_ms=32, end_silence_ms=64, pre_roll_ms=320), max_turn_seconds=0.096)
    assert detector.feed(SILENCE * 10) == []
    events = detector.feed(SPEECH)
    assert events == [TurnEvent("start", SILENCE * 2 + SPEECH), TurnEvent("end", reason="limit")]


@pytest.mark.parametrize("maximum", [0.001, 0.031, float("inf"), float("nan")])
def test_maximum_duration_must_fit_one_finite_vad_window(maximum: float) -> None:
    """Недопустимый лимит отвергается до загрузки модели и экспорта лишнего аудио.

    :param maximum: Невыполнимый или бесконечный предел длительности, с.
    """
    with pytest.raises(ValueError, match="32 мс"):
        TurnDetector(TurnSettings(), max_turn_seconds=maximum)
