"""Проверяет конечные пороги CPU VAD и сохранение начала подтверждаемой речи."""

import pytest
from pydantic import ValidationError

from settings import Settings, TurnSettings


def test_turn_defaults_preserve_confirmed_speech_in_pre_roll() -> None:
    """Pre-roll покрывает подтверждение речи и более строгий порог перебивания."""
    settings = TurnSettings()
    assert settings.pre_roll_ms >= max(settings.min_speech_ms, settings.barge_in_speech_ms)
    assert settings.end_silence_ms == 640
    assert settings.speech_threshold == 0.6
    assert isinstance(Settings(_env_file=None).turn, TurnSettings)
    assert TurnSettings(pre_roll_ms=128, min_speech_ms=96, barge_in_speech_ms=128).pre_roll_ms == 128


@pytest.mark.parametrize(
    "values",
    [
        {"speech_threshold": 0},
        {"speech_threshold": 1.01},
        {"speech_threshold": float("nan")},
        {"min_speech_ms": 0},
        {"barge_in_speech_ms": 0},
        {"end_silence_ms": 0},
        {"pre_roll_ms": 0},
        {"pre_roll_ms": 96, "min_speech_ms": 128, "barge_in_speech_ms": 64},
        {"pre_roll_ms": 96, "min_speech_ms": 64, "barge_in_speech_ms": 128},
    ],
)
def test_invalid_turn_limits(values: dict[str, int | float]) -> None:
    """Отклоняет неограниченное ожидание и потерю начала речи до подтверждения VAD.

    :param values: Невалидная группа параметров VAD.
    """
    with pytest.raises(ValidationError):
        TurnSettings.model_validate(values)
