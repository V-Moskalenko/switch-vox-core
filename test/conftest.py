"""Изолирует настройки тестов от пользовательского окружения и файлов проекта."""

import os

import pytest


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Удаляет только переменные групп настроек на время проверки.

    :param monkeypatch: Восстановление исходного окружения после теста.
    """
    groups = {"api", "audio", "turn", "inference", "loging_settings"}
    for name in os.environ:
        if name.lower().split("__", maxsplit=1)[0] in groups:
            monkeypatch.delenv(name)
