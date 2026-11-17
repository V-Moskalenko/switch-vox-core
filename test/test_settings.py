"""Проверяет источники конфигурации, аудиолимиты и ограничения локального MVP."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import orjson
import pytest
from pydantic import BaseModel, ValidationError

from settings import PROJECT_ROOT, ApiSettings, AudioSettings, InferenceSettings, LogingSettings, Settings, TurnSettings


def test_default_settings() -> None:
    """Проверяет локальный запуск и согласованный PCM без внешних моделей."""
    settings = Settings(_env_file=None)
    assert settings.api.host == "127.0.0.1"
    assert settings.api.workers == settings.api.max_sessions == 1
    assert settings.inference.mode == "cascade"
    assert settings.audio.input_sample_rate == 16000
    assert settings.audio.output_sample_rate == 24000
    assert settings.audio.channels == 1
    assert settings.audio.sample_width_bytes == 2
    assert settings.audio.frame_ms == 20
    assert settings.audio.max_frame_ms == 100
    assert settings.audio.max_turn_seconds == 30
    assert settings.audio.max_output_buffer_seconds == 2
    assert settings.loging_settings.file_path is None


def test_documented_env_example_starts_with_valid_settings() -> None:
    """Проверяет загрузку всех значений файла, который пользователь копирует в .env."""
    settings = Settings(_env_file=PROJECT_ROOT / ".env.example")
    assert settings.api.workers == settings.api.max_sessions == 1
    assert settings.audio.input_sample_rate == 16000
    assert settings.audio.output_sample_rate == 24000
    assert settings.audio.channels == 1
    assert settings.audio.sample_width_bytes == 2


def test_dotenv_uses_project_root_from_another_cwd(tmp_path: Path) -> None:
    """Проверяет реальное определение корня без изменения пользовательского .env.

    :param tmp_path: Временная копия модуля и рабочий каталог дочернего процесса.
    """
    project = tmp_path / "проект"
    source = project / "sources"
    source.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / "sources" / "settings.py", source / "settings.py")
    (project / ".env").write_text("API__PORT=8123\nLOGING_SETTINGS__FILE_PATH=logs/проверка.log\nUNRELATED=ignored\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    environment = dict(os.environ, PYTHONPATH=str(source), INFERENCE__MODE="cascade")
    command = "from settings import config; print(config.api.port); print(config.loging_settings.file_path.as_posix())"
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", command], cwd=elsewhere, env=environment, text=True, encoding="utf-8", capture_output=True, timeout=15, check=True)
    assert result.stdout.splitlines() == ["8123", "logs/проверка.log"]
    assert not (project / "logs").exists()
    assert not (elsewhere / "logs").exists()


def test_environment_overrides_nested_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверяет приоритет env, смешанный регистр и вложенные JSON-значения.

    :param tmp_path: Каталог временного файла конфигурации.
    :param monkeypatch: Изолированное окружение процесса.
    """
    dotenv = tmp_path / ".env"
    dotenv.write_text("API__PORT=8123\nINFERENCE__MODE=cascade\nUNKNOWN=ignored\n", encoding="utf-8")
    monkeypatch.setenv("api__port", "9123")
    monkeypatch.setenv("inference", orjson.dumps({"mode": "e2e", "connect_timeout_seconds": 7}).decode("utf-8"))
    monkeypatch.setenv("INFERENCE__CONNECT_TIMEOUT_SECONDS", "3")
    settings = Settings(_env_file=dotenv)
    assert settings.api.port == 9123
    assert settings.inference.mode == "e2e"
    assert settings.inference.connect_timeout_seconds == 3


@pytest.mark.parametrize("values", [{"workers": 0}, {"workers": 2}, {"max_sessions": 2}, {"port": 0}, {"port": 65536}])
def test_invalid_api_limits(values: dict[str, int]) -> None:
    """Отклоняет нерабочий порт и несколько независимых владельцев сессии.

    :param values: Недопустимое значение одного из API-ограничений.
    """
    with pytest.raises(ValidationError):
        ApiSettings.model_validate(values)


@pytest.mark.parametrize(
    "values",
    [
        {"input_sample_rate": 48000},
        {"output_sample_rate": 16000},
        {"channels": 2},
        {"sample_width_bytes": 4},
        {"frame_ms": 0},
        {"frame_ms": 40, "max_frame_ms": 20},
        {"max_frame_ms": 101},
        {"max_turn_seconds": 0},
        {"max_turn_seconds": 0.031},
        {"max_output_buffer_seconds": 0},
        {"max_output_buffer_seconds": float("inf")},
    ],
)
def test_invalid_audio_limits(values: dict[str, int | float]) -> None:
    """Отклоняет неверный PCM и неограниченные либо противоречивые буферы.

    :param values: Невалидные настройки аудиоконтракта.
    """
    with pytest.raises(ValidationError):
        AudioSettings.model_validate(values)


@pytest.mark.parametrize(
    "values",
    [
        {"mode": "unknown"},
        {"connect_timeout_seconds": 0},
        {"idle_timeout_seconds": -1},
        {"first_response_timeout_seconds": float("inf")},
        {"llm_base_url": "file:///models"},
        {"max_response_tokens": 0},
        {"max_history_turns": 0},
        {"max_phrase_characters": 501},
        {"phrase_queue_size": 0},
    ],
)
def test_invalid_inference_settings(values: dict[str, str | float]) -> None:
    """Отклоняет неизвестный режим, бесконечные ожидания и неподходящий URL.

    :param values: Невалидные параметры выбранного модельного контура.
    """
    with pytest.raises(ValidationError):
        InferenceSettings.model_validate(values)


@pytest.mark.parametrize("model", [Settings, ApiSettings, AudioSettings, TurnSettings, InferenceSettings, LogingSettings])
def test_settings_have_russian_descriptions(model: type[BaseModel]) -> None:
    """Проверяет опубликованную JSON Schema каждой группы и её полей.

    :param model: Проверяемая модель настроек.
    """
    schema = model.model_json_schema()
    assert re.search("[А-Яа-яЁё]", schema["description"])
    for field in schema["properties"].values():
        assert re.search("[А-Яа-яЁё]", field["description"])


def test_settings_import_has_no_model_runtime() -> None:
    """Проверяет импорт в чистом процессе без Torch/vLLM/Transformers."""
    command = "import sys; import settings; assert not {'torch', 'vllm', 'transformers'} & sys.modules.keys(); print(settings.config.api.workers)"
    environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT / "sources"), INFERENCE__MODE="cascade")
    result = subprocess.run([sys.executable, "-c", command], env=environment, capture_output=True, text=True, timeout=15, check=True)
    assert result.stdout.strip() == "1"


def test_valid_boundary_settings() -> None:
    """Разрешает допустимые границы порта, кадра и альтернативный режим."""
    assert ApiSettings(port=65535).port == 65535
    audio = AudioSettings(frame_ms=100, max_frame_ms=100)
    assert audio.frame_ms == audio.max_frame_ms
    inference = InferenceSettings(mode="e2e", connect_timeout_seconds=0.5)
    assert inference.mode == "e2e"
    assert inference.connect_timeout_seconds == 0.5
