"""Проверяет конфигурацию пары reference WAV/transcript до загрузки GPU-весов."""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import server
from backend import REFERENCE_TEXT


def make_engine(model_dir: Path, reference: Path, reference_text: str = REFERENCE_TEXT) -> SimpleNamespace:
    """Заменяет только загрузку весов с обычным связыванием аргументов Python.

    :param model_dir: Каталог модели.
    :param reference: WAV фиксируемого голоса.
    :param reference_text: Полный prompt CV3 с точным transcript референса.
    :return: Параметры вызова без загрузки весов и выполнения GPU-операций.
    """
    return SimpleNamespace(model_dir=model_dir, reference=reference, reference_text=reference_text)


class ReferenceSettingsTests(unittest.IsolatedAsyncioTestCase):
    """Не подменяет lifespan, заменяет только тяжёлую загрузку модели."""

    async def test_default_reference_keeps_existing_voice(self) -> None:
        """Отсутствующие и пустые переменные окружения выбирают штатный образец голоса CosyVoice."""
        for environment in ({}, {"TTS_REFERENCE_WAV": "", "TTS_REFERENCE_TEXT": ""}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), patch("backend.CosyVoiceEngine", side_effect=make_engine):
                async with server.lifespan(server.app):
                    self.assertFalse(server.app.state.busy)
                    loaded = server.app.state.engine
                self.assertEqual(loaded.reference, Path("/opt/CosyVoice/asset/zero_shot_prompt.wav"))
                self.assertEqual(loaded.reference_text, "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。")

    async def test_custom_reference_preserves_required_prefix(self) -> None:
        """Русский transcript передаётся с обязательным CV3 endofprompt prefix."""
        environment = {"TTS_REFERENCE_WAV": "/models/fixtures/reference.wav", "TTS_REFERENCE_TEXT": "Привет. Это русский голос."}
        with patch.dict(os.environ, environment, clear=True), patch("backend.CosyVoiceEngine", side_effect=make_engine):
            async with server.lifespan(server.app):
                loaded = server.app.state.engine
            self.assertEqual(loaded.reference, Path(environment["TTS_REFERENCE_WAV"]))
            self.assertEqual(loaded.reference_text, "You are a helpful assistant.<|endofprompt|>Привет. Это русский голос.")

    async def test_partial_reference_is_rejected_before_model_loading(self) -> None:
        """WAV без transcript и transcript без WAV не дают несовпадающую пару модели."""
        for environment in ({"TTS_REFERENCE_WAV": "/models/fixtures/reference.wav"}, {"TTS_REFERENCE_TEXT": "Русский текст."}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), patch("backend.CosyVoiceEngine") as engine:
                with self.assertRaisesRegex(ValueError, "reference|референс"):
                    async with server.lifespan(server.app):
                        self.fail("Частичная настройка reference принята")
                engine.assert_not_called()
