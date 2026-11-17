"""Проверяет фоновые записи lifecycle TTS без загрузки нейросетевых весов."""

import asyncio
import importlib.util
import io
import os
import sys
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class ThreadRecordingStream(io.StringIO):
    """Запоминает потоки, которые действительно записывали сообщения в stderr."""

    def __init__(self) -> None:
        """Инициализирует журнал и идентификаторы потоков записи."""
        super().__init__()
        self.writers: list[int] = []

    def write(self, value: str) -> int:
        """Сохраняет поток-писатель вместе с текстом записи.

        :param value: Готовая строка журнала.
        :return: Число записанных символов.
        """
        self.writers.append(threading.get_ident())
        return super().write(value)

    def fileno(self) -> int:
        """Сохраняет реальный stderr для регистрации faulthandler на Linux.

        :return: Дескриптор исходного стандартного потока ошибок.
        """
        assert sys.__stderr__ is not None
        return sys.__stderr__.fileno()


class LoggingTests(unittest.TestCase):
    """Проверяет очередь журнала средствами стандартного модельного окружения."""

    def test_lifespan_flushes_background_logs_to_stderr(self) -> None:
        """TTS сбрасывает startup/shutdown из фонового потока до возврата lifespan."""
        directory = Path(__file__).resolve().parents[1]
        backend = ModuleType("backend")
        backend.REFERENCE_PREFIX = "test-prefix"
        backend.REFERENCE_TEXT = "PRIVATE_REFERENCE_TEXT"
        backend.CosyVoiceEngine = lambda **kwargs: SimpleNamespace()
        stream = ThreadRecordingStream()
        caller = threading.get_ident()

        # Подмена только загрузчика весов сохраняет настоящий lifecycle, очередь и stderr sink.
        with (
            patch.object(sys, "path", [str(directory), *sys.path]),
            patch.dict(sys.modules, {"backend": backend}),
            patch.dict(os.environ, {"TTS_REFERENCE_WAV": "", "TTS_REFERENCE_TEXT": "", "RUNTIME_LOG_LEVEL": "INFO"}),
            patch.object(sys, "stderr", stream),
        ):
            spec = importlib.util.spec_from_file_location("tts_logging_under_test", directory / "server.py")
            assert spec is not None and spec.loader is not None
            runtime = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runtime)

            async def scenario() -> None:
                """Запускает настоящий lifespan с заменой только загрузки модели."""
                async with runtime.lifespan(runtime.app):
                    await asyncio.sleep(0)

            asyncio.run(scenario())
        output = stream.getvalue()
        self.assertGreaterEqual(output.count("TTS"), 2)
        self.assertNotIn("PRIVATE_REFERENCE_TEXT", output)
        self.assertTrue(stream.writers)
        self.assertTrue(all(writer != caller for writer in stream.writers))
