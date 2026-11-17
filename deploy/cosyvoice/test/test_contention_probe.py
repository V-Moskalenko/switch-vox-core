"""Проверки диагностического сравнения flow с замороженным и активным token-worker."""

import importlib.util
import threading
import time
import unittest
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from backend import CancellableCosyVoice3Model


def load_probe() -> ModuleType:
    """Загружает общий CPU-сценарий конкурентной генерации из тестового пакета.

    :return: Модуль пробы, не запускающий main при импорте.
    """
    path = Path(__file__).with_name("contention_fixture.py")
    spec = importlib.util.spec_from_file_location("probe_tts_contention_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContentionProbeTests(unittest.TestCase):
    """Сохраняет настоящий token-worker, заменяя только нейросети на CPU."""

    def test_trial_controls_overlap_and_keeps_flow_tokens_unchanged(self) -> None:
        """Проверяет парковку, активное перекрытие и cleanup при одинаковых flow-токенах."""
        for active in (False, True):
            with self.subTest(active=active):
                self.check_trial(active)

    def check_trial(self, active: bool) -> None:
        """Проверяет независимый сценарий парковки и cleanup.

        :param active: Разрешить продолжение token-worker во время flow.
        """
        probe = load_probe()
        model = CancellableCosyVoice3Model.__new__(CancellableCosyVoice3Model)
        model.device = torch.device("cpu")
        model.llm_context = nullcontext()
        model.fp16 = False
        model.silent_tokens = set()
        model.tts_speech_token_dict = {}
        model.llm_end_dict = {}
        model.hift_cache_dict = {}
        progressed = threading.Event()
        exited = threading.Event()

        def inference(**kwargs: object) -> Iterator[int]:
            """Даёт конечные токены и подтверждает реальный выход генератора.

            :param kwargs: Параметры закреплённого inference.
            :yield: Речевые токены до отмены настоящим worker.
            """
            kwargs["text_len"].add_(1)
            try:
                for token in range(100):
                    yield token
                    if token:
                        progressed.set()
                    time.sleep(0.001)
            finally:
                exited.set()

        fixed = torch.tensor([[11, 12, 13]])

        def token2wav(token: torch.Tensor, *args: object, expected_active: bool = active, **kwargs: object) -> torch.Tensor:
            """Наблюдает вход flow и продолжение параллельного реального worker.

            :param token: Один и тот же фиксированный префикс в обоих режимах.
            :param args: Prompt и embedding.
            :param expected_active: Закреплённый режим данного запуска worker.
            :param kwargs: Параметры stream/finalize.
            :return: Короткий CPU-аудиоблок вместо нейросетевого flow/vocoder.
            """
            self.assertTrue(torch.equal(token, fixed))
            self.assertTrue(kwargs["stream"])
            self.assertFalse(kwargs.get("finalize", False))
            self.assertEqual(args[3], 0)
            self.assertIsNone(model.hift_cache_dict[args[4]])
            if expected_active:
                self.assertTrue(progressed.wait(timeout=1))
            else:
                self.assertFalse(progressed.is_set())
            return torch.ones(1, 480)

        model.llm = SimpleNamespace(inference=inference)
        model.token2wav = token2wav
        keys = [
            "text",
            "text_len",
            "prompt_text",
            "prompt_text_len",
            "llm_prompt_speech_token",
            "llm_prompt_speech_token_len",
            "llm_embedding",
            "flow_prompt_speech_token",
            "prompt_speech_feat",
            "flow_embedding",
        ]
        inputs = {key: torch.zeros(1, 0) for key in keys}
        inputs["text_len"] = torch.zeros(1, dtype=torch.int64)
        result, audio = probe.run_trial(model, inputs, fixed, active=active)
        self.assertEqual(result["tokens_before"], 1)
        self.assertEqual(result["tokens_after"] > 1, active)
        self.assertTrue(torch.equal(fixed, torch.tensor([[11, 12, 13]])))
        self.assertEqual(audio.shape, (1, 480))
        self.assertEqual(inputs["text_len"].sum().item(), 0)
        self.assertTrue(exited.is_set())
        self.assertIs(model.llm.inference, inference)
        self.assertFalse(model.tts_speech_token_dict)
        self.assertFalse(model.llm_end_dict)
        self.assertFalse(model.hift_cache_dict)

        def fail_flow(*args: object, **kwargs: object) -> torch.Tensor:
            """Проверяет cleanup после исключения в flow при припаркованном worker.

            :param args: Неизменные входные тензоры.
            :param kwargs: Параметры flow.
            :return: Аудио не появляется.
            :raises RuntimeError: Управляемый отказ flow.
            """
            raise RuntimeError("Пробный отказ flow")

        exited.clear()
        model.token2wav = fail_flow
        with self.assertRaisesRegex(RuntimeError, "Пробный отказ flow"):
            probe.run_trial(model, inputs, fixed, active=False)
        self.assertTrue(exited.is_set())
        self.assertIs(model.llm.inference, inference)
        self.assertFalse(model.tts_speech_token_dict)
        self.assertFalse(model.llm_end_dict)
        self.assertFalse(model.hift_cache_dict)
        self.assertEqual(inputs["text_len"].sum().item(), 0)
