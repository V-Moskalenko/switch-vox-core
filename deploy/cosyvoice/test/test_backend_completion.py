"""Проверяет завершение реального backend generator без загрузки весов и CUDA."""

import queue
import threading
import time
import unittest
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from backend import CancellableCosyVoice3Model
from streaming import AudioStream


class BackendCompletionTests(unittest.TestCase):
    """Сохраняет настоящий stream lifecycle, заменяя только нейросетевые вычисления."""

    def make_model(self, stopped: threading.Event | None = None) -> CancellableCosyVoice3Model:
        """Создаёт CPU-модель с детерминированными speech tokens и PCM.

        :param stopped: При наличии LLM ждёт отмены и подтверждает своё завершение.
        :return: Реальный streaming backend без весов, frontend и CUDA.
        """
        model = CancellableCosyVoice3Model.__new__(CancellableCosyVoice3Model)
        model.tts_speech_token_dict = {}
        model.llm_end_dict = {}
        model.hift_cache_dict = {}
        model._flow_idle = threading.Event()
        model._flow_idle.set()
        model.flow = SimpleNamespace(pre_lookahead_len=3)

        def generate(inputs: dict[str, torch.Tensor], request_id: str, cancel: threading.Event, failures: queue.Queue[Exception], worker_cancel: threading.Event | None = None) -> None:
            """Публикует завершённую последовательность вместо тяжёлой LLM.

            :param inputs: Неиспользуемые features CPU-пробы.
            :param request_id: Ключ настоящего lifecycle модели.
            :param cancel: Сигнал внешней отмены.
            :param failures: Канал ошибок настоящего backend.
            :param worker_cancel: Остановка LLM при завершении или ошибке flow.
            """
            model.tts_speech_token_dict[request_id].extend(range(60))
            if stopped is not None:
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    if cancel.is_set() or (worker_cancel is not None and worker_cancel.is_set()):
                        stopped.set()
                        break
                    time.sleep(0.001)
            model.llm_end_dict[request_id] = True

        def token2wav(*args: object, **kwargs: object) -> torch.Tensor:
            """Заменяет flow/vocoder коротким PCM-блоком.

            :param args: Аргументы закреплённого token2wav.
            :param kwargs: Параметры streaming/finalize.
            :return: Детерминированные CPU audio samples.
            """
            return torch.ones(1, 480)

        model._generate_tokens = generate
        model.token2wav = token2wav
        return model

    def test_completed_backend_closes_pcm_stream_without_idle_timeout(self) -> None:
        """Нормальный EOF доходит через настоящий AudioStream после последнего PCM."""
        model = self.make_model()
        inputs = {key: torch.zeros(1, 0) for key in ["flow_prompt_speech_token", "prompt_speech_feat", "flow_embedding"]}

        def synthesize(cancel: threading.Event) -> Iterator[bytes]:
            """Пропускает реальный backend generator через тот же worker, что HTTP.

            :param cancel: Общий с HTTP-мостом сигнал отмены.
            :yield: Реальные CPU samples, упакованные в PCM16.
            """
            for audio in model.stream(inputs, cancel):
                yield audio.numpy().astype("<i2").tobytes()

        stream = AudioStream(synthesize)
        stream.start()
        chunks = []
        deadline = time.monotonic() + 3
        try:
            while time.monotonic() < deadline:
                try:
                    chunk = stream.read(timeout=0.02)
                except queue.Empty:
                    continue
                if chunk is None:
                    self.assertTrue(chunks)
                    self.assertFalse(stream.cancel.is_set(), "Успешный EOF не является внешней отменой")
                    return
                chunks.append(chunk)
            self.fail("Backend завершился, но HTTP-мост не получил EOF после PCM")
        finally:
            stream.close()

    def test_profile_orders_first_token_flow_and_cpu_audio_for_both_paths(self) -> None:
        """Профиль различает первый streaming chunk и финал короткой реплики без CUDA."""
        for token_count in (8, 60):
            with self.subTest(token_count=token_count):
                model = self.make_model()

                def tokens(count: int = token_count, target: CancellableCosyVoice3Model = model, **kwargs: object) -> Iterator[int]:
                    """Заменяет только нейросеть, сохраняя настоящий token-worker.

                    :param count: Число токенов данного сценария, закреплённое до запуска worker.
                    :param target: Модель, чей общий lock защищает нейросетевой token step.
                    :param kwargs: Параметры закреплённого speech-token inference.
                    :yield: Конечные речевые токены без пауз и GPU.
                    """
                    for token in range(count):
                        self.assertTrue(target._compute_lock.locked())
                        yield token

                model._generate_tokens = CancellableCosyVoice3Model._generate_tokens.__get__(model)
                model.llm = SimpleNamespace(inference=tokens)
                model.llm_context = nullcontext()
                model.device = torch.device("cpu")
                model.fp16 = False
                model.silent_tokens = set()
                original_flow = model.token2wav

                def checked_flow(*args: object, target: CancellableCosyVoice3Model = model, flow: Callable[..., torch.Tensor] = original_flow, **kwargs: object) -> torch.Tensor:
                    """Проверяет общий compute-lock на обеих ветках первого flow.

                    :param args: Входы настоящего token2wav.
                    :param target: Модель с общим lock этого запроса.
                    :param flow: Нейросетевой stub текущего сценария.
                    :param kwargs: Признаки streaming/finalize.
                    :return: Непустое CPU-аудио тестового flow.
                    """
                    self.assertTrue(target._compute_lock.locked())
                    self.assertFalse(target._flow_idle.is_set())
                    return flow(*args, **kwargs)

                model.token2wav = checked_flow
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
                self.assertTrue(list(model.stream(inputs, threading.Event())))
                progress = model.progress
                self.assertLessEqual(0, progress["first_token_us"])
                self.assertLessEqual(progress["first_token_us"], progress["first_flow_us"])
                self.assertLessEqual(progress["first_flow_us"], progress["first_audio_us"])
                self.assertLessEqual(progress["first_audio_us"], progress["finished_us"])
                self.assertLessEqual(progress["first_token_us"], progress["llm_done_us"])
                self.assertEqual(progress["first_flow_finalize"], int(token_count == 8))
                self.assertEqual(progress["buffered_tokens"], token_count)
                self.assertFalse(model.tts_speech_token_dict)

    def test_cancelled_profile_does_not_retain_previous_audio_timings(self) -> None:
        """Новая отменённая реплика не получает временные отметки прошлого ответа."""
        model = self.make_model()
        inputs = {key: torch.zeros(1, 0) for key in ["flow_prompt_speech_token", "prompt_speech_feat", "flow_embedding"]}
        self.assertTrue(list(model.stream(inputs, threading.Event())))
        cancel = threading.Event()
        cancel.set()
        self.assertFalse(list(model.stream(inputs, cancel)))
        self.assertNotIn("first_flow_us", model.progress)
        self.assertNotIn("first_audio_us", model.progress)
        self.assertIn("finished_us", model.progress)

    def test_cancel_waiting_for_flow_does_not_start_another_token_step(self) -> None:
        """Worker видит отмену во время заблокированного flow и не запускает новый шаг."""
        model = self.make_model()
        model._compute_lock = threading.Lock()
        cancel = threading.Event()
        stopped = threading.Event()
        source_called = threading.Event()
        errors: queue.Queue[Exception] = queue.Queue()

        def source() -> Iterator[int]:
            """Отмечает нежелательный вызов нейросетевого шага после отмены.

            :yield: Единственный токен, до которого отмена не должна допустить.
            """
            source_called.set()
            yield 1

        def read_tokens() -> None:
            """Имитирует token-worker, пока flow держит общий lock."""
            try:
                list(model._speech_tokens(source(), cancel, threading.Event()))
            except Exception as exc:
                errors.put(exc)
            finally:
                stopped.set()

        model._compute_lock.acquire()
        worker = threading.Thread(target=read_tokens, daemon=True)
        worker.start()
        try:
            cancel.set()
            self.assertTrue(stopped.wait(timeout=0.5))
            self.assertTrue(errors.empty(), list(errors.queue))
            self.assertFalse(source_called.is_set())
        finally:
            model._compute_lock.release()
            worker.join(timeout=1)

    def test_internal_cancel_closes_open_tokens_without_waiting_for_flow(self) -> None:
        """Остановка flow освобождает живой upstream generator/KV, пока compute-lock занят."""
        model = self.make_model()
        model._compute_lock = threading.Lock()
        closed = threading.Event()
        stopped = threading.Event()
        worker_cancel = threading.Event()
        steps = []
        errors: queue.Queue[Exception] = queue.Queue()

        def source() -> Iterator[int]:
            """Сохраняет факт освобождения уже начатого генератора.

            :yield: Два токена; после первого должна сработать внутренняя отмена.
            """
            try:
                for token in (1, 2):
                    steps.append(token)
                    yield token
            finally:
                closed.set()

        tokens = model._speech_tokens(source(), threading.Event(), worker_cancel)
        self.assertEqual(next(tokens), 1)

        def finish_tokens() -> None:
            """Дочитывает поток с отменой, не дожидаясь освобождения flow-lock."""
            try:
                list(tokens)
            except Exception as exc:
                errors.put(exc)
            finally:
                stopped.set()

        model._compute_lock.acquire()
        worker = threading.Thread(target=finish_tokens, daemon=True)
        worker.start()
        try:
            worker_cancel.set()
            self.assertTrue(stopped.wait(timeout=0.5))
            self.assertTrue(errors.empty(), list(errors.queue))
            self.assertTrue(closed.is_set())
            self.assertEqual(steps, [1])
        finally:
            model._compute_lock.release()
            worker.join(timeout=1)

    def test_cancel_waiting_for_token_step_does_not_start_a_new_flow(self) -> None:
        """Отмена во время ожидания token step запрещает следующий flow-вызов."""
        model = self.make_model()
        model._compute_lock = threading.Lock()
        model._stream_started_ns = time.perf_counter_ns()
        model.progress = {"flow_chunks": 0}
        called = threading.Event()
        started = threading.Event()
        cancel = threading.Event()
        outputs = []
        errors: queue.Queue[Exception] = queue.Queue()

        def flow(*args: object, **kwargs: object) -> torch.Tensor:
            """Отмечает нежелательный запуск новой GPU-работы после отмены.

            :param args: Тензоры и cache ID запроса.
            :param kwargs: Признаки stream/finalize.
            :return: Непустой блок только при ошибочном вызове.
            """
            called.set()
            return torch.ones(1, 480)

        def read_audio() -> None:
            """Ждёт завершения текущего token step в общем compute-lock."""
            started.set()
            try:
                inputs = {key: torch.zeros(1, 0) for key in ("flow_prompt_speech_token", "prompt_speech_feat", "flow_embedding")}
                outputs.append(model._audio_chunk(inputs, torch.zeros(1, 0), 0, "cancelled", cancel, finalize=False))
            except Exception as exc:
                errors.put(exc)

        model.token2wav = flow
        model._compute_lock.acquire()
        worker = threading.Thread(target=read_audio, daemon=True)
        worker.start()
        try:
            self.assertTrue(started.wait(timeout=1))
            cancel.set()
        finally:
            model._compute_lock.release()
            worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertTrue(errors.empty(), list(errors.queue))
        self.assertFalse(called.is_set())
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].numel(), 0)
        self.assertNotIn("first_flow_us", model.progress)

    def test_closing_backend_stops_internal_token_worker(self) -> None:
        """Закрытие generator до конца синтеза останавливает LLM и очищает buffers."""
        stopped = threading.Event()
        model = self.make_model(stopped)
        inputs = {key: torch.zeros(1, 0) for key in ["flow_prompt_speech_token", "prompt_speech_feat", "flow_embedding"]}
        cancel = threading.Event()
        chunks = model.stream(inputs, cancel)
        self.assertTrue(next(chunks).numel())
        cancel.set()
        chunks.close()
        self.assertTrue(stopped.is_set())
        self.assertFalse(model.tts_speech_token_dict)
        self.assertFalse(model.llm_end_dict)
        self.assertFalse(model.hift_cache_dict)

    def test_flow_failure_reaches_reader_and_stops_token_worker(self) -> None:
        """Ошибка flow передаётся HTTP-мосту, даже если для остановки LLM нужна отмена."""
        stopped = threading.Event()
        model = self.make_model(stopped)
        inputs = {key: torch.zeros(1, 0) for key in ["flow_prompt_speech_token", "prompt_speech_feat", "flow_embedding"]}
        failure = RuntimeError("Наблюдаемая ошибка flow")

        def token2wav(*args: object, **kwargs: object) -> torch.Tensor:
            """Воспроизводит отказ нейросетевого flow до появления первого PCM.

            :param args: Аргументы закреплённого token2wav.
            :param kwargs: Параметры streaming/finalize.
            :return: Аудио не появляется из-за ошибки.
            :raises RuntimeError: Наблюдаемая ошибка модели.
            """
            raise failure

        def synthesize(cancel: threading.Event) -> Iterator[bytes]:
            """Передаёт реальный отказ backend в настоящий HTTP-worker.

            :param cancel: Сигнал внешней отмены.
            :yield: PCM отсутствует из-за ошибки flow.
            """
            for audio in model.stream(inputs, cancel):
                yield audio.numpy().astype("<i2").tobytes()

        model.token2wav = token2wav
        stream = AudioStream(synthesize)
        stream.start()
        try:
            with self.assertRaises(RuntimeError) as raised:
                stream.read(timeout=3)
            self.assertIs(raised.exception, failure)
            self.assertTrue(stopped.is_set())
            self.assertFalse(model.tts_speech_token_dict)
            self.assertFalse(model.hift_cache_dict)
        finally:
            stream.close()

    def test_client_cancel_stops_tokens_while_flow_is_blocked(self) -> None:
        """Внешняя отмена доходит до LLM независимо от завершения медленного flow."""
        stopped = threading.Event()
        flow_started = threading.Event()
        release_flow = threading.Event()
        model = self.make_model(stopped)
        inputs = {key: torch.zeros(1, 0) for key in ["flow_prompt_speech_token", "prompt_speech_feat", "flow_embedding"]}

        def token2wav(*args: object, **kwargs: object) -> torch.Tensor:
            """Блокирует только flow, оставляя token-worker доступным для отмены.

            :param args: Аргументы закреплённого token2wav.
            :param kwargs: Параметры streaming/finalize.
            :return: Короткий блок после разрешения тестового flow.
            """
            flow_started.set()
            release_flow.wait(timeout=3)
            return torch.ones(1, 480)

        def synthesize(cancel: threading.Event) -> Iterator[bytes]:
            """Передаёт внешний cancel из настоящего HTTP-моста.

            :param cancel: Сигнал отключившегося клиента.
            :yield: PCM только после завершения flow.
            """
            for audio in model.stream(inputs, cancel):
                yield audio.numpy().astype("<i2").tobytes()

        model.token2wav = token2wav
        stream = AudioStream(synthesize)
        stream.start()
        try:
            self.assertTrue(flow_started.wait(timeout=1))
            stream.cancel.set()
            self.assertTrue(stopped.wait(timeout=0.5), "LLM не увидел отмену, пока flow заблокирован")
        finally:
            release_flow.set()
            stream.close()


if __name__ == "__main__":
    unittest.main()
