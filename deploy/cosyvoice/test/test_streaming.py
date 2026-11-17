"""Проверки ограниченного PCM-потока и прекращения вычислений при отмене."""

import queue
import threading
import time
import unittest
from collections.abc import Iterator

from deploy.cosyvoice.streaming import AudioStream


class AudioStreamTests(unittest.TestCase):
    """Проверяет наблюдаемое поведение очереди без загрузки CUDA-моделей."""

    def test_stream_is_visible_before_synthesis_finishes(self) -> None:
        """Первый PCM приходит, пока синтез последующих частей ещё заблокирован."""
        release = threading.Event()

        def synthesize(cancel: threading.Event) -> Iterator[bytes]:
            """Имитирует независимый результат модели перед концом инференса.

            :param cancel: Сигнал отмены производителя.
            :yield: Один тестовый PCM-кадр до и после разрешения продолжения.
            """
            yield b"\x01\x00" * 480
            release.wait(timeout=1)
            if not cancel.is_set():
                yield b"\x02\x00" * 480

        stream = AudioStream(synthesize)
        stream.start()
        self.assertEqual(stream.read(timeout=1), b"\x01\x00" * 480)
        self.assertFalse(release.is_set())
        release.set()
        self.assertEqual(stream.read(timeout=1), b"\x02\x00" * 480)
        self.assertIsNone(stream.read(timeout=1))
        stream.close()

    def test_slow_reader_is_bounded_and_cancel_stops_producer(self) -> None:
        """Медленный клиент не создаёт бесконечный буфер и не оставляет worker."""
        produced = []
        exited = threading.Event()

        def synthesize(cancel: threading.Event) -> Iterator[bytes]:
            """Генерирует PCM до отмены и отмечает завершение реального worker.

            :param cancel: Сигнал остановки тестового производителя.
            :yield: PCM до отмены.
            """
            try:
                while not cancel.is_set():
                    produced.append(1)
                    yield b"\x00\x00" * 480
            finally:
                exited.set()

        stream = AudioStream(synthesize, queue_size=4)
        stream.start()
        time.sleep(0.1)
        self.assertLessEqual(len(produced), 5)
        stream.close(timeout=1)
        self.assertTrue(exited.is_set())

    def test_model_failure_is_not_converted_to_successful_eof(self) -> None:
        """Ошибка модели сохраняется для обработки HTTP-границей."""

        def synthesize(cancel: threading.Event) -> Iterator[bytes]:
            """Выдаёт часть PCM, затем моделирует отказ инференса.

            :param cancel: Сигнал отмены, не успевающий наступить до отказа.
            :yield: Единственный PCM-кадр перед ошибкой.
            """
            yield b"\x01\x00" * 480
            raise RuntimeError("Ошибка модельного runtime")

        stream = AudioStream(synthesize)
        stream.start()
        self.assertEqual(len(stream.read(timeout=1)), 960)
        with self.assertRaisesRegex(RuntimeError, "Ошибка модельного runtime"):
            stream.read(timeout=1)
        stream.close()

    def test_cancel_before_first_audio_does_not_wait_for_output(self) -> None:
        """Отмена активирует сигнал worker даже до появления первого PCM."""
        started = threading.Event()
        stopped = threading.Event()

        def synthesize(cancel: threading.Event) -> Iterator[bytes]:
            """Ожидает отмену во время подготовки первого результата.

            :param cancel: Сигнал остановки вместо ожидания PCM.
            :yield: Аудио не появляется, поскольку подготовка отменена.
            """
            started.set()
            cancel.wait(timeout=5)
            stopped.set()
            return
            yield b""

        stream = AudioStream(synthesize)
        stream.start()
        self.assertTrue(started.wait(timeout=1))
        with self.assertRaises(queue.Empty):
            stream.read()
        stream.close(timeout=1)
        self.assertTrue(stopped.is_set())


if __name__ == "__main__":
    unittest.main()
