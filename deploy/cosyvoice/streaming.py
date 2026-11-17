"""Ограниченный мост синхронного TTS-worker в поток HTTP без накопления аудио."""

import queue
import threading
from collections.abc import Callable, Iterator


class AudioStream:
    """Передаёт PCM через ограниченную очередь и явно останавливает производителя."""

    def __init__(self, synthesize: Callable[[threading.Event], Iterator[bytes]], queue_size: int = 4) -> None:
        """Создаёт один worker; запуск выполняется отдельным вызовом.

        :param synthesize: Конкретный синхронный генератор PCM, принимающий сигнал отмены.
        :param queue_size: Максимальное число результатов в очереди; размер PCM-кадров задаёт производитель.
        """
        self.cancel = threading.Event()
        self._synthesize = synthesize
        self._queue: queue.Queue[bytes | Exception | None] = queue.Queue(maxsize=queue_size)
        self._worker = threading.Thread(target=self._produce, name="cosyvoice-synthesis", daemon=True)

    def start(self) -> None:
        """Запускает синтез без ожидания первого аудиокадра."""
        self._worker.start()

    def _put(self, value: bytes | Exception | None) -> None:
        """Передаёт результат с backpressure и проверкой отмены.

        :param value: PCM, ошибка либо маркер завершения.
        """
        while not self.cancel.is_set():
            try:
                self._queue.put(value, timeout=0.02)
                return
            except queue.Full:
                continue

    def _produce(self) -> None:
        """Производит только реальные части аудио и сохраняет ошибки модели."""
        try:
            for chunk in self._synthesize(self.cancel):
                if self.cancel.is_set():
                    break
                self._put(chunk)
        except Exception as exc:
            self._put(exc)
        finally:
            self._put(None)

    def read(self, timeout: float = 0) -> bytes | None:
        """Возвращает следующий кадр, не подменяя отказ успешным завершением.

        :param timeout: Максимальное ожидание кадра, секунды; ноль не блокирует.
        :return: PCM либо None после завершения синтеза.
        :raises queue.Empty: Результат пока не появился.
        :raises Exception: Конкретная ошибка модельного worker.
        """
        result = self._queue.get(timeout=timeout)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self, timeout: float = 2) -> None:
        """Останавливает вычисления независимо от очереди подготовленного PCM.

        :param timeout: Допустимое время завершения worker, секунды.
        :raises TimeoutError: Worker не подтвердил остановку в отведённое время.
        """
        self.cancel.set()
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            raise TimeoutError("TTS-worker не остановился после отмены")
