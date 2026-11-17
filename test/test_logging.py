"""Проверяет реальные Loguru sinks без утечки локальных переменных."""

import sys
from pathlib import Path
from threading import Event, Thread, get_ident

import pytest
from loguru import logger

from settings import LogingSettings, setup_logging


def test_logging_reconfiguration_and_exception_privacy(tmp_path: Path) -> None:
    """Пишет одно сообщение после повторной настройки, не раскрывая locals.

    :param tmp_path: Временный каталог журналов.
    """
    log_file = tmp_path / "nested" / "test.log"
    settings = LogingSettings(file_path=log_file)
    setup_logging(settings)
    setup_logging(settings)
    sensitive_value = "".join(["private", "value", "12345"])
    try:
        logger.info("Проверка единственного обработчика")
        try:
            len(sensitive_value) // 0
        except ZeroDivisionError:
            logger.exception("Проверка исключения без локальных переменных")
    finally:
        logger.remove()
    contents = log_file.read_text(encoding="utf-8")
    assert contents.count("Проверка единственного обработчика") == 1
    assert "ZeroDivisionError" in contents
    assert sensitive_value not in contents


def test_slow_console_does_not_block_log_producer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Вызов логгера завершается, пока реальная запись в stderr ещё заблокирована.

    :param monkeypatch: Подмена только выходного потока, без замены Loguru.
    """
    entered, release, returned = Event(), Event(), Event()
    writer_threads: list[int] = []
    producer_threads: list[int] = []

    class SlowConsole:
        """Удерживает запись до явного разрешения теста."""

        def write(self, message: str) -> int:
            """Блокирует sink независимо от вызывающего логгер потока.

            :param message: Подготовленная запись Loguru.
            :return: Число принятых символов.
            """
            writer_threads.append(get_ident())
            entered.set()
            assert release.wait(timeout=5), "Тест не освободил консоль"
            return len(message)

        def flush(self) -> None:
            """Соответствует интерфейсу stderr без дополнительной буферизации."""

    def produce() -> None:
        """Отмечает возврат из настоящего logger.info до освобождения sink."""
        producer_threads.append(get_ident())
        logger.info("Проверка очереди вывода")
        returned.set()

    monkeypatch.setattr(sys, "stderr", SlowConsole())
    setup_logging(LogingSettings())
    producer = Thread(target=produce)
    producer.start()
    try:
        assert entered.wait(timeout=2), "Обработчик не начал запись"
        assert returned.wait(timeout=1), "Медленный stderr блокирует вызывающий поток"
        assert writer_threads[0] != producer_threads[0]
    finally:
        # Освобождение sink обязательно и при RED, иначе очередь не завершит остановку.
        release.set()
        producer.join(timeout=2)
        logger.complete()
        logger.remove()
    assert not producer.is_alive()
