"""Проверяет остановку TTS при разрыве соединения до первого байта body."""

import asyncio
import threading
import unittest
from collections.abc import Iterator

import server
from fastapi import Request
from schemas import SynthesisRequest


class EarlyDisconnectTests(unittest.IsolatedAsyncioTestCase):
    """Проверяет HTTP-lifecycle с реальным worker и отказом отправки headers."""

    async def test_header_send_failure_stops_worker_and_releases_slot(self) -> None:
        """Разрыв до запуска body generator не оставляет prefill и busy slot."""
        started = threading.Event()
        stopped = threading.Event()
        cancel_events = []

        class WaitingEngine:
            """Заменяет только тяжёлый inference, сохраняя настоящий worker."""

            def synthesize(self, text: str, cancel: threading.Event) -> Iterator[bytes]:
                """Ждёт отмены до первого PCM.

                :param text: Валидная фраза запроса.
                :param cancel: Настоящий сигнал остановки stream worker.
                :yield: PCM отсутствует при отменённом prefill.
                """
                cancel_events.append(cancel)
                started.set()
                cancel.wait(timeout=5)
                stopped.set()
                return
                yield b""

        async def receive() -> dict[str, object]:
            """Ожидает событие ASGI до отмены тестового соединения.

            :return: Не возникает: операция отменяется response lifecycle.
            """
            await asyncio.Future()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            """Моделирует разрыв уже при отправке первого заголовка.

            :param message: Событие http.response.start.
            """
            raise BrokenPipeError("Клиент отключился до HTTP body")

        server.app.state.engine = WaitingEngine()
        server.app.state.busy = False
        scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
        response = await server.synthesize(SynthesisRequest(text="Привет"), Request(scope))
        self.assertTrue(started.wait(timeout=1))
        try:
            with self.assertRaisesRegex(Exception, "TaskGroup|Клиент отключился"):
                await response(scope, receive, send)
            self.assertTrue(stopped.is_set())
            self.assertFalse(server.app.state.busy)
        finally:
            for cancel in cancel_events:
                cancel.set()
            stopped.wait(timeout=1)
            del server.app.state.engine
            server.app.state.busy = False


if __name__ == "__main__":
    unittest.main()
