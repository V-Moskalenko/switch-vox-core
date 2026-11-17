"""Проверяет живой ASR: раннюю гипотезу, EOF, отмену и отказ потока микрофона."""

import asyncio
import base64
from collections.abc import AsyncGenerator
from contextlib import suppress

import orjson
import pytest
from pydantic import AnyHttpUrl
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from clients.asr_client import ASRClient
from services.session_service import SessionError
from settings import Settings


def settings_for(port: int) -> Settings:
    """Настраивает конкретный ASR endpoint тестового сервера.

    :param port: Случайный TCP-порт локального WebSocket сервера.
    :return: Отдельная конфигурация без .env.
    """
    settings = Settings(_env_file=None)
    settings.inference.asr_base_url = AnyHttpUrl(f"http://127.0.0.1:{port}")
    return settings


async def handshake(socket: ServerConnection) -> None:
    """Проверяет стандартное открытие одной ASR-реплики.

    :param socket: Настоящее соединение с concrete model client.
    """
    await socket.send(orjson.dumps({"type": "session.created"}).decode("utf-8"))
    assert orjson.loads(await socket.recv()) == {"type": "session.update", "model": "qwen3-asr"}
    assert orjson.loads(await socket.recv()) == {"type": "input_audio_buffer.commit"}


def test_hypothesis_arrives_before_live_audio_eof_and_commit_is_last() -> None:
    """Результат появляется до EOF, а большой preroll не нарушает размер фреймов."""

    async def scenario() -> None:
        """Удерживает живой источник открытым после первого PCM."""
        release = asyncio.Event()
        source_eof = asyncio.Event()
        committed = asyncio.Event()
        received = bytearray()
        first = b"\x01\0" * 4000
        last = b"\x02\0" * 320

        async def source() -> AsyncGenerator[bytes]:
            """Отдаёт preroll и ждёт подтверждённого завершения записи.

            :yield: Реальные PCM chunks входной реплики.
            """
            yield first
            await release.wait()
            yield last
            source_eof.set()

        async def recognize(socket: ServerConnection) -> None:
            """Публикует гипотезу до final commit, проверяя каждый transport frame.

            :param socket: Настоящее ASR-соединение.
            """
            await handshake(socket)
            sent_hypothesis = False
            while True:
                command = orjson.loads(await socket.recv())
                if command.get("final") is True:
                    assert source_eof.is_set()
                    committed.set()
                    await socket.send(orjson.dumps({"type": "transcription.done", "text": "Живая русская реплика."}).decode("utf-8"))
                    break
                assert command["type"] == "input_audio_buffer.append"
                pcm = base64.b64decode(command["audio"], validate=True)
                assert len(pcm) <= 3200 and len(pcm) % 2 == 0
                received.extend(pcm)
                if not sent_hypothesis:
                    await socket.send(orjson.dumps({"type": "transcription.hypothesis", "text": "Живая"}).decode("utf-8"))
                    sent_hypothesis = True
            await socket.wait_closed()

        async with serve(recognize, "127.0.0.1", 0) as server:
            client = ASRClient(settings_for(server.sockets[0].getsockname()[1]))
            results = client.transcribe_stream(source())
            try:
                assert await asyncio.wait_for(anext(results), timeout=1) == ("Живая", False)
                assert not source_eof.is_set() and not committed.is_set()
                release.set()
                assert [event async for event in results] == [("Живая русская реплика.", True)]
                assert committed.is_set() and bytes(received) == first + last
            finally:
                release.set()
                await results.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("stage", ["before_hypothesis", "after_hypothesis"])
def test_cancel_during_live_input_closes_source_and_websocket(stage: str) -> None:
    """Отмена до первого transcript либо на yield прекращает upload и закрывает ASR WS.

    :param stage: Ожидание первого результата либо уже опубликованная гипотеза.
    """

    async def scenario() -> None:
        """Отменяет только после наблюдаемого приёма первого PCM сервером."""
        received = asyncio.Event()
        disconnected = asyncio.Event()
        source_closed = asyncio.Event()

        async def source() -> AsyncGenerator[bytes]:
            """Удерживает микрофон открытым без окончания реплики.

            :yield: Первый PCM, после которого новых кадров пока нет.
            """
            try:
                yield b"\0\0" * 320
                await asyncio.Future()
            finally:
                source_closed.set()

        async def recognize(socket: ServerConnection) -> None:
            """Не выдаёт гипотез, ожидая закрытия клиента.

            :param socket: Настоящее внутреннее соединение.
            """
            await handshake(socket)
            with suppress(ConnectionClosed):
                while True:
                    command = orjson.loads(await socket.recv())
                    assert command.get("final") is not True
                    received.set()
                    if stage == "after_hypothesis":
                        await socket.send(orjson.dumps({"type": "transcription.hypothesis", "text": "Гипотеза"}).decode("utf-8"))
            disconnected.set()

        async with serve(recognize, "127.0.0.1", 0) as server:
            results = ASRClient(settings_for(server.sockets[0].getsockname()[1])).transcribe_stream(source())
            task = asyncio.create_task(anext(results))
            await asyncio.wait_for(received.wait(), timeout=1)
            if stage == "before_hypothesis":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                assert await task == ("Гипотеза", False)
                await results.aclose()
            await asyncio.wait_for(disconnected.wait(), timeout=1)
            assert source_closed.is_set()
            assert not [task for task in asyncio.all_tasks() if task.get_name().startswith("asr-")]

    asyncio.run(scenario())


def test_final_before_input_eof_is_rejected_without_waiting_for_source() -> None:
    """Нештатный ранний final не объявляет успех и не оставляет upload ждать микрофон."""

    async def scenario() -> None:
        """Публикует final при ещё открытом источнике PCM."""
        source_closed = asyncio.Event()

        async def source() -> AsyncGenerator[bytes]:
            """Удерживает вход открытым после первого кадра.

            :yield: PCM кадр незавершённой реплики.
            """
            try:
                yield b"\0\0" * 320
                await asyncio.Future()
            finally:
                source_closed.set()

        async def recognize(socket: ServerConnection) -> None:
            """Нарушает контракт, отвечая final до завершающего commit.

            :param socket: Настоящее соединение с concrete ASR-клиентом.
            """
            await handshake(socket)
            command = orjson.loads(await socket.recv())
            assert command["type"] == "input_audio_buffer.append"
            await socket.send(orjson.dumps({"type": "transcription.done", "text": "Преждевременно"}).decode("utf-8"))
            await socket.wait_closed()

        async with serve(recognize, "127.0.0.1", 0) as server:
            results = ASRClient(settings_for(server.sockets[0].getsockname()[1])).transcribe_stream(source())
            with pytest.raises(SessionError) as error:
                await asyncio.wait_for(anext(results), timeout=0.5)
            assert error.value.code == "asr_invalid_response"
            assert source_closed.is_set()

    asyncio.run(scenario())


def test_input_failure_interrupts_pending_receive_without_model_timeout() -> None:
    """Ошибка источника не ждёт 60-секундного deadline ожидания ASR."""

    async def scenario() -> None:
        """Ломает источник после первого фрейма при молчащем ASR runtime."""
        disconnected = asyncio.Event()

        async def source() -> AsyncGenerator[bytes]:
            """Воспроизводит конкретный отказ чтения входного PCM.

            :yield: Один PCM frame перед ошибкой.
            """
            yield b"\0\0" * 320
            raise OSError("test microphone input failed")

        async def recognize(socket: ServerConnection) -> None:
            """Принимает кадры, но не публикует ни одного результата.

            :param socket: Настоящее model-client соединение.
            """
            await handshake(socket)
            with suppress(ConnectionClosed):
                while True:
                    await socket.recv()
            disconnected.set()

        async with serve(recognize, "127.0.0.1", 0) as server:
            results = ASRClient(settings_for(server.sockets[0].getsockname()[1])).transcribe_stream(source())
            with pytest.raises(SessionError) as error:
                await asyncio.wait_for(anext(results), timeout=0.5)
            assert error.value.code == "asr_unavailable"
            assert isinstance(error.value.__cause__, OSError)
            await asyncio.wait_for(disconnected.wait(), timeout=1)

    asyncio.run(scenario())


def test_live_input_does_not_expire_model_idle_deadline_before_eof() -> None:
    """Открытый поток микрофона не запускает тайм-аут ожидания окончательного результата ASR."""

    async def scenario() -> None:
        """Удерживает вход дольше короткого тестового deadline до первой гипотезы."""
        received = asyncio.Event()
        release = asyncio.Event()

        async def source() -> AsyncGenerator[bytes]:
            """Отдаёт кадр и продолжает активную реплику до разрешения EOF.

            :yield: Первый PCM frame.
            """
            yield b"\0\0" * 320
            await release.wait()

        async def recognize(socket: ServerConnection) -> None:
            """Не распознаёт до завершения пользовательского ввода.

            :param socket: Настоящее соединение ASR gateway.
            """
            await handshake(socket)
            while not orjson.loads(await socket.recv()).get("final"):
                received.set()
            await socket.send(orjson.dumps({"type": "transcription.done", "text": "Готово."}).decode("utf-8"))
            await socket.wait_closed()

        async with serve(recognize, "127.0.0.1", 0) as server:
            settings = settings_for(server.sockets[0].getsockname()[1])
            settings.inference.first_response_timeout_seconds = 0.03
            settings.inference.idle_timeout_seconds = 0.03
            results = ASRClient(settings).transcribe_stream(source())
            task = asyncio.create_task(anext(results))
            try:
                await asyncio.wait_for(received.wait(), timeout=1)
                await asyncio.sleep(0.06)
                assert not task.done()
                release.set()
                assert await asyncio.wait_for(task, timeout=1) == ("Готово.", True)
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await results.aclose()

    asyncio.run(scenario())


def test_missing_final_after_input_eof_has_bounded_deadline() -> None:
    """После настоящего EOF молчащий ASR получает конкретный timeout, а WS закрывается."""

    async def scenario() -> None:
        """Подтверждает final commit на сервере перед ожиданием отказа клиента."""
        committed = asyncio.Event()

        async def source() -> AsyncGenerator[bytes]:
            """Выдаёт завершённую короткую реплику.

            :yield: Единственный PCM frame.
            """
            yield b"\0\0" * 320

        async def recognize(socket: ServerConnection) -> None:
            """Молчит после принятого final commit.

            :param socket: Настоящее ASR-соединение.
            """
            await handshake(socket)
            while not orjson.loads(await socket.recv()).get("final"):
                pass
            committed.set()
            await socket.wait_closed()

        async with serve(recognize, "127.0.0.1", 0) as server:
            settings = settings_for(server.sockets[0].getsockname()[1])
            settings.inference.first_response_timeout_seconds = 0.03
            results = ASRClient(settings).transcribe_stream(source())
            with pytest.raises(SessionError) as error:
                await asyncio.wait_for(anext(results), timeout=1)
            assert error.value.code == "asr_unavailable" and committed.is_set()

    asyncio.run(scenario())
