"""CPU-сценарий проверки конкурентного token-worker и очистки модельных буферов."""

import queue
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import closing

import torch
from backend import CancellableCosyVoice3Model


@torch.inference_mode()
def run_trial(model: CancellableCosyVoice3Model, inputs: dict[str, torch.Tensor], tokens: torch.Tensor, *, active: bool) -> tuple[dict[str, object], torch.Tensor]:
    """Сравнивает неизменный flow-вход, сохраняя живой KV в обоих условиях.

    :param model: Единственная резидентная модель без других запросов.
    :param inputs: Один закреплённый frontend input speech-token генератора.
    :param tokens: Неизменный префикс для первого streaming flow chunk.
    :param active: Освободить припаркованный token-worker перед flow.
    :return: Наблюдения перекрытия/времени и первый непустой CPU-аудиоблок.
    :raises RuntimeError: Невозможность парковки, ошибка worker или невалидный audio.
    :raises TimeoutError: Worker не подтвердил остановку после опыта.
    """
    load_id, flow_id = str(uuid.uuid4()), str(uuid.uuid4())
    paused, release, cancel, worker_cancel = (threading.Event() for _ in range(4))
    failures: queue.Queue[Exception] = queue.Queue(maxsize=1)
    original_inference = model.llm.inference

    def controlled_inference(**kwargs: object) -> Iterator[int]:
        """Паркует реальный генератор после публикации первого принятого токена.

        :param kwargs: Неизменные параметры upstream inference.
        :yield: Настоящие speech tokens; KV остаётся в живом upstream generator.
        """
        with closing(original_inference(**kwargs)) as source:
            yield next(source)
            paused.set()
            # Владелец всегда освобождает gate в finally до join, включая ошибку flow.
            release.wait()
            if cancel.is_set() or worker_cancel.is_set():
                return
            yield from source

    model._stream_started_ns = time.perf_counter_ns()
    model._compute_lock = threading.Lock()
    model._flow_idle = threading.Event()
    model._flow_idle.set()
    model.progress = {"generated_tokens": 0, "buffered_tokens": 0, "flow_chunks": 0, "llm_done": 0}
    model.tts_speech_token_dict[load_id] = []
    model.llm_end_dict[load_id] = False
    model.hift_cache_dict[flow_id] = None
    # Upstream изменяет text_len на месте; копирование и перенос выполняются до измерения.
    load_inputs = {key: value.detach().cpu().clone() for key, value in inputs.items()}
    model.llm.inference = controlled_inference
    worker = threading.Thread(target=model._generate_tokens, args=(load_inputs, load_id, cancel, failures, worker_cancel), daemon=True)
    worker.start()
    try:
        deadline = time.perf_counter() + 10
        while not paused.wait(timeout=0.005):
            if not failures.empty():
                raise failures.get_nowait()
            if not worker.is_alive() or time.perf_counter() >= deadline:
                raise RuntimeError("Token-worker не подтвердил парковку после первого токена")
        before = model.progress["generated_tokens"]
        if (before, model.llm_end_dict[load_id]) != (1, False):
            raise RuntimeError("Token-worker не сохранил ожидаемый живой KV")
        if active:
            release.set()
        started = time.perf_counter_ns()
        # Обходим production-сериализацию намеренно: эта проба измеряет исходное перекрытие.
        audio = model.token2wav(tokens, inputs["flow_prompt_speech_token"], inputs["prompt_speech_feat"], inputs["flow_embedding"], 0, flow_id, stream=True, finalize=False).cpu()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        after = model.progress["generated_tokens"]
        result = {"active": active, "elapsed_ms": elapsed_ms, "tokens_before": before, "tokens_after": after, "worker_finished_at_end": model.llm_end_dict[load_id], "overlap_observed": after > before}
        if not audio.numel() or not torch.isfinite(audio).all():
            raise RuntimeError("Проба flow вернула пустой или нечисловой audio")
    finally:
        cancel.set()
        worker_cancel.set()
        release.set()
        worker.join(timeout=2)
        if worker.is_alive():
            raise TimeoutError("Диагностический token-worker не остановился")
        model.llm.inference = original_inference
        model.tts_speech_token_dict.pop(load_id, None)
        model.llm_end_dict.pop(load_id, None)
        model.hift_cache_dict.pop(flow_id, None)
    if not failures.empty():
        raise failures.get_nowait()
    return result, audio
