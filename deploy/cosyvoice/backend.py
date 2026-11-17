"""Синтез речи CosyVoice3 с выбранным образцом голоса и поддержкой отмены."""

import io
import os
import queue
import threading
import time
import uuid
from collections.abc import Generator, Iterator
from contextlib import closing
from pathlib import Path

import numpy as np
import orjson
import torch
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoice3Model
from hyperpyyaml import load_hyperpyyaml
from loguru import logger

REFERENCE_PREFIX = "You are a helpful assistant.<|endofprompt|>"
REFERENCE_TEXT = REFERENCE_PREFIX + "希望你以后能够做的比我还好呦。"
REFERENCE_ID = "upstream-development-reference"
PCM_FRAME_BYTES = 960


class CancellableCosyVoice3Model(CosyVoice3Model):
    """Добавляет сигнал отмены в token loop закреплённой upstream-реализации."""

    def _speech_tokens(self, source: Generator[int, None, None], cancel: threading.Event, worker_cancel: threading.Event) -> Iterator[int]:
        """Чередует speech-token шаги с flow, позволяя отменить ожидание общего lock.

        :param source: Закреплённый upstream generator; next завершает шаг через sampler.item().
        :param cancel: Внешняя отмена запроса независимо от текущего flow.
        :param worker_cancel: Внутренняя остановка при завершении либо отказе flow.
        :yield: Готовый speech token после освобождения compute-lock.
        """
        with closing(source):
            while not cancel.is_set() and not worker_cancel.is_set():
                if not self._flow_idle.wait(timeout=0.01):
                    continue
                if not self._compute_lock.acquire(timeout=0.01):
                    continue
                try:
                    if cancel.is_set() or worker_cancel.is_set():
                        return
                    if not self._flow_idle.is_set():
                        continue
                    token = next(source, None)
                finally:
                    self._compute_lock.release()
                if token is None:
                    return
                yield token

    def _generate_tokens(self, inputs: dict[str, torch.Tensor], request_id: str, cancel: threading.Event, failures: queue.Queue[Exception], worker_cancel: threading.Event) -> None:
        """Генерирует speech tokens, проверяя отмену между CUDA kernel calls.

        :param inputs: Результат concrete frontend CosyVoice3.
        :param request_id: Ключ временных буферов модели.
        :param cancel: Сигнал отмены запроса.
        :param failures: Передача ошибки из LLM-thread владельцу синтеза.
        :param worker_cancel: Остановка LLM при завершении или ошибке flow, не отменяющая HTTP.
        """
        try:
            if cancel.is_set() or worker_cancel.is_set():
                return
            with torch.inference_mode(), self.llm_context, torch.cuda.amp.autocast(self.fp16):
                tokens = self.llm.inference(
                    text=inputs["text"].to(self.device),
                    text_len=inputs["text_len"].to(self.device),
                    prompt_text=inputs["prompt_text"].to(self.device),
                    prompt_text_len=inputs["prompt_text_len"].to(self.device),
                    prompt_speech_token=inputs["llm_prompt_speech_token"].to(self.device),
                    prompt_speech_token_len=inputs["llm_prompt_speech_token_len"].to(self.device),
                    embedding=inputs["llm_embedding"].to(self.device),
                    uuid=request_id,
                )
                silent_count = 0
                for token in self._speech_tokens(tokens, cancel, worker_cancel):
                    self.progress["generated_tokens"] += 1
                    if cancel.is_set() or worker_cancel.is_set():
                        break
                    silent_count = silent_count + 1 if token in self.silent_tokens else 0
                    if silent_count <= 5:
                        if not self.progress["buffered_tokens"]:
                            self.progress["first_token_us"] = (time.perf_counter_ns() - self._stream_started_ns) // 1000
                        self.tts_speech_token_dict[request_id].append(token)
                        self.progress["buffered_tokens"] += 1
        except Exception as exc:
            failures.put(exc)
        finally:
            self.progress["llm_done_us"] = (time.perf_counter_ns() - self._stream_started_ns) // 1000
            self.llm_end_dict[request_id] = True
            self.progress["llm_done"] = 1

    def _audio_chunk(self, inputs: dict[str, torch.Tensor], tokens: torch.Tensor, offset: int, request_id: str, cancel: threading.Event, *, finalize: bool) -> torch.Tensor:
        """Исключает одновременный token step до готовности flow-аудио на CPU.

        :param inputs: Фиксированные prompt features текущей реплики.
        :param tokens: Готовый речевой префикс либо полный финальный набор.
        :param offset: Число уже обработанных speech tokens.
        :param request_id: Ключ flow/vocoder cache текущей реплики.
        :param cancel: Внешняя отмена; отменённый streaming chunk не копируется на CPU.
        :param finalize: Завершить хвост вместо очередного streaming chunk.
        :return: Аудиоблок на CPU либо неиспользуемый GPU-блок при отмене streaming.
        """
        # Готовый flow имеет приоритет перед следующим token step; Lock сам по себе не даёт fairness.
        self._flow_idle.clear()
        try:
            with self._compute_lock:
                if cancel.is_set():
                    return torch.empty(1, 0)
                self.progress.setdefault("first_flow_us", (time.perf_counter_ns() - self._stream_started_ns) // 1000)
                self.progress.setdefault("first_flow_finalize", int(finalize))
                audio = self.token2wav(tokens, inputs["flow_prompt_speech_token"], inputs["prompt_speech_feat"], inputs["flow_embedding"], offset, request_id, stream=not finalize, finalize=finalize)
                self.progress["flow_chunks"] += 1
                if audio.numel() and (finalize or not cancel.is_set()):
                    audio = audio.cpu()
                    self.progress.setdefault("first_audio_us", (time.perf_counter_ns() - self._stream_started_ns) // 1000)
            return audio
        finally:
            self._flow_idle.set()

    @torch.inference_mode()
    def stream(self, inputs: dict[str, torch.Tensor], cancel: threading.Event) -> Iterator[torch.Tensor]:
        """Выдаёт аудиоблоки по мере готовности речевых токенов и очищает буферы запроса.

        Если поток генерации токенов не остановился после отмены, завершает процесс с кодом 70.

        :param inputs: Токены и фиксированные prompt features.
        :param cancel: Сигнал отмены, независимый от появления следующего PCM.
        :yield: Новые float32 audio samples формы [1, N].
        :raises Exception: Ошибка генерации речевых токенов или синтеза аудио.
        """
        request_id = str(uuid.uuid4())
        self._stream_started_ns = time.perf_counter_ns()
        self._compute_lock = threading.Lock()
        self._flow_idle = threading.Event()
        self._flow_idle.set()
        self.progress = {"generated_tokens": 0, "buffered_tokens": 0, "flow_chunks": 0, "llm_done": 0}
        self.tts_speech_token_dict[request_id] = []
        self.llm_end_dict[request_id] = False
        self.hift_cache_dict[request_id] = None
        failures: queue.Queue[Exception] = queue.Queue(maxsize=1)
        worker_cancel = threading.Event()
        worker = threading.Thread(target=self._generate_tokens, args=(inputs, request_id, cancel, failures, worker_cancel), daemon=True)
        worker.start()
        try:
            offset = yield from self._stream_audio(inputs, cancel, request_id, failures)
            if not cancel.is_set():
                worker.join(timeout=2)
                if not failures.empty():
                    raise failures.get_nowait()
                tokens = torch.tensor(self.tts_speech_token_dict[request_id]).unsqueeze(0)
                if tokens.numel():
                    audio = self._audio_chunk(inputs, tokens, offset, request_id, cancel, finalize=True)
                    if audio.numel():
                        yield audio
        finally:
            # Остановка внутреннего LLM не должна подавлять HTTP EOF или ошибку модели.
            worker_cancel.set()
            worker.join(timeout=2)
            # Не оставляем потерянную CUDA-генерацию после client disconnect.
            if worker.is_alive():
                logger.error("Генератор речевых токенов TTS не остановился после отмены")
                logger.complete()
                os._exit(70)
            self.tts_speech_token_dict.pop(request_id, None)
            self.llm_end_dict.pop(request_id, None)
            self.hift_cache_dict.pop(request_id, None)
            self.progress["finished_us"] = (time.perf_counter_ns() - self._stream_started_ns) // 1000

    def _stream_audio(self, inputs: dict[str, torch.Tensor], cancel: threading.Event, request_id: str, failures: queue.Queue[Exception]) -> Generator[torch.Tensor, None, int]:
        """Синтезирует готовые порции токенов, сохраняя смещение для финального хвоста.

        :param inputs: Фиксированные prompt features текущей реплики.
        :param cancel: Внешняя отмена запроса независимо от готовности аудио.
        :param request_id: Ключ токенов и flow/vocoder cache текущего запроса.
        :param failures: Ошибки параллельного производителя speech tokens.
        :yield: Непустые CPU audio chunks до окончания генерации токенов.
        :return: Число уже обработанных speech tokens для финального синтеза.
        """
        offset = 0
        hop = 25
        prompt_length = inputs["flow_prompt_speech_token"].shape[1]
        padding = int(np.ceil(prompt_length / hop) * hop - prompt_length)
        while not cancel.wait(timeout=0.01):
            if not failures.empty():
                raise failures.get_nowait()
            current_hop = hop + padding if offset == 0 else hop
            ready = len(self.tts_speech_token_dict[request_id]) - offset
            if ready >= current_hop + self.flow.pre_lookahead_len:
                tokens = torch.tensor(self.tts_speech_token_dict[request_id][: offset + current_hop + self.flow.pre_lookahead_len]).unsqueeze(0)
                audio = self._audio_chunk(inputs, tokens, offset, request_id, cancel, finalize=False)
                offset += current_hop
                hop = min(100, hop * 2)
                if not cancel.is_set() and audio.numel():
                    yield audio
            if self.llm_end_dict[request_id] and len(self.tts_speech_token_dict[request_id]) - offset < current_hop + self.flow.pre_lookahead_len:
                break
        return offset


class CosyVoiceEngine:
    """Резидентная модель CosyVoice3 с признаками голоса, вычисленными при загрузке."""

    def __init__(self, model_dir: Path, reference: Path, reference_text: str = REFERENCE_TEXT) -> None:
        """Загружает веса и вычисляет признаки образца голоса для последующего синтеза.

        :param model_dir: Каталог весов и конфигурации CosyVoice3, доступный для чтения.
        :param reference: WAV выбранного фиксированного reference voice.
        :param reference_text: Точная расшифровка образца с префиксом CV3; по умолчанию текст штатного образца CosyVoice.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CosyVoice3 требует доступную CUDA GPU")
        torch.set_num_threads(2)
        original = (model_dir / "cosyvoice3.yaml").read_text(encoding="utf-8")
        marker = "# gan related module"
        if marker not in original:
            raise RuntimeError("Изменился закреплённый формат CosyVoice3 YAML")
        # Training/discriminator/dataset nodes не нужны inference и не загружаются.
        graph = (
            original.split(marker, maxsplit=1)[0]
            + """
get_tokenizer: !name:cosyvoice.tokenizer.tokenizer.get_qwen_tokenizer
    token_path: !ref <qwen_pretrain_path>
    skip_special_tokens: True
    version: cosyvoice3
allowed_special: 'all'
feat_extractor: !name:matcha.utils.audio.mel_spectrogram
    n_fft: 1920
    num_mels: 80
    sampling_rate: 24000
    hop_size: 480
    win_size: 1920
    fmin: 0
    fmax: null
    center: False
"""
        )
        configs = load_hyperpyyaml(io.StringIO(graph), overrides={"qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN")})
        self.frontend = CosyVoiceFrontEnd(configs["get_tokenizer"], configs["feat_extractor"], str(model_dir / "campplus.onnx"), str(model_dir / "speech_tokenizer_v3.onnx"), allowed_special="all")
        self.model = CancellableCosyVoice3Model(configs["llm"], configs["flow"], configs["hift"], fp16=True)
        # LLM и flow переводятся в FP16 на CPU до переноса на GPU; vocoder сохраняет исходную точность.
        for component, filename, half_precision in [(self.model.llm, "llm.pt", True), (self.model.flow, "flow.pt", True), (self.model.hift, "hift.pt", False)]:
            weights = torch.load(model_dir / filename, map_location="cpu", weights_only=True)
            if filename == "hift.pt":
                weights = {key.removeprefix("generator."): value for key, value in weights.items()}
            component.load_state_dict(weights, strict=True)
            if half_precision:
                component.half()
            component.to(self.model.device).eval()
            del weights
        fixed = self.frontend.frontend_zero_shot("", reference_text, str(reference), 24000, "")
        fixed.pop("text")
        fixed.pop("text_len")
        self.frontend.spk2info[REFERENCE_ID] = fixed

    def synthesize(self, text: str, cancel: threading.Event) -> Iterator[bytes]:
        """Синтезирует текст в кадры PCM16LE mono 24 кГц длительностью не более 20 мс.

        :param text: Текст одной короткой русской фразы.
        :param cancel: Сигнал прекращения вычислений.
        :yield: Реальные PCM chunks, максимум 960 байт каждый.
        """
        started = time.perf_counter_ns()
        inputs = self.frontend.frontend_zero_shot(text, "", "", 24000, REFERENCE_ID)
        frontend_us = (time.perf_counter_ns() - started) // 1000
        try:
            with closing(self.model.stream(inputs, cancel)) as audio_chunks:
                for audio in audio_chunks:
                    pcm = (audio.squeeze(0).float().clamp(-1, 1).numpy() * 32767).astype("<i2").tobytes()
                    self.model.progress.setdefault("first_pcm_us", (time.perf_counter_ns() - self.model._stream_started_ns) // 1000)
                    for offset in range(0, len(pcm), PCM_FRAME_BYTES):
                        if cancel.is_set():
                            return
                        yield pcm[offset : offset + PCM_FRAME_BYTES]
        finally:
            self.model.progress["frontend_us"] = frontend_us
            if os.getenv("TTS_PROFILE") == "1":
                logger.info("Профиль TTS: {}", orjson.dumps(self.model.progress, option=orjson.OPT_SORT_KEYS).decode())
