"""Подготавливает закреплённые модели каскада и повторно использует полный локальный snapshot."""

import os
import sys
import time
from pathlib import Path

import orjson
from huggingface_hub import snapshot_download
from loguru import logger

MODELS = {
    "asr": ("Qwen/Qwen3-ASR-0.6B", "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"),
    "llm": ("Qwen/Qwen3-4B-AWQ", "74d4bd2bd4bff9cafc9345221320bffb08b406a3"),
    "tts": ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"),
}
REQUIRED_FILES = {
    "asr": ("config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json"),
    "llm": ("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"),
    "tts": (
        "config.json",
        "cosyvoice3.yaml",
        "llm.pt",
        "flow.pt",
        "hift.pt",
        "campplus.onnx",
        "speech_tokenizer_v3.onnx",
        "CosyVoice-BlankEN/config.json",
        "CosyVoice-BlankEN/model.safetensors",
        "CosyVoice-BlankEN/tokenizer_config.json",
        "CosyVoice-BlankEN/vocab.json",
        "CosyVoice-BlankEN/merges.txt",
    ),
}


def missing_files(directory: Path, role: str) -> list[str]:
    """Проверяет наличие непустых конфигураций, tokenizer и всех частей весов.

    Это проверка комплектности, а не криптографическая проверка содержимого весов.

    :param directory: Каталог snapshot одной модели.
    :param role: Роль закреплённой модели: asr, llm или tts.
    :return: Отсутствующие/пустые файлы либо некорректный индекс частей весов.
    """
    required = list(REQUIRED_FILES[role])
    if role in {"asr", "llm"}:
        index = directory / "model.safetensors.index.json"
        if index.exists():
            try:
                value = orjson.loads(index.read_bytes())
                weights = value.get("weight_map") if isinstance(value, dict) else None
                if not isinstance(weights, dict) or not weights:
                    return [index.name]
                for name in weights.values():
                    if not isinstance(name, str) or not name or not (directory / name).resolve().is_relative_to(directory.resolve()):
                        return [index.name]
                    required.append(name)
            except orjson.JSONDecodeError:
                return [index.name]
        else:
            required.append("model.safetensors")
    return sorted({name for name in required if not (directory / name).is_file() or (directory / name).stat().st_size == 0})


def main(models_dir: Path = Path("/models")) -> None:
    """Пропускает готовые модели и возобновляет загрузку недостающих файлов.

    При первом запуске нужен интернет; Hub сохраняет метаданные загрузки в .cache
    каждого snapshot. После ошибки повторный запуск использует уже скачанные файлы.
    Каталог с маркером другой модели или версии автоматически не перезаписывается.

    :param models_dir: Общий каталог локальных snapshot, доступный контейнеру на запись.
    :raises RuntimeError: Чужая версия, повреждённый маркер или неполный результат загрузки.
    """
    for role, (repo, revision) in MODELS.items():
        directory = models_dir / role
        marker = directory / "revision.json"
        expected = {"role": role, "repo": repo, "revision": revision}
        if marker.exists():
            try:
                manifest = orjson.loads(marker.read_bytes())
            except orjson.JSONDecodeError as exc:
                raise RuntimeError(f"Повреждён маркер модели {marker}; проверьте каталог перед повторным запуском") from exc
            if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"Другая модель или версия в {directory}; ожидается {repo}@{revision}. Переместите прежний каталог перед загрузкой закреплённой версии")
            if not missing_files(directory, role):
                logger.info("Закреплённая модель готова локально: role={}", role)
                continue

        logger.info("Загрузка закреплённой модели: role={}", role)
        started = time.perf_counter()
        directory.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(directory), max_workers=4)
        missing = missing_files(directory, role)
        if missing:
            logger.warning("Не завершена загрузка модели: role={}, missing_files={}", role, len(missing))
            raise RuntimeError(f"Неполная модель {role}: {', '.join(missing)}. Повторите запуск для продолжения загрузки")
        # Маркер становится видимым целиком только после завершения загрузки и проверки файлов.
        temporary = marker.with_suffix(".json.tmp")
        temporary.write_bytes(orjson.dumps({**expected, "path": str(directory)}, option=orjson.OPT_INDENT_2))
        temporary.replace(marker)
        logger.info("Модель загружена: role={}, elapsed_seconds={:.1f}", role, time.perf_counter() - started)


if __name__ == "__main__":
    # Настройка выполняется только в процессе подготовки моделей, а не при импорте тестами.
    logger.remove()
    sink = logger.add(
        sys.stderr, level=os.getenv("RUNTIME_LOG_LEVEL", "INFO"), enqueue=True, diagnose=False, backtrace=False, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message} | {extra}"
    )
    try:
        main()
    except Exception as exc:
        logger.error("Не удалось подготовить модели: type={}", type(exc).__name__)
        raise SystemExit(1) from None
    finally:
        logger.complete()
        logger.remove(sink)
