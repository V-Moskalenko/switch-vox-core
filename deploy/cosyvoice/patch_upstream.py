"""Исключает только побочные training/MultiGPU импорты Matcha в single-worker runtime."""

from pathlib import Path

MATCHA = Path("/opt/CosyVoice/third_party/Matcha-TTS/matcha/utils")
initializer = MATCHA / "__init__.py"
logger = MATCHA / "pylogger.py"
initial_text = initializer.read_text(encoding="utf-8")
logger_text = logger.read_text(encoding="utf-8")
if "from matcha.utils.instantiators import instantiate_callbacks, instantiate_loggers" not in initial_text:
    raise RuntimeError("Изменился закреплённый Matcha initializer")
if "from lightning.pytorch.utilities import rank_zero_only" not in logger_text:
    raise RuntimeError("Изменился закреплённый Matcha logger")
initializer.write_text("# Inference-only single-worker package: training helpers are not imported.\n", encoding="utf-8")
logger_text = logger_text.replace("from lightning.pytorch.utilities import rank_zero_only\n", "")
logger_text = logger_text.replace("rank_zero_only(getattr(logger, level))", "getattr(logger, level)")
logger.write_text(logger_text, encoding="utf-8")
