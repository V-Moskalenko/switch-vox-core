"""Запуск единственного Uvicorn worker с настройками локального API."""

import uvicorn

from settings import config

if __name__ == "__main__":
    # Лимит транспорта должен вмещать и JSON-команду, и максимальный PCM-кадр.
    maximum_frame_bytes = config.audio.input_sample_rate * config.audio.max_frame_ms // 1000 * config.audio.sample_width_bytes
    uvicorn.run(
        "api.fastapi_app:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        access_log=False,
        proxy_headers=False,
        ws_max_size=max(config.api.max_json_bytes, maximum_frame_bytes),
    )
