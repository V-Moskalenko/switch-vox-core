"""Проверяет обычный запуск Compose, изоляцию моделей и воспроизводимость образов."""

import subprocess
from pathlib import Path
from typing import Any

import orjson

from settings import PROJECT_ROOT


def compose_config() -> dict[str, Any]:
    """Читает фактически разобранный Compose без обращения к Docker daemon.

    :return: Нормализованная конфигурация каскада.
    """
    result = subprocess.run(
        ["docker", "compose", "--project-name", "switch-vox-core", "--file", str(PROJECT_ROOT / "docker-compose.yml"), "config", "--format", "json"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return dict(orjson.loads(result.stdout))


def test_cascade_isolated_models_and_readiness_configuration() -> None:
    """Проверяет опубликованные порты, единственный режим и readiness зависимостей."""
    config = compose_config()
    services = config["services"]
    assert set(services) == {"model-init", "app-cascade", "asr", "llm", "tts"}
    assert set(config["networks"]) == {"voice", "entry"}
    assert config["networks"]["voice"]["internal"] is True
    assert not config["networks"]["voice"].get("external")
    assert not config["networks"]["entry"].get("internal")
    assert not config["networks"]["entry"].get("external")
    app = services["app-cascade"]
    assert app["environment"]["INFERENCE__MODE"] == "cascade"
    assert app["environment"]["API__HOST"] == "0.0.0.0"
    assert len(app["ports"]) == 1
    assert {key: app["ports"][0][key] for key in ["host_ip", "target", "published", "protocol"]} == {"host_ip": "127.0.0.1", "target": 8000, "published": "8000", "protocol": "tcp"}
    assert "/health/ready" in " ".join(app["healthcheck"]["test"])
    assert all(app["depends_on"][role]["condition"] == "service_healthy" for role in ["asr", "llm", "tts"])
    for role, service in services.items():
        assert not service.get("profiles")
        expected_networks = {"voice", "entry"} if role == "app-cascade" else ({"entry"} if role == "model-init" else {"voice"})
        assert set(service["networks"]) == expected_networks
        assert "network_mode" not in service
        assert not service.get("privileged")
        if role == "app-cascade":
            continue
        assert not service.get("ports")
        if role == "model-init":
            assert not service.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices")
            assert service["read_only"] is True
            assert service["cap_drop"] == ["ALL"]
            assert set(service["cap_add"]) == {"DAC_OVERRIDE", "FOWNER"}
            assert service["security_opt"] == ["no-new-privileges:true"]
            models = next(volume for volume in service["volumes"] if volume["target"] == "/models")
            assert not models.get("read_only")
            assert models["bind"].get("create_host_path", True) is True
            continue
        assert service["depends_on"]["model-init"]["condition"] == "service_completed_successfully"
        assert service["deploy"]["resources"]["reservations"]["devices"] == [{"capabilities": ["gpu"], "driver": "nvidia", "device_ids": ["0"]}]
        models = next(volume for volume in service["volumes"] if volume["target"] == "/models")
        assert models["read_only"] is True
        assert models["bind"].get("create_host_path") is False
        assert service["environment"]["HF_HUB_OFFLINE"] == "1"
        assert "/health" in " ".join(service["healthcheck"]["test"])
    assert services["llm"]["depends_on"]["asr"]["condition"] == "service_healthy"
    assert services["tts"]["depends_on"]["llm"]["condition"] == "service_healthy"
    assert "@sha256:" in services["llm"]["image"]
    assert "/app/run_llm.py" in services["llm"]["entrypoint"]
    assert Path(services["asr"]["build"]["context"]) == PROJECT_ROOT / "deploy" / "asr"
    assert Path(services["tts"]["build"]["context"]) == PROJECT_ROOT / "deploy" / "cosyvoice"


def test_app_image_installs_locked_runtime_and_excludes_local_state() -> None:
    """Не допускает mutable базовые images и копирование окружения/весов в app image."""
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    bases = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
    assert bases and all("@sha256:" in line for line in bases)
    assert "python:3.15.1-slim-trixie@sha256:" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert 'CMD ["python", "sources/main.py"]' in dockerfile
    assert "COPY . " not in dockerfile
    ignored = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored and "data/" in ignored and ".venv/" in ignored


def test_model_init_image_has_locked_cpu_dependencies() -> None:
    """Загрузка весов использует закреплённые зависимости без CUDA-образа."""
    context = PROJECT_ROOT / "deploy" / "models"
    dockerfile = (context / "Dockerfile").read_text(encoding="utf-8")
    bases = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
    assert bases and all("@sha256:" in line for line in bases)
    assert "--require-hashes" in dockerfile
    assert "nvidia/" not in dockerfile and "vllm/" not in dockerfile
    requirements = (context / "requirements.lock").read_text(encoding="utf-8")
    assert "huggingface-hub==" in requirements
    assert "orjson==3.13.0" in requirements
    assert "--hash=sha256:" in requirements
