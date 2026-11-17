"""Проверяет подготовку локальных моделей без сети и загрузки GPU-весов."""

from pathlib import Path
from unittest.mock import Mock

import orjson
import pytest
from deploy.models import download_models


@pytest.fixture
def llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Создаёт минимальный локальный snapshot и ограничивает запуск одной моделью.

    :param tmp_path: Каталог временных моделей.
    :param monkeypatch: Изолированные подмены состава моделей и сетевого клиента.
    :return: Каталог модели LLM.
    """
    monkeypatch.setattr(download_models, "MODELS", {"llm": download_models.MODELS["llm"]})
    path = tmp_path / "llm"
    path.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "model.safetensors"):
        (path / name).write_bytes(b"{}")
    return path


def write_manifest(path: Path, revision: str | None = None) -> bytes:
    """Записывает совместимый маркер загруженного snapshot.

    :param path: Каталог LLM.
    :param revision: Версия для проверки несовместимого локального каталога.
    :return: Исходные байты для проверки сохранности маркера при отказе.
    """
    repo, pinned = download_models.MODELS["llm"]
    data = orjson.dumps({"role": "llm", "repo": repo, "revision": revision or pinned, "path": str(path)})
    (path / "revision.json").write_bytes(data)
    return data


def test_ready_snapshot_never_uses_network(llm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Полный snapshot нужной версии запускается без повторного обращения к Hub.

    :param llm: Локальный snapshot LLM.
    :param monkeypatch: Подмена только сетевой загрузки.
    """
    write_manifest(llm)
    download = Mock(side_effect=AssertionError("Готовая модель не требует сети"))
    monkeypatch.setattr(download_models, "snapshot_download", download)
    download_models.main(llm.parent)
    download.assert_not_called()


def test_unmarked_snapshot_downloads_pinned_revision_before_manifest(llm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Маркер создаётся только после успешного получения закреплённого snapshot.

    :param llm: Файлы без маркера готовности.
    :param monkeypatch: Подмена скачивания с проверкой порядка действий.
    """

    def download(**kwargs: object) -> str:
        """Проверяет адрес локального каталога и отсутствие преждевременного маркера.

        :param kwargs: Параметры обращения к Hub.
        :return: Каталог завершённого snapshot.
        """
        assert kwargs["repo_id"] == download_models.MODELS["llm"][0]
        assert kwargs["revision"] == download_models.MODELS["llm"][1]
        assert kwargs["local_dir"] == str(llm)
        assert not (llm / "revision.json").exists()
        return str(llm)

    monkeypatch.setattr(download_models, "snapshot_download", download)
    download_models.main(llm.parent)
    assert orjson.loads((llm / "revision.json").read_bytes())["revision"] == download_models.MODELS["llm"][1]
    assert not (llm / "revision.json.tmp").exists()


@pytest.mark.parametrize("broken", ["model.safetensors", "config.json", "tokenizer.json"])
def test_incomplete_snapshot_is_repaired(llm: Path, monkeypatch: pytest.MonkeyPatch, broken: str) -> None:
    """Совпадающий маркер не скрывает отсутствующий файл весов или конфигурации.

    :param llm: Локальный snapshot с маркером.
    :param monkeypatch: Подмена сетевой загрузки.
    :param broken: Удаляемый обязательный файл.
    """
    write_manifest(llm)
    (llm / broken).unlink()
    download = Mock(side_effect=lambda **kwargs: (llm / broken).write_bytes(b"{}"))
    monkeypatch.setattr(download_models, "snapshot_download", download)
    download_models.main(llm.parent)
    download.assert_called_once()
    assert (llm / broken).stat().st_size > 0


def test_foreign_revision_is_preserved(llm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Чужой snapshot не перезаписывается автоматически.

    :param llm: Каталог с другой версией модели.
    :param monkeypatch: Подмена обращения к Hub.
    """
    original = write_manifest(llm, revision="other-revision")
    download = Mock(return_value=str(llm))
    monkeypatch.setattr(download_models, "snapshot_download", download)
    with pytest.raises(RuntimeError, match="revision|верси"):
        download_models.main(llm.parent)
    assert (llm / "revision.json").read_bytes() == original
    download.assert_not_called()


def test_failed_download_never_creates_ready_marker(llm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Сетевая ошибка оставляет каталог пригодным для следующей попытки.

    :param llm: Незавершённый локальный snapshot.
    :param monkeypatch: Подмена отказавшего обращения к Hub.
    """
    monkeypatch.setattr(download_models, "snapshot_download", Mock(side_effect=ConnectionError("Hub unavailable")))
    with pytest.raises(ConnectionError):
        download_models.main(llm.parent)
    assert not (llm / "revision.json").exists()


def test_download_without_required_weights_is_not_ready(llm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Возврат download API сам по себе не подтверждает наличие весов.

    :param llm: Каталог без файла весов.
    :param monkeypatch: Загрузчик, завершившийся без полного snapshot.
    """
    (llm / "model.safetensors").write_bytes(b"")
    monkeypatch.setattr(download_models, "snapshot_download", Mock(return_value=str(llm)))
    with pytest.raises(RuntimeError, match="model.safetensors"):
        download_models.main(llm.parent)
    assert not (llm / "revision.json").exists()


def test_shard_index_requires_every_referenced_file(llm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Отсутствующий shard не скрывается валидным index и маркером версии.

    :param llm: Snapshot с индексом двух частей весов.
    :param monkeypatch: Подмена загрузки, не восстановившей потерянную часть.
    """
    write_manifest(llm)
    (llm / "model.safetensors").unlink()
    (llm / "model.safetensors.index.json").write_bytes(orjson.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}))
    (llm / "part-1.safetensors").write_bytes(b"weights")
    download = Mock(return_value=str(llm))
    monkeypatch.setattr(download_models, "snapshot_download", download)
    with pytest.raises(RuntimeError, match="part-2.safetensors"):
        download_models.main(llm.parent)
    download.assert_called_once()


@pytest.mark.parametrize("contents", [b"invalid-json", b"[]"])
def test_invalid_manifest_never_overwrites_local_files(llm: Path, monkeypatch: pytest.MonkeyPatch, contents: bytes) -> None:
    """Нечитаемый или неверно устроенный маркер требует проверки владельцем каталога.

    :param llm: Локальный snapshot.
    :param monkeypatch: Подмена сетевого обращения.
    :param contents: Некорректное содержимое маркера.
    """
    (llm / "revision.json").write_bytes(contents)
    download = Mock(return_value=str(llm))
    monkeypatch.setattr(download_models, "snapshot_download", download)
    with pytest.raises(RuntimeError):
        download_models.main(llm.parent)
    assert (llm / "revision.json").read_bytes() == contents
    download.assert_not_called()


def test_empty_data_directory_is_created_before_download(llm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Новый bind-каталог не требует предварительного ручного создания model subdirectory.

    :param llm: Путь будущего snapshot.
    :param monkeypatch: Подмена скачивания, создающая полный набор файлов.
    """
    names = [path.name for path in llm.iterdir()]
    for path in llm.iterdir():
        path.unlink()
    llm.rmdir()

    def download(**kwargs: object) -> str:
        """Создаёт файлы в заранее подготовленном каталоге.

        :param kwargs: Параметры обращения к Hub.
        :return: Каталог локального snapshot.
        """
        assert llm.is_dir()
        assert kwargs["local_dir"] == str(llm)
        for name in names:
            (llm / name).write_bytes(b"{}")
        return str(llm)

    monkeypatch.setattr(download_models, "snapshot_download", download)
    download_models.main(llm.parent)
    assert (llm / "revision.json").is_file()


def test_complete_sharded_snapshot_is_reused(llm: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Полный набор частей весов не требует монолитного model.safetensors или сети.

    :param llm: Snapshot с индексом частей весов.
    :param monkeypatch: Подмена обращения к Hub.
    """
    write_manifest(llm)
    (llm / "model.safetensors").unlink()
    (llm / "model.safetensors.index.json").write_bytes(orjson.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}))
    for name in ("part-1.safetensors", "part-2.safetensors"):
        (llm / name).write_bytes(b"weights")
    download = Mock(side_effect=AssertionError("Полные части весов уже на диске"))
    monkeypatch.setattr(download_models, "snapshot_download", download)
    download_models.main(llm.parent)
    download.assert_not_called()
