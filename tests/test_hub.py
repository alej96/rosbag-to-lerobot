"""--push-to-hub: dataset card generation and the Hugging Face Hub upload call."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from bag2lerobot.cli import main
from bag2lerobot.config import load_config
from bag2lerobot.convert import convert
from bag2lerobot.hub import HubError, dataset_card, push_to_hub
from tests.conftest import BagSpec


def test_push_to_hub_without_extra_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(HubError, match="hub"):
        push_to_hub(tmp_path, "me/my-dataset")


def test_push_to_hub_creates_repo_and_uploads(tmp_path, monkeypatch):
    calls = {}

    class FakeApi:
        def create_repo(self, repo_id, repo_type, private, exist_ok):
            calls["create_repo"] = (repo_id, repo_type, private, exist_ok)

        def upload_folder(self, repo_id, repo_type, folder_path):
            calls["upload_folder"] = (repo_id, repo_type, folder_path)

    fake_module = type(sys)("huggingface_hub")
    fake_module.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    url = push_to_hub(tmp_path, "me/my-dataset", private=True)

    assert url == "https://huggingface.co/datasets/me/my-dataset"
    assert calls["create_repo"] == ("me/my-dataset", "dataset", True, True)
    assert calls["upload_folder"] == ("me/my-dataset", "dataset", str(tmp_path))


def test_push_to_hub_with_changed_files_uploads_only_those(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "new.parquet").write_text("x")
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}")
    calls = {"upload_file": [], "upload_folder": 0}

    class FakeApi:
        def create_repo(self, repo_id, repo_type, private, exist_ok):
            pass

        def upload_folder(self, repo_id, repo_type, folder_path):
            calls["upload_folder"] += 1

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type):
            calls["upload_file"].append((path_or_fileobj, path_in_repo))

    fake_module = type(sys)("huggingface_hub")
    fake_module.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    changed = [Path("data/new.parquet"), Path("meta/info.json")]
    push_to_hub(tmp_path, "me/my-dataset", changed_files=changed)

    assert calls["upload_folder"] == 0
    assert calls["upload_file"] == [
        (str(tmp_path / "data" / "new.parquet"), "data/new.parquet"),
        (str(tmp_path / "meta" / "info.json"), "meta/info.json"),
    ]


def test_dataset_card_summarizes_bags_config_and_sync(make_bag, config_file, tmp_path):
    bag = make_bag(name="demo")
    out = tmp_path / "ds"
    reports = convert([bag], load_config(config_file), out)
    info = json.loads((out / "meta" / "info.json").read_text())

    card = dataset_card("me/my-dataset", load_config(config_file), info, reports)

    assert "me/my-dataset" in card
    assert "demo" in card  # source bag name
    assert "pick the cube" in card  # task, from config_file
    assert "so101_follower" in card  # robot_type, from config_file
    assert "observation.state" in card  # a real feature
    assert "observation.images" not in card  # no image feature in this config
    assert "timestamp" not in card  # index/bookkeeping column, not a user feature
    assert str(info["total_episodes"]) in card


def test_cli_push_to_hub_writes_card_and_uploads(
    make_bag, config_file, tmp_path, monkeypatch, capsys
):
    calls = []

    def fake_push(out, repo_id, private=False, changed_files=None):
        calls.append((out, repo_id, private))
        return f"https://huggingface.co/datasets/{repo_id}"

    monkeypatch.setattr("bag2lerobot.cli.push_to_hub", fake_push)
    bag = make_bag()
    out = tmp_path / "ds"

    rc = main(
        [
            "convert",
            str(bag),
            "-c",
            str(config_file),
            "-o",
            str(out),
            "--push-to-hub",
            "me/my-dataset",
            "--private",
        ]
    )

    assert rc == 0
    assert calls == [(out, "me/my-dataset", True)]
    assert "pushed to https://huggingface.co/datasets/me/my-dataset" in capsys.readouterr().out
    assert (out / "README.md").exists()
    assert "me/my-dataset" in (out / "README.md").read_text()


def test_cli_push_to_hub_after_append_uploads_only_new_and_metadata_files(
    make_bag, config_file, tmp_path, monkeypatch
):
    calls = []

    def fake_push(out, repo_id, private=False, changed_files=None):
        calls.append(changed_files)
        return f"https://huggingface.co/datasets/{repo_id}"

    monkeypatch.setattr("bag2lerobot.cli.push_to_hub", fake_push)
    out = tmp_path / "ds"

    rc = main(
        [
            "convert",
            str(make_bag(name="a")),
            "-c",
            str(config_file),
            "-o",
            str(out),
            "--push-to-hub",
            "me/my-dataset",
        ]
    )
    assert rc == 0
    assert calls == [None]  # the first (non-append) push still uploads the whole folder

    rc = main(
        [
            "convert",
            str(make_bag(name="b", spec=BagSpec(seed=2))),
            "-c",
            str(config_file),
            "-o",
            str(out),
            "--append",
            "--push-to-hub",
            "me/my-dataset",
        ]
    )
    assert rc == 0
    changed_files = calls[1]
    assert changed_files is not None
    # the new data/episode files are included...
    assert Path("data/chunk-000/file-001.parquet") in changed_files
    assert Path("meta/episodes/chunk-000/file-001.parquet") in changed_files
    # ...the rewritten-in-place metadata files are included...
    assert Path("meta/info.json") in changed_files
    assert Path("meta/stats.json") in changed_files
    assert Path("meta/tasks.parquet") in changed_files
    assert Path("README.md") in changed_files
    # ...but the first episode's untouched data file is not re-uploaded
    assert Path("data/chunk-000/file-000.parquet") not in changed_files
