"""--push-to-hub: dataset card generation and the Hugging Face Hub upload call."""

from __future__ import annotations

import json
import sys

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


def test_dataset_card_summarizes_bags_config_and_sync(make_bag, config_file, tmp_path):
    bag = make_bag(name="demo")
    out = tmp_path / "ds"
    convert([bag], load_config(config_file), out)
    info = json.loads((out / "meta" / "info.json").read_text())
    source_log = json.loads((out / "meta" / "source_bags.json").read_text())

    card = dataset_card("me/my-dataset", load_config(config_file), info, source_log)

    assert "me/my-dataset" in card
    assert "demo" in card  # source bag name
    assert "pick the cube" in card  # task, from config_file
    assert "so101_follower" in card  # robot_type, from config_file
    assert "observation.state" in card  # a real feature
    assert "observation.images" not in card  # no image feature in this config
    assert "timestamp" not in card  # index/bookkeeping column, not a user feature
    assert str(info["total_episodes"]) in card


def test_dataset_card_after_append_still_lists_earlier_bags(make_bag, config_file, tmp_path):
    """Before this, the card was built straight from the ConversionReports of the most
    recent `convert` call, so pushing again after `--append` produced a README that
    silently dropped every bag converted in earlier calls, even though their episodes
    are still in the dataset."""
    out = tmp_path / "ds"
    convert([make_bag(name="first_session")], load_config(config_file), out)
    convert(
        [make_bag(name="second_session", spec=BagSpec(seed=2))],
        load_config(config_file),
        out,
        append=True,
    )
    info = json.loads((out / "meta" / "info.json").read_text())
    source_log = json.loads((out / "meta" / "source_bags.json").read_text())
    assert len(source_log) == 2

    card = dataset_card("me/my-dataset", load_config(config_file), info, source_log)

    assert "first_session" in card
    assert "second_session" in card


def test_cli_push_to_hub_writes_card_and_uploads(
    make_bag, config_file, tmp_path, monkeypatch, capsys
):
    calls = []

    def fake_push(out, repo_id, private=False):
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
