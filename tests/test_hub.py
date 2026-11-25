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


def test_dataset_card_stale_frames_not_double_counted_across_streams(tmp_path, make_bag):
    """The card's per-episode 'stale frames' column summed StreamReport.stale_frames
    across every configured stream, so a frame that's stale on more than one stream at
    once (e.g. two features reading the same topic that loses data) was counted once per
    affected stream instead of once per frame -- it could report more stale frames than
    the episode even has. SyncResult.stale_frames (a max across streams) already exists
    to answer this correctly; the card just didn't use it."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "fps: 30\n"
        "features:\n"
        "  action:\n"
        "    topic: /leader/joint_states\n"
        "    field: position\n"
        "    names_from: name\n"
        "  action2:\n"
        "    topic: /leader/joint_states\n"
        "    field: velocity\n"
        "    names_from: name\n"
    )
    bag = make_bag(spec=BagSpec(leader_gap=(1.0, 2.0)))
    out = tmp_path / "ds"
    cfg = load_config(cfg_path)
    reports = convert([bag], cfg, out)
    info = json.loads((out / "meta" / "info.json").read_text())

    # both features read the same topic, so the same frames go stale on both at once:
    # the episode-level count must equal that shared per-stream count, not their sum.
    streams = reports[0].sync.streams
    assert reports[0].sync.stale_frames > 0
    assert {s.stale_frames for s in streams} == {reports[0].sync.stale_frames}

    card = dataset_card("me/my-dataset", cfg, info, reports)
    row = next(line for line in card.splitlines() if line.startswith(f"| {bag.name} |"))
    reported_stale = int(row.split("|")[4].strip())
    assert reported_stale == reports[0].sync.stale_frames


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
