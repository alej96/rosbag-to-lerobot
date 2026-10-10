import json

import numpy as np
import pandas as pd
import pytest

from bag2lerobot.config import ConfigError, load_config
from bag2lerobot.convert import convert
from bag2lerobot.extract import ExtractError, extract_streams
from tests.conftest import JOINTS, BagSpec, signal


def _read(out):
    info = json.loads((out / "meta" / "info.json").read_text())
    data = pd.read_parquet(out / "data" / "chunk-000" / "file-000.parquet")
    eps = pd.read_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return info, data, eps


@pytest.mark.parametrize("kind", ["ros1", "sqlite3", "mcap"])
def test_converts_every_bag_format(make_bag, config_file, tmp_path, kind):
    bag = make_bag(kind)
    out = tmp_path / "ds"
    reports = convert([bag], load_config(config_file), out)
    info, data, eps = _read(out)

    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == 30
    # leader starts 0.2s late and everything ends ~3.0s: ~2.8s overlap at 30 fps
    assert 80 <= info["total_frames"] <= 86
    assert info["total_frames"] == len(data) == reports[0].sync.frames
    assert info["features"]["observation.state"]["names"] == JOINTS
    assert info["features"]["observation.state"]["shape"] == [3]
    assert info["features"]["observation.ee_pose"]["names"] == [
        "pose.position.x",
        "pose.position.y",
        "pose.position.z",
        "pose.orientation.x",
        "pose.orientation.y",
        "pose.orientation.z",
        "pose.orientation.w",
    ]
    np.testing.assert_allclose(data["timestamp"], np.arange(len(data)) / 30, atol=1e-6)
    assert list(data["frame_index"]) == list(range(len(data)))
    assert eps.loc[0, "length"] == len(data)
    assert reports[0].sync.trimmed_start_s == pytest.approx(0.2, abs=0.03)


def test_hold_sampling_uses_latest_sample(make_bag, config_file, tmp_path):
    out = tmp_path / "ds"
    reports = convert([make_bag()], load_config(config_file), out)
    _, data, _ = _read(out)
    state = np.stack(data["observation.state"])
    t = reports[0].sync.t0 - 1_700_000_000.0 + data["timestamp"].to_numpy()
    # follower publishes at 100 Hz: the held value is at most ~1 period old
    err = np.abs(state - signal(t)).max()
    assert err < 0.02, err


def test_linear_sync_is_more_accurate(make_bag, config_file, tmp_path):
    cfg = load_config(config_file)
    cfg.sync = "linear"
    out = tmp_path / "ds"
    reports = convert([make_bag()], cfg, out)
    _, data, _ = _read(out)
    state = np.stack(data["observation.state"])
    t = reports[0].sync.t0 - 1_700_000_000.0 + data["timestamp"].to_numpy()
    assert np.abs(state - signal(t)).max() < 1e-3


def test_gap_in_a_stream_is_reported(make_bag, config_file, tmp_path):
    bag = make_bag(spec=BagSpec(leader_gap=(1.0, 2.0)))
    reports = convert([bag], load_config(config_file), tmp_path / "ds")
    leader = next(s for s in reports[0].sync.streams if s.key == "action")
    assert leader.max_gap_s == pytest.approx(1.0, abs=0.05)
    # stale once data is older than 0.25s: roughly 0.75s of the 1s gap at 30 fps
    assert 18 <= leader.stale_frames <= 26
    others = [s for s in reports[0].sync.streams if s.key != "action"]
    assert all(s.stale_frames == 0 for s in others)


def test_joint_order_is_stable_when_publisher_reorders(make_bag, config_file, tmp_path):
    bag = make_bag(spec=BagSpec(shuffle_joints_every=3))
    out = tmp_path / "ds"
    reports = convert([bag], load_config(config_file), out)
    _, data, _ = _read(out)
    state = np.stack(data["observation.state"])
    t = reports[0].sync.t0 - 1_700_000_000.0 + data["timestamp"].to_numpy()
    assert np.abs(state - signal(t)).max() < 0.02


def test_multiple_bags_become_episodes(make_bag, config_file, tmp_path):
    bags = [
        make_bag("mcap", BagSpec(seconds=2.0, seed=1), name="a"),
        make_bag("ros1", BagSpec(seconds=3.0, seed=2), name="b"),
    ]
    out = tmp_path / "ds"
    convert(bags, load_config(config_file), out)
    info, data, eps = _read(out)

    assert info["total_episodes"] == 2
    assert list(eps["episode_index"]) == [0, 1]
    assert eps.loc[0, "dataset_to_index"] == eps.loc[1, "dataset_from_index"]
    assert eps.loc[1, "dataset_to_index"] == len(data)
    assert list(data["index"]) == list(range(len(data)))
    for _, group in data.groupby("episode_index"):
        assert list(group["frame_index"]) == list(range(len(group)))
        assert group["timestamp"].iloc[0] == 0


PER_FEATURE_SYNC_CONFIG = """
fps: 30
features:
  observation.state:
    topic: /follower/joint_states
    field: position
    names_from: name
    sync: hold
  action:
    topic: /leader/joint_states
    field: position
    names_from: name
sync:
  method: linear
"""


def test_per_feature_sync_overrides_global(make_bag, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(PER_FEATURE_SYNC_CONFIG)
    out = tmp_path / "ds"
    reports = convert([make_bag()], load_config(cfg_path), out)
    _, data, _ = _read(out)
    t = reports[0].sync.t0 - 1_700_000_000.0 + data["timestamp"].to_numpy()

    # observation.state pins "hold" even though sync.method is "linear" globally.
    state = np.stack(data["observation.state"])
    assert np.abs(state - signal(t)).max() > 0.003
    # action keeps the global "linear" method.
    action = np.stack(data["action"])
    assert np.abs(action - (signal(t) + 1.0)).max() < 1e-3


PER_FEATURE_STALENESS_CONFIG = """
fps: 30
features:
  observation.state:
    topic: /follower/joint_states
    field: position
    names_from: name
  action:
    topic: /leader/joint_states
    field: position
    names_from: name
    max_staleness_s: 5.0
"""


def test_per_feature_max_staleness_overrides_global(make_bag, tmp_path):
    bag = make_bag(spec=BagSpec(leader_gap=(1.0, 2.0)))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(PER_FEATURE_STALENESS_CONFIG)
    reports = convert([bag], load_config(cfg_path), tmp_path / "ds")

    action = next(s for s in reports[0].sync.streams if s.key == "action")
    assert action.max_staleness_s == 5.0
    assert action.stale_frames == 0  # the lenient override absorbs the 1s gap

    state = next(s for s in reports[0].sync.streams if s.key == "observation.state")
    assert state.max_staleness_s == 0.25  # unaffected default


def test_invalid_feature_sync_override_rejected(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("fps: 30\nfeatures:\n  action:\n    topic: /x\n    sync: nearest\n")
    with pytest.raises(ConfigError, match="sync must be one of"):
        load_config(cfg_path)


def test_negative_feature_max_staleness_rejected(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("fps: 30\nfeatures:\n  action:\n    topic: /x\n    max_staleness_s: -1\n")
    with pytest.raises(ConfigError, match="must not be negative"):
        load_config(cfg_path)


def test_missing_topic_lists_available(make_bag, tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("fps: 30\nfeatures:\n  action:\n    topic: /nope\n")
    with pytest.raises(ExtractError, match="/nope.*available: /ee_pose"):
        extract_streams(make_bag(), load_config(cfg_path))


def test_refuses_to_overwrite(make_bag, config_file, tmp_path):
    out = tmp_path / "ds"
    convert([make_bag()], load_config(config_file), out)
    with pytest.raises(FileExistsError):
        convert([make_bag(name="again")], load_config(config_file), out)
    convert([make_bag(name="again2")], load_config(config_file), out, overwrite=True)
