import json

import numpy as np
import pandas as pd
import pytest

from bag2lerobot.config import load_config
from bag2lerobot.convert import convert
from bag2lerobot.extract import ExtractError
from tests.conftest import BagSpec

IMAGE_CONFIG = """
fps: 10
task: pick the cube
robot_type: so101_follower
features:
  observation.state:
    topic: /follower/joint_states
    field: position
    names_from: name
  observation.images.cam:
    topic: /cam/image_raw
    type: image
"""


def _read(out):
    info = json.loads((out / "meta" / "info.json").read_text())
    eps = pd.concat(
        pd.read_parquet(p)
        for p in sorted((out / "meta" / "episodes" / "chunk-000").glob("file-*.parquet"))
    ).reset_index(drop=True)
    data = pd.concat(
        pd.read_parquet(p) for p in sorted((out / "data" / "chunk-000").glob("file-*.parquet"))
    ).reset_index(drop=True)
    return info, data, eps


def test_append_adds_episodes_to_existing_dataset(make_bag, config_file, tmp_path):
    out = tmp_path / "ds"
    convert([make_bag(name="a")], load_config(config_file), out)
    info1, data1, eps1 = _read(out)
    assert info1["total_episodes"] == 1

    convert([make_bag(name="b", spec=BagSpec(seed=2))], load_config(config_file), out, append=True)
    info2, data2, eps2 = _read(out)

    assert info2["total_episodes"] == 2
    assert info2["total_frames"] == len(data2)
    assert list(eps2["episode_index"]) == [0, 1]
    # the first episode's frames are untouched, just a new file was added alongside them
    assert len(data2) > len(data1)
    pd.testing.assert_frame_equal(data2.iloc[: len(data1)].reset_index(drop=True), data1)
    assert (out / "data" / "chunk-000" / "file-001.parquet").exists()
    assert list(data2["index"]) == list(range(len(data2)))
    assert eps2.loc[1, "dataset_from_index"] == eps1.loc[0, "dataset_to_index"]


def test_append_keeps_frame_and_episode_indices_continuous(make_bag, config_file, tmp_path):
    out = tmp_path / "ds"
    bags = [make_bag(name="a"), make_bag(name="b", spec=BagSpec(seed=1))]
    convert(bags, load_config(config_file), out)
    convert([make_bag(name="c", spec=BagSpec(seed=2))], load_config(config_file), out, append=True)
    _, data, eps = _read(out)

    assert list(eps["episode_index"]) == [0, 1, 2]
    assert list(data["index"]) == list(range(len(data)))
    for _, group in data.groupby("episode_index"):
        assert list(group["frame_index"]) == list(range(len(group)))


def test_append_twice_keeps_incrementing_files_and_indices(make_bag, config_file, tmp_path):
    out = tmp_path / "ds"
    convert([make_bag(name="a")], load_config(config_file), out)
    convert([make_bag(name="b", spec=BagSpec(seed=2))], load_config(config_file), out, append=True)
    convert([make_bag(name="c", spec=BagSpec(seed=3))], load_config(config_file), out, append=True)

    info, data, eps = _read(out)
    assert info["total_episodes"] == 3
    assert list(eps["episode_index"]) == [0, 1, 2]
    assert list(data["index"]) == list(range(len(data)))
    assert (out / "data" / "chunk-000" / "file-002.parquet").exists()
    assert (out / "meta" / "episodes" / "chunk-000" / "file-002.parquet").exists()


def test_append_merges_tasks(make_bag, config_file, tmp_path):
    cfg = load_config(config_file)
    out = tmp_path / "ds"
    cfg.task = "pick the cube"
    convert([make_bag(name="a")], cfg, out)

    cfg2 = load_config(config_file)
    cfg2.task = "stack the cube"
    convert([make_bag(name="b", spec=BagSpec(seed=2))], cfg2, out, append=True)

    tasks = pd.read_parquet(out / "meta" / "tasks.parquet")
    assert set(tasks.index) == {"pick the cube", "stack the cube"}
    assert tasks.loc["pick the cube", "task_index"] == 0  # the original task keeps its index
    _, data, eps = _read(out)
    assert eps.loc[0, "tasks"] == ["pick the cube"]
    assert eps.loc[1, "tasks"] == ["stack the cube"]


def test_append_merges_global_stats(make_bag, config_file, tmp_path):
    out = tmp_path / "ds"
    convert([make_bag(name="a")], load_config(config_file), out)
    convert([make_bag(name="b", spec=BagSpec(seed=2))], load_config(config_file), out, append=True)

    _, data, _ = _read(out)
    stats = json.loads((out / "meta" / "stats.json").read_text())
    state = np.stack(data["observation.state"]).astype(np.float64)
    assert stats["observation.state"]["count"] == [len(data)]
    np.testing.assert_allclose(stats["observation.state"]["mean"], state.mean(axis=0), atol=1e-4)
    np.testing.assert_allclose(stats["observation.state"]["std"], state.std(axis=0), atol=1e-4)
    np.testing.assert_allclose(stats["observation.state"]["min"], state.min(axis=0))
    np.testing.assert_allclose(stats["observation.state"]["max"], state.max(axis=0))


def test_append_rejects_fps_mismatch(make_bag, config_file, tmp_path):
    out = tmp_path / "ds"
    convert([make_bag(name="a")], load_config(config_file), out)
    cfg = load_config(config_file)
    cfg.fps = 60
    with pytest.raises(ValueError, match="fps mismatch"):
        convert([make_bag(name="b")], cfg, out, append=True)


def test_append_rejects_feature_mismatch(make_bag, config_file, tmp_path):
    out = tmp_path / "ds"
    convert([make_bag(name="a")], load_config(config_file), out)
    cfg = load_config(config_file)
    cfg.features = cfg.features[:-1]  # drop observation.ee_pose
    with pytest.raises(ValueError, match="feature mismatch"):
        convert([make_bag(name="b")], cfg, out, append=True)


def test_append_rejects_overwrite_combo(make_bag, config_file, tmp_path):
    out = tmp_path / "ds"
    convert([make_bag(name="a")], load_config(config_file), out)
    with pytest.raises(ValueError, match="--overwrite and --append"):
        convert([make_bag(name="b")], load_config(config_file), out, overwrite=True, append=True)


def test_append_to_missing_dataset_rejected(make_bag, config_file, tmp_path):
    out = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError, match="no dataset to append to"):
        convert([make_bag()], load_config(config_file), out, append=True)


def test_append_keeps_joint_order_stable_across_calls(make_bag, config_file, tmp_path):
    out = tmp_path / "ds"
    convert([make_bag(name="a")], load_config(config_file), out)
    # second bag's follower publishes joints in reverse order
    reversed_bag = make_bag(name="b", spec=BagSpec(seed=2, shuffle_joints_every=1))
    convert([reversed_bag], load_config(config_file), out, append=True)

    info, data, eps = _read(out)
    assert info["features"]["observation.state"]["names"] == ["shoulder", "elbow", "gripper"]
    ep1 = data[data["episode_index"] == 1]
    assert len(ep1) == eps.loc[1, "length"]


def test_seed_joint_order_reads_names_from_existing_dataset(make_bag, config_file, tmp_path):
    from bag2lerobot.convert import _seed_joint_order
    from bag2lerobot.extract import _canonical_order

    out = tmp_path / "ds"
    convert([make_bag(name="a")], load_config(config_file), out)
    cfg = load_config(config_file)
    joint_order = _seed_joint_order(out, cfg)
    assert joint_order == {
        "observation.state": ["shoulder", "elbow", "gripper"],
        "action": ["shoulder", "elbow", "gripper"],
    }

    # a later bag whose driver publishes a genuinely different joint set is rejected,
    # the same way it would be within a single multi-bag conversion.
    spec = cfg.feature("observation.state")
    with pytest.raises(ExtractError, match="joint names changed"):
        _canonical_order(np.zeros(3), ["shoulder", "elbow", "wrist"], spec, joint_order)


def test_append_video_feature_adds_new_file(make_bag, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(IMAGE_CONFIG)
    out = tmp_path / "ds"
    bag_a = make_bag(spec=BagSpec(seconds=1.0, camera_hz=10, seed=1), name="a")
    bag_b = make_bag(spec=BagSpec(seconds=1.0, camera_hz=10, seed=2), name="b")
    convert([bag_a], load_config(cfg_path), out)
    convert([bag_b], load_config(cfg_path), out, append=True)

    info, _, eps = _read(out)
    video_dir = out / "videos" / "observation.images.cam" / "chunk-000"
    assert (video_dir / "file-000.mp4").exists()
    assert (video_dir / "file-001.mp4").exists()
    assert eps.loc[1, "videos/observation.images.cam/file_index"] == 1
    assert eps.loc[1, "videos/observation.images.cam/from_timestamp"] == 0.0  # new file's own clock


def test_append_rejects_image_resolution_mismatch(make_bag, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(IMAGE_CONFIG)
    out = tmp_path / "ds"
    convert(
        [make_bag(spec=BagSpec(seconds=1.0, camera_hz=10, camera_size=(8, 6)), name="a")],
        load_config(cfg_path),
        out,
    )
    with pytest.raises(ValueError, match="resolution"):
        convert(
            [make_bag(spec=BagSpec(seconds=1.0, camera_hz=10, camera_size=(4, 3)), name="b")],
            load_config(cfg_path),
            out,
            append=True,
        )
