import pytest

from bag2lerobot.cli import main
from tests.conftest import BagSpec


def test_inspect(make_bag, capsys):
    assert main(["inspect", str(make_bag())]) == 0
    out = capsys.readouterr().out
    assert "/follower/joint_states" in out
    assert "sensor_msgs/msg/JointState" in out
    assert "100.0Hz" in out or "99." in out


def test_convert_cli_and_strict(make_bag, config_file, tmp_path, capsys):
    clean = make_bag(name="clean")
    gappy = make_bag(spec=BagSpec(leader_gap=(1.0, 2.0)), name="gappy")
    assert main(["convert", str(clean), "-c", str(config_file), "-o", str(tmp_path / "a")]) == 0
    assert "wrote 1 episode(s)" in capsys.readouterr().out

    args = ["convert", str(gappy), "-c", str(config_file), "-o", str(tmp_path / "b")]
    assert main(args) == 0
    assert "stale frame" in capsys.readouterr().out
    assert main([*args, "--overwrite", "--strict"]) == 1


def test_bad_config(tmp_path, make_bag, capsys):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("features: {}\n")
    assert main(["convert", str(make_bag()), "-c", str(cfg), "-o", str(tmp_path / "x")]) == 2
    assert "fps" in capsys.readouterr().err


@pytest.mark.lerobot
def test_output_loads_in_lerobot(make_bag, config_file, tmp_path):
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    bags = [make_bag("mcap", name="a"), make_bag("ros1", BagSpec(seed=3), name="b")]
    out = tmp_path / "ds"
    assert main(["convert", *map(str, bags), "-c", str(config_file), "-o", str(out)]) == 0

    ds = lerobot_dataset.LeRobotDataset("local/bag2lerobot-test", root=out)
    assert ds.num_episodes == 2
    assert ds.fps == 30
    item = ds[len(ds) - 1]
    assert tuple(item["observation.state"].shape) == (3,)
    assert tuple(item["action"].shape) == (3,)
    assert tuple(item["observation.ee_pose"].shape) == (7,)
    assert item["task"] == "pick the cube"


@pytest.mark.lerobot
def test_append_loads_in_lerobot(make_bag, config_file, tmp_path):
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    out = tmp_path / "ds"
    first = ["convert", str(make_bag("mcap", name="a")), "-c", str(config_file), "-o", str(out)]
    assert main(first) == 0
    bag_b = make_bag("ros1", BagSpec(seed=3), name="b")
    args = ["convert", str(bag_b), "-c", str(config_file), "-o", str(out), "--append"]
    assert main(args) == 0

    ds = lerobot_dataset.LeRobotDataset("local/bag2lerobot-append-test", root=out)
    assert ds.num_episodes == 2
    item = ds[len(ds) - 1]
    assert tuple(item["observation.state"].shape) == (3,)
    assert item["task"] == "pick the cube"


IMAGE_CONFIG = """
fps: 10
task: pick the cube
features:
  observation.state:
    topic: /follower/joint_states
    field: position
    names_from: name
  observation.images.cam:
    topic: /cam/image_raw
    type: image
"""


@pytest.mark.lerobot
def test_image_feature_loads_in_lerobot(make_bag, tmp_path):
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(IMAGE_CONFIG)
    bags = [
        make_bag(spec=BagSpec(seconds=1.0, camera_hz=10, seed=1), name="a"),
        make_bag(spec=BagSpec(seconds=1.0, camera_hz=10, seed=2), name="b"),
    ]
    out = tmp_path / "ds"
    assert main(["convert", *map(str, bags), "-c", str(cfg_path), "-o", str(out)]) == 0

    # pyav avoids torchcodec, whose compiled ABI is sensitive to the exact torch/ffmpeg
    # build on the runner; pyav is the other officially supported LeRobotDataset backend.
    ds = lerobot_dataset.LeRobotDataset(
        "local/bag2lerobot-video-test", root=out, video_backend="pyav"
    )
    assert ds.num_episodes == 2
    item = ds[0]
    assert tuple(item["observation.images.cam"].shape) == (3, 6, 8)
    last = ds[len(ds) - 1]
    assert tuple(last["observation.images.cam"].shape) == (3, 6, 8)
