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
