import pytest

from bag2lerobot import init as init_mod
from bag2lerobot.cli import main
from bag2lerobot.config import load_config
from bag2lerobot.extract import ExtractError, TopicInfo
from bag2lerobot.init import _assign_joint_roles, _camera_names, generate_config
from tests.conftest import BagSpec

FOLLOWER, LEADER, AUX = "/follower/joint_states", "/leader/joint_states", "/aux/joint_states"


def test_assign_joint_roles_by_name_hint():
    assigned = _assign_joint_roles([FOLLOWER, LEADER])
    assert assigned == [
        ("observation.state", FOLLOWER, False),
        ("action", LEADER, False),
    ]


def test_assign_joint_roles_single_topic_falls_back_to_state():
    assigned = _assign_joint_roles(["/robot/joint_states"])
    assert assigned == [("observation.state", "/robot/joint_states", True)]


def test_assign_joint_roles_two_unclassified_use_order():
    assigned = _assign_joint_roles(["/armA/joint_states", "/armB/joint_states"])
    assert assigned == [
        ("observation.state", "/armA/joint_states", True),
        ("action", "/armB/joint_states", True),
    ]


def test_assign_joint_roles_extra_topic_is_flagged():
    assigned = _assign_joint_roles([FOLLOWER, LEADER, AUX])
    assert assigned == [
        ("observation.state", FOLLOWER, False),
        ("action", LEADER, False),
        ("observation.state_2", AUX, True),
    ]


def test_camera_names_dedupes_on_collision():
    names = _camera_names(["/cam/image_compressed", "/cam/image_raw"])
    assert names == {"/cam/image_compressed": "cam", "/cam/image_raw": "cam_2"}


def test_camera_names_two_cameras():
    names = _camera_names(["/cam_left/image_raw", "/cam_right/image_raw"])
    assert names == {"/cam_left/image_raw": "cam_left", "/cam_right/image_raw": "cam_right"}


def test_generate_config_default(make_bag, tmp_path):
    text = generate_config(make_bag())
    cfg_path = tmp_path / "gen.yaml"
    cfg_path.write_text(text)
    cfg = load_config(cfg_path)

    assert cfg.fps == 100  # follower publishes at ~100Hz, the fastest JointState topic
    assert cfg.task == ""
    assert cfg.feature("observation.state").topic == FOLLOWER
    assert cfg.feature("observation.state").names_from == "name"
    assert cfg.feature("action").topic == LEADER
    assert not any(f.type == "image" for f in cfg.features)


def test_generate_config_with_camera(make_bag, tmp_path):
    bag = make_bag(spec=BagSpec(camera_hz=15.0))
    cfg_path = tmp_path / "gen.yaml"
    cfg_path.write_text(generate_config(bag))
    cfg = load_config(cfg_path)

    image_feats = [f for f in cfg.features if f.type == "image"]
    assert len(image_feats) == 1
    assert image_feats[0].key == "observation.images.cam"
    assert image_feats[0].topic == "/cam/image_raw"


def test_generate_config_fps_override(make_bag):
    text = generate_config(make_bag(), fps=30)
    assert "fps: 30" in text


def test_generate_config_empty_bag_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(init_mod, "inspect_bag", lambda path, msg_dir=None: [])
    with pytest.raises(ExtractError):
        generate_config(tmp_path / "x")


def test_generate_config_no_candidates_raises(monkeypatch, tmp_path):
    topics = [TopicInfo("/ee_pose", "geometry_msgs/msg/PoseStamped", 10, 1.0)]
    monkeypatch.setattr(init_mod, "inspect_bag", lambda path, msg_dir=None: topics)
    with pytest.raises(ExtractError):
        generate_config(tmp_path / "x")


def test_cli_init_writes_loadable_config(make_bag, tmp_path):
    out_cfg = tmp_path / "gen.yaml"
    assert main(["init", str(make_bag()), "-o", str(out_cfg)]) == 0
    cfg = load_config(out_cfg)
    assert cfg.feature("observation.state").topic == FOLLOWER
    assert cfg.feature("action").topic == LEADER


def test_cli_init_prints_to_stdout(make_bag, capsys):
    assert main(["init", str(make_bag())]) == 0
    out = capsys.readouterr().out
    assert "observation.state" in out
    assert FOLLOWER in out


def test_generated_config_converts(make_bag, tmp_path):
    bag = make_bag()
    cfg_path = tmp_path / "gen.yaml"
    cfg_path.write_text(generate_config(bag).replace("task:", "task: pick the cube"))
    out = tmp_path / "ds"
    assert main(["convert", str(bag), "-c", str(cfg_path), "-o", str(out)]) == 0
