"""Tests for --msg-dir: registering custom message types for a bag with no embedded
message definitions (common for sqlite3 bags from rosbag2 releases older than the
type-description-hash feature)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from rosbags.rosbag2 import StoragePlugin
from rosbags.rosbag2 import Writer as Writer2
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from bag2lerobot.cli import main
from bag2lerobot.config import parse_config
from bag2lerobot.convert import convert
from bag2lerobot.extract import ExtractError, _msgtype_from_path, extract_streams

GRIPPER_MSGDEF = "float64 force\nbool closed\n"
GRIPPER_MSGTYPE = "my_robot/msg/GripperCmd"

CONFIG = {
    "fps": 10,
    "task": "squeeze",
    "features": {"action.gripper": {"topic": "/gripper_cmd", "field": "force"}},
}


def _write_undocumented_bag(path: Path, forces: list[float]) -> Path:
    """A bag with a custom message type, then stripped of embedded definitions."""
    ts = get_typestore(Stores.ROS2_HUMBLE)
    ts.register(get_types_from_msg(GRIPPER_MSGDEF, GRIPPER_MSGTYPE))
    gripper_cmd = ts.types[GRIPPER_MSGTYPE]
    writer = Writer2(path, version=9, storage_plugin=StoragePlugin.SQLITE3)
    with writer:
        conn = writer.add_connection("/gripper_cmd", GRIPPER_MSGTYPE, typestore=ts)
        for i, force in enumerate(forces):
            msg = gripper_cmd(force=force, closed=False)
            writer.write(conn, i * 10_000_000, ts.serialize_cdr(msg, GRIPPER_MSGTYPE))
    db = sqlite3.connect(str(path / f"{path.name}.db3"))
    db.execute("DELETE FROM message_definitions")
    db.commit()
    db.close()
    return path


@pytest.fixture
def custom_bag(tmp_path):
    return _write_undocumented_bag(tmp_path / "custom_bag", [0.1, 0.2, 0.3, 0.4, 0.5])


@pytest.fixture
def msg_dir(tmp_path):
    pkg_dir = tmp_path / "msgs" / "my_robot" / "msg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "GripperCmd.msg").write_text(GRIPPER_MSGDEF)
    return tmp_path / "msgs"


def test_custom_type_fails_without_msg_dir(custom_bag):
    cfg = parse_config(CONFIG)
    with pytest.raises(ExtractError, match="--msg-dir"):
        extract_streams(custom_bag, cfg)


def test_custom_type_succeeds_with_msg_dir(custom_bag, msg_dir):
    cfg = parse_config(CONFIG)
    streams = extract_streams(custom_bag, cfg, msg_dir=msg_dir)
    assert streams["action.gripper"].values[:, 0].tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5]
    )


def test_convert_cli_with_msg_dir(custom_bag, msg_dir, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "fps: 10\ntask: squeeze\nfeatures:\n"
        "  action.gripper:\n    topic: /gripper_cmd\n    field: force\n"
    )
    out = tmp_path / "ds"
    assert (
        main(
            [
                "convert",
                str(custom_bag),
                "-c",
                str(cfg_path),
                "-o",
                str(out),
                "--msg-dir",
                str(msg_dir),
            ]
        )
        == 0
    )
    assert (out / "meta" / "info.json").exists()


def test_convert_without_msg_dir_reports_clear_error(custom_bag, tmp_path, capsys):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "fps: 10\ntask: squeeze\nfeatures:\n"
        "  action.gripper:\n    topic: /gripper_cmd\n    field: force\n"
    )
    out = tmp_path / "ds"
    assert main(["convert", str(custom_bag), "-c", str(cfg_path), "-o", str(out)]) == 2
    assert "--msg-dir" in capsys.readouterr().err


def test_msg_dir_missing_type_still_fails(custom_bag, tmp_path):
    other_dir = tmp_path / "other_msgs" / "other_pkg" / "msg"
    other_dir.mkdir(parents=True)
    (other_dir / "Unrelated.msg").write_text("int32 x\n")
    cfg = parse_config(CONFIG)
    with pytest.raises(ExtractError, match="--msg-dir"):
        extract_streams(custom_bag, cfg, msg_dir=tmp_path / "other_msgs")


def test_msg_dir_not_a_directory(custom_bag, tmp_path):
    cfg = parse_config(CONFIG)
    with pytest.raises(ExtractError, match="not a directory"):
        extract_streams(custom_bag, cfg, msg_dir=tmp_path / "does_not_exist")


def test_msg_dir_empty(custom_bag, tmp_path):
    empty = tmp_path / "empty_msgs"
    empty.mkdir()
    cfg = parse_config(CONFIG)
    with pytest.raises(ExtractError, match="no .msg files"):
        extract_streams(custom_bag, cfg, msg_dir=empty)


def test_msgtype_from_path_standard_layout(tmp_path):
    msg = tmp_path / "my_robot" / "msg" / "GripperCmd.msg"
    assert _msgtype_from_path(msg, tmp_path) == "my_robot/msg/GripperCmd"


def test_msgtype_from_path_flat_layout(tmp_path):
    msg = tmp_path / "my_robot" / "GripperCmd.msg"
    assert _msgtype_from_path(msg, tmp_path) == "my_robot/msg/GripperCmd"


def test_msgtype_from_path_too_shallow(tmp_path):
    msg = tmp_path / "GripperCmd.msg"
    with pytest.raises(ExtractError, match="package"):
        _msgtype_from_path(msg, tmp_path)


def test_dry_run_split_boundaries_needs_msg_dir(custom_bag, msg_dir, tmp_path, capsys):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "fps: 10\ntask: squeeze\nsplit_on: /gripper_cmd\nfeatures:\n"
        "  action.gripper:\n    topic: /gripper_cmd\n    field: force\n"
    )
    assert main(["convert", str(custom_bag), "-c", str(cfg_path), "--dry-run"]) == 2
    assert "--msg-dir" in capsys.readouterr().err

    assert (
        main(
            [
                "convert",
                str(custom_bag),
                "-c",
                str(cfg_path),
                "--dry-run",
                "--msg-dir",
                str(msg_dir),
            ]
        )
        == 0
    )
    assert "episode(s)" in capsys.readouterr().out


@pytest.mark.lerobot
def test_custom_type_output_loads_in_lerobot(custom_bag, msg_dir, tmp_path):
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    cfg = parse_config(CONFIG)
    out = tmp_path / "ds"
    convert([custom_bag], cfg, out, msg_dir=msg_dir)
    ds = lerobot_dataset.LeRobotDataset(repo_id="x", root=out)
    assert len(ds) > 0
