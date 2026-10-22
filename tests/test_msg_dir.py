"""'msg_dir' registers custom message types for bags whose types rosbags doesn't bundle.

Bags written by rosbags' own writer always embed a usable definition (even for custom
types), so reading them never actually needs 'msg_dir'. To exercise the real failure this
option fixes -- some ROS 2 storage plugins/distros only ever record the type *name*, not
its schema -- these tests strip the embedded definition rosbags writes before reading the
bag back, the same way such a bag would arrive with no schema at all.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from rosbags.rosbag2 import StoragePlugin
from rosbags.rosbag2 import Writer as Writer2
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from bag2lerobot.cli import main
from bag2lerobot.config import ConfigError, load_config, parse_config
from bag2lerobot.extract import ExtractError, extract_streams, inspect_bag
from bag2lerobot.init import generate_config

GRIPPER_MSGDEF = "float64 force\n"
GRIPPER_TYPE = "test_msgs/msg/GripperCmd"


def _write_bag_without_schema(path: Path, samples: list[tuple[float, float]]) -> None:
    ts = get_typestore(Stores.ROS2_HUMBLE)
    ts.register(get_types_from_msg(GRIPPER_MSGDEF, GRIPPER_TYPE))
    GripperCmd = ts.types[GRIPPER_TYPE]
    writer = Writer2(path, version=9, storage_plugin=StoragePlugin.SQLITE3)
    with writer:
        conn = writer.add_connection("/gripper_cmd", GRIPPER_TYPE, typestore=ts)
        for t, force in samples:
            writer.write(
                conn, int(t * 1e9), ts.serialize_cdr(GripperCmd(force=force), GRIPPER_TYPE)
            )
    # Simulate a storage plugin that never records the schema at all.
    db = sqlite3.connect(str(path / "bag.db3"))
    db.execute("delete from message_definitions")
    db.commit()
    db.close()


def _msg_dir(tmp_path: Path) -> Path:
    pkg = tmp_path / "msg_defs" / "test_msgs" / "msg"
    pkg.mkdir(parents=True)
    (pkg / "GripperCmd.msg").write_text(GRIPPER_MSGDEF)
    return tmp_path / "msg_defs"


def _config(msg_dir: Path | None = None) -> dict:
    data = {"fps": 10, "features": {"gripper": {"topic": "/gripper_cmd", "field": "force"}}}
    if msg_dir is not None:
        data["msg_dir"] = str(msg_dir)
    return data


def test_custom_type_without_msg_dir_fails_clearly(tmp_path):
    bag = tmp_path / "bag"
    _write_bag_without_schema(bag, [(0.0, 1.0), (0.1, 2.0)])
    cfg = parse_config(_config())
    with pytest.raises(ExtractError, match="msg_dir"):
        extract_streams(bag, cfg)


def test_custom_type_with_msg_dir_reads_correctly(tmp_path):
    bag = tmp_path / "bag"
    _write_bag_without_schema(bag, [(0.0, 1.0), (0.1, 2.0)])
    cfg = parse_config(_config(_msg_dir(tmp_path)))
    streams = extract_streams(bag, cfg)
    assert streams["gripper"].values[:, 0].tolist() == [1.0, 2.0]


def test_inspect_bag_lists_custom_type_without_msg_dir(tmp_path):
    # inspect only reads connection metadata, never deserializes, so it works even
    # without the schema registered.
    bag = tmp_path / "bag"
    _write_bag_without_schema(bag, [(0.0, 1.0)])
    topics = inspect_bag(bag)
    assert topics[0].topic == "/gripper_cmd"
    assert topics[0].msgtype == GRIPPER_TYPE


def test_msg_dir_must_be_a_directory(tmp_path):
    with pytest.raises(ConfigError, match="not a directory"):
        parse_config(_config(tmp_path / "does-not-exist"))


def test_msg_dir_without_package_layout_errors(tmp_path):
    bag = tmp_path / "bag"
    _write_bag_without_schema(bag, [(0.0, 1.0)])
    empty = tmp_path / "empty"
    empty.mkdir()
    cfg = parse_config(_config(empty))
    with pytest.raises(ExtractError, match="no '<package>/msg"):
        extract_streams(bag, cfg)


def test_cli_inspect_accepts_msg_dir(tmp_path, capsys):
    bag = tmp_path / "bag"
    _write_bag_without_schema(bag, [(0.0, 1.0)])
    assert main(["inspect", str(bag), "--msg-dir", str(_msg_dir(tmp_path))]) == 0
    assert GRIPPER_TYPE in capsys.readouterr().out


def test_cli_convert_msg_dir_flag_registers_custom_type(tmp_path):
    bag = tmp_path / "bag"
    _write_bag_without_schema(bag, [(0.0, 1.0), (0.1, 2.0)])
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "fps: 10\nfeatures:\n  gripper:\n    topic: /gripper_cmd\n    field: force\n"
    )
    out = tmp_path / "ds"
    args = [
        "convert",
        str(bag),
        "-c",
        str(cfg_path),
        "-o",
        str(out),
        "--msg-dir",
        str(_msg_dir(tmp_path)),
    ]
    assert main(args) == 0


def test_generate_config_with_msg_dir_round_trips(make_bag, tmp_path):
    msg_dir = _msg_dir(tmp_path)
    text = generate_config(make_bag(), msg_dir=msg_dir)
    assert f"msg_dir: {msg_dir}" in text
    cfg_path = tmp_path / "gen.yaml"
    cfg_path.write_text(text)
    cfg = load_config(cfg_path)
    assert cfg.msg_dir == str(msg_dir)
