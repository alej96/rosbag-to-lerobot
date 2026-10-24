"""Validation against a real, publicly recorded bag (not a synthetic fixture).

Synthetic bags in ``conftest.py`` cover specific sync/extraction behaviors, but none of
them prove the converter actually handles a bag recorded off real hardware: real message
layouts, clock jitter, dropped frames and multiple compressed cameras at once. These
tests download one such bag from the internet, so they're opt-in ("network") rather than
part of the default suite -- see docs/validated-bags.md for the bag's source and what's
been checked, and run `pytest -m network` to exercise them.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pytest

from bag2lerobot.config import parse_config
from bag2lerobot.convert import convert
from bag2lerobot.extract import inspect_bag

# RobotisSW/rosbag_test4 on the Hugging Face Hub: a ROS 2 Jazzy MCAP recording of a
# dual-arm mobile manipulator teleop session (25-joint JointState plus three
# CompressedImage cameras). Public and ungated, ~43MB. Not affiliated with this
# project and not guaranteed to stay published; a download failure skips rather than
# fails these tests.
_BAG_URL = "https://huggingface.co/datasets/RobotisSW/rosbag_test4/resolve/main"
_BAG_FILES = ("0_0.mcap", "metadata.yaml")
_CAMERA_TOPIC = "/camera_left/camera_left/color/image_rect_raw/compressed"


@pytest.fixture
def real_bag(pytestconfig, tmp_path) -> Path:
    """The downloaded bag directory, cached under pytest's cache dir across runs."""
    bag_dir = Path(pytestconfig.cache.mkdir("bag2lerobot-real-bags")) / "rosbag_test4"
    try:
        bag_dir.mkdir(parents=True, exist_ok=True)
        for name in _BAG_FILES:
            dest = bag_dir / name
            if not dest.exists():
                with urllib.request.urlopen(f"{_BAG_URL}/{name}", timeout=30) as resp:
                    dest.write_bytes(resp.read())
    except (urllib.error.URLError, OSError) as exc:
        pytest.skip(f"could not download real-bag fixture (no network?): {exc}")
    return bag_dir


@pytest.mark.network
def test_real_bag_inspects(real_bag):
    by_topic = {t.topic: t for t in inspect_bag(real_bag)}
    assert by_topic["/joint_states"].msgtype == "sensor_msgs/msg/JointState"
    assert by_topic["/joint_states"].count > 0
    assert by_topic[_CAMERA_TOPIC].msgtype == "sensor_msgs/msg/CompressedImage"


def _real_bag_config() -> dict:
    return {
        "fps": 100,
        "task": "teleop demo",
        "features": {
            "observation.state": {
                "topic": "/joint_states",
                "field": "position",
                "names_from": "name",
            },
            "observation.images.camera_left": {"topic": _CAMERA_TOPIC, "type": "image"},
        },
    }


@pytest.mark.network
def test_real_bag_converts_cleanly(real_bag, tmp_path):
    pytest.importorskip("av")
    pytest.importorskip("PIL")
    cfg = parse_config(_real_bag_config())
    reports = convert([real_bag], cfg, tmp_path / "ds")

    assert len(reports) == 1
    sync = reports[0].sync
    assert sync.frames > 500  # ~5.7s at 100fps in this recording
    for stream in sync.streams:
        # real hardware jitter is expected; the bag must still convert without ever
        # dropping a frame, and nearly every one should be fresh, not filled from stale
        # data (hold/linear only fills real gaps, like a camera at a lower rate than fps)
        assert stream.stale_frames <= sync.frames * 0.05


@pytest.mark.network
@pytest.mark.lerobot
def test_real_bag_loads_in_lerobot(real_bag, tmp_path):
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    pytest.importorskip("av")
    pytest.importorskip("PIL")
    cfg = parse_config(_real_bag_config())
    out = tmp_path / "ds"
    convert([real_bag], cfg, out)

    ds = lerobot_dataset.LeRobotDataset("local/bag2lerobot-real-bag-test", root=out)
    assert ds.num_episodes == 1
    item = ds[len(ds) - 1]
    assert tuple(item["observation.state"].shape) == (25,)
