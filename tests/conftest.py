"""Synthetic ROS 1 / ROS 2 (sqlite3, mcap) bags with known signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from rosbags.rosbag1 import Writer as Writer1
from rosbags.rosbag2 import StoragePlugin
from rosbags.rosbag2 import Writer as Writer2
from rosbags.typesys import Stores, get_typestore

JOINTS = ["shoulder", "elbow", "gripper"]
T0 = 1_700_000_000.0  # realistic epoch seconds, exercises float precision


def signal(t: np.ndarray | float) -> np.ndarray:
    """Joint positions as a function of time (s since start)."""
    t = np.asarray(t, dtype=np.float64)
    return np.stack([np.sin(t), np.cos(t), 0.1 * t], axis=-1)


@dataclass
class BagSpec:
    seconds: float = 3.0
    follower_hz: float = 100.0
    leader_hz: float = 50.0
    pose_hz: float = 30.0
    leader_start_s: float = 0.2  # leader starts late: exercises overlap trimming
    leader_gap: tuple[float, float] | None = None  # (start, end) seconds with no leader msgs
    shuffle_joints_every: int = 0  # publish follower joints in reversed order every N msgs
    jitter_s: float = 0.002
    seed: int = 0
    extra_topics: dict[str, float] = field(default_factory=dict)


def _times(hz: float, start: float, end: float, jitter: float, rng) -> np.ndarray:
    t = np.arange(start, end, 1.0 / hz)
    return np.sort(t + rng.uniform(-jitter, jitter, len(t)).clip(-0.4 / hz, 0.4 / hz))


def write_bag(path: Path, kind: str, spec: BagSpec) -> Path:
    rng = np.random.default_rng(spec.seed)
    store = Stores.ROS1_NOETIC if kind == "ros1" else Stores.ROS2_HUMBLE
    ts = get_typestore(store)
    T = ts.types
    Header, Time = T["std_msgs/msg/Header"], T["builtin_interfaces/msg/Time"]

    def header(t: float):
        sec = int(T0 + t)
        kw = dict(stamp=Time(sec=sec, nanosec=int(round((T0 + t - sec) * 1e9))), frame_id="base")
        if "seq" in Header.__dataclass_fields__:
            kw["seq"] = 0
        return Header(**kw)

    def joint_state(t: float, reverse: bool = False, offset: float = 0.0):
        pos = signal(t) + offset
        names = list(JOINTS)
        if reverse:
            names, pos = names[::-1], pos[::-1]
        empty = np.array([], dtype=np.float64)
        return T["sensor_msgs/msg/JointState"](
            header=header(t),
            name=names,
            position=np.ascontiguousarray(pos, dtype=np.float64),
            velocity=empty,
            effort=empty,
        )

    def pose(t: float):
        P, Pt, Q = (
            T["geometry_msgs/msg/Pose"],
            T["geometry_msgs/msg/Point"],
            T["geometry_msgs/msg/Quaternion"],
        )
        return T["geometry_msgs/msg/PoseStamped"](
            header=header(t),
            pose=P(
                position=Pt(x=float(np.sin(t)), y=0.5, z=float(t)),
                orientation=Q(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )

    msgs = []  # (t, topic, msgtype, msg)
    for i, t in enumerate(_times(spec.follower_hz, 0, spec.seconds, spec.jitter_s, rng)):
        rev = bool(spec.shuffle_joints_every) and i % spec.shuffle_joints_every == 1
        msgs.append(
            (t, "/follower/joint_states", "sensor_msgs/msg/JointState", joint_state(t, rev))
        )
    for t in _times(spec.leader_hz, spec.leader_start_s, spec.seconds, spec.jitter_s, rng):
        if spec.leader_gap and spec.leader_gap[0] <= t < spec.leader_gap[1]:
            continue
        msgs.append(
            (t, "/leader/joint_states", "sensor_msgs/msg/JointState", joint_state(t, offset=1.0))
        )
    for t in _times(spec.pose_hz, 0, spec.seconds, spec.jitter_s, rng):
        msgs.append((t, "/ee_pose", "geometry_msgs/msg/PoseStamped", pose(t)))
    msgs.sort(key=lambda m: m[0])

    if kind == "ros1":
        writer = Writer1(path)
        serialize = ts.serialize_ros1
    else:
        plugin = StoragePlugin.MCAP if kind == "mcap" else StoragePlugin.SQLITE3
        writer = Writer2(path, version=9, storage_plugin=plugin)
        serialize = ts.serialize_cdr
    with writer:
        conns = {}
        for topic, msgtype in {(m[1], m[2]) for m in msgs}:
            conns[topic] = writer.add_connection(topic, msgtype, typestore=ts)
        for t, topic, msgtype, msg in msgs:
            # bag receive time lags the header stamp by a few ms, like a real recorder
            writer.write(conns[topic], int((T0 + t + 0.003) * 1e9), serialize(msg, msgtype))
    return path


CONFIG = """
fps: 30
robot_type: so101_follower
task: pick the cube
features:
  observation.state:
    topic: /follower/joint_states
    field: position
    names_from: name
  action:
    topic: /leader/joint_states
    field: position
    names_from: name
  observation.ee_pose:
    topic: /ee_pose
    fields: [pose.position, pose.orientation]
"""


@pytest.fixture
def config_file(tmp_path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG)
    return path


@pytest.fixture
def make_bag(tmp_path):
    def _make(kind: str = "mcap", spec: BagSpec | None = None, name: str | None = None) -> Path:
        suffix = ".bag" if kind == "ros1" else ""
        path = tmp_path / f"{name or kind}{suffix}"
        return write_bag(path, kind, spec or BagSpec())

    return _make
