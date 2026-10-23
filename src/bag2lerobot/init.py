"""Generate a starter YAML config from a bag's topics (``bag2lerobot init``)."""

from __future__ import annotations

from pathlib import Path

from bag2lerobot.extract import ExtractError, inspect_bag

_ACTION_HINTS = ("leader", "cmd", "command", "teleop", "target")
_STATE_HINTS = ("follower", "state", "actual", "feedback")
_CAMERA_STRIP = {"image_raw", "image_compressed", "image", "compressed", "color", "rgb", "raw"}
# std_msgs wrappers around a single numeric 'data' field (or array of them): the common
# shape for a simple command/reading that rides outside a JointState, e.g. a gripper
# command on std_msgs/Float64.
_SCALAR_TYPES = {
    "Bool",
    "Float32",
    "Float64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32MultiArray",
    "Float64MultiArray",
    "Int32MultiArray",
    "Int64MultiArray",
}
_SCALAR_STRIP = set(_ACTION_HINTS) | set(_STATE_HINTS) | {"data"}


def _tokens(topic: str) -> set[str]:
    return {p.lower() for part in topic.strip("/").split("/") for p in part.split("_") if p}


def _assign_joint_roles(topics: list[str]) -> list[tuple[str, str, bool]]:
    """Pick which JointState topic is state, which is action, from topic name hints.

    Returns ``(key, topic, is_guess)`` triples. With no hints the first topic (in
    ``inspect``'s alphabetical order) becomes ``observation.state`` and the second
    becomes ``action``; anything left over is flagged with ``is_guess=True`` so the
    generated config calls it out as needing a human look.
    """
    state_topic = next((t for t in topics if _tokens(t) & set(_STATE_HINTS)), None)
    action_topic = next((t for t in topics if _tokens(t) & set(_ACTION_HINTS)), None)
    remaining = [t for t in topics if t not in (state_topic, action_topic)]
    state_guess = state_topic is None
    action_guess = action_topic is None
    if state_topic is None and remaining:
        state_topic = remaining.pop(0)
    if action_topic is None and remaining:
        action_topic = remaining.pop(0)

    assigned = []
    if state_topic is not None:
        assigned.append(("observation.state", state_topic, state_guess))
    if action_topic is not None:
        assigned.append(("action", action_topic, action_guess))
    for i, t in enumerate(remaining, start=2):
        assigned.append((f"observation.state_{i}", t, True))
    return assigned


def _scalar_names(topics: list[str]) -> dict[str, str]:
    """Derive a short, unique name per scalar topic, dropping role-hint segments (e.g.
    'cmd', 'state') so '/gripper/cmd' and '/gripper/state' both read 'gripper'."""
    used: set[str] = set()
    names = {}
    for topic in topics:
        parts = [p for p in topic.strip("/").split("/") if p]
        trimmed = [p for p in parts if p.lower() not in _SCALAR_STRIP]
        base = "_".join(trimmed) if trimmed else "_".join(parts) or "value"
        name, i = base, 2
        while name in used:
            name = f"{base}_{i}"
            i += 1
        used.add(name)
        names[topic] = name
    return names


def _assign_scalar_roles(topics: list[str]) -> list[tuple[str, str, bool]]:
    """Pick an ``action.<name>``/``observation.state.<name>`` key for each non-JointState
    numeric topic (e.g. a gripper command on ``std_msgs/Float64``), from the same
    leader/follower-style hints ``_assign_joint_roles`` uses. With no hint, the topic
    becomes an ``observation.state.<name>`` candidate flagged for review: a plain sensor
    reading is the more common case for a topic that doesn't name itself a command.

    Returns ``(key, topic, is_guess)`` triples, like ``_assign_joint_roles``.
    """
    roles: dict[str, tuple[str, bool]] = {}
    for topic in topics:
        tokens = _tokens(topic)
        if tokens & set(_ACTION_HINTS):
            roles[topic] = ("action", False)
        elif tokens & set(_STATE_HINTS):
            roles[topic] = ("observation.state", False)
        else:
            roles[topic] = ("observation.state", True)

    # name collisions only matter within the same prefix (action vs observation.state),
    # since e.g. '/gripper/cmd' and '/gripper/state' both reading 'gripper' is exactly
    # the point: one becomes action.gripper, the other observation.state.gripper
    assigned = []
    for prefix in ("action", "observation.state"):
        bucket = [t for t in topics if roles[t][0] == prefix]
        names = _scalar_names(bucket)
        for topic in bucket:
            assigned.append((f"{prefix}.{names[topic]}", topic, roles[topic][1]))
    return [next(a for a in assigned if a[1] == t) for t in topics]


def _camera_names(topics: list[str]) -> dict[str, str]:
    """Derive a short, unique ``observation.images.<name>`` suffix per camera topic."""
    used: set[str] = set()
    names = {}
    for topic in topics:
        parts = [p for p in topic.strip("/").split("/") if p]
        trimmed = [p for p in parts if p.lower() not in _CAMERA_STRIP]
        base = "_".join(trimmed) if trimmed else "_".join(parts) or "cam"
        name, i = base, 2
        while name in used:
            name = f"{base}_{i}"
            i += 1
        used.add(name)
        names[topic] = name
    return names


def generate_config(
    path: str | Path, fps: float | None = None, msg_dir: str | Path | None = None
) -> str:
    """Build a starter YAML config from the topics found in the bag at ``path``.

    JointState topics become ``observation.state``/``action`` candidates (picked by
    name hints like "leader"/"follower", falling back to topic order); Image and
    CompressedImage topics become ``observation.images.*`` video features; simple
    numeric std_msgs topics outside any JointState (e.g. a gripper command on
    ``std_msgs/Float64``) become ``action.<name>``/``observation.state.<name>``
    candidates, picked by the same hints. The result loads with ``load_config`` as-is,
    but comments point out guesses to double check.
    ``msg_dir``, when the bag uses custom message types rosbags doesn't ship (see
    ``extract._custom_types``), is carried into the generated config so later ``convert``
    calls pick it up too.
    """
    topics = inspect_bag(path, msg_dir)
    if not topics:
        raise ExtractError(f"{path}: no messages found")

    joint_topics = [t.topic for t in topics if t.msgtype.endswith("JointState")]
    image_topics = [
        t.topic
        for t in topics
        if t.msgtype.endswith("CompressedImage") or t.msgtype.endswith("Image")
    ]
    scalar_topics = [
        t.topic
        for t in topics
        if t.topic not in joint_topics
        and t.topic not in image_topics
        and t.msgtype.startswith("std_msgs/")
        and t.msgtype.rsplit("/", 1)[-1] in _SCALAR_TYPES
    ]
    if not joint_topics and not image_topics and not scalar_topics:
        raise ExtractError(
            f"{path}: no JointState, Image/CompressedImage or simple numeric std_msgs "
            "topics to build a config from (write one by hand; see README.md)"
        )

    if fps is None:
        rates = [t.rate_hz for t in topics if t.topic in joint_topics and t.rate_hz > 0]
        if not rates:
            rates = [t.rate_hz for t in topics if t.rate_hz > 0]
        fps = max(1, round(max(rates))) if rates else 30

    lines = [
        f"# Starter config generated by `bag2lerobot init {Path(path).name}`.",
        "# Review every guess below (marked '# check'), then fill in 'task'.",
        f"fps: {fps}",
        "robot_type:",
        "task:  # e.g. 'pick the cube'",
    ]
    if msg_dir:
        lines.append(f"msg_dir: {msg_dir}")
    lines.append("features:")
    for key, topic, is_guess in _assign_joint_roles(joint_topics):
        suffix = "  # check: is this really state/action?" if is_guess else ""
        lines += [
            f"  {key}:{suffix}",
            f"    topic: {topic}",
            "    field: position",
            "    names_from: name",
        ]
    for key, topic, is_guess in _assign_scalar_roles(scalar_topics):
        suffix = "  # check: is this really state/action?" if is_guess else ""
        lines += [
            f"  {key}:{suffix}",
            f"    topic: {topic}",
            "    field: data",
        ]
    for topic, name in _camera_names(image_topics).items():
        lines += [
            f"  observation.images.{name}:",
            f"    topic: {topic}",
            "    type: image  # needs the 'video' extra: pip install bag2lerobot[video]",
        ]
    return "\n".join(lines) + "\n"
