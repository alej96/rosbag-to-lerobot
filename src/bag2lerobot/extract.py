"""Reading bags and turning messages into timestamped numeric vectors."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from bag2lerobot.config import Config, FeatureSpec


@dataclass
class Stream:
    """All samples of one feature: times in seconds and an (N, D) value matrix."""

    key: str
    topic: str
    times: np.ndarray
    values: np.ndarray
    names: list[str]


@dataclass
class ImageStream:
    """All frames of one image feature: times in seconds and the raw, not-yet-decoded
    message for each. Decoding is deferred to ``sync.synchronize`` (via ``decode_image``),
    which only needs the frames the output grid actually samples; a bag's camera often
    publishes faster than the configured fps, so most raw frames are never decoded at
    all."""

    key: str
    topic: str
    times: np.ndarray
    frames: list  # raw deserialized messages, decode with decode_image()


# numpy dtype/channel layout for raw sensor_msgs/Image encodings we can decode without a
# compressed-image codec. Values are (channels, is_bgr).
_RAW_ENCODINGS = {
    "rgb8": (3, False),
    "bgr8": (3, True),
    "rgba8": (4, False),
    "bgra8": (4, True),
    "mono8": (1, False),
}


@dataclass
class TopicInfo:
    topic: str
    msgtype: str
    count: int
    duration_s: float

    @property
    def rate_hz(self) -> float:
        return (self.count - 1) / self.duration_s if self.duration_s > 0 and self.count > 1 else 0.0


class ExtractError(ValueError):
    pass


@lru_cache(maxsize=8)
def _typestore(msg_dir: str | None):
    store = get_typestore(Stores.LATEST)
    if msg_dir:
        store.register(_custom_types(Path(msg_dir)))
    return store


def _custom_types(msg_dir: Path) -> dict:
    """Parse every ``<package>/msg/<Name>.msg`` file under ``msg_dir`` (the layout of a
    ROS package's message definitions) into a typesdict ready for ``Typestore.register``."""
    if not msg_dir.is_dir():
        raise ExtractError(f"msg_dir '{msg_dir}' is not a directory")
    msg_files = sorted(msg_dir.glob("*/msg/*.msg"))
    if not msg_files:
        raise ExtractError(
            f"msg_dir '{msg_dir}': no '<package>/msg/*.msg' files found; lay out custom "
            "definitions the same way a ROS package would (e.g. 'my_pkg/msg/MyMsg.msg')"
        )
    types: dict = {}
    for msg_file in msg_files:
        name = f"{msg_file.parent.parent.name}/msg/{msg_file.stem}"
        types.update(get_types_from_msg(msg_file.read_text(), name))
    return types


def _reader(path: Path, msg_dir: str | Path | None = None) -> AnyReader:
    # Most bags carry their own message definitions; a bag recorded without them (seen
    # on some ROS 2 distros/storage plugins for message types outside the bundled
    # standard interfaces) falls back to this default typestore, extended with 'msg_dir'
    # when the config or CLI points at custom '.msg' files.
    return AnyReader([Path(path)], default_typestore=_typestore(str(msg_dir) if msg_dir else None))


def _deserialize(reader: AnyReader, raw: bytes, conn, path: str | Path):
    try:
        return reader.deserialize(raw, conn.msgtype)
    except KeyError as exc:
        raise ExtractError(
            f"{path}: '{conn.msgtype}' on '{conn.topic}' has no known definition (not bundled "
            "with rosbags and not embedded in the bag); set 'msg_dir' to a directory with its "
            f"'<package>/msg/{conn.msgtype.rsplit('/', 1)[-1]}.msg' file"
        ) from exc


def inspect_bag(path: str | Path, msg_dir: str | Path | None = None) -> list[TopicInfo]:
    counts: Counter[str] = Counter()
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    types: dict[str, str] = {}
    with _reader(Path(path), msg_dir) as reader:
        for conn, t, _ in reader.messages():
            counts[conn.topic] += 1
            first.setdefault(conn.topic, t)
            last[conn.topic] = t
            types[conn.topic] = conn.msgtype
    return [
        TopicInfo(topic, types[topic], counts[topic], (last[topic] - first[topic]) / 1e9)
        for topic in sorted(counts)
    ]


def extract_streams(
    path: str | Path, cfg: Config, joint_order: dict[str, list[str]] | None = None
) -> dict[str, Stream]:
    """Read every configured feature from one bag.

    ``joint_order`` is shared across the bags of one conversion so that named elements
    (e.g. JointState joints) keep the same column order in every episode.
    """
    joint_order = {} if joint_order is None else joint_order
    specs = [s for s in cfg.features if s.type == "numeric"]
    by_topic: dict[str, list[FeatureSpec]] = {}
    for spec in specs:
        by_topic.setdefault(spec.topic, []).append(spec)

    samples: dict[str, tuple[list[float], list[np.ndarray]]] = {s.key: ([], []) for s in specs}
    names: dict[str, list[str]] = {}
    with _reader(Path(path), cfg.msg_dir) as reader:
        available = {c.topic for c in reader.connections}
        missing = sorted(set(by_topic) - available)
        if missing:
            raise ExtractError(
                f"{path}: topic(s) not in bag: {', '.join(missing)} "
                f"(available: {', '.join(sorted(available))})"
            )
        conns = [c for c in reader.connections if c.topic in by_topic]
        for conn, bag_t, raw in reader.messages(connections=conns):
            msg = _deserialize(reader, raw, conn, path)
            t = _stamp(msg, bag_t) if cfg.time_source == "header" else bag_t / 1e9
            for spec in by_topic[conn.topic]:
                vec, vec_names = _flatten(msg, spec, joint_order)
                times, values = samples[spec.key]
                if values and len(vec) != len(values[0]):
                    raise ExtractError(
                        f"{path}: '{spec.key}' changed size from {len(values[0])} to {len(vec)} "
                        f"at t={t:.3f}s; pin the element order with 'names'"
                    )
                times.append(t)
                values.append(vec)
                names.setdefault(spec.key, vec_names)

    streams = {}
    for spec in specs:
        times, values = samples[spec.key]
        if not values:
            raise ExtractError(f"{path}: no messages on {spec.topic} for '{spec.key}'")
        t = np.asarray(times, dtype=np.float64)
        v = np.vstack(values)
        order = np.argsort(t, kind="stable")
        streams[spec.key] = Stream(spec.key, spec.topic, t[order], v[order], names[spec.key])
    return streams


def extract_image_streams(path: str | Path, cfg: Config) -> dict[str, ImageStream]:
    """Read every configured image feature's messages from one bag, without decoding."""
    specs = [s for s in cfg.features if s.type == "image"]
    if not specs:
        return {}
    by_topic: dict[str, list[FeatureSpec]] = {}
    for spec in specs:
        by_topic.setdefault(spec.topic, []).append(spec)

    samples: dict[str, tuple[list[float], list]] = {s.key: ([], []) for s in specs}
    with _reader(Path(path), cfg.msg_dir) as reader:
        available = {c.topic for c in reader.connections}
        missing = sorted(set(by_topic) - available)
        if missing:
            raise ExtractError(
                f"{path}: topic(s) not in bag: {', '.join(missing)} "
                f"(available: {', '.join(sorted(available))})"
            )
        conns = [c for c in reader.connections if c.topic in by_topic]
        for conn, bag_t, raw in reader.messages(connections=conns):
            msg = _deserialize(reader, raw, conn, path)
            t = _stamp(msg, bag_t) if cfg.time_source == "header" else bag_t / 1e9
            for spec in by_topic[conn.topic]:
                times, frames = samples[spec.key]
                times.append(t)
                frames.append(msg)

    streams = {}
    for spec in specs:
        times, frames = samples[spec.key]
        if not frames:
            raise ExtractError(f"{path}: no messages on {spec.topic} for '{spec.key}'")
        t = np.asarray(times, dtype=np.float64)
        order = np.argsort(t, kind="stable")
        streams[spec.key] = ImageStream(spec.key, spec.topic, t[order], [frames[i] for i in order])
    return streams


def split_boundaries(path: str | Path, topic: str, cfg: Config) -> list[float]:
    """Episode-start timestamps (s) from messages on ``topic``.

    A ``std_msgs/Bool`` message starts a new episode only on a rising edge (``data``
    True); any other message type (e.g. ``std_msgs/Empty``) starts a new episode on
    every message. Each boundary marks where one episode ends and the next begins; data
    before the first boundary belongs to no episode and is dropped.
    """
    with _reader(Path(path), cfg.msg_dir) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            available = sorted({c.topic for c in reader.connections})
            raise ExtractError(
                f"{path}: split topic '{topic}' not in bag (available: {', '.join(available)})"
            )
        is_bool = conns[0].msgtype.endswith("Bool")
        times = []
        for conn, bag_t, raw in reader.messages(connections=conns):
            msg = _deserialize(reader, raw, conn, path)
            if is_bool and not bool(msg.data):
                continue
            t = _stamp(msg, bag_t) if cfg.time_source == "header" else bag_t / 1e9
            times.append(t)
    if not times:
        raise ExtractError(f"{path}: no episode-start messages found on '{topic}'")
    return sorted(times)


def idle_gap_boundaries(
    streams: dict[str, Stream], image_streams: dict[str, ImageStream], min_gap_s: float
) -> list[float]:
    """Episode-start timestamps (s) from idle gaps of at least ``min_gap_s`` in the merged
    timeline of every configured stream.

    Unlike ``split_boundaries``, every sample belongs to some episode: the first boundary
    is the earliest timestamp across all streams, and each later boundary is the first
    sample after a gap at least ``min_gap_s`` long, so nothing is dropped.
    """
    arrays = [s.times for s in streams.values()] + [s.times for s in image_streams.values()]
    if not arrays or all(a.size == 0 for a in arrays):
        raise ExtractError("no data to split on idle gaps")
    t = np.sort(np.concatenate(arrays))
    gaps = np.diff(t)
    idle = np.flatnonzero(gaps >= min_gap_s)
    return [float(t[0]), *(float(t[i + 1]) for i in idle)]


def csv_windows(
    csv_path: str | Path, streams: dict[str, Stream], image_streams: dict[str, ImageStream]
) -> list[tuple[tuple[float, float], str | None]]:
    """Episode ``(start, end)`` windows (absolute, s) and optional per-episode task from a CSV
    of ``start,end[,task]`` rows, e.g. written by an external annotation tool.

    ``start``/``end`` are seconds since the earliest sample across every configured stream
    (the same anchor ``idle_gap_boundaries`` uses for its first boundary), not bag/header
    epoch time, since that's what an external tool labeling a recording would know. Rows
    are read in file order; each becomes one episode in that order, independent of the
    others (unlike ``split_boundaries``, a row's end doesn't have to be the next row's
    start, so gaps between labeled ranges are simply excluded).
    """
    arrays = [s.times for s in streams.values()] + [s.times for s in image_streams.values()]
    if not arrays or all(a.size == 0 for a in arrays):
        raise ExtractError("no data to anchor 'split_csv' time ranges")
    anchor = min(float(a[0]) for a in arrays)

    path = Path(csv_path)
    if not path.exists():
        raise ExtractError(f"split_csv file not found: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "start" not in fields or "end" not in fields:
            raise ExtractError(f"{path}: csv needs 'start' and 'end' columns, got {fields}")
        windows = []
        for i, row in enumerate(reader, start=1):
            try:
                start, end = float(row["start"]), float(row["end"])
            except (TypeError, ValueError):
                raise ExtractError(
                    f"{path}: row {i} has non-numeric start/end ({row['start']!r}, {row['end']!r})"
                ) from None
            if end <= start:
                raise ExtractError(f"{path}: row {i} has end ({end}) <= start ({start})")
            task = (row.get("task") or "").strip() or None
            windows.append(((anchor + start, anchor + end), task))
    if not windows:
        raise ExtractError(f"{path}: no episode rows found")
    return windows


def decode_image(msg) -> np.ndarray:
    """Decode a sensor_msgs/Image or CompressedImage message into an (H, W, 3) RGB array."""
    msgtype = getattr(msg, "__msgtype__", "")
    if msgtype.endswith("CompressedImage"):
        try:
            from PIL import Image as PILImage
        except ImportError as exc:
            raise ExtractError(
                "decoding CompressedImage needs the 'video' extra: pip install 'bag2lerobot[video]'"
            ) from exc
        import io

        with PILImage.open(io.BytesIO(bytes(msg.data))) as img:
            return np.array(img.convert("RGB"))
    if msgtype.endswith("Image"):
        channels, is_bgr = _RAW_ENCODINGS.get(msg.encoding, (None, None))
        if channels is None:
            raise ExtractError(
                f"unsupported image encoding '{msg.encoding}' (supported: "
                f"{', '.join(_RAW_ENCODINGS)}, or publish CompressedImage)"
            )
        data = np.asarray(msg.data, dtype=np.uint8)
        rows = data[: msg.height * msg.step].reshape(msg.height, msg.step)
        frame = rows[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
        if channels == 1:
            return np.repeat(frame, 3, axis=2)
        if channels == 4:
            frame = frame[..., :3]
        return frame[..., ::-1] if is_bgr else frame
    raise ExtractError(f"'{msgtype}' is not an image message type")


def _stamp(msg, bag_t: int) -> float:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        sec = getattr(stamp, "sec", getattr(stamp, "secs", 0))
        nsec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0))
        if sec or nsec:
            return sec + nsec / 1e9
    return bag_t / 1e9


def _get(obj, path: str):
    for part in path.split("."):
        if not hasattr(obj, part):
            raise ExtractError(f"message {type(obj).__name__} has no field '{part}' (in '{path}')")
        obj = getattr(obj, part)
    return obj


def _flatten(
    msg, spec: FeatureSpec, joint_order: dict[str, list[str]]
) -> tuple[np.ndarray, list[str]]:
    parts: list[np.ndarray] = []
    names: list[str] = []
    for path in spec.fields or [""]:
        value = _get(msg, path) if path else msg
        vec, leaf_names = _numeric(value, path or spec.topic.strip("/"))
        parts.append(vec)
        names.extend(leaf_names)
    vec = np.concatenate(parts) if parts else np.zeros(0)

    if spec.names:
        names = list(spec.names)
    elif spec.names_from:
        names = [str(n) for n in _get(msg, spec.names_from)]
    if spec.names and len(spec.names) != len(vec):
        raise ExtractError(f"'{spec.key}': {len(spec.names)} names for {len(vec)} values")
    if spec.names_from and spec.names is None:
        # reorder-safe: some drivers publish joints in varying order
        vec, names = _canonical_order(vec, names, spec, joint_order)
    return vec, names


def _canonical_order(
    vec: np.ndarray, names: list[str], spec: FeatureSpec, joint_order: dict[str, list[str]]
):
    if len(names) != len(vec):
        return vec, names
    order = joint_order.setdefault(spec.key, list(names))
    if names == order:
        return vec, names
    if sorted(names) != sorted(order):
        raise ExtractError(f"'{spec.key}': joint names changed from {order} to {names}")
    index = {n: i for i, n in enumerate(names)}
    return vec[[index[n] for n in order]], list(order)


def _numeric(value, prefix: str) -> tuple[np.ndarray, list[str]]:
    """Flatten a field into floats: numbers, arrays and nested messages (leaf order)."""
    if isinstance(value, bool | int | float | np.number | np.bool_):
        return np.array([float(value)]), [prefix]
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ExtractError(f"field '{prefix}' is not numeric ({value.dtype})")
        flat = value.astype(np.float64).ravel()
        return flat, [f"{prefix}.{i}" for i in range(len(flat))]
    if isinstance(value, list | tuple):
        vecs = [_numeric(v, f"{prefix}.{i}") for i, v in enumerate(value)]
        if not vecs:
            return np.zeros(0), []
        return np.concatenate([v for v, _ in vecs]), [n for _, ns in vecs for n in ns]
    fields = getattr(value, "__dataclass_fields__", None)
    if fields:
        vecs, names = [], []
        for name in fields:
            if name in ("header", "__msgtype__"):
                continue
            child = getattr(value, name)
            if isinstance(child, str):
                continue
            try:
                v, n = _numeric(child, f"{prefix}.{name}" if prefix else name)
            except ExtractError:
                continue
            vecs.append(v)
            names.extend(n)
        if not vecs:
            raise ExtractError(f"field '{prefix}' has no numeric content")
        return np.concatenate(vecs), names
    raise ExtractError(f"field '{prefix}' of type {type(value).__name__} is not numeric")
