"""Reading bags and turning messages into timestamped numeric vectors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

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


def _reader(path: Path) -> AnyReader:
    # ROS 2 bags carry their message definitions; the default store covers ROS 1 bags
    # and ROS 2 bags recorded without definitions.
    return AnyReader([Path(path)], default_typestore=get_typestore(Stores.LATEST))


def inspect_bag(path: str | Path) -> list[TopicInfo]:
    counts: Counter[str] = Counter()
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    types: dict[str, str] = {}
    with _reader(Path(path)) as reader:
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
    by_topic: dict[str, list[FeatureSpec]] = {}
    for spec in cfg.features:
        by_topic.setdefault(spec.topic, []).append(spec)

    samples: dict[str, tuple[list[float], list[np.ndarray]]] = {
        s.key: ([], []) for s in cfg.features
    }
    names: dict[str, list[str]] = {}
    with _reader(Path(path)) as reader:
        available = {c.topic for c in reader.connections}
        missing = sorted(set(by_topic) - available)
        if missing:
            raise ExtractError(
                f"{path}: topic(s) not in bag: {', '.join(missing)} "
                f"(available: {', '.join(sorted(available))})"
            )
        conns = [c for c in reader.connections if c.topic in by_topic]
        for conn, bag_t, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
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
    for spec in cfg.features:
        times, values = samples[spec.key]
        if not values:
            raise ExtractError(f"{path}: no messages on {spec.topic} for '{spec.key}'")
        t = np.asarray(times, dtype=np.float64)
        v = np.vstack(values)
        order = np.argsort(t, kind="stable")
        streams[spec.key] = Stream(spec.key, spec.topic, t[order], v[order], names[spec.key])
    return streams


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
