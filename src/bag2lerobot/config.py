"""Conversion config: which topics and fields become which LeRobot features."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

SYNC_METHODS = ("hold", "linear")
TIME_SOURCES = ("header", "bag")
FEATURE_TYPES = ("numeric", "image")


@dataclass
class FeatureSpec:
    key: str  # LeRobot feature name, e.g. "observation.state"
    topic: str
    fields: list[str]  # dotted paths into the message; empty means the whole message
    names_from: str | None = None  # field holding per-element names (e.g. JointState.name)
    names: list[str] | None = None  # explicit names, overrides names_from
    sync: str | None = None  # per-feature override of Config.sync
    max_staleness_s: float | None = None  # per-feature override of Config.max_staleness_s
    type: str = "numeric"  # "numeric" or "image" (sensor_msgs/Image or CompressedImage -> video)


@dataclass
class Config:
    fps: float
    features: list[FeatureSpec]
    task: str = ""
    robot_type: str | None = None
    sync: str = "hold"
    max_staleness_s: float = 0.25
    time_source: str = "header"
    trim_to_overlap: bool = True
    split_on: str | None = None  # topic whose messages mark episode-start boundaries
    split_idle_gap_s: float | None = None  # idle gap (s) across all streams that splits episodes
    split_csv: str | None = None  # CSV of start,end[,task] ranges that mark episode boundaries
    msg_dir: str | None = None  # directory of custom <package>/msg/*.msg definitions
    extra: dict = field(default_factory=dict)

    def feature(self, key: str) -> FeatureSpec:
        return next(f for f in self.features if f.key == key)


class ConfigError(ValueError):
    pass


def _as_float(value, what: str) -> float:
    """Convert a config value expected to be a number, raising a clear ConfigError
    instead of a raw TypeError (e.g. a list or mapping from a YAML indentation typo,
    which ``float()`` rejects with a message naming neither the field nor the config)."""
    if not isinstance(value, int | float | str):
        raise ConfigError(f"{what} must be a number, got {value!r}")
    try:
        return float(value)
    except ValueError:
        raise ConfigError(f"{what} must be a number, got {value!r}") from None


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return parse_config(data)


def parse_config(data: dict) -> Config:
    if not isinstance(data, dict):
        raise ConfigError(
            "config must be a YAML mapping with top-level keys like 'fps' and 'features', "
            f"got {type(data).__name__}"
        )
    if "fps" not in data:
        raise ConfigError("config needs 'fps' (the output frame rate)")
    raw_features = data.get("features") or {}
    if not isinstance(raw_features, dict):
        raise ConfigError(
            "'features' must be a mapping of feature name -> spec, "
            f"got {type(raw_features).__name__}"
        )
    if not raw_features:
        raise ConfigError("config needs at least one entry under 'features'")

    features = []
    for key, spec in raw_features.items():
        if not isinstance(spec, dict) or "topic" not in spec:
            raise ConfigError(f"feature '{key}' needs a 'topic'")
        feat_type = spec.get("type", "numeric")
        if feat_type not in FEATURE_TYPES:
            raise ConfigError(
                f"feature '{key}': type must be one of {FEATURE_TYPES}, got '{feat_type}'"
            )
        if "field" in spec and "fields" in spec:
            raise ConfigError(f"feature '{key}': use either 'field' or 'fields', not both")
        if spec.get("field") is not None and not isinstance(spec["field"], str):
            raise ConfigError(f"feature '{key}': 'field' must be a string, got {spec['field']!r}")
        raw_fields = spec.get("fields")
        if raw_fields is not None and (
            not isinstance(raw_fields, list) or not all(isinstance(f, str) for f in raw_fields)
        ):
            raise ConfigError(
                f"feature '{key}': 'fields' must be a list of strings, got {raw_fields!r}"
            )
        raw_names = spec.get("names")
        if raw_names is not None and (
            not isinstance(raw_names, list) or not all(isinstance(n, str) for n in raw_names)
        ):
            raise ConfigError(
                f"feature '{key}': 'names' must be a list of strings, got {raw_names!r}"
            )
        raw_names_from = spec.get("names_from")
        if raw_names_from is not None and not isinstance(raw_names_from, str):
            raise ConfigError(
                f"feature '{key}': 'names_from' must be a string, got {raw_names_from!r}"
            )
        fields = raw_fields
        if fields is None:
            fields = [spec["field"]] if spec.get("field") else []
        feat_sync = spec.get("sync")
        if feat_sync is not None and feat_sync not in SYNC_METHODS:
            raise ConfigError(
                f"feature '{key}': sync must be one of {SYNC_METHODS}, got '{feat_sync}'"
            )
        feat_staleness = spec.get("max_staleness_s")
        if feat_staleness is not None:
            feat_staleness = _as_float(feat_staleness, f"feature '{key}': max_staleness_s")
            if feat_staleness < 0:
                raise ConfigError(f"feature '{key}': max_staleness_s must not be negative")
        if feat_type == "image":
            if fields or spec.get("names_from") or spec.get("names"):
                raise ConfigError(f"feature '{key}': image features don't support fields/names")
            if feat_sync is not None and feat_sync != "hold":
                raise ConfigError(f"feature '{key}': image features only support sync 'hold'")
        features.append(
            FeatureSpec(
                key=key,
                topic=spec["topic"],
                fields=list(fields),
                names_from=spec.get("names_from"),
                names=spec.get("names"),
                sync=feat_sync,
                max_staleness_s=feat_staleness,
                type=feat_type,
            )
        )

    raw_task = data.get("task")
    if raw_task is not None and not isinstance(raw_task, str):
        raise ConfigError(f"'task' must be a string, got {raw_task!r}")
    raw_robot_type = data.get("robot_type")
    if raw_robot_type is not None and not isinstance(raw_robot_type, str):
        raise ConfigError(f"'robot_type' must be a string, got {raw_robot_type!r}")

    sync = data.get("sync", {}) or {}
    if not isinstance(sync, dict):
        raise ConfigError(f"'sync' must be a mapping, got {type(sync).__name__}")
    trim_to_overlap = sync.get("trim_to_overlap", True)
    if not isinstance(trim_to_overlap, bool):
        raise ConfigError(f"sync.trim_to_overlap must be true or false, got {trim_to_overlap!r}")
    split_on = data.get("split_on")
    if split_on is not None and not isinstance(split_on, str):
        raise ConfigError("'split_on' must be a topic name (string)")
    split_idle_gap_s = data.get("split_idle_gap_s")
    if split_idle_gap_s is not None:
        split_idle_gap_s = _as_float(split_idle_gap_s, "'split_idle_gap_s'")
    split_csv = data.get("split_csv")
    if split_csv is not None and not isinstance(split_csv, str):
        raise ConfigError("'split_csv' must be a path to a CSV file (string)")
    msg_dir = data.get("msg_dir")
    if msg_dir is not None and not isinstance(msg_dir, str):
        raise ConfigError(
            "'msg_dir' must be a path to a directory of custom '<package>/msg/*.msg' files"
        )
    set_splits = [
        name
        for name, val in (
            ("split_on", split_on),
            ("split_idle_gap_s", split_idle_gap_s),
            ("split_csv", split_csv),
        )
        if val
    ]
    if len(set_splits) > 1:
        raise ConfigError(
            f"only one of 'split_on', 'split_idle_gap_s', 'split_csv' can be set "
            f"(got {', '.join(set_splits)})"
        )
    cfg = Config(
        fps=_as_float(data["fps"], "'fps'"),
        features=features,
        task=raw_task or "",
        robot_type=raw_robot_type,
        sync=sync.get("method", "hold"),
        max_staleness_s=_as_float(sync.get("max_staleness_s", 0.25), "sync.max_staleness_s"),
        time_source=data.get("time_source", "header"),
        trim_to_overlap=trim_to_overlap,
        split_on=split_on or None,
        split_idle_gap_s=split_idle_gap_s,
        split_csv=split_csv or None,
        msg_dir=msg_dir or None,
    )
    if cfg.fps <= 0:
        raise ConfigError("'fps' must be positive")
    if cfg.sync not in SYNC_METHODS:
        raise ConfigError(f"sync.method must be one of {SYNC_METHODS}, got '{cfg.sync}'")
    if cfg.max_staleness_s < 0:
        raise ConfigError("sync.max_staleness_s must not be negative")
    if cfg.time_source not in TIME_SOURCES:
        raise ConfigError(f"time_source must be one of {TIME_SOURCES}, got '{cfg.time_source}'")
    if cfg.split_idle_gap_s is not None and cfg.split_idle_gap_s <= 0:
        raise ConfigError("'split_idle_gap_s' must be positive")
    if cfg.msg_dir is not None and not Path(cfg.msg_dir).is_dir():
        raise ConfigError(f"msg_dir '{cfg.msg_dir}' is not a directory")
    return cfg
