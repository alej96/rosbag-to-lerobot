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


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return parse_config(data)


def parse_config(data: dict) -> Config:
    if "fps" not in data:
        raise ConfigError("config needs 'fps' (the output frame rate)")
    raw_features = data.get("features") or {}
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
        fields = spec.get("fields")
        if fields is None:
            fields = [spec["field"]] if spec.get("field") else []
        feat_sync = spec.get("sync")
        if feat_sync is not None and feat_sync not in SYNC_METHODS:
            raise ConfigError(
                f"feature '{key}': sync must be one of {SYNC_METHODS}, got '{feat_sync}'"
            )
        feat_staleness = spec.get("max_staleness_s")
        if feat_staleness is not None:
            feat_staleness = float(feat_staleness)
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

    sync = data.get("sync", {}) or {}
    split_on = data.get("split_on")
    if split_on is not None and not isinstance(split_on, str):
        raise ConfigError("'split_on' must be a topic name (string)")
    split_idle_gap_s = data.get("split_idle_gap_s")
    if split_idle_gap_s is not None:
        split_idle_gap_s = float(split_idle_gap_s)
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
        fps=float(data["fps"]),
        features=features,
        task=str(data.get("task") or ""),
        robot_type=data.get("robot_type"),
        sync=sync.get("method", "hold"),
        max_staleness_s=float(sync.get("max_staleness_s", 0.25)),
        time_source=data.get("time_source", "header"),
        trim_to_overlap=bool(sync.get("trim_to_overlap", True)),
        split_on=split_on or None,
        split_idle_gap_s=split_idle_gap_s,
        split_csv=split_csv or None,
        msg_dir=msg_dir or None,
    )
    if cfg.fps <= 0:
        raise ConfigError("'fps' must be positive")
    if cfg.sync not in SYNC_METHODS:
        raise ConfigError(f"sync.method must be one of {SYNC_METHODS}, got '{cfg.sync}'")
    if cfg.time_source not in TIME_SOURCES:
        raise ConfigError(f"time_source must be one of {TIME_SOURCES}, got '{cfg.time_source}'")
    if cfg.split_idle_gap_s is not None and cfg.split_idle_gap_s <= 0:
        raise ConfigError("'split_idle_gap_s' must be positive")
    if cfg.msg_dir is not None and not Path(cfg.msg_dir).is_dir():
        raise ConfigError(f"msg_dir '{cfg.msg_dir}' is not a directory")
    return cfg
