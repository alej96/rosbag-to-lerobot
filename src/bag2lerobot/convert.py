"""End-to-end conversion: bags -> synchronized episodes -> LeRobot dataset."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from bag2lerobot.config import Config
from bag2lerobot.extract import (
    extract_image_streams,
    extract_streams,
    idle_gap_boundaries,
    split_boundaries,
)
from bag2lerobot.sync import SyncResult, synchronize
from bag2lerobot.writer import DEFAULT_DATA_FILES_SIZE_IN_MB, Episode, write_dataset


@dataclass
class ConversionReport:
    bag: Path
    sync: SyncResult
    segment: int = 0  # index of this episode within its source bag (when splitting is on)


def _windows(boundaries: list[float]) -> list[tuple[float, float]]:
    ends = [*boundaries[1:], float("inf")]
    return list(zip(boundaries, ends, strict=True))


def _boundaries(
    bag: Path, cfg: Config, streams=None, image_streams=None, msg_dir: str | Path | None = None
) -> list[float] | None:
    if cfg.split_on:
        return split_boundaries(bag, cfg.split_on, cfg, msg_dir)
    if cfg.split_idle_gap_s:
        if streams is None:
            streams = extract_streams(bag, cfg, msg_dir=msg_dir)
        if image_streams is None:
            image_streams = extract_image_streams(bag, cfg, msg_dir=msg_dir)
        return idle_gap_boundaries(streams, image_streams, cfg.split_idle_gap_s)
    return None


def preview_splits(
    bags: list[Path], cfg: Config, msg_dir: str | Path | None = None
) -> dict[Path, list[float]]:
    """Episode-start timestamps (s) ``convert`` would split each bag on, without writing."""
    if not cfg.split_on and not cfg.split_idle_gap_s:
        raise ValueError("config has no 'split_on' topic or 'split_idle_gap_s' to preview")
    return {Path(bag): _boundaries(Path(bag), cfg, msg_dir=msg_dir) for bag in bags}


def convert(
    bags: list[Path],
    cfg: Config,
    out: Path,
    overwrite: bool = False,
    append: bool = False,
    max_data_file_mb: float = DEFAULT_DATA_FILES_SIZE_IN_MB,
    msg_dir: str | Path | None = None,
) -> list[ConversionReport]:
    """Convert each bag into one or more episodes of a LeRobot dataset at ``out``.

    When ``cfg.split_on`` is set, each bag is split into multiple episodes at the
    timestamps of messages on that topic (see ``extract.split_boundaries``). When
    ``cfg.split_idle_gap_s`` is set instead, each bag is split wherever every stream goes
    idle for at least that long (see ``extract.idle_gap_boundaries``). Otherwise each bag
    becomes exactly one episode. With ``append=True``, episodes are added to the dataset
    already at ``out`` instead of replacing it (``out`` must hold one written by a
    previous ``convert`` call, with matching fps and features). ``max_data_file_mb`` caps
    the size of each data parquet file; a conversion producing more data than that is
    split across several files instead of one unboundedly large one. ``msg_dir`` registers
    custom message types from ``.msg`` files for a bag with no embedded definitions.
    """
    if overwrite and append:
        raise ValueError("--overwrite and --append can't be used together")
    out = Path(out)
    joint_order: dict[str, list[str]] = {}
    if append:
        joint_order = _seed_joint_order(out, cfg)
    elif out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty (use --overwrite)")
        shutil.rmtree(out)

    episodes, reports = [], []
    for bag in bags:
        streams = extract_streams(bag, cfg, joint_order, msg_dir)
        image_streams = extract_image_streams(bag, cfg, msg_dir)
        boundaries = _boundaries(bag, cfg, streams, image_streams, msg_dir)
        windows = _windows(boundaries) if boundaries is not None else [None]
        for segment, window in enumerate(windows):
            result = synchronize(streams, cfg, image_streams, window=window)
            episodes.append(Episode(result, cfg.task))
            reports.append(ConversionReport(Path(bag), result, segment))
    write_dataset(
        out, episodes, cfg.fps, cfg.robot_type, append=append, max_data_file_mb=max_data_file_mb
    )
    return reports


def _seed_joint_order(out: Path, cfg: Config) -> dict[str, list[str]]:
    """Pin ``names_from`` features to the element order already established in an
    existing dataset, so an appended bag whose driver reorders joints still lines up
    with earlier episodes instead of silently shifting columns."""
    info_path = Path(out) / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"--append: {out} has no dataset to append to (missing {info_path})"
        )
    info = json.loads(info_path.read_text())
    joint_order = {}
    for spec in cfg.features:
        if spec.type == "numeric" and spec.names_from and spec.key in info["features"]:
            names = info["features"][spec.key].get("names")
            if names:
                joint_order[spec.key] = list(names)
    return joint_order
