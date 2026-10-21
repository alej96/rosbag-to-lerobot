"""End-to-end conversion: bags -> synchronized episodes -> LeRobot dataset."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from bag2lerobot.config import Config
from bag2lerobot.extract import (
    csv_windows,
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


def _episode_windows(
    bag: Path, cfg: Config, streams=None, image_streams=None
) -> list[tuple[tuple[float, float], str | None]] | None:
    """Each episode's ``(window, task)`` for one bag, or ``None`` when no splitting is
    configured (the whole bag is one episode). ``task`` is ``None`` unless ``split_csv``
    gives a per-episode task, in which case it overrides ``cfg.task`` for that episode."""
    if cfg.split_on:
        return [(w, None) for w in _windows(split_boundaries(bag, cfg.split_on, cfg))]
    if cfg.split_idle_gap_s or cfg.split_csv:
        if streams is None:
            streams = extract_streams(bag, cfg)
        if image_streams is None:
            image_streams = extract_image_streams(bag, cfg)
        if cfg.split_csv:
            return csv_windows(cfg.split_csv, streams, image_streams)
        return [
            (w, None)
            for w in _windows(idle_gap_boundaries(streams, image_streams, cfg.split_idle_gap_s))
        ]
    return None


def preview_splits(
    bags: list[Path], cfg: Config
) -> dict[Path, list[tuple[tuple[float, float], str | None]]]:
    """Each episode's ``(window, task)`` ``convert`` would split each bag into, without
    writing anything."""
    if not (cfg.split_on or cfg.split_idle_gap_s or cfg.split_csv):
        raise ValueError(
            "config has no 'split_on' topic, 'split_idle_gap_s' or 'split_csv' to preview"
        )
    return {Path(bag): _episode_windows(Path(bag), cfg) for bag in bags}


def convert(
    bags: list[Path],
    cfg: Config,
    out: Path,
    overwrite: bool = False,
    append: bool = False,
    max_data_file_mb: float = DEFAULT_DATA_FILES_SIZE_IN_MB,
) -> list[ConversionReport]:
    """Convert each bag into one or more episodes of a LeRobot dataset at ``out``.

    When ``cfg.split_on`` is set, each bag is split into multiple episodes at the
    timestamps of messages on that topic (see ``extract.split_boundaries``). When
    ``cfg.split_idle_gap_s`` is set instead, each bag is split wherever every stream goes
    idle for at least that long (see ``extract.idle_gap_boundaries``). When
    ``cfg.split_csv`` is set, episode windows (and optionally a per-episode task) come
    from a CSV of time ranges instead (see ``extract.csv_windows``). Otherwise each bag
    becomes exactly one episode. With ``append=True``, episodes are added to the dataset
    already at ``out`` instead of replacing it (``out`` must hold one written by a
    previous ``convert`` call, with matching fps and features). ``max_data_file_mb`` caps
    the size of each data parquet file; a conversion producing more data than that is
    split across several files instead of one unboundedly large one.

    Bags are read and synchronized one at a time, and ``write_dataset`` writes (and, for
    image features, encodes) each episode as soon as it's produced, instead of first
    collecting every bag's episodes (decoded video frames included) into one big list:
    peak memory scales with the largest single bag, not with the sum of all of them. A
    fresh (non-append) conversion that fails partway removes the partial dataset it
    created, so a failure never leaves ``out`` holding only some of the bags.
    """
    if overwrite and append:
        raise ValueError("--overwrite and --append can't be used together")
    if not bags:
        raise ValueError("nothing to write")
    out = Path(out)
    joint_order: dict[str, list[str]] = {}
    created = not append
    if append:
        joint_order = _seed_joint_order(out, cfg)
    elif out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty (use --overwrite)")
        shutil.rmtree(out)

    reports: list[ConversionReport] = []

    def episodes():
        for bag in bags:
            streams = extract_streams(bag, cfg, joint_order)
            image_streams = extract_image_streams(bag, cfg)
            episode_windows = _episode_windows(bag, cfg, streams, image_streams)
            episode_windows = episode_windows if episode_windows is not None else [(None, None)]
            for segment, (window, task) in enumerate(episode_windows):
                result = synchronize(streams, cfg, image_streams, window=window)
                reports.append(ConversionReport(Path(bag), result, segment))
                yield Episode(result, task or cfg.task)
                # write_dataset has already written (and, for images, encoded) this
                # episode by the time it asks for the next one: drop the per-frame
                # data so the kept `reports` list doesn't pin every bag's values and
                # decoded image frames in memory for the rest of the conversion.
                result.values = {}
                result.images = {}

    try:
        write_dataset(
            out,
            episodes(),
            cfg.fps,
            cfg.robot_type,
            append=append,
            max_data_file_mb=max_data_file_mb,
        )
    except Exception:
        if created and out.exists():
            shutil.rmtree(out)
        raise
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
