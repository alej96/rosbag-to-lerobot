"""End-to-end conversion: bags -> synchronized episodes -> LeRobot dataset."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from bag2lerobot.config import Config
from bag2lerobot.extract import extract_image_streams, extract_streams, split_boundaries
from bag2lerobot.sync import SyncResult, synchronize
from bag2lerobot.writer import Episode, write_dataset


@dataclass
class ConversionReport:
    bag: Path
    sync: SyncResult
    segment: int = 0  # index of this episode within its source bag (when split_on is set)


def _windows(boundaries: list[float]) -> list[tuple[float, float]]:
    ends = [*boundaries[1:], float("inf")]
    return list(zip(boundaries, ends, strict=True))


def preview_splits(bags: list[Path], cfg: Config) -> dict[Path, list[float]]:
    """Episode-start timestamps (s) ``convert`` would split each bag on, without writing."""
    if not cfg.split_on:
        raise ValueError("config has no 'split_on' topic to preview")
    return {Path(bag): split_boundaries(bag, cfg.split_on, cfg) for bag in bags}


def convert(
    bags: list[Path], cfg: Config, out: Path, overwrite: bool = False
) -> list[ConversionReport]:
    """Convert each bag into one or more episodes of a new LeRobot dataset at ``out``.

    When ``cfg.split_on`` is set, each bag is split into multiple episodes at the
    timestamps of messages on that topic (see ``extract.split_boundaries``); otherwise
    each bag becomes exactly one episode.
    """
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty (use --overwrite)")
        shutil.rmtree(out)

    joint_order: dict[str, list[str]] = {}
    episodes, reports = [], []
    for bag in bags:
        streams = extract_streams(bag, cfg, joint_order)
        image_streams = extract_image_streams(bag, cfg)
        windows = _windows(split_boundaries(bag, cfg.split_on, cfg)) if cfg.split_on else [None]
        for segment, window in enumerate(windows):
            result = synchronize(streams, cfg, image_streams, window=window)
            episodes.append(Episode(result, cfg.task))
            reports.append(ConversionReport(Path(bag), result, segment))
    write_dataset(out, episodes, cfg.fps, cfg.robot_type)
    return reports
