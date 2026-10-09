"""End-to-end conversion: bags -> synchronized episodes -> LeRobot dataset."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from bag2lerobot.config import Config
from bag2lerobot.extract import extract_streams
from bag2lerobot.sync import SyncResult, synchronize
from bag2lerobot.writer import Episode, write_dataset


@dataclass
class ConversionReport:
    bag: Path
    sync: SyncResult


def convert(
    bags: list[Path], cfg: Config, out: Path, overwrite: bool = False
) -> list[ConversionReport]:
    """Convert each bag into one episode of a new LeRobot dataset at ``out``."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty (use --overwrite)")
        shutil.rmtree(out)

    joint_order: dict[str, list[str]] = {}
    episodes, reports = [], []
    for bag in bags:
        streams = extract_streams(bag, cfg, joint_order)
        result = synchronize(streams, cfg)
        episodes.append(Episode(result, cfg.task))
        reports.append(ConversionReport(Path(bag), result))
    write_dataset(out, episodes, cfg.fps, cfg.robot_type)
    return reports
