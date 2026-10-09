"""Resampling asynchronous topic streams onto one fixed-rate frame grid."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bag2lerobot.config import Config
from bag2lerobot.extract import Stream


@dataclass
class StreamReport:
    key: str
    topic: str
    messages: int
    rate_hz: float
    max_gap_s: float
    stale_frames: int  # frames whose newest sample is older than max_staleness_s


@dataclass
class SyncResult:
    t0: float  # absolute time (s) of frame 0
    frames: int
    values: dict[str, np.ndarray]  # key -> (frames, D) float32
    names: dict[str, list[str]]
    streams: list[StreamReport] = field(default_factory=list)
    trimmed_start_s: float = 0.0
    trimmed_end_s: float = 0.0

    @property
    def stale_frames(self) -> int:
        return max((s.stale_frames for s in self.streams), default=0)


class SyncError(ValueError):
    pass


def synchronize(streams: dict[str, Stream], cfg: Config) -> SyncResult:
    """Sample every stream at ``cfg.fps`` over the window where all of them have data.

    Frames are evenly spaced (timestamp = frame_index / fps), which is what LeRobot's
    loader requires. Gaps inside a stream are filled (hold or linear) and counted as
    stale frames rather than silently dropped.
    """
    starts = [s.times[0] for s in streams.values()]
    ends = [s.times[-1] for s in streams.values()]
    earliest, latest = min(starts), max(ends)
    t0, t1 = (max(starts), min(ends)) if cfg.trim_to_overlap else (earliest, latest)
    if t1 <= t0:
        raise SyncError(
            "streams don't overlap in time: "
            + ", ".join(f"{s.key} [{s.times[0]:.2f}, {s.times[-1]:.2f}]s" for s in streams.values())
        )

    dt = 1.0 / cfg.fps
    frames = int(np.floor((t1 - t0) / dt + 1e-9)) + 1
    grid = t0 + np.arange(frames) * dt

    values, names, reports = {}, {}, []
    for key, s in streams.items():
        idx = np.clip(np.searchsorted(s.times, grid, side="right") - 1, 0, len(s.times) - 1)
        if cfg.sync == "linear" and len(s.times) > 1:
            out = np.column_stack(
                [np.interp(grid, s.times, s.values[:, d]) for d in range(s.values.shape[1])]
            )
        else:
            out = s.values[idx]
        age = grid - s.times[idx]
        # before a stream's first sample (only possible without trimming) is also stale
        stale = int(((age > cfg.max_staleness_s) | (grid < s.times[0])).sum())
        gaps = np.diff(s.times)
        reports.append(
            StreamReport(
                key=key,
                topic=s.topic,
                messages=len(s.times),
                rate_hz=(len(s.times) - 1) / (s.times[-1] - s.times[0])
                if s.times[-1] > s.times[0]
                else 0.0,
                max_gap_s=float(gaps.max()) if len(gaps) else 0.0,
                stale_frames=stale,
            )
        )
        values[key] = out.astype(np.float32)
        names[key] = s.names

    return SyncResult(
        t0=t0,
        frames=frames,
        values=values,
        names=names,
        streams=reports,
        trimmed_start_s=t0 - earliest,
        trimmed_end_s=latest - t1,
    )
