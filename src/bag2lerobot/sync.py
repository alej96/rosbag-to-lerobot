"""Resampling asynchronous topic streams onto one fixed-rate frame grid."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bag2lerobot.config import Config
from bag2lerobot.extract import ImageStream, Stream


@dataclass
class StreamReport:
    key: str
    topic: str
    messages: int
    rate_hz: float
    max_gap_s: float
    stale_frames: int  # frames whose newest sample is older than max_staleness_s
    max_staleness_s: float  # threshold used for this stream (may be a per-feature override)


@dataclass
class SyncResult:
    t0: float  # absolute time (s) of frame 0
    frames: int
    values: dict[str, np.ndarray]  # key -> (frames, D) float32
    names: dict[str, list[str]]
    streams: list[StreamReport] = field(default_factory=list)
    trimmed_start_s: float = 0.0
    trimmed_end_s: float = 0.0
    images: dict[str, list] = field(default_factory=dict)  # key -> one frame per output row

    @property
    def stale_frames(self) -> int:
        return max((s.stale_frames for s in self.streams), default=0)


class SyncError(ValueError):
    pass


def synchronize(
    streams: dict[str, Stream],
    cfg: Config,
    image_streams: dict[str, ImageStream] | None = None,
    window: tuple[float, float] | None = None,
) -> SyncResult:
    """Sample every stream at ``cfg.fps`` over the window where all of them have data.

    Frames are evenly spaced (timestamp = frame_index / fps), which is what LeRobot's
    loader requires. Gaps inside a stream are filled (hold or linear) and counted as
    stale frames rather than silently dropped. Image streams are always held (the most
    recent decoded frame) rather than interpolated. ``window``, when given, further
    restricts the frame grid to ``[window[0], window[1])`` (e.g. one episode split out of
    a longer bag via ``cfg.split_on`` or ``cfg.split_idle_gap_s``); ``window[1]`` may be
    ``float("inf")``.
    """
    image_streams = image_streams or {}
    all_times = [s.times for s in streams.values()] + [s.times for s in image_streams.values()]
    starts = [t[0] for t in all_times]
    ends = [t[-1] for t in all_times]
    earliest, latest = min(starts), max(ends)
    t0, t1 = (max(starts), min(ends)) if cfg.trim_to_overlap else (earliest, latest)
    if window is not None:
        t0, t1 = max(t0, window[0]), min(t1, window[1])
    if t1 <= t0:
        if window is not None:
            end = "end" if window[1] == float("inf") else f"{window[1]:.3f}s"
            raise SyncError(f"no overlapping data in episode window [{window[0]:.3f}s, {end})")
        raise SyncError(
            "streams don't overlap in time: "
            + ", ".join(
                f"{s.key} [{s.times[0]:.2f}, {s.times[-1]:.2f}]s"
                for s in (*streams.values(), *image_streams.values())
            )
        )

    dt = 1.0 / cfg.fps
    frames = int(np.floor((t1 - t0) / dt + 1e-9)) + 1
    grid = t0 + np.arange(frames) * dt

    specs = {f.key: f for f in cfg.features}
    values, names, reports = {}, {}, []
    for key, s in streams.items():
        spec = specs.get(key)
        method = (spec.sync if spec and spec.sync else None) or cfg.sync
        staleness = (
            spec.max_staleness_s
            if spec and spec.max_staleness_s is not None
            else cfg.max_staleness_s
        )
        idx = np.clip(np.searchsorted(s.times, grid, side="right") - 1, 0, len(s.times) - 1)
        if method == "linear" and len(s.times) > 1:
            out = np.column_stack(
                [np.interp(grid, s.times, s.values[:, d]) for d in range(s.values.shape[1])]
            )
        else:
            out = s.values[idx]
        age = grid - s.times[idx]
        # before a stream's first sample (only possible without trimming) is also stale
        stale = int(((age > staleness) | (grid < s.times[0])).sum())
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
                max_staleness_s=staleness,
            )
        )
        values[key] = out.astype(np.float32)
        names[key] = s.names

    images: dict[str, list] = {}
    for key, s in image_streams.items():
        spec = specs.get(key)
        staleness = (
            spec.max_staleness_s
            if spec and spec.max_staleness_s is not None
            else cfg.max_staleness_s
        )
        idx = np.clip(np.searchsorted(s.times, grid, side="right") - 1, 0, len(s.times) - 1)
        age = grid - s.times[idx]
        stale = int(((age > staleness) | (grid < s.times[0])).sum())
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
                max_staleness_s=staleness,
            )
        )
        images[key] = [s.frames[i] for i in idx]

    return SyncResult(
        t0=t0,
        frames=frames,
        values=values,
        names=names,
        streams=reports,
        trimmed_start_s=t0 - earliest,
        trimmed_end_s=latest - t1,
        images=images,
    )
