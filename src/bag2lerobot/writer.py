"""Writing a LeRobot v3.0 dataset (tabular features; no video yet)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bag2lerobot.sync import SyncResult

CODEBASE_VERSION = "v3.0"
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
_INDEX_FEATURES = {
    "timestamp": "float32",
    "frame_index": "int64",
    "episode_index": "int64",
    "index": "int64",
    "task_index": "int64",
}


@dataclass
class Episode:
    sync: SyncResult
    task: str


def write_dataset(out: Path, episodes: list[Episode], fps: float, robot_type: str | None) -> dict:
    """Write episodes as a LeRobot v3.0 dataset under ``out``. Returns info.json."""
    if not episodes:
        raise ValueError("nothing to write")
    keys = list(episodes[0].sync.values)
    for ep in episodes[1:]:
        if list(ep.sync.values) != keys:
            raise ValueError("all episodes must have the same features")
        for k in keys:
            if ep.sync.values[k].shape[1] != episodes[0].sync.values[k].shape[1]:
                raise ValueError(f"feature '{k}' has a different size across episodes")

    tasks = list(dict.fromkeys(ep.task for ep in episodes))
    task_index = {t: i for i, t in enumerate(tasks)}

    frames, ep_rows = [], []
    index = 0
    for ep_idx, ep in enumerate(episodes):
        n = ep.sync.frames
        frame = {k: list(ep.sync.values[k]) for k in keys}
        frame.update(
            timestamp=(np.arange(n) / fps).astype(np.float32),
            frame_index=np.arange(n, dtype=np.int64),
            episode_index=np.full(n, ep_idx, dtype=np.int64),
            index=np.arange(index, index + n, dtype=np.int64),
            task_index=np.full(n, task_index[ep.task], dtype=np.int64),
        )
        df = pd.DataFrame(frame)
        frames.append(df)
        row = {
            "episode_index": ep_idx,
            "tasks": [ep.task],
            "length": n,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": index,
            "dataset_to_index": index + n,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for col, stats in _stats(df, keys).items():
            for stat, value in stats.items():
                row[f"stats/{col}/{stat}"] = value
        ep_rows.append(row)
        index += n

    data = pd.concat(frames, ignore_index=True)
    for k in keys:
        data[k] = data[k].map(lambda v: np.asarray(v, dtype=np.float32))

    out = Path(out)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    data.to_parquet(out / DATA_PATH.format(chunk_index=0, file_index=0), index=False)
    pd.DataFrame(ep_rows).to_parquet(
        out / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False
    )
    pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task")).to_parquet(
        out / "meta" / "tasks.parquet"
    )
    (out / "meta" / "stats.json").write_text(json.dumps(_stats(data, keys), indent=2))

    names = episodes[0].sync.names
    features = {
        k: {
            "dtype": "float32",
            "shape": [episodes[0].sync.values[k].shape[1]],
            "names": names[k],
            "fps": fps,
        }
        for k in keys
    }
    features.update(
        {
            k: {"dtype": d, "shape": [1], "names": None, "fps": fps}
            for k, d in _INDEX_FEATURES.items()
        }
    )
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": len(episodes),
        "total_frames": int(index),
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": DATA_PATH,
        "video_path": None,
        "features": features,
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    return info


def _stats(df: pd.DataFrame, keys: list[str]) -> dict:
    out = {}
    for col in [*keys, *_INDEX_FEATURES]:
        arr = np.stack(df[col].to_numpy()) if col in keys else df[col].to_numpy()[:, None]
        arr = arr.astype(np.float64)
        out[col] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [len(arr)],
        }
    return out
