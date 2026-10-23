"""Writing a LeRobot v3.0 dataset: tabular features plus video for image features."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bag2lerobot.sync import SyncResult

CODEBASE_VERSION = "v3.0"
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
VIDEO_CODEC = "libx264"
VIDEO_PIX_FMT = "yuv420p"
DEFAULT_DATA_FILES_SIZE_IN_MB = 100.0
INDEX_FEATURES = {
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


@dataclass
class _Existing:
    info: dict
    task_index: dict[str, int]
    episode_offset: int
    index_offset: int
    data_file_index: int
    episodes_file_index: int
    video_file_index: dict[str, int]
    stats: dict


def write_dataset(
    out: Path,
    episodes: Iterable[Episode],
    fps: float,
    robot_type: str | None,
    append: bool = False,
    max_data_file_mb: float = DEFAULT_DATA_FILES_SIZE_IN_MB,
) -> dict:
    """Write episodes as a LeRobot v3.0 dataset under ``out``. Returns info.json.

    With ``append=True``, ``out`` must already hold a dataset written by this function;
    the new episodes are added to it (new episode/frame indices continuing from the
    existing totals, new data file(s), a new per-key video file and merged tasks/stats)
    instead of replacing it. The existing dataset's fps and features must match.

    The data parquet is split across multiple files, each kept under
    ``max_data_file_mb`` (an episode is never split across files, so a single episode
    larger than the limit still gets one file to itself), instead of writing every
    episode from one call into a single, unboundedly large file.

    ``episodes`` is consumed in one pass, writing (and, for image features, encoding)
    each episode before asking for the next, instead of first collecting every episode
    (decoded video frames included) into a list: peak memory is bounded by one episode,
    not by how many episodes the caller passes in. A caller may still pass a plain
    list; only a generator actually benefits. If an episode partway through turns out
    to have a mismatched feature or image resolution, the episodes before it have
    already been written to ``out``; callers that need all-or-nothing writes should
    remove a freshly-created ``out`` on failure themselves (``append=True`` already
    only ever adds new files, so a failure there just leaves the earlier ones appended).
    """
    episodes = iter(episodes)
    try:
        first = next(episodes)
    except StopIteration:
        raise ValueError("nothing to write") from None
    episodes = itertools.chain([first], episodes)

    keys = list(first.sync.values)
    image_keys = list(first.sync.images)
    value_widths = {k: first.sync.values[k].shape[1] for k in keys}
    image_shapes = {k: first.sync.images[k][0].shape for k in image_keys}
    # LeRobot's hf_features reads a width-1 feature as a scalar datasets.Value, not a
    # length-1 datasets.Sequence; write those columns as plain floats, not 1-element
    # arrays, so the parquet dtype matches what info.json's shape [1] tells it to expect.
    scalar_keys = {k for k in keys if value_widths[k] == 1}

    out = Path(out)
    existing = _load_existing(out) if append else None
    if existing is not None:
        _check_append_schema(existing.info, [first], fps, robot_type)
        task_index = dict(existing.task_index)
        next_task_idx = max(task_index.values(), default=-1) + 1
        episode_offset = existing.episode_offset
        index_offset = existing.index_offset
        data_file_index = existing.data_file_index
        episodes_file_index = existing.episodes_file_index
        video_file_index = existing.video_file_index
    else:
        task_index = {}
        next_task_idx = 0
        episode_offset = 0
        index_offset = 0
        data_file_index = 0
        episodes_file_index = 0
        video_file_index = dict.fromkeys(image_keys, 0)

    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    max_bytes = max_data_file_mb * 1024 * 1024

    ep_rows: list[dict] = []
    index = index_offset
    video_t = dict.fromkeys(image_keys, 0.0)  # cumulative seconds written so far, per video
    file_index = data_file_index
    chunk_frames: list[pd.DataFrame] = []
    chunk_bytes = 0
    running_stats: dict | None = None

    def flush_chunk() -> None:
        nonlocal file_index, chunk_frames, chunk_bytes
        if not chunk_frames:
            return
        chunk = pd.concat(chunk_frames, ignore_index=True)
        for k in keys:
            if k in scalar_keys:
                chunk[k] = chunk[k].astype(np.float32)
            else:
                chunk[k] = chunk[k].map(lambda v: np.asarray(v, dtype=np.float32))
        chunk.to_parquet(out / DATA_PATH.format(chunk_index=0, file_index=file_index), index=False)
        file_index += 1
        chunk_frames = []
        chunk_bytes = 0

    video_containers, video_streams = _open_video_streams(
        out, image_keys, video_file_index, fps, image_shapes
    )

    try:
        n_episodes = 0
        for i, ep in enumerate(episodes):
            if list(ep.sync.values) != keys:
                raise ValueError("all episodes must have the same features")
            if list(ep.sync.images) != image_keys:
                raise ValueError("all episodes must have the same image features")
            for k in keys:
                if ep.sync.values[k].shape[1] != value_widths[k]:
                    raise ValueError(f"feature '{k}' has a different size across episodes")
            for k in image_keys:
                if ep.sync.images[k][0].shape != image_shapes[k]:
                    raise ValueError(
                        f"image feature '{k}' has a different resolution across episodes"
                    )
            if ep.task not in task_index:
                task_index[ep.task] = next_task_idx
                next_task_idx += 1

            ep_idx = episode_offset + i
            n = ep.sync.frames
            frame = {
                k: (
                    ep.sync.values[k][:, 0].astype(np.float32)
                    if k in scalar_keys
                    else list(ep.sync.values[k])
                )
                for k in keys
            }
            frame.update(
                timestamp=(np.arange(n) / fps).astype(np.float32),
                frame_index=np.arange(n, dtype=np.int64),
                episode_index=np.full(n, ep_idx, dtype=np.int64),
                index=np.arange(index, index + n, dtype=np.int64),
                task_index=np.full(n, task_index[ep.task], dtype=np.int64),
            )
            df = pd.DataFrame(frame)
            df_bytes = int(df.memory_usage(deep=True).sum())
            if chunk_frames and chunk_bytes + df_bytes > max_bytes:
                flush_chunk()
            chunk_frames.append(df)
            chunk_bytes += df_bytes
            row = {
                "episode_index": ep_idx,
                "tasks": [ep.task],
                "length": n,
                "data/chunk_index": 0,
                "data/file_index": file_index,
                "dataset_from_index": index,
                "dataset_to_index": index + n,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": episodes_file_index,
            }
            for key in image_keys:
                from_t = video_t[key]
                to_t = from_t + n / fps
                row[f"videos/{key}/chunk_index"] = 0
                row[f"videos/{key}/file_index"] = video_file_index[key]
                row[f"videos/{key}/from_timestamp"] = from_t
                row[f"videos/{key}/to_timestamp"] = to_t
                video_t[key] = to_t
                _encode_frames(video_containers[key], video_streams[key], ep.sync.images[key])
            ep_stats = _stats(df, keys, scalar_keys)
            for col, stats in ep_stats.items():
                for stat, value in stats.items():
                    row[f"stats/{col}/{stat}"] = value
            running_stats = (
                ep_stats if running_stats is None else _merge_stats(running_stats, ep_stats)
            )
            ep_rows.append(row)
            index += n
            n_episodes += 1
    except Exception:
        for container in video_containers.values():
            container.close()
        raise

    flush_chunk()
    for key in image_keys:
        for packet in video_streams[key].encode():
            video_containers[key].mux(packet)
        video_containers[key].close()

    pd.DataFrame(ep_rows).to_parquet(
        out / "meta" / "episodes" / "chunk-000" / f"file-{episodes_file_index:03d}.parquet",
        index=False,
    )
    tasks = sorted(task_index, key=task_index.get)
    pd.DataFrame(
        {"task_index": [task_index[t] for t in tasks]}, index=pd.Index(tasks, name="task")
    ).to_parquet(out / "meta" / "tasks.parquet")
    new_stats = running_stats or {}
    stats = _merge_stats(existing.stats, new_stats) if existing is not None else new_stats
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    names = first.sync.names
    features = {
        k: {
            "dtype": "float32",
            "shape": [value_widths[k]],
            "names": names[k],
            "fps": fps,
        }
        for k in keys
    }
    features.update(
        {
            k: {"dtype": d, "shape": [1], "names": None, "fps": fps}
            for k, d in INDEX_FEATURES.items()
        }
    )
    for key in image_keys:
        h, w, c = image_shapes[key]
        features[key] = {
            "dtype": "video",
            "shape": [h, w, c],
            "names": ["height", "width", "channel"],
            "fps": fps,
            "video_info": {
                "video.fps": fps,
                "video.height": h,
                "video.width": w,
                "video.channels": c,
                "video.codec": VIDEO_CODEC,
                "video.pix_fmt": VIDEO_PIX_FMT,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    total_episodes = episode_offset + n_episodes
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": int(index),
        "total_tasks": len(task_index),
        "chunks_size": 1000,
        "data_files_size_in_mb": max_data_file_mb,
        "video_files_size_in_mb": 500,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH if image_keys else None,
        "features": features,
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    return info


def _load_existing(out: Path) -> _Existing:
    info_path = out / "meta" / "info.json"
    if not info_path.exists():
        raise ValueError(f"--append: {out} has no dataset to append to (missing {info_path})")
    info = json.loads(info_path.read_text())
    tasks_df = pd.read_parquet(out / "meta" / "tasks.parquet")
    task_index = dict(zip(tasks_df.index, (int(i) for i in tasks_df["task_index"]), strict=True))
    stats = json.loads((out / "meta" / "stats.json").read_text())
    data_files = sorted((out / "data" / "chunk-000").glob("file-*.parquet"))
    episodes_files = sorted((out / "meta" / "episodes" / "chunk-000").glob("file-*.parquet"))
    video_file_index = {}
    for key, feat in info["features"].items():
        if feat["dtype"] == "video":
            video_dir = out / "videos" / key / "chunk-000"
            files = list(video_dir.glob("file-*.mp4")) if video_dir.exists() else []
            video_file_index[key] = len(files)
    return _Existing(
        info=info,
        task_index=task_index,
        episode_offset=info["total_episodes"],
        index_offset=info["total_frames"],
        data_file_index=len(data_files),
        episodes_file_index=len(episodes_files),
        video_file_index=video_file_index,
        stats=stats,
    )


def _check_append_schema(
    info: dict, episodes: list[Episode], fps: float, robot_type: str | None
) -> None:
    if info["fps"] != fps:
        raise ValueError(
            f"--append: fps mismatch (existing dataset is {info['fps']}, config is {fps})"
        )
    if info.get("robot_type") != robot_type:
        raise ValueError(
            f"--append: robot_type mismatch (existing dataset is {info.get('robot_type')!r}, "
            f"config is {robot_type!r})"
        )
    keys = list(episodes[0].sync.values)
    image_keys = list(episodes[0].sync.images)
    existing_keys = set(info["features"]) - set(INDEX_FEATURES)
    new_keys = set(keys) | set(image_keys)
    if existing_keys != new_keys:
        raise ValueError(
            f"--append: feature mismatch (existing dataset has {sorted(existing_keys)}, "
            f"config produces {sorted(new_keys)})"
        )
    for k in keys:
        shape = [episodes[0].sync.values[k].shape[1]]
        if info["features"][k]["shape"] != shape:
            raise ValueError(
                f"--append: feature '{k}' has shape {info['features'][k]['shape']} in the "
                f"existing dataset, but {shape} now"
            )
    for k in image_keys:
        h, w, c = episodes[0].sync.images[k][0].shape
        if info["features"][k]["shape"] != [h, w, c]:
            raise ValueError(
                f"--append: image feature '{k}' has resolution {info['features'][k]['shape']} in "
                f"the existing dataset, but [{h}, {w}, {c}] now"
            )


def _merge_stats(old: dict, new: dict) -> dict:
    merged = {}
    for col, new_stat in new.items():
        old_stat = old.get(col)
        if old_stat is None:
            merged[col] = new_stat
            continue
        n1 = np.asarray(old_stat["count"], dtype=np.float64)
        n2 = np.asarray(new_stat["count"], dtype=np.float64)
        mean1, mean2 = np.asarray(old_stat["mean"]), np.asarray(new_stat["mean"])
        std1, std2 = np.asarray(old_stat["std"]), np.asarray(new_stat["std"])
        n = n1 + n2
        mean = (mean1 * n1 + mean2 * n2) / n
        var = (n1 * (std1**2 + (mean1 - mean) ** 2) + n2 * (std2**2 + (mean2 - mean) ** 2)) / n
        merged[col] = {
            "min": np.minimum(old_stat["min"], new_stat["min"]).tolist(),
            "max": np.maximum(old_stat["max"], new_stat["max"]).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "count": n.tolist(),
        }
    return merged


def _open_video_streams(
    out: Path,
    image_keys: list[str],
    video_file_index: dict[str, int],
    fps: float,
    image_shapes: dict[str, tuple[int, int, int]],
):
    """Open one MP4 container+stream per image key, ready to encode frames as episodes
    arrive, instead of collecting every episode's decoded frames before writing any of
    them."""
    if not image_keys:
        return {}, {}
    try:
        import av
    except ImportError as exc:
        raise ValueError(
            "writing image features needs the 'video' extra: pip install 'bag2lerobot[video]'"
        ) from exc
    containers, streams = {}, {}
    for key in image_keys:
        path = out / VIDEO_PATH.format(
            video_key=key, chunk_index=0, file_index=video_file_index[key]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        container = av.open(str(path), mode="w")
        stream = container.add_stream(VIDEO_CODEC, rate=round(fps))
        h, w, _ = image_shapes[key]
        stream.height = h
        stream.width = w
        stream.pix_fmt = VIDEO_PIX_FMT
        containers[key] = container
        streams[key] = stream
    return containers, streams


def _encode_frames(container, stream, frames: list[np.ndarray]) -> None:
    import av

    for frame in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)


def _stats(df: pd.DataFrame, keys: list[str], scalar_keys: set[str] = frozenset()) -> dict:
    out = {}
    for col in [*keys, *INDEX_FEATURES]:
        # a scalar (width-1) feature column holds plain floats, like an index column,
        # not one row-vector per row, so it needs the same [:, None] reshape as those
        stack = col in keys and col not in scalar_keys
        arr = np.stack(df[col].to_numpy()) if stack else df[col].to_numpy()[:, None]
        arr = arr.astype(np.float64)
        out[col] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [len(arr)],
        }
    return out
