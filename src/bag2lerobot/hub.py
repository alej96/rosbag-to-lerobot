"""Publishing a converted dataset to the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

from bag2lerobot.config import Config
from bag2lerobot.convert import ConversionReport
from bag2lerobot.writer import INDEX_FEATURES


class HubError(ValueError):
    pass


def dataset_card(repo_id: str, cfg: Config, info: dict, reports: list[ConversionReport]) -> str:
    """Build a Hub dataset card (README.md) describing the bags, config and sync quality
    that produced this dataset, so it's reviewable without re-running the conversion."""
    lines = [
        "---",
        "license: mit",
        "tags:",
        "  - LeRobot",
        "  - robotics",
        "---",
        "",
        f"# {repo_id}",
        "",
        "Converted from ROS bags with [bag2lerobot](https://github.com/alej96/rosbag-to-lerobot).",
        "",
        f"- **Episodes:** {info['total_episodes']}",
        f"- **Frames:** {info['total_frames']}",
        f"- **FPS:** {info['fps']:g}",
        f"- **Robot type:** {cfg.robot_type or 'unspecified'}",
        f"- **Task:** {cfg.task or 'unspecified'}",
        "",
        "## Source bags",
        "",
        "| bag | episode | frames | stale frames | max gap (s) |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(reports):
        label = r.bag.name if r.segment == 0 else f"{r.bag.name} (segment {r.segment})"
        stale = sum(s.stale_frames for s in r.sync.streams)
        gap = max((s.max_gap_s for s in r.sync.streams), default=0.0)
        lines.append(f"| {label} | {i} | {r.sync.frames} | {stale} | {gap:.3f} |")

    lines += ["", "## Features", "", "| key | dtype | shape |", "|---|---|---|"]
    for key, feat in info["features"].items():
        if key in INDEX_FEATURES:
            continue
        lines.append(f"| {key} | {feat['dtype']} | {feat['shape']} |")
    lines.append("")
    return "\n".join(lines)


def push_to_hub(
    out: Path, repo_id: str, private: bool = False, changed_files: list[Path] | None = None
) -> str:
    """Create (or update) a Hub dataset repo and upload ``out``'s contents to it.

    ``changed_files``, when given, uploads only those paths (relative to ``out``)
    instead of the whole folder: a re-push after ``--append`` only adds new data/
    episode/video files and rewrites the small metadata files, so re-uploading
    episodes already on the Hub (potentially a lot of encoded video) is wasted
    bandwidth every time more demos come in.

    Returns the dataset's URL on the Hub.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise HubError(
            "--push-to-hub needs the 'hub' extra: pip install 'bag2lerobot[hub]'"
        ) from exc
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    if changed_files is None:
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out))
    else:
        for rel in changed_files:
            api.upload_file(
                path_or_fileobj=str(out / rel),
                path_in_repo=str(rel),
                repo_id=repo_id,
                repo_type="dataset",
            )
    return f"https://huggingface.co/datasets/{repo_id}"
