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


def push_to_hub(out: Path, repo_id: str, private: bool = False) -> str:
    """Create (or update) a Hub dataset repo and upload ``out``'s contents to it.

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
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out))
    return f"https://huggingface.co/datasets/{repo_id}"
