"""Publishing a converted dataset to the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

from bag2lerobot.config import Config
from bag2lerobot.writer import INDEX_FEATURES


class HubError(ValueError):
    pass


def dataset_card(repo_id: str, cfg: Config, info: dict, source_log: list[dict]) -> str:
    """Build a Hub dataset card (README.md) describing the bags, config and sync quality
    that produced this dataset, so it's reviewable without re-running the conversion.

    ``source_log`` is bag2lerobot's own per-episode provenance record
    (``meta/source_bags.json``, written by ``convert``), covering every episode the
    dataset has ever had -- not just the ones from the most recent ``convert`` call --
    so a card built after ``--append`` still lists the bags from earlier calls too.
    """
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
    for i, rec in enumerate(source_log):
        label = rec["bag"] if rec["segment"] == 0 else f"{rec['bag']} (segment {rec['segment']})"
        lines.append(
            f"| {label} | {i} | {rec['frames']} | {rec['stale_frames']} | {rec['max_gap_s']:.3f} |"
        )

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
