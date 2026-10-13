"""Command-line interface: ``bag2lerobot inspect`` and ``bag2lerobot convert``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bag2lerobot import __version__
from bag2lerobot.config import ConfigError, load_config
from bag2lerobot.convert import convert, preview_splits
from bag2lerobot.extract import ExtractError, inspect_bag
from bag2lerobot.init import generate_config
from bag2lerobot.sync import SyncError


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bag2lerobot", description=__doc__)
    p.add_argument("--version", action="version", version=f"bag2lerobot {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    ins = sub.add_parser("inspect", help="list topics, message types and rates in a bag")
    ins.add_argument(
        "bag", type=Path, help="ROS 1 .bag file, or ROS 2 bag directory (sqlite3/mcap)"
    )

    ini = sub.add_parser("init", help="generate a starter config from a bag's topics")
    ini.add_argument(
        "bag", type=Path, help="ROS 1 .bag file, or ROS 2 bag directory (sqlite3/mcap)"
    )
    ini.add_argument("--out", "-o", type=Path, help="write the config here (default: stdout)")
    ini.add_argument(
        "--fps", type=float, help="output frame rate (default: fastest JointState topic's rate)"
    )

    conv = sub.add_parser("convert", help="convert bags into a LeRobot dataset")
    conv.add_argument("bags", type=Path, nargs="+", help="one episode per bag")
    conv.add_argument("--config", "-c", type=Path, required=True, help="YAML feature mapping")
    conv.add_argument("--out", "-o", type=Path, help="output dataset directory")
    conv.add_argument("--task", help="task description (overrides the config)")
    conv.add_argument("--overwrite", action="store_true", help="replace an existing output dir")
    conv.add_argument(
        "--strict", action="store_true", help="fail if any frame had to use stale data"
    )
    conv.add_argument(
        "--dry-run",
        action="store_true",
        help="print the episode boundaries from the config's 'split_on' topic and exit",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "init":
            return _init(args)
        return _convert(args)
    except (
        ConfigError,
        ExtractError,
        SyncError,
        ValueError,
        FileExistsError,
        FileNotFoundError,
    ) as exc:
        print(f"bag2lerobot: {exc}", file=sys.stderr)
        return 2


def _inspect(args) -> int:
    topics = inspect_bag(args.bag)
    width = max((len(t.topic) for t in topics), default=5)
    print(f"{'topic':<{width}}  {'type':<40} {'msgs':>7} {'rate':>9}")
    for t in topics:
        print(f"{t.topic:<{width}}  {t.msgtype:<40} {t.count:>7} {t.rate_hz:>7.1f}Hz")
    return 0


def _init(args) -> int:
    text = generate_config(args.bag, fps=args.fps)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _dry_run(args, cfg) -> int:
    if not cfg.split_on:
        print("bag2lerobot: --dry-run needs 'split_on' in the config", file=sys.stderr)
        return 2
    for bag, boundaries in preview_splits(args.bags, cfg).items():
        print(f"{bag.name}: {len(boundaries)} episode(s) on '{cfg.split_on}'")
        for i, start in enumerate(boundaries):
            end = f"{boundaries[i + 1]:.3f}s" if i + 1 < len(boundaries) else "end"
            print(f"  ep {i}: {start:.3f}s -> {end}")
    return 0


def _convert(args) -> int:
    cfg = load_config(args.config)
    if args.task:
        cfg.task = args.task
    if args.dry_run:
        return _dry_run(args, cfg)
    if not args.out:
        print("bag2lerobot: --out is required (unless --dry-run)", file=sys.stderr)
        return 2
    reports = convert(args.bags, cfg, args.out, overwrite=args.overwrite)
    total = sum(r.sync.frames for r in reports)
    print(f"wrote {len(reports)} episode(s), {total} frames at {cfg.fps:g} fps to {args.out}")
    any_stale = False
    for i, r in enumerate(reports):
        s = r.sync
        label = r.bag.name if not cfg.split_on else f"{r.bag.name} seg {r.segment}"
        print(
            f"  ep {i}: {label}  {s.frames} frames  "
            f"(trimmed {s.trimmed_start_s:.2f}s start, {s.trimmed_end_s:.2f}s end)"
        )
        for st in s.streams:
            line = (
                f"      {st.key:<24} {st.topic:<28} {st.rate_hz:6.1f}Hz  "
                f"max gap {st.max_gap_s:.3f}s"
            )
            if st.stale_frames:
                any_stale = True
                line += f"  !! {st.stale_frames} stale frame(s) > {st.max_staleness_s}s"
            print(line)
    if any_stale and args.strict:
        print("bag2lerobot: stale frames found (--strict)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
