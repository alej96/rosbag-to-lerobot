import json

import pandas as pd
import pytest

from bag2lerobot.cli import main
from bag2lerobot.config import ConfigError, load_config, parse_config
from bag2lerobot.convert import convert, preview_splits
from bag2lerobot.extract import (
    ExtractError,
    csv_windows,
    extract_image_streams,
    extract_streams,
    idle_gap_boundaries,
    split_boundaries,
)
from tests.conftest import T0, BagSpec

SPLIT_CONFIG = """
fps: 30
task: pick the cube
split_on: /episode_start
features:
  observation.state:
    topic: /follower/joint_states
    field: position
    names_from: name
  action:
    topic: /leader/joint_states
    field: position
    names_from: name
"""

IDLE_GAP_CONFIG = """
fps: 30
task: pick the cube
split_idle_gap_s: 0.3
features:
  observation.state:
    topic: /follower/joint_states
    field: position
    names_from: name
  action:
    topic: /leader/joint_states
    field: position
    names_from: name
"""


def _split_cfg(topic="/episode_start"):
    return parse_config(
        {
            "fps": 30,
            "split_on": topic,
            "features": {"a": {"topic": "/follower/joint_states", "field": "position"}},
        }
    )


@pytest.fixture
def split_config_file(tmp_path):
    path = tmp_path / "split.yaml"
    path.write_text(SPLIT_CONFIG)
    return path


@pytest.fixture
def idle_gap_config_file(tmp_path):
    path = tmp_path / "idle_gap.yaml"
    path.write_text(IDLE_GAP_CONFIG)
    return path


CSV_SPLIT_CONFIG = """
fps: 30
task: pick the cube
split_csv: {csv_path}
features:
  observation.state:
    topic: /follower/joint_states
    field: position
    names_from: name
  action:
    topic: /leader/joint_states
    field: position
    names_from: name
"""


@pytest.fixture
def csv_split_config_file(tmp_path):
    csv_path = tmp_path / "episodes.csv"
    csv_path.write_text("start,end,task\n0.0,1.5,pick red cube\n1.5,3.0,pick blue cube\n")
    cfg_path = tmp_path / "csv_split.yaml"
    cfg_path.write_text(CSV_SPLIT_CONFIG.format(csv_path=csv_path))
    return cfg_path, csv_path


def test_split_on_must_be_a_string():
    with pytest.raises(ConfigError, match="split_on"):
        parse_config({"fps": 30, "split_on": 3, "features": {"a": {"topic": "/t"}}})


@pytest.mark.parametrize("start_type", ["bool", "empty"])
def test_split_boundaries_reads_timestamps(make_bag, start_type):
    bag = make_bag(spec=BagSpec(episode_starts=[0.5, 1.7], episode_start_type=start_type))
    boundaries = split_boundaries(bag, "/episode_start", _split_cfg())
    assert len(boundaries) == 2
    assert boundaries[0] < boundaries[1]
    assert boundaries[0] == pytest.approx(T0 + 0.5, abs=0.02)
    assert boundaries[1] == pytest.approx(T0 + 1.7, abs=0.02)


def test_split_topic_missing_lists_available(make_bag):
    bag = make_bag()
    with pytest.raises(ExtractError, match="not in bag"):
        split_boundaries(bag, "/nope", _split_cfg("/nope"))


def test_convert_splits_one_bag_into_multiple_episodes(make_bag, split_config_file, tmp_path):
    bag = make_bag(spec=BagSpec(seconds=3.0, episode_starts=[0.0, 1.5]))
    out = tmp_path / "ds"
    reports = convert([bag], load_config(split_config_file), out)

    assert len(reports) == 2
    assert [r.segment for r in reports] == [0, 1]
    assert all(r.bag.name == bag.name for r in reports)
    # episode 0 spans ~[0.2, 1.5)s (leader starts 0.2s late), episode 1 spans ~[1.5, 3.0)s
    assert 35 <= reports[0].sync.frames <= 41
    assert 42 <= reports[1].sync.frames <= 46

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == reports[0].sync.frames + reports[1].sync.frames


def test_preview_splits_does_not_write(make_bag, split_config_file, tmp_path):
    bag = make_bag(spec=BagSpec(seconds=3.0, episode_starts=[0.0, 1.5]))
    cfg = load_config(split_config_file)
    boundaries = preview_splits([bag], cfg)
    assert len(boundaries[bag]) == 2
    assert not (tmp_path / "ds").exists()


def test_preview_splits_needs_split_on(make_bag, config_file):
    bag = make_bag()
    with pytest.raises(ValueError, match="split_on"):
        preview_splits([bag], load_config(config_file))


def test_cli_dry_run_prints_boundaries_without_writing(
    make_bag, split_config_file, tmp_path, capsys
):
    bag = make_bag(spec=BagSpec(seconds=3.0, episode_starts=[0.0, 1.5]))
    assert main(["convert", str(bag), "-c", str(split_config_file), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 episode(s)" in out
    assert "ep 0:" in out and "ep 1:" in out
    assert not (tmp_path / "ds").exists()


def test_cli_dry_run_needs_split_on(make_bag, config_file, capsys):
    bag = make_bag()
    assert main(["convert", str(bag), "-c", str(config_file), "--dry-run"]) == 2
    assert "split_on" in capsys.readouterr().err


def test_cli_convert_needs_out_without_dry_run(make_bag, config_file, capsys):
    bag = make_bag()
    assert main(["convert", str(bag), "-c", str(config_file)]) == 2
    assert "--out is required" in capsys.readouterr().err


@pytest.mark.lerobot
def test_split_output_loads_in_lerobot(make_bag, split_config_file, tmp_path):
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

    bag = make_bag(spec=BagSpec(seconds=3.0, episode_starts=[0.0, 1.5]))
    out = tmp_path / "ds"
    reports = convert([bag], load_config(split_config_file), out)
    assert len(reports) == 2

    ds = lerobot_dataset.LeRobotDataset("local/bag2lerobot-split-test", root=out)
    assert ds.num_episodes == 2
    assert ds.fps == 30


def test_split_on_and_idle_gap_mutually_exclusive():
    with pytest.raises(ConfigError, match="split_on.*split_idle_gap_s"):
        parse_config(
            {
                "fps": 30,
                "split_on": "/episode_start",
                "split_idle_gap_s": 0.5,
                "features": {"a": {"topic": "/t"}},
            }
        )


def test_split_idle_gap_s_must_be_positive():
    with pytest.raises(ConfigError, match="split_idle_gap_s"):
        parse_config({"fps": 30, "split_idle_gap_s": 0, "features": {"a": {"topic": "/t"}}})


def test_idle_gap_boundaries_finds_gap(make_bag, idle_gap_config_file):
    cfg = load_config(idle_gap_config_file)
    bag = make_bag(spec=BagSpec(seconds=3.0, idle_gap=(1.0, 1.5)))
    streams = extract_streams(bag, cfg)
    boundaries = idle_gap_boundaries(streams, extract_image_streams(bag, cfg), 0.3)
    assert len(boundaries) == 2
    assert boundaries[0] == pytest.approx(T0, abs=0.02)
    assert boundaries[1] == pytest.approx(T0 + 1.5, abs=0.05)


def test_idle_gap_boundaries_needs_data():
    with pytest.raises(ExtractError, match="no data"):
        idle_gap_boundaries({}, {}, 0.3)


def test_convert_splits_on_idle_gap(make_bag, idle_gap_config_file, tmp_path):
    bag = make_bag(spec=BagSpec(seconds=3.0, idle_gap=(1.0, 1.5)))
    out = tmp_path / "ds"
    reports = convert([bag], load_config(idle_gap_config_file), out)

    assert len(reports) == 2
    assert [r.segment for r in reports] == [0, 1]
    assert all(r.bag.name == bag.name for r in reports)

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == reports[0].sync.frames + reports[1].sync.frames


def test_preview_splits_with_idle_gap(make_bag, idle_gap_config_file, tmp_path):
    bag = make_bag(spec=BagSpec(seconds=3.0, idle_gap=(1.0, 1.5)))
    cfg = load_config(idle_gap_config_file)
    boundaries = preview_splits([bag], cfg)
    assert len(boundaries[bag]) == 2
    assert not (tmp_path / "ds").exists()


def test_cli_dry_run_with_idle_gap(make_bag, idle_gap_config_file, tmp_path, capsys):
    bag = make_bag(spec=BagSpec(seconds=3.0, idle_gap=(1.0, 1.5)))
    assert main(["convert", str(bag), "-c", str(idle_gap_config_file), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 episode(s)" in out
    assert "idle gaps" in out
    assert not (tmp_path / "ds").exists()


@pytest.mark.lerobot
def test_idle_gap_split_output_loads_in_lerobot(make_bag, idle_gap_config_file, tmp_path):
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

    bag = make_bag(spec=BagSpec(seconds=3.0, idle_gap=(1.0, 1.5)))
    out = tmp_path / "ds"
    reports = convert([bag], load_config(idle_gap_config_file), out)
    assert len(reports) == 2

    ds = lerobot_dataset.LeRobotDataset("local/bag2lerobot-idle-gap-test", root=out)
    assert ds.num_episodes == 2
    assert ds.fps == 30


def test_split_csv_must_be_a_string():
    with pytest.raises(ConfigError, match="split_csv"):
        parse_config({"fps": 30, "split_csv": 3, "features": {"a": {"topic": "/t"}}})


def test_split_csv_and_split_on_mutually_exclusive():
    with pytest.raises(ConfigError, match="split_on.*split_csv"):
        parse_config(
            {
                "fps": 30,
                "split_on": "/episode_start",
                "split_csv": "episodes.csv",
                "features": {"a": {"topic": "/t"}},
            }
        )


def test_csv_windows_reads_ranges(make_bag, tmp_path):
    bag = make_bag(spec=BagSpec(seconds=3.0))
    cfg = _split_cfg()
    streams = extract_streams(bag, cfg)
    csv_path = tmp_path / "episodes.csv"
    csv_path.write_text("start,end,task\n0.0,1.5,pick red cube\n1.5,3.0,\n")

    windows = csv_windows(csv_path, streams, extract_image_streams(bag, cfg))

    assert len(windows) == 2
    (start0, end0), task0 = windows[0]
    (start1, end1), task1 = windows[1]
    assert start0 == pytest.approx(T0, abs=0.02)
    assert end0 == pytest.approx(T0 + 1.5, abs=0.02)
    assert end0 == pytest.approx(start1, abs=1e-6)
    assert end1 == pytest.approx(T0 + 3.0, abs=0.02)
    assert task0 == "pick red cube"
    assert task1 is None


def test_csv_windows_missing_file(make_bag, tmp_path):
    bag = make_bag()
    cfg = _split_cfg()
    streams = extract_streams(bag, cfg)
    with pytest.raises(ExtractError, match="not found"):
        csv_windows(tmp_path / "nope.csv", streams, extract_image_streams(bag, cfg))


def test_csv_windows_needs_start_end_columns(make_bag, tmp_path):
    bag = make_bag()
    cfg = _split_cfg()
    streams = extract_streams(bag, cfg)
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("begin,finish\n0.0,1.0\n")
    with pytest.raises(ExtractError, match="start.*end"):
        csv_windows(csv_path, streams, extract_image_streams(bag, cfg))


def test_csv_windows_rejects_end_before_start(make_bag, tmp_path):
    bag = make_bag()
    cfg = _split_cfg()
    streams = extract_streams(bag, cfg)
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("start,end\n1.0,0.5\n")
    with pytest.raises(ExtractError, match="row 1"):
        csv_windows(csv_path, streams, extract_image_streams(bag, cfg))


def test_convert_splits_on_csv_ranges(make_bag, csv_split_config_file, tmp_path):
    cfg_path, _ = csv_split_config_file
    bag = make_bag(spec=BagSpec(seconds=3.0))
    out = tmp_path / "ds"
    reports = convert([bag], load_config(cfg_path), out)

    assert len(reports) == 2
    assert [r.segment for r in reports] == [0, 1]
    # episode 0 spans ~[0.2, 1.5)s (leader starts 0.2s late), episode 1 spans ~[1.5, 3.0)s
    assert 35 <= reports[0].sync.frames <= 41
    assert 42 <= reports[1].sync.frames <= 46

    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_tasks"] == 2
    tasks_df = pd.read_parquet(out / "meta" / "tasks.parquet")
    assert set(tasks_df.index) == {"pick red cube", "pick blue cube"}


def test_preview_splits_with_csv(make_bag, csv_split_config_file, tmp_path):
    cfg_path, _ = csv_split_config_file
    bag = make_bag(spec=BagSpec(seconds=3.0))
    cfg = load_config(cfg_path)
    windows = preview_splits([bag], cfg)
    assert len(windows[bag]) == 2
    assert windows[bag][0][1] == "pick red cube"
    assert windows[bag][1][1] == "pick blue cube"
    assert not (tmp_path / "ds").exists()


def test_cli_dry_run_with_csv(make_bag, csv_split_config_file, tmp_path, capsys):
    cfg_path, csv_path = csv_split_config_file
    bag = make_bag(spec=BagSpec(seconds=3.0))
    assert main(["convert", str(bag), "-c", str(cfg_path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 episode(s)" in out
    assert str(csv_path) in out
    assert "pick red cube" in out
    assert not (tmp_path / "ds").exists()


@pytest.mark.lerobot
def test_csv_split_output_loads_in_lerobot(make_bag, csv_split_config_file, tmp_path):
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    cfg_path, _ = csv_split_config_file

    bag = make_bag(spec=BagSpec(seconds=3.0))
    out = tmp_path / "ds"
    reports = convert([bag], load_config(cfg_path), out)
    assert len(reports) == 2

    ds = lerobot_dataset.LeRobotDataset("local/bag2lerobot-csv-split-test", root=out)
    assert ds.num_episodes == 2
    assert ds.fps == 30
