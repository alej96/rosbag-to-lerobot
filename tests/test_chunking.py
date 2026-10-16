import json

import pandas as pd

from bag2lerobot.config import load_config
from bag2lerobot.convert import convert
from tests.conftest import BagSpec


def _read(out):
    info = json.loads((out / "meta" / "info.json").read_text())
    eps = pd.concat(
        pd.read_parquet(p)
        for p in sorted((out / "meta" / "episodes" / "chunk-000").glob("file-*.parquet"))
    ).reset_index(drop=True)
    data = pd.concat(
        pd.read_parquet(p) for p in sorted((out / "data" / "chunk-000").glob("file-*.parquet"))
    ).reset_index(drop=True)
    return info, data, eps


def test_large_conversion_splits_data_into_multiple_files(make_bag, config_file, tmp_path):
    bags = [make_bag("mcap", BagSpec(seconds=2.0, seed=i), name=f"bag{i}") for i in range(4)]
    out = tmp_path / "ds"
    reports = convert(bags, load_config(config_file), out, max_data_file_mb=0.001)
    info, data, eps = _read(out)

    data_files = sorted((out / "data" / "chunk-000").glob("file-*.parquet"))
    assert len(data_files) > 1
    assert info["data_files_size_in_mb"] == 0.001
    assert info["total_episodes"] == len(bags) == len(eps)
    assert info["total_frames"] == len(data) == sum(r.sync.frames for r in reports)
    # frame/global indices stay continuous across the chunk boundary
    assert list(data["index"]) == list(range(len(data)))
    for _, group in data.groupby("episode_index"):
        assert list(group["frame_index"]) == list(range(len(group)))

    # each episode's recorded file actually holds that episode's frames, intact
    for _, row in eps.iterrows():
        path = out / "data" / "chunk-000" / f"file-{row['data/file_index']:03d}.parquet"
        assert path.exists()
        df = pd.read_parquet(path)
        n = row["length"]
        episode_rows = df[df["episode_index"] == row["episode_index"]]
        assert len(episode_rows) == n

    # episode metadata itself stays in one file regardless of data chunking
    assert len(list((out / "meta" / "episodes" / "chunk-000").glob("file-*.parquet"))) == 1


def test_default_data_file_size_keeps_a_small_conversion_in_one_file(
    make_bag, config_file, tmp_path
):
    out = tmp_path / "ds"
    convert([make_bag()], load_config(config_file), out)
    info, _, _ = _read(out)
    assert info["data_files_size_in_mb"] == 100.0
    assert len(list((out / "data" / "chunk-000").glob("file-*.parquet"))) == 1


def test_data_file_chunking_continues_across_append(make_bag, config_file, tmp_path):
    out = tmp_path / "ds"
    convert(
        [make_bag("mcap", BagSpec(seed=1), name="a")],
        load_config(config_file),
        out,
        max_data_file_mb=0.001,
    )
    files_after_first = len(list((out / "data" / "chunk-000").glob("file-*.parquet")))
    assert files_after_first >= 1

    convert(
        [make_bag("mcap", BagSpec(seed=2), name="b")],
        load_config(config_file),
        out,
        append=True,
        max_data_file_mb=0.001,
    )
    info, data, eps = _read(out)
    files_after_second = sorted((out / "data" / "chunk-000").glob("file-*.parquet"))
    assert len(files_after_second) > files_after_first
    assert info["total_episodes"] == 2
    assert info["total_frames"] == len(data)
    assert list(data["index"]) == list(range(len(data)))
    assert eps.loc[1, "dataset_from_index"] == eps.loc[0, "dataset_to_index"]
