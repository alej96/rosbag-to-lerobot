import json
from dataclasses import dataclass

import av
import numpy as np
import pandas as pd
import pytest

from bag2lerobot.config import ConfigError, load_config, parse_config
from bag2lerobot.convert import convert
from bag2lerobot.extract import ExtractError, decode_image, extract_image_streams, extract_streams
from bag2lerobot.sync import synchronize
from tests.conftest import BagSpec

IMAGE_CONFIG = """
fps: 10
task: pick the cube
features:
  observation.state:
    topic: /follower/joint_states
    field: position
    names_from: name
  observation.images.cam:
    topic: /cam/image_raw
    type: image
"""


def _read(out):
    info = json.loads((out / "meta" / "info.json").read_text())
    eps = pd.read_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return info, eps


def test_image_feature_is_encoded_to_mp4(make_bag, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(IMAGE_CONFIG)
    bag = make_bag(spec=BagSpec(seconds=1.0, camera_hz=10))
    out = tmp_path / "ds"
    reports = convert([bag], load_config(cfg_path), out)
    info, eps = _read(out)

    video_path = out / "videos" / "observation.images.cam" / "chunk-000" / "file-000.mp4"
    assert video_path.exists()
    feat = info["features"]["observation.images.cam"]
    assert feat["dtype"] == "video"
    assert feat["shape"] == [6, 8, 3]
    assert feat["video_info"]["video.fps"] == 10

    container = av.open(str(video_path))
    decoded = list(container.decode(video=0))
    n = reports[0].sync.frames
    assert len(decoded) == n
    assert eps.loc[0, "videos/observation.images.cam/from_timestamp"] == 0.0
    assert eps.loc[0, "videos/observation.images.cam/to_timestamp"] == pytest.approx(n / 10)

    # the red channel ramps with time; the first and last frame should bracket it
    first = decoded[0].to_ndarray(format="rgb24")
    last = decoded[-1].to_ndarray(format="rgb24")
    assert int(first[0, 0, 0]) < int(last[0, 0, 0])


def test_image_frames_span_episode_boundaries_in_one_video(make_bag, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(IMAGE_CONFIG)
    bags = [
        make_bag(spec=BagSpec(seconds=1.0, camera_hz=10, seed=1), name="a"),
        make_bag(spec=BagSpec(seconds=1.0, camera_hz=10, seed=2), name="b"),
    ]
    out = tmp_path / "ds"
    reports = convert(bags, load_config(cfg_path), out)
    _, eps = _read(out)

    n0, n1 = reports[0].sync.frames, reports[1].sync.frames
    assert eps.loc[0, "videos/observation.images.cam/from_timestamp"] == 0.0
    assert eps.loc[0, "videos/observation.images.cam/to_timestamp"] == pytest.approx(n0 / 10)
    assert eps.loc[1, "videos/observation.images.cam/from_timestamp"] == pytest.approx(n0 / 10)
    assert eps.loc[1, "videos/observation.images.cam/to_timestamp"] == pytest.approx((n0 + n1) / 10)

    video_path = out / "videos" / "observation.images.cam" / "chunk-000" / "file-000.mp4"
    container = av.open(str(video_path))
    assert len(list(container.decode(video=0))) == n0 + n1


def test_decoded_frames_are_not_retained_in_reports_after_writing(make_bag, tmp_path):
    """Once an episode's frames are muxed into the video file, convert() must not keep
    them alive in the report it returns: that would hold every bag's decoded images in
    memory for the whole conversion, the exact blowup streaming video encoding (writing
    each episode's frames as they're produced, instead of collecting every episode's
    decoded frames before encoding one video) is meant to avoid."""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(IMAGE_CONFIG)
    bag = make_bag(spec=BagSpec(seconds=1.0, camera_hz=10))
    reports = convert([bag], load_config(cfg_path), tmp_path / "ds")
    assert reports[0].sync.images == {}
    assert reports[0].sync.frames > 0


def test_mismatched_image_resolution_across_episodes_rejected(make_bag, tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(IMAGE_CONFIG)
    bags = [
        make_bag(spec=BagSpec(seconds=1.0, camera_hz=10, camera_size=(8, 6)), name="a"),
        make_bag(spec=BagSpec(seconds=1.0, camera_hz=10, camera_size=(4, 3)), name="b"),
    ]
    with pytest.raises(ValueError, match="different resolution"):
        convert(bags, load_config(cfg_path), tmp_path / "ds")


def test_decodes_raw_and_compressed_images(make_bag):
    bag = make_bag(spec=BagSpec(seconds=1.0, camera_hz=5, compressed_camera_hz=5))
    cfg = parse_config(
        {
            "fps": 10,
            "features": {
                "raw": {"topic": "/cam/image_raw", "type": "image"},
                "compressed": {"topic": "/cam/image_compressed", "type": "image"},
            },
        }
    )
    streams = extract_image_streams(bag, cfg)
    raw_frame = decode_image(streams["raw"].frames[-1])
    compressed_frame = decode_image(streams["compressed"].frames[-1])
    assert raw_frame.shape == compressed_frame.shape == (6, 8, 3)
    # jpeg is lossy but should stay close to the raw frame
    assert np.abs(raw_frame.astype(int) - compressed_frame.astype(int)).max() < 10


def test_image_decoding_is_deferred(make_bag):
    """Frames aren't decoded until sync picks the ones the output grid needs."""
    bag = make_bag(spec=BagSpec(seconds=1.0, camera_hz=10))
    cfg = parse_config(
        {"fps": 10, "features": {"cam": {"topic": "/cam/image_raw", "type": "image"}}}
    )
    streams = extract_image_streams(bag, cfg)
    assert not isinstance(streams["cam"].frames[-1], np.ndarray)


def test_image_decoding_skips_unselected_frames_and_dedupes_repeats(monkeypatch, make_bag):
    """A camera publishing faster than the output fps shouldn't have every frame decoded,
    and a frame held across several output rows shouldn't be decoded more than once."""
    from bag2lerobot import sync as sync_mod

    bag = make_bag(spec=BagSpec(seconds=1.0, camera_hz=20))
    cfg = parse_config(
        {
            "fps": 5,
            "features": {
                "state": {
                    "topic": "/follower/joint_states",
                    "field": "position",
                    "names_from": "name",
                },
                "cam": {"topic": "/cam/image_raw", "type": "image"},
            },
        }
    )
    streams = extract_streams(bag, cfg)
    image_streams = extract_image_streams(bag, cfg)
    n_raw = len(image_streams["cam"].frames)

    decoded_ids = []
    real_decode = sync_mod.decode_image

    def spy(msg):
        decoded_ids.append(id(msg))
        return real_decode(msg)

    monkeypatch.setattr(sync_mod, "decode_image", spy)
    result = synchronize(streams, cfg, image_streams)

    assert len(decoded_ids) == len(set(decoded_ids))  # each raw frame decoded at most once
    assert len(decoded_ids) < n_raw  # frames the grid never samples are never decoded
    assert len(result.images["cam"]) == result.frames


def test_unsupported_image_encoding_rejected():
    @dataclass
    class FakeImage:
        __msgtype__ = "sensor_msgs/msg/Image"
        height: int = 2
        width: int = 2
        encoding: str = "yuv422"
        is_bigendian: int = 0
        step: int = 4
        data: np.ndarray = None

    with pytest.raises(ExtractError, match="unsupported image encoding"):
        decode_image(FakeImage(data=np.zeros(16, dtype=np.uint8)))


@pytest.mark.parametrize(
    ("encoding", "channels", "pixel", "expected_rgb"),
    [
        ("rgb8", 3, (10, 20, 30), (10, 20, 30)),
        ("bgr8", 3, (30, 20, 10), (10, 20, 30)),
        ("mono8", 1, (42,), (42, 42, 42)),
        ("rgba8", 4, (10, 20, 30, 255), (10, 20, 30)),
        ("bgra8", 4, (30, 20, 10, 255), (10, 20, 30)),
    ],
)
def test_decodes_every_supported_raw_encoding(encoding, channels, pixel, expected_rgb):
    h, w = 2, 2
    raw = np.tile(np.array(pixel, dtype=np.uint8), (h, w, 1))
    data = np.ascontiguousarray(raw.reshape(-1))
    enc = encoding

    @dataclass
    class FakeImage:
        __msgtype__ = "sensor_msgs/msg/Image"
        height: int = h
        width: int = w
        encoding: str = enc
        is_bigendian: int = 0
        step: int = w * channels
        data: np.ndarray = None

    out = decode_image(FakeImage(data=data))
    assert out.shape == (h, w, 3)
    assert tuple(int(c) for c in out[0, 0]) == expected_rgb


def test_image_feature_rejects_fields_and_names(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        "fps: 10\nfeatures:\n  cam:\n    topic: /cam\n    type: image\n    field: data\n"
    )
    with pytest.raises(ConfigError, match="don't support fields/names"):
        load_config(cfg_path)


def test_image_feature_rejects_linear_sync(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        "fps: 10\nfeatures:\n  cam:\n    topic: /cam\n    type: image\n    sync: linear\n"
    )
    with pytest.raises(ConfigError, match="only support sync 'hold'"):
        load_config(cfg_path)
