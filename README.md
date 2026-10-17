# bag2lerobot: ROS bags to LeRobot datasets

Teams with existing robot logs (ROS 1 `.bag`, ROS 2 sqlite3 or MCAP) want to train
imitation-learning policies with [LeRobot](https://github.com/huggingface/lerobot), and
converting by hand is error-prone. Topics arrive at different rates and with jitter,
joints are published in varying order, and the LeRobot loader rejects frames that
aren't evenly spaced. `bag2lerobot` turns bags into a valid LeRobot v3.0 dataset using
one small YAML file.

- Reads ROS 1 and ROS 2 (sqlite3 and MCAP) bags without a ROS install (via
  [`rosbags`](https://gitlab.com/ternaris/rosbags))
- Resamples every topic onto one fixed-rate grid (`hold` or `linear`) and reports stale
  frames and gaps per stream instead of hiding them; a feature can override the sync
  method and staleness threshold (e.g. `hold` for a discrete gripper command while joint
  positions use `linear`)
- Uses header stamps by default, trims each episode to the window where all topics have
  data, and keeps joint order stable when a driver reorders `JointState`
- Flattens nested messages (`pose.position`, `pose.orientation`) into named columns
- Maps `sensor_msgs/Image` and `CompressedImage` topics to `observation.images.*` video
  features: frames are held on the same frame grid as everything else and encoded to MP4
  (H.264 via [PyAV](https://github.com/PyAV-Org/PyAV)); needs the `video` extra
- Splits one bag with many demos into multiple episodes, either on a topic (`split_on`)
  or automatically wherever every stream goes idle (`split_idle_gap_s`), with
  `convert --dry-run` to preview the boundaries before writing anything
- `--append` adds episodes to an existing dataset instead of starting over, checking
  that fps and features still match
- `--max-data-file-mb` caps the size of each data parquet file (default 100MB, matching
  `info.json`'s `data_files_size_in_mb`); a large conversion is split across several data
  files instead of growing one file without bound
- `bag2lerobot init` writes a starter config from a bag's topics, so the first
  conversion doesn't start from a blank YAML file
- `--push-to-hub user/name` uploads the dataset to the Hugging Face Hub and writes a
  dataset card (source bags, sync quality, features); needs the `hub` extra
- Output is checked in CI by loading it with the real `lerobot` `LeRobotDataset`

## Install

```bash
pip install git+https://github.com/alej96/rosbag-to-lerobot
```

Add `[video]` (e.g. `pip install "bag2lerobot[video] @ git+..."`) to convert image topics to
video features; without it, configs with an image feature raise a clear error telling you
to install it.

## Usage

```bash
bag2lerobot inspect my_run.mcap_dir/          # topics, types, counts, rates
bag2lerobot init my_run.mcap_dir/ -o cfg.yaml # starter config from the bag's topics
bag2lerobot convert run1/ run2/ run3.bag -c examples/so101_teleop.yaml -o ./my_dataset
bag2lerobot convert run*/ -c cfg.yaml -o ./ds --strict   # fail on stale frames
```

Each bag becomes one episode. See [examples/so101_teleop.yaml](examples/so101_teleop.yaml)
for the config format, including an `observation.images.*` feature (`type: image`) that
maps a camera topic to video.

### Generating a starter config

`bag2lerobot init` looks at one bag's topics and writes a config to get a first
conversion running: `JointState` topics become `observation.state`/`action` candidates
(picked by name, e.g. "leader"/"follower" or "cmd"/"state"; topic order breaks ties),
and `Image`/`CompressedImage` topics become `observation.images.*` video features. Edit
the result before converting: fill in `task`, and double-check any feature commented
`# check` — those are guesses `init` couldn't confirm from topic names alone.

```bash
bag2lerobot init my_run.mcap_dir/ -o cfg.yaml --fps 30
```

### Splitting many demos out of one bag

If a bag holds several demonstrations recorded back to back, set `split_on` in the config
to a topic that marks each demo's start, e.g. a `std_msgs/Bool` (only `data: true`
messages count, so you can publish `false` to mark an end without splitting on it) or
`std_msgs/Empty` published once per demo:

```yaml
split_on: /episode_start
```

Every bag is then split into one episode per boundary, from that message's timestamp up
to the next one (or the end of the bag for the last one); data before the first boundary
is dropped. Preview the boundaries before converting anything:

```bash
bag2lerobot convert run1.bag -c cfg.yaml --dry-run
```

If demos aren't marked by a topic but are separated by idle time (the operator pauses
between demos and every topic stops publishing), set `split_idle_gap_s` instead: a new
episode starts wherever all topics combined go quiet for at least that many seconds. No
data is dropped (unlike `split_on`, the first sample starts the first episode).

```yaml
split_idle_gap_s: 2.0
```

`split_on` and `split_idle_gap_s` can't both be set; `--dry-run` previews either one.

### Adding more episodes later

If more demos show up after the first conversion, add them with `--append` instead of
converting everything again:

```bash
bag2lerobot convert new_run.bag -c cfg.yaml -o ./my_dataset --append
```

The config must still describe the same fps and features (same keys, shapes and, for a
`names_from` feature, the same joint names) as the dataset already at `-o`; `--append`
checks this and fails with a clear error otherwise. Episode and frame indices continue
from the existing dataset, new data/video files are added alongside the old ones, and
`meta/stats.json` is recomputed over the combined dataset. `--append` and `--overwrite`
can't be used together.

### Large conversions

By default the data parquet for one `convert` call is split into files of at most
100MB (an episode is never split across files, so one unusually long episode can still
produce a file over the limit). Tune it with `--max-data-file-mb` if you need smaller
files, e.g. to stay under a Hub upload limit:

```bash
bag2lerobot convert run*/ -c cfg.yaml -o ./ds --max-data-file-mb 20
```

Reading still holds one bag's extracted samples and decoded frames in memory at a time;
converting bags that are individually huge (hours of data, or video, in one file) is not
yet streamed and is tracked as a follow-up.

## Development

```bash
uv venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check . && python -m pytest -q
pip install "lerobot[dataset]" && python -m pytest -m lerobot   # loader compatibility
```

## License

MIT
