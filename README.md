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
  frames and gaps per stream instead of hiding them
- Uses header stamps by default, trims each episode to the window where all topics have
  data, and keeps joint order stable when a driver reorders `JointState`
- Flattens nested messages (`pose.position`, `pose.orientation`) into named columns
- Output is checked in CI by loading it with the real `lerobot` `LeRobotDataset`

## Install

```bash
pip install git+https://github.com/alej96/rosbag-to-lerobot
```

## Usage

```bash
bag2lerobot inspect my_run.mcap_dir/          # topics, types, counts, rates
bag2lerobot convert run1/ run2/ run3.bag -c examples/so101_teleop.yaml -o ./my_dataset
bag2lerobot convert run*/ -c cfg.yaml -o ./ds --strict   # fail on stale frames
```

Each bag becomes one episode. See [examples/so101_teleop.yaml](examples/so101_teleop.yaml)
for the config format. Images and video are not converted yet (see the roadmap).

## Development

```bash
uv venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check . && python -m pytest -q
pip install "lerobot[dataset]" && python -m pytest -m lerobot   # loader compatibility
```

## License

MIT
