# Validated against real bags

The synthetic bags in `tests/conftest.py` pin down specific sync/extraction behaviors,
but they can't prove the converter handles a bag actually recorded off real hardware:
real message layouts, clock jitter, several compressed cameras at once, a 25-element
`JointState`. `tests/test_real_bags.py` (marked `network`, run with `pytest -m network`)
downloads real public bags and converts them for that reason. It's opt-in rather than
part of the default suite since it needs internet access and the bags aren't ours to
vendor into the repo.

## RobotisSW/rosbag_test4 (Hugging Face Hub)

<https://huggingface.co/datasets/RobotisSW/rosbag_test4>

- ROS 2 Jazzy, MCAP storage, ~43MB, 5297 messages across 18 topics.
- A dual-arm mobile-manipulator teleop session: `/joint_states`
  (`sensor_msgs/msg/JointState`, 25 joints: two 7-DoF arms, two grippers, a pan/tilt
  head, a lift and 3 steer/drive wheel pairs) at 100Hz, plus three
  `sensor_msgs/msg/CompressedImage` cameras (`/camera_left/...`, `/camera_right/...`,
  `/zed/zed_node/left/...`) at 15Hz, odometry, `/tf`, and teleop command topics we don't
  convert.
- Public and ungated, published without a license tag (Robotis' own test recording, not
  a curated dataset release) -- read at test time over HTTPS, not committed to this repo.

**Result:** `inspect_bag` reads every topic and type with no `msg_dir` needed (all
standard `sensor_msgs`/`nav_msgs`/`tf2_msgs` types). Converting `/joint_states` plus the
left camera into a LeRobot episode at 100fps produces 558 frames with 0 stale frames on
either stream (real 100Hz/15Hz hardware rates land cleanly on the 100fps grid), and the
output loads in the real `lerobot.datasets.lerobot_dataset.LeRobotDataset` with the
expected `(25,)` `observation.state` shape.

One unrelated quirk found along the way: `bag2lerobot inspect` printed `/tf`'s rate as
`95000000000.0Hz`. `/tf` isn't a topic any config would ever map to a feature, so it
doesn't affect conversion, but `TopicInfo.rate_hz` doesn't clamp out the degenerate case
where two messages share (or nearly share) a bag timestamp -- tracked as a follow-up
below rather than fixed blind in this change, since it needs its own test bag that
reproduces the clustered timestamps rather than a guess at the cause.

## Running it

```bash
python -m pip install -e ".[dev,video]"
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "lerobot[dataset]"
python -m pytest -q -m network
```

The first two real-bag tests only need `[video]`; the third (`test_real_bag_loads_in_lerobot`)
also needs `lerobot` installed, same as the rest of the `lerobot`-marked suite.
