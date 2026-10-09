# CLAUDE.md

Guidance for AI agents (including the daily autonomous maintainer) working in this repo.

## What this project is

`bag2lerobot` converts ROS 1/ROS 2 bags (sqlite3, MCAP) into LeRobot v3.0 datasets for
imitation learning. Users have robot logs and want to train LeRobot policies. Read
README.md for usage and ROADMAP.md for priorities.

## Layout

- `src/bag2lerobot/config.py`: YAML config (`Config`, `FeatureSpec`) and validation
- `src/bag2lerobot/extract.py`: reads bags with `rosbags.highlevel.AnyReader` and
  flattens fields into numeric vectors, with stable joint order via `names_from`
- `src/bag2lerobot/sync.py`: resamples streams onto a fixed-rate grid; reports gaps
  and stale frames
- `src/bag2lerobot/writer.py`: LeRobot v3.0 layout (data parquet, meta/episodes,
  tasks, stats, info.json)
- `src/bag2lerobot/convert.py`, `cli.py`: orchestration and CLI
- `tests/conftest.py`: writes real ROS 1, ROS 2 sqlite3 and MCAP bags with known signals

## Rules

- **Real problems only.** Every change must make conversion more correct, more complete
  or easier for real users. No cosmetic churn or speculative abstractions.
- **LeRobot compatibility is the contract.** Timestamps must equal `frame_index / fps`
  exactly. Never drop frames inside an episode (fill them and report stale frames
  instead). `pytest -m lerobot` loads the output with the real `LeRobotDataset` and must
  pass. CI runs it.
- **Tests for every behavior change**, using synthetic bags in all three formats where
  relevant. Make sure the test fails without your change.
- Keep core dependencies small (rosbags, numpy, pandas, pyarrow, pyyaml). Video or Hub
  features go behind optional extras.
- Never modify `.github/workflows/`. Changes there block auto-merge and need a human.

## Commands

```bash
python -m pip install -e ".[dev]"
ruff check . && ruff format --check . && python -m pytest -q
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install "lerobot[dataset]"
python -m pytest -q -m lerobot
```

## Shipping

Push a branch named `claude/<yyyy-mm-dd>-<slug>`. The `auto-merge` workflow runs CI
(including LeRobot compatibility) and rebase-merges the branch into `main` when green.
Write commit subjects in the imperative mood with a body covering the problem and how
you verified the change. Update ROADMAP.md (check items off, add follow-ups, append to
the Log) in the same branch.
