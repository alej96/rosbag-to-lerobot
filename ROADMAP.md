# Roadmap

The objective: be the reliable way to turn existing ROS robot logs into LeRobot datasets
that train well. Every item must fix a real conversion problem people hit. Work top-down
through **Next up**; each item ships in one session with tests (synthetic bags in
`tests/conftest.py`) and stays loadable by real LeRobot (`pytest -m lerobot`).

## Next up

- [x] **Images to video.** Map `sensor_msgs/Image` and `CompressedImage` topics to
      `observation.images.*` video features: decode, hold-sample on the frame grid,
      encode to MP4 (H.264 via PyAV behind the optional `[video]` extra), and write
      `videos/...` paths plus the per-episode `from_timestamp`/`to_timestamp` metadata.
      Follow-ups: an AV1 codec option, and chunking large videos (currently one file per
      camera, covered by the "Large bags" item below).
- [x] **Episode splitting on a topic.** Many logs hold many demos in one file. A
      `split_on` config topic (e.g. `/episode_start` std_msgs/Bool rising edge, or Empty)
      marks each episode boundary, and `convert --dry-run` previews the boundaries
      without writing. Follow-up below covers the other two splitting sources from the
      original item (CSV time ranges, idle gaps).
- [x] **`bag2lerobot init`** generates a starter YAML from `inspect` (JointState topics
      become state and action candidates, image topics become cameras) to make the
      first conversion quick. Follow-up: once `--msg-dir` (custom message types) or
      `--push-to-hub` land, teach `init` to pick up non-JointState numeric topics (e.g.
      a gripper command on `std_msgs/Float64`) as extra features too.
- [x] **Push to the Hub.** `--push-to-hub user/name` uploads with `huggingface_hub` and
      writes a dataset card (source bags, sync stats, config used). Follow-up: once
      `--append` lands, re-push should update the card and only upload new/changed files
      instead of re-uploading the whole dataset.
- [x] **Per-feature resampling policies.** Per-feature `sync` override (e.g. hold for
      gripper commands, linear for joint positions) and `max_staleness_s`.
- [x] **Chunk data files by size.** The writer now splits the data parquet across
      multiple files once a conversion exceeds `--max-data-file-mb` (default 100, same
      as the `data_files_size_in_mb` already declared in `info.json`), instead of always
      writing one file per `convert` call regardless of size. Follow-up below covers the
      other half of the original "Large bags" item: streaming extraction itself.
- [ ] **Stream extraction of huge bags.** `extract_streams`/`extract_image_streams` hold
      every sample (and every decoded frame, for images) of one bag in memory before
      sync. For a single bag that is itself huge (hours of data, or dense video), that
      still blows up memory even with chunked output files. Needs incremental/windowed
      resampling in `sync.py` so a bag can be read and written in pieces.
- [x] **Append mode.** Add episodes to an existing dataset (`--append`), checking that
      features and fps match. Follow-up: once Hub re-push can diff instead of
      re-uploading everything, make it skip files `--append` didn't touch.
- [ ] **Real-bag validation.** Find small public ROS bags (robot arms with JointState
      plus a camera), add a `network` test tier, and record results in
      `docs/validated-bags.md`.
- [ ] **Custom message types** from `.msg` files (`--msg-dir`) for ROS 1/2 bags recorded
      without embedded definitions.
- [x] **Episode splitting on idle gaps.** `split_idle_gap_s` splits a bag into episodes
      wherever every configured stream goes idle for at least that long, reusing the
      `--dry-run` machinery added for topic-based splitting. Follow-up below covers the
      other splitting source from the original item (CSV time ranges).
- [ ] **Episode splitting on time ranges from a CSV.** Split a bag into episodes from a
      CSV of time ranges, reusing the `split_on`/`--dry-run` machinery.

## Needs the owner

- Publish to PyPI. Announce on the LeRobot Discord or ROS Discourse once video ships.

## Log

- 2026-10-09: v0.1.0. `inspect` and `convert` for ROS 1, ROS 2 sqlite3 and MCAP; YAML
  feature mapping (dotted fields, nested messages, names_from with stable joint order);
  hold/linear sync with stale-frame and gap reports; LeRobot v3.0 writer (data, episodes,
  tasks, stats). Output verified to load in lerobot 0.6.1.
- 2026-10-10: Per-feature `sync`/`max_staleness_s` overrides, so e.g. a discrete gripper
  command can stay `hold` while joint positions use `linear` in the same config.
- 2026-10-11: `observation.images.*` video features (`type: image`) decode
  `sensor_msgs/Image`/`CompressedImage`, hold-sample onto the frame grid and encode to
  MP4 via the optional `[video]` extra.
- 2026-10-12: `split_on` config topic splits one bag with many demos into multiple
  episodes (rising-edge `std_msgs/Bool` or every `std_msgs/Empty` message marks a
  boundary); `convert --dry-run` previews the boundaries first.
- 2026-10-13: `bag2lerobot init` writes a starter config from one bag's topics
  (JointState -> state/action candidates by name hint, Image/CompressedImage ->
  `observation.images.*`), flagging guesses to double-check. Also fixed `task:` with no
  value in a config parsing to the literal string `"None"` instead of empty.
- 2026-10-14: `--push-to-hub user/name` (optional `hub` extra) uploads a converted
  dataset to the Hugging Face Hub and writes a dataset card summarizing source bags,
  per-episode sync quality (stale frames, max gap) and features.
- 2026-10-15: `--append` adds new episodes to an existing dataset (new data/episodes/video
  files, merged tasks and globally combined stats) instead of requiring a full
  reconversion, after checking fps, features and (for `names_from` features) joint order
  all still match.
- 2026-10-16: `--max-data-file-mb` splits the data parquet into multiple files once a
  conversion exceeds the limit (default 100MB), instead of always writing one unbounded
  file per `convert` call; `info.json`'s `data_files_size_in_mb` now reflects the limit
  actually used. Streaming extraction for a single huge bag is split out as a follow-up.
- 2026-10-17: `split_idle_gap_s` splits a bag into episodes wherever every configured
  stream goes idle for at least that long, for demos that aren't marked by a topic;
  `--dry-run` previews these boundaries too. Splitting episodes from a CSV of time
  ranges is split out as a follow-up.
