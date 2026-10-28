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
      first conversion quick.
- [x] **`init` picks up non-JointState numeric topics.** A simple numeric `std_msgs`
      topic outside any `JointState` (e.g. a gripper command on `std_msgs/Float64`)
      becomes an `action.<name>`/`observation.state.<name>` candidate, picked by the
      same leader/follower-style name hints as joints.
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
- [x] **Decode only the image frames sync actually samples.** `extract_image_streams` used
      to decode every camera message up front and hold the full RGB frame list in memory,
      even though a camera recorded faster than the configured `fps` throws most of those
      frames away, and `split_idle_gap_s` only ever needed their timestamps. It now keeps
      the raw message and decodes lazily (deduped) once `synchronize` knows which frames
      the output grid actually needs. First slice of the "stream extraction" item below.
- [x] **Write each bag's episode as it's produced, not after every bag.** `convert` used
      to synchronize every bag into one list of episodes (decoded video frames included)
      before calling `write_dataset` once at the end; `write_dataset` now consumes
      episodes one at a time (encoding each episode's video frames as they arrive instead
      of collecting every frame first), and the `ConversionReport`s `convert` returns no
      longer pin each episode's bulk value/image arrays in memory afterwards. A
      conversion of many bags now peaks at the memory of its largest single bag, not the
      sum of all of them. Second slice of the "stream extraction" item below.
- [x] **Sane per-stream `rate_hz` in the conversion report.** Yesterday's `TopicInfo.rate_hz`
      fix (median gap instead of count/span) only landed in `inspect_bag`; `sync.py`'s
      `StreamReport.rate_hz` — shown after every real `convert` and for every configured
      feature, not just `inspect` — still divided message count by the first/last span,
      so a feature topic republished as a burst (the same real `/tf`-style pattern found
      during real-bag validation) could still report an absurd rate in the actual
      conversion output. Both now share one `extract.rate_hz` implementation.
- [ ] **Stream extraction of a single huge bag.** `extract_streams`/`extract_image_streams`
      still hold every sample of one bag in memory before sync runs. For a bag that is
      itself huge (hours of data, or dense video, in one file), that still blows up
      memory even though a conversion of many such bags no longer does (shipped above).
      Needs incremental/windowed resampling in `sync.py`, so even one bag can be read and
      synchronized in pieces instead of loading all of it first.
- [x] **Append mode.** Add episodes to an existing dataset (`--append`), checking that
      features and fps match. Follow-up: once Hub re-push can diff instead of
      re-uploading everything, make it skip files `--append` didn't touch.
- [x] **Real-bag validation.** Find small public ROS bags (robot arms with JointState
      plus a camera), add a `network` test tier, and record results in
      `docs/validated-bags.md`.
- [x] **Custom message types** from `.msg` files (`msg_dir`/`--msg-dir`) for bags whose
      types aren't bundled with `rosbags` and aren't embedded in the bag either (some ROS
      2 storage plugins/distros only record the type name). `inspect`, `init` and
      `convert` all accept it.
- [x] **Episode splitting on idle gaps.** `split_idle_gap_s` splits a bag into episodes
      wherever every configured stream goes idle for at least that long, reusing the
      `--dry-run` machinery added for topic-based splitting. Follow-up below covers the
      other splitting source from the original item (CSV time ranges).
- [x] **Episode splitting on time ranges from a CSV.** `split_csv` splits a bag into
      episodes from a CSV of `start,end[,task]` rows (seconds since the first configured
      sample), for demos already labeled by an external tool; a row's `task` overrides
      the config's `task` for that episode. Reuses the `--dry-run` machinery added for
      the other two splitting sources.
- [x] **Sane `TopicInfo.rate_hz`.** It divided message count by the first/last timestamp
      span, which blows up to an absurd rate for a topic republished as a burst (nearly
      all messages sharing one bag timestamp, like a real `/tf` static-transform replay
      seen during real-bag validation). `rate_hz` is now the median gap between
      consecutive messages, which stays accurate for a steadily published topic and
      collapses to the existing "no reliable rate" `0.0` for a burst. This feeds
      `bag2lerobot init`'s fps guess, so a bursty topic could previously have produced a
      nonsensical `fps:` in the generated config, not just a confusing `inspect` display.

- [x] **Reject an odd video resolution with a clear error.** A camera with an odd width
      or height (e.g. a cropped ROI) reached PyAV/ffmpeg before failing, which raised an
      opaque `avcodec_open2(...) Generic error in an external library` instead of
      explaining that H.264's yuv420p needs both dimensions even. `_open_video_streams`
      now checks every image feature's resolution before opening any container and
      raises a clear `ValueError` naming the feature and its resolution.

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
- 2026-10-19: `extract_image_streams` no longer decodes every camera message up front;
  frames are decoded lazily (and only once per unique frame) once `synchronize` knows
  which ones the output grid actually samples, so a camera recorded faster than the
  configured fps no longer wastes memory and CPU decoding frames that get thrown away.
  First slice of the "stream extraction of huge bags" item; batching every bag's episodes
  in memory before `write_dataset` runs is the remaining follow-up.
- 2026-10-20: `write_dataset` consumes episodes one at a time instead of requiring a
  materialized list, encoding each episode's video frames as they arrive instead of
  collecting every episode's decoded frames before writing one video; `convert`'s
  generator drops each episode's bulk value/image arrays once it's written, instead of
  the reports it returns pinning them in memory for the rest of the conversion. A
  conversion of many bags now peaks at the memory of its largest single bag. Second
  slice of "stream extraction of huge bags"; incrementally resampling a single huge bag
  is the remaining follow-up.
- 2026-10-21: `split_csv` splits a bag into episodes from a CSV of `start,end[,task]`
  ranges (seconds since the first configured sample), for logs whose demos are already
  labeled by an external tool rather than a topic or idle gaps; a row's `task` column
  overrides the config's `task` for that one episode. `--dry-run` and `preview_splits`
  now report windows (and any per-episode task) uniformly across all three splitting
  sources instead of `split_on`/`split_idle_gap_s`-only start lists.
- 2026-10-22: `msg_dir` (config) / `--msg-dir` (`inspect`, `init`, `convert`) registers
  custom message types from `.msg` files laid out like a ROS package, for bags whose
  types `rosbags` doesn't bundle and that aren't embedded in the bag either (some ROS 2
  storage plugins/distros only ever record a topic's type name, not its schema).
  Deserializing such a topic used to raise a raw `KeyError`; it now raises a clear
  `ExtractError` naming the type and pointing at `msg_dir`, with or without a custom
  type registered. Picked over "Stream extraction of a single huge bag" (still top of
  "Next up"): that item needs `sync.py` to resample incrementally while a window's
  bounds are still being discovered, which risks the "never drop a frame" and
  "timestamp == frame_index / fps" invariants on a branch that auto-merges with no
  review; it deserves its own session rather than a rushed slice.
- 2026-10-23: `bag2lerobot init` now also turns a simple numeric `std_msgs` topic
  outside any `JointState` (`Bool`, `Float32`/`Float64`, `Int*`/`UInt*`, or a
  `*MultiArray` of one of those — e.g. a gripper command on `std_msgs/Float64`) into
  an `action.<name>`/`observation.state.<name>` feature candidate, picked by the same
  leader/follower-style name hints joints already use. Previously `init` ignored these
  topics entirely, so a real teleop setup where the gripper rides on its own topic
  (common once the arm's `JointState` doesn't include it) needed a hand-edited config
  before the first conversion could even run. Along the way, found and fixed a
  pre-existing `writer.py` bug the real `lerobot` loader test caught: any width-1
  numeric feature (exactly what a scalar gripper command is) was written as a
  length-1 list column, but LeRobot reads a `shape: [1]` feature as a scalar
  `datasets.Value`, so loading raised `DatasetGenerationCastError`. No existing config
  had a width-1 feature, so this was invisible until now; it would have broken any
  single-value feature, not just the ones `init` generates. Still top of "Next up":
  "Stream extraction of a single huge bag", for the same reason as yesterday.
- 2026-10-24: Real-bag validation: `tests/test_real_bags.py` (opt-in `network` marker,
  skipped on a plain `pytest` run via a `pytest_collection_modifyitems` hook, not just
  `importorskip` like the `lerobot` marker, since this one makes a live HTTP call)
  downloads a real ROS 2 Jazzy MCAP recording (dual-arm mobile manipulator teleop, 18
  topics, `RobotisSW/rosbag_test4` on the Hugging Face Hub) and converts its 25-joint
  `JointState` plus a `CompressedImage` camera into a LeRobot episode, asserting it
  converts with ~0 stale frames and loads in the real `LeRobotDataset`. `inspect_bag`
  needed no `msg_dir` for any of this bag's (all-standard) message types. Picked over
  "Stream extraction of a single huge bag" again (now deferred three sessions running
  for the same reason: it needs `sync.py` to resample incrementally while a window's
  bounds are still being discovered, which risks the "never drop a frame"/"timestamp ==
  frame_index / fps" invariants on a branch that auto-merges with no review) because
  this one was concretely actionable and the project had never been run against a real
  bag before. Results and the one quirk found (`/tf`'s displayed rate) are in
  `docs/validated-bags.md`.
- 2026-10-25: Fixed `TopicInfo.rate_hz` (`inspect_bag`): it divided message count by the
  first/last timestamp span, which reports an absurd rate (`95000000000.0Hz`, seen on a
  real bag's `/tf` topic) for a topic republished as a burst where nearly every message
  shares one bag timestamp. It now uses the median gap between consecutive messages,
  which stays accurate for a steadily published topic and collapses to the existing "no
  reliable rate" `0.0` for a burst; `tests/test_extract.py` reproduces the burst with a
  new `BagSpec.burst_topic`/`burst_count` and checks both cases. `bag2lerobot init` reads
  `rate_hz` to guess `fps`, so this wasn't just a cosmetic `inspect` display bug -- a
  bursty topic picked as a joint/state candidate could have produced a nonsensical `fps:`
  in the generated config. Picked over "Stream extraction of a single huge bag" again
  (now deferred four sessions running, for the same reason as before: it needs `sync.py`
  to resample incrementally while a window's bounds are still being discovered, which
  risks the "never drop a frame"/"timestamp == frame_index / fps" invariants on a branch
  that auto-merges with no review) because this one was a concrete, fully-scoped bug with
  a known real-world repro, left as a follow-up from yesterday's real-bag validation.
- 2026-10-26: Fixed `sync.StreamReport.rate_hz`: yesterday's median-gap fix for a bursty
  topic's rate only landed in `extract.inspect_bag`'s `TopicInfo.rate_hz`; `sync.py` kept
  its own copy of the old count/span formula for the per-stream rate shown after every
  real `convert`, so a *configured feature* topic republished as a burst (not just an
  `inspect`-only display topic like `/tf`) could still report an absurd rate in actual
  conversion output. Factored the median-gap calculation into one shared
  `extract.rate_hz`, used by both `inspect_bag` and `sync.synchronize`, and added
  `BagSpec.burst_numeric` so the burst-topic test helper can produce a numeric feature
  stream (not just an `inspect`-only topic) to catch this -
  `tests/test_convert.py::test_stream_report_rate_is_sane_for_burst_topic` fails on the
  old code with a 51M Hz reading and passes now. Picked over "Stream extraction of a
  single huge bag" again (now deferred five sessions running, for the same reason as
  before: it needs `sync.py` to resample incrementally while a window's bounds are still
  being discovered, which risks the "never drop a frame"/"timestamp == frame_index / fps"
  invariants on a branch that auto-merges with no review) because this was a concrete gap
  left by yesterday's fix, found while re-reading `sync.py` for that invariant risk.
- 2026-10-27: Fixed `extract._flatten`: a feature combining several equal-length arrays
  from one message via `fields` (e.g. `fields: [position, velocity]` on a `JointState`,
  both indexed by the same `names_from: name`) silently stopped reordering to the
  canonical joint order. The code reordered the *concatenated* vector against the raw
  `names_from` array, which is only sized for one field, so the length check inside
  `_canonical_order` always failed and the reorder never ran -- not just skipped
  reordering, but also left the feature's `names` list shorter than its vector (3 names
  for a width-6 feature), which is exactly the kind of shape/names mismatch LeRobot's
  loader rejects. Fixed by reordering each field's block to the canonical order on its
  own, before concatenating. `tests/test_convert.py::
  test_joint_order_is_stable_with_multiple_fields_from_one_names_from` reproduces it with
  a `position`+`velocity` feature over a driver that reshuffles joints every few
  messages: fails on the old code (names list only 3 long, columns drift out of order)
  and passes now. `tests/conftest.py`'s synthetic `JointState` now publishes real
  `velocity` values (previously always empty) so a test can combine it with `position`.
  Picked over "Stream extraction of a single huge bag" again (now deferred six sessions
  running, for the same reason as before: it needs `sync.py` to resample incrementally
  while a window's bounds are still being discovered, which risks the "never drop a
  frame"/"timestamp == frame_index / fps" invariants on a branch that auto-merges with no
  review) because this was a concrete, silent correctness bug found while reading
  `extract.py` for that same risk, and it fit a realistic config (state = position +
  velocity, both joint-ordered) that no existing test exercised.
- 2026-10-28: Fixed `writer._open_video_streams`: an image feature with an odd width or
  height (e.g. a cropped camera ROI) used to reach PyAV/ffmpeg before failing, raising an
  opaque `av.error.ExternalError: avcodec_open2("libx264", ...) Generic error in an
  external library` with no indication of the actual cause. H.264's `yuv420p` pixel
  format subsamples chroma 2x2, so it needs both dimensions even; every image feature's
  resolution is now checked before any video container is opened, raising a clear
  `ValueError` naming the feature and its resolution instead.
  `tests/test_video.py::test_odd_resolution_rejected_with_a_clear_error` reproduces it
  (fails on the old code with the opaque ffmpeg error, passes now with the clear one).
  Picked over "Stream extraction of a single huge bag" again (now deferred seven sessions
  running, for the same reason as before: it needs `sync.py` to resample incrementally
  while a window's bounds are still being discovered, which risks the "never drop a
  frame"/"timestamp == frame_index / fps" invariants on a branch that auto-merges with no
  review) because this was a concrete crash-to-confusing-error bug, found while reading
  `writer.py`'s video path for realistic camera configurations (a cropped or non-standard
  resolution) the existing tests never exercised.
