import pytest

from bag2lerobot.config import ConfigError, parse_config

BASE = {"fps": 30, "features": {"a": {"topic": "/t"}}}


def test_task_defaults_to_empty_string():
    cfg = parse_config(BASE)
    assert cfg.task == ""


def test_null_task_is_empty_not_the_string_none():
    cfg = parse_config({**BASE, "task": None})
    assert cfg.task == ""


@pytest.mark.parametrize("bad", [["fps", 30], "just text", 42, None])
def test_non_mapping_config_raises_a_clear_error(bad):
    """A config that doesn't parse to a YAML mapping (e.g. 'features:' indented as a
    top-level list by mistake, or an empty/scalar file) used to reach dict-only code
    (``"fps" not in data``, ``data.get``) and crash with a raw TypeError/AttributeError
    instead of the usual 'bag2lerobot: ...' message every other bad-config path gives."""
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        parse_config(bad)


@pytest.mark.parametrize("bad_features", ["oops", ["a", "b"], 5])
def test_non_mapping_features_raises_a_clear_error(bad_features):
    """'features:' written as a list or scalar instead of a mapping used to crash with a
    raw AttributeError from '.items()' instead of a clear ConfigError."""
    with pytest.raises(ConfigError, match="'features' must be a mapping"):
        parse_config({"fps": 30, "features": bad_features})
