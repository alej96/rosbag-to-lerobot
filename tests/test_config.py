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


def test_names_as_a_string_raises_a_clear_error_instead_of_corrupting_silently():
    """A YAML typo of 'names: xyz' (a scalar) instead of 'names: [xyz]' (a one-element
    list) used to pass through unvalidated: FeatureSpec.names ended up as the string
    "xyz", and extract._flatten's `list(spec.names)` then split it into one name per
    *character* ('x', 'y', 'z'). For a feature whose width happens to match the string's
    length (e.g. a 3-character name on a 3-DoF joint), this silently wrote garbled
    per-column names into the dataset with no error at all, instead of the clear
    'must be a list of strings' every other malformed field already gets."""
    with pytest.raises(ConfigError, match="'names' must be a list of strings"):
        parse_config(
            {"fps": 30, "features": {"a": {"topic": "/t", "field": "data", "names": "xyz"}}}
        )


def test_fields_as_a_string_raises_a_clear_error_instead_of_corrupting_silently():
    """Like 'names', 'fields: position' (a scalar) instead of 'fields: [position]' used to
    reach extract._flatten's `for path in spec.fields` and iterate one *character* at a
    time as a bogus field path, instead of raising a clear config error up front."""
    with pytest.raises(ConfigError, match="'fields' must be a list of strings"):
        parse_config({"fps": 30, "features": {"a": {"topic": "/t", "fields": "position"}}})


def test_field_as_a_list_raises_a_clear_error():
    """'field: [position]' (a list) instead of 'field: position' (a string) used to reach
    extract._get's `path.split(".")` with a list instead of a string and crash with a raw
    AttributeError."""
    with pytest.raises(ConfigError, match="'field' must be a string"):
        parse_config({"fps": 30, "features": {"a": {"topic": "/t", "field": ["position"]}}})


def test_names_from_as_a_list_raises_a_clear_error():
    """'names_from: [name]' (a list) instead of 'names_from: name' (a string) used to
    reach extract._get's `path.split(".")` with a list instead of a string and crash with
    a raw AttributeError."""
    with pytest.raises(ConfigError, match="'names_from' must be a string"):
        parse_config({"fps": 30, "features": {"a": {"topic": "/t", "names_from": ["name"]}}})


def test_names_as_a_proper_list_is_unaffected():
    cfg = parse_config(
        {"fps": 30, "features": {"a": {"topic": "/t", "field": "data", "names": ["x", "y", "z"]}}}
    )
    assert cfg.features[0].names == ["x", "y", "z"]
