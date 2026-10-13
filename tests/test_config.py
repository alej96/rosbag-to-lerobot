from bag2lerobot.config import parse_config

BASE = {"fps": 30, "features": {"a": {"topic": "/t"}}}


def test_task_defaults_to_empty_string():
    cfg = parse_config(BASE)
    assert cfg.task == ""


def test_null_task_is_empty_not_the_string_none():
    cfg = parse_config({**BASE, "task": None})
    assert cfg.task == ""
